"""Turn raw Slack messages into structured ErrorEvents.

The renewal service does not post JSON. It posts Slack rich-text blocks laid out
as alternating label / preformatted-value pairs, which render in the fallback
text as:

    methodName ``` renewalHandler ```
    correlationId ``` 3f7a… ```
    error ``` type: payment-failure
              message: Renewal Flow: Card declined: Your card was declined. ```
    message ``` Renewal Flow for 1e7c… : Renewal Flow: Card declined… ```

That structure is far richer than a bare error string, and the original script
threw all of it away by regex-matching `"error": "..."` against the flattened
text. Three fields matter and are now captured:

  methodName     which handler failed — the discriminator that separates several
                 unrelated "Request failed with status code 400" failures
  error.type     the service's own taxonomy (payment-failure, unknown, …), used
                 directly as the cluster category instead of having a model guess
  correlationId  explicitly *not* the quote, so it can be excluded from quote
                 attribution rather than inflating it

Quote IDs appear inline in prose — "for quote: <uuid>", "Renewal Flow for <uuid>"
— so they are matched by context rather than by grabbing every UUID present.

Three input shapes are handled, in order: labelled blocks (the real format),
JSON payloads, and a regex fallback for anything else.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from .models import ErrorEvent

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_UUID = UUID_RE.pattern

# Keys whose string values are candidate error messages, best first.
_MESSAGE_KEYS = (
    "errorMessage",
    "error_message",
    "error",
    "message",
    "msg",
    "reason",
    "detail",
    "details",
    "description",
)
_QUOTE_KEYS = ("quoteId", "quote_id", "quoteID", "quoteid", "quoteReference", "quoteRef")
_STATUS_KEYS = ("status", "statusCode", "status_code", "httpStatus", "code")
_SERVICE_KEYS = ("methodName", "service", "source", "stage", "step", "flow", "handler", "endpoint")
_NOISE_VALUES = {"", "error", "null", "none", "n/a", "-", "undefined"}

# Quote IDs appear in prose. Match on the surrounding words so a correlation ID
# or document ID sitting in the same sentence is never mistaken for the quote.
_QUOTE_CONTEXT = [
    # JSON form, which survives even when Slack truncates the payload mid-object.
    re.compile(rf'"quote_?id"\s*:\s*"({_UUID})"', re.I),
    re.compile(rf"\bfor\s+quote\s*:?\s*({_UUID})", re.I),
    re.compile(rf"\bquote\s*(?:id|ref|reference)?\s*[:=]\s*({_UUID})", re.I),
    re.compile(rf"\bquote\s+docs\s+for\s*:?\s*({_UUID})", re.I),
    re.compile(rf"\brenewal\s+flow\s+for\s+({_UUID})", re.I),
    re.compile(rf"\bpolicy\s+renewal\s+for\s+({_UUID})", re.I),
]

# Label names inside the rich-text payload.
_LABEL_ERROR = {"error", "err", "exception"}
_LABEL_MESSAGE = {"message", "msg", "text"}
_LABEL_METHOD = {"methodname", "method", "handler", "service"}
_LABEL_CORRELATION = {"correlationid", "correlation_id", "requestid", "traceid"}

# Sub-fields inside an `error` value block. Deliberately not line-anchored:
# handlers emit both newline-separated and space-separated forms.
_SUBFIELD_ANY = re.compile(
    r"(?:^|[\s,;])(type|message|name|code|statusCode|errorType|error)\s*:\s*", re.M
)
_JSON_START = re.compile(r"(?:^|\s)error\s*:\s*(?=[{\[])", re.M)

# Slack system events and non-error payloads. Counting these as extraction
# failures makes the coverage number meaningless — they are not errors at all.
_NON_ERROR_SUBTYPES = {
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "bot_add",
    "bot_remove",
    "pinned_item",
    "thread_broadcast",
}
_ERROR_HINT = re.compile(
    r"\berror\b|\bfail(?:ed|ure|s)?\b|\bexception\b|\bdeclined\b|\btimeout\b|"
    r"\btimed out\b|\bnot found\b|\brejected\b|\binvalid\b|\bcannot\b|\bcan't\b|"
    r"\bunable\b|\bmissing\b|\bstatus code\b|\bvalidation\b",
    re.I,
)

# Policy documents and endorsement clauses are posted to the same channel. They
# are not errors and must not be reported as extraction failures.
_MARKUP = re.compile(r"&lt;|&gt;|<div|<span|<p[ >]|<ul|<li|MsoNormal|clauseText", re.I)


# --- shared helpers --------------------------------------------------------


def _walk(obj: Any, path: str = "") -> list[tuple[str, str, Any]]:
    out: list[tuple[str, str, Any]] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            child = f"{path}.{key}" if path else str(key)
            out.append((str(key), child, val))
            out.extend(_walk(val, child))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            out.extend(_walk(item, f"{path}[{idx}]"))
    return out


def _score_message(key: str, value: str) -> float:
    try:
        rank = _MESSAGE_KEYS.index(key)
    except ValueError:
        return -1.0
    score = float(len(_MESSAGE_KEYS) - rank) + min(len(value.split()), 25) * 0.4
    if len(value.split()) > 200:
        score -= 10.0
    if value.strip().startswith(("{", "[")):
        score -= 8.0
    return score


def _first_string(entries: list[tuple[str, str, Any]], keys: tuple[str, ...]) -> str | None:
    for wanted in keys:
        for key, _path, val in entries:
            if key == wanted and isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _first_int(entries: list[tuple[str, str, Any]], keys: tuple[str, ...]) -> int | None:
    for wanted in keys:
        for key, _path, val in entries:
            if key != wanted or isinstance(val, bool):
                continue
            if isinstance(val, int) and 100 <= val <= 599:
                return val
            if isinstance(val, str) and val.strip().isdigit() and 100 <= int(val.strip()) <= 599:
                return int(val.strip())
    return None


def find_quote_id(text: str, exclude: set[str] | None = None) -> tuple[str | None, float]:
    """Locate the quote ID by surrounding context, not by grabbing every UUID.

    Returns (quote_id, confidence): 0.8 when a contextual pattern matched, 0.5
    when exactly one candidate UUID remained after excluding known non-quote
    identifiers, 0.0 when we could not tell. Unattributed is reported honestly
    rather than guessed — the old script's union-of-all-UUIDs approach claimed
    162 affected quotes in a week where it could evidence far fewer.
    """
    exclude = exclude or set()
    for pattern in _QUOTE_CONTEXT:
        match = pattern.search(text)
        if match and match.group(1) not in exclude:
            return match.group(1), 0.8

    candidates = [u for u in dict.fromkeys(UUID_RE.findall(text)) if u not in exclude]
    if len(candidates) == 1:
        return candidates[0], 0.5
    return None, 0.0


# --- shape 1: labelled rich-text blocks (the real format) ------------------


def labelled_pairs(msg: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract (label, value) pairs from a rich-text label/preformatted layout."""
    pairs: list[tuple[str, str]] = []
    pending: str | None = None

    for block in msg.get("blocks") or []:
        if block.get("type") != "rich_text":
            continue
        for element in block.get("elements", []):
            text = "".join(
                inner.get("text", "")
                for inner in element.get("elements", [])
                if inner.get("type") == "text"
            )
            if element.get("type") == "rich_text_section":
                stripped = text.strip()
                if stripped:
                    pending = stripped
            elif element.get("type") == "rich_text_preformatted":
                pairs.append((pending or "", text))
                pending = None

    # The flattened text rendering fences preformatted values in triple
    # backticks. Parse it too, and keep whichever view recovered more fields.
    #
    # This is not merely a fallback for missing blocks. When a payload is long
    # enough for Slack to truncate it, the trailing `rich_text_preformatted`
    # element is dropped from `blocks` while its opening still survives in
    # `text`. Returning the blocks view unconditionally lost the entire `error`
    # field on exactly those messages — a whole class of partner-api validation
    # failures went unreported, and they were the ones with the most content.
    from_text = _pairs_from_text(msg.get("text") or "")
    return from_text if len(from_text) > len(pairs) else pairs


def _pairs_from_text(text: str) -> list[tuple[str, str]]:
    if "```" not in text:
        return []
    parts = text.split("```")
    pairs: list[tuple[str, str]] = []
    # Values sit at odd indices; the label is the tail of the preceding chunk.
    # An unterminated final fence still yields its value, which is the point.
    for i in range(1, len(parts), 2):
        pairs.append((parts[i - 1].strip().split("\n")[-1].strip(), parts[i]))
    return pairs


def parse_error_value(value: str) -> dict[str, Any]:
    """Parse the inner structure of an `error` value block.

    Sub-fields are not reliably newline-separated. Several handlers emit
    "type: partner-api message: response-validation error: {…}" on one line, and a
    line-anchored parser finds only `type` on those, loses the message entirely
    and drops the event. So the scan tolerates both forms, and the trailing JSON
    blob — frequently truncated by Slack's message length limit — is parsed
    separately with a regex recovery path for when it will not load.
    """
    out: dict[str, Any] = {}

    head, json_part = value, ""
    marker = _JSON_START.search(value)
    if marker:
        head, json_part = value[: marker.start()], value[marker.end() :]
    else:
        brace = value.find("{")
        if brace > 0 and len(value) - brace > 60:
            head, json_part = value[:brace], value[brace:]

    matches = list(_SUBFIELD_ANY.finditer(head))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(head)
        name = "type" if match.group(1) == "errorType" else match.group(1)
        body = head[match.end() : end].strip()
        if body:
            out.setdefault(name, body)
    if not out and head.strip():
        out["message"] = head.strip()

    if json_part:
        nested: Any = None
        try:
            nested = json.loads(json_part)
        except json.JSONDecodeError:
            nested = None

        if isinstance(nested, dict):
            cause = nested.get("cause")
            if isinstance(cause, dict):
                out.setdefault("cause_type", cause.get("type"))
                out.setdefault("cause_name", cause.get("name"))
            out.setdefault("type", nested.get("type") or nested.get("errorType"))
            out.setdefault("message", nested.get("message"))
        else:
            # Truncated payload: recover what we can rather than lose the event.
            for key, pattern in (
                ("type", r'"errorType"\s*:\s*"([^"]+)"'),
                ("message", r'"message"\s*:\s*"([^"]{3,300})"'),
                ("cause_type", r'"cause"\s*:\s*\{[^{}]*?"type"\s*:\s*"([^"]+)"'),
                ("cause_name", r'"cause"\s*:\s*\{[^{}]*?"name"\s*:\s*"([^"]+)"'),
            ):
                found = re.search(pattern, json_part, re.S)
                if found:
                    out.setdefault(key, found.group(1))

    return {key: val for key, val in out.items() if val}


def labelled_meta(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Handler and correlation ID, independent of which path parses the message.

    These must survive even when message extraction falls through to the JSON
    path, because `methodName` is the only thing separating several unrelated
    "Request failed with status code 400" failures from each other.
    """
    meta: dict[str, str] = {}
    for label, value in pairs:
        key = label.lower().strip(": ")
        if key in _LABEL_METHOD and value.strip() and "service" not in meta:
            meta["service"] = value.strip()[:80]
        elif key in _LABEL_CORRELATION and "correlation_id" not in meta:
            found = UUID_RE.search(value)
            if found:
                meta["correlation_id"] = found.group(0)
    return meta


def _from_labelled(msg: dict[str, Any], pairs: list[tuple[str, str]]) -> dict[str, Any] | None:
    fields: dict[str, Any] = {}
    error_block: dict[str, Any] = {}
    message_text = ""
    raw_error = ""

    for label, value in pairs:
        key = label.lower().strip(": ")
        if key in _LABEL_ERROR:
            error_block = parse_error_value(value)
            raw_error = value
        elif key in _LABEL_MESSAGE:
            message_text = value.strip()
        elif key in _LABEL_METHOD:
            fields["service"] = value.strip() or None
        elif key in _LABEL_CORRELATION:
            found = UUID_RE.search(value)
            fields["correlation_id"] = found.group(0) if found else value.strip() or None

    primary = (error_block.get("message") or "").strip()
    if not primary:
        # Some payloads carry only the message block.
        primary = message_text.strip()
    if not primary and error_block.get("cause_type"):
        primary = str(error_block["cause_type"])
    # A trailing colon is left behind when a serialised object is split off the
    # end of the message; it carries no meaning and only makes labels look broken.
    primary = primary.strip().rstrip(" :;,-")
    if not primary:
        return None

    others = [
        text
        for text in (message_text, error_block.get("error") or "", error_block.get("name") or "")
        if text and text.strip() and text.strip() != primary
    ]

    fields.update(
        {
            "message": primary,
            "all_messages": [primary, *[o.strip()[:400] for o in others]],
            "error_type": error_block.get("type"),
            "cause_type": error_block.get("cause_type"),
            "cause_name": error_block.get("cause_name"),
        }
    )

    status = error_block.get("statusCode") or error_block.get("code")
    if isinstance(status, str) and status.strip().isdigit() and 100 <= int(status) <= 599:
        fields["http_status"] = int(status)
    else:
        match = re.search(r"status(?:\s*code)?\D{0,3}(\d{3})", primary, re.I)
        fields["http_status"] = int(match.group(1)) if match else None

    exclude = {fields["correlation_id"]} if fields.get("correlation_id") else set()
    # Search the *raw* error value, not just the parsed fields: when Slack
    # truncates a payload mid-object the quote ID often survives only as
    # `"quoteId": "…"` inside the JSON tail that parsing discarded.
    search_space = f"{message_text}\n{primary}\n{raw_error}"
    quote_id, confidence = find_quote_id(search_space, exclude)
    fields["quote_id"] = quote_id
    fields["quote_id_confidence"] = confidence
    fields["related_ids"] = [
        u for u in dict.fromkeys(UUID_RE.findall(search_space)) if u != quote_id
    ][:20]
    return fields


# --- shape 2 and 3: JSON payloads and loose text ---------------------------


def _parse_candidates(msg: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for block in msg.get("blocks") or []:
        if block.get("type") != "rich_text":
            continue
        for element in block.get("elements", []):
            if element.get("type") not in {"rich_text_preformatted", "rich_text_section"}:
                continue
            for inner in element.get("elements", []):
                if inner.get("type") == "text" and inner.get("text"):
                    texts.append(inner["text"])
    for attachment in msg.get("attachments") or []:
        for field in ("text", "fallback"):
            if attachment.get(field):
                texts.append(attachment[field])
    if msg.get("text"):
        texts.append(msg["text"])

    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        key = text.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(text)
    return unique


def _loads(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        brace = stripped.find("{")
        if brace == -1:
            return None
        stripped = stripped[brace:]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _from_json(text: str, parsed: Any) -> dict[str, Any] | None:
    entries = _walk(parsed)
    scored = [
        (_score_message(key, val), val.strip())
        for key, _path, val in entries
        if isinstance(val, str) and val.strip() and val.strip().lower() not in _NOISE_VALUES
    ]
    scored = [(score, val) for score, val in scored if score > 0]
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)

    ordered: list[str] = []
    for _score, val in scored:
        if val not in ordered:
            ordered.append(val)

    correlation = _first_string(entries, ("correlationId", "correlation_id", "requestId"))
    exclude = {correlation} if correlation else set()

    quote_id, confidence = (None, 0.0)
    for wanted in _QUOTE_KEYS:
        for key, _path, val in entries:
            if key == wanted and isinstance(val, str) and UUID_RE.fullmatch(val.strip()):
                quote_id, confidence = val.strip(), 1.0
                break
        if quote_id:
            break
    if not quote_id:
        for key, path, val in entries:
            if (
                key.lower() in {"id", "uuid"}
                and isinstance(val, str)
                and UUID_RE.fullmatch(val.strip())
                and "quote" in path.lower()
            ):
                quote_id, confidence = val.strip(), 0.5
                break
    if not quote_id:
        quote_id, confidence = find_quote_id(text, exclude)

    error_type = _first_string(entries, ("type", "errorType", "kind"))
    return {
        "message": ordered[0],
        "all_messages": ordered[:8],
        "quote_id": quote_id,
        "quote_id_confidence": confidence,
        "related_ids": [u for u in dict.fromkeys(UUID_RE.findall(text)) if u != quote_id][:20],
        "http_status": _first_int(entries, _STATUS_KEYS),
        "service": _first_string(entries, _SERVICE_KEYS),
        "error_type": error_type,
        "correlation_id": correlation,
    }


def _from_loose_text(text: str) -> dict[str, Any] | None:
    pairs = re.findall(r'"(?:error|message|errorMessage|reason)"\s*:\s*"(.*?)"', text)
    ordered = [p.strip() for p in dict.fromkeys(pairs) if p.strip()]
    if not ordered:
        return None
    status = re.search(r"status(?:\s*code)?\D{0,3}(\d{3})", text, re.I)
    quote_id, confidence = find_quote_id(text)
    return {
        "message": ordered[0],
        "all_messages": ordered[:8],
        "quote_id": quote_id,
        "quote_id_confidence": confidence,
        "related_ids": [u for u in dict.fromkeys(UUID_RE.findall(text)) if u != quote_id][:20],
        "http_status": int(status.group(1)) if status else None,
        "service": None,
        "error_type": None,
        "correlation_id": None,
    }


# --- entry point -----------------------------------------------------------


def is_document_noise(msg: dict[str, Any]) -> bool:
    """Policy wordings and endorsement clauses share the channel with errors."""
    text = msg.get("text") or ""
    if not text:
        return False
    hits = len(_MARKUP.findall(text))
    return hits >= 3 or (hits >= 1 and len(text) > 1500)


def is_reportable(msg: dict[str, Any]) -> bool:
    """Is this message something the daily report should ever describe?

    The channel carries policy wordings, quote answer payloads, and Slack join
    events alongside real failures. None of those are errors, so treating them as
    extraction misses would make coverage look far worse than it is — and would
    push us to "fix" a parser that is working correctly.
    """
    if msg.get("subtype") in _NON_ERROR_SUBTYPES:
        return False
    if is_document_noise(msg):
        return False
    text = msg.get("text") or ""
    if not text.strip():
        return False
    if any(label.lower().strip(": ") in _LABEL_ERROR for label, _v in labelled_pairs(msg)):
        return True
    return bool(_ERROR_HINT.search(text))


def extract_events(messages: list[dict[str, Any]]) -> list[ErrorEvent]:
    """Extract at most one ErrorEvent per Slack message."""
    events: list[ErrorEvent] = []

    for msg in messages:
        if not is_reportable(msg):
            continue

        ts = str(msg.get("ts", "0"))
        try:
            occurred_at = datetime.fromtimestamp(float(ts), tz=UTC)
        except (ValueError, OSError):
            occurred_at = datetime.now(tz=UTC)

        fields: dict[str, Any] | None = None
        excerpt = ""

        pairs = labelled_pairs(msg)
        meta = labelled_meta(pairs) if pairs else {}
        if pairs:
            fields = _from_labelled(msg, pairs)
            excerpt = (msg.get("text") or "")[:2000]

        if fields is None:
            for text in _parse_candidates(msg):
                parsed = _loads(text)
                fields = _from_json(text, parsed) if parsed is not None else _from_loose_text(text)
                if fields:
                    excerpt = text[:2000]
                    break

        if not fields:
            continue

        # Whichever path produced the message, the labelled handler and
        # correlation ID still apply. Losing methodName here is what left
        # forty generic errors with nothing to separate them.
        for key, value in meta.items():
            if not fields.get(key):
                fields[key] = value

        # A correlation ID is never a quote. If a fallback path picked one up,
        # re-attribute with it excluded rather than reporting it as a quote.
        correlation = fields.get("correlation_id")
        if correlation and fields.get("quote_id") == correlation:
            quote_id, confidence = find_quote_id(excerpt, {correlation})
            fields["quote_id"] = quote_id
            fields["quote_id_confidence"] = confidence

        events.append(
            ErrorEvent(
                slack_ts=ts,
                occurred_at=occurred_at,
                raw_excerpt=excerpt,
                **{k: v for k, v in fields.items() if v is not None or k == "quote_id"},
            )
        )

    return events
