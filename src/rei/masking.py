"""Volatile-token masking: turn a concrete error string into a stable template.

This runs *before* embedding and does most of the heavy lifting for free.

Worked example from the current production report:

    "Renewal Flow: Failed to get quote docs for: 8afd54b3-e236-4dcf-806e-3024680a3c9b"

Because the quote ID is embedded in the message text, every occurrence is a
different string, so the old exact-match grouping reported each one as its own
"1 x ..." line. After masking, all of them collapse to:

    "Renewal Flow: Failed to get quote docs for: <UUID>"

which is one cluster with a real frequency count.

Masking is also what keeps the embedding bill at approximately zero: identical
templates hash to the same key and hit the cache, so a normal day embeds only
the handful of genuinely new templates.

Number handling is deliberately asymmetric. Integers of one to three digits are
preserved, because "status code 400" and "status code 500" are different
failures and must not merge. Anything with a decimal point, or four or more
digits, is a quantity or an identifier and is masked.
"""

from __future__ import annotations

import hashlib
import re

# Order matters: the most specific patterns run first so that, for example, a
# Slack-wrapped mailto link is recognised as an email rather than a URL.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Slack auto-link markup: <mailto:a@example.com|a@example.com> and <http://x|x>
    (re.compile(r"<mailto:[^|>]+(?:\|[^>]*)?>"), "<EMAIL>"),
    (re.compile(r"<https?://[^|>]+(?:\|[^>]*)?>"), "<URL>"),
    # UUIDs (canonical 8-4-4-4-12)
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "<UUID>",
    ),
    # Stripe-style prefixed object IDs. `pm_` matters in particular: payment
    # method IDs appear inline in "Failed to create pm_… setup intent", and
    # missing the prefix split one failure across several templates.
    (
        re.compile(
            r"\b(?:cus|sub|pi|ch|in|price|prod|card|src|seti|evt|acct|txn|re|py|po|tok|"
            r"pm|ba|btok|cs|sk|pk|whsec|ii|inv|dp|fee|tr|trr|sub_sched)_"
            r"[A-Za-z0-9]{6,}\b"
        ),
        "<STRIPE_ID>",
    ),
    # Bare emails
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<EMAIL>"),
    # Bare URLs
    (re.compile(r"\bhttps?://\S+"), "<URL>"),
    # ISO-8601 timestamps and plain dates
    (
        re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
        "<TS>",
    ),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<DATE>"),
    # JWT / bearer-ish blobs
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "<JWT>"),
    # Mongo ObjectIds and other long hex runs
    (re.compile(r"\b[0-9a-fA-F]{24,}\b"), "<HEX>"),
    # Decimals and long integers -> quantities/identifiers. Short integers stay.
    (re.compile(r"\b\d+\.\d+\b"), "<NUM>"),
    (re.compile(r"\b\d{4,}\b"), "<NUM>"),
]

_WS = re.compile(r"\s+")
_MAX_TEMPLATE_CHARS = 1000

# A serialised exception appended to a message — "Can't update CRM
# contacts: {"message":"…","name":"AxiosError","config":{…}}". The blob carries
# request headers, timeouts and payloads that differ on every occurrence, so
# leaving it in fragments one failure into a separate template per event. In a
# real week it turned one CRM-sync failure into eight singleton lines. The
# failure is identified by the prose before the colon; the blob is noise.
_EMBEDDED_JSON = re.compile(r":\s*[{\[]\s*[\"']?.{60,}$", re.S)


def mask(text: str) -> str:
    """Replace volatile tokens with stable placeholders."""
    out = text
    for pattern, placeholder in _PATTERNS:
        out = pattern.sub(placeholder, out)
    return out


_INNER_MESSAGE = re.compile(r'"message"\s*:\s*"([^"]{3,200})"')


def strip_embedded_json(text: str) -> str:
    """Drop a trailing serialised object, keeping the prose that names the failure.

    When the prose is only a prefix — "Renewal Flow: {…}" — discarding the blob
    would leave nothing diagnostic at all, so the object's own `message` field is
    promoted in its place. That recovers a dozen otherwise-anonymous errors per
    week while still collapsing the variable request metadata around it.
    """
    match = _EMBEDDED_JSON.search(text)
    if not match:
        return text
    inner = _INNER_MESSAGE.search(match.group(0))
    replacement = f": {inner.group(1)}" if inner else ": <JSON>"
    return text[: match.start()] + replacement


def normalize(text: str) -> str:
    """Full normalisation: mask, collapse whitespace, trim, and bound length."""
    out = _WS.sub(" ", mask(strip_embedded_json(text))).strip()
    # Collapse runs of the same placeholder, e.g. "<UUID>, <UUID>, <UUID>" ->
    # "<UUID>", so a list of ten IDs and a list of two produce one template.
    out = re.sub(r"(<[A-Z_]+>)(\s*[,;|]\s*\1)+", r"\1", out)
    if len(out) > _MAX_TEMPLATE_CHARS:
        out = out[:_MAX_TEMPLATE_CHARS] + " …"
    return out


def template_hash(template: str) -> str:
    """Stable content hash used as the embedding cache key."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:32]


# Messages that are transport-level noise: they describe *that* a call failed,
# never *what* failed. Two of these strings being identical says nothing about
# whether the underlying failures are related, so they must be enriched with the
# emitting service and any deeper message before they are embedded. This is the
# fix for the "12 x Request failed with status code 400" line in the current
# report, which merged several unrelated upstream failures into one number.
_GENERIC_PATTERNS = [
    re.compile(r"^request failed(?: with status(?: code)? \d{3})?\.?$", re.I),
    re.compile(r"^(?:http )?(?:error )?\d{3}(?: error)?\.?$", re.I),
    re.compile(
        r"^(?:bad request|unauthorized|forbidden|not found|conflict|"
        r"internal server error|bad gateway|service unavailable|gateway timeout)\.?$",
        re.I,
    ),
    re.compile(
        r"^(?:an )?(?:unknown|unexpected|internal|network|generic) error(?: occurred)?\.?$", re.I
    ),
    re.compile(r"^(?:request|operation|call) (?:failed|errored)\.?$", re.I),
    re.compile(r"^error\.?$", re.I),
]

_GENERIC_EXACT = {
    "error",
    "failed",
    "validation",
    "response-validation",
    "request failed",
    "invalid",
    "invalid request",
    "unknown",
    "null",
}


def is_low_information(template: str) -> bool:
    """True when a template needs surrounding context to be diagnostically useful.

    Two cases qualify. First, strings too short to say anything — the original
    script relabelled these as "Likely validation error", and in the sample
    report that one bucket swallowed four unrelated messages and thirty quote
    IDs. Second, generic transport errors: "Request failed with status code 400"
    is emitted by every upstream in the system and identifies none of them.

    Flagging these does not discard them. The pipeline enriches them with the
    emitting service and the next-most-specific message in the payload, so
    genuinely different failures stay separate and each gets an honest count.
    """
    stripped = re.sub(r"<[A-Z_]+>", "", template).strip(" .,:;|-")
    if len(stripped) < 12:
        return True
    # Three tokens or fewer is a code identifier, not a sentence:
    # "response-validation boundError" names a failure mode but not which of the
    # several handlers that emit it produced this one. Prose of four tokens or
    # more ("Please select the option.") is specific enough to stand alone and
    # must not be split per handler.
    if len(stripped.split()) <= 3:
        return True
    lowered = stripped.lower()
    if lowered in _GENERIC_EXACT:
        return True
    return any(pattern.match(lowered) for pattern in _GENERIC_PATTERNS)
