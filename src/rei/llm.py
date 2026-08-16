"""Claude labelling layer.

Scope discipline, stated plainly because it is the part worth defending in a
review: the model names and categorises clusters and, optionally, proposes a root
cause. It never counts anything. Frequencies, unique-quote tallies and
new-vs-recurring are computed in SQL from the events table. An LLM that is asked
to both interpret and tally will quietly get the tallies wrong, and a report whose
numbers cannot be trusted is worse than no report.

The model is called only for clusters that are new on a given day. On a steady
day that is zero calls and the pipeline costs nothing.

Structured output is obtained via forced tool use rather than "reply in JSON",
so the response is schema-validated by the API instead of by a regex on our side.
"""

from __future__ import annotations

import json
import logging
import re

from .config import Settings
from .http import HttpClient
from .rag import RetrievedCluster

log = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

CATEGORIES = [
    "payment",
    "validation",
    "timeout",
    "integration",
    "data",
    "auth",
    "config",
    "unknown",
]
SEVERITIES = ["blocking", "degraded", "cosmetic", "unknown"]

SYSTEM_PROMPT = """You label clusters of production error messages from a subscription \
renewal pipeline. Each cluster is a group of error strings that an embedding \
model judged to be the same underlying failure.

For each cluster you are given:
  - the masked templates in the cluster (volatile IDs replaced with placeholders \
like <UUID>, <EMAIL>, <STRIPE_ID>, <NUM>)
  - real example messages
  - the most similar clusters already in the historical corpus, with any human \
resolutions recorded against them

Your job for each cluster:
  1. label: a short, specific, human-readable title an on-call engineer can scan. \
Describe the failure, not the symptom string. Prefer "Stripe customer has no default \
payment method" over "Could not find a default payment method for Stripe user \
<STRIPE_ID>". Keep it under 80 characters. Do not include IDs or placeholders.
  2. category: one of payment, validation, timeout, integration, data, auth, config, \
unknown.
  3. severity: blocking if the renewal cannot complete, degraded if it completes with \
problems or retries, cosmetic if it is noise, unknown if you cannot tell.
  4. duplicate_of: if this cluster is clearly the same failure as one of the supplied \
historical clusters and was only separated by wording, return that cluster id. \
Otherwise null. Be conservative: only say duplicate when the underlying cause is the \
same, not merely when the wording is similar.

Rules:
  - Some clusters carry metadata the renewal service published itself, such as \
error_type (payment-failure, unknown, …), cause_type (card-declined, …) and the \
emitting handler. That metadata is authoritative. Where it maps cleanly onto a \
category, use it rather than re-deriving one from the text, and use cause_type to \
make the label specific.
  - Never invent counts, frequencies or dates. You are not given them and must not \
guess them.
  - If the templates are too generic to identify a failure (for example a bare \
"Request failed with status code 400" with no context), say so in the label rather \
than inventing a cause. "Unattributed HTTP 400 from renewal request" is a good label \
for that case.
  - Base everything on the supplied text only."""

DIAGNOSIS_FIELDS = """
  5. root_cause: one or two sentences on the most likely cause, grounded strictly in \
the supplied templates, examples and historical resolutions. If the evidence does not \
support a specific cause, say what additional information would be needed.
  6. suggested_action: one concrete next step for an engineer."""


def _tool_schema(include_diagnosis: bool) -> dict:
    properties = {
        "index": {"type": "integer", "description": "The index of the cluster as supplied."},
        "label": {"type": "string", "maxLength": 90},
        "category": {"type": "string", "enum": CATEGORIES},
        "severity": {"type": "string", "enum": SEVERITIES},
        "duplicate_of": {
            "type": ["integer", "null"],
            "description": "Historical cluster id this duplicates, or null.",
        },
    }
    required = ["index", "label", "category", "severity"]
    if include_diagnosis:
        properties["root_cause"] = {"type": "string"}
        properties["suggested_action"] = {"type": "string"}
        required += ["root_cause", "suggested_action"]

    return {
        "name": "record_labels",
        "description": "Record the canonical label for every cluster supplied.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clusters": {
                    "type": "array",
                    # Bounded: the model has been observed returning 149 entries
                    # for a single input cluster.
                    "maxItems": 64,
                    "items": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                }
            },
            "required": ["clusters"],
        },
    }


class Labeller:
    """Labels clusters. Falls back to a deterministic rule-based labeller when
    offline or when no Anthropic key is configured, so the pipeline always
    produces a complete report."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.api_calls = 0
        self.offline = settings.offline or not settings.anthropic_api_key.get_secret_value()
        if self.offline and not settings.offline:
            log.warning("No REI_ANTHROPIC_API_KEY set — using the rule-based labeller.")

    def label_clusters(
        self,
        clusters: list[dict],
        neighbours_by_index: dict[int, list[RetrievedCluster]],
    ) -> list[dict]:
        """`clusters` is a list of {index, templates: [...], examples: [...]}"""
        if not clusters:
            return []
        if self.offline:
            return [_rule_based_label(c) for c in clusters]
        try:
            return self._call_claude(clusters, neighbours_by_index)
        except Exception as exc:
            log.error("Claude labelling failed (%s); falling back to rules", exc)
            return [_rule_based_label(c) for c in clusters]

    def _call_claude(
        self, clusters: list[dict], neighbours_by_index: dict[int, list[RetrievedCluster]]
    ) -> list[dict]:
        blocks: list[str] = []
        for cluster in clusters:
            index = cluster["index"]
            lines = [f"### Cluster {index}", "Masked templates:"]
            lines += [f"  - {t[:400]}" for t in cluster["templates"][:6]]
            lines.append("Example messages:")
            lines += [f"  - {e[:400]}" for e in cluster["examples"][:4]]
            hints = cluster.get("hints") or {}
            if hints:
                lines.append("Metadata published by the renewal service itself:")
                lines += [f"  {key}: {value}" for key, value in hints.items()]
            lines.append("Similar historical clusters:")
            retrieved = neighbours_by_index.get(index, [])
            lines.append(
                "\n".join(r.render() for r in retrieved)
                if retrieved
                else "  (none above the similarity floor)"
            )
            blocks.append("\n".join(lines))

        system = SYSTEM_PROMPT + (DIAGNOSIS_FIELDS if self.settings.enable_diagnosis else "")
        tool = _tool_schema(self.settings.enable_diagnosis)
        payload = {
            "model": self.settings.anthropic_model,
            "max_tokens": 4096,
            "system": system,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "record_labels"},
            "messages": [
                {
                    "role": "user",
                    # The count is stated explicitly at both ends. Given a single
                    # cluster the model has been observed returning 149 and 156
                    # entries; naming the exact expected shape stops it.
                    "content": (
                        f"Label the {len(clusters)} cluster(s) below.\n\n"
                        + "\n\n".join(blocks)
                        + f"\n\nReturn exactly {len(clusters)} object(s) — one per cluster — "
                        f"with index values {', '.join(str(c['index']) for c in clusters)} "
                        "and nothing else."
                    ),
                }
            ],
        }
        headers = {
            "x-api-key": self.settings.anthropic_api_key.get_secret_value(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        with HttpClient("anthropic", timeout=120.0) as client:
            response = client.request_json(
                "POST", ANTHROPIC_URL, headers=headers, json_body=payload
            )
        self.api_calls += 1

        for block in response.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "record_labels":
                results = block["input"].get("clusters", [])
                log.info("Claude labelled %d clusters", len(results))
                return _fill_gaps(results, clusters)

        raise RuntimeError(f"no tool_use block in response: {json.dumps(response)[:400]}")


def _fill_gaps(results: list, clusters: list[dict]) -> list[dict]:
    """Guarantee exactly one well-formed result per input cluster.

    A schema-validated tool call is not a guarantee about semantics. On a real
    run the model was handed one cluster and returned an array of 149 entries,
    one of which was a bare string rather than an object — enough to crash the
    merge and drop the whole batch to rule-based labels. Anything that is not a
    dict carrying an in-range integer index is discarded here, and the first
    result for a given index wins.
    """
    valid = len(clusters)
    by_index: dict[int, dict] = {}
    discarded = 0
    for item in results:
        if not isinstance(item, dict):
            discarded += 1
            continue
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < valid or index in by_index:
            discarded += 1
            continue
        by_index[index] = item

    if discarded:
        log.warning(
            "discarded %d malformed or out-of-range label result(s) for %d cluster(s)",
            discarded,
            valid,
        )
    return [by_index.get(c["index"]) or _rule_based_label(c) for c in clusters]


# --- deterministic fallback ------------------------------------------------

# Kept short on purpose: the rule name is only the prefix, and every character
# spent here is a character unavailable to the distinguishing detail appended by
# _qualify().
_RULES: list[tuple[re.Pattern[str], str, str, str]] = [
    (
        re.compile(r"payment method|stripe|card declin|charge fail", re.I),
        "payment",
        "blocking",
        "Payment method problem",
    ),
    (
        re.compile(r"timeout|timed out|deadline exceeded", re.I),
        "timeout",
        "degraded",
        "Downstream timeout",
    ),
    (
        re.compile(r"not allowed|must be|is required|validation|invalid|please select", re.I),
        "validation",
        "blocking",
        "Validation failure",
    ),
    (
        re.compile(r"status code 5\d\d|internal server error", re.I),
        "integration",
        "blocking",
        "Upstream server error",
    ),
    (
        re.compile(r"status code 4\d\d|bad request", re.I),
        "integration",
        "degraded",
        "Upstream rejected request",
    ),
    (re.compile(r"unauthor|forbidden|token|credential", re.I), "auth", "blocking", "Auth failure"),
    (
        re.compile(r"not found|failed to get|could not find|missing", re.I),
        "data",
        "blocking",
        "Missing record",
    ),
]


def _elide(text: str, budget: int) -> str:
    """Shorten to `budget` characters keeping both ends.

    A plain prefix is wrong here: the token that distinguishes two otherwise
    identical templates is usually at the tail — the emitting service, the
    offending field. "Request failed with status code 400 from quote-docs-service"
    and "... from pricing-engine" truncate to the same prefix and become
    indistinguishable, so keep the head and the tail and drop the middle.
    """
    if len(text) <= budget:
        return text
    head_budget = max(int(budget * 0.55), 1)
    tail_budget = max(budget - head_budget - 1, 1)
    head = text[:head_budget].rsplit(" ", 1)[0] if " " in text[:head_budget] else text[:head_budget]
    tail = text[-tail_budget:]
    if " " in tail:
        tail = tail.split(" ", 1)[1]
    return f"{head.strip()}…{tail.strip()}"


def _qualify(label: str, template: str) -> str:
    """Keep rule-based labels distinct.

    The rules are coarse by design, so several unrelated clusters match the same
    one — two different validators both look like "validation". A report with two
    lines reading "Validation failure" is worse than useless, so the
    distinguishing part of the template is appended. Claude produces distinct
    labels on its own; this only guards the fallback path.
    """
    cleaned = re.sub(r"\s*<[A-Z_]+>\s*", " ", " ".join(template.split())).strip(" .,:;|-")
    budget = max(88 - len(label) - 2, 0)
    if not cleaned or budget < 10:
        return label
    hint = _elide(cleaned, budget).strip(" .,:;|-")
    if len(hint) < 8 or hint.lower() in label.lower():
        return label
    return f"{label}: {hint}"


# The renewal service's own error taxonomy, mapped onto report categories. This
# is authoritative — it comes from the code that raised the error — so it wins
# over any inference from the message text.
_TYPE_CATEGORY: dict[str, tuple[str, str]] = {
    "payment-failure": ("payment", "blocking"),
    "card-declined": ("payment", "blocking"),
    "stripeserviceerror": ("payment", "blocking"),
    "response-validation": ("validation", "blocking"),
    "validation": ("validation", "blocking"),
    "timeout": ("timeout", "degraded"),
    "auth": ("auth", "blocking"),
    "not-found": ("data", "blocking"),
}


def _from_hints(hints: dict) -> tuple[str, str] | None:
    for key in ("cause_type", "error_type", "cause_name"):
        value = str(hints.get(key) or "").strip().lower()
        if value and value in _TYPE_CATEGORY:
            return _TYPE_CATEGORY[value]
    return None


def _rule_based_label(cluster: dict) -> dict:
    templates = cluster.get("templates") or ["Unclassified error"]
    hints = cluster.get("hints") or {}
    text = " ".join(templates[:3])

    from_service = _from_hints(hints)
    for pattern, category, severity, label in _RULES:
        if pattern.search(text):
            if from_service:
                category, severity = from_service
            return {
                "index": cluster["index"],
                "label": _qualify(label, templates[0]),
                "category": category,
                "severity": severity,
                "duplicate_of": None,
            }

    collapsed = " ".join(templates[0].split())
    category, severity = from_service or ("unknown", "unknown")
    return {
        "index": cluster["index"],
        "label": collapsed[:88] + ("…" if len(collapsed) > 88 else ""),
        "category": category,
        "severity": severity,
        "duplicate_of": None,
    }
