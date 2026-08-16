"""The original algorithm, preserved verbatim in behaviour.

This exists so the eval can measure the new pipeline against the thing it
replaces on the same input, rather than against an assertion that it is better.
It is a faithful port of slack_error_parser.py: the same recursive scrape of any
`error`/`message` key, the same " || " join, the same "Likely validation error"
relabelling of short strings, and the same union-of-all-UUIDs quote attribution.

It is never imported by the pipeline. It is measurement apparatus.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

_UUID_LOOSE = re.compile(r"\b[a-f0-9]{8}-[a-f0-9\-]{4,}\b")


def _collect_error_and_message_strings(obj: Any, results: list[str] | None = None) -> list[str]:
    if results is None:
        results = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key in ("error", "message") and isinstance(val, str):
                results.append(val.strip())
            else:
                _collect_error_and_message_strings(val, results)
    elif isinstance(obj, list):
        for item in obj:
            _collect_error_and_message_strings(item, results)
    return results


def _extract_uuids(obj: Any, found: set[str] | None = None) -> set[str]:
    if found is None:
        found = set()
    if isinstance(obj, dict):
        for val in obj.values():
            _extract_uuids(val, found)
    elif isinstance(obj, list):
        for item in obj:
            _extract_uuids(item, found)
    elif isinstance(obj, str):
        found.update(_UUID_LOOSE.findall(obj))
    return found


def _normalize_combined_message(msgs: list[str]) -> str:
    msgs = [m.strip() for m in msgs if m.strip()]
    if not msgs:
        return ""
    for m in msgs:
        lower = m.lower()
        if "can't update crm contacts" in lower:
            return "Renewal Flow: Can't update CRM contacts"
        if re.search(r"total payable \([\d.]+\) does not match total paid \([\d.]+\)", lower):
            return "Renewal Flow: Total Payable does not match Total Paid"
    combined = " || ".join(sorted(set(msgs)))
    if len(combined) < 20:
        return f"Likely validation error || {combined}"
    return combined


def baseline_groups(messages: list[dict[str, Any]]) -> dict[str, list[tuple[str, list[str]]]]:
    """Reproduces extract_errors_with_quotes() from the original script."""
    error_groups: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)

    for msg in messages:
        texts_to_try: list[str] = []
        if "blocks" in msg:
            for block in msg["blocks"]:
                if block.get("type") != "rich_text":
                    continue
                for element in block.get("elements", []):
                    if element.get("type") == "rich_text_preformatted":
                        for inner in element.get("elements", []):
                            if inner.get("type") == "text":
                                texts_to_try.append(inner["text"])
        if msg.get("text"):
            texts_to_try.append(msg["text"])

        for text in texts_to_try:
            try:
                parsed = json.loads(text)
                messages_found = _collect_error_and_message_strings(parsed)
                quote_ids = _extract_uuids(parsed)
            except json.JSONDecodeError:
                messages_found = [
                    m.strip() for m in re.findall(r'"(?:error|message)":\s*"(.*?)"', text)
                ]
                quote_ids = _extract_uuids(text)

            if not messages_found:
                continue

            is_total_premium = any(
                "totalAmount is not allowed to be an empty array" in m for m in messages_found
            )
            is_partner_api = any(
                "Can't fetch quote from partner-api-validation" in m for m in messages_found
            )
            if is_total_premium or is_partner_api:
                group_key = (
                    "Renewal Flow: Can't fetch quote from partner-api-validation || "
                    "totalAmount is not allowed to be an empty array"
                )
            else:
                group_key = _normalize_combined_message(messages_found)

            if group_key:
                error_groups[group_key].append((group_key, sorted(quote_ids)))

    return dict(error_groups)


def baseline_labels(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map slack_ts -> baseline group key, for cluster-quality comparison."""
    labels: dict[str, str] = {}
    for msg in messages:
        single = baseline_groups([msg])
        if single:
            labels[str(msg.get("ts"))] = next(iter(single))
    return labels
