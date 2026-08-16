"""Report on how well the extractor handles a real Slack dump — safely.

Everything printed is passed through the masking layer first, so UUIDs, emails,
Stripe customer IDs, amounts and timestamps come out as <UUID>, <EMAIL>,
<STRIPE_ID>, <NUM>. Label and key *names* are printed as-is because they are
schema, not customer data. The output is safe to paste into a chat or a ticket.

    python scripts/inspect_payloads.py fixtures/slack_dump.json

Section 6 matters most: every message listed there is an error that would
silently vanish from the daily report.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rei.baseline import baseline_groups
from rei.extract import (
    _loads,
    _parse_candidates,
    _walk,
    extract_events,
    is_document_noise,
    is_reportable,
    labelled_meta,
    labelled_pairs,
    parse_error_value,
)
from rei.masking import is_low_information, normalize
from rei.pipeline import build_templates

RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main(path: str) -> None:
    raw = json.loads(Path(path).read_text())
    messages = raw["messages"] if isinstance(raw, dict) else raw
    print(f"Loaded {len(messages)} messages from {path}")

    documents = [m for m in messages if is_document_noise(m)]
    system = [m for m in messages if m.get("subtype") in {"channel_join", "channel_leave"}]
    candidates = [m for m in messages if is_reportable(m)]
    other = len(messages) - len(candidates) - len(documents) - len(system)
    events = extract_events(messages)
    by_ts = {e.slack_ts: e for e in events}
    missed = [m for m in candidates if str(m.get("ts")) not in by_ts]

    # --- 1 -----------------------------------------------------------------
    section("1. COVERAGE")
    print(f"messages in dump          : {len(messages)}")
    print(f"  policy/endorsement docs : {len(documents)}")
    print(f"  slack system events     : {len(system)}")
    print(f"  other non-error payloads: {other}  (quote answers, etc)")
    print(f"candidate error messages  : {len(candidates)}")
    print(f"errors extracted          : {len(events)}")
    print(f"GENUINELY MISSED          : {len(missed)}")
    if candidates:
        print(f"coverage of candidates    : {len(events) / len(candidates) * 100:.1f}%")

    # --- 2 -----------------------------------------------------------------
    labels: Counter[str] = Counter()
    subfields: Counter[str] = Counter()
    json_keys: Counter[str] = Counter()
    with_pairs = 0

    for msg in candidates:
        pairs = labelled_pairs(msg)
        if pairs:
            with_pairs += 1
            for label, value in pairs:
                clean = label.lower().strip(": ") or "(no label)"
                labels[clean] += 1
                if clean in {"error", "err", "exception"}:
                    subfields.update(parse_error_value(value).keys())
            continue
        for text in _parse_candidates(msg):
            parsed = _loads(text)
            if parsed is None:
                continue
            for key, _path, val in _walk(parsed):
                if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                    json_keys[key] += 1
            break

    section("2. PAYLOAD SHAPE  (label and key names only, no values)")
    print(f"messages in label/value rich-text format : {with_pairs}")
    print(f"messages parsed as JSON instead          : {len(candidates) - with_pairs}\n")
    print("labels found:")
    for name, count in labels.most_common(20):
        print(f"  {count:5d}  {name}")
    print("\nsub-fields inside the `error` value:")
    for name, count in subfields.most_common(12):
        print(f"  {count:5d}  {name}")
    if json_keys:
        print("\nJSON keys (non-labelled messages):")
        for name, count in json_keys.most_common(20):
            print(f"  {count:5d}  {name}")

    # --- 3 -----------------------------------------------------------------
    section("3. SERVICE-SUPPLIED METADATA  (drives categories without a model)")
    for field, title in (
        ("error_type", "error.type"),
        ("cause_type", "error.cause.type"),
        ("service", "methodName"),
    ):
        values = Counter(getattr(e, field) for e in events if getattr(e, field))
        print(f"\n{title}:")
        if not values:
            print("  (none found)")
        for name, count in values.most_common(15):
            print(f"  {count:5d}  {name}")

    # --- 4 -----------------------------------------------------------------
    section("4. QUOTE ATTRIBUTION")
    confidence = Counter(e.quote_id_confidence for e in events)
    meaning = {
        1.0: "named quoteId key",
        0.8: "matched by prose context, e.g. 'for quote: <uuid>'",
        0.5: "only one candidate UUID after excluding correlationId",
        0.0: "could not attribute -- reported honestly as unattributed",
    }
    for score in sorted(confidence, reverse=True):
        print(f"  {confidence[score]:5d}  conf {score}  {meaning.get(score, '')}")
    attributed = sum(1 for e in events if e.quote_id)
    if events:
        print(f"\n  {attributed}/{len(events)} attributed ({attributed / len(events) * 100:.1f}%)")
    unattributed = [e for e in events if not e.quote_id]
    if unattributed:
        print("\nUnattributed samples (masked):")
        for event in unattributed[:5]:
            print(f"  - {normalize(event.message)[:130]}")
            if event.all_messages[1:]:
                print(f"      context: {normalize(event.all_messages[1])[:130]}")

    # --- 5 -----------------------------------------------------------------
    templates, counts, _examples, _by = build_templates(events)
    section("5. TEMPLATES AFTER MASKING  (fully masked, safe to share)")
    print(f"{len(events)} errors collapsed into {len(templates)} distinct templates\n")
    ordered = sorted(templates.items(), key=lambda kv: -counts.get(kv[0], 0))
    for digest, text in ordered[:30]:
        print(f"  {counts.get(digest, 0):5d} x  {text[:160]}")
    if len(ordered) > 30:
        print(f"  ... and {len(ordered) - 30} more")

    singletons = sum(1 for d in templates if counts.get(d, 0) == 1)
    print(f"\n  {singletons} templates occur exactly once.")
    print("  A high number here means enrichment is pulling in variable text and")
    print("  fragmenting the taxonomy -- tell me if this is more than ~25% of the total.")

    # --- 6 -----------------------------------------------------------------
    section("6. GENUINELY MISSED MESSAGES  (most important section)")
    if not missed:
        print("None. Every candidate error message produced an event.")
    else:
        print(f"{len(missed)} produced nothing. Diagnosis per message:\n")
        for msg in missed[:12]:
            pairs = labelled_pairs(msg)
            meta = labelled_meta(pairs) if pairs else {}
            print(f"  labels found : {[label for label, _v in pairs] or 'NONE'}")
            print(f"  meta         : {meta or 'none'}")
            for label, value in pairs:
                if label.lower().strip(": ") in {"error", "err", "exception"}:
                    parsed = parse_error_value(value)
                    print(f"  error parsed : keys={sorted(parsed)}")
                    print(f"  error head   : {normalize(value[:200])}")
            print(f"  text         : {normalize((msg.get('text') or '')[:200]) or '(no text)'}\n")
        if len(missed) > 12:
            print(f"  ... and {len(missed) - 12} more")

    # --- 7 -----------------------------------------------------------------
    generic = [e for e in events if is_low_information(normalize(e.message))]
    stranded = [
        e for e in generic if not e.service and not e.cause_type and len(e.all_messages) < 2
    ]
    section("7. GENERIC MESSAGES")
    print(f"{len(generic)} of {len(events)} errors are transport noise on their own.")
    print(f"{len(stranded)} of those have no handler, no cause and no second message,")
    print("so nothing can separate them. Ideally this is zero.")

    # --- 8 -----------------------------------------------------------------
    old = baseline_groups(messages)
    old_quotes = {q for entries in old.values() for _k, ids in entries for q in ids}
    new_quotes = {e.quote_id for e in events if e.quote_id}
    section("8. OLD SCRIPT vs NEW PIPELINE ON THIS DATA")
    print(f"old script   : {len(old)} groups, claims {len(old_quotes)} affected quotes")
    print(f"new pipeline : {len(templates)} templates, attributes {len(new_quotes)} quotes")
    if new_quotes:
        print(f"\nold script over-claimed quotes by {len(old_quotes) / len(new_quotes):.1f}x")

    print(f"\n{RULE}\nDone. This entire output is masked and safe to share.\n{RULE}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/slack_dump.json")
