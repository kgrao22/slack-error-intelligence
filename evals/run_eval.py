"""Does the new pipeline actually beat the script it replaces?

Runs four arms over the same labelled input and prints a comparison table:

  1. baseline (exact string)   the original slack_error_parser.py grouping
  2. masking only              mask volatile tokens, then exact template match.
                               Isolates how much of the gain is free, before any
                               model is involved.
  3. masking + lexical vectors the full pipeline with the offline hashed n-gram
                               encoder. Runs with no API key, so CI can assert on it.
  4. masking + voyage          the full pipeline as deployed. Skipped unless
                               REI_VOYAGE_API_KEY is set.

Reporting arm 2 separately is deliberate. It would be easy to attribute the whole
improvement to "we added AI"; most of it is masking, and the honest thing is to
show where the gain comes from.

    python -m evals.run_eval
    python -m evals.run_eval --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rei.baseline import baseline_groups
from rei.config import load_settings
from rei.extract import extract_events
from rei.masking import normalize, template_hash
from rei.pipeline import Pipeline, build_templates
from rei.store import Store

from .metrics import AttributionQuality, ClusterQuality, attribution_quality, cluster_quality

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESSAGES = ROOT / "fixtures" / "slack_messages.json"
DEFAULT_LABELS = ROOT / "fixtures" / "labelled.json"


def load_dataset(
    messages_path: Path = DEFAULT_MESSAGES, labels_path: Path = DEFAULT_LABELS
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    messages = json.loads(messages_path.read_text())["messages"]
    labels = json.loads(labels_path.read_text())
    return messages, labels


# --- arms ------------------------------------------------------------------


def arm_baseline(messages: list[dict]) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Returns (ts -> group key, ts -> attributed quote IDs)."""
    assignment: dict[str, str] = {}
    attribution: dict[str, set[str]] = {}
    for msg in messages:
        groups = baseline_groups([msg])
        if not groups:
            continue
        key = next(iter(groups))
        assignment[str(msg["ts"])] = key
        quotes: set[str] = set()
        for _group, entries in groups.items():
            for _k, ids in entries:
                quotes.update(ids)
        attribution[str(msg["ts"])] = quotes
    return assignment, attribution


def arm_masking_only(messages: list[dict]) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Mask volatile tokens, then group by exact template. No model involved."""
    events = extract_events(messages)
    assignment = {e.slack_ts: template_hash(normalize(e.message)) for e in events}
    attribution = {e.slack_ts: ({e.quote_id} if e.quote_id else set()) for e in events}
    return assignment, attribution


def arm_enriched(messages: list[dict]) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Masking plus context enrichment of uninformative messages, exact match.

    Isolates the contribution of pulling the emitting service and the deeper
    message into generic transport errors, before any vector search happens.
    """
    events = extract_events(messages)
    _templates, _counts, _examples, by_template = build_templates(events)
    assignment: dict[str, str] = {}
    attribution: dict[str, set[str]] = {}
    for digest, grouped in by_template.items():
        for event in grouped:
            assignment[event.slack_ts] = digest
            attribution[event.slack_ts] = {event.quote_id} if event.quote_id else set()
    return assignment, attribution


def arm_pipeline(
    messages: list[dict], *, offline: bool, threshold: float
) -> tuple[dict[str, str], dict[str, set[str]]]:
    with tempfile.TemporaryDirectory() as tmp:
        settings = load_settings(
            db_path=str(Path(tmp) / "eval.db"),
            offline=offline,
            dry_run=True,
            # Both, because the pipeline selects the threshold matching the
            # encoder in use. Setting only one would silently leave the swept
            # value unused and make every row of the sweep identical.
            cluster_threshold=threshold,
            offline_cluster_threshold=threshold,
            enable_diagnosis=False,
        )
        with Store(settings.db_path) as store:
            pipeline = Pipeline(settings, store)
            now = datetime.now(tz=UTC)
            pipeline.run(messages, now, now)
            rows = store.conn.execute(
                "SELECT slack_ts, cluster_id, quote_id FROM events"
            ).fetchall()
    assignment = {row["slack_ts"]: f"c{row['cluster_id']}" for row in rows}
    attribution = {
        row["slack_ts"]: ({row["quote_id"]} if row["quote_id"] else set()) for row in rows
    }
    return assignment, attribution


# --- scoring ---------------------------------------------------------------


def score(
    assignment: dict[str, str],
    attribution: dict[str, set[str]],
    labels: dict[str, dict[str, str]],
) -> tuple[ClusterQuality, AttributionQuality]:
    shared = [ts for ts in labels if ts in assignment]
    truth = [labels[ts]["true_cluster"] for ts in shared]
    pred = [assignment[ts] for ts in shared]
    quality = cluster_quality(truth, pred)
    truth_quotes = {ts: labels[ts]["true_quote"] for ts in shared}
    return quality, attribution_quality(attribution, truth_quotes)


def run(threshold: float | None = None) -> dict[str, ClusterQuality]:
    messages, labels = load_dataset()
    settings = load_settings()
    # Arms 1-4 run on the offline encoder, so the offline threshold is the one
    # under test unless an explicit value is given.
    tau = threshold if threshold is not None else settings.offline_cluster_threshold

    arms: list[tuple[str, dict[str, str], dict[str, set[str]]]] = []
    arms.append(("1 baseline (exact string)", *arm_baseline(messages)))
    arms.append(("2  + token masking", *arm_masking_only(messages)))
    arms.append(("3  + context enrichment", *arm_enriched(messages)))
    arms.append(("4  + lexical vectors", *arm_pipeline(messages, offline=True, threshold=tau)))
    if os.environ.get("REI_VOYAGE_API_KEY"):
        arms.append(
            ("5  + voyage embeddings", *arm_pipeline(messages, offline=False, threshold=tau))
        )

    n_true = len({v["true_cluster"] for v in labels.values()})
    print(f"\nDataset: {len(messages)} messages, {n_true} true failure types, threshold {tau}\n")
    print("Cluster quality (1.000 is perfect). Each row adds one stage to the row above.")
    print(ClusterQuality.header())
    print("-" * 78)
    results: dict[str, ClusterQuality] = {}
    attributions: list[tuple[str, AttributionQuality]] = []
    for name, assignment, attribution in arms:
        quality, attr = score(assignment, attribution, labels)
        results[name] = quality
        attributions.append((name, attr))
        print(quality.as_row(name))

    print("\nQuote attribution (does the report blame the right quote?)")
    print(AttributionQuality.header())
    print("-" * 72)
    for name, attr in attributions:
        print(attr.as_row(name))

    if "5  + voyage embeddings" not in results:
        print("\n(arm 5 skipped: set REI_VOYAGE_API_KEY to score real embeddings)")
    print()
    return results


def sweep(offline: bool = True) -> None:
    """Threshold sweep, used by `rei tune-threshold`."""
    messages, labels = load_dataset()
    print(f"\nThreshold sweep ({'offline lexical' if offline else 'voyage'} vectors)\n")
    print(f"{'tau':>6} {'homog.':>8} {'complet.':>9} {'V':>7} {'#clusters':>10}")
    print("-" * 44)
    best = (0.0, 0.0)
    for tau in [round(0.60 + 0.02 * i, 2) for i in range(21)]:
        assignment, _attr = arm_pipeline(messages, offline=offline, threshold=tau)
        quality, _ = score(assignment, {}, labels)
        print(
            f"{tau:>6.2f} {quality.homogeneity:>8.3f} {quality.completeness:>9.3f} "
            f"{quality.v_measure:>7.3f} {quality.n_pred_clusters:>10d}"
        )
        if quality.v_measure > best[1]:
            best = (tau, quality.v_measure)
    print(f"\nbest V-measure {best[1]:.3f} at threshold {best[0]:.2f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate clustering quality.")
    parser.add_argument("--sweep", action="store_true", help="Sweep the similarity threshold.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--json", dest="json_path", default=None, help="Also write results as JSON, for CI gates."
    )
    args = parser.parse_args()
    if args.sweep:
        sweep()
        return

    results = run(args.threshold)
    if args.json_path:
        payload = {
            name: {
                "homogeneity": q.homogeneity,
                "completeness": q.completeness,
                "v_measure": q.v_measure,
                "adjusted_rand": q.adjusted_rand,
                "n_pred_clusters": q.n_pred_clusters,
                "n_true_clusters": q.n_true_clusters,
            }
            for name, q in results.items()
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_path}")


if __name__ == "__main__":
    main()
