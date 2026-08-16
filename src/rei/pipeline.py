"""Orchestration: messages in, DailyReport out.

Order matters and is worth reading once:

    fetch → extract → mask → embed (cached) → cluster → label (new only) →
    persist → aggregate → render

Counting happens in the aggregate step, from the events table, after clustering
has settled. The LLM has already run and cannot influence any number.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from .clustering import Assignment, IncrementalClusterer
from .config import Settings
from .embeddings import Embedder
from .extract import extract_events
from .llm import Labeller
from .masking import is_low_information, normalize, strip_embedded_json, template_hash
from .models import ClusterReport, DailyReport, ErrorEvent
from .rag import Retriever
from .store import Store

log = logging.getLogger(__name__)


def build_templates(
    events: list[ErrorEvent],
) -> tuple[dict[str, str], dict[str, int], dict[str, str], dict[str, list[ErrorEvent]]]:
    """Mask every message into a template, enriching the uninformative ones.

    Returns (templates, counts, examples, events_by_template). Kept at module
    level and free of state so the eval harness can score this step on its own
    and show how much of the improvement comes from here rather than from the
    model.
    """
    templates: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, str] = {}
    by_template: dict[str, list[ErrorEvent]] = defaultdict(list)

    for event in events:
        text = normalize(event.message)
        # A bare "Request failed with status code 400" or "response-validation"
        # says nothing on its own, and in the current report a dozen unrelated
        # upstream failures hid behind one such string. Rather than dropping them
        # into a catch-all bucket, pull in the emitting handler and the service's
        # own error taxonomy so the embedding has something real to separate on.
        if is_low_information(text):
            parts = [text]
            if event.http_status:
                parts.append(f"[status {event.http_status}]")
            if event.service:
                parts.append(f"from {event.service}")
            if event.cause_type:
                parts.append(f"cause {event.cause_type}")
            elif event.error_type and event.error_type.lower() != "unknown":
                parts.append(f"type {event.error_type}")

            # Only the two most specific sibling messages, each bounded, so a
            # long and highly variable payload cannot explode the template space.
            extras: list[str] = []
            for sibling in event.all_messages[1:4]:
                masked = normalize(sibling)[:160]
                if masked and not is_low_information(masked) and masked != text:
                    extras.append(masked)
                if len(extras) == 2:
                    break
            if extras:
                parts.append("| " + " | ".join(dict.fromkeys(extras)))
            text = " ".join(parts)

        digest = template_hash(text)
        templates[digest] = text
        counts[digest] += 1
        examples.setdefault(digest, event.message[:600])
        by_template[digest].append(event)

    return templates, dict(counts), examples, dict(by_template)


def _dominant_hints(events: list[ErrorEvent]) -> dict[str, str]:
    """The most common service-supplied metadata across a cluster's events."""
    hints: dict[str, str] = {}
    for field in ("error_type", "cause_type", "cause_name", "service"):
        values = [getattr(e, field) for e in events if getattr(e, field)]
        if values:
            hints[field] = Counter(values).most_common(1)[0][0]
    return hints


class Pipeline:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.embedder = Embedder(settings, store)
        # The threshold has to match the encoder actually in use, not the one
        # configured — falling back to the offline encoder without also falling
        # back to its threshold would silently wreck the clustering.
        threshold = (
            settings.offline_cluster_threshold
            if self.embedder.is_offline
            else settings.cluster_threshold
        )
        self.clusterer = IncrementalClusterer(store, threshold)
        self.labeller = Labeller(settings)
        self.retriever = Retriever(store)

    # -- steps --------------------------------------------------------------

    def label_new_clusters(
        self, assignments: list[Assignment], by_template: dict[str, list[ErrorEvent]]
    ) -> int:
        """Label clusters whose meaning changed in this run.

        That is both clusters created now *and* existing clusters that a new
        template joined. The second case matters: a cluster seeded by
        "card has insufficient funds" was still carrying that label after
        expired-card and transaction-not-allowed declines joined it, so the
        report named a specific decline reason while counting five of them.
        Relabelling only fires when membership actually changed, so a steady day
        still makes zero model calls.
        """
        changed = [a for a in assignments if a.is_new_template]
        if not changed:
            return 0

        # One entry per cluster, not per template — several new templates can
        # land in the same cluster in a single run.
        seen: set[int] = set()
        payload: list[dict[str, Any]] = []
        neighbours_by_index: dict[int, Any] = {}
        for assignment in changed:
            if assignment.cluster_id in seen:
                continue
            seen.add(assignment.cluster_id)
            index = len(payload)
            events = by_template.get(assignment.template_hash, [])
            # For an existing cluster, show the model everything it now contains,
            # so the label describes the whole group rather than its seed.
            templates = (
                [assignment.template_text]
                if assignment.is_new_cluster
                else self.store.cluster_templates(assignment.cluster_id, limit=8)
                or [assignment.template_text]
            )
            payload.append(
                {
                    "index": index,
                    "cluster_id": assignment.cluster_id,
                    "templates": templates,
                    "examples": self.store.cluster_exemplars(assignment.cluster_id, limit=3)
                    or [assignment.template_text],
                    # The renewal service publishes its own error taxonomy. Pass
                    # it through rather than asking a model to re-derive it.
                    "hints": _dominant_hints(events),
                }
            )
            # Never show a cluster itself among its own "similar historical
            # clusters" — when relabelling, the cluster the template just joined
            # is by construction the nearest neighbour, and offering it back as a
            # duplicate candidate produced results like "cluster 17 is a
            # duplicate of 17".
            neighbours_by_index[index] = self.retriever.neighbours(
                [(cid, sim) for cid, sim in assignment.neighbours if cid != assignment.cluster_id]
            )

        results = self.labeller.label_clusters(payload, neighbours_by_index)

        for item, entry in zip(results, payload, strict=False):
            self.store.set_cluster_label(
                cluster_id=entry["cluster_id"],
                label=item.get("label") or entry["templates"][0][:88],
                category=item.get("category", "unknown"),
                severity=item.get("severity", "unknown"),
                root_cause=item.get("root_cause"),
                suggested_action=item.get("suggested_action"),
                labeled_by="rule" if self.labeller.offline else "claude",
            )
            duplicate_of = item.get("duplicate_of")
            if duplicate_of:
                log.info(
                    "Claude flagged cluster %s as a duplicate of %s (recorded, not auto-merged)",
                    entry["cluster_id"],
                    duplicate_of,
                )
        return len(payload)

    def persist(
        self,
        assignments: list[Assignment],
        by_template: dict[str, list[ErrorEvent]],
        examples: dict[str, str],
    ) -> None:
        # Cumulative counters may only advance for events we have not seen
        # before, or an overlapping window inflates every historical baseline.
        already_ingested = self.store.existing_event_ids(
            [event.slack_ts for events in by_template.values() for event in events]
        )

        for assignment in assignments:
            events = by_template.get(assignment.template_hash, [])
            fresh = sum(1 for e in events if e.slack_ts not in already_ingested)
            seen_at = (
                max(e.occurred_at for e in events).isoformat()
                if events
                else datetime.now(tz=UTC).isoformat()
            )
            self.store.upsert_template(
                template_hash=assignment.template_hash,
                text=assignment.template_text,
                example_message=examples.get(assignment.template_hash, ""),
                cluster_id=assignment.cluster_id,
                seen_at=seen_at,
                added_events=fresh,
            )
            for event in events:
                self.store.insert_event(
                    slack_ts=event.slack_ts,
                    occurred_at=event.occurred_at.isoformat(),
                    message=event.message,
                    template_hash=assignment.template_hash,
                    cluster_id=assignment.cluster_id,
                    quote_id=event.quote_id,
                    quote_id_confidence=event.quote_id_confidence,
                    correlation_id=event.correlation_id,
                    http_status=event.http_status,
                    service=event.service,
                    raw_excerpt=event.raw_excerpt,
                )
            self.store.bump_cluster(assignment.cluster_id, fresh, seen_at)
        self.store.commit()

    def aggregate(
        self,
        assignments: list[Assignment],
        by_template: dict[str, list[ErrorEvent]],
        window_start: datetime,
        window_end: datetime,
        messages_scanned: int,
    ) -> DailyReport:
        """All counting happens here, in Python and SQL, never in the model."""
        grouped: dict[int, list[ErrorEvent]] = defaultdict(list)
        templates_by_cluster: dict[int, list[str]] = defaultdict(list)
        created_now: set[int] = {a.cluster_id for a in assignments if a.is_new_cluster}

        for assignment in assignments:
            grouped[assignment.cluster_id].extend(by_template.get(assignment.template_hash, []))
            templates_by_cluster[assignment.cluster_id].append(assignment.template_text)

        entries: list[ClusterReport] = []
        for cluster_id, events in grouped.items():
            cluster = self.store.get_cluster(cluster_id)
            if cluster is None:
                continue
            quotes = sorted({e.quote_id for e in events if e.quote_id})
            # Real IDs are kept — an on-call engineer wants a clickable
            # example — but a serialised AxiosError blob would swamp the
            # Slack card, so it is trimmed the same way templates are.
            samples = [strip_embedded_json(e.message)[:400] for e in events[:3]]
            entries.append(
                ClusterReport(
                    cluster=cluster,
                    event_count=len(events),
                    unique_quotes=quotes,
                    sample_messages=samples,
                    templates=sorted(set(templates_by_cluster[cluster_id])),
                    is_new=cluster_id in created_now,
                    prior_daily_mean=self.store.prior_daily_mean(
                        cluster_id, window_start.isoformat(), days=7
                    ),
                )
            )

        entries.sort(key=lambda e: (-e.event_count, -e.unique_quote_count, e.cluster.label))
        window_total = sum(e.event_count for e in entries)
        for entry in entries:
            entry.window_total = window_total

        return DailyReport(
            run_date=window_end.strftime("%a %d %b %Y"),
            window_start=window_start,
            window_end=window_end,
            messages_scanned=messages_scanned,
            events_extracted=sum(e.event_count for e in entries),
            unique_templates=len({a.template_hash for a in assignments}),
            clusters=entries,
        )

    # -- entry point --------------------------------------------------------

    def run(
        self,
        messages: list[dict[str, Any]],
        window_start: datetime,
        window_end: datetime,
    ) -> DailyReport:
        run_date = window_end.strftime("%Y-%m-%d")
        self.store.start_run(run_date)

        events = extract_events(messages)
        log.info("extracted %d error events from %d messages", len(events), len(messages))

        templates, counts, examples, by_template = build_templates(events)
        log.info("%d distinct templates after masking", len(templates))

        vectors = self.embedder.embed(templates)
        assignments = self.clusterer.assign(
            templates, vectors, counts, seen_at=window_end.isoformat()
        )

        new_count = self.label_new_clusters(assignments, by_template)
        self.persist(assignments, by_template, examples)
        report = self.aggregate(assignments, by_template, window_start, window_end, len(messages))

        self.store.finish_run(
            run_date,
            messages_scanned=len(messages),
            events_extracted=report.events_extracted,
            new_clusters=new_count,
            embed_calls=self.embedder.api_calls,
            llm_calls=self.labeller.api_calls,
        )
        log.info(
            "run complete: %d clusters (%d new), %d embed calls, %d llm calls",
            len(report.clusters),
            new_count,
            self.embedder.api_calls,
            self.labeller.api_calls,
        )
        return report
