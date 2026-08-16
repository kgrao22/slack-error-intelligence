from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from rei.blocks import MAX_BLOCKS, MAX_SECTION_CHARS, build_report_blocks, build_thread_text
from rei.config import load_settings
from rei.pipeline import Pipeline
from rei.store import Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_messages() -> list[dict]:
    return json.loads((FIXTURES / "slack_messages.json").read_text())["messages"]


def error_count() -> int:
    """Errors only. The fixture also contains endorsement documents, which share
    the channel with errors and must be filtered rather than counted."""
    return len(json.loads((FIXTURES / "labelled.json").read_text()))


def make_settings(tmp: str, **kw):
    return load_settings(db_path=str(Path(tmp) / "t.db"), offline=True, dry_run=True, **kw)


class TestPipelineEndToEnd(unittest.TestCase):
    def test_offline_run_produces_a_report(self):
        messages = load_messages()
        with tempfile.TemporaryDirectory() as tmp:
            settings = make_settings(tmp)
            with Store(settings.db_path) as store:
                report = Pipeline(settings, store).run(
                    messages, datetime.now(tz=UTC), datetime.now(tz=UTC)
                )
        self.assertEqual(report.messages_scanned, len(messages))
        self.assertEqual(report.events_extracted, error_count())
        self.assertLess(report.events_extracted, len(messages))  # documents filtered
        self.assertGreater(len(report.clusters), 5)
        # Every cluster on day one is new, by definition.
        self.assertEqual(len(report.new_clusters), len(report.clusters))
        # Counts must reconcile: cluster totals equal events extracted.
        self.assertEqual(sum(c.event_count for c in report.clusters), report.events_extracted)

    def test_cluster_ids_are_stable_across_runs(self):
        """The property that makes 'new' and 'trending' meaningful at all.

        Keyed on template hash rather than label: the label is descriptive text
        that may legitimately be rewritten, while the template-to-cluster mapping
        is the identity guarantee downstream consumers depend on.
        """
        messages = load_messages()
        half = len(messages) // 2
        with tempfile.TemporaryDirectory() as tmp:
            settings = make_settings(tmp)
            with Store(settings.db_path) as store:
                pipeline = Pipeline(settings, store)
                now = datetime.now(tz=UTC)
                pipeline.run(messages[:half], now, now)
                before = dict(
                    store.conn.execute("SELECT template_hash, cluster_id FROM templates").fetchall()
                )
                pipeline.run(messages[half:], now, now)
                after = dict(
                    store.conn.execute("SELECT template_hash, cluster_id FROM templates").fetchall()
                )

        self.assertTrue(before, "first pass produced no templates")
        for template_hash, cluster_id in before.items():
            self.assertEqual(
                after[template_hash],
                cluster_id,
                f"cluster id moved for template {template_hash}",
            )

    def test_labels_are_unique_per_cluster(self):
        """Two report lines with the same label are indistinguishable to a reader."""
        messages = load_messages()
        with tempfile.TemporaryDirectory() as tmp:
            settings = make_settings(tmp)
            with Store(settings.db_path) as store:
                report = Pipeline(settings, store).run(
                    messages, datetime.now(tz=UTC), datetime.now(tz=UTC)
                )
        labels = [c.cluster.label for c in report.clusters]
        self.assertEqual(len(labels), len(set(labels)), f"duplicate labels: {labels}")

    def test_second_pass_creates_no_new_clusters(self):
        """Re-processing the same input must be a no-op for the taxonomy."""
        messages = load_messages()
        with tempfile.TemporaryDirectory() as tmp:
            settings = make_settings(tmp)
            with Store(settings.db_path) as store:
                pipeline = Pipeline(settings, store)
                now = datetime.now(tz=UTC)
                pipeline.run(messages, now, now)
                before = store.conn.execute("SELECT COUNT(*) c FROM clusters").fetchone()["c"]
                second = pipeline.run(messages, now, now)
                after = store.conn.execute("SELECT COUNT(*) c FROM clusters").fetchone()["c"]
        self.assertEqual(before, after)
        self.assertEqual(len(second.new_clusters), 0)

    def test_counters_do_not_inflate_when_a_window_is_reprocessed(self):
        """Overlapping windows are normal — a manual --days 3, a retried job.

        If cumulative counters advanced on every pass, the 7-day baselines that
        trend detection reads would drift upward with no failure ever occurring.
        """
        messages = load_messages()
        with tempfile.TemporaryDirectory() as tmp:
            settings = make_settings(tmp)
            with Store(settings.db_path) as store:
                pipeline = Pipeline(settings, store)
                now = datetime.now(tz=UTC)
                pipeline.run(messages, now, now)

                def totals():
                    return (
                        store.conn.execute(
                            "SELECT COALESCE(SUM(total_events), 0) t FROM clusters"
                        ).fetchone()["t"],
                        store.conn.execute(
                            "SELECT COALESCE(SUM(event_count), 0) t FROM templates"
                        ).fetchone()["t"],
                        store.conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"],
                    )

                first = totals()
                pipeline.run(messages, now, now)
                pipeline.run(messages, now, now)
                third = totals()

        self.assertEqual(first, third)
        self.assertEqual(first[0], error_count())  # cluster totals == real events
        self.assertEqual(first[1], error_count())  # template counts == real events

    def test_embeddings_are_cached_between_runs(self):
        messages = load_messages()
        with tempfile.TemporaryDirectory() as tmp:
            settings = make_settings(tmp)
            with Store(settings.db_path) as store:
                pipeline = Pipeline(settings, store)
                now = datetime.now(tz=UTC)
                pipeline.run(messages, now, now)
                cached = store.conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"]
                pipeline.run(messages, now, now)
                still = store.conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"]
        self.assertGreater(cached, 0)
        self.assertEqual(cached, still)

    def test_run_is_deterministic(self):
        messages = load_messages()
        signatures = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                settings = make_settings(tmp)
                with Store(settings.db_path) as store:
                    report = Pipeline(settings, store).run(
                        messages, datetime.now(tz=UTC), datetime.now(tz=UTC)
                    )
                signatures.append(sorted((c.cluster.label, c.event_count) for c in report.clusters))
        self.assertEqual(signatures[0], signatures[1])


class TestBlockKit(unittest.TestCase):
    def setUp(self):
        messages = load_messages()
        self.tmp = tempfile.TemporaryDirectory()
        settings = make_settings(self.tmp.name)
        self.settings = settings
        with Store(settings.db_path) as store:
            self.report = Pipeline(settings, store).run(
                messages, datetime.now(tz=UTC), datetime.now(tz=UTC)
            )

    def tearDown(self):
        self.tmp.cleanup()

    def test_respects_slack_limits(self):
        blocks = build_report_blocks(self.report, "<@U0123456789>", 5, 12)
        self.assertLessEqual(len(blocks), MAX_BLOCKS)
        for block in blocks:
            self.assertIn(block["type"], {"header", "section", "context", "divider"})
            if block["type"] == "section":
                self.assertLessEqual(len(block["text"]["text"]), MAX_SECTION_CHARS)
            if block["type"] == "header":
                self.assertEqual(block["text"]["type"], "plain_text")
                self.assertLessEqual(len(block["text"]["text"]), 150)
            if block["type"] == "context":
                self.assertLessEqual(len(block["elements"]), 10)

    def test_payload_is_json_serialisable(self):
        blocks = build_report_blocks(self.report, "", 5, 12)
        json.dumps({"channel": "C1", "blocks": blocks})

    def test_thread_carries_every_quote_id(self):
        thread = build_thread_text(self.report)
        for entry in self.report.clusters:
            for quote in entry.unique_quotes:
                self.assertIn(quote, thread)

    def test_empty_report_renders(self):
        from rei.models import DailyReport

        now = datetime.now(tz=UTC)
        empty = DailyReport(
            run_date="Sun 16 Aug 2026",
            window_start=now,
            window_end=now,
            messages_scanned=0,
            events_extracted=0,
            unique_templates=0,
        )
        blocks = build_report_blocks(empty, "<@U1>", 5, 12)
        self.assertTrue(any("No renewal errors" in json.dumps(b) for b in blocks))


class TestLabelRobustness(unittest.TestCase):
    """A schema-validated tool call is not a guarantee about semantics."""

    def setUp(self):
        from rei.llm import _fill_gaps

        self.fill = _fill_gaps
        self.clusters = [
            {"index": 0, "templates": ["Renewal Flow: Card declined"], "hints": {}},
            {"index": 1, "templates": ["Renewal Flow: Payment needs 3D secure"], "hints": {}},
        ]

    def test_string_results_do_not_crash(self):
        """Observed in production: the model returned 149 entries for one input
        cluster, one of them a bare string, which crashed the merge and dropped
        the whole batch to rule-based labels."""
        results = ["oops", {"index": 0, "label": "Card declined", "category": "payment"}]
        out = self.fill(results, self.clusters)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["label"], "Card declined")
        self.assertTrue(out[1]["label"])  # rule fallback filled the gap

    def test_out_of_range_and_duplicate_indexes_are_dropped(self):
        results = [
            {"index": 99, "label": "hallucinated"},
            {"index": 0, "label": "first wins", "category": "payment"},
            {"index": 0, "label": "second ignored"},
            {"index": "1", "label": "wrong type"},
        ]
        out = self.fill(results, self.clusters)
        self.assertEqual(out[0]["label"], "first wins")
        self.assertNotEqual(out[1].get("label"), "wrong type")

    def test_always_one_result_per_cluster(self):
        for results in ([], [{}], [{"index": None}], list(range(200))):
            with self.subTest(results=str(results)[:40]):
                self.assertEqual(len(self.fill(results, self.clusters)), 2)


class TestCorrelationLinking(unittest.TestCase):
    """One failure emitting two log lines should be reported as one incident."""

    def _store(self, tmp):
        from rei.store import Store

        store = Store(Path(tmp) / "link.db")
        store.create_cluster(
            "Duplicate Stripe customers",
            __import__("numpy").array([1.0], dtype="float32"),
            "2026-01-01",
        )
        store.create_cluster(
            "Failed to create accounts",
            __import__("numpy").array([0.0], dtype="float32"),
            "2026-01-01",
        )
        store.create_cluster(
            "Unrelated timeout", __import__("numpy").array([0.5], dtype="float32"), "2026-01-01"
        )
        return store

    def test_paired_clusters_are_linked_and_ordered(self):
        from rei.digest import build_digest

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            for i in range(40):
                corr = f"corr-{i}"
                # Cluster 1 always precedes cluster 2 under the same trace.
                store.insert_event(
                    slack_ts=f"a{i}",
                    occurred_at=f"2026-01-02T00:00:{i:02d}",
                    message="dup",
                    template_hash="h1",
                    cluster_id=1,
                    quote_id=None,
                    quote_id_confidence=0.0,
                    correlation_id=corr,
                    http_status=None,
                    service="s",
                    raw_excerpt="",
                )
                store.insert_event(
                    slack_ts=f"b{i}",
                    occurred_at=f"2026-01-02T00:01:{i:02d}",
                    message="acct",
                    template_hash="h2",
                    cluster_id=2,
                    quote_id=None,
                    quote_id_confidence=0.0,
                    correlation_id=corr,
                    http_status=None,
                    service="s",
                    raw_excerpt="",
                )
            # A third cluster on its own traces must not be linked to anything.
            for i in range(20):
                store.insert_event(
                    slack_ts=f"c{i}",
                    occurred_at=f"2026-01-03T00:00:{i:02d}",
                    message="timeout",
                    template_hash="h3",
                    cluster_id=3,
                    quote_id=None,
                    quote_id_confidence=0.0,
                    correlation_id=f"solo-{i}",
                    http_status=None,
                    service="s",
                    raw_excerpt="",
                )
            store.commit()
            digest = build_digest(store)
            store.close()

        self.assertEqual(len(digest.links), 1, f"expected one link, got {digest.links}")
        link = digest.links[0]
        self.assertEqual({link.a_id, link.b_id}, {1, 2})
        self.assertEqual(link.shared, 40)
        self.assertAlmostEqual(link.containment, 1.0)
        self.assertAlmostEqual(link.a_first_pct, 1.0)
        self.assertIn("plausibly causes the second", link.verdict)
        self.assertEqual(link.first_label, "Duplicate Stripe customers")
        # Three issue lines, but only two real incidents.
        self.assertEqual(len(digest.issues), 3)
        self.assertEqual(digest.distinct_incidents, 2)

    def test_no_links_when_traces_do_not_overlap(self):
        from rei.digest import build_digest

        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            for i in range(30):
                store.insert_event(
                    slack_ts=f"a{i}",
                    occurred_at=f"2026-01-02T00:00:{i:02d}",
                    message="x",
                    template_hash="h1",
                    cluster_id=1,
                    quote_id=None,
                    quote_id_confidence=0.0,
                    correlation_id=f"a-{i}",
                    http_status=None,
                    service="s",
                    raw_excerpt="",
                )
                store.insert_event(
                    slack_ts=f"b{i}",
                    occurred_at=f"2026-01-02T00:01:{i:02d}",
                    message="y",
                    template_hash="h2",
                    cluster_id=2,
                    quote_id=None,
                    quote_id_confidence=0.0,
                    correlation_id=f"b-{i}",
                    http_status=None,
                    service="s",
                    raw_excerpt="",
                )
            store.commit()
            digest = build_digest(store)
            store.close()

        self.assertEqual(digest.links, [])
        self.assertEqual(digest.distinct_incidents, len(digest.issues))


class TestOwnershipIsDeterministic(unittest.TestCase):
    """Ownership drives the report's headline number, so it must not depend on a
    label the model is free to change. Across two runs on identical data,
    "Multiple Stripe customers exist" moved from `data` to `payment` and shifted
    the engineering share by five points.
    """

    def _issue(self, label, category, template):
        from rei.digest import Issue

        return Issue(1, label, category, "blocking", 10, 5, 1, "", "", 10, {}, template)

    def test_customer_owns_only_payment_instrument_failures(self):
        for label, template in [
            (
                "Stripe card declined due to insufficient funds",
                "Card declined: Your card has insufficient funds. (insufficient_funds)",
            ),
            (
                "Monthly subscription setup failed due to insufficient funds",
                "Renewal Flow: Failed to setup a monthly subscription",
            ),
            (
                "Payment requires 3D secure verification",
                "Renewal Flow: Payment needs 3D secure verification",
            ),
            (
                "Stripe customer has no default payment method",
                "Could not find a default payment method for Stripe user <STRIPE_ID>",
            ),
        ]:
            with self.subTest(label=label):
                self.assertEqual(self._issue(label, "payment", template).owner, "customer")

    def test_payment_adjacent_data_defects_stay_engineering(self):
        """The regression this rule exists to prevent."""
        issue = self._issue(
            "Multiple Stripe customers found for same email",
            "payment",
            "Renewal Flow <UUID>: Multiple Stripe customers exist with the given email",
        )
        self.assertEqual(issue.owner, "engineering")

    def test_category_flip_does_not_change_ownership(self):
        label = "Multiple Stripe customers found for same email"
        template = "Multiple Stripe customers exist with the given email"
        owners = {
            self._issue(label, cat, template).owner
            for cat in ("data", "payment", "integration", "validation")
        }
        self.assertEqual(owners, {"engineering"}, "ownership moved with the category")

    def test_genuinely_unknown_stays_unclassified(self):
        issue = self._issue(
            "Unattributed HTTP 400 during renewal case check",
            "unknown",
            "Renewal Flow: Request failed with status code 400",
        )
        self.assertEqual(issue.owner, "unclassified")


if __name__ == "__main__":
    unittest.main()
