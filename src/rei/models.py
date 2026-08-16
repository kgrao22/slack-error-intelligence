"""Domain model.

The important idea here is that an ErrorEvent knows *which* quote it belongs to,
separately from the other identifiers that happen to appear in its payload. The
original script conflated the two: it collected every UUID anywhere in a message
and attached the whole set to whatever error group the message landed in. That is
why the sample report says "12 x Request failed with status code 400" and then
lists nine quote IDs, and why one group listed thirty. Counts and IDs were
measuring different things.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ErrorEvent(BaseModel):
    """One error occurrence extracted from one Slack message."""

    slack_ts: str
    occurred_at: datetime
    # The most specific human-readable error string found in the payload.
    message: str
    # Every error-ish string found, deepest-first. Kept for the LLM's context.
    all_messages: list[str] = Field(default_factory=list)
    # The quote this error is about, when we can attribute it confidently.
    quote_id: str | None = None
    quote_id_confidence: float = 0.0
    # Other identifiers seen in the payload (customer ids, doc ids, ...).
    related_ids: list[str] = Field(default_factory=list)
    http_status: int | None = None
    # The emitting handler, e.g. "renewalHandler". Taken from the
    # methodName field in the Slack payload.
    service: str | None = None
    # The error taxonomy the renewal service already publishes, e.g.
    # "payment-failure" or "unknown". This is authoritative and is used to set
    # the cluster category directly rather than having a model guess it.
    error_type: str | None = None
    # Nested cause, e.g. "card-declined" / "StripeServiceError".
    cause_type: str | None = None
    cause_name: str | None = None
    correlation_id: str | None = None
    raw_excerpt: str = ""

    @property
    def has_quote(self) -> bool:
        return self.quote_id is not None


class Cluster(BaseModel):
    """A persistent semantic group. IDs are stable across days, which is what
    makes 'is this new?' and week-over-week trends possible at all."""

    cluster_id: int
    label: str
    category: str = "unknown"
    severity: str = "unknown"
    root_cause: str | None = None
    suggested_action: str | None = None
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    template_count: int = 0
    total_events: int = 0


class ClusterReport(BaseModel):
    """A cluster as it appears in one day's report."""

    cluster: Cluster
    event_count: int
    unique_quotes: list[str] = Field(default_factory=list)
    sample_messages: list[str] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list)
    is_new: bool = False
    prior_daily_mean: float | None = None
    # Denominator for share-of-total. A raw count reads as alarming or trivial
    # depending on the reader's guess at the total; the share removes the guess.
    window_total: int = 0

    @property
    def share_pct(self) -> float:
        return self.event_count / self.window_total * 100 if self.window_total else 0.0

    @property
    def unique_quote_count(self) -> int:
        return len(self.unique_quotes)

    @property
    def delta_pct(self) -> float | None:
        if not self.prior_daily_mean:
            return None
        return (self.event_count - self.prior_daily_mean) / self.prior_daily_mean * 100.0


class DailyReport(BaseModel):
    run_date: str
    window_start: datetime
    window_end: datetime
    messages_scanned: int
    events_extracted: int
    unique_templates: int
    clusters: list[ClusterReport] = Field(default_factory=list)

    @property
    def new_clusters(self) -> list[ClusterReport]:
        return [c for c in self.clusters if c.is_new]

    @property
    def total_unique_quotes(self) -> int:
        seen: set[str] = set()
        for c in self.clusters:
            seen.update(c.unique_quotes)
        return len(seen)
