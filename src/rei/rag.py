"""Retrieval over the historical corpus.

The corpus is self-bootstrapping: every cluster the pipeline has ever created,
its exemplar messages, and any human resolution harvested from thread replies in
the report channel. Nothing external needs to be wired up for it to start being
useful, and it gets better every day the job runs.

Retrieval is used at exactly one decision point — when a template does not match
any existing cluster closely enough to join it automatically. At that moment we
hand Claude the near-misses: clusters that were similar but under threshold. This
is the case where a purely geometric decision is least reliable ("is this the
same failure worded differently, or a genuinely new one?"), so it is the case
worth spending a model call on. Everything above threshold is decided by cosine
distance alone, with no LLM in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .store import Store


@dataclass
class RetrievedCluster:
    cluster_id: int
    label: str
    category: str
    similarity: float
    exemplars: list[str]
    resolutions: list[str]
    total_events: int
    last_seen: str | None

    def render(self, max_exemplars: int = 2) -> str:
        lines = [
            f"- cluster #{self.cluster_id} (similarity {self.similarity:.3f}, "
            f"{self.total_events} events all-time, category={self.category})",
            f"  label: {self.label}",
        ]
        for exemplar in self.exemplars[:max_exemplars]:
            lines.append(f"  example: {exemplar[:240]}")
        for resolution in self.resolutions[:2]:
            lines.append(f"  previously resolved by: {resolution[:240]}")
        return "\n".join(lines)


class Retriever:
    def __init__(self, store: Store) -> None:
        self.store = store

    def neighbours(
        self, ranked: list[tuple[int, float]], limit: int = 3, min_similarity: float = 0.55
    ) -> list[RetrievedCluster]:
        out: list[RetrievedCluster] = []
        for cluster_id, similarity in ranked[:limit]:
            if similarity < min_similarity:
                continue
            cluster = self.store.get_cluster(cluster_id)
            if cluster is None:
                continue
            out.append(
                RetrievedCluster(
                    cluster_id=cluster_id,
                    label=cluster.label,
                    category=cluster.category,
                    similarity=similarity,
                    exemplars=self.store.cluster_exemplars(cluster_id, limit=3),
                    resolutions=self.store.get_resolutions(cluster_id, limit=2),
                    total_events=cluster.total_events,
                    last_seen=cluster.last_seen.isoformat() if cluster.last_seen else None,
                )
            )
        return out

    def render_context(self, retrieved: list[RetrievedCluster]) -> str:
        if not retrieved:
            return "(no similar historical clusters found)"
        return "\n".join(item.render() for item in retrieved)
