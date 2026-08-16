"""Incremental clustering against persistent centroids.

Why incremental rather than re-running k-means or HDBSCAN each morning: cluster
identity has to survive across days. If today's cluster 3 is not the same thing
as yesterday's cluster 3, then "is this new?", "is this getting worse?" and any
week-over-week comparison are all meaningless, and the RAG corpus has nothing
stable to attach resolutions to. Batch clustering gives better-shaped clusters on
a fixed corpus and no continuity at all, which is the wrong trade here.

The algorithm:

  1. Exact-template short circuit. A template hash we have assigned before goes
     straight back to its cluster with no vector search. On a steady-state day
     this handles almost everything, which is what makes the daily run cheap.
  2. Otherwise embed, then cosine-search the centroid matrix.
  3. Best similarity >= threshold: join that cluster and fold the vector into its
     centroid as a count-weighted running mean.
  4. Below threshold: open a new cluster seeded at this vector.

Templates are processed in descending frequency order so the day's dominant
template seeds a cluster before its rarer variants arrive. That makes the result
deterministic given the same input, which matters for the eval and for tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .store import Store

log = logging.getLogger(__name__)


@dataclass
class Assignment:
    template_hash: str
    template_text: str
    cluster_id: int
    similarity: float
    is_new_cluster: bool
    # A template seen for the first time, whether it opened a new cluster or
    # joined an existing one. Joining changes what an existing cluster *means*,
    # so its label has to be revisited — otherwise a cluster seeded by
    # "insufficient funds" keeps that name after expired-card and
    # transaction-not-allowed declines join it.
    is_new_template: bool = False
    event_count: int = 0
    neighbours: list[tuple[int, float]] = field(default_factory=list)


class IncrementalClusterer:
    def __init__(self, store: Store, threshold: float) -> None:
        self.store = store
        self.threshold = threshold

    def assign(
        self,
        templates: dict[str, str],
        vectors: dict[str, np.ndarray],
        counts: dict[str, int],
        seen_at: str,
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        ordered = sorted(templates, key=lambda h: (-counts.get(h, 0), h))

        for template_hash in ordered:
            text = templates[template_hash]
            count = counts.get(template_hash, 0)

            known = self.store.get_template_cluster(template_hash)
            if known is not None:
                assignments.append(Assignment(template_hash, text, known, 1.0, False, False, count))
                continue

            vector = vectors.get(template_hash)
            if vector is None:
                log.warning("no vector for template %s, skipping", template_hash)
                continue

            neighbours = self.store.search_clusters(vector, top_k=5)
            best_id, best_sim = neighbours[0] if neighbours else (None, -1.0)

            if best_id is not None and best_sim >= self.threshold:
                self.store.update_centroid(best_id, vector, weight=1)
                assignments.append(
                    Assignment(
                        template_hash, text, best_id, best_sim, False, True, count, neighbours
                    )
                )
            else:
                # Seed the new cluster with a provisional label; the LLM layer
                # replaces it with a canonical one before the report renders.
                cluster_id = self.store.create_cluster(
                    label=_provisional_label(text), centroid=vector, seen_at=seen_at
                )
                self.store.update_centroid(cluster_id, vector, weight=1)
                assignments.append(
                    Assignment(
                        template_hash, text, cluster_id, best_sim, True, True, count, neighbours
                    )
                )

        self.store.commit()
        return assignments


def _provisional_label(text: str) -> str:
    """A readable stand-in used until the LLM assigns a canonical label, and the
    permanent label when running offline or without an Anthropic key."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= 90:
        return collapsed
    cut = collapsed[:90].rsplit(" ", 1)[0]
    return f"{cut}…"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0
