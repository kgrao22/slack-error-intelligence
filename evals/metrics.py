"""Clustering-quality metrics, implemented with no third-party dependency.

Homogeneity, completeness and V-measure follow Rosenberg & Hirschberg (2007);
the adjusted Rand index follows Hubert & Arabie (1985). `tests/test_metrics.py`
checks these against scikit-learn when it is installed, so the zero-dependency
implementations are held to the reference behaviour rather than trusted.

Why these three and not accuracy: the predicted clusters have no natural
correspondence to the true ones — the pipeline invents its own IDs — so any
metric requiring aligned labels is inapplicable. Homogeneity ("does each
predicted cluster contain only one true failure?") and completeness ("is each
true failure kept in one predicted cluster?") name the two ways this can go
wrong, and the two failure modes have very different operational costs:

  - low homogeneity means unrelated failures are merged, so the report hides one
    behind another. This is what the old "Likely validation error" bucket did.
  - low completeness means one failure is split across several lines, so its true
    frequency is understated. This is what happened to every message with a UUID
    embedded in it.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


def _entropy(counts: list[int], total: int) -> float:
    if total == 0:
        return 0.0
    result = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            result -= p * math.log(p)
    return result


def _mutual_information(truth: list[str], pred: list[str]) -> float:
    n = len(truth)
    if n == 0:
        return 0.0
    joint = Counter(zip(truth, pred, strict=True))
    truth_counts = Counter(truth)
    pred_counts = Counter(pred)
    mi = 0.0
    for (t, p), n_tp in joint.items():
        p_tp = n_tp / n
        p_t = truth_counts[t] / n
        p_p = pred_counts[p] / n
        if p_tp > 0:
            mi += p_tp * math.log(p_tp / (p_t * p_p))
    return mi


@dataclass
class ClusterQuality:
    homogeneity: float
    completeness: float
    v_measure: float
    adjusted_rand: float
    n_true_clusters: int
    n_pred_clusters: int
    n_items: int

    def as_row(self, name: str) -> str:
        return (
            f"{name:<28} {self.homogeneity:>7.3f} {self.completeness:>8.3f} "
            f"{self.v_measure:>7.3f} {self.adjusted_rand:>7.3f} "
            f"{self.n_pred_clusters:>7d} {self.n_true_clusters:>7d}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'variant':<28} {'homog.':>7} {'complet.':>8} {'V':>7} {'ARI':>7} "
            f"{'#pred':>7} {'#true':>7}"
        )


def cluster_quality(truth: list[str], pred: list[str]) -> ClusterQuality:
    n = len(truth)
    if n != len(pred):
        raise ValueError("truth and pred must be the same length")
    if n == 0:
        return ClusterQuality(1.0, 1.0, 1.0, 1.0, 0, 0, 0)

    h_truth = _entropy(list(Counter(truth).values()), n)
    h_pred = _entropy(list(Counter(pred).values()), n)
    mi = _mutual_information(truth, pred)

    homogeneity = 1.0 if h_truth == 0 else mi / h_truth
    completeness = 1.0 if h_pred == 0 else mi / h_pred
    if homogeneity + completeness == 0:
        v_measure = 0.0
    else:
        v_measure = 2 * homogeneity * completeness / (homogeneity + completeness)

    return ClusterQuality(
        homogeneity=max(0.0, min(1.0, homogeneity)),
        completeness=max(0.0, min(1.0, completeness)),
        v_measure=max(0.0, min(1.0, v_measure)),
        adjusted_rand=adjusted_rand_index(truth, pred),
        n_true_clusters=len(set(truth)),
        n_pred_clusters=len(set(pred)),
        n_items=n,
    )


def adjusted_rand_index(truth: list[str], pred: list[str]) -> float:
    n = len(truth)
    if n < 2:
        return 1.0
    joint = Counter(zip(truth, pred, strict=True))
    truth_counts = Counter(truth)
    pred_counts = Counter(pred)

    def c2(x: int) -> float:
        return x * (x - 1) / 2.0

    index = sum(c2(v) for v in joint.values())
    sum_a = sum(c2(v) for v in truth_counts.values())
    sum_b = sum(c2(v) for v in pred_counts.values())
    total = c2(n)
    expected = sum_a * sum_b / total if total else 0.0
    maximum = (sum_a + sum_b) / 2.0
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


@dataclass
class AttributionQuality:
    """How well each report attributes an error to the right quote ID."""

    precision: float
    recall: float
    f1: float
    predicted: int
    correct: int
    expected: int

    def as_row(self, name: str) -> str:
        return (
            f"{name:<28} {self.precision:>9.3f} {self.recall:>7.3f} {self.f1:>7.3f} "
            f"{self.predicted:>9d} {self.correct:>8d}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'variant':<28} {'precision':>9} {'recall':>7} {'F1':>7} "
            f"{'claimed':>9} {'correct':>8}"
        )


def attribution_quality(
    predicted: dict[str, set[str]], truth: dict[str, str]
) -> AttributionQuality:
    """`predicted` maps slack_ts -> the set of quote IDs the report attributes to
    that message. `truth` maps slack_ts -> the one correct quote ID."""
    claimed = 0
    correct = 0
    recalled = 0
    for ts, true_quote in truth.items():
        got = predicted.get(ts, set())
        claimed += len(got)
        if true_quote and true_quote in got:
            correct += 1
            recalled += 1
    expected = len([q for q in truth.values() if q])
    precision = correct / claimed if claimed else 0.0
    recall = recalled / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return AttributionQuality(precision, recall, f1, claimed, correct, expected)
