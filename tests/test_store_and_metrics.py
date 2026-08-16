from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evals.metrics import adjusted_rand_index, cluster_quality
from rei.store import Store, from_blob, to_blob


def unit(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_blob_roundtrip_preserves_vector(self):
        vector = unit([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(from_blob(to_blob(vector)), vector, rtol=1e-6)

    def test_embedding_cache_roundtrip(self):
        vectors = {"a" * 32: unit([1, 0, 0]), "b" * 32: unit([0, 1, 0])}
        self.store.put_embeddings(vectors, "m1")
        got = self.store.get_cached_embeddings(list(vectors), "m1")
        self.assertEqual(set(got), set(vectors))
        # A different model must not hit the same cache entries.
        self.assertEqual(self.store.get_cached_embeddings(list(vectors), "m2"), {})

    def test_search_returns_nearest_centroid_first(self):
        a = self.store.create_cluster("a", unit([1, 0, 0]), "2026-08-16T00:00:00+00:00")
        b = self.store.create_cluster("b", unit([0, 1, 0]), "2026-08-16T00:00:00+00:00")
        results = self.store.search_clusters(unit([0.9, 0.1, 0.0]), top_k=2)
        self.assertEqual(results[0][0], a)
        self.assertEqual(results[1][0], b)
        self.assertGreater(results[0][1], results[1][1])

    def test_search_handles_empty_store_and_dimension_mismatch(self):
        self.assertEqual(self.store.search_clusters(unit([1, 0, 0])), [])
        self.store.create_cluster("a", unit([1, 0, 0]), "2026-08-16T00:00:00+00:00")
        self.assertEqual(self.store.search_clusters(unit([1, 0, 0, 0])), [])

    def test_centroid_update_stays_normalised(self):
        cid = self.store.create_cluster("a", unit([1, 0, 0]), "2026-08-16T00:00:00+00:00")
        self.store.update_centroid(cid, unit([1, 0, 0]))
        self.store.update_centroid(cid, unit([0, 1, 0]))
        _ids, matrix = self.store.load_centroids()
        self.assertAlmostEqual(float(np.linalg.norm(matrix[0])), 1.0, places=5)

    def test_cached_embeddings_handles_large_hash_lists(self):
        """The IN(...) chunking path — SQLite caps host parameters at 999."""
        vectors = {f"{i:032d}": unit([1, 0, 0]) for i in range(1500)}
        self.store.put_embeddings(vectors, "m1")
        got = self.store.get_cached_embeddings(list(vectors), "m1")
        self.assertEqual(len(got), 1500)


class TestMetrics(unittest.TestCase):
    def test_perfect_and_worst_cases(self):
        truth = ["a", "a", "b", "b"]
        perfect = cluster_quality(truth, ["x", "x", "y", "y"])
        self.assertAlmostEqual(perfect.v_measure, 1.0)
        self.assertAlmostEqual(perfect.adjusted_rand, 1.0)

        merged = cluster_quality(truth, ["x", "x", "x", "x"])
        self.assertAlmostEqual(merged.completeness, 1.0)
        self.assertLess(merged.homogeneity, 0.5)

        split = cluster_quality(truth, ["w", "x", "y", "z"])
        self.assertAlmostEqual(split.homogeneity, 1.0)
        self.assertLess(split.completeness, 1.0)

    def test_matches_scikit_learn_when_available(self):
        """Hold the dependency-free implementations to the reference behaviour."""
        try:
            from sklearn.metrics import (
                adjusted_rand_score,
                completeness_score,
                homogeneity_score,
                v_measure_score,
            )
        except ImportError:
            self.skipTest("scikit-learn not installed")

        rng = random.Random(7)
        for _ in range(25):
            n = rng.randint(10, 200)
            truth = [rng.choice("abcdef") for _ in range(n)]
            pred = [rng.choice("wxyz01234") for _ in range(n)]
            got = cluster_quality(truth, pred)
            self.assertAlmostEqual(got.homogeneity, homogeneity_score(truth, pred), places=6)
            self.assertAlmostEqual(got.completeness, completeness_score(truth, pred), places=6)
            self.assertAlmostEqual(got.v_measure, v_measure_score(truth, pred), places=6)
            self.assertAlmostEqual(
                adjusted_rand_index(truth, pred), adjusted_rand_score(truth, pred), places=6
            )

    def test_empty_input(self):
        result = cluster_quality([], [])
        self.assertEqual(result.n_items, 0)
        self.assertAlmostEqual(result.v_measure, 1.0)


if __name__ == "__main__":
    unittest.main()
