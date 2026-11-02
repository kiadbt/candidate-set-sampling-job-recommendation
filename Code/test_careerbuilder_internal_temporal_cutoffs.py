#!/usr/bin/env python3
"""Regression tests for CareerBuilder cutoff admissibility."""

from __future__ import annotations

import inspect
import unittest
from collections import defaultdict
from types import SimpleNamespace

import careerbuilder_internal_temporal_evaluation as evaluator


class SupportedCutoffTests(unittest.TestCase):
    def test_full_catalog_uses_realized_support(self) -> None:
        self.assertEqual(
            evaluator.supported_cutoffs((5, 10, 20, 150), 100),
            (5, 10, 20),
        )
        self.assertEqual(
            evaluator.supported_cutoffs((5, 10, 20, 150), 200),
            (5, 10, 20, 150),
        )

    def test_nominal_size_100_never_admits_k150(self) -> None:
        for realized_size in (100, 149, 150, 151, 500):
            with self.subTest(realized_size=realized_size):
                admitted = evaluator.supported_cutoffs(
                    (5, 10, 20, 150),
                    realized_size,
                    requested_candidate_size=100,
                )
                self.assertNotIn(150, admitted)

    def test_larger_nominal_size_can_admit_k150(self) -> None:
        self.assertEqual(
            evaluator.supported_cutoffs(
                (5, 10, 20, 150), 200, requested_candidate_size=500
            ),
            (5, 10, 20, 150),
        )

    def test_evaluate_user_passes_nominal_size_to_gate(self) -> None:
        source = inspect.getsource(evaluator.evaluate_user)
        self.assertIn("requested_candidate_size=nominal", source)
        self.assertNotIn(
            "tuple(k for k in args.ks if k <= len(candidate_ids))", source
        )

    def test_high_relevance_query_preserves_fixed_row_contract(self) -> None:
        np = evaluator.np
        pd = evaluator.pd
        job_ids = np.arange(1, 12_001, dtype=np.int64)
        relevant = set(map(int, job_ids[:160]))
        assets = {
            "users": pd.DataFrame([{
                "UserID": 42,
                "precut_application_count": 0,
                "activity_group": "cold_0",
                "mean_target_precut_popularity": 0.0,
                "target_popularity_group": "mean_unseen_0",
                "inclusion_probability": 1.0,
                "population_weight": 1.0,
            }]),
            "truth": {42: relevant},
            "job_ids": job_ids,
            "job_index": {int(job_id): index for index, job_id in enumerate(job_ids)},
            "prior": {},
            "popularity": np.zeros(len(job_ids), dtype=np.float32),
            "sampling_cdf": np.linspace(
                1.0 / len(job_ids), 1.0, len(job_ids), dtype=np.float64
            ),
            "window": 1,
        }
        zero_scores = np.zeros(len(job_ids), dtype=np.float32)
        scores = {
            "tfidf_combined_static": zero_scores,
            "bm25_full_static": zero_scores,
            "bm25_title_static": zero_scores,
            "minilm_title_static": zero_scores,
        }
        args = SimpleNamespace(
            protocol="First7Last2",
            ks=(5, 10, 20, 150),
            skip_semantic=False,
            candidate_sizes=(100, 500, 1000, 5000, 10000),
            strategies=("uniform", "popularity_weighted"),
            seeds=(1, 2, 3, 4, 5),
        )
        integrity = defaultdict(int)

        rows = evaluator.evaluate_user(args, assets, 0, scores, integrity)

        self.assertEqual(len(rows), 1_164)
        self.assertEqual(integrity["target_count_exceeds_nominal_user_sizes"], 10)
        self.assertEqual(integrity["target_inclusion_failures"], 0)
        self.assertEqual(integrity["nesting_failures"], 0)
        self.assertFalse(any(
            row["candidate_regime"] == "sampled"
            and row["requested_candidate_size"] == 100
            and row["k"] == 150
            for row in rows
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
