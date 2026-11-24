"""Unit tests for the fixed RRF sensitivity implementation."""

from __future__ import annotations

import unittest

import numpy as np

import careerbuilder_internal_temporal_evaluation as careerbuilder
import jth_sampling_evaluation as jth


class ReciprocalRankFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.job_ids = np.asarray([10, 20, 30, 40, 50], dtype=np.int64)
        self.lexical = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
        self.semantic = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_dataset_implementations_agree(self) -> None:
        expected = careerbuilder.reciprocal_rank_fusion_within_candidates(
            self.job_ids,
            self.lexical,
            self.semantic,
            output_depth=5,
            constant=60.0,
        )
        observed = jth.reciprocal_rank_fusion_within_candidates(
            self.job_ids,
            np.arange(len(self.job_ids), dtype=np.int32),
            self.lexical,
            self.semantic,
            output_depth=5,
            constant=60.0,
        )
        np.testing.assert_array_equal(observed, expected)

    def test_ranks_are_recomputed_within_subset(self) -> None:
        subset = np.asarray([0, 1, 2], dtype=np.int32)
        ranked = jth.reciprocal_rank_fusion_within_candidates(
            self.job_ids,
            subset,
            self.lexical,
            self.semantic,
            output_depth=3,
            constant=60.0,
        )
        self.assertEqual(set(map(int, ranked)), {10, 20, 30})
        self.assertEqual(len(ranked), 3)

    def test_job_identifier_breaks_exact_fusion_ties(self) -> None:
        lexical = np.asarray([2.0, 1.0])
        semantic = np.asarray([1.0, 2.0])
        ranked = careerbuilder.reciprocal_rank_fusion_within_candidates(
            np.asarray([20, 10], dtype=np.int64),
            lexical,
            semantic,
            output_depth=2,
            constant=60.0,
        )
        self.assertEqual(ranked.tolist(), [10, 20])

    def test_invalid_parameters_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            careerbuilder.reciprocal_rank_fusion_within_candidates(
                self.job_ids,
                self.lexical,
                self.semantic,
                output_depth=5,
                constant=0.0,
            )


if __name__ == "__main__":
    unittest.main()
