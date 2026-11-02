#!/usr/bin/env python3
"""Static and synthetic tests for the fail-closed synthesis workflow."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd


SCRIPT = Path(__file__).with_name("build_cross_dataset_synthesis.py")
SPEC = importlib.util.spec_from_file_location("cross_dataset_synthesis", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"Cannot import {SCRIPT}")
SYNTH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNTH)
TEST_TMP_ROOT = SCRIPT.parents[1] / ".cross_synthesis_test_tmp"


def workspace_tempdir() -> tempfile.TemporaryDirectory:
    TEST_TMP_ROOT.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT)


class CrossDatasetSynthesisTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if TEST_TMP_ROOT.exists() and not any(TEST_TMP_ROOT.iterdir()):
            TEST_TMP_ROOT.rmdir()

    def test_source_compiles(self) -> None:
        compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")

    def test_only_aggregate_evidence_is_named(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in (
            "per_query_metrics.csv", "per_user_metrics.csv",
            "checkpoint_manifest.json", "aggregate_metrics.csv",
        ):
            self.assertNotIn(forbidden, source)

    def test_completion_sentinel_is_written_after_integrity(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertGreater(
            source.index('(output / "SYNTHESIS_COMPLETE.json").write_text'),
            source.index('integrity_path = output / "synthesis_integrity.json"'),
        )

    def test_interval_classification(self) -> None:
        self.assertEqual(
            SYNTH.classify_interval_directions(
                (0.2, 0.1, 0.3), (0.4, 0.2, 0.6)
            ),
            "bounded_replication_interval_supported_direction",
        )
        self.assertEqual(
            SYNTH.classify_interval_directions(
                (0.2, 0.1, 0.3), (-0.2, -0.4, -0.1)
            ),
            "dataset_specific_interval_supported_opposite_direction",
        )
        self.assertEqual(
            SYNTH.classify_interval_directions(
                (0.0, -0.1, 0.1), (0.2, 0.1, 0.3)
            ),
            "bounded_or_inconclusive_interval",
        )

    def test_classification_matrix_uses_measured_cells(self) -> None:
        bias = pd.DataFrame({
            "dataset": ["JTH", "CareerBuilder"],
            "strategy": ["uniform", "uniform"],
            "candidate_size": [100, 100],
            "metric": ["hitrate", "hitrate"],
            "model_family": ["BM25", "BM25"],
            "k": [20, 20],
            "signed_bias_sampled_minus_full": [0.1, 0.2],
        })
        rank = pd.DataFrame({
            "dataset": ["JTH", "CareerBuilder"],
            "strategy": ["uniform", "uniform"],
            "candidate_size": [100, 100],
            "metric": ["hitrate", "hitrate"],
            "kendall_tau_b_mean": [0.8, 0.7],
            "pairwise_order_agreement_mean": [0.9, 0.8],
            "winner_exact_rate": [0.6, 0.4],
            "winner_regret_worst_mean": [0.02, 0.03],
        })
        contrasts = pd.DataFrame({
            "dataset": ["JTH", "CareerBuilder"],
            "condition": ["sampled", "sampled"],
            "strategy": ["uniform", "uniform"],
            "candidate_size": [100, 100],
            "metric": ["hitrate", "hitrate"],
            "k": [20, 20],
            "contrast_family": ["BM25_minus_MiniLM", "BM25_minus_MiniLM"],
            "effect_type": ["contrast", "contrast"],
            "estimate": [0.2, -0.2],
            "ci_low": [0.1, -0.3],
            "ci_high": [0.3, -0.1],
        })
        subgroup = pd.DataFrame({
            "dataset": ["JTH", "CareerBuilder"],
            "boundary_type": ["subgroup", "subgroup"],
            "canonical_subgroup": ["activity::cold_0", "activity::cold_0"],
            "model_family": ["BM25", "BM25"],
            "metric": ["hitrate", "hitrate"],
            "strategy": ["uniform", "uniform"],
            "candidate_size": [100, 100],
            "estimate": [0.01, 0.02],
            "ci_low": [-0.01, -0.02],
            "ci_high": [0.03, 0.04],
        })
        result = SYNTH.classification_matrix(bias, rank, contrasts, subgroup)
        by_domain = result.set_index("evidence_domain")
        self.assertEqual(
            by_domain.loc["metric_bias_direction", "classification_category"],
            "bounded_replication",
        )
        self.assertEqual(
            by_domain.loc["prespecified_bm25_contrast", "classification_category"],
            "dataset_specific",
        )
        self.assertEqual(
            by_domain.loc["subgroup_distortion", "classification_category"],
            "not_comparable",
        )

    def test_candidate_scale_collapses_career_seeds(self) -> None:
        with workspace_tempdir() as raw:
            root = Path(raw)
            jth, career = root / "jth", root / "career"
            jth.mkdir()
            career.mkdir()
            pd.DataFrame({
                "candidate_protocol": ["sampled"],
                "strategy": ["uniform"],
                "nominal_candidate_size": [100],
                "eligible_catalog_mean": [1000.0],
                "eligible_catalog_median": [1000.0],
                "actual_candidate_count_mean": [100.0],
                "realized_candidate_fraction_mean": [0.1],
            }).to_csv(jth / "publication_table_7_candidate_scale.csv", index=False)
            pd.DataFrame({
                "candidate_regime": ["sampled", "sampled"],
                "sampling_strategy": ["uniform", "uniform"],
                "seed": [1, 2],
                "requested_candidate_size": [100, 100],
                "eligible_candidate_count_mean": [2000.0, 2000.0],
                "eligible_candidate_count_median": [2000.0, 2000.0],
                "actual_candidate_count_mean": [100.0, 100.0],
                "actual_candidate_fraction_mean": [0.05, 0.05],
            }).to_csv(
                career / "publication_table_6_candidate_scale.csv", index=False
            )
            result = SYNTH.normalize_candidate_scale(jth, career)
            career_rows = result[result.dataset.eq("CareerBuilder")]
            self.assertEqual(len(career_rows), 1)
            self.assertAlmostEqual(
                float(career_rows.iloc[0].realized_candidate_fraction_mean), 0.05
            )

    def test_candidate_scale_accepts_authoritative_aggregate_schema(self) -> None:
        with workspace_tempdir() as raw:
            root = Path(raw)
            jth, career = root / "jth", root / "career"
            jth.mkdir()
            career.mkdir()
            pd.DataFrame({
                "candidate_protocol": ["sampled"],
                "strategy": ["uniform"],
                "nominal_candidate_size": [100],
                "eligible_catalog_mean": [1000.0],
                "eligible_catalog_median": [1000.0],
                "actual_candidate_count_mean": [100.0],
                "realized_candidate_fraction_mean": [0.1],
            }).to_csv(jth / "publication_table_7_candidate_scale.csv", index=False)
            pd.DataFrame({
                "sampling_strategy": ["full_catalog", "uniform"],
                "requested_candidate_size": [float("nan"), 100],
                "seeds": [1, 5],
                "eligible_candidate_count_mean": [2000.0, 2000.0],
                "eligible_candidate_count_median": [2000.0, 2000.0],
                "actual_candidate_count_mean": [2000.0, 100.0],
                "actual_candidate_fraction_mean": [1.0, 0.05],
            }).to_csv(
                career / "publication_table_6_candidate_scale.csv", index=False
            )
            result = SYNTH.normalize_candidate_scale(jth, career)
            career_rows = result[result.dataset.eq("CareerBuilder")]
            self.assertEqual(len(career_rows), 2)
            sampled = career_rows[career_rows.candidate_regime.eq("sampled")]
            self.assertEqual(len(sampled), 1)
            self.assertAlmostEqual(
                float(sampled.iloc[0].realized_candidate_fraction_mean), 0.05
            )

    def test_compute_table_matches_full_catalog_fraction(self) -> None:
        with workspace_tempdir() as raw:
            benchmark = Path(raw)
            pd.DataFrame({
                "model": ["bm25_full_static", "bm25_full_static"],
                "candidate_regime": ["full_catalog", "sampled_uniform"],
                "requested_candidate_size": [float("nan"), 100],
                "total_seconds_mean": [10.0, 1.0],
                "total_seconds_sd": [1.0, 0.1],
                "score_seconds_mean": [9.0, 0.8],
                "rank_seconds_mean": [0.5, 0.1],
                "metric_seconds_mean": [0.5, 0.1],
                "scored_pairs_mean": [2000.0, 100.0],
                "peak_delta_bytes_max": [1000, 100],
            }).to_csv(benchmark / "benchmark_primary_summary.csv", index=False)
            scale = pd.DataFrame({
                "dataset": ["CareerBuilder", "CareerBuilder"],
                "candidate_regime": ["full_catalog", "sampled"],
                "strategy": ["full_catalog", "uniform"],
                "candidate_size": [float("nan"), 100],
                "realized_candidate_fraction_mean": [1.0, 0.05],
            })
            result = SYNTH.compute_table(benchmark, scale)
            career = result[result.dataset.eq("CareerBuilder")]
            self.assertEqual(len(career), 2)
            full = career[career.candidate_regime.eq("full_catalog")].iloc[0]
            sampled = career[career.candidate_regime.eq("sampled_uniform")].iloc[0]
            self.assertAlmostEqual(float(full.realized_candidate_fraction_mean), 1.0)
            self.assertEqual(sampled.candidate_regime, "sampled_uniform")
            self.assertAlmostEqual(float(sampled.relative_to_full_time), 0.1)

    def test_jth_hash_mismatch_fails(self) -> None:
        with workspace_tempdir() as raw:
            directory = Path(raw)
            (directory / "integrity_checks.json").write_text(
                json.dumps({"status": "passed"}), encoding="utf-8"
            )
            (directory / "VERIFIED_FINDINGS.md").write_text("x", encoding="utf-8")
            (directory / "AUTHORITATIVE_COMPLETE.json").write_text(
                json.dumps({
                    "status": "passed",
                    "required_files_present": True,
                    "integrity_checks_sha256": "0" * 64,
                    "verified_findings_sha256": hashlib.sha256(b"x").hexdigest(),
                }),
                encoding="utf-8",
            )
            with self.assertRaises(SYNTH.EvidenceError):
                SYNTH.validate_jth(directory)

    def test_main_does_not_create_output_when_sources_are_incomplete(self) -> None:
        with workspace_tempdir() as raw:
            root = Path(raw)
            jth, career, benchmark = root / "j", root / "c", root / "b"
            for directory in (jth, career, benchmark):
                directory.mkdir()
            output = root / "must_not_exist"
            argv = [
                str(SCRIPT), "--jth-dir", str(jth), "--career-dir", str(career),
                "--benchmark-dir", str(benchmark), "--output-dir", str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SYNTH.EvidenceError):
                    SYNTH.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
