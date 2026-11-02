#!/usr/bin/env python3
"""Fail-closed cross-dataset synthesis for the evaluation-validity study.

This program never reads per-user/per-query records. It consumes only completed,
integrity-gated aggregate packages from JTH, CareerBuilder, and the separate
CareerBuilder direct-scoring benchmark. Raw effectiveness is never pooled across
datasets. All cross-dataset classifications are computed from the input tables;
no empirical conclusion is hard-coded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMMON_SIZES = (100, 500, 1000, 5000)
COMMON_STRATEGIES = ("uniform", "popularity_weighted")
COMMON_METRICS = ("hitrate", "map")
JTH_ANALYZER = ROOT / "Code" / "analyze_jth_strengthened.py"
CAREER_ANALYZER = ROOT / "Code" / "analyze_careerbuilder_internal_temporal.py"
BENCHMARK_SCRIPT = (
    ROOT / "Code" / "careerbuilder_internal_temporal_direct_scoring_benchmark.py"
)
CAREER_EVALUATOR = ROOT / "Code" / "careerbuilder_internal_temporal_evaluation.py"
MODEL_MAP_JTH = {
    "bm25": "BM25",
    "semantic": "MiniLM",
    "tfidf": "TFIDF",
    "popularity": "Popularity",
    "random": "Random",
}
MODEL_MAP_CB = {
    "bm25_full_static": "BM25",
    "bm25_title_static": "BM25 title sensitivity",
    "minilm_title_static": "MiniLM",
    "tfidf_combined_static": "TFIDF",
    "popularity_precut": "Popularity",
    "random_hash": "Random",
}
CONTRAST_FAMILY_JTH = {
    "bm25_minus_semantic": "BM25_minus_MiniLM",
    "bm25_minus_tfidf": "BM25_minus_TFIDF",
}
CONTRAST_FAMILY_CB = {
    "BM25-full_minus_MiniLM-title": "BM25_minus_MiniLM",
    "BM25-full_minus_TFIDF-combined": "BM25_minus_TFIDF",
}
SUBGROUP_MAP_JTH = {
    ("training-interaction activity", "cold_0"): ("activity", "cold_0"),
    ("training-interaction activity", "one_1"): ("activity", "one_1"),
    ("target popularity", "unseen_0"): ("target_popularity", "unseen_0"),
    ("target popularity", "rare_gt0_to_2"): (
        "target_popularity", "positive_le2",
    ),
    ("target popularity", "popular_gt2"): ("target_popularity", "gt2"),
}
SUBGROUP_MAP_CB = {
    ("training-interaction activity", "cold_0"): ("activity", "cold_0"),
    ("training-interaction activity", "one_1"): ("activity", "one_1"),
    ("query-mean target pre-cutoff popularity", "mean_unseen_0"): (
        "target_popularity", "unseen_0",
    ),
    ("query-mean target pre-cutoff popularity", "mean_positive_le2"): (
        "target_popularity", "positive_le2",
    ),
    ("query-mean target pre-cutoff popularity", "mean_gt2"): (
        "target_popularity", "gt2",
    ),
}
TOL = 1e-12
RELATIVE_BIAS_MIN_DENOMINATOR = 0.01


class EvidenceError(RuntimeError):
    pass


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise EvidenceError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise EvidenceError(f"{label} is missing columns: {missing}")


def require_file(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file() or path.stat().st_size <= 0:
        raise EvidenceError(f"Missing or empty required file: {path}")
    return path


def verify_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual.lower() != str(expected).lower():
        raise EvidenceError(
            f"{label} SHA-256 mismatch for {path}: expected {expected}, found {actual}"
        )


def verify_generated_hashes(
    directory: Path,
    mapping: dict,
    required: Iterable[str],
    *,
    verify_every_recorded_file: bool = True,
) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise EvidenceError(f"Missing generated-file hash mapping in {directory}")
    for name in required:
        if name not in mapping:
            raise EvidenceError(f"Integrity record does not hash required file: {name}")
    names = mapping if verify_every_recorded_file else required
    for name in names:
        if Path(name).name != name:
            raise EvidenceError(f"Unsafe generated-file hash key: {name!r}")
        verify_hash(require_file(directory, name), mapping[name], "generated output")


def validate_jth(directory: Path) -> dict:
    sentinel = read_json(directory / "AUTHORITATIVE_COMPLETE.json")
    if sentinel.get("status") != "passed" or not sentinel.get("required_files_present"):
        raise EvidenceError("JTH authoritative sentinel did not pass")
    integrity_path = require_file(directory, "integrity_checks.json")
    findings_path = require_file(directory, "VERIFIED_FINDINGS.md")
    verify_hash(integrity_path, sentinel.get("integrity_checks_sha256"), "JTH integrity")
    verify_hash(findings_path, sentinel.get("verified_findings_sha256"), "JTH findings")
    integrity = read_json(integrity_path)
    if integrity.get("status") != "passed":
        raise EvidenceError("JTH integrity status is not passed")
    recorded_code_hash = integrity.get("analysis_code_sha256")
    if sentinel.get("analysis_code_sha256") != recorded_code_hash:
        raise EvidenceError("JTH sentinel/integrity analysis-code hashes disagree")
    verify_hash(JTH_ANALYZER, recorded_code_hash, "current JTH analyzer")
    required = [
        "publication_table_2_cluster_sensitivity.csv",
        "rank_fidelity_metric_summary_k20.csv",
        "publication_table_7_candidate_scale.csv",
        "publication_table_8_evaluation_bias_across_sizes.csv",
        "publication_table_9_subgroup_evaluation_bias.csv",
    ]
    verify_generated_hashes(directory, integrity.get("generated_file_sha256", {}), required)
    return {
        "directory": str(directory),
        "sentinel_sha256": sha256(directory / "AUTHORITATIVE_COMPLETE.json"),
        "integrity_sha256": sha256(integrity_path),
        "analysis_code_sha256": sentinel.get("analysis_code_sha256"),
    }


def validate_career(directory: Path) -> dict:
    integrity_path = require_file(directory, "analysis_integrity.json")
    integrity = read_json(integrity_path)
    if integrity.get("status") != "passed":
        raise EvidenceError("CareerBuilder analysis integrity status is not passed")
    authority = integrity.get("authority_validation", {})
    if (
        authority.get("status") != "passed"
        or authority.get("study_role") != "primary"
        or authority.get("protocol") != "First7Last2"
        or authority.get("users_per_window") != 1000
        or authority.get("users") != 7000
    ):
        raise EvidenceError(
            "CareerBuilder package is not the passed authoritative primary design"
        )
    recorded_code_hash = integrity.get("analysis_code", {}).get("sha256")
    verify_hash(CAREER_ANALYZER, recorded_code_hash, "current CareerBuilder analyzer")
    required = [
        "per_window_results.csv",
        "publication_table_2_sampling_bias.csv",
        "publication_table_3_rank_fidelity.csv",
        "publication_table_4_contrast_bootstrap.csv",
        "publication_table_5_subgroups.csv",
        "publication_table_6_candidate_scale.csv",
        "publication_table_9_subgroup_evaluation_bias.csv",
        "VERIFIED_FINDINGS.md",
    ]
    verify_generated_hashes(directory, integrity.get("generated_sha256", {}), required)
    return {
        "directory": str(directory),
        "integrity_sha256": sha256(integrity_path),
        "analysis_code_sha256": integrity.get("analysis_code", {}).get("sha256"),
        "run_label": integrity.get("run_label"),
        "protocol": integrity.get("protocol"),
    }


def validate_benchmark(directory: Path) -> dict:
    report_path = require_file(directory, "BENCHMARK_RUN_REPORT.json")
    audit_path = require_file(directory, "BENCHMARK_INTEGRITY_AUDIT.json")
    report = read_json(report_path)
    authority = report.get("authority", {})
    if authority != {
        "authoritative": True,
        "study_role": "direct_scoring_benchmark",
        "status": "completed",
    }:
        raise EvidenceError(f"Benchmark authority marker is invalid: {authority}")
    audit = read_json(audit_path)
    if report.get("integrity") != audit:
        raise EvidenceError("Benchmark run-report/integrity identity failed")
    outputs = report.get("outputs", {})
    required = [
        "BENCHMARK_CONFIG.json",
        "benchmark_primary_summary.csv",
        "benchmark_primary_window_summary.csv",
        "BENCHMARK_INTEGRITY_AUDIT.json",
    ]
    if not isinstance(outputs, dict) or not outputs:
        raise EvidenceError("Benchmark run report has no output manifest")
    if not set(required).issubset(outputs):
        raise EvidenceError("Benchmark run report omits a required output")
    for name, record in outputs.items():
        if Path(name).name != name:
            raise EvidenceError(f"Unsafe benchmark output key: {name!r}")
        record = outputs.get(name)
        if not isinstance(record, dict):
            raise EvidenceError(f"Benchmark report does not describe {name}")
        path = require_file(directory, name)
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise EvidenceError(f"Benchmark byte-count mismatch for {name}")
        verify_hash(path, record.get("sha256"), "benchmark output")
    config_path = require_file(directory, "BENCHMARK_CONFIG.json")
    config = read_json(config_path)
    if report.get("config_sha256") != outputs["BENCHMARK_CONFIG.json"].get("sha256"):
        raise EvidenceError("Benchmark run-report/config hash identity failed")
    verify_hash(
        BENCHMARK_SCRIPT, config.get("benchmark_script_sha256"),
        "current benchmark script",
    )
    verify_hash(
        CAREER_EVALUATOR, config.get("evaluator_module_sha256"),
        "current CareerBuilder evaluator",
    )
    if (
        config.get("authority", {}).get("authoritative") is not True
        or config.get("authority", {}).get("study_role") != "direct_scoring_benchmark"
        or config.get("not_effectiveness_runtime") is not True
        or config.get("protocol") != "First7Last2"
        or config.get("windows") != list(range(1, 8))
        or config.get("candidate_sizes") != [100, 500, 1000, 5000, 10000]
        or config.get("primary_repeats") != 5
    ):
        raise EvidenceError("Benchmark configuration violates the frozen design")
    for field in (
        "pair_invariance_failures", "checksum_invariance_failures",
        "duplicate_result_keys",
    ):
        if int(audit.get(field, -1)) != 0:
            raise EvidenceError(f"Benchmark integrity failure: {field}")
    return {
        "directory": str(directory),
        "run_report_sha256": sha256(report_path),
        "integrity_sha256": sha256(audit_path),
        "elapsed_seconds": report.get("elapsed_seconds"),
    }


def direction(value: float) -> str:
    if not np.isfinite(value) or abs(value) <= TOL:
        return "approximately_zero"
    return "positive" if value > 0 else "negative"


def classify_directions(left: float, right: float) -> str:
    a, b = direction(left), direction(right)
    if "approximately_zero" in (a, b):
        return "bounded_or_inconclusive_near_zero"
    if a == b:
        return "bounded_replication_same_direction"
    return "dataset_specific_opposite_direction"


def interval_direction(estimate: float, low: float, high: float) -> str:
    if not all(np.isfinite(value) for value in (estimate, low, high)):
        return "not_estimated"
    if low > TOL:
        return "positive_supported"
    if high < -TOL:
        return "negative_supported"
    return "interval_includes_zero"


def classify_interval_directions(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> str:
    a = interval_direction(*left)
    b = interval_direction(*right)
    if "not_estimated" in (a, b):
        return "not_comparable_interval_missing"
    if "interval_includes_zero" in (a, b):
        return "bounded_or_inconclusive_interval"
    if a == b:
        return "bounded_replication_interval_supported_direction"
    return "dataset_specific_interval_supported_opposite_direction"


def classification_category(label: str) -> str:
    if label.startswith("bounded_replication"):
        return "bounded_replication"
    if label.startswith("dataset_specific"):
        return "dataset_specific"
    if label.startswith("replicated"):
        return "replicated"
    return "not_comparable"


def normalize_candidate_scale(jth_dir: Path, cb_dir: Path) -> pd.DataFrame:
    j = pd.read_csv(jth_dir / "publication_table_7_candidate_scale.csv")
    require_columns(j, [
        "candidate_protocol", "strategy", "nominal_candidate_size",
        "eligible_catalog_mean", "eligible_catalog_median",
        "actual_candidate_count_mean", "realized_candidate_fraction_mean",
    ], "JTH candidate scale")
    j = j.rename(columns={"nominal_candidate_size": "candidate_size"})
    j["dataset"] = "JTH"
    j["candidate_regime"] = j["candidate_protocol"]
    j.loc[j["candidate_regime"].eq("full_catalog"), "strategy"] = "full_catalog"
    c = pd.read_csv(cb_dir / "publication_table_6_candidate_scale.csv")
    require_columns(c, [
        "sampling_strategy", "requested_candidate_size",
        "eligible_candidate_count_mean", "eligible_candidate_count_median",
        "actual_candidate_count_mean", "actual_candidate_fraction_mean",
    ], "CareerBuilder candidate scale")
    # Older analyzer extracts exposed one row per seed and explicitly carried
    # candidate_regime.  The authoritative analyzer now emits the already
    # collapsed publication table, records the number of contributing seeds in
    # `seeds`, and identifies the full catalog through sampling_strategy.  Both
    # representations encode the same candidate-scale estimand, so normalize
    # either schema while refusing an ambiguous third form.
    seed_level = "seed" in c.columns
    aggregate_level = "seeds" in c.columns
    if not seed_level and not aggregate_level:
        raise EvidenceError(
            "CareerBuilder candidate scale must declare either seed-level "
            "rows (`seed`) or an authoritative aggregate (`seeds`)"
        )
    if "candidate_regime" not in c.columns:
        c["candidate_regime"] = np.where(
            c["sampling_strategy"].eq("full_catalog"),
            "full_catalog",
            "sampled",
        )
    c = c.rename(columns={
        "sampling_strategy": "strategy",
        "requested_candidate_size": "candidate_size",
        "eligible_candidate_count_mean": "eligible_catalog_mean",
        "eligible_candidate_count_median": "eligible_catalog_median",
        "actual_candidate_fraction_mean": "realized_candidate_fraction_mean",
    })
    c["dataset"] = "CareerBuilder"
    c["candidate_size"] = pd.to_numeric(c["candidate_size"], errors="coerce")
    c.loc[c["candidate_regime"].eq("full_catalog"), "strategy"] = "full_catalog"
    cell_keys = ["dataset", "candidate_regime", "strategy", "candidate_size"]
    if seed_level:
        # Collapse equally sized seed-level query panels before cross-dataset
        # comparison.  This preserves compatibility with earlier extracts.
        c = c.groupby(cell_keys, dropna=False, as_index=False).agg(
            eligible_catalog_mean=("eligible_catalog_mean", "mean"),
            eligible_catalog_median=("eligible_catalog_median", "mean"),
            actual_candidate_count_mean=("actual_candidate_count_mean", "mean"),
            realized_candidate_fraction_mean=("realized_candidate_fraction_mean", "mean"),
        )
    else:
        seeds = pd.to_numeric(c["seeds"], errors="coerce")
        if seeds.isna().any() or (seeds < 1).any():
            raise EvidenceError(
                "CareerBuilder aggregate candidate scale has invalid seed counts"
            )
        if c.duplicated(cell_keys).any():
            raise EvidenceError(
                "CareerBuilder aggregate candidate scale has duplicate cells"
            )
    columns = [
        "dataset", "candidate_regime", "strategy", "candidate_size",
        "eligible_catalog_mean", "eligible_catalog_median",
        "actual_candidate_count_mean", "realized_candidate_fraction_mean",
    ]
    out = pd.concat([j[columns], c[columns]], ignore_index=True)
    out["candidate_size"] = pd.to_numeric(out["candidate_size"], errors="coerce")
    if out.duplicated(["dataset", "candidate_regime", "strategy", "candidate_size"]).any():
        raise EvidenceError("Candidate-scale normalization produced duplicate cells")
    return out.sort_values(["dataset", "candidate_regime", "strategy", "candidate_size"])


def normalize_bias(jth_dir: Path, cb_dir: Path, scale: pd.DataFrame) -> pd.DataFrame:
    j = pd.read_csv(jth_dir / "publication_table_8_evaluation_bias_across_sizes.csv")
    require_columns(j, [
        "strategy", "candidate_size", "model", "metric", "k", "estimate",
        "full_catalog_value", "sampled_value", "ci_low", "ci_high",
        "cluster_method",
    ], "JTH evaluation bias")
    j = j[j["cluster_method"].eq("two_way_candidate_date")].copy()
    j["dataset"] = "JTH"
    j["model_family"] = j["model"].map(MODEL_MAP_JTH)
    j = j.rename(columns={"estimate": "signed_bias_sampled_minus_full"})
    j["uncertainty_method"] = "two_way_candidate_date_cluster_bootstrap"
    c = pd.read_csv(cb_dir / "publication_table_2_sampling_bias.csv")
    require_columns(c, [
        "estimand", "sampling_strategy", "requested_candidate_size", "model",
        "metric", "k", "mean_signed_bias", "full_value", "sampled_seed_mean",
        "actual_candidate_fraction_mean",
    ], "CareerBuilder evaluation bias")
    c = c[c["estimand"].eq("design_weighted_population")].copy()
    c["dataset"] = "CareerBuilder"
    c["model_family"] = c["model"].map(MODEL_MAP_CB)
    c = c.rename(columns={
        "sampling_strategy": "strategy",
        "requested_candidate_size": "candidate_size",
        "mean_signed_bias": "signed_bias_sampled_minus_full",
        "full_value": "full_catalog_value",
        "sampled_seed_mean": "sampled_value",
    })
    c["ci_low"] = np.nan
    c["ci_high"] = np.nan
    c["uncertainty_method"] = "five_seed_descriptive_no_bias_bootstrap_interval"
    for frame in (j, c):
        frame["metric"] = frame["metric"].replace({"hit_rate": "hitrate"})
        frame["candidate_size"] = pd.to_numeric(frame["candidate_size"], errors="raise")
    columns = [
        "dataset", "strategy", "candidate_size", "model", "model_family",
        "metric", "k", "full_catalog_value", "sampled_value",
        "signed_bias_sampled_minus_full", "ci_low", "ci_high",
        "uncertainty_method", "actual_candidate_fraction_mean",
    ]
    for col in columns:
        if col not in j:
            j[col] = np.nan
        if col not in c:
            c[col] = np.nan
    out = pd.concat([j[columns], c[columns]], ignore_index=True)
    fractions = scale[scale["candidate_regime"].eq("sampled")][[
        "dataset", "strategy", "candidate_size", "realized_candidate_fraction_mean",
    ]]
    out = out.drop(columns=["actual_candidate_fraction_mean"]).merge(
        fractions,
        on=["dataset", "strategy", "candidate_size"],
        how="left",
        validate="many_to_one",
    ).rename(columns={
        "realized_candidate_fraction_mean": "actual_candidate_fraction_mean",
    })
    if out["actual_candidate_fraction_mean"].isna().any():
        raise EvidenceError("A bias cell lacks a realized candidate fraction")
    out["bias_direction"] = out["signed_bias_sampled_minus_full"].map(direction)
    out["absolute_bias"] = out["signed_bias_sampled_minus_full"].abs()
    out["signed_bias_percentage_points"] = (
        100.0 * out["signed_bias_sampled_minus_full"]
    )
    stable_denominator = (
        out["full_catalog_value"].abs() >= RELATIVE_BIAS_MIN_DENOMINATOR
    )
    out["relative_bias_vs_full"] = np.nan
    out.loc[stable_denominator, "relative_bias_vs_full"] = (
        out.loc[stable_denominator, "signed_bias_sampled_minus_full"]
        / out.loc[stable_denominator, "full_catalog_value"]
    )
    out["relative_bias_percent"] = 100.0 * out["relative_bias_vs_full"]
    out["relative_bias_status"] = np.where(
        stable_denominator,
        "reported",
        f"suppressed_abs_full_below_{RELATIVE_BIAS_MIN_DENOMINATOR:g}",
    )
    return out.sort_values(
        ["metric", "strategy", "model_family", "candidate_size", "dataset"]
    )


def normalize_rank(jth_dir: Path, cb_dir: Path, scale: pd.DataFrame) -> pd.DataFrame:
    j = pd.read_csv(jth_dir / "rank_fidelity_metric_summary_k20.csv")
    require_columns(j, [
        "strategy", "candidate_size", "metric", "kendall_tau_b",
        "pairwise_order_agreement", "winner_set_exact_agreement",
        "winner_set_overlap", "winner_regret_best_case",
        "winner_regret_worst_case", "n_seed_decisions",
    ], "JTH rank fidelity")
    j["dataset"] = "JTH"
    j = j.rename(columns={
        "kendall_tau_b": "kendall_tau_b_mean",
        "pairwise_order_agreement": "pairwise_order_agreement_mean",
        "winner_set_exact_agreement": "winner_exact_rate",
        "winner_set_overlap": "winner_overlap_rate",
        "winner_regret_best_case": "winner_regret_best_mean",
        "winner_regret_worst_case": "winner_regret_worst_mean",
    })
    c = pd.read_csv(cb_dir / "publication_table_3_rank_fidelity.csv")
    require_columns(c, [
        "rank_scope", "estimand", "sampling_strategy",
        "requested_candidate_size", "metric", "k", "kendall_tau_b_mean",
        "pairwise_order_agreement_mean", "winner_set_exact_match_rate",
        "winner_set_overlap_rate", "sampled_winner_regret_best_mean",
        "sampled_winner_regret_worst_mean",
    ], "CareerBuilder rank fidelity")
    c = c[
        c["rank_scope"].eq("pooled")
        & c["estimand"].eq("design_weighted_population")
        & c["k"].eq(20)
    ].copy()
    c["dataset"] = "CareerBuilder"
    c = c.rename(columns={
        "sampling_strategy": "strategy",
        "requested_candidate_size": "candidate_size",
        "winner_set_exact_match_rate": "winner_exact_rate",
        "winner_set_overlap_rate": "winner_overlap_rate",
        "sampled_winner_regret_best_mean": "winner_regret_best_mean",
        "sampled_winner_regret_worst_mean": "winner_regret_worst_mean",
    })
    for frame in (j, c):
        frame["metric"] = frame["metric"].replace({"hit_rate": "hitrate"})
        frame["candidate_size"] = pd.to_numeric(
            frame["candidate_size"], errors="raise"
        )
    columns = [
        "dataset", "strategy", "candidate_size", "metric",
        "kendall_tau_b_mean", "pairwise_order_agreement_mean",
        "winner_exact_rate", "winner_overlap_rate",
        "winner_regret_best_mean", "winner_regret_worst_mean",
    ]
    out = pd.concat([j[columns], c[columns]], ignore_index=True)
    fractions = scale[scale["candidate_regime"].eq("sampled")][
        ["dataset", "strategy", "candidate_size", "realized_candidate_fraction_mean"]
    ].drop_duplicates()
    return out.merge(
        fractions, on=["dataset", "strategy", "candidate_size"], how="left",
        validate="many_to_one",
    )


def normalize_contrasts(jth_dir: Path, cb_dir: Path) -> pd.DataFrame:
    j = pd.read_csv(jth_dir / "publication_table_2_cluster_sensitivity.csv")
    require_columns(j, [
        "strategy", "candidate_size", "seed_aggregation", "metric", "contrast",
        "cluster_method", "estimate", "ci_low", "ci_high",
    ], "JTH contrasts")
    j = j[
        j["cluster_method"].eq("two_way_candidate_date")
        & (
            j["strategy"].eq("full")
            | j["seed_aggregation"].eq("mean_over_5_seeds")
        )
    ].copy()
    j["dataset"] = "JTH"
    j["condition"] = np.where(j["strategy"].eq("full"), "full_catalog", "sampled")
    j.loc[j["condition"].eq("full_catalog"), "strategy"] = "full_catalog"
    j["effect_type"] = "contrast"
    j["contrast_family"] = j["contrast"].map(CONTRAST_FAMILY_JTH)
    j["uncertainty_method"] = "two_way_candidate_date_cluster_bootstrap"
    c = pd.read_csv(cb_dir / "publication_table_4_contrast_bootstrap.csv")
    require_columns(c, [
        "estimand", "candidate_regime", "sampling_strategy", "seed_aggregation",
        "requested_candidate_size", "contrast", "metric", "k", "effect_type",
        "estimate", "ci_low", "ci_high",
    ], "CareerBuilder contrasts")
    c = c[
        c["estimand"].eq("design_weighted_population")
        & (
            (
                c["candidate_regime"].eq("full_catalog")
                & c["effect_type"].eq("contrast")
            )
            | (
                c["candidate_regime"].eq("sampled")
                & c["seed_aggregation"].eq("mean_over_5_seeds")
                & c["effect_type"].isin([
                    "contrast", "effect_distortion_sampled_minus_full",
                ])
            )
        )
    ].copy()
    c["dataset"] = "CareerBuilder"
    c["condition"] = c["candidate_regime"]
    c["contrast_family"] = c["contrast"].map(CONTRAST_FAMILY_CB)
    c["uncertainty_method"] = "window_stratified_paired_user_bootstrap"
    c = c.rename(columns={
        "sampling_strategy": "strategy",
        "requested_candidate_size": "candidate_size",
    })
    c.loc[c["condition"].eq("full_catalog"), "strategy"] = "full_catalog"
    j["k"] = 20
    columns = [
        "dataset", "condition", "strategy", "candidate_size", "metric", "k",
        "contrast", "contrast_family", "effect_type", "seed_aggregation",
        "estimate", "ci_low", "ci_high", "uncertainty_method",
    ]
    for frame in (j, c):
        frame["metric"] = frame["metric"].replace({"hit_rate": "hitrate"})
        frame["candidate_size"] = pd.to_numeric(
            frame["candidate_size"], errors="coerce"
        ).fillna(-1)
    return pd.concat([j[columns], c[columns]], ignore_index=True).sort_values(
        ["contrast", "metric", "condition", "strategy", "candidate_size", "dataset"]
    )


def subgroup_boundary_table(jth_dir: Path, cb_dir: Path) -> pd.DataFrame:
    j = pd.read_csv(jth_dir / "publication_table_9_subgroup_evaluation_bias.csv")
    require_columns(j, [
        "subgroup_dimension", "subgroup_value", "metric", "strategy", "model",
        "candidate_size", "estimate", "full_catalog_value", "sampled_value",
        "ci_low", "ci_high", "cluster_adequacy", "n_queries",
        "bootstrap_replicates",
    ], "JTH subgroup evaluation bias")
    j = j.copy()
    j["metric"] = j["metric"].replace({"hit_rate": "hitrate"})
    j["dataset"] = "JTH"
    j["boundary_type"] = np.where(
        j["subgroup_dimension"].eq("timestamp-rank quartile"),
        "temporal", "subgroup",
    )
    j["model_family"] = j["model"].map(MODEL_MAP_JTH)
    j["canonical_subgroup"] = [
        SUBGROUP_MAP_JTH.get((dimension, value))
        for dimension, value in zip(j["subgroup_dimension"], j["subgroup_value"])
    ]
    j["canonical_subgroup"] = j["canonical_subgroup"].map(
        lambda value: "::".join(value) if isinstance(value, tuple) else np.nan
    )
    j["evidence_type"] = "sampled_minus_full_distortion"
    j["within_dataset_interval_status"] = np.where(
        j["cluster_adequacy"].astype(str).str.startswith("adequate"),
        "dependence_aware_interval_available",
        "interval_not_estimated",
    )
    j["n_windows"] = np.nan
    adequate = j["cluster_adequacy"].astype(str).str.startswith("adequate")
    if not j.loc[adequate, "bootstrap_replicates"].eq(2000).all():
        raise EvidenceError("JTH subgroup-distortion intervals are not the frozen 2,000-replicate design")
    rows = [j]
    c = pd.read_csv(cb_dir / "publication_table_9_subgroup_evaluation_bias.csv")
    require_columns(c, [
        "subgroup_dimension", "subgroup_value", "metric",
        "sampling_strategy", "model", "requested_candidate_size", "estimand",
        "effect_type", "seed_aggregation", "k", "estimate",
        "full_catalog_value", "sampled_value", "ci_low", "ci_high",
        "bootstrap_replicates", "n_queries", "n_windows",
    ], "CareerBuilder subgroup evaluation bias")
    c = c[
        c["estimand"].eq("design_weighted_population")
    ].copy()
    c = c.rename(columns={
        "sampling_strategy": "strategy",
        "requested_candidate_size": "candidate_size",
    })
    c["metric"] = c["metric"].replace({"hit_rate": "hitrate"})
    if (
        len(c) != 840
        or not c["effect_type"].eq("sampled_minus_full_metric_bias").all()
        or not c["seed_aggregation"].eq("per_user_mean_over_5_seeds").all()
        or not c["bootstrap_replicates"].eq(2000).all()
        or not c["n_windows"].eq(7).all()
    ):
        raise EvidenceError(
            "CareerBuilder primary subgroup-distortion extract violates its frozen design"
        )
    c["dataset"] = "CareerBuilder"
    c["boundary_type"] = "subgroup"
    c["model_family"] = c["model"].map(MODEL_MAP_CB)
    c["canonical_subgroup"] = [
        SUBGROUP_MAP_CB.get((dimension, value))
        for dimension, value in zip(c["subgroup_dimension"], c["subgroup_value"])
    ]
    c["canonical_subgroup"] = c["canonical_subgroup"].map(
        lambda value: "::".join(value) if isinstance(value, tuple) else np.nan
    )
    c["evidence_type"] = "sampled_minus_full_distortion"
    c["cluster_adequacy"] = "window-stratified paired user bootstrap"
    c["within_dataset_interval_status"] = "dependence_aware_interval_available"
    rows.append(c)

    temporal = career_temporal_bias_rows(cb_dir)
    rows.append(temporal)
    columns = [
        "dataset", "boundary_type", "subgroup_dimension", "subgroup_value",
        "canonical_subgroup", "metric", "strategy", "model", "model_family",
        "candidate_size", "full_catalog_value", "sampled_value", "estimate",
        "ci_low", "ci_high", "n_queries", "n_windows", "bootstrap_replicates",
        "cluster_adequacy", "evidence_type", "within_dataset_interval_status",
    ]
    return pd.concat([x[columns] for x in rows], ignore_index=True)


def career_temporal_bias_rows(cb_dir: Path) -> pd.DataFrame:
    """Create descriptive per-window bias rows without inventing intervals."""
    frame = pd.read_csv(cb_dir / "per_window_results.csv")
    require_columns(frame, [
        "window", "estimand", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k", "users", "hit_rate", "map",
    ], "CareerBuilder per-window results")
    frame = frame[
        frame["estimand"].eq("audit_cohort")
        & frame["k"].eq(20)
    ].copy()
    full = frame[frame["candidate_regime"].eq("full_catalog")]
    if full.duplicated(["window", "model", "k"]).any():
        raise EvidenceError("CareerBuilder per-window full-catalog cells are duplicated")
    full_long = full.melt(
        id_vars=["window", "model", "k", "users"],
        value_vars=["hit_rate", "map"],
        var_name="metric", value_name="full_catalog_value",
    )
    sampled = frame[frame["candidate_regime"].eq("sampled")]
    sampled_long = sampled.melt(
        id_vars=[
            "window", "sampling_strategy", "requested_candidate_size",
            "model", "k", "seed", "users",
        ],
        value_vars=["hit_rate", "map"],
        var_name="metric", value_name="sampled_seed_value",
    )
    sampled_mean = sampled_long.groupby([
        "window", "sampling_strategy", "requested_candidate_size", "model",
        "k", "users", "metric",
    ], as_index=False).agg(
        sampled_value=("sampled_seed_value", "mean"),
        seeds=("seed", "nunique"),
    )
    if not sampled_mean["seeds"].eq(5).all():
        raise EvidenceError("CareerBuilder per-window temporal cells lack five seeds")
    merged = sampled_mean.merge(
        full_long,
        on=["window", "model", "k", "users", "metric"],
        how="left", validate="many_to_one",
    )
    if merged["full_catalog_value"].isna().any():
        raise EvidenceError("CareerBuilder temporal bias lacks a full reference")
    merged["dataset"] = "CareerBuilder"
    merged["boundary_type"] = "temporal"
    merged["subgroup_dimension"] = "provider-assigned temporal window"
    merged["subgroup_value"] = "W" + merged["window"].astype(int).astype(str)
    merged["canonical_subgroup"] = np.nan
    merged["strategy"] = merged["sampling_strategy"]
    merged["candidate_size"] = merged["requested_candidate_size"]
    merged["metric"] = merged["metric"].replace({"hit_rate": "hitrate"})
    merged["model_family"] = merged["model"].map(MODEL_MAP_CB)
    merged["estimate"] = merged["sampled_value"] - merged["full_catalog_value"]
    merged["ci_low"] = np.nan
    merged["ci_high"] = np.nan
    merged["n_queries"] = merged["users"]
    merged["n_windows"] = 1
    merged["bootstrap_replicates"] = 0
    merged["cluster_adequacy"] = "descriptive only; no per-window interval"
    merged["evidence_type"] = "sampled_minus_full_distortion"
    merged["within_dataset_interval_status"] = "descriptive_no_interval"
    return merged


def compute_table(benchmark_dir: Path, scale: pd.DataFrame) -> pd.DataFrame:
    b = pd.read_csv(benchmark_dir / "benchmark_primary_summary.csv")
    require_columns(b, [
        "model", "candidate_regime", "requested_candidate_size",
        "total_seconds_mean", "total_seconds_sd", "score_seconds_mean",
        "rank_seconds_mean", "metric_seconds_mean", "scored_pairs_mean",
        "peak_delta_bytes_max",
    ], "CareerBuilder direct-scoring benchmark")
    b["dataset"] = "CareerBuilder"
    b["strategy"] = np.where(b["candidate_regime"].eq("full_catalog"), "full_catalog", "uniform")
    b = b.rename(columns={"requested_candidate_size": "candidate_size"})
    b["candidate_size"] = pd.to_numeric(b["candidate_size"], errors="coerce").fillna(-1)
    fractions = scale[scale["dataset"].eq("CareerBuilder")][
        ["strategy", "candidate_size", "realized_candidate_fraction_mean"]
    ].drop_duplicates().copy()
    # Pandas does not match NaN join keys.  Use the same explicit sentinel for
    # the full-catalog row on both sides; sampled sizes remain unchanged.
    fractions["candidate_size"] = pd.to_numeric(
        fractions["candidate_size"], errors="coerce"
    ).fillna(-1)
    b = b.merge(
        fractions, on=["strategy", "candidate_size"], how="left",
        validate="many_to_one",
    )
    if b["realized_candidate_fraction_mean"].isna().any():
        raise EvidenceError("A benchmark condition lacks a candidate-fraction match")
    full_reference = b[b["candidate_regime"].eq("full_catalog")][[
        "model", "total_seconds_mean", "scored_pairs_mean",
    ]].rename(columns={
        "total_seconds_mean": "full_total_seconds_mean",
        "scored_pairs_mean": "full_scored_pairs_mean",
    })
    if full_reference.duplicated("model").any():
        raise EvidenceError("Benchmark has duplicate full-catalog model references")
    b = b.merge(full_reference, on="model", how="left", validate="many_to_one")
    b["relative_to_full_time"] = (
        b["total_seconds_mean"] / b["full_total_seconds_mean"]
    )
    b["scored_pair_fraction_vs_full"] = (
        b["scored_pairs_mean"] / b["full_scored_pairs_mean"]
    )
    b["evidence_boundary"] = (
        "direct scoring plus deterministic ranking and metric calculation; "
        "loading fitting encoding candidate generation and I/O excluded"
    )
    j = scale[
        scale["dataset"].eq("JTH") & scale["candidate_regime"].eq("sampled")
    ][["dataset", "strategy", "candidate_size", "realized_candidate_fraction_mean"]].copy()
    for col in (
        "model", "candidate_regime", "total_seconds_mean", "total_seconds_sd",
        "score_seconds_mean", "rank_seconds_mean", "metric_seconds_mean",
        "scored_pairs_mean", "peak_delta_bytes_max", "full_total_seconds_mean",
        "full_scored_pairs_mean", "relative_to_full_time",
        "scored_pair_fraction_vs_full",
    ):
        j[col] = np.nan
    j["candidate_regime"] = "sampled"
    j["evidence_boundary"] = (
        "analytical candidate fraction only; no directly comparable JTH timing benchmark"
    )
    columns = [
        "dataset", "model", "candidate_regime", "strategy", "candidate_size",
        "realized_candidate_fraction_mean", "total_seconds_mean", "total_seconds_sd",
        "score_seconds_mean", "rank_seconds_mean", "metric_seconds_mean",
        "scored_pairs_mean", "peak_delta_bytes_max", "full_total_seconds_mean",
        "full_scored_pairs_mean", "relative_to_full_time",
        "scored_pair_fraction_vs_full", "evidence_boundary",
    ]
    return pd.concat([b[columns], j[columns]], ignore_index=True)


def classification_matrix(
    bias: pd.DataFrame,
    rank: pd.DataFrame,
    contrasts: pd.DataFrame,
    subgroup: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict] = []

    def add(
        *, evidence_domain: str, strategy: str, candidate_size: float,
        metric: str, model_or_scope: str, jth_value: float = np.nan,
        careerbuilder_value: float = np.nan, jth_ci_low: float = np.nan,
        jth_ci_high: float = np.nan, careerbuilder_ci_low: float = np.nan,
        careerbuilder_ci_high: float = np.nan, classification: str,
        claim_boundary: str,
    ) -> None:
        records.append({
            "evidence_domain": evidence_domain,
            "strategy": strategy,
            "candidate_size": candidate_size,
            "metric": metric,
            "model_or_scope": model_or_scope,
            "jth_value": jth_value,
            "careerbuilder_value": careerbuilder_value,
            "jth_ci_low": jth_ci_low,
            "jth_ci_high": jth_ci_high,
            "careerbuilder_ci_low": careerbuilder_ci_low,
            "careerbuilder_ci_high": careerbuilder_ci_high,
            "classification": classification,
            "classification_category": classification_category(classification),
            "claim_boundary": claim_boundary,
        })

    common_bias = bias[
        bias["candidate_size"].isin(COMMON_SIZES)
        & bias["strategy"].isin(COMMON_STRATEGIES)
        & bias["metric"].isin(COMMON_METRICS)
        & bias["k"].eq(20)
        & bias["model_family"].isin(["BM25", "MiniLM", "TFIDF", "Popularity", "Random"])
    ]
    keys = ["strategy", "candidate_size", "metric", "model_family"]
    j_bias = common_bias[common_bias.dataset.eq("JTH")][
        keys + ["signed_bias_sampled_minus_full"]
    ].rename(columns={"signed_bias_sampled_minus_full": "jth_value"})
    c_bias = common_bias[common_bias.dataset.eq("CareerBuilder")][
        keys + ["signed_bias_sampled_minus_full"]
    ].rename(columns={"signed_bias_sampled_minus_full": "careerbuilder_value"})
    paired_bias = j_bias.merge(c_bias, on=keys, how="inner", validate="one_to_one")
    for row in paired_bias.itertuples(index=False):
        add(
            evidence_domain="metric_bias_direction",
            strategy=row.strategy, candidate_size=row.candidate_size,
            metric=row.metric, model_or_scope=row.model_family,
            jth_value=row.jth_value,
            careerbuilder_value=row.careerbuilder_value,
            classification=classify_directions(
                row.jth_value, row.careerbuilder_value
            ),
            claim_boundary=(
                "directional comparison only; raw effectiveness and bias magnitude "
                "are not pooled; model configurations and native tasks differ"
            ),
        )

    common_rank = rank[
        rank["candidate_size"].isin(COMMON_SIZES)
        & rank["strategy"].isin(COMMON_STRATEGIES)
        & rank["metric"].isin(COMMON_METRICS)
    ].copy()
    rank_keys = ["strategy", "candidate_size", "metric"]
    measures = {
        "rank_concordance": "kendall_tau_b_mean",
        "pairwise_order_fidelity": "pairwise_order_agreement_mean",
        "winner_fidelity": "winner_exact_rate",
        "winner_regret": "winner_regret_worst_mean",
    }
    jr = common_rank[common_rank.dataset.eq("JTH")][rank_keys + list(measures.values())]
    cr = common_rank[common_rank.dataset.eq("CareerBuilder")][rank_keys + list(measures.values())]
    paired_rank = jr.merge(
        cr, on=rank_keys, how="inner", suffixes=("_jth", "_career"),
        validate="one_to_one",
    )
    for row in paired_rank.itertuples(index=False):
        for domain, column in measures.items():
            left = float(getattr(row, f"{column}_jth"))
            right = float(getattr(row, f"{column}_career"))
            if domain == "rank_concordance":
                left_loss, right_loss = 1.0 - left, 1.0 - right
            elif domain in {"pairwise_order_fidelity", "winner_fidelity"}:
                left_loss, right_loss = 1.0 - left, 1.0 - right
            else:
                left_loss, right_loss = left, right
            a, b = direction(left_loss), direction(right_loss)
            if a == b == "approximately_zero":
                cls = f"bounded_replication_no_{domain}_loss"
            elif a != "approximately_zero" and b != "approximately_zero":
                cls = f"bounded_replication_nonzero_{domain}_loss"
            else:
                cls = f"dataset_specific_{domain}_loss_presence"
            add(
                evidence_domain=domain,
                strategy=row.strategy, candidate_size=row.candidate_size,
                metric=row.metric, model_or_scope="native evaluated model set",
                jth_value=left, careerbuilder_value=right,
                classification=cls,
                claim_boundary=(
                    "metric-matched descriptive comparison with native model sets; "
                    "model membership and text fields differ across datasets"
                ),
            )

    comparable_contrasts = contrasts[
        contrasts["contrast_family"].notna()
        & contrasts["effect_type"].eq("contrast")
        & contrasts["metric"].isin(COMMON_METRICS)
        & contrasts["k"].eq(20)
        & (
            contrasts["condition"].eq("full_catalog")
            | (
                contrasts["strategy"].isin(COMMON_STRATEGIES)
                & contrasts["candidate_size"].isin(COMMON_SIZES)
            )
        )
    ].copy()
    contrast_keys = [
        "condition", "strategy", "candidate_size", "metric", "contrast_family",
    ]
    jc = comparable_contrasts[comparable_contrasts.dataset.eq("JTH")][
        contrast_keys + ["estimate", "ci_low", "ci_high"]
    ]
    cc = comparable_contrasts[comparable_contrasts.dataset.eq("CareerBuilder")][
        contrast_keys + ["estimate", "ci_low", "ci_high"]
    ]
    paired_contrasts = jc.merge(
        cc, on=contrast_keys, how="inner", suffixes=("_jth", "_career"),
        validate="one_to_one",
    )
    for row in paired_contrasts.itertuples(index=False):
        cls = classify_interval_directions(
            (row.estimate_jth, row.ci_low_jth, row.ci_high_jth),
            (row.estimate_career, row.ci_low_career, row.ci_high_career),
        )
        add(
            evidence_domain="prespecified_bm25_contrast",
            strategy=row.strategy, candidate_size=row.candidate_size,
            metric=row.metric, model_or_scope=row.contrast_family,
            jth_value=row.estimate_jth, careerbuilder_value=row.estimate_career,
            jth_ci_low=row.ci_low_jth, jth_ci_high=row.ci_high_jth,
            careerbuilder_ci_low=row.ci_low_career,
            careerbuilder_ci_high=row.ci_high_career,
            classification=cls,
            claim_boundary=(
                "interval-supported direction only; BM25/MiniLM and BM25/TF-IDF "
                "field configurations are named and differ across datasets"
            ),
        )

    comparable_subgroups = subgroup[
        subgroup["boundary_type"].eq("subgroup")
        & subgroup["canonical_subgroup"].notna()
        & subgroup["model_family"].isin([
            "BM25", "MiniLM", "TFIDF", "Popularity", "Random",
        ])
        & subgroup["metric"].isin(COMMON_METRICS)
        & subgroup["strategy"].isin(COMMON_STRATEGIES)
        & subgroup["candidate_size"].isin(COMMON_SIZES)
    ].copy()
    subgroup_keys = [
        "canonical_subgroup", "strategy", "candidate_size", "metric",
        "model_family",
    ]
    js = comparable_subgroups[comparable_subgroups.dataset.eq("JTH")][
        subgroup_keys + ["estimate", "ci_low", "ci_high"]
    ]
    cs = comparable_subgroups[comparable_subgroups.dataset.eq("CareerBuilder")][
        subgroup_keys + ["estimate", "ci_low", "ci_high"]
    ]
    paired_subgroups = js.merge(
        cs, on=subgroup_keys, how="inner", suffixes=("_jth", "_career"),
        validate="one_to_one",
    )
    for row in paired_subgroups.itertuples(index=False):
        cls = classify_interval_directions(
            (row.estimate_jth, row.ci_low_jth, row.ci_high_jth),
            (row.estimate_career, row.ci_low_career, row.ci_high_career),
        )
        add(
            evidence_domain="subgroup_distortion",
            strategy=row.strategy, candidate_size=row.candidate_size,
            metric=row.metric,
            model_or_scope=f"{row.canonical_subgroup}; {row.model_family}",
            jth_value=row.estimate_jth, careerbuilder_value=row.estimate_career,
            jth_ci_low=row.ci_low_jth, jth_ci_high=row.ci_high_jth,
            careerbuilder_ci_low=row.ci_low_career,
            careerbuilder_ci_high=row.ci_high_career,
            classification=cls,
            claim_boundary=(
                "descriptive subgroup distortion only; threshold-aligned cold, "
                "one-event, and target-popularity groups retain dataset-native "
                "event semantics; target groups are outcome-derived"
            ),
        )

    add(
        evidence_domain="activity_subgroup_bin_definition",
        strategy="all", candidate_size=np.nan, metric="all",
        model_or_scope="JTH 2-3/4+ versus CareerBuilder 2-4/5+",
        classification="not_comparable_activity_bin_boundaries",
        claim_boundary="Only cold_0 and one_1 activity groups are directly aligned.",
    )
    add(
        evidence_domain="temporal_boundary_structure",
        strategy="all", candidate_size=np.nan, metric="all",
        model_or_scope="JTH chronological quartiles versus provider-assigned windows",
        classification="not_comparable_temporal_partitions",
        claim_boundary=(
            "JTH T1-T4 query-time quartiles and CareerBuilder's seven randomly "
            "assigned provider windows answer different temporal questions."
        ),
    )
    add(
        evidence_domain="compute_timing",
        strategy="uniform", candidate_size=np.nan, metric="seconds",
        model_or_scope="direct scoring, ranking, and metrics",
        classification="not_comparable_jth_timing_not_measured",
        claim_boundary=(
            "CareerBuilder has a controlled direct-scoring benchmark; JTH has "
            "candidate fractions but no directly comparable timing measurement."
        ),
    )
    return pd.DataFrame(records)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(output / f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def make_figures(
    output: Path, bias: pd.DataFrame, rank: pd.DataFrame, compute: pd.DataFrame
) -> list[str]:
    stems = []
    colors = {"JTH": "#1f77b4", "CareerBuilder": "#d62728"}
    styles = {"BM25": "-", "MiniLM": "--", "TFIDF": ":"}
    metric_labels = {"hitrate": "HitRate@20", "map": "MAP@20"}
    model_labels = {
        ("JTH", "BM25"): "JTH BM25",
        ("JTH", "MiniLM"): "JTH MiniLM",
        ("JTH", "TFIDF"): "JTH TF–IDF",
        ("CareerBuilder", "BM25"): "CareerBuilder full-text BM25",
        ("CareerBuilder", "MiniLM"): "CareerBuilder title MiniLM",
        ("CareerBuilder", "TFIDF"): "CareerBuilder combined TF–IDF",
    }
    selected = bias[
        bias["k"].eq(20)
        & bias["metric"].isin(COMMON_METRICS)
        & bias["model_family"].isin(styles)
        & bias["strategy"].isin(COMMON_STRATEGIES)
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=False)
    for row, metric in enumerate(COMMON_METRICS):
        for col, strategy in enumerate(COMMON_STRATEGIES):
            ax = axes[row, col]
            cell = selected[selected.metric.eq(metric) & selected.strategy.eq(strategy)]
            for (dataset, model), part in cell.groupby(["dataset", "model_family"]):
                part = part.sort_values("actual_candidate_fraction_mean")
                ax.plot(
                    part.actual_candidate_fraction_mean, part.signed_bias_sampled_minus_full,
                    color=colors[dataset], linestyle=styles[model], marker="o", markersize=3,
                    label=model_labels[(dataset, model)],
                )
            ax.axhline(0, color="#777777", linewidth=.8)
            ax.set_xscale("log")
            strategy_label = strategy.replace("_", "-").capitalize()
            ax.set_title(f"{metric_labels[metric]} — {strategy_label}")
            ax.set_xlabel("Mean realized candidate fraction")
            ax.set_ylabel("Sampled − sampling-free reference")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Metric differences relative to sampling-free evaluation")
    fig.tight_layout(rect=(0, .08, 1, .96))
    save_figure(fig, output, "figure_1_cross_dataset_metric_bias")
    stems.append("figure_1_cross_dataset_metric_bias")

    collapsed = rank.groupby(
        ["dataset", "strategy", "candidate_size", "realized_candidate_fraction_mean"],
        as_index=False,
    ).agg(
        kendall_tau_b_mean=("kendall_tau_b_mean", "mean"),
        winner_exact_rate=("winner_exact_rate", "mean"),
        pairwise_order_agreement_mean=("pairwise_order_agreement_mean", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=False)
    measures = [
        ("kendall_tau_b_mean", "Mean Kendall’s τb"),
        ("pairwise_order_agreement_mean", "Mean pairwise-order agreement"),
        ("winner_exact_rate", "Mean top-model-set agreement"),
    ]
    for ax, (column, label) in zip(axes, measures):
        for (dataset, strategy), part in collapsed.groupby(["dataset", "strategy"]):
            part = part.sort_values("realized_candidate_fraction_mean")
            ax.plot(
                part.realized_candidate_fraction_mean, part[column], marker="o",
                color=colors[dataset], linestyle="-" if strategy == "uniform" else "--",
                label=f"{dataset} {strategy.replace('_', ' ')}",
            )
        ax.set_xscale("log")
        ax.set_ylim(-.05, 1.05)
        ax.set_xlabel("Mean realized candidate fraction")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Model-ranking and selection agreement across candidate fractions")
    fig.tight_layout(rect=(0, .15, 1, .94))
    save_figure(fig, output, "figure_2_cross_dataset_rank_winner_fidelity")
    stems.append("figure_2_cross_dataset_rank_winner_fidelity")

    c = compute[
        compute.dataset.eq("CareerBuilder")
        & compute.candidate_regime.astype(str).str.startswith("sampled")
    ].copy()
    if c.empty:
        raise EvidenceError(
            "Compute figure has no CareerBuilder sampled benchmark rows"
        )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    compute_labels = {
        "bm25_full_static": "BM25 (full text)",
        "minilm_title_static": "MiniLM (title)",
        "tfidf_combined_static": "TF-IDF (combined)",
    }
    for model, part in c.groupby("model"):
        part = part.sort_values("realized_candidate_fraction_mean")
        ax.errorbar(
            part.realized_candidate_fraction_mean, part.total_seconds_mean,
            yerr=part.total_seconds_sd,
            marker="o", capsize=2.5, linewidth=1.5,
            label=compute_labels.get(model, model),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean realized candidate fraction")
    ax.set_ylabel("Post-warm-up scoring, ranking, and metric time (s)")
    ax.set_title("CareerBuilder time across sampled candidate fractions")
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        raise EvidenceError("Compute figure has no plotted model series")
    ax.legend(handles, labels, frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "figure_3_compute_evidence_boundary")
    stems.append("figure_3_compute_evidence_boundary")
    return stems


def protocol_markdown() -> str:
    return """# Cross-dataset synthesis protocol

## Evidence gate

The synthesis reads only a completed JTH v2 package, a passed authoritative
CareerBuilder primary analysis package, and a completed authoritative CareerBuilder
direct-scoring benchmark. It verifies the source packages and hashes before
creating an output directory. Per-query and per-user files are never read.

## Comparability rules

Raw effectiveness values are not pooled, averaged, or used to rank datasets.
Metric bias is compared directionally within shared sampler, candidate-size,
metric, cutoff, and named model-family cells. Signed bias, absolute bias, and
percentage points stay within each dataset. Relative bias is reported only when
the absolute full-catalog value is at least 0.01; otherwise it is explicitly
suppressed as denominator-unstable. Rank and winner fidelity are
metric matched but remain descriptive native-model-set properties because the
evaluated model sets differ.
Candidate fractions are reported beside nominal sizes. BM25 contrasts retain
their exact configuration names; family-level parallels use interval-supported
direction only and are bounded by field and task differences. Subgroup distortion
is compared only for prespecified threshold-aligned groups when both datasets
supply paired sampled-minus-full intervals; event semantics remain dataset
native. Higher activity bins are not aligned. JTH's
chronological quartiles and CareerBuilder's provider-assigned windows remain
separate temporal descriptions, not replications. Measured direct-scoring time is
a CareerBuilder-only boundary; JTH contributes analytical candidate fractions,
not a timing result.

## Classification vocabulary

- `bounded_replication_*`: the datasets share a direction or loss-presence
  finding, subject to the recorded task, model, and configuration boundary.
- `dataset_specific_*`: the direction or decision-loss finding differs.
- `bounded_or_inconclusive_*`: an effect is near zero or its
  dependence-aware interval includes zero.
- `not_comparable_*`: a construct, interval, matched definition, or measured
  boundary is not available in both datasets.

The `classification_category` column collapses detailed labels to `replicated`,
`bounded_replication`, `dataset_specific`, or `not_comparable`. Because datasets,
tasks, model sets, and text fields differ, exact `replicated` labels are not
issued automatically; supported parallels are bounded replications.

These labels describe this study's datasets, protocols, models, and metrics. They
are not universal candidate-size recommendations.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jth-dir", type=Path, default=ROOT / "Results" / "JTH_EvaluationValidity_Strengthened_v2")
    parser.add_argument("--career-dir", type=Path, default=ROOT / "Results" / "CareerBuilder_InternalTemporal_Authoritative_Analysis_v1" / "First7Last2")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "Results" / "CareerBuilder_InternalTemporal_EvaluationValidity" / "RQ5_DirectScoring_Benchmark_Authoritative_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Results" / "CrossDataset_EvaluationValidity_Synthesis_v1")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def synthetic_self_test() -> None:
    assert direction(0.0) == "approximately_zero"
    assert direction(0.01) == "positive"
    assert direction(-0.01) == "negative"
    assert classify_directions(0.1, 0.2) == "bounded_replication_same_direction"
    assert classify_directions(-0.1, 0.2) == "dataset_specific_opposite_direction"
    assert classify_directions(0.0, 0.2) == "bounded_or_inconclusive_near_zero"
    assert interval_direction(0.2, 0.1, 0.3) == "positive_supported"
    assert interval_direction(-0.2, -0.3, -0.1) == "negative_supported"
    assert interval_direction(0.0, -0.1, 0.1) == "interval_includes_zero"
    assert classify_interval_directions(
        (0.2, 0.1, 0.3), (0.4, 0.2, 0.6)
    ) == "bounded_replication_interval_supported_direction"
    frame = pd.DataFrame({
        "dataset": ["JTH", "CareerBuilder"],
        "strategy": ["uniform", "uniform"],
        "candidate_size": [100, 100],
        "metric": ["hitrate", "hitrate"],
        "model_family": ["BM25", "BM25"],
        "k": [20, 20],
        "signed_bias_sampled_minus_full": [0.1, 0.2],
    })
    rank = pd.DataFrame(columns=[
        "dataset", "strategy", "candidate_size", "metric",
        "kendall_tau_b_mean", "pairwise_order_agreement_mean",
        "winner_exact_rate", "winner_regret_worst_mean",
    ])
    contrasts = pd.DataFrame(columns=[
        "dataset", "condition", "strategy", "candidate_size", "metric", "k",
        "contrast_family", "effect_type", "estimate", "ci_low", "ci_high",
    ])
    subgroup = pd.DataFrame(columns=[
        "dataset", "boundary_type", "canonical_subgroup", "model_family",
        "metric", "strategy", "candidate_size", "estimate", "ci_low", "ci_high",
    ])
    result = classification_matrix(frame, rank, contrasts, subgroup)
    assert len(result) == 4
    assert result.iloc[0]["classification"] == "bounded_replication_same_direction"
    assert set(result["classification_category"]) == {
        "bounded_replication", "not_comparable",
    }
    print(json.dumps({"status": "passed", "tests": 13}, indent=2))


def main() -> None:
    args = parse_args()
    if args.self_test:
        synthetic_self_test()
        return
    jth_dir = args.jth_dir.expanduser().resolve()
    cb_dir = args.career_dir.expanduser().resolve()
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse synthesis directory: {output}")
    for source in (jth_dir, cb_dir, benchmark_dir):
        if output == source or output.is_relative_to(source):
            raise EvidenceError("Synthesis output cannot be nested inside a source package")

    # Complete every source validation before creating the output directory.
    sources = {
        "jth": validate_jth(jth_dir),
        "careerbuilder": validate_career(cb_dir),
        "careerbuilder_direct_benchmark": validate_benchmark(benchmark_dir),
    }
    scale = normalize_candidate_scale(jth_dir, cb_dir)
    bias = normalize_bias(jth_dir, cb_dir, scale)
    rank = normalize_rank(jth_dir, cb_dir, scale)
    contrasts = normalize_contrasts(jth_dir, cb_dir)
    subgroup = subgroup_boundary_table(jth_dir, cb_dir)
    compute = compute_table(benchmark_dir, scale)
    classifications = classification_matrix(bias, rank, contrasts, subgroup)
    if classifications.empty:
        raise EvidenceError("No cross-dataset comparable classification cells were produced")

    output.mkdir(parents=True, exist_ok=False)
    tables = {
        "table_1_cross_dataset_candidate_scale.csv": scale,
        "table_2_cross_dataset_metric_bias.csv": bias,
        "table_3_cross_dataset_rank_fidelity.csv": rank,
        "table_4_cross_dataset_bm25_contrasts.csv": contrasts,
        "table_5_cross_dataset_temporal_subgroup_boundary.csv": subgroup,
        "table_6_cross_dataset_compute_boundary.csv": compute,
        "cross_dataset_classification_matrix.csv": classifications,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    stems = make_figures(output, bias, rank, compute)
    (output / "STUDY_PROTOCOL.md").write_text(protocol_markdown(), encoding="utf-8")

    generated = [path for path in output.iterdir() if path.is_file()]
    integrity = {
        "status": "passed",
        "scope": "bounded two-dataset synthesis; raw effectiveness is not pooled",
        "source_packages": sources,
        "analysis_code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "row_counts": {name: int(len(frame)) for name, frame in tables.items()},
        "figures": {
            stem: {ext: (output / f"{stem}.{ext}").stat().st_size for ext in ("png", "pdf", "svg")}
            for stem in stems
        },
        "classification_vocabulary": sorted(classifications["classification"].unique()),
        "classification_categories": sorted(
            classifications["classification_category"].unique()
        ),
        "subgroup_cross_dataset_distortion_available": bool(
            classifications["evidence_domain"].eq("subgroup_distortion").any()
        ),
        "raw_effectiveness_pooled": False,
        "temporal_partitions_directly_compared": False,
        "jth_direct_timing_available": False,
        "generated_sha256": {path.name: sha256(path) for path in generated},
    }
    integrity_path = output / "synthesis_integrity.json"
    integrity_path.write_text(json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8")
    sentinel = {
        "status": "passed",
        "scope": "authoritative cross-dataset synthesis",
        "analysis_code_sha256": sha256(Path(__file__)),
        "integrity_sha256": sha256(integrity_path),
        "output_directory": str(output),
        "required_tables": sorted(tables),
        "figure_stems": stems,
        "interpretation": "Written last; absence means the synthesis is incomplete and non-citable.",
    }
    (output / "SYNTHESIS_COMPLETE.json").write_text(
        json.dumps(sentinel, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "output": str(output), "rows": integrity["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
