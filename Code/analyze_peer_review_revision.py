#!/usr/bin/env python3
"""Post-review reanalysis for the temporal job-recommendation study.

This script reads retained authoritative query-level and aggregate evidence.  It
does not train, encode, or score a recommender.  All outputs are written to a
new directory so the authoritative experiment records remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_REPS = 2000
CB_BOOTSTRAP_SEED = 20261130
JTH_DATE_BOOTSTRAP_SEED = 20261131
JTH_CANDIDATE_BOOTSTRAP_SEED = 20261132
JTH_TWO_WAY_BOOTSTRAP_SEED = 20261133

METRICS = ("hitrate", "ndcg", "map")
CB_METRIC_COLUMNS = {"hitrate": "hit_rate", "ndcg": "ndcg", "map": "map"}
JTH_METRIC_COLUMNS = {"hitrate": "hitrate", "ndcg": "ndcg", "map": "map"}

CB_PRINCIPAL_MODELS = (
    "bm25_full_static",
    "tfidf_combined_static",
    "minilm_title_static",
)
CB_SAMPLER_MODELS = CB_PRINCIPAL_MODELS
JTH_PRINCIPAL_MODELS = ("bm25", "tfidf", "semantic")

CB_MODEL_LABELS = {
    "random_hash": "Deterministic random",
    "popularity_precut": "Pre-cutoff popularity",
    "tfidf_combined_static": "Combined-profile TF-IDF",
    "bm25_full_static": "Full-text BM25",
    "bm25_title_static": "Title BM25",
    "minilm_title_static": "Title MiniLM",
}
JTH_MODEL_LABELS = {
    "random": "Deterministic random",
    "popularity": "Training popularity",
    "tfidf": "JTH TF-IDF",
    "bm25": "JTH BM25",
    "recency_pop": "Recency-popularity",
    "item_knn": "Item-kNN",
    "semantic": "JTH MiniLM",
}

EQUIVALENCE_MARGINS = (
    ("numerical_isclose", None),
    ("absolute_0.001", 0.001),
    ("absolute_0.005", 0.005),
    ("absolute_0.010", 0.010),
)

BIAS_THRESHOLDS = (0.01, 0.02, 0.05)
PAIRWISE_THRESHOLDS = (0.90, 0.95, 1.00)
REGRET_THRESHOLDS = (0.00, 0.01, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results") / "PeerReview_Revision_2026-11-04",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_summary(values: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.std(values, ddof=1)),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values > 0)),
    )


def top_set(scores: pd.Series, margin: float | None) -> tuple[str, ...]:
    best = float(scores.max())
    values = scores.to_numpy(dtype=float)
    if margin is None:
        keep = np.isclose(values, best, rtol=1e-10, atol=1e-12)
    else:
        keep = best - values <= margin + 1e-15
    return tuple(sorted(scores.index.to_numpy()[keep].astype(str)))


def read_cb_sample_rows(checkpoint_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(checkpoint_root.glob("window_*/*.csv.gz"))
    if len(files) != 140:
        raise RuntimeError(f"Expected 140 CareerBuilder checkpoints, found {len(files)}")
    usecols = [
        "window",
        "user_id",
        "population_weight",
        "relevant_jobs",
        "candidate_regime",
        "sampling_strategy",
        "seed",
        "requested_candidate_size",
        "model",
        "k",
        "hit_rate",
        "ndcg",
        "map",
    ]
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, usecols=usecols)
        frame = frame.loc[
            frame["k"].eq(20)
            & frame["candidate_regime"].eq("sampled")
            & frame["model"].isin(CB_SAMPLER_MODELS)
            & frame["requested_candidate_size"].isin((100.0, 10000.0))
        ]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise RuntimeError("No CareerBuilder sampled rows were retained")
    return result, files


def read_jth_sample_rows(paths: list[Path]) -> pd.DataFrame:
    usecols = [
        "query_index",
        "candidate_id",
        "timestamp",
        "group",
        "n_relevant",
        "eligible_catalog",
        "model",
        "strategy",
        "seed",
        "candidate_size",
        "k",
        "hitrate",
        "ndcg",
        "map",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, usecols=usecols)
        # Per-query files partition the complete task into mutually exclusive
        # warm and cold rows; the aggregate files additionally provide an
        # "all" group.  Retain both per-query partitions here to reconstruct
        # the full query population exactly once.
        frame = frame.loc[
            frame["group"].isin(("warm", "cold"))
            & frame["k"].eq(20)
            & frame["model"].isin(JTH_PRINCIPAL_MODELS)
            & frame["strategy"].isin(("uniform", "popularity_weighted"))
            & frame["candidate_size"].isin((100, 5000))
        ]
        if not frame.empty:
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.empty:
        raise RuntimeError("No JTH sampled rows were retained")
    return result


def bootstrap_cb_window_stratified(
    values: np.ndarray,
    windows: np.ndarray,
    weights: np.ndarray,
    reps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    numerator = np.zeros((reps, values.shape[1]), dtype=np.float64)
    denominator = np.zeros(reps, dtype=np.float64)
    for window in np.unique(windows):
        idx = np.flatnonzero(windows == window)
        n = len(idx)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=reps)
        numerator += counts @ (values[idx] * weights[idx, None])
        denominator += counts @ weights[idx]
    return numerator / denominator[:, None]


def bootstrap_one_way_clusters(
    values: np.ndarray,
    clusters: np.ndarray,
    reps: int,
    seed: int,
) -> np.ndarray:
    unique, codes = np.unique(clusters, return_inverse=True)
    cluster_sums = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    np.add.at(cluster_sums, codes, values)
    cluster_counts = np.bincount(codes).astype(np.float64)
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(
        len(unique), np.full(len(unique), 1.0 / len(unique)), size=reps
    )
    numerator = multiplicities @ cluster_sums
    denominator = multiplicities @ cluster_counts
    return numerator / denominator[:, None]


def bootstrap_two_way_clusters(
    values: np.ndarray,
    candidates: np.ndarray,
    dates: np.ndarray,
    reps: int,
    seed: int,
    block_size: int = 100,
) -> np.ndarray:
    candidate_values, candidate_codes = np.unique(candidates, return_inverse=True)
    date_values, date_codes = np.unique(dates, return_inverse=True)
    rng = np.random.default_rng(seed)
    output = np.empty((reps, values.shape[1]), dtype=np.float64)
    for start in range(0, reps, block_size):
        stop = min(reps, start + block_size)
        block = stop - start
        candidate_mult = rng.multinomial(
            len(candidate_values),
            np.full(len(candidate_values), 1.0 / len(candidate_values)),
            size=block,
        )
        date_mult = rng.multinomial(
            len(date_values),
            np.full(len(date_values), 1.0 / len(date_values)),
            size=block,
        )
        query_weights = (
            candidate_mult[:, candidate_codes] * date_mult[:, date_codes]
        ).astype(np.float64, copy=False)
        denominator = query_weights.sum(axis=1)
        if np.any(denominator == 0):
            raise RuntimeError("A two-way bootstrap replicate contained no queries")
        output[start:stop] = (query_weights @ values) / denominator[:, None]
    return output


def build_sampler_contrasts(
    cb_rows: pd.DataFrame,
    jth_rows: pd.DataFrame,
    reps: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    cb_group = (
        cb_rows.groupby(
            [
                "window",
                "user_id",
                "population_weight",
                "model",
                "requested_candidate_size",
                "sampling_strategy",
            ],
            as_index=False,
        )[["hit_rate", "ndcg", "map"]]
        .mean()
    )
    cb_wide = cb_group.pivot(
        index=["window", "user_id", "population_weight"],
        columns=["model", "requested_candidate_size", "sampling_strategy"],
        values=["hit_rate", "ndcg", "map"],
    ).sort_index()
    cb_cells: list[tuple[str, int, str]] = []
    cb_values: list[np.ndarray] = []
    for model in CB_SAMPLER_MODELS:
        for size in (100, 10000):
            for metric, source_metric in CB_METRIC_COLUMNS.items():
                uniform = cb_wide[(source_metric, model, float(size), "uniform")]
                popularity = cb_wide[
                    (source_metric, model, float(size), "popularity_weighted")
                ]
                diff = (uniform - popularity).to_numpy(dtype=np.float64)
                cb_cells.append((model, size, metric))
                cb_values.append(diff)
    cb_matrix = np.column_stack(cb_values)
    cb_index = cb_wide.index.to_frame(index=False)
    if np.isnan(cb_matrix).any():
        raise RuntimeError("CareerBuilder sampler-contrast matrix contains missing values")
    cb_weights = cb_index["population_weight"].to_numpy(dtype=np.float64)
    cb_estimates = np.average(cb_matrix, axis=0, weights=cb_weights)
    cb_boot = bootstrap_cb_window_stratified(
        cb_matrix,
        cb_index["window"].to_numpy(),
        cb_weights,
        reps,
        CB_BOOTSTRAP_SEED,
    )
    for column, (model, size, metric) in enumerate(cb_cells):
        se, low, high, probability = percentile_summary(cb_boot[:, column])
        records.append(
            {
                "dataset": "CareerBuilder",
                "model": CB_MODEL_LABELS[model],
                "metric": metric,
                "k": 20,
                "candidate_size": size,
                "contrast": "uniform bias minus popularity-weighted bias",
                "estimate": cb_estimates[column],
                "bootstrap_se": se,
                "ci_low": low,
                "ci_high": high,
                "probability_gt_zero": probability,
                "bootstrap_method": "paired users resampled within provider window",
                "bootstrap_replicates": reps,
                "bootstrap_seed": CB_BOOTSTRAP_SEED,
                "n_queries": len(cb_index),
                "n_clusters_dimension_1": int(cb_index["window"].nunique()),
                "n_clusters_dimension_2": np.nan,
                "estimand": "design-weighted population mean",
            }
        )

    jth_group = (
        jth_rows.groupby(
            [
                "query_index",
                "candidate_id",
                "timestamp",
                "model",
                "candidate_size",
                "strategy",
            ],
            as_index=False,
        )[["hitrate", "ndcg", "map"]]
        .mean()
    )
    jth_wide = jth_group.pivot(
        index=["query_index", "candidate_id", "timestamp"],
        columns=["model", "candidate_size", "strategy"],
        values=["hitrate", "ndcg", "map"],
    ).sort_index()
    jth_cells: list[tuple[str, int, str]] = []
    jth_values: list[np.ndarray] = []
    for model in JTH_PRINCIPAL_MODELS:
        for size in (100, 5000):
            for metric, source_metric in JTH_METRIC_COLUMNS.items():
                uniform = jth_wide[(source_metric, model, size, "uniform")]
                popularity = jth_wide[
                    (source_metric, model, size, "popularity_weighted")
                ]
                jth_cells.append((model, size, metric))
                jth_values.append((uniform - popularity).to_numpy(dtype=np.float64))
    jth_matrix = np.column_stack(jth_values)
    jth_index = jth_wide.index.to_frame(index=False)
    if np.isnan(jth_matrix).any():
        raise RuntimeError("JTH sampler-contrast matrix contains missing values")
    jth_estimates = jth_matrix.mean(axis=0)
    bootstrap_sets = {
        "date-cluster": bootstrap_one_way_clusters(
            jth_matrix,
            jth_index["timestamp"].astype(str).to_numpy(),
            reps,
            JTH_DATE_BOOTSTRAP_SEED,
        ),
        "candidate-cluster": bootstrap_one_way_clusters(
            jth_matrix,
            jth_index["candidate_id"].to_numpy(),
            reps,
            JTH_CANDIDATE_BOOTSTRAP_SEED,
        ),
        "two-way candidate/date": bootstrap_two_way_clusters(
            jth_matrix,
            jth_index["candidate_id"].to_numpy(),
            jth_index["timestamp"].astype(str).to_numpy(),
            reps,
            JTH_TWO_WAY_BOOTSTRAP_SEED,
        ),
    }
    seeds = {
        "date-cluster": JTH_DATE_BOOTSTRAP_SEED,
        "candidate-cluster": JTH_CANDIDATE_BOOTSTRAP_SEED,
        "two-way candidate/date": JTH_TWO_WAY_BOOTSTRAP_SEED,
    }
    for method, bootstrap_values in bootstrap_sets.items():
        for column, (model, size, metric) in enumerate(jth_cells):
            se, low, high, probability = percentile_summary(bootstrap_values[:, column])
            records.append(
                {
                    "dataset": "JTH",
                    "model": JTH_MODEL_LABELS[model],
                    "metric": metric,
                    "k": 20,
                    "candidate_size": size,
                    "contrast": "uniform bias minus popularity-weighted bias",
                    "estimate": jth_estimates[column],
                    "bootstrap_se": se,
                    "ci_low": low,
                    "ci_high": high,
                    "probability_gt_zero": probability,
                    "bootstrap_method": method,
                    "bootstrap_replicates": reps,
                    "bootstrap_seed": seeds[method],
                    "n_queries": len(jth_index),
                    "n_clusters_dimension_1": (
                        int(jth_index["timestamp"].nunique())
                        if method == "date-cluster"
                        else int(jth_index["candidate_id"].nunique())
                    ),
                    "n_clusters_dimension_2": (
                        int(jth_index["timestamp"].nunique())
                        if method == "two-way candidate/date"
                        else np.nan
                    ),
                    "estimand": "query macro-mean",
                }
            )
    return pd.DataFrame.from_records(records)


def load_top_equivalence_aggregates(
    cb_analysis: Path,
    jth_summary_paths: list[Path],
) -> pd.DataFrame:
    cb = pd.read_csv(cb_analysis / "pooled_estimates.csv")
    cb = cb.loc[cb["estimand"].eq("design_weighted_population") & cb["k"].eq(20)]
    cb_records: list[pd.DataFrame] = []
    for metric, source_metric in CB_METRIC_COLUMNS.items():
        part = cb[
            [
                "candidate_regime",
                "sampling_strategy",
                "seed",
                "requested_candidate_size",
                "model",
                source_metric,
            ]
        ].copy()
        part["dataset"] = "CareerBuilder"
        part["metric"] = metric
        part["value"] = part[source_metric].astype(float)
        part["protocol"] = np.where(
            part["candidate_regime"].eq("full_catalog"), "full", "sampled"
        )
        part["strategy"] = np.where(
            part["protocol"].eq("full"), "full", part["sampling_strategy"]
        )
        part["candidate_size"] = part["requested_candidate_size"].fillna(-1).astype(int)
        part["model_label"] = part["model"].map(CB_MODEL_LABELS)
        cb_records.append(
            part[
                [
                    "dataset",
                    "metric",
                    "protocol",
                    "strategy",
                    "seed",
                    "candidate_size",
                    "model_label",
                    "value",
                ]
            ]
        )

    jth_frames = [pd.read_csv(path) for path in jth_summary_paths]
    jth = pd.concat(jth_frames, ignore_index=True)
    jth = jth.loc[jth["group"].eq("all") & jth["k"].eq(20)]
    jth_records: list[pd.DataFrame] = []
    for metric in METRICS:
        part = jth[
            ["model", "strategy", "seed", "candidate_size", metric]
        ].copy()
        part["dataset"] = "JTH"
        part["metric"] = metric
        part["value"] = part[metric].astype(float)
        part["protocol"] = np.where(part["strategy"].eq("full"), "full", "sampled")
        part["candidate_size"] = part["candidate_size"].astype(int)
        part["model_label"] = part["model"].map(JTH_MODEL_LABELS)
        jth_records.append(
            part[
                [
                    "dataset",
                    "metric",
                    "protocol",
                    "strategy",
                    "seed",
                    "candidate_size",
                    "model_label",
                    "value",
                ]
            ]
        )
    return pd.concat(cb_records + jth_records, ignore_index=True)


def build_top_equivalence(
    aggregates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail: list[dict[str, object]] = []
    for (dataset, metric), cell in aggregates.groupby(["dataset", "metric"]):
        reference = (
            cell.loc[cell["protocol"].eq("full")]
            .drop_duplicates("model_label")
            .set_index("model_label")["value"]
            .sort_index()
        )
        sampled = cell.loc[cell["protocol"].eq("sampled")]
        for (strategy, size, seed), condition in sampled.groupby(
            ["strategy", "candidate_size", "seed"]
        ):
            scores = condition.set_index("model_label")["value"].reindex(reference.index)
            if scores.isna().any():
                raise RuntimeError(f"Incomplete top-set scores for {dataset}, {metric}")
            for margin_label, margin in EQUIVALENCE_MARGINS:
                reference_top = top_set(reference, margin)
                sampled_top = top_set(scores, margin)
                intersection = set(reference_top).intersection(sampled_top)
                union = set(reference_top).union(sampled_top)
                reference_best = float(reference.max())
                selected_reference_scores = reference.loc[list(sampled_top)]
                detail.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "k": 20,
                        "strategy": strategy,
                        "candidate_size": int(size),
                        "seed": int(seed),
                        "equivalence_rule": margin_label,
                        "absolute_margin": np.nan if margin is None else margin,
                        "reference_top_set": " | ".join(reference_top),
                        "sampled_top_set": " | ".join(sampled_top),
                        "reference_top_set_size": len(reference_top),
                        "sampled_top_set_size": len(sampled_top),
                        "top_set_exact_agreement": reference_top == sampled_top,
                        "top_set_overlap": bool(intersection),
                        "top_set_jaccard": len(intersection) / len(union),
                        "reference_regret_best_case": reference_best
                        - float(selected_reference_scores.max()),
                        "reference_regret_worst_case": reference_best
                        - float(selected_reference_scores.min()),
                    }
                )
    detail_frame = pd.DataFrame.from_records(detail)
    summary = (
        detail_frame.groupby(
            [
                "dataset",
                "metric",
                "k",
                "strategy",
                "candidate_size",
                "equivalence_rule",
                "absolute_margin",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            seed_decisions=("seed", "size"),
            top_set_exact_agreement_rate=("top_set_exact_agreement", "mean"),
            top_set_overlap_rate=("top_set_overlap", "mean"),
            top_set_jaccard_mean=("top_set_jaccard", "mean"),
            reference_regret_best_case_mean=("reference_regret_best_case", "mean"),
            reference_regret_worst_case_mean=("reference_regret_worst_case", "mean"),
            reference_regret_worst_case_max=("reference_regret_worst_case", "max"),
        )
    )
    seed_detail = detail_frame.loc[
        detail_frame["equivalence_rule"].eq("numerical_isclose")
    ].copy()
    seed_stability = (
        seed_detail.groupby(
            ["dataset", "metric", "k", "strategy", "candidate_size"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "size"),
            exact_top_set_agreement_count=("top_set_exact_agreement", "sum"),
            exact_top_set_agreement_rate=("top_set_exact_agreement", "mean"),
            overlap_count=("top_set_overlap", "sum"),
            overlap_rate=("top_set_overlap", "mean"),
            worst_case_regret_mean=("reference_regret_worst_case", "mean"),
            worst_case_regret_max=("reference_regret_worst_case", "max"),
        )
    )
    return detail_frame, summary, seed_detail, seed_stability


def build_decision_calibration(
    cb_analysis: Path,
    jth_analysis: Path,
    jth_summary_paths: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cb_rank = pd.read_csv(cb_analysis / "rank_fidelity_summary.csv")
    cb_rank = cb_rank.loc[
        cb_rank["rank_scope"].eq("pooled")
        & cb_rank["estimand"].eq("design_weighted_population")
        & cb_rank["k"].eq(20)
        & cb_rank["metric"].isin(("hit_rate", "ndcg", "map"))
    ].copy()
    cb_rank["metric"] = cb_rank["metric"].replace({"hit_rate": "hitrate"})
    cb_cal = cb_rank[
        [
            "sampling_strategy",
            "requested_candidate_size",
            "metric",
            "winner_set_exact_match_rate",
            "pairwise_order_agreement_mean",
            "sampled_winner_regret_worst_max",
        ]
    ].rename(
        columns={
            "sampling_strategy": "strategy",
            "requested_candidate_size": "candidate_size",
            "winner_set_exact_match_rate": "winner_exact_rate",
            "sampled_winner_regret_worst_max": "worst_case_regret_max",
        }
    )
    cb_cal["dataset"] = "CareerBuilder"
    cb_bias = pd.read_csv(cb_analysis / "sampling_bias_seed_summary.csv")
    cb_bias = cb_bias.loc[
        cb_bias["estimand"].eq("design_weighted_population")
        & cb_bias["k"].eq(20)
        & cb_bias["model"].isin(CB_PRINCIPAL_MODELS)
        & cb_bias["metric"].isin(("hit_rate", "ndcg", "map"))
    ].copy()
    cb_bias["metric"] = cb_bias["metric"].replace({"hit_rate": "hitrate"})
    cb_bias = (
        cb_bias.groupby(
            ["sampling_strategy", "requested_candidate_size", "metric"],
            as_index=False,
        )["absolute_mean_bias"]
        .max()
        .rename(
            columns={
                "sampling_strategy": "strategy",
                "requested_candidate_size": "candidate_size",
                "absolute_mean_bias": "max_principal_absolute_bias",
            }
        )
    )
    cb_scale = pd.read_csv(cb_analysis / "candidate_scale_and_realized_fraction.csv")
    cb_scale = cb_scale.loc[cb_scale["sampling_strategy"].isin(("uniform", "popularity_weighted"))]
    cb_scale = cb_scale[
        ["sampling_strategy", "requested_candidate_size", "actual_candidate_fraction_mean"]
    ].rename(
        columns={
            "sampling_strategy": "strategy",
            "requested_candidate_size": "candidate_size",
            "actual_candidate_fraction_mean": "candidate_fraction_mean",
        }
    )
    cb_cal = cb_cal.merge(cb_bias, on=["strategy", "candidate_size", "metric"], validate="one_to_one")
    cb_cal = cb_cal.merge(cb_scale, on=["strategy", "candidate_size"], validate="many_to_one")

    jth_rank = pd.read_csv(jth_analysis / "rank_fidelity_metric_summary_k20.csv")
    jth_rank = jth_rank.loc[jth_rank["metric"].isin(METRICS)].copy()
    jth_cal = jth_rank[
        [
            "strategy",
            "candidate_size",
            "metric",
            "winner_set_exact_agreement",
            "pairwise_order_agreement",
        ]
    ].rename(
        columns={
            "winner_set_exact_agreement": "winner_exact_rate",
            "pairwise_order_agreement": "pairwise_order_agreement_mean",
        }
    )
    # The metric summary stores a seed mean.  Calibration instead requires the
    # maximum observed regret across the five retained seed decisions.
    jth_rank_detail = pd.read_csv(jth_analysis / "rank_fidelity_explicit_ties.csv")
    jth_regret = (
        jth_rank_detail.loc[
            jth_rank_detail["group"].eq("all")
            & jth_rank_detail["k"].eq(20)
            & jth_rank_detail["metric"].isin(METRICS)
        ]
        .groupby(["strategy", "candidate_size", "metric"], as_index=False)[
            "winner_regret_worst_case"
        ]
        .max()
        .rename(columns={"winner_regret_worst_case": "worst_case_regret_max"})
    )
    jth_cal = jth_cal.merge(
        jth_regret,
        on=["strategy", "candidate_size", "metric"],
        validate="one_to_one",
    )
    jth_cal["dataset"] = "JTH"
    # The strengthened-v2 bias table intentionally reports HitRate and MAP.
    # Recover the same seed-mean bias estimand for NDCG (and the other two
    # metrics for consistency) from the retained authoritative aggregate rows.
    jth_summary = pd.concat(
        [pd.read_csv(path) for path in jth_summary_paths], ignore_index=True
    )
    jth_summary = jth_summary.loc[
        jth_summary["group"].eq("all")
        & jth_summary["k"].eq(20)
        & jth_summary["model"].isin(JTH_PRINCIPAL_MODELS)
    ].copy()
    jth_bias_parts: list[pd.DataFrame] = []
    for metric in METRICS:
        metric_rows = jth_summary[
            ["model", "strategy", "seed", "candidate_size", metric]
        ].copy()
        full_values = (
            metric_rows.loc[metric_rows["strategy"].eq("full"), ["model", metric]]
            .drop_duplicates("model")
            .rename(columns={metric: "full_value"})
        )
        sampled_values = metric_rows.loc[~metric_rows["strategy"].eq("full")].merge(
            full_values, on="model", validate="many_to_one"
        )
        sampled_values["signed_bias"] = (
            sampled_values[metric] - sampled_values["full_value"]
        )
        model_bias = (
            sampled_values.groupby(
                ["strategy", "candidate_size", "model"], as_index=False
            )["signed_bias"]
            .mean()
        )
        model_bias["absolute_mean_bias"] = model_bias["signed_bias"].abs()
        model_bias["metric"] = metric
        jth_bias_parts.append(model_bias)
    jth_bias = (
        pd.concat(jth_bias_parts, ignore_index=True)
        .groupby(["strategy", "candidate_size", "metric"], as_index=False)[
            "absolute_mean_bias"
        ]
        .max()
        .rename(
            columns={"absolute_mean_bias": "max_principal_absolute_bias"}
        )
    )
    jth_scale = pd.read_csv(jth_analysis / "catalog_size_and_candidate_fraction.csv")
    jth_scale = jth_scale.loc[jth_scale["candidate_protocol"].eq("sampled")]
    jth_scale = jth_scale[
        ["strategy", "nominal_candidate_size", "realized_candidate_fraction_mean"]
    ].rename(
        columns={
            "nominal_candidate_size": "candidate_size",
            "realized_candidate_fraction_mean": "candidate_fraction_mean",
        }
    )
    jth_cal = jth_cal.merge(jth_bias, on=["strategy", "candidate_size", "metric"], validate="one_to_one")
    jth_cal = jth_cal.merge(jth_scale, on=["strategy", "candidate_size"], validate="many_to_one")

    calibration = pd.concat([cb_cal, jth_cal], ignore_index=True)
    calibration["candidate_size"] = calibration["candidate_size"].astype(int)
    calibration["winner_all_five_seeds"] = np.isclose(calibration["winner_exact_rate"], 1.0)
    calibration["pairwise_agreement_at_least_0_90"] = calibration[
        "pairwise_order_agreement_mean"
    ].ge(0.90)
    calibration["worst_case_regret_at_most_0_01"] = calibration[
        "worst_case_regret_max"
    ].le(0.01 + 1e-15)
    calibration["max_principal_bias_at_most_0_02"] = calibration[
        "max_principal_absolute_bias"
    ].le(0.02 + 1e-15)
    calibration["selection_focused_illustrative_pass"] = (
        calibration["winner_all_five_seeds"]
        & calibration["pairwise_agreement_at_least_0_90"]
        & calibration["worst_case_regret_at_most_0_01"]
    )
    calibration["metric_and_selection_illustrative_pass"] = (
        calibration["selection_focused_illustrative_pass"]
        & calibration["max_principal_bias_at_most_0_02"]
    )
    calibration["analysis_status"] = "post-review illustrative calibration; not preregistered"

    sensitivity_records: list[dict[str, object]] = []
    for row in calibration.itertuples(index=False):
        for bias_threshold in BIAS_THRESHOLDS:
            for pairwise_threshold in PAIRWISE_THRESHOLDS:
                for regret_threshold in REGRET_THRESHOLDS:
                    sensitivity_records.append(
                        {
                            "dataset": row.dataset,
                            "strategy": row.strategy,
                            "candidate_size": row.candidate_size,
                            "candidate_fraction_mean": row.candidate_fraction_mean,
                            "metric": row.metric,
                            "bias_threshold": bias_threshold,
                            "pairwise_agreement_threshold": pairwise_threshold,
                            "regret_threshold": regret_threshold,
                            "winner_all_five_seeds": row.winner_all_five_seeds,
                            "passes_bias": row.max_principal_absolute_bias <= bias_threshold + 1e-15,
                            "passes_pairwise": row.pairwise_order_agreement_mean >= pairwise_threshold - 1e-15,
                            "passes_regret": row.worst_case_regret_max <= regret_threshold + 1e-15,
                            "passes_all": (
                                row.winner_all_five_seeds
                                and row.max_principal_absolute_bias <= bias_threshold + 1e-15
                                and row.pairwise_order_agreement_mean >= pairwise_threshold - 1e-15
                                and row.worst_case_regret_max <= regret_threshold + 1e-15
                            ),
                        }
                    )
    sensitivity = pd.DataFrame.from_records(sensitivity_records)

    criteria = {
        "winner_all_five_seeds": "winner_all_five_seeds",
        "pairwise_agreement_at_least_0.90": "pairwise_agreement_at_least_0_90",
        "worst_case_regret_at_most_0.01": "worst_case_regret_at_most_0_01",
        "max_principal_bias_at_most_0.02": "max_principal_bias_at_most_0_02",
        "selection_focused_illustrative": "selection_focused_illustrative_pass",
        "metric_and_selection_illustrative": "metric_and_selection_illustrative_pass",
    }
    minimum_records: list[dict[str, object]] = []
    for (dataset, strategy, metric), group in calibration.groupby(
        ["dataset", "strategy", "metric"]
    ):
        group = group.sort_values("candidate_size")
        for criterion_label, column in criteria.items():
            passing = group.loc[group[column]]
            selected = passing.iloc[0] if not passing.empty else None
            minimum_records.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "metric": metric,
                    "criterion": criterion_label,
                    "minimum_tested_candidate_size": (
                        int(selected["candidate_size"]) if selected is not None else np.nan
                    ),
                    "candidate_fraction_mean": (
                        float(selected["candidate_fraction_mean"])
                        if selected is not None
                        else np.nan
                    ),
                    "status": "reached" if selected is not None else "not reached",
                    "analysis_status": "post-review illustrative calibration; not preregistered",
                }
            )
    minimums = pd.DataFrame.from_records(minimum_records)
    return calibration, sensitivity, minimums


def build_timing_summary(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.loc[data["phase"].eq("primary_warm_cache")].copy()
    full = data.loc[data["candidate_regime"].eq("full_catalog")][
        ["window", "repeat", "model", "total_seconds", "scored_pairs"]
    ].rename(
        columns={
            "total_seconds": "full_total_seconds",
            "scored_pairs": "full_scored_pairs",
        }
    )
    data = data.merge(full, on=["window", "repeat", "model"], how="left", validate="many_to_one")
    data["paired_time_ratio"] = data["total_seconds"] / data["full_total_seconds"]
    data["paired_pair_ratio"] = data["scored_pairs"] / data["full_scored_pairs"]
    records: list[dict[str, object]] = []
    group_columns = ["model", "candidate_regime", "requested_candidate_size"]
    for keys, group in data.groupby(group_columns, dropna=False):
        model, regime, size = keys
        window_medians = group.groupby("window")["paired_time_ratio"].median()
        records.append(
            {
                "model": CB_MODEL_LABELS.get(model, model),
                "candidate_regime": regime,
                "candidate_size": -1 if pd.isna(size) else int(size),
                "timed_observations": len(group),
                "windows": int(group["window"].nunique()),
                "repetitions_per_window": int(group["repeat"].nunique()),
                "total_seconds_mean": float(group["total_seconds"].mean()),
                "total_seconds_sd": float(group["total_seconds"].std(ddof=1)),
                "total_seconds_median": float(group["total_seconds"].median()),
                "total_seconds_q1": float(group["total_seconds"].quantile(0.25)),
                "total_seconds_q3": float(group["total_seconds"].quantile(0.75)),
                "paired_time_ratio_mean": float(group["paired_time_ratio"].mean()),
                "paired_time_ratio_median": float(group["paired_time_ratio"].median()),
                "paired_time_ratio_q1": float(group["paired_time_ratio"].quantile(0.25)),
                "paired_time_ratio_q3": float(group["paired_time_ratio"].quantile(0.75)),
                "window_median_time_ratio_min": float(window_medians.min()),
                "window_median_time_ratio_max": float(window_medians.max()),
                "paired_pair_ratio_mean": float(group["paired_pair_ratio"].mean()),
                "timed_boundary": "scoring, deterministic ranking, and metric calculation",
            }
        )
    return pd.DataFrame.from_records(records)


def describe_relevance(values: pd.Series, dataset: str) -> dict[str, object]:
    numeric = values.astype(float)
    return {
        "dataset": dataset,
        "queries": int(len(numeric)),
        "mean_relevant_jobs": float(numeric.mean()),
        "sd_relevant_jobs": float(numeric.std(ddof=1)),
        "minimum": int(numeric.min()),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "p75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "maximum": int(numeric.max()),
        "queries_with_multiple_relevant_jobs": int(numeric.gt(1).sum()),
        "fraction_with_multiple_relevant_jobs": float(numeric.gt(1).mean()),
    }


def build_relevance_distribution(
    cb_rows: pd.DataFrame,
    jth_primary_path: Path,
) -> pd.DataFrame:
    cb_queries = cb_rows[["window", "user_id", "relevant_jobs"]].drop_duplicates()
    if cb_queries.duplicated(["window", "user_id"]).any():
        raise RuntimeError("CareerBuilder relevance count is not unique per query")
    jth = pd.read_csv(jth_primary_path, usecols=["query_index", "n_relevant"])
    jth_queries = jth.drop_duplicates()
    if jth_queries.duplicated("query_index").any():
        raise RuntimeError("JTH relevance count is not unique per query")
    return pd.DataFrame.from_records(
        [
            describe_relevance(cb_queries["relevant_jobs"], "CareerBuilder"),
            describe_relevance(jth_queries["n_relevant"], "JTH"),
        ]
    )


def build_matched_fraction(calibration: pd.DataFrame) -> pd.DataFrame:
    cb = calibration.loc[
        calibration["dataset"].eq("CareerBuilder")
        & calibration["candidate_size"].eq(10000)
    ].copy()
    jth = calibration.loc[
        calibration["dataset"].eq("JTH") & calibration["candidate_size"].eq(500)
    ].copy()
    keys = ["strategy", "metric"]
    merged = cb.merge(jth, on=keys, suffixes=("_careerbuilder", "_jth"), validate="one_to_one")
    output = merged[
        [
            "strategy",
            "metric",
            "candidate_fraction_mean_careerbuilder",
            "candidate_fraction_mean_jth",
            "max_principal_absolute_bias_careerbuilder",
            "max_principal_absolute_bias_jth",
            "pairwise_order_agreement_mean_careerbuilder",
            "pairwise_order_agreement_mean_jth",
            "winner_exact_rate_careerbuilder",
            "winner_exact_rate_jth",
            "worst_case_regret_max_careerbuilder",
            "worst_case_regret_max_jth",
        ]
    ].copy()
    output.insert(2, "careerbuilder_candidate_size", 10000)
    output.insert(3, "jth_candidate_size", 500)
    output["absolute_fraction_difference"] = (
        output["candidate_fraction_mean_careerbuilder"]
        - output["candidate_fraction_mean_jth"]
    ).abs()
    output["interpretation"] = (
        "descriptive near-matched-fraction comparison; dataset and task differences remain"
    )
    return output


def protocol_text(reps: int) -> str:
    return f"""# Post-review calibration and uncertainty analysis protocol

## Status

These analyses were added after independent manuscript review. They were not
preregistered and are reported as sensitivity or worked-calibration analyses.
They do not replace the prespecified primary contrasts.

## Evidence boundary

The script reads only the completed CareerBuilder First7Last2 authoritative-v2
checkpoints, their admitted analysis package, the completed CareerBuilder
direct-scoring benchmark, the three hash-pinned JTH per-query result files, and
the JTH strengthened-v2 analysis package. It performs no model training, text
encoding, candidate scoring, or resampling of new evaluation candidates.

## Analyses

1. Top-model sets are recomputed under the primary numerical tie rule and
   illustrative absolute equivalence margins of 0.001, 0.005, and 0.010.
2. Uniform-minus-popularity-weighted paired metric differences are calculated
   for the three principal content models after averaging five seeds within
   query. CareerBuilder uses {reps}
   user-bootstrap replicates stratified by provider window and retains design
   weights. JTH reports date-cluster, candidate-cluster, and two-way
   candidate/date bootstrap sensitivities with {reps} replicates each. Because
   both sampled protocols share the same full-catalog reference, their sampled
   metric difference is algebraically identical to the difference between
   their two sampled-minus-full biases.
3. Seed-level top-model decisions are reported descriptively. Five seeds do not
   support a precise binomial probability claim.
4. Illustrative decision calibration reports each criterion separately and a
   transparent combined rule. The worked rule requires top-model agreement in
   all five seeds, mean pairwise agreement at least 0.90, maximum reference
   regret at most 0.01, and, for the metric-plus-decision variant, maximum
   absolute bias across the three principal content configurations at most
   0.02. A threshold grid is released; no threshold is claimed as universal.
5. Timing summaries use the 630 primary warm-cache records. Sampled/full ratios
   are paired within model, provider window, and timed repetition.
6. Relevance-set sizes are deduplicated to one record per query.
7. CareerBuilder size 10,000 and JTH size 500 are compared descriptively because
   their mean realized catalog fractions are both approximately 8.5%. This
   single near-match does not isolate candidate fraction from task differences.

## Interpretation

The outputs quantify robustness of the reported decisions and make the
calibration recommendation operational. They must not be described as
preregistered criteria, universal operating points, or a causal decomposition
of cross-dataset differences.
"""


def main() -> None:
    analysis_started = time.perf_counter()
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cb_run = root / "Results" / "CareerBuilder_InternalTemporal_EvaluationValidity" / "Primary_First7Last2_Authoritative_v2"
    cb_analysis = root / "Results" / "CareerBuilder_InternalTemporal_Authoritative_Analysis_v1" / "First7Last2"
    cb_benchmark = root / "Results" / "CareerBuilder_InternalTemporal_EvaluationValidity" / "RQ5_DirectScoring_Benchmark_Authoritative_v1" / "benchmark_results.csv"
    jth_raw_paths = [
        root / "Results" / "JTH_EvaluationValidity" / "per_query_metrics.csv.gz",
        root / "Results" / "JTH_EvaluationValidity_Expanded" / "per_query_metrics.csv.gz",
        root / "Results" / "JTH_EvaluationValidity_Semantic" / "per_query_metrics.csv.gz",
    ]
    jth_summary_paths = [path.with_name("summary_metrics.csv") for path in jth_raw_paths]
    jth_analysis = root / "Results" / "JTH_EvaluationValidity_Strengthened_v2"
    required = [cb_run, cb_analysis, cb_benchmark, jth_analysis, *jth_raw_paths, *jth_summary_paths]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing authoritative inputs: {missing}")

    cb_rows, cb_checkpoint_files = read_cb_sample_rows(cb_run / "checkpoints")
    jth_rows = read_jth_sample_rows(jth_raw_paths)

    sampler = build_sampler_contrasts(cb_rows, jth_rows, args.bootstrap_reps)
    aggregates = load_top_equivalence_aggregates(cb_analysis, jth_summary_paths)
    top_detail, top_summary, seed_detail, seed_stability = build_top_equivalence(aggregates)
    calibration, threshold_sensitivity, calibration_minimums = build_decision_calibration(
        cb_analysis, jth_analysis, jth_summary_paths
    )
    timing = build_timing_summary(cb_benchmark)
    relevance = build_relevance_distribution(cb_rows, jth_raw_paths[0])
    matched = build_matched_fraction(calibration)

    outputs: dict[str, pd.DataFrame] = {
        "top_set_equivalence_sensitivity_detail.csv": top_detail,
        "top_set_equivalence_sensitivity_summary.csv": top_summary,
        "sampler_paired_contrasts.csv": sampler,
        "seed_decision_detail.csv": seed_detail,
        "seed_decision_stability.csv": seed_stability,
        "decision_calibration.csv": calibration,
        "decision_calibration_threshold_sensitivity.csv": threshold_sensitivity,
        "decision_calibration_minimums.csv": calibration_minimums,
        "timing_robust_summary.csv": timing,
        "relevance_set_distribution.csv": relevance,
        "matched_fraction_descriptive.csv": matched,
    }
    for name, frame in outputs.items():
        if frame.empty:
            raise RuntimeError(f"Refusing to write empty output {name}")
        frame.to_csv(output_dir / name, index=False)

    protocol_path = output_dir / "STUDY_PROTOCOL.md"
    protocol_path.write_text(protocol_text(args.bootstrap_reps), encoding="utf-8")

    if int(relevance.loc[relevance["dataset"].eq("CareerBuilder"), "queries"].iloc[0]) != 7000:
        raise RuntimeError("CareerBuilder relevance distribution did not contain 7,000 queries")
    if int(relevance.loc[relevance["dataset"].eq("JTH"), "queries"].iloc[0]) != 8341:
        raise RuntimeError("JTH relevance distribution did not contain 8,341 queries")
    if len(sampler) != 72:
        raise RuntimeError(f"Expected 72 sampler contrast rows, found {len(sampler)}")
    if len(top_detail) != 1200:
        raise RuntimeError(f"Expected 1,200 equivalence-detail rows, found {len(top_detail)}")
    if len(seed_detail) != 300:
        raise RuntimeError(f"Expected 300 seed-decision rows, found {len(seed_detail)}")
    if len(calibration) != 60:
        raise RuntimeError(f"Expected 60 calibration rows, found {len(calibration)}")
    if len(matched) != 6:
        raise RuntimeError(f"Expected 6 matched-fraction rows, found {len(matched)}")
    if float(matched["absolute_fraction_difference"].max()) >= 0.002:
        raise RuntimeError("Matched-fraction conditions are not sufficiently close")
    required_numeric_blocks = {
        "sampler": sampler[["estimate", "bootstrap_se", "ci_low", "ci_high"]],
        "top_equivalence": top_detail[
            [
                "top_set_jaccard",
                "reference_regret_best_case",
                "reference_regret_worst_case",
            ]
        ],
        "calibration": calibration[
            [
                "winner_exact_rate",
                "pairwise_order_agreement_mean",
                "worst_case_regret_max",
                "max_principal_absolute_bias",
                "candidate_fraction_mean",
            ]
        ],
        "timing": timing[
            [
                "total_seconds_median",
                "total_seconds_q1",
                "total_seconds_q3",
                "paired_time_ratio_median",
            ]
        ],
        "relevance": relevance[
            ["queries", "mean_relevant_jobs", "median", "maximum"]
        ],
    }
    all_key_numerics_finite = all(
        np.isfinite(frame.to_numpy(dtype=float)).all()
        for frame in required_numeric_blocks.values()
    )
    if not all_key_numerics_finite:
        raise RuntimeError("A required output field contains a missing or non-finite value")

    source_paths = [
        root / "Code" / "analyze_peer_review_revision.py",
        cb_analysis / "pooled_estimates.csv",
        cb_analysis / "rank_fidelity_summary.csv",
        cb_analysis / "sampling_bias_seed_summary.csv",
        cb_analysis / "candidate_scale_and_realized_fraction.csv",
        cb_benchmark,
        jth_analysis / "rank_fidelity_metric_summary_k20.csv",
        jth_analysis / "rank_fidelity_explicit_ties.csv",
        jth_analysis / "catalog_size_and_candidate_fraction.csv",
        *jth_raw_paths,
        *jth_summary_paths,
        *cb_checkpoint_files,
    ]
    source_hashes = {
        str(path.relative_to(root)): sha256(path) for path in source_paths
    }
    output_paths = [output_dir / name for name in outputs] + [protocol_path]
    output_hashes = {
        path.name: sha256(path) for path in output_paths
    }
    manifest = {
        "status": "passed",
        "analysis_status": "post-review sensitivity and worked calibration; not preregistered",
        "no_model_scoring_or_encoding": True,
        "authoritative_sources_only": True,
        "bootstrap_replicates": args.bootstrap_reps,
        "bootstrap_seeds": {
            "careerbuilder_window_stratified": CB_BOOTSTRAP_SEED,
            "jth_date_cluster": JTH_DATE_BOOTSTRAP_SEED,
            "jth_candidate_cluster": JTH_CANDIDATE_BOOTSTRAP_SEED,
            "jth_two_way_candidate_date": JTH_TWO_WAY_BOOTSTRAP_SEED,
        },
        "equivalence_margins": [label for label, _ in EQUIVALENCE_MARGINS],
        "illustrative_calibration": {
            "winner_exact_all_five_seeds": True,
            "pairwise_agreement_minimum": 0.90,
            "worst_case_regret_maximum": 0.01,
            "max_principal_absolute_bias": 0.02,
            "threshold_grid": {
                "bias": list(BIAS_THRESHOLDS),
                "pairwise": list(PAIRWISE_THRESHOLDS),
                "regret": list(REGRET_THRESHOLDS),
            },
        },
        "row_counts": {name: int(len(frame)) for name, frame in outputs.items()},
        "query_counts": {"CareerBuilder": 7000, "JTH": 8341},
        "source_files": source_hashes,
        "output_files": output_hashes,
        "runtime": {
            "analysis_elapsed_seconds": time.perf_counter() - analysis_started,
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "integrity_checks": {
            "careerbuilder_checkpoint_files": len(cb_checkpoint_files),
            "sampler_contrast_rows_expected_72": len(sampler) == 72,
            "top_equivalence_rows_expected_1200": len(top_detail) == 1200,
            "seed_decision_rows_expected_300": len(seed_detail) == 300,
            "calibration_rows_expected_60": len(calibration) == 60,
            "matched_fraction_rows_expected_6": len(matched) == 6,
            "matched_fraction_max_absolute_difference_below_0.002": float(
                matched["absolute_fraction_difference"].max()
            )
            < 0.002,
            "all_outputs_nonempty": all(not frame.empty for frame in outputs.values()),
            "all_required_numeric_fields_finite": all_key_numerics_finite,
            "source_hashes_recorded": len(source_hashes),
            "output_hashes_recorded": len(output_hashes),
        },
    }
    manifest_path = output_dir / "POST_REVIEW_ANALYSIS_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output_dir": str(output_dir), "row_counts": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
