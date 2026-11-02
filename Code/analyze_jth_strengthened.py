#!/usr/bin/env python3
"""Strengthened, evaluation-validity analysis for the JTH experiment.

This script is intentionally independent of the invalidated CareerBuilder branch.
It performs four linked audits:

1. exact reproduction of the retained JTH K=20 full/100-candidate rows;
2. date-, candidate-, and two-way candidate/date-cluster bootstrap sensitivity;
3. known-item/repeat-target sensitivity; and
4. rank fidelity with explicit tau-b, tie, winner-set, pairwise-order, and
   regret rules.

The verified findings file is written only after the reproduction gate and all
integrity checks pass.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import kendalltau


ROOT = Path("Results")
INPUT_ROOTS = [
    ROOT / "JTH_EvaluationValidity",
    ROOT / "JTH_EvaluationValidity_Expanded",
    ROOT / "JTH_EvaluationValidity_Semantic",
]
SEVEN_MODEL = ROOT / "JTH_EvaluationValidity_SevenModel"
SOURCE_OUT = ROOT / "JTH_EvaluationValidity_Strengthened"
OUT = ROOT / "JTH_EvaluationValidity_Strengthened_v2"
DATA = Path("Dataset/external/JTH/data/binary")

EXPECTED_KNOWN_ITEM_SHA256 = (
    "0c2dffa21d695edd9982fb31599f23ba7681a3ef1f7cbb0b2509ea5a57806601"
)
EXPECTED_KNOWN_ITEM_AUDIT_SHA256 = (
    "8e1195bce4ce42dcfa27503843a30cbdff5598333587eda3eccf610c035c59d8"
)

MODELS = [
    "bm25",
    "item_knn",
    "popularity",
    "random",
    "recency_pop",
    "semantic",
    "tfidf",
]
DISPLAY = {
    "bm25": "BM25",
    "item_knn": "Item-kNN",
    "popularity": "Popularity",
    "random": "Random",
    "recency_pop": "Recency-popularity",
    "semantic": "MiniLM",
    "tfidf": "TF-IDF",
}
COLORS = {
    "bm25": "#0072B2",
    "semantic": "#D55E00",
    "tfidf": "#009E73",
    "popularity": "#CC79A7",
    "random": "#777777",
    "recency_pop": "#E69F00",
    "item_knn": "#56B4E9",
}
METRICS = ["precision", "recall", "hitrate", "ndcg", "map", "mrr"]
PRIMARY_METRICS = ["hitrate", "ndcg", "map"]
CHALLENGERS = ["semantic", "tfidf"]
BOOTSTRAPS = 2_000
BOOTSTRAP_SEEDS = {
    "date_cluster": 20261020,
    "candidate_cluster": 20261021,
    "two_way_candidate_date": 20261022,
}
SUBGROUP_BOOTSTRAP_SEEDS = {
    "date_cluster": 20261120,
    "candidate_cluster": 20261121,
    "two_way_candidate_date": 20261122,
}
BIAS_BOOTSTRAP_SEEDS = {
    "date_cluster": 20261220,
    "candidate_cluster": 20261221,
    "two_way_candidate_date": 20261222,
}
BIAS_METRICS = ["hitrate", "map"]
BIAS_SIZES = [100, 500, 1000, 2500, 5000]
TIE_RTOL = 1e-10
TIE_ATOL = 1e-12
PARITY_ATOL = 1e-12
EXPECTED_K20_ROWS = 2_977_737
EXPECTED_SELECTED_ROWS = 642_257
EXPECTED_ORIGINAL_ROWS = 642_257
EXPECTED_NOVEL_ROWS = 640_871
EXPECTED_QUERIES = 8_341


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2), encoding="utf-8")


def initialize_clean_output() -> tuple[Path, Path, dict]:
    """Create a fresh v2 package from two hash-pinned immutable inputs.

    The former strengthened directory is quarantined after an interrupted
    extension run and is never used as an output target.  Only the two retained
    producer artifacts required by this analyzer are copied from it, and both
    must match hashes recorded before the interruption.
    """
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a new run with existing v2 artifacts: {OUT}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    source_known = SOURCE_OUT / "known_item_per_query_k20.csv.gz"
    source_audit = SOURCE_OUT / "known_item_sensitivity_audit.json"
    expected = {
        source_known: EXPECTED_KNOWN_ITEM_SHA256,
        source_audit: EXPECTED_KNOWN_ITEM_AUDIT_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing immutable JTH producer input: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Immutable JTH producer input hash mismatch for {path}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
    known_path = OUT / source_known.name
    producer_audit_path = OUT / source_audit.name
    shutil.copy2(source_known, known_path)
    shutil.copy2(source_audit, producer_audit_path)
    copied = {
        "status": "passed",
        "source_directory": str(SOURCE_OUT),
        "source_directory_status": (
            "quarantined and non-citable after an interrupted 2026-10-27 run"
        ),
        "destination_directory": str(OUT),
        "copied_immutable_inputs": {
            known_path.name: {
                "sha256": sha256(known_path),
                "bytes": known_path.stat().st_size,
            },
            producer_audit_path.name: {
                "sha256": sha256(producer_audit_path),
                "bytes": producer_audit_path.stat().st_size,
            },
        },
        "rule": (
            "No generated result from the quarantined source directory is "
            "copied or cited; all v2 tables, figures, and narratives are rebuilt."
        ),
    }
    write_json(OUT / "SOURCE_INPUT_MANIFEST.json", copied)
    return known_path, producer_audit_path, copied


def load_query_metadata() -> tuple[pd.DataFrame, dict]:
    train = pd.read_csv(DATA / "train.csv")
    test_raw = pd.read_csv(DATA / "test.csv")
    test_raw["timestamp"] = pd.to_datetime(test_raw["timestamp"])
    test = test_raw.drop_duplicates().copy()
    q = (
        test.groupby(["candidate_id", "timestamp"], sort=True)["job_id"]
        .agg(lambda x: sorted(set(map(int, x))))
        .reset_index(name="relevant_jobs")
    )
    q.insert(0, "query_index", np.arange(len(q), dtype=np.int64))
    q["n_relevant"] = q.relevant_jobs.map(len).astype(int)

    # These names deliberately distinguish released interaction history from
    # static CV/work-history fields.  They reproduce the audited subgroup rules
    # used in the earlier JTH analyzer, with clearer labels.
    training_activity = train.candidate_id.value_counts()
    training_popularity = train.job_id.value_counts()
    q["training_interaction_count"] = (
        q.candidate_id.map(training_activity).fillna(0).astype(int)
    )
    q["training_interaction_activity"] = pd.cut(
        q.training_interaction_count,
        bins=[-1, 0, 1, 3, np.inf],
        labels=["cold_0", "one_1", "two_to_three_2_3", "four_plus_4_plus"],
    ).astype(str)
    q["target_mean_training_popularity"] = q.relevant_jobs.map(
        lambda ids: float(np.mean([training_popularity.get(job_id, 0) for job_id in ids]))
    )
    q["target_popularity"] = pd.cut(
        q.target_mean_training_popularity,
        bins=[-1, 0, 2, np.inf],
        labels=["unseen_0", "rare_gt0_to_2", "popular_gt2"],
    ).astype(str)

    # T1--T4 are bins of the dense chronological timestamp rank.  They are
    # neither calendar quarters nor equal-query quantiles.
    q["timestamp_dense_rank"] = q.timestamp.rank(method="dense").astype(int)
    n_timestamps = int(q.timestamp_dense_rank.max())
    q["timestamp_rank_fraction"] = q.timestamp_dense_rank / n_timestamps
    q["timestamp_rank_quartile"] = pd.cut(
        q.timestamp_rank_fraction,
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["T1", "T2", "T3", "T4"],
        include_lowest=True,
    ).astype(str)

    train_pairs = train[["candidate_id", "job_id"]].drop_duplicates()
    overlap = test.merge(train_pairs, on=["candidate_id", "job_id"], how="inner")
    overlap_query_keys = overlap[["candidate_id", "timestamp"]].drop_duplicates()
    overlap_query_ids = q.merge(
        overlap_query_keys,
        on=["candidate_id", "timestamp"],
        how="inner",
    ).query_index
    audit = {
        "raw_test_rows": int(len(test_raw)),
        "deduplicated_test_events": int(len(test)),
        "queries": int(len(q)),
        "unique_candidates": int(q.candidate_id.nunique()),
        "unique_timestamps": int(q.timestamp.nunique()),
        "target_events_after_deduplication": int(q.n_relevant.sum()),
        "overlap_unique_candidate_job_pairs": int(
            overlap[["candidate_id", "job_id"]].drop_duplicates().shape[0]
        ),
        "overlap_target_events": int(len(overlap)),
        "overlap_queries": int(len(overlap_query_keys)),
        "overlap_query_indices": sorted(map(int, overlap_query_ids)),
        "timestamp_group_rule": (
            "Dense-rank the 389 unique query timestamps chronologically; divide "
            "rank by 389; bin (0,.25], (.25,.50], (.50,.75], (.75,1] as T1--T4."
        ),
        "subgroup_rules": {
            "training-interaction activity": {
                "source": "count of released binary/train.csv rows per candidate",
                "groups": {
                    "cold_0": "0 training interactions",
                    "one_1": "1 training interaction",
                    "two_to_three_2_3": "2--3 training interactions",
                    "four_plus_4_plus": "4 or more training interactions",
                },
            },
            "target popularity": {
                "source": "mean released-training interaction count of a query's relevant jobs",
                "groups": {
                    "unseen_0": "mean count exactly 0",
                    "rare_gt0_to_2": "mean count greater than 0 and at most 2",
                    "popular_gt2": "mean count greater than 2",
                },
            },
        },
    }
    return q, audit


def load_retained_k20(
    qmeta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stream retained rows, retaining parity subset and temporal aggregates."""
    use = [
        "query_index",
        "group",
        "n_relevant",
        "eligible_catalog",
        "model",
        "strategy",
        "seed",
        "candidate_size",
        "k",
        *METRICS,
    ]
    quartile = qmeta.set_index("query_index").timestamp_rank_quartile
    selected_parts: list[pd.DataFrame] = []
    temporal_parts: list[pd.DataFrame] = []
    root_rows: dict[str, int] = {}
    total_rows = 0
    temporal_keys = [
        "timestamp_rank_quartile",
        "model",
        "strategy",
        "seed",
        "candidate_size",
    ]

    for root in INPUT_ROOTS:
        root_count = 0
        for chunk in pd.read_csv(
            root / "per_query_metrics.csv.gz", usecols=use, chunksize=350_000
        ):
            chunk = chunk[chunk.k.eq(20)].drop(columns="k")
            root_count += len(chunk)
            total_rows += len(chunk)
            keep = chunk.strategy.eq("full") | chunk.candidate_size.eq(100)
            selected_parts.append(chunk.loc[keep].copy())

            chunk["timestamp_rank_quartile"] = chunk.query_index.map(quartile)
            if chunk.timestamp_rank_quartile.isna().any():
                raise ValueError("Retained row has an unknown query_index")
            sums = (
                chunk.groupby(temporal_keys, observed=True)[METRICS]
                .sum()
                .reset_index()
            )
            counts = (
                chunk.groupby(temporal_keys, observed=True)
                .size()
                .rename("n_queries")
                .reset_index()
            )
            temporal_parts.append(
                sums.merge(counts, on=temporal_keys, validate="one_to_one")
            )
        root_rows[str(root)] = int(root_count)

    if total_rows != EXPECTED_K20_ROWS:
        raise ValueError(
            f"Unexpected retained K=20 rows: {total_rows:,} != {EXPECTED_K20_ROWS:,}"
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) != EXPECTED_SELECTED_ROWS:
        raise ValueError(
            f"Unexpected retained full/100 rows: {len(selected):,} != "
            f"{EXPECTED_SELECTED_ROWS:,}"
        )
    if selected.duplicated(
        ["query_index", "model", "strategy", "seed", "candidate_size"]
    ).any():
        raise ValueError("Duplicate retained full/100 rows")

    temporal_raw = pd.concat(temporal_parts, ignore_index=True)
    temporal_sums = (
        temporal_raw.groupby(temporal_keys, observed=True)[[*METRICS, "n_queries"]]
        .sum()
        .reset_index()
    )
    temporal = temporal_sums.copy()
    for metric in METRICS:
        temporal[metric] = temporal_sums[metric] / temporal_sums.n_queries
    temporal = temporal[[*temporal_keys, "n_queries", *METRICS]].sort_values(
        temporal_keys
    )
    return selected, temporal, {
        "retained_k20_rows": int(total_rows),
        "retained_full_and_100_rows": int(len(selected)),
        "retained_rows_by_root": root_rows,
    }


def load_evaluation_bias_matrix() -> tuple[pd.DataFrame, dict]:
    """Load exact K=20 full/sampled outcomes into a compact per-query matrix."""
    use = [
        "query_index", "model", "strategy", "seed", "candidate_size", "k",
        *BIAS_METRICS,
    ]
    wide_parts: list[pd.DataFrame] = []
    root_audit: dict[str, dict] = {}
    for root in INPUT_ROOTS:
        chunks = []
        for chunk in pd.read_csv(
            root / "per_query_metrics.csv.gz", usecols=use, chunksize=350_000
        ):
            chunks.append(chunk[chunk.k.eq(20)].drop(columns="k"))
        data = pd.concat(chunks, ignore_index=True)
        keys = ["query_index", "model", "strategy", "seed", "candidate_size"]
        if data.duplicated(keys).any():
            raise ValueError(f"Duplicate K=20 bias keys in {root}")
        root_wides = []
        for metric in BIAS_METRICS:
            wide = data.pivot(
                index="query_index",
                columns=["model", "strategy", "seed", "candidate_size"],
                values=metric,
            )
            wide.columns = pd.MultiIndex.from_tuples(
                [(metric, *column) for column in wide.columns],
                names=["metric", "model", "strategy", "seed", "candidate_size"],
            )
            root_wides.append(wide)
        root_wide = pd.concat(root_wides, axis=1).sort_index(axis=1)
        if root_wide.isna().any().any():
            raise ValueError(f"Incomplete per-query evaluation-bias matrix in {root}")
        wide_parts.append(root_wide)
        root_audit[str(root)] = {
            "k20_rows": int(len(data)),
            "models": sorted(map(str, data.model.unique())),
            "matrix_shape": [int(root_wide.shape[0]), int(root_wide.shape[1])],
        }

    wide = pd.concat(wide_parts, axis=1).sort_index().sort_index(axis=1)
    if wide.index.tolist() != list(range(EXPECTED_QUERIES)):
        raise ValueError("Evaluation-bias matrix query index is not the audited 0..8340 range")
    if wide.columns.duplicated().any() or wide.isna().any().any():
        raise ValueError("Combined evaluation-bias matrix has duplicate/missing cells")
    expected_columns = len(BIAS_METRICS) * len(MODELS) * (
        1 + 2 * 5 * len(BIAS_SIZES)
    )
    if wide.shape != (EXPECTED_QUERIES, expected_columns):
        raise ValueError(
            f"Unexpected evaluation-bias matrix shape {wide.shape}; "
            f"expected ({EXPECTED_QUERIES}, {expected_columns})"
        )
    return wide, {
        "shape": [int(wide.shape[0]), int(wide.shape[1])],
        "bytes": int(wide.memory_usage(index=True, deep=True).sum()),
        "roots": root_audit,
        "metrics": BIAS_METRICS,
        "candidate_sizes": BIAS_SIZES,
        "seed_aggregation": "per-query arithmetic mean over seeds 1--5",
    }


def parity_check(new: pd.DataFrame, retained: pd.DataFrame) -> dict:
    keys = ["query_index", "model", "strategy", "seed", "candidate_size"]
    original = new[new.estimand.eq("original")].copy()
    if len(original) != EXPECTED_ORIGINAL_ROWS:
        raise ValueError(
            f"Known-item original rows {len(original):,} != {EXPECTED_ORIGINAL_ROWS:,}"
        )
    if original.duplicated(keys).any():
        raise ValueError("Duplicate original-condition known-item rows")
    left = original.set_index(keys).sort_index()
    right = retained.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        only_new = int((~left.index.isin(right.index)).sum())
        only_retained = int((~right.index.isin(left.index)).sum())
        raise ValueError(
            f"Parity keys differ: new_only={only_new}, retained_only={only_retained}"
        )

    metric_audit: dict[str, dict] = {}
    hard_failure = False
    for metric in METRICS:
        a = left[metric].to_numpy(float)
        b = right[metric].to_numpy(float)
        difference = np.abs(a - b)
        mismatches = int(np.count_nonzero(difference > PARITY_ATOL))
        metric_audit[metric] = {
            "max_absolute_difference": float(difference.max(initial=0.0)),
            "differences_above_tolerance": mismatches,
        }
        hard_failure |= mismatches > 0

    structural = {
        "group_mismatches": int(
            np.count_nonzero(left.group.astype(str).to_numpy() != right.group.astype(str).to_numpy())
        ),
        "n_relevant_mismatches": int(
            np.count_nonzero(
                left.n_relevant_original.to_numpy(int)
                != right.n_relevant.to_numpy(int)
            )
        ),
        "eligible_catalog_mismatches": int(
            np.count_nonzero(
                left.eligible_catalog_original.to_numpy(int)
                != right.eligible_catalog.to_numpy(int)
            )
        ),
    }
    hard_failure |= any(structural.values())
    result = {
        "status": "failed" if hard_failure else "passed",
        "rows_compared": int(len(left)),
        "absolute_tolerance": PARITY_ATOL,
        "metrics": metric_audit,
        "structural": structural,
    }
    if hard_failure:
        write_json(OUT / "PARITY_FAILURE.json", result)
        raise ValueError("Original-condition parity failed; findings were not generated")
    return result


def add_seed_means(frame: pd.DataFrame, estimand_label: str) -> pd.DataFrame:
    """Aggregate known-item output and append per-query five-seed means."""
    keys = ["model", "strategy", "seed", "candidate_size"]
    base = (
        frame.groupby(keys, observed=True)
        .agg(
            n_queries=("query_index", "nunique"),
            n_relevant_mean=("n_relevant_eval", "mean"),
            eligible_catalog_mean=("eligible_catalog_eval", "mean"),
            actual_candidate_size_mean=("actual_candidate_size", "mean"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
        .reset_index()
    )
    base.insert(0, "estimand", estimand_label)
    base["seed_aggregation"] = "individual"

    sampled = frame[~frame.strategy.eq("full")]
    per_query_seed_mean = (
        sampled.groupby(
            ["query_index", "model", "strategy", "candidate_size"], observed=True
        )[[*METRICS, "n_relevant_eval", "eligible_catalog_eval", "actual_candidate_size"]]
        .mean()
        .reset_index()
    )
    seed_mean = (
        per_query_seed_mean.groupby(
            ["model", "strategy", "candidate_size"], observed=True
        )
        .agg(
            n_queries=("query_index", "nunique"),
            n_relevant_mean=("n_relevant_eval", "mean"),
            eligible_catalog_mean=("eligible_catalog_eval", "mean"),
            actual_candidate_size_mean=("actual_candidate_size", "mean"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
        .reset_index()
    )
    seed_mean.insert(0, "estimand", estimand_label)
    seed_mean.insert(3, "seed", 0)
    seed_mean["seed_aggregation"] = "mean_over_5_seeds"
    return pd.concat([base, seed_mean], ignore_index=True)


def known_item_summaries(known: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    novel_ids = set(map(int, known.loc[known.estimand.eq("novel_item"), "query_index"].unique()))
    original = known[known.estimand.eq("original")]
    novel = known[known.estimand.eq("novel_item")]
    original_matched = original[original.query_index.isin(novel_ids)]
    pieces = [
        add_seed_means(original, "original_all"),
        add_seed_means(original_matched, "original_matched_novel_cohort"),
        add_seed_means(novel, "novel_item"),
    ]
    summary = pd.concat(pieces, ignore_index=True)
    compare_keys = [
        "model",
        "strategy",
        "seed",
        "candidate_size",
        "seed_aggregation",
    ]
    novelty = summary[summary.estimand.eq("novel_item")]
    differences = []
    for reference in ["original_all", "original_matched_novel_cohort"]:
        ref = summary[summary.estimand.eq(reference)]
        merged = novelty.merge(
            ref,
            on=compare_keys,
            suffixes=("_novel", "_reference"),
            validate="one_to_one",
        )
        merged.insert(0, "reference_estimand", reference)
        for metric in METRICS:
            merged[f"{metric}_difference_novel_minus_reference"] = (
                merged[f"{metric}_novel"] - merged[f"{metric}_reference"]
            )
        differences.append(merged)
    return summary, pd.concat(differences, ignore_index=True), novel_ids


def build_primary_contrast_matrix(
    original: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build one query-by-contrast matrix for all preselected protocols."""
    data = original[original.model.isin(["bm25", *CHALLENGERS])]
    query_ids = np.sort(data.query_index.unique())
    matrices: list[np.ndarray] = []
    metadata: list[dict] = []

    protocols: list[tuple[str, int, int, str, pd.DataFrame]] = []
    full = data[data.strategy.eq("full")]
    protocols.append(("full", 0, -1, "individual", full))
    for strategy in ["uniform", "popularity_weighted"]:
        for seed in range(1, 6):
            subset = data[data.strategy.eq(strategy) & data.seed.eq(seed)]
            protocols.append((strategy, seed, 100, "individual", subset))
        subset = data[data.strategy.eq(strategy)]
        averaged = (
            subset.groupby(["query_index", "model"], observed=True)[METRICS]
            .mean()
            .reset_index()
        )
        protocols.append((strategy, 0, 100, "mean_over_5_seeds", averaged))

    for strategy, seed, size, seed_aggregation, subset in protocols:
        pivot = subset.set_index(["query_index", "model"])[PRIMARY_METRICS].unstack("model")
        pivot = pivot.reindex(query_ids)
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete primary contrast protocol: {strategy}, seed={seed}")
        for metric in PRIMARY_METRICS:
            for challenger in CHALLENGERS:
                values = (
                    pivot[(metric, "bm25")].to_numpy(float)
                    - pivot[(metric, challenger)].to_numpy(float)
                )
                matrices.append(values)
                metadata.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "candidate_size": size,
                        "seed_aggregation": seed_aggregation,
                        "metric": metric,
                        "contrast": f"bm25_minus_{challenger}",
                    }
                )
    return query_ids, np.column_stack(matrices), metadata


def grouped_sums(
    values: np.ndarray, codes: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((n_groups, values.shape[1]), dtype=np.float64)
    np.add.at(sums, codes, values)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    return sums, counts


def one_way_cluster_draws(
    values: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    seed: int,
    batch_size: int = 100,
) -> np.ndarray:
    sums, counts = grouped_sums(values, codes, n_groups)
    rng = np.random.default_rng(seed)
    draws = np.empty((BOOTSTRAPS, values.shape[1]), dtype=np.float64)
    probabilities = np.full(n_groups, 1.0 / n_groups)
    for start in range(0, BOOTSTRAPS, batch_size):
        stop = min(BOOTSTRAPS, start + batch_size)
        multiplicity = rng.multinomial(
            n_groups, probabilities, size=stop - start
        ).astype(np.float64)
        denominator = multiplicity @ counts
        draws[start:stop] = (multiplicity @ sums) / denominator[:, None]
    return draws


def two_way_cluster_draws(
    values: np.ndarray,
    candidate_codes: np.ndarray,
    n_candidates: int,
    date_codes: np.ndarray,
    n_dates: int,
    seed: int,
    batch_size: int = 100,
) -> np.ndarray:
    """Two-way pigeonhole bootstrap using products of cluster multiplicities."""
    rng = np.random.default_rng(seed)
    draws = np.empty((BOOTSTRAPS, values.shape[1]), dtype=np.float64)
    p_candidate = np.full(n_candidates, 1.0 / n_candidates)
    p_date = np.full(n_dates, 1.0 / n_dates)
    for start in range(0, BOOTSTRAPS, batch_size):
        stop = min(BOOTSTRAPS, start + batch_size)
        n = stop - start
        candidate_multiplicity = rng.multinomial(
            n_candidates, p_candidate, size=n
        ).astype(np.float64)
        date_multiplicity = rng.multinomial(n_dates, p_date, size=n).astype(np.float64)
        query_weight = (
            candidate_multiplicity[:, candidate_codes]
            * date_multiplicity[:, date_codes]
        )
        denominator = query_weight.sum(axis=1)
        if np.any(denominator == 0):
            raise ValueError("Zero-weight replicate in two-way cluster bootstrap")
        draws[start:stop] = (query_weight @ values) / denominator[:, None]
    return draws


def bootstrap_primary_contrasts(
    known: pd.DataFrame, qmeta: pd.DataFrame
) -> pd.DataFrame:
    original = known[known.estimand.eq("original")]
    query_ids, values, column_meta = build_primary_contrast_matrix(original)
    meta = qmeta.set_index("query_index").loc[query_ids]
    candidate_codes, candidate_levels = pd.factorize(meta.candidate_id, sort=True)
    date_codes, date_levels = pd.factorize(meta.timestamp, sort=True)

    method_draws = {
        "date_cluster": one_way_cluster_draws(
            values,
            date_codes,
            len(date_levels),
            BOOTSTRAP_SEEDS["date_cluster"],
        ),
        "candidate_cluster": one_way_cluster_draws(
            values,
            candidate_codes,
            len(candidate_levels),
            BOOTSTRAP_SEEDS["candidate_cluster"],
        ),
        "two_way_candidate_date": two_way_cluster_draws(
            values,
            candidate_codes,
            len(candidate_levels),
            date_codes,
            len(date_levels),
            BOOTSTRAP_SEEDS["two_way_candidate_date"],
        ),
    }
    estimate = values.mean(axis=0)
    rows = []
    for method, draws in method_draws.items():
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        probability_positive = (draws > 0).mean(axis=0)
        standard_error = draws.std(axis=0, ddof=1)
        for j, meta_row in enumerate(column_meta):
            rows.append(
                {
                    "estimand": "original_all_queries",
                    **meta_row,
                    "cluster_method": method,
                    "estimate": estimate[j],
                    "ci_low": low[j],
                    "ci_high": high[j],
                    "bootstrap_standard_error": standard_error[j],
                    "bootstrap_probability_positive": probability_positive[j],
                    "n_queries": len(query_ids),
                    "n_candidate_clusters": len(candidate_levels),
                    "n_date_clusters": len(date_levels),
                    "bootstrap_replicates": BOOTSTRAPS,
                    "bootstrap_seed": BOOTSTRAP_SEEDS[method],
                }
            )
    return pd.DataFrame(rows)


def bootstrap_full_catalog_subgroups(
    retained: pd.DataFrame, qmeta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dependence-aware descriptive CIs for audited JTH subgroups.

    The output is intentionally restricted to the three preselected content
    models and three primary metrics.  It describes effectiveness heterogeneity;
    it is not a fairness or causal analysis.
    """
    selected_models = ["bm25", "semantic", "tfidf"]
    full = retained[
        retained.strategy.eq("full") & retained.model.isin(selected_models)
    ]
    pivot = full.set_index(["query_index", "model"])[PRIMARY_METRICS].unstack("model")
    pivot = pivot.reindex(qmeta.query_index)
    if pivot.isna().any().any():
        raise ValueError("Incomplete full-catalog subgroup matrix")

    value_columns: list[tuple[str, str]] = []
    values_parts: list[np.ndarray] = []
    for metric in PRIMARY_METRICS:
        for model in selected_models:
            value_columns.append((metric, model))
            values_parts.append(pivot[(metric, model)].to_numpy(float))
    all_values = np.column_stack(values_parts)

    dimension_columns = {
        "timestamp-rank quartile": "timestamp_rank_quartile",
        "training-interaction activity": "training_interaction_activity",
        "target popularity": "target_popularity",
    }
    subgroup_order = {
        "T1": 1,
        "T2": 2,
        "T3": 3,
        "T4": 4,
        "cold_0": 1,
        "one_1": 2,
        "two_to_three_2_3": 3,
        "four_plus_4_plus": 4,
        "unseen_0": 1,
        "rare_gt0_to_2": 2,
        "popular_gt2": 3,
    }
    rows: list[dict] = []
    definition_rows: list[dict] = []
    q_indexed = qmeta.set_index("query_index")
    for dimension, column in dimension_columns.items():
        for subgroup, subgroup_meta in qmeta.groupby(column, sort=True):
            query_ids = subgroup_meta.query_index.to_numpy(int)
            row_positions = q_indexed.index.get_indexer(query_ids)
            if np.any(row_positions < 0):
                raise ValueError("Unknown subgroup query index")
            values = all_values[row_positions]
            candidate_codes, candidate_levels = pd.factorize(
                subgroup_meta.candidate_id, sort=True
            )
            date_codes, date_levels = pd.factorize(subgroup_meta.timestamp, sort=True)
            method_draws = {
                "date_cluster": one_way_cluster_draws(
                    values,
                    date_codes,
                    len(date_levels),
                    SUBGROUP_BOOTSTRAP_SEEDS["date_cluster"],
                ),
                "candidate_cluster": one_way_cluster_draws(
                    values,
                    candidate_codes,
                    len(candidate_levels),
                    SUBGROUP_BOOTSTRAP_SEEDS["candidate_cluster"],
                ),
                "two_way_candidate_date": two_way_cluster_draws(
                    values,
                    candidate_codes,
                    len(candidate_levels),
                    date_codes,
                    len(date_levels),
                    SUBGROUP_BOOTSTRAP_SEEDS["two_way_candidate_date"],
                ),
            }
            estimate = values.mean(axis=0)
            for method, draws in method_draws.items():
                low, high = np.quantile(draws, [0.025, 0.975], axis=0)
                standard_error = draws.std(axis=0, ddof=1)
                for j, (metric, model) in enumerate(value_columns):
                    rows.append(
                        {
                            "subgroup_dimension": dimension,
                            "subgroup_value": subgroup,
                            "subgroup_order": subgroup_order[subgroup],
                            "model": model,
                            "metric": metric,
                            "protocol": "full_catalog_k20",
                            "cluster_method": method,
                            "estimate": estimate[j],
                            "ci_low": low[j],
                            "ci_high": high[j],
                            "bootstrap_standard_error": standard_error[j],
                            "n_queries": len(subgroup_meta),
                            "n_candidates": len(candidate_levels),
                            "n_dates": len(date_levels),
                            "bootstrap_replicates": BOOTSTRAPS,
                            "bootstrap_seed": SUBGROUP_BOOTSTRAP_SEEDS[method],
                            "interpretation": "descriptive; not fairness or causal inference",
                        }
                    )
            definition = {
                "subgroup_dimension": dimension,
                "subgroup_value": subgroup,
                "subgroup_order": subgroup_order[subgroup],
                "n_queries": int(len(subgroup_meta)),
                "n_candidates": int(subgroup_meta.candidate_id.nunique()),
                "n_dates": int(subgroup_meta.timestamp.nunique()),
            }
            if dimension == "timestamp-rank quartile":
                definition.update(
                    {
                        "source_measure": "dense chronological timestamp rank fraction",
                        "source_min": float(subgroup_meta.timestamp_dense_rank.min()),
                        "source_max": float(subgroup_meta.timestamp_dense_rank.max()),
                    }
                )
            elif dimension == "training-interaction activity":
                definition.update(
                    {
                        "source_measure": "released training interactions per candidate",
                        "source_min": float(subgroup_meta.training_interaction_count.min()),
                        "source_max": float(subgroup_meta.training_interaction_count.max()),
                    }
                )
            else:
                definition.update(
                    {
                        "source_measure": "mean released-training popularity of query targets",
                        "source_min": float(subgroup_meta.target_mean_training_popularity.min()),
                        "source_max": float(subgroup_meta.target_mean_training_popularity.max()),
                    }
                )
            definition_rows.append(definition)
    return pd.DataFrame(rows), pd.DataFrame(definition_rows)


def build_evaluation_bias_arrays(
    wide: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], pd.DataFrame]:
    """Create paired full, five-seed sampled, and sampled-minus-full arrays."""
    full_columns: list[np.ndarray] = []
    sampled_columns: list[np.ndarray] = []
    bias_columns: list[np.ndarray] = []
    descriptors: list[dict] = []
    seed_rows: list[dict] = []
    for metric in BIAS_METRICS:
        for model in MODELS:
            full = wide[(metric, model, "full", 0, -1)].to_numpy(float)
            for strategy in ["uniform", "popularity_weighted"]:
                for size in BIAS_SIZES:
                    sampled_by_seed = []
                    for seed in range(1, 6):
                        sampled = wide[
                            (metric, model, strategy, seed, size)
                        ].to_numpy(float)
                        sampled_by_seed.append(sampled)
                        seed_rows.append({
                            "model": model,
                            "metric": metric,
                            "k": 20,
                            "strategy": strategy,
                            "candidate_size": size,
                            "seed": seed,
                            "seed_aggregation": "individual",
                            "n_queries": EXPECTED_QUERIES,
                            "full_catalog_value": float(full.mean()),
                            "sampled_value": float(sampled.mean()),
                            "signed_bias_sampled_minus_full": float(
                                (sampled - full).mean()
                            ),
                            "absolute_bias": abs(float((sampled - full).mean())),
                        })
                    sampled_mean = np.mean(sampled_by_seed, axis=0)
                    full_columns.append(full)
                    sampled_columns.append(sampled_mean)
                    bias_columns.append(sampled_mean - full)
                    descriptors.append({
                        "model": model,
                        "metric": metric,
                        "k": 20,
                        "strategy": strategy,
                        "candidate_size": size,
                        "seed_aggregation": "per_query_mean_over_5_seeds",
                    })
    return (
        np.column_stack(full_columns),
        np.column_stack(sampled_columns),
        np.column_stack(bias_columns),
        descriptors,
        pd.DataFrame(seed_rows),
    )


def _cluster_bias_rows(
    full_values: np.ndarray,
    sampled_values: np.ndarray,
    bias_values: np.ndarray,
    descriptors: list[dict],
    meta: pd.DataFrame,
    scope: dict,
    seed_offset: int = 0,
) -> list[dict]:
    candidate_codes, candidate_levels = pd.factorize(meta.candidate_id, sort=True)
    date_codes, date_levels = pd.factorize(meta.timestamp, sort=True)
    n_candidates = len(candidate_levels)
    n_dates = len(date_levels)
    adequate = n_candidates >= 30 and n_dates >= 30
    if not adequate:
        return [{
            **scope,
            **descriptor,
            "cluster_method": "not_estimated",
            "estimate": float(bias_values[:, j].mean()),
            "full_catalog_value": float(full_values[:, j].mean()),
            "sampled_value": float(sampled_values[:, j].mean()),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_standard_error": np.nan,
            "bootstrap_probability_positive": np.nan,
            "n_queries": len(meta),
            "n_candidate_clusters": n_candidates,
            "n_date_clusters": n_dates,
            "cluster_adequacy": "insufficient (<30 candidate or date clusters)",
            "bootstrap_replicates": 0,
            "bootstrap_seed": np.nan,
        } for j, descriptor in enumerate(descriptors)]

    methods = {
        "date_cluster": one_way_cluster_draws(
            bias_values, date_codes, n_dates,
            BIAS_BOOTSTRAP_SEEDS["date_cluster"] + seed_offset,
        ),
        "candidate_cluster": one_way_cluster_draws(
            bias_values, candidate_codes, n_candidates,
            BIAS_BOOTSTRAP_SEEDS["candidate_cluster"] + seed_offset,
        ),
        "two_way_candidate_date": two_way_cluster_draws(
            bias_values,
            candidate_codes, n_candidates,
            date_codes, n_dates,
            BIAS_BOOTSTRAP_SEEDS["two_way_candidate_date"] + seed_offset,
        ),
    }
    estimate = bias_values.mean(axis=0)
    full_estimate = full_values.mean(axis=0)
    sampled_estimate = sampled_values.mean(axis=0)
    rows: list[dict] = []
    for method, draws in methods.items():
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        se = draws.std(axis=0, ddof=1)
        probability_positive = (draws > 0).mean(axis=0)
        for j, descriptor in enumerate(descriptors):
            rows.append({
                **scope,
                **descriptor,
                "cluster_method": method,
                "estimate": float(estimate[j]),
                "full_catalog_value": float(full_estimate[j]),
                "sampled_value": float(sampled_estimate[j]),
                "ci_low": float(low[j]),
                "ci_high": float(high[j]),
                "bootstrap_standard_error": float(se[j]),
                "bootstrap_probability_positive": float(probability_positive[j]),
                "n_queries": len(meta),
                "n_candidate_clusters": n_candidates,
                "n_date_clusters": n_dates,
                "cluster_adequacy": "adequate (>=30 candidate and date clusters)",
                "bootstrap_replicates": BOOTSTRAPS,
                "bootstrap_seed": BIAS_BOOTSTRAP_SEEDS[method] + seed_offset,
            })
    return rows


def bootstrap_evaluation_bias(
    wide: pd.DataFrame, qmeta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Paired sampled-minus-full bias overall and in descriptive subgroups."""
    full, sampled, bias, descriptors, seed_detail = build_evaluation_bias_arrays(wide)
    meta = qmeta.set_index("query_index").loc[wide.index]
    overall_rows = _cluster_bias_rows(
        full, sampled, bias, descriptors, meta,
        {"scope": "all_queries", "subgroup_dimension": "all",
         "subgroup_value": "all", "subgroup_order": 1},
    )

    dimension_columns = {
        "timestamp-rank quartile": "timestamp_rank_quartile",
        "training-interaction activity": "training_interaction_activity",
        "target popularity": "target_popularity",
    }
    subgroup_order = {
        "T1": 1, "T2": 2, "T3": 3, "T4": 4,
        "cold_0": 1, "one_1": 2, "two_to_three_2_3": 3,
        "four_plus_4_plus": 4,
        "unseen_0": 1, "rare_gt0_to_2": 2, "popular_gt2": 3,
    }
    subgroup_rows: list[dict] = []
    group_index = 1
    for dimension, column in dimension_columns.items():
        for group in sorted(meta[column].unique(), key=lambda x: subgroup_order[x]):
            mask = meta[column].eq(group).to_numpy()
            subgroup_rows.extend(_cluster_bias_rows(
                full[mask], sampled[mask], bias[mask], descriptors, meta.loc[mask],
                {
                    "scope": "descriptive_subgroup",
                    "subgroup_dimension": dimension,
                    "subgroup_value": group,
                    "subgroup_order": subgroup_order[group],
                },
                seed_offset=group_index * 100,
            ))
            group_index += 1
    return pd.DataFrame(overall_rows), pd.DataFrame(subgroup_rows), seed_detail


def tied(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=TIE_RTOL, atol=TIE_ATOL))


def pairwise_state(a: float, b: float) -> int:
    if tied(a, b):
        return 0
    return 1 if a > b else -1


def winner_set(values: np.ndarray, models: list[str]) -> list[str]:
    maximum = float(np.max(values))
    return [
        model
        for model, value in zip(models, values)
        if np.isclose(value, maximum, rtol=TIE_RTOL, atol=TIE_ATOL)
    ]


def rank_levels(values: np.ndarray, models: list[str]) -> str:
    order = sorted(range(len(models)), key=lambda j: (-values[j], models[j]))
    levels: list[list[str]] = []
    for index in order:
        if not levels:
            levels.append([models[index]])
            continue
        representative = models.index(levels[-1][0])
        if tied(values[index], values[representative]):
            levels[-1].append(models[index])
        else:
            levels.append([models[index]])
    return ">".join("=".join(sorted(level)) for level in levels)


def compare_rank_vectors(
    reference: np.ndarray, comparison: np.ndarray, models: list[str]
) -> dict:
    ref_winners = winner_set(reference, models)
    comp_winners = winner_set(comparison, models)
    ref_states: list[int] = []
    comp_states: list[int] = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ref_states.append(pairwise_state(reference[i], reference[j]))
            comp_states.append(pairwise_state(comparison[i], comparison[j]))
    ref_states_array = np.asarray(ref_states)
    comp_states_array = np.asarray(comp_states)
    agreement = ref_states_array == comp_states_array
    reversals = ref_states_array * comp_states_array == -1
    tie_disagreements = (ref_states_array == 0) ^ (comp_states_array == 0)
    sampled_ref_values = np.array(
        [reference[models.index(model)] for model in comp_winners], dtype=float
    )
    ref_max = float(np.max(reference))
    tau = kendalltau(
        reference,
        comparison,
        variant="b",
        method="auto",
        nan_policy="propagate",
    ).statistic
    return {
        "reference_rank_levels": rank_levels(reference, models),
        "comparison_rank_levels": rank_levels(comparison, models),
        "reference_winner_set": "|".join(sorted(ref_winners)),
        "comparison_winner_set": "|".join(sorted(comp_winners)),
        "reference_winner_set_size": len(ref_winners),
        "comparison_winner_set_size": len(comp_winners),
        "winner_set_exact_agreement": int(set(ref_winners) == set(comp_winners)),
        "winner_set_overlap": int(bool(set(ref_winners) & set(comp_winners))),
        "kendall_tau_b": float(tau) if np.isfinite(tau) else np.nan,
        "pairwise_order_agreement": float(agreement.mean()),
        "pairwise_agree_count": int(agreement.sum()),
        "pairwise_reverse_count": int(reversals.sum()),
        "pairwise_tie_disagreement_count": int(tie_disagreements.sum()),
        "pairwise_comparisons": int(len(ref_states)),
        "winner_regret_best_case": float(ref_max - sampled_ref_values.max()),
        "winner_regret_worst_case": float(ref_max - sampled_ref_values.min()),
    }


def regenerate_rank_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SEVEN_MODEL / "seven_model_summary_metrics.csv")
    if sorted(summary.model.unique()) != MODELS:
        raise ValueError("Seven-model summary has unexpected models")
    full = summary[summary.strategy.eq("full")]
    sampled = summary[~summary.strategy.eq("full")]
    rows = []
    for (group, strategy, seed, size, k), subset in sampled.groupby(
        ["group", "strategy", "seed", "candidate_size", "k"], observed=True
    ):
        reference = full[(full.group.eq(group)) & (full.k.eq(k))].set_index("model")
        comparison = subset.set_index("model")
        if set(reference.index) != set(MODELS) or set(comparison.index) != set(MODELS):
            raise ValueError("Incomplete rank cell")
        for metric in METRICS:
            ref_values = reference.loc[MODELS, metric].to_numpy(float)
            comp_values = comparison.loc[MODELS, metric].to_numpy(float)
            rows.append(
                {
                    "group": group,
                    "strategy": strategy,
                    "seed": seed,
                    "candidate_size": size,
                    "k": k,
                    "metric": metric,
                    **compare_rank_vectors(ref_values, comp_values, MODELS),
                }
            )
    ranks = pd.DataFrame(rows)
    k20 = ranks[ranks.group.eq("all") & ranks.k.eq(20)]
    summary_columns = [
        "winner_set_exact_agreement",
        "winner_set_overlap",
        "kendall_tau_b",
        "pairwise_order_agreement",
        "winner_regret_best_case",
        "winner_regret_worst_case",
    ]
    rank_summary = (
        k20.groupby(["strategy", "candidate_size"], observed=True)[summary_columns]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    rank_summary.columns = [
        "_".join(map(str, col)).rstrip("_") if isinstance(col, tuple) else col
        for col in rank_summary.columns
    ]
    decision_counts = (
        k20.groupby(["strategy", "candidate_size"], observed=True)
        .size()
        .rename("n_metric_seed_decisions")
        .reset_index()
    )
    rank_summary = rank_summary.merge(
        decision_counts, on=["strategy", "candidate_size"], validate="one_to_one"
    )
    metric_summary = (
        k20.groupby(["strategy", "candidate_size", "metric"], observed=True)[
            summary_columns
        ]
        .mean()
        .reset_index()
    )
    metric_summary["n_seed_decisions"] = 5
    return ranks, rank_summary, metric_summary


def known_item_rank_table(summary: pd.DataFrame) -> pd.DataFrame:
    protocol_keys = ["strategy", "seed", "candidate_size", "seed_aggregation"]
    novel = summary[summary.estimand.eq("novel_item")]
    rows = []
    for reference_label in ["original_all", "original_matched_novel_cohort"]:
        reference = summary[summary.estimand.eq(reference_label)]
        for protocol, comp_cell in novel.groupby(protocol_keys, observed=True):
            mask = np.ones(len(reference), dtype=bool)
            for key, value in zip(protocol_keys, protocol):
                mask &= reference[key].eq(value).to_numpy()
            ref_cell = reference.loc[mask].set_index("model")
            comp_cell = comp_cell.set_index("model")
            if set(ref_cell.index) != set(MODELS) or set(comp_cell.index) != set(MODELS):
                raise ValueError("Incomplete known-item rank cell")
            protocol_dict = dict(zip(protocol_keys, protocol))
            for metric in METRICS:
                rows.append(
                    {
                        "reference_estimand": reference_label,
                        "comparison_estimand": "novel_item",
                        **protocol_dict,
                        "metric": metric,
                        **compare_rank_vectors(
                            ref_cell.loc[MODELS, metric].to_numpy(float),
                            comp_cell.loc[MODELS, metric].to_numpy(float),
                            MODELS,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def temporal_definitions(qmeta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, subset in qmeta.groupby("timestamp_rank_quartile", sort=True):
        rows.append(
            {
                "timestamp_rank_quartile": label,
                "dense_rank_min": int(subset.timestamp_dense_rank.min()),
                "dense_rank_max": int(subset.timestamp_dense_rank.max()),
                "timestamp_min": subset.timestamp.min().isoformat(),
                "timestamp_max": subset.timestamp.max().isoformat(),
                "n_unique_timestamps": int(subset.timestamp.nunique()),
                "n_queries": int(len(subset)),
                "n_unique_candidates": int(subset.candidate_id.nunique()),
                "n_target_events": int(subset.n_relevant.sum()),
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(
            OUT / f"{stem}.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def make_figures(
    bootstrap: pd.DataFrame,
    known_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    temporal: pd.DataFrame,
    subgroup_bootstrap: pd.DataFrame,
    evaluation_bias: pd.DataFrame,
    subgroup_evaluation_bias: pd.DataFrame,
) -> list[str]:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        }
    )
    stems: list[str] = []

    # Figure 1: inferential sensitivity to dependence assumption.
    b = bootstrap[
        bootstrap.metric.eq("hitrate")
        & (
            bootstrap.strategy.eq("full")
            | bootstrap.seed_aggregation.eq("mean_over_5_seeds")
        )
    ].copy()
    protocol_order = [
        ("full", "Sampling-free reference"),
        ("uniform", "Uniform 100"),
        ("popularity_weighted", "Popularity-weighted 100"),
    ]
    method_style = {
        "date_cluster": ("Date clusters", "o", "#0072B2"),
        "candidate_cluster": ("Candidate clusters", "s", "#D55E00"),
        "two_way_candidate_date": ("Two-way", "^", "#009E73"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=False)
    for ax, challenger in zip(axes, CHALLENGERS):
        contrast = f"bm25_minus_{challenger}"
        for offset, (method, (label, marker, color)) in zip(
            [-0.16, 0.0, 0.16], method_style.items()
        ):
            rows = []
            for strategy, _ in protocol_order:
                row = b[b.strategy.eq(strategy) & b.cluster_method.eq(method) & b.contrast.eq(contrast)]
                if len(row) != 1:
                    raise ValueError(f"Figure 1 cell count {len(row)} for {strategy}/{method}/{contrast}")
                rows.append(row.iloc[0])
            x = np.arange(3) + offset
            estimates = np.array([row.estimate for row in rows])
            lower = estimates - np.array([row.ci_low for row in rows])
            upper = np.array([row.ci_high for row in rows]) - estimates
            ax.errorbar(
                x,
                estimates,
                yerr=np.vstack([lower, upper]),
                fmt=marker,
                color=color,
                capsize=3,
                label=label,
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(range(3), [label for _, label in protocol_order], rotation=15)
        ax.set_title(f"BM25 minus {DISPLAY[challenger]}")
        ax.set_ylabel("HitRate@20 difference")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Dependence-aware uncertainty for selected JTH contrasts")
    fig.tight_layout()
    save_figure(fig, "figure_1_cluster_sensitivity_hitrate")
    stems.append("figure_1_cluster_sensitivity_hitrate")

    # Figure 2: signed novel-item sensitivity on the matched query cohort.
    k = known_summary[
        known_summary.estimand.isin(["original_matched_novel_cohort", "novel_item"])
        & (
            known_summary.strategy.eq("full")
            | known_summary.seed_aggregation.eq("mean_over_5_seeds")
        )
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    for ax, (strategy, title) in zip(axes, protocol_order):
        cell = k[k.strategy.eq(strategy)]
        x = np.arange(len(MODELS))
        original = (
            cell[cell.estimand.eq("original_matched_novel_cohort")]
            .set_index("model")
            .loc[MODELS, "hitrate"]
        )
        novel = (
            cell[cell.estimand.eq("novel_item")]
            .set_index("model")
            .loc[MODELS, "hitrate"]
        )
        difference = novel.to_numpy(float) - original.to_numpy(float)
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.vlines(x, 0, difference, color=[COLORS[m] for m in MODELS], linewidth=2)
        ax.scatter(x, difference, color=[COLORS[m] for m in MODELS], s=35, zorder=3)
        ax.set_title(title)
        ax.set_xticks(x, [DISPLAY[m] for m in MODELS], rotation=55, ha="right")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Novel-item − matched original HitRate@20")
    fig.suptitle("Known-item sensitivity on the matched JTH query cohort")
    fig.tight_layout()
    save_figure(fig, "figure_2_known_item_hitrate")
    stems.append("figure_2_known_item_hitrate")

    # Figure 3: explicit-tie rank fidelity.
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5))
    rank_metrics = [
        ("kendall_tau_b_mean", "Kendall tau-b"),
        ("pairwise_order_agreement_mean", "Pairwise order agreement"),
        ("winner_set_overlap_mean", "Winner-set overlap rate"),
    ]
    for strategy, linestyle in [("uniform", "-"), ("popularity_weighted", "--")]:
        cell = rank_summary[rank_summary.strategy.eq(strategy)].sort_values("candidate_size")
        label = "Uniform" if strategy == "uniform" else "Popularity-weighted"
        for ax, (column, ylabel) in zip(axes, rank_metrics):
            ax.plot(
                cell.candidate_size,
                cell[column],
                marker="o",
                linestyle=linestyle,
                label=label,
            )
            ax.set_xscale("log")
            ax.set_xlabel("Nominal candidate size")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("JTH aggregate rank fidelity at K=20 (six metrics, five seeds)")
    fig.tight_layout()
    save_figure(fig, "figure_3_rank_fidelity_explicit_ties")
    stems.append("figure_3_rank_fidelity_explicit_ties")

    # Figure 4: correctly labelled dense timestamp-rank groups with two-way CIs.
    t = subgroup_bootstrap[
        subgroup_bootstrap.subgroup_dimension.eq("timestamp-rank quartile")
        & subgroup_bootstrap.metric.eq("hitrate")
        & subgroup_bootstrap.cluster_method.eq("two_way_candidate_date")
    ]
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    for model in ["bm25", "semantic", "tfidf"]:
        cell = t[t.model.eq(model)].copy()
        cell["order"] = cell.subgroup_value.map({"T1": 1, "T2": 2, "T3": 3, "T4": 4})
        cell = cell.sort_values("order")
        estimate = cell.estimate.to_numpy(float)
        ax.errorbar(
            cell.subgroup_value,
            estimate,
            yerr=np.vstack(
                [
                    estimate - cell.ci_low.to_numpy(float),
                    cell.ci_high.to_numpy(float) - estimate,
                ]
            ),
            marker="o",
            color=COLORS[model],
            capsize=3,
            label=DISPLAY[model],
        )
    ax.set_xlabel("Dense timestamp-rank quartile")
    ax.set_ylabel("Full-catalog HitRate@20")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "figure_4_timestamp_rank_quartiles")
    stems.append("figure_4_timestamp_rank_quartiles")

    # Figure 5: descriptive subgroup effectiveness with two-way intervals.
    subgroup_order = {
        "training-interaction activity": [
            "cold_0",
            "one_1",
            "two_to_three_2_3",
            "four_plus_4_plus",
        ],
        "target popularity": ["unseen_0", "rare_gt0_to_2", "popular_gt2"],
    }
    subgroup_labels = {
        "cold_0": "0 (cold)",
        "one_1": "1",
        "two_to_three_2_3": "2–3",
        "four_plus_4_plus": "4+",
        "unseen_0": "0 (unseen)",
        "rare_gt0_to_2": ">0–2",
        "popular_gt2": ">2",
    }
    s = subgroup_bootstrap[
        subgroup_bootstrap.metric.eq("hitrate")
        & subgroup_bootstrap.cluster_method.eq("two_way_candidate_date")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for ax, dimension in zip(axes, subgroup_order):
        order = subgroup_order[dimension]
        for offset, model in zip([-0.14, 0.0, 0.14], ["bm25", "semantic", "tfidf"]):
            cell = s[
                s.subgroup_dimension.eq(dimension) & s.model.eq(model)
            ].set_index("subgroup_value").loc[order]
            x = np.arange(len(order)) + offset
            estimate = cell.estimate.to_numpy(float)
            ax.errorbar(
                x,
                estimate,
                yerr=np.vstack(
                    [
                        estimate - cell.ci_low.to_numpy(float),
                        cell.ci_high.to_numpy(float) - estimate,
                    ]
                ),
                fmt="o",
                capsize=3,
                color=COLORS[model],
                label=DISPLAY[model],
            )
        ax.set_xticks(range(len(order)), [subgroup_labels[value] for value in order])
        ax.set_title(dimension.capitalize())
        ax.set_xlabel(
            "Released-training count group"
            if dimension == "training-interaction activity"
            else "Target-popularity group"
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Sampling-free HitRate@20")
    axes[0].legend(frameon=False)
    fig.suptitle("Descriptive JTH subgroup effectiveness (two-way cluster intervals)")
    fig.tight_layout()
    save_figure(fig, "figure_5_subgroup_hitrate_two_way")
    stems.append("figure_5_subgroup_hitrate_two_way")

    # Figure 6: signed sampled-minus-full bias across every retained size.
    bias = evaluation_bias[
        evaluation_bias.cluster_method.eq("two_way_candidate_date")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0), sharex=True)
    for row, metric in enumerate(BIAS_METRICS):
        for column, strategy in enumerate(["uniform", "popularity_weighted"]):
            ax = axes[row, column]
            cell = bias[bias.metric.eq(metric) & bias.strategy.eq(strategy)]
            for model in MODELS:
                model_cell = cell[cell.model.eq(model)].sort_values("candidate_size")
                x = model_cell.candidate_size.to_numpy(float)
                y = model_cell.estimate.to_numpy(float)
                ax.plot(
                    x, y, marker="o", linewidth=1.2, markersize=3.5,
                    color=COLORS[model], label=DISPLAY[model],
                )
                ax.fill_between(
                    x, model_cell.ci_low.to_numpy(float),
                    model_cell.ci_high.to_numpy(float),
                    color=COLORS[model], alpha=0.08,
                )
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_xscale("log")
            ax.grid(alpha=0.2)
            ax.set_title(
                f"{'HitRate@20' if metric == 'hitrate' else 'MAP@20'}; "
                f"{'Uniform' if strategy == 'uniform' else 'Popularity-weighted'}"
            )
            ax.set_xlabel("Nominal candidate size")
            ax.set_ylabel("Sampled − full")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("JTH evaluation bias across candidate-set sizes", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.0,
    )
    # Reserve a dedicated header band for the title and the two-row legend.
    # This avoids their overlap in all three publication formats.
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    save_figure(fig, "figure_6_evaluation_bias_across_sizes")
    stems.append("figure_6_evaluation_bias_across_sizes")

    # Figure 7: descriptive subgroup distortion at the most aggressive protocol.
    subgroup = subgroup_evaluation_bias[
        subgroup_evaluation_bias.cluster_method.eq("two_way_candidate_date")
        & subgroup_evaluation_bias.metric.eq("hitrate")
        & subgroup_evaluation_bias.strategy.eq("uniform")
        & subgroup_evaluation_bias.candidate_size.eq(100)
        & subgroup_evaluation_bias.model.isin(["bm25", "semantic", "tfidf"])
    ]
    dimensions = [
        "timestamp-rank quartile", "training-interaction activity",
        "target popularity",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), sharey=True)
    for ax, dimension in zip(axes, dimensions):
        cell = subgroup[subgroup.subgroup_dimension.eq(dimension)]
        groups = (
            cell[["subgroup_value", "subgroup_order"]]
            .drop_duplicates().sort_values("subgroup_order").subgroup_value.tolist()
        )
        offsets = [-0.16, 0.0, 0.16]
        for offset, model in zip(offsets, ["bm25", "semantic", "tfidf"]):
            model_cell = cell[cell.model.eq(model)].set_index("subgroup_value").loc[groups]
            x = np.arange(len(groups)) + offset
            y = model_cell.estimate.to_numpy(float)
            ax.errorbar(
                x, y,
                yerr=np.vstack([
                    y - model_cell.ci_low.to_numpy(float),
                    model_cell.ci_high.to_numpy(float) - y,
                ]),
                fmt="o", capsize=2.5, color=COLORS[model], label=DISPLAY[model],
            )
        ax.axhline(0, color="#333333", linewidth=0.8)
        readable_groups = {
            "T1": "T1",
            "T2": "T2",
            "T3": "T3",
            "T4": "T4",
            "cold_0": "0 interactions",
            "one_1": "1 interaction",
            "two_to_three_2_3": "2–3 interactions",
            "four_plus_4_plus": "4+ interactions",
            "unseen_0": "Unseen (0)",
            "rare_gt0_to_2": "Rare (>0–2)",
            "popular_gt2": "Popular (>2)",
        }
        ax.set_xticks(
            range(len(groups)),
            [readable_groups.get(value, value.replace("_", " ")) for value in groups],
        )
        ax.tick_params(axis="x", rotation=24)
        ax.set_title(dimension.capitalize())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Uniform-100 − sampling-free HitRate@20")
    axes[0].legend(frameon=False)
    fig.suptitle("JTH HitRate@20 differences across descriptive subgroups")
    fig.tight_layout()
    save_figure(fig, "figure_7_subgroup_evaluation_bias")
    stems.append("figure_7_subgroup_evaluation_bias")
    return stems


def publication_tables(
    known_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    rank_summary: pd.DataFrame,
    temporal: pd.DataFrame,
    subgroup_bootstrap: pd.DataFrame,
    evaluation_bias: pd.DataFrame,
    subgroup_evaluation_bias: pd.DataFrame,
) -> None:
    original = known_summary[
        known_summary.estimand.eq("original_all")
        & known_summary.strategy.eq("full")
        & known_summary.seed_aggregation.eq("individual")
    ].copy()
    original["model_label"] = original.model.map(DISPLAY)
    original["hitrate_rank"] = original.hitrate.rank(
        method="min", ascending=False
    ).astype(int)
    original[
        ["model", "model_label", "n_queries", *METRICS, "hitrate_rank"]
    ].sort_values(["hitrate_rank", "model"]).to_csv(
        OUT / "publication_table_1_full_catalog_k20.csv", index=False
    )

    bootstrap[
        bootstrap.strategy.eq("full")
        | bootstrap.seed_aggregation.eq("mean_over_5_seeds")
    ].sort_values(
        ["metric", "contrast", "strategy", "cluster_method"]
    ).to_csv(OUT / "publication_table_2_cluster_sensitivity.csv", index=False)

    known_summary[
        known_summary.estimand.isin(
            ["original_all", "original_matched_novel_cohort", "novel_item"]
        )
        & (
            known_summary.strategy.eq("full")
            | known_summary.seed_aggregation.eq("mean_over_5_seeds")
        )
    ].sort_values(["strategy", "model", "estimand"]).to_csv(
        OUT / "publication_table_3_known_item_sensitivity_k20.csv", index=False
    )

    rank_summary.to_csv(OUT / "publication_table_4_rank_fidelity_k20.csv", index=False)

    subgroup_bootstrap[
        subgroup_bootstrap.subgroup_dimension.eq("timestamp-rank quartile")
        & subgroup_bootstrap.cluster_method.eq("two_way_candidate_date")
    ].sort_values(["subgroup_order", "metric", "model"]).to_csv(
        OUT / "publication_table_5_timestamp_rank_quartiles.csv", index=False
    )

    subgroup_bootstrap[
        subgroup_bootstrap.metric.eq("hitrate")
        & subgroup_bootstrap.cluster_method.eq("two_way_candidate_date")
        & ~subgroup_bootstrap.subgroup_dimension.eq("timestamp-rank quartile")
    ].sort_values(["subgroup_dimension", "subgroup_order", "model"]).to_csv(
        OUT / "publication_table_6_subgroup_full_catalog_hitrate.csv", index=False
    )

    evaluation_bias[
        evaluation_bias.cluster_method.eq("two_way_candidate_date")
    ].sort_values(["metric", "strategy", "model", "candidate_size"]).to_csv(
        OUT / "publication_table_8_evaluation_bias_across_sizes.csv", index=False
    )

    subgroup_evaluation_bias[
        subgroup_evaluation_bias.cluster_method.eq("two_way_candidate_date")
    ].sort_values([
        "subgroup_dimension", "subgroup_order", "metric", "strategy",
        "model", "candidate_size",
    ]).to_csv(
        OUT / "publication_table_9_subgroup_evaluation_bias.csv", index=False
    )


def catalog_scale_summary() -> pd.DataFrame:
    """Summarize exact eligible catalogs and realized sampled-catalog fractions.

    The retained K=20 file repeats each query over models. This routine streams
    one parity-verified model and reduces it to one row per query/protocol draw,
    so catalog summaries do not accidentally model-weight the estimand.
    """
    source = INPUT_ROOTS[0] / "per_query_metrics.csv.gz"
    parts = []
    usecols = [
        "query_index", "eligible_catalog", "model", "strategy", "seed",
        "candidate_size", "k",
    ]
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=350_000):
        parts.append(chunk[chunk.k.eq(20) & chunk.model.eq("tfidf")].drop(columns=["k", "model"]))
    protocol = pd.concat(parts, ignore_index=True)
    if protocol.duplicated(
        ["query_index", "strategy", "seed", "candidate_size"]
    ).any():
        raise ValueError("Duplicate retained candidate-scale units")
    full = protocol[protocol.strategy.eq("full")].drop_duplicates("query_index")
    if len(full) != EXPECTED_QUERIES:
        raise ValueError(
            f"Expected {EXPECTED_QUERIES:,} unique full-catalog queries; found {len(full):,}"
        )

    rows = [{
        "candidate_protocol": "full_catalog",
        "strategy": "full",
        "nominal_candidate_size": -1,
        "n_queries": int(full.query_index.nunique()),
        "n_query_seed_units": int(len(full)),
        "eligible_catalog_mean": float(full.eligible_catalog.mean()),
        "eligible_catalog_median": float(full.eligible_catalog.median()),
        "eligible_catalog_min": int(full.eligible_catalog.min()),
        "eligible_catalog_max": int(full.eligible_catalog.max()),
        "actual_candidate_count_mean": float(full.eligible_catalog.mean()),
        "actual_candidate_count_median": float(full.eligible_catalog.median()),
        "realized_candidate_fraction_mean": 1.0,
        "realized_candidate_fraction_median": 1.0,
        "realized_candidate_fraction_min": 1.0,
        "realized_candidate_fraction_max": 1.0,
    }]

    sampled = protocol[protocol.strategy.ne("full")].copy()
    sampled["actual_candidate_size"] = np.minimum(
        sampled.candidate_size, sampled.eligible_catalog
    )
    if (sampled.actual_candidate_size != sampled.candidate_size).any():
        raise ValueError("A retained nominal candidate size exceeds an eligible catalog")
    sampled["realized_candidate_fraction"] = (
        sampled.actual_candidate_size / sampled.eligible_catalog
    )
    for (strategy, nominal), cell in sampled.groupby(
        ["strategy", "candidate_size"], sort=True
    ):
        expected_units = EXPECTED_QUERIES * int(cell.seed.nunique())
        if len(cell) != expected_units or int(cell.query_index.nunique()) != EXPECTED_QUERIES:
            raise ValueError(
                f"Incomplete candidate-scale cell {strategy}/{nominal}: "
                f"rows={len(cell):,}, queries={cell.query_index.nunique():,}"
            )
        rows.append({
            "candidate_protocol": "sampled",
            "strategy": str(strategy),
            "nominal_candidate_size": int(nominal),
            "n_queries": int(cell.query_index.nunique()),
            "n_query_seed_units": int(len(cell)),
            "eligible_catalog_mean": float(cell.eligible_catalog.mean()),
            "eligible_catalog_median": float(cell.eligible_catalog.median()),
            "eligible_catalog_min": int(cell.eligible_catalog.min()),
            "eligible_catalog_max": int(cell.eligible_catalog.max()),
            "actual_candidate_count_mean": float(cell.actual_candidate_size.mean()),
            "actual_candidate_count_median": float(cell.actual_candidate_size.median()),
            "realized_candidate_fraction_mean": float(
                cell.realized_candidate_fraction.mean()
            ),
            "realized_candidate_fraction_median": float(
                cell.realized_candidate_fraction.median()
            ),
            "realized_candidate_fraction_min": float(
                cell.realized_candidate_fraction.min()
            ),
            "realized_candidate_fraction_max": float(
                cell.realized_candidate_fraction.max()
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["candidate_protocol", "strategy", "nominal_candidate_size"]
    ).reset_index(drop=True)


def protocol_markdown() -> str:
    return """# Strengthened JTH analysis protocol

## Scope

This analysis applies only to the retained JTH experiment. It does not use or
rehabilitate any CareerBuilder result.

## Reproduction gate

Before any finding is written, all 642,257 original-condition K=20 rows for the
full catalog and 100-candidate protocols are joined one-to-one to the retained
production rows on query, model, sampler, seed, and candidate size. All six
metrics must agree within an absolute tolerance of 1e-12, and group, relevance
count, and eligible-catalog size must agree exactly.

## Dependence-aware bootstrap

The preselected contrasts are BM25 minus MiniLM and BM25 minus TF-IDF for
HitRate@20, NDCG@20, and MAP@20. They are evaluated for the full catalog and for
each of five seeds under uniform and popularity-weighted 100-candidate sampling.
The five-seed, per-query mean is also reported as a summary estimand.

Each interval uses 2,000 nonparametric replicates and preserves the query-weighted
mean:

- **Date cluster:** resample the 389 distinct query dates with replacement and
  retain every query on each selected date with its resampling multiplicity.
- **Candidate cluster:** resample the 6,398 candidates with replacement and
  retain every query for each selected candidate with its multiplicity.
- **Two-way candidate/date sensitivity:** independently resample candidate and
  date clusters; each query receives the product of its candidate and date
  multiplicities (a two-way pigeonhole cluster bootstrap). This is a dependence
  sensitivity analysis, not asserted to be a universally optimal small-sample
  correction. The product-weight construction follows the pigeonhole-style
  principle in Owen and Eckles (2012), *Bootstrapping data arrays of arbitrary
  order*, *Annals of Applied Statistics*, 6(3), 895--927,
  https://doi.org/10.1214/12-AOAS547.

Percentile 95% intervals, bootstrap standard errors, and the proportion of
replicates above zero are reported. No bootstrap probability is labelled a
p-value.

## Known-item sensitivity

A job seen for a candidate in the released training split is removed from that
candidate's test-time eligible catalog. Repeat targets are removed from the
relevance set; queries with no novel target are excluded. Scores are unchanged.
For sampled protocols, the original seeded negative order is filtered and then
continued until the requested pool is refilled, avoiding a new random draw.

Three aggregates separate effects: all original queries, original scores on the
novel-query cohort, and the novel-item estimand. The matched original cohort
separates catalog filtering from the effect of excluding repeat-only queries.

## Descriptive subgroup uncertainty

Full-catalog HitRate@20, NDCG@20, and MAP@20 are reported for BM25, MiniLM, and
TF-IDF with date-, candidate-, and two-way cluster intervals across the temporal
groups and both audited subgroup dimensions. The first non-temporal dimension
is named **training-interaction activity** and uses counts from the released
training interactions: 0 (cold), 1, 2--3, and 4+. It must not be described as a
static CV or work-history-depth measure. The second dimension is target popularity,
defined by the mean released-training count of a query's relevant jobs: exactly
0 (unseen), greater than 0 to 2, and greater than 2. Query, candidate, and date
counts accompany every group. These are descriptive heterogeneity results, not
fairness evidence and not causal effects.

## Rank and tie rules

Kendall's coefficient is explicitly SciPy 1.18.1 tau-b (`variant="b"`,
`method="auto"`); SciPy's native exact-value tie handling is retained for that
coefficient. For the separately reported winner sets and pairwise order states,
two values are tied under `numpy.isclose` with rtol=1e-10 and atol=1e-12. A winner
is therefore a set, not an alphabetically selected model. Exact winner agreement
requires identical sets; overlap requires a nonempty intersection. Pairwise order
agreement treats less-than, tie, and greater-than as three distinct states over
all 21 model pairs. Best-case and worst-case regret evaluate every sampled winner
under the full-catalog aggregate score.

## Time groups

T1--T4 are dense timestamp-rank quartiles: each of the 389 unique timestamps is
ranked chronologically and its rank fraction is binned at .25, .50, and .75.
They are not calendar quarters and are not equal-query quantiles.

## Evaluation-bias estimand across sizes

For HitRate@20 and MAP@20, each sampled query/model/sampler/size value is first
averaged over seeds 1--5. Its exact per-query full-catalog value is then
subtracted. The paired difference is summarized over all queries and within
timestamp-rank, training-interaction-activity, and target-popularity groups.
Date-cluster, candidate-cluster, and two-way candidate/date intervals use 2,000
replicates with seeds 20261220--20261222 (offset reproducibly by subgroup).
Subgroup results are descriptive; target popularity is outcome-derived.

## Interpretation limits

The released JTH entity text is static and has not been verified as an as-of-query
snapshot. Candidate sampling was applied after full model scores were available.
Consequently, computational statements concern only post-scoring candidate
ranking and metric calculation; they do not establish model-scoring or end-to-end
latency savings.
"""


def figure_source_map(stems: Iterable[str]) -> str:
    entries = {
        "figure_1_cluster_sensitivity_hitrate": (
            "primary_contrasts_cluster_bootstrap.csv",
            "BM25-minus-comparator HitRate@20 estimates and 95% percentile intervals under date-cluster, candidate-cluster, and two-way candidate/date pigeonhole-style bootstraps. Sampled estimates average five seeds per query before inference.",
        ),
        "figure_2_known_item_hitrate": (
            "known_item_summary_k20.csv",
            "Signed change in HitRate@20 from the matched-original to novel-item estimand for full-catalog, uniform-100, and popularity-weighted-100 evaluation. Sampled points use per-query five-seed means; the matched cohort isolates eligibility filtering from repeat-only query exclusion.",
        ),
        "figure_3_rank_fidelity_explicit_ties": (
            "rank_fidelity_summary_k20.csv",
            "Full-versus-sampled aggregate rank fidelity at K=20, averaged across six metrics and five seeds. Tau-b uses SciPy's native tie handling; pairwise and winner-set measures use the prespecified numerical tolerance.",
        ),
        "figure_4_timestamp_rank_quartiles": (
            "subgroup_full_catalog_cluster_bootstrap.csv",
            "Full-catalog HitRate@20 across dense timestamp-rank quartiles T1--T4, with two-way candidate/date cluster intervals. These labels are not calendar quarters or equal-query bins.",
        ),
        "figure_5_subgroup_hitrate_two_way": (
            "subgroup_full_catalog_cluster_bootstrap.csv",
            "Descriptive full-catalog HitRate@20 by training-interaction activity and target popularity, with two-way candidate/date cluster intervals. The plot is not a fairness or causal analysis.",
        ),
        "figure_6_evaluation_bias_across_sizes": (
            "evaluation_bias_cluster_bootstrap.csv",
            "Signed sampled-minus-full HitRate@20 and MAP@20 bias across every retained candidate size. Sampled values are averaged over all five seeds per query before paired inference; ribbons are two-way candidate/date cluster intervals.",
        ),
        "figure_7_subgroup_evaluation_bias": (
            "subgroup_evaluation_bias_cluster_bootstrap.csv",
            "Descriptive subgroup variation in uniform-100 HitRate@20 sampled-minus-full bias for BM25, MiniLM, and TF-IDF, with paired two-way candidate/date cluster intervals. Target popularity is outcome-derived; no causal or fairness claim is made.",
        ),
    }
    lines = ["# Figure source map", ""]
    for stem in stems:
        source, caption = entries[stem]
        lines.extend(
            [
                f"## `{stem}`",
                "",
                f"- Source: `{source}`",
                f"- Caption: {caption}",
                "",
            ]
        )
    lines.append(
        "All PNG files are rendered at 300 dpi. PDF and SVG versions preserve vector text and lines."
    )
    return "\n".join(lines) + "\n"


def verified_findings_markdown(
    parity: dict,
    metadata_audit: dict,
    producer_audit: dict,
    bootstrap: pd.DataFrame,
    known_summary: pd.DataFrame,
    known_ranks: pd.DataFrame,
    rank_summary: pd.DataFrame,
    subgroup_bootstrap: pd.DataFrame,
    subgroup_definitions: pd.DataFrame,
    evaluation_bias: pd.DataFrame,
    subgroup_evaluation_bias: pd.DataFrame,
    catalog_scale: pd.DataFrame,
) -> str:
    known_full = known_summary[
        known_summary.strategy.eq("full")
        & known_summary.seed_aggregation.eq("individual")
    ]
    original_full = known_full[known_full.estimand.eq("original_all")].set_index("model")
    novel_full = known_full[known_full.estimand.eq("novel_item")].set_index("model")
    original_winners = winner_set(
        original_full.loc[MODELS, "hitrate"].to_numpy(float), MODELS
    )
    novel_winners = winner_set(
        novel_full.loc[MODELS, "hitrate"].to_numpy(float), MODELS
    )
    matched = known_ranks[
        known_ranks.reference_estimand.eq("original_matched_novel_cohort")
        & known_ranks.strategy.eq("full")
        & known_ranks.metric.eq("hitrate")
    ].iloc[0]
    rank_lines = []
    for strategy in ["uniform", "popularity_weighted"]:
        cell = rank_summary[rank_summary.strategy.eq(strategy)].sort_values("candidate_size")
        smallest_rank = cell.iloc[0]
        largest_rank = cell.iloc[-1]
        label = "uniform" if strategy == "uniform" else "popularity-weighted"
        rank_lines.append(
            f"- {label}: at size {int(smallest_rank.candidate_size)}, mean tau-b "
            f"{smallest_rank.kendall_tau_b_mean:.3f}, pairwise agreement "
            f"{smallest_rank.pairwise_order_agreement_mean:.3f}, and winner-set "
            f"overlap {smallest_rank.winner_set_overlap_mean:.3f}; at size "
            f"{int(largest_rank.candidate_size)}, the values are "
            f"{largest_rank.kendall_tau_b_mean:.3f}, "
            f"{largest_rank.pairwise_order_agreement_mean:.3f}, and "
            f"{largest_rank.winner_set_overlap_mean:.3f}."
        )

    contrast_lines = []
    for contrast, label in [
        ("bm25_minus_semantic", "BM25 minus MiniLM"),
        ("bm25_minus_tfidf", "BM25 minus TF-IDF"),
    ]:
        protocol_results = bootstrap[
            bootstrap.metric.eq("hitrate")
            & bootstrap.contrast.eq(contrast)
            & bootstrap.cluster_method.eq("two_way_candidate_date")
            & (
                bootstrap.strategy.eq("full")
                | bootstrap.seed_aggregation.eq("mean_over_5_seeds")
            )
        ]
        for strategy, protocol_label in [
            ("full", "full catalog"),
            ("uniform", "uniform 100, five-seed query mean"),
            (
                "popularity_weighted",
                "popularity-weighted 100, five-seed query mean",
            ),
        ]:
            row = protocol_results[protocol_results.strategy.eq(strategy)].iloc[0]
            contrast_lines.append(
                f"- {label}, {protocol_label}: {row.estimate:.6f} "
                f"(two-way 95% interval {row.ci_low:.6f} to {row.ci_high:.6f}; "
                f"bootstrap Pr[difference > 0] = {row.bootstrap_probability_positive:.3f})."
            )

    known_matched_rank = known_ranks[
        known_ranks.reference_estimand.eq("original_matched_novel_cohort")
    ]
    winner_set_changes = int(
        (known_matched_rank.winner_set_exact_agreement == 0).sum()
    )
    pairwise_order_changes = int(
        (known_matched_rank.pairwise_order_agreement < 1.0).sum()
    )
    matched_summary = known_summary[
        known_summary.estimand.eq("original_matched_novel_cohort")
    ]
    novel_summary = known_summary[known_summary.estimand.eq("novel_item")]
    summary_keys = [
        "model",
        "strategy",
        "seed",
        "candidate_size",
        "seed_aggregation",
    ]
    known_delta = novel_summary.merge(
        matched_summary,
        on=summary_keys,
        suffixes=("_novel", "_matched"),
        validate="one_to_one",
    )
    max_absolute_hitrate_change = float(
        (known_delta.hitrate_novel - known_delta.hitrate_matched).abs().max()
    )

    subgroup_lines = []
    subgroup_display = {
        "cold_0": "0 (cold)",
        "one_1": "1",
        "two_to_three_2_3": "2--3",
        "four_plus_4_plus": "4+",
        "unseen_0": "0 (unseen)",
        "rare_gt0_to_2": ">0--2",
        "popular_gt2": ">2",
    }
    subgroup_hr = subgroup_bootstrap[
        subgroup_bootstrap.metric.eq("hitrate")
        & subgroup_bootstrap.cluster_method.eq("two_way_candidate_date")
    ]
    for dimension in ["training-interaction activity", "target popularity"]:
        definitions = subgroup_definitions[
            subgroup_definitions.subgroup_dimension.eq(dimension)
        ].sort_values("subgroup_order")
        count_text = ", ".join(
            f"{subgroup_display[row.subgroup_value]}: {int(row.n_queries):,} queries"
            for row in definitions.itertuples(index=False)
        )
        model_text = []
        for model in ["bm25", "semantic", "tfidf"]:
            cell = subgroup_hr[
                subgroup_hr.subgroup_dimension.eq(dimension)
                & subgroup_hr.model.eq(model)
            ]
            model_text.append(
                f"{DISPLAY[model]} {cell.estimate.min():.6f}--{cell.estimate.max():.6f}"
            )
        subgroup_lines.append(
            f"- {dimension}: {count_text}. Across groups, full-catalog HitRate@20 "
            f"ranges were {', '.join(model_text)}."
        )

    temporal_lines = []
    temporal_hr = subgroup_bootstrap[
        subgroup_bootstrap.subgroup_dimension.eq("timestamp-rank quartile")
        & subgroup_bootstrap.metric.eq("hitrate")
        & subgroup_bootstrap.cluster_method.eq("two_way_candidate_date")
    ]
    for group in ["T1", "T2", "T3", "T4"]:
        cell = temporal_hr[temporal_hr.subgroup_value.eq(group)].set_index("model")
        entries = []
        for model in ["bm25", "semantic", "tfidf"]:
            row = cell.loc[model]
            entries.append(
                f"{DISPLAY[model]} {row.estimate:.6f} "
                f"[{row.ci_low:.6f}, {row.ci_high:.6f}]"
            )
        temporal_lines.append(f"- {group}: " + "; ".join(entries) + ".")

    bias_lines = []
    bias_two_way = evaluation_bias[
        evaluation_bias.cluster_method.eq("two_way_candidate_date")
    ]
    for strategy in ["uniform", "popularity_weighted"]:
        for metric in BIAS_METRICS:
            entries = []
            for model in ["bm25", "semantic", "tfidf"]:
                endpoints = bias_two_way[
                    bias_two_way.strategy.eq(strategy)
                    & bias_two_way.metric.eq(metric)
                    & bias_two_way.model.eq(model)
                    & bias_two_way.candidate_size.isin([100, 5000])
                ].set_index("candidate_size")
                entries.append(
                    f"{DISPLAY[model]} {endpoints.loc[100, 'estimate']:.6f} at 100 "
                    f"and {endpoints.loc[5000, 'estimate']:.6f} at 5,000"
                )
            bias_lines.append(
                f"- {strategy.replace('_', '-')} {metric.title()}@20: "
                + "; ".join(entries) + "."
            )

    subgroup_bias_lines = []
    subgroup_bias_cell = subgroup_evaluation_bias[
        subgroup_evaluation_bias.cluster_method.eq("two_way_candidate_date")
        & subgroup_evaluation_bias.metric.eq("hitrate")
        & subgroup_evaluation_bias.strategy.eq("uniform")
        & subgroup_evaluation_bias.candidate_size.eq(100)
    ]
    for dimension in [
        "timestamp-rank quartile", "training-interaction activity",
        "target popularity",
    ]:
        entries = []
        for model in ["bm25", "semantic", "tfidf"]:
            cell = subgroup_bias_cell[
                subgroup_bias_cell.subgroup_dimension.eq(dimension)
                & subgroup_bias_cell.model.eq(model)
            ]
            entries.append(
                f"{DISPLAY[model]} {cell.estimate.min():.6f}--{cell.estimate.max():.6f}"
            )
        subgroup_bias_lines.append(
            f"- {dimension}: " + "; ".join(entries) + "."
        )

    full_scale = catalog_scale[catalog_scale.strategy.eq("full")].iloc[0]
    sampled_scale = catalog_scale[catalog_scale.candidate_protocol.eq("sampled")]
    fraction_lines = []
    for size in BIAS_SIZES:
        value = sampled_scale[
            sampled_scale.nominal_candidate_size.eq(size)
        ].realized_candidate_fraction_mean.mean()
        fraction_lines.append(f"{size:,}: {value:.4f}")

    return f"""# Verified statistical findings: strengthened JTH branch

## Integrity gate

The regenerated original condition reproduced all {parity['rows_compared']:,}
retained full-catalog and 100-candidate K=20 rows. Across precision, recall,
HitRate, NDCG, MAP, and MRR, the maximum absolute discrepancy was
{max(v['max_absolute_difference'] for v in parity['metrics'].values()):.3g}; no
value exceeded the prespecified {PARITY_ATOL:g} tolerance. Structural metadata
also matched exactly. This gate passed before the findings below were written.

## Dependence-aware primary contrasts

{chr(10).join(contrast_lines)}

The complete table reports date-cluster, candidate-cluster, and two-way
candidate/date sensitivity intervals for the full catalog, every sampled seed,
and the per-query five-seed sampled mean. These intervals address repeated dates
and/or repeated candidates; the earlier date-only interval should not be treated
as the sole inferential result.

## Known-item and repeat-target sensitivity

The audit found {metadata_audit['overlap_unique_candidate_job_pairs']} unique
training/test candidate-job overlaps, representing
{metadata_audit['overlap_target_events']} target events in
{metadata_audit['overlap_queries']} queries. All affected queries were repeat-only,
so the novel-item estimand contains {producer_audit['novel_item_queries']:,} of
{producer_audit['evaluated_queries']:,} original queries and excludes
{producer_audit['repeat_only_queries_excluded']} queries. The full-catalog
HitRate@20 winner set is
`{'|'.join(DISPLAY[model] for model in original_winners)}` in the original
estimand and `{'|'.join(DISPLAY[model] for model in novel_winners)}` in the
novel-item estimand. On the matched query cohort, the explicit winner-set comparison is
`{matched.reference_winner_set}` versus `{matched.comparison_winner_set}`;
winner-set overlap is {int(matched.winner_set_overlap)} and pairwise-order
agreement is {matched.pairwise_order_agreement:.3f}.
Across all {len(known_matched_rank)} model-ranking decisions (six metrics over
the full and sampled protocols), the winner set changed in
{winner_set_changes} cells. Pairwise ordering changed in
{pairwise_order_changes} cells, and the maximum absolute HitRate@20 change over
all model/protocol cells was {max_absolute_hitrate_change:.6f}. Thus the
known-item restriction is a robustness check, not the source of the much larger
full-versus-sampled winner reversal above.

## Rank fidelity with explicit ties

Across six metrics and five seeds:

{chr(10).join(rank_lines)}

Both best- and worst-case winner regret are in
`rank_fidelity_summary_k20.csv`.

## Descriptive subgroup uncertainty

{chr(10).join(subgroup_lines)}

Every subgroup row reports query, candidate, and date counts together with date-,
candidate-, and two-way cluster intervals. These patterns are descriptive only;
they are not fairness findings and do not identify causal effects. The label
`training-interaction activity` is the released training interaction count, not
CV or work-history depth. The target-popularity analysis includes `unseen_0`
explicitly.

## Corrected time-group label and scope boundary

T1--T4 are dense timestamp-rank quartiles over
{metadata_audit['unique_timestamps']} unique timestamps—not calendar quarters and
not equal-query groups. Descriptive full-catalog HitRate@20 estimates with two-way
candidate/date 95% intervals are:

{chr(10).join(temporal_lines)}

The static profile text is not verified as an as-of-query
snapshot. Candidate sampling here is post-scoring, so no result supports a claim
of reduced model-scoring or end-to-end computational cost.

## Evaluation-bias estimand across sizes

The exact eligible catalog has mean {full_scale.eligible_catalog_mean:.2f},
median {full_scale.eligible_catalog_median:.0f}, and range
{int(full_scale.eligible_catalog_min):,}--{int(full_scale.eligible_catalog_max):,}
jobs. Mean realized candidate fractions by nominal size are
{'; '.join(fraction_lines)}.

Each sampled query is averaged over all five seeds before subtracting its exact
full-catalog value. Two-way cluster point estimates at the smallest and largest
candidate sizes are:

{chr(10).join(bias_lines)}

At uniform size 100, the ranges of HitRate@20 bias across the descriptive groups
are:

{chr(10).join(subgroup_bias_lines)}

Complete date-, candidate-, and two-way cluster intervals are in
`evaluation_bias_cluster_bootstrap.csv` and
`subgroup_evaluation_bias_cluster_bootstrap.csv`. All subgroup cells exceed 30
candidate and 30 date clusters, but the patterns remain descriptive. Target
popularity is outcome-derived and cannot be read as an ex-ante or causal stratum.
"""


def main() -> None:
    if scipy.__version__ != "1.18.1":
        raise RuntimeError(
            f"This audited run requires SciPy 1.18.1; found {scipy.__version__}"
        )
    known_path, producer_audit_path, source_input_manifest = initialize_clean_output()

    qmeta, metadata_audit = load_query_metadata()
    retained, temporal, retained_audit = load_retained_k20(qmeta)
    evaluation_bias_wide, evaluation_bias_matrix_audit = load_evaluation_bias_matrix()
    known = pd.read_csv(known_path)
    producer_audit = json.loads(producer_audit_path.read_text(encoding="utf-8"))

    expected_total = EXPECTED_ORIGINAL_ROWS + EXPECTED_NOVEL_ROWS
    if len(known) != expected_total:
        raise ValueError(
            f"Known-item row count {len(known):,} != expected {expected_total:,}"
        )
    if int(known.isna().sum().sum()) != 0:
        raise ValueError("Known-item per-query file contains missing values")
    if sorted(known.model.unique()) != MODELS:
        raise ValueError("Known-item output has unexpected models")

    parity = parity_check(known, retained)
    known_summary, known_differences, novel_ids = known_item_summaries(known)
    if len(novel_ids) != 8_323:
        raise ValueError(f"Unexpected novel query count: {len(novel_ids):,}")
    if producer_audit["train_test_overlap_target_events"] != 18:
        raise ValueError("Producer overlap target-event count changed")
    excluded_query_ids = set(map(int, qmeta.query_index)).difference(novel_ids)
    documented_overlap_ids = set(metadata_audit["overlap_query_indices"])
    if excluded_query_ids != documented_overlap_ids:
        raise ValueError(
            "Novel-item exclusions do not equal the independently audited "
            "repeat-target query set"
        )
    overlap_fields = {
        "train_test_overlap_unique_candidate_job_pairs": "overlap_unique_candidate_job_pairs",
        "train_test_overlap_target_events": "overlap_target_events",
        "train_test_overlap_queries": "overlap_queries",
    }
    for producer_field, independent_field in overlap_fields.items():
        if int(producer_audit[producer_field]) != int(metadata_audit[independent_field]):
            raise ValueError(
                f"Overlap audit mismatch: {producer_field} versus {independent_field}"
            )
    if int(producer_audit["original_target_events"]) - int(
        producer_audit["novel_target_events"]
    ) != metadata_audit["overlap_target_events"]:
        raise ValueError("Novel relevance-event subtraction does not equal overlap events")

    bootstrap = bootstrap_primary_contrasts(known, qmeta)
    subgroup_bootstrap, subgroup_definitions = bootstrap_full_catalog_subgroups(
        retained, qmeta
    )
    catalog_scale = catalog_scale_summary()
    evaluation_bias, subgroup_evaluation_bias, evaluation_bias_seed_detail = (
        bootstrap_evaluation_bias(evaluation_bias_wide, qmeta)
    )
    ranks, rank_summary, rank_metric_summary = regenerate_rank_tables()
    known_ranks = known_item_rank_table(known_summary)
    time_definitions = temporal_definitions(qmeta)

    # Complete analysis outputs.
    bootstrap.to_csv(OUT / "primary_contrasts_cluster_bootstrap.csv", index=False)
    bootstrap[
        bootstrap.strategy.eq("full")
        | bootstrap.seed_aggregation.eq("mean_over_5_seeds")
    ].to_csv(OUT / "primary_contrasts_cluster_summary.csv", index=False)
    known_summary.to_csv(OUT / "known_item_summary_k20.csv", index=False)
    known_differences.to_csv(OUT / "known_item_protocol_differences.csv", index=False)
    known_ranks.to_csv(OUT / "known_item_rank_sensitivity_k20.csv", index=False)
    ranks.to_csv(OUT / "rank_fidelity_explicit_ties.csv", index=False)
    rank_summary.to_csv(OUT / "rank_fidelity_summary_k20.csv", index=False)
    rank_metric_summary.to_csv(OUT / "rank_fidelity_metric_summary_k20.csv", index=False)
    time_definitions.to_csv(OUT / "temporal_rank_quartile_definitions.csv", index=False)
    temporal.to_csv(OUT / "temporal_rank_quartile_summary_k20.csv", index=False)
    subgroup_bootstrap.to_csv(
        OUT / "subgroup_full_catalog_cluster_bootstrap.csv", index=False
    )
    subgroup_definitions.to_csv(OUT / "subgroup_definitions_and_counts.csv", index=False)
    catalog_scale.to_csv(OUT / "catalog_size_and_candidate_fraction.csv", index=False)
    catalog_scale.to_csv(OUT / "publication_table_7_candidate_scale.csv", index=False)
    evaluation_bias.to_csv(
        OUT / "evaluation_bias_cluster_bootstrap.csv", index=False
    )
    subgroup_evaluation_bias.to_csv(
        OUT / "subgroup_evaluation_bias_cluster_bootstrap.csv", index=False
    )
    evaluation_bias_seed_detail.to_csv(
        OUT / "evaluation_bias_seed_detail.csv", index=False
    )

    publication_tables(
        known_summary, bootstrap, rank_summary, temporal, subgroup_bootstrap,
        evaluation_bias, subgroup_evaluation_bias,
    )
    stems = make_figures(
        bootstrap, known_summary, rank_summary, temporal, subgroup_bootstrap,
        evaluation_bias, subgroup_evaluation_bias,
    )
    (OUT / "STUDY_PROTOCOL.md").write_text(protocol_markdown(), encoding="utf-8")
    (OUT / "FIGURE_SOURCE_MAP.md").write_text(
        figure_source_map(stems), encoding="utf-8"
    )

    analysis_frames = {
        "bootstrap": bootstrap,
        "subgroup_bootstrap": subgroup_bootstrap,
        "known_summary": known_summary,
        "known_differences": known_differences,
        "known_ranks": known_ranks,
        "rank_fidelity": ranks,
        "rank_summary": rank_summary,
        "temporal": temporal,
        "time_definitions": time_definitions,
        "catalog_scale": catalog_scale,
        "evaluation_bias": evaluation_bias,
        "subgroup_evaluation_bias": subgroup_evaluation_bias,
        "evaluation_bias_seed_detail": evaluation_bias_seed_detail,
    }
    missing_by_frame = {
        name: int(frame.isna().sum().sum())
        for name, frame in analysis_frames.items()
    }
    primary_interval_violations = int(
        (
            (bootstrap.ci_low > bootstrap.ci_high)
            | (bootstrap.estimate < bootstrap.ci_low)
            | (bootstrap.estimate > bootstrap.ci_high)
        ).sum()
    )
    subgroup_interval_violations = int(
        (
            (subgroup_bootstrap.ci_low > subgroup_bootstrap.ci_high)
            | (subgroup_bootstrap.estimate < subgroup_bootstrap.ci_low)
            | (subgroup_bootstrap.estimate > subgroup_bootstrap.ci_high)
        ).sum()
    )
    evaluation_bias_interval_violations = int((
        (evaluation_bias.ci_low > evaluation_bias.ci_high)
        | (evaluation_bias.estimate < evaluation_bias.ci_low)
        | (evaluation_bias.estimate > evaluation_bias.ci_high)
    ).sum())
    subgroup_bias_interval_violations = int((
        (subgroup_evaluation_bias.ci_low > subgroup_evaluation_bias.ci_high)
        | (subgroup_evaluation_bias.estimate < subgroup_evaluation_bias.ci_low)
        | (subgroup_evaluation_bias.estimate > subgroup_evaluation_bias.ci_high)
    ).sum())
    subgroup_query_sums = (
        subgroup_definitions.groupby("subgroup_dimension").n_queries.sum().astype(int).to_dict()
    )
    figure_variants_complete = all(
        (OUT / f"{stem}.{extension}").exists()
        and (OUT / f"{stem}.{extension}").stat().st_size > 0
        for stem in stems
        for extension in ["png", "pdf", "svg"]
    )
    if (
        any(missing_by_frame.values())
        or primary_interval_violations
        or subgroup_interval_violations
        or evaluation_bias_interval_violations
        or subgroup_bias_interval_violations
        or set(subgroup_query_sums.values()) != {len(qmeta)}
        or not figure_variants_complete
    ):
        raise ValueError("Post-analysis integrity checks failed")
    post_analysis_checks = {
        "missing_values_by_frame": missing_by_frame,
        "primary_interval_violations": primary_interval_violations,
        "subgroup_interval_violations": subgroup_interval_violations,
        "evaluation_bias_interval_violations": evaluation_bias_interval_violations,
        "subgroup_bias_interval_violations": subgroup_bias_interval_violations,
        "subgroup_query_sums": subgroup_query_sums,
        "rank_tau_b_range": [float(ranks.kendall_tau_b.min()), float(ranks.kendall_tau_b.max())],
        "rank_pairwise_agreement_range": [
            float(ranks.pairwise_order_agreement.min()),
            float(ranks.pairwise_order_agreement.max()),
        ],
        "all_figure_variants_nonempty": figure_variants_complete,
    }

    output_paths = [
        path
        for path in OUT.iterdir()
        if path.is_file() and path.name not in {"integrity_checks.json", "VERIFIED_FINDINGS.md"}
    ]
    integrity = {
        "status": "passed",
        "scope": "JTH only; CareerBuilder not read",
        "input_files_sha256": {
            str(path / "per_query_metrics.csv.gz"): sha256(path / "per_query_metrics.csv.gz")
            for path in INPUT_ROOTS
        },
        "known_item_output_sha256": sha256(known_path),
        "source_input_manifest": source_input_manifest,
        "analysis_code_sha256": sha256(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "pythonpath_resolution_order": os.environ.get("PYTHONPATH", "").split(";"),
            "module_paths": {
                "numpy": str(Path(np.__file__).resolve()),
                "pandas": str(Path(pd.__file__).resolve()),
                "scipy": str(Path(scipy.__file__).resolve()),
                "matplotlib": str(Path(matplotlib.__file__).resolve()),
            },
            "environment_note": (
                ".semantic_packages is first so the working NumPy and SciPy "
                "binaries resolve there; .python_packages is second to provide "
                "pandas and matplotlib, which are absent from the first path. "
                "The recorded module paths are authoritative."
            ),
        },
        "retained": retained_audit,
        "query_metadata": metadata_audit,
        "producer_audit": producer_audit,
        "parity": parity,
        "bootstrap": {
            "replicates": BOOTSTRAPS,
            "seeds": BOOTSTRAP_SEEDS,
            "rows": int(len(bootstrap)),
            "methods": sorted(bootstrap.cluster_method.unique()),
            "algorithms": {
                "date_cluster": "multinomial resampling of all date clusters; query-count denominator",
                "candidate_cluster": "multinomial resampling of all candidate clusters; query-count denominator",
                "two_way_candidate_date": "independent multinomial candidate/date multiplicities; product weight per query",
            },
        },
        "subgroup_bootstrap": {
            "scope": "full-catalog K=20 BM25, MiniLM, and TF-IDF effectiveness",
            "replicates": BOOTSTRAPS,
            "seeds": SUBGROUP_BOOTSTRAP_SEEDS,
            "rows": int(len(subgroup_bootstrap)),
            "definitions": subgroup_definitions.to_dict(orient="records"),
            "interpretation": "descriptive performance heterogeneity; not fairness or causal inference",
        },
        "evaluation_bias": {
            "matrix": evaluation_bias_matrix_audit,
            "definition": (
                "per-query arithmetic mean over seeds 1--5 minus exact "
                "per-query full-catalog metric"
            ),
            "metrics": BIAS_METRICS,
            "candidate_sizes": BIAS_SIZES,
            "replicates": BOOTSTRAPS,
            "seeds": BIAS_BOOTSTRAP_SEEDS,
            "overall_rows": int(len(evaluation_bias)),
            "subgroup_rows": int(len(subgroup_evaluation_bias)),
            "subgroup_interpretation": (
                "descriptive only; target popularity is outcome-derived"
            ),
        },
        "post_analysis_checks": post_analysis_checks,
        "rank_rules": {
            "kendall": "scipy.stats.kendalltau variant=b, method=auto; native exact-value ties; SciPy 1.18.1 expected",
            "tie_rtol": TIE_RTOL,
            "tie_atol": TIE_ATOL,
            "winner": "set of every model tied with the maximum",
            "pairwise": "exact agreement over {-1, 0, +1}; tie is a third state",
            "regret": "full maximum minus best/worst full score among sampled winner set",
        },
        "row_counts": {
            "known_item_total": int(len(known)),
            "known_item_original": int(known.estimand.eq("original").sum()),
            "known_item_novel": int(known.estimand.eq("novel_item").sum()),
            "known_summary": int(len(known_summary)),
            "rank_fidelity": int(len(ranks)),
            "temporal_summary": int(len(temporal)),
            "subgroup_bootstrap": int(len(subgroup_bootstrap)),
            "candidate_scale_summary": int(len(catalog_scale)),
            "evaluation_bias": int(len(evaluation_bias)),
            "subgroup_evaluation_bias": int(len(subgroup_evaluation_bias)),
            "evaluation_bias_seed_detail": int(len(evaluation_bias_seed_detail)),
        },
        "time_group_labels": "T1--T4 dense timestamp-rank quartiles",
        "rq5_boundary": (
            "Only post-scoring candidate ranking and metric work was varied; no "
            "model-scoring or end-to-end speedup was measured."
        ),
        "static_profile_limitation": (
            "JTH entity text is static and is not verified as an as-of-query snapshot."
        ),
        "generated_file_sha256": {path.name: sha256(path) for path in output_paths},
    }
    write_json(OUT / "integrity_checks.json", integrity)

    # This is deliberately last: no verified narrative exists if any check above fails.
    findings = verified_findings_markdown(
        parity,
        metadata_audit,
        producer_audit,
        bootstrap,
        known_summary,
        known_ranks,
        rank_summary,
        subgroup_bootstrap,
        subgroup_definitions,
        evaluation_bias,
        subgroup_evaluation_bias,
        catalog_scale,
    )
    findings_path = OUT / "VERIFIED_FINDINGS.md"
    findings_path.write_text(findings, encoding="utf-8")
    completion = {
        "status": "passed",
        "scope": "authoritative strengthened JTH v2 result package",
        "output_directory": str(OUT.resolve()),
        "analysis_code_sha256": sha256(Path(__file__)),
        "integrity_checks_sha256": sha256(OUT / "integrity_checks.json"),
        "verified_findings_sha256": sha256(findings_path),
        "required_files_present": all(
            (OUT / name).exists()
            for name in [
                "SOURCE_INPUT_MANIFEST.json",
                "integrity_checks.json",
                "VERIFIED_FINDINGS.md",
                "evaluation_bias_cluster_bootstrap.csv",
                "subgroup_evaluation_bias_cluster_bootstrap.csv",
                "publication_table_8_evaluation_bias_across_sizes.csv",
                "publication_table_9_subgroup_evaluation_bias.csv",
            ]
        ),
        "interpretation": (
            "This sentinel is written last. Its absence means the package is "
            "incomplete and non-citable even if earlier files exist."
        ),
    }
    if not completion["required_files_present"]:
        raise FileNotFoundError("Required v2 result-package file is missing")
    write_json(OUT / "AUTHORITATIVE_COMPLETE.json", completion)
    print(json.dumps({
        "status": "passed",
        "parity": parity["status"],
        "known_rows": len(known),
        "bootstrap_rows": len(bootstrap),
        "rank_rows": len(ranks),
        "subgroup_bootstrap_rows": len(subgroup_bootstrap),
        "figures": stems,
    }, indent=2))


if __name__ == "__main__":
    main()
