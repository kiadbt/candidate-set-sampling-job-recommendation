#!/usr/bin/env python3
"""Field-matched BM25/MiniLM query-level transition sensitivity.

The authoritative CareerBuilder checkpoints retain one row per user, model,
candidate condition, and cutoff.  This post-decision analysis compares title
BM25 with title MiniLM at HitRate@20, where both systems use the job-title
field.  It reports how a user's four-state hit pattern changes from the exact
eligible catalog to each sampled condition and summarizes the first relevant
rank recoverable from MRR@20.

This script does not rescore recommendations or modify authoritative outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("bm25_title_static", "minilm_title_static")
EXPECTED_STRATEGIES = ("uniform", "popularity_weighted")
EXPECTED_SIZES = (100, 500, 1000, 5000, 10000)
EXPECTED_SEEDS = (1, 2, 3, 4, 5)
MODEL_LABELS = {
    "bm25_title_static": "Title BM25",
    "minilm_title_static": "Title MiniLM",
}
STATE_ORDER = ("neither_hits", "bm25_only", "minilm_only", "both_hit")
USECOLS = (
    "window",
    "user_id",
    "population_weight",
    "candidate_regime",
    "sampling_strategy",
    "seed",
    "requested_candidate_size",
    "model",
    "k",
    "hit_rate",
    "mrr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_relevant_rows(checkpoint_root: Path, k: int) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    files = sorted(checkpoint_root.glob("window_*/users_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No checkpoint files under {checkpoint_root}")
    pieces: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    for path in files:
        frame = pd.read_csv(path, usecols=list(USECOLS))
        frame = frame.loc[frame["model"].isin(MODELS) & frame["k"].eq(k)].copy()
        if frame.empty:
            raise ValueError(f"No requested model/K rows in {path}")
        pieces.append(frame)
        records.append(
            {
                "file": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "selected_rows": int(len(frame)),
            }
        )
    data = pd.concat(pieces, ignore_index=True)
    return data, records


def hit_state(bm25: pd.Series, minilm: pd.Series) -> pd.Series:
    b = bm25.gt(0.0)
    m = minilm.gt(0.0)
    return pd.Series(
        np.select(
            (b & m, b & ~m, ~b & m),
            ("both_hit", "bm25_only", "minilm_only"),
            default="neither_hits",
        ),
        index=bm25.index,
    )


def weighted_quantile(values: pd.Series, weights: pd.Series, q: float) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return float("nan")
    v = values.loc[mask].to_numpy(dtype=float)
    w = weights.loc[mask].to_numpy(dtype=float)
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    threshold = q * w.sum()
    return float(v[np.searchsorted(np.cumsum(w), threshold, side="left")])


def validate_and_pivot(data: pd.DataFrame, regime: str) -> pd.DataFrame:
    keys = [
        "window",
        "user_id",
        "population_weight",
        "candidate_regime",
        "sampling_strategy",
        "seed",
        "requested_candidate_size",
    ]
    subset = data.loc[data["candidate_regime"].eq(regime)].copy()
    duplicated = subset.duplicated(keys + ["model"], keep=False)
    if duplicated.any():
        raise ValueError(f"Duplicate model rows in {regime} data")
    expected = subset.groupby(keys, dropna=False)["model"].nunique()
    if not expected.eq(len(MODELS)).all():
        raise ValueError(f"Incomplete BM25/MiniLM pairs in {regime} data")
    wide = subset.pivot(index=keys, columns="model", values=["hit_rate", "mrr"]).reset_index()
    wide.columns = [
        column if isinstance(column, str) else "__".join(str(value) for value in column if value)
        for column in wide.columns
    ]
    return wide


def add_state(wide: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = wide.copy()
    bm25_hit = result["hit_rate__bm25_title_static"]
    minilm_hit = result["hit_rate__minilm_title_static"]
    result[f"{prefix}_state"] = hit_state(bm25_hit, minilm_hit)
    return result


def validate_primary_grid(full: pd.DataFrame, sampled: pd.DataFrame, k: int) -> None:
    """Fail closed unless the retained rows form the frozen primary grid."""
    if full["user_id"].nunique() != 7000 or len(full) != 7000:
        raise ValueError("Full-catalog rows do not contain exactly 7,000 distinct users")
    if set(sampled["sampling_strategy"].unique()) != set(EXPECTED_STRATEGIES):
        raise ValueError("Unexpected sampled strategies")
    if set(sampled["requested_candidate_size"].astype(int).unique()) != set(EXPECTED_SIZES):
        raise ValueError("Unexpected sampled candidate-size grid")
    if set(sampled["seed"].astype(int).unique()) != set(EXPECTED_SEEDS):
        raise ValueError("Unexpected sampled seed grid")
    expected_rows = 7000 * len(EXPECTED_STRATEGIES) * len(EXPECTED_SIZES) * len(EXPECTED_SEEDS)
    if len(sampled) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} sampled user-conditions; found {len(sampled):,}")
    conditions_per_user = sampled.groupby(["window", "user_id"]).size()
    expected_conditions = len(EXPECTED_STRATEGIES) * len(EXPECTED_SIZES) * len(EXPECTED_SEEDS)
    if not conditions_per_user.eq(expected_conditions).all():
        raise ValueError("At least one user lacks a sampled strategy/size/seed condition")
    if not np.isfinite(sampled["population_weight"]).all() or not sampled["population_weight"].gt(0).all():
        raise ValueError("Population weights must be finite and positive")

    for label, wide in (("full", full), ("sampled", sampled)):
        for model in MODELS:
            hit = wide[f"hit_rate__{model}"]
            mrr = wide[f"mrr__{model}"]
            if not hit.isin((0.0, 1.0)).all():
                raise ValueError(f"{label} HitRate is not binary for {model}")
            if not mrr.loc[hit.eq(0.0)].eq(0.0).all():
                raise ValueError(f"{label} misses have nonzero MRR for {model}")
            positive = mrr.loc[hit.eq(1.0)]
            if not positive.gt(0.0).all():
                raise ValueError(f"{label} hits have nonpositive MRR for {model}")
            ranks = 1.0 / positive
            if not (
                ranks.between(1.0, float(k)).all()
                and np.allclose(ranks, np.rint(ranks), rtol=1e-8, atol=1e-8)
            ):
                raise ValueError(f"{label} MRR does not encode an integer first-relevant rank for {model}")


def transition_detail(full: pd.DataFrame, sampled: pd.DataFrame) -> pd.DataFrame:
    full_columns = [
        "window",
        "user_id",
        "population_weight",
        "full_state",
        "hit_rate__bm25_title_static",
        "hit_rate__minilm_title_static",
        "mrr__bm25_title_static",
        "mrr__minilm_title_static",
    ]
    renamed = full[full_columns].rename(
        columns={
            "population_weight": "full_population_weight",
            "hit_rate__bm25_title_static": "full_bm25_hit",
            "hit_rate__minilm_title_static": "full_minilm_hit",
            "mrr__bm25_title_static": "full_bm25_mrr",
            "mrr__minilm_title_static": "full_minilm_mrr",
        }
    )
    merged = sampled.merge(renamed, on=["window", "user_id"], how="left", validate="many_to_one")
    if merged["full_state"].isna().any():
        raise ValueError("Sampled rows lack full-catalog references")
    if not np.allclose(merged["population_weight"], merged["full_population_weight"]):
        raise ValueError("Population weights differ between full and sampled rows")
    merged["transition"] = merged["full_state"] + " -> " + merged["sampled_state"]
    return merged


def summarize_transitions(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = ["sampling_strategy", "requested_candidate_size", "seed"]
    rows: list[dict[str, object]] = []
    for condition, local in detail.groupby(group, dropna=False, sort=True):
        total_weight = float(local["population_weight"].sum())
        for full_state in STATE_ORDER:
            for sampled_state in STATE_ORDER:
                selected = local["full_state"].eq(full_state) & local["sampled_state"].eq(sampled_state)
                rows.append(
                    {
                        **dict(zip(group, condition)),
                        "full_state": full_state,
                        "sampled_state": sampled_state,
                        "users_n": int(selected.sum()),
                        "weighted_users": float(local.loc[selected, "population_weight"].sum()),
                        "weighted_share": (
                            float(local.loc[selected, "population_weight"].sum()) / total_weight
                            if total_weight else float("nan")
                        ),
                        "total_users": int(len(local)),
                        "total_weight": total_weight,
                    }
                )
    by_seed = pd.DataFrame(rows)
    summary = (
        by_seed.groupby(
            ["sampling_strategy", "requested_candidate_size", "full_state", "sampled_state"],
            as_index=False,
            dropna=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            weighted_share_mean=("weighted_share", "mean"),
            weighted_share_sd=("weighted_share", "std"),
            weighted_share_min=("weighted_share", "min"),
            weighted_share_max=("weighted_share", "max"),
            users_n_mean=("users_n", "mean"),
        )
    )
    return by_seed, summary


def rank_rows(wide: pd.DataFrame, condition: str) -> list[dict[str, object]]:
    if condition == "full_catalog":
        groups = [((), wide)]
        keys: list[str] = []
    else:
        keys = ["sampling_strategy", "requested_candidate_size", "seed"]
        groups = wide.groupby(keys, dropna=False, sort=True)
    rows: list[dict[str, object]] = []
    for condition_key, local in groups:
        if not isinstance(condition_key, tuple):
            condition_key = (condition_key,)
        base = dict(zip(keys, condition_key))
        total_weight = float(local["population_weight"].sum())
        for model in MODELS:
            hit = local[f"hit_rate__{model}"].gt(0.0)
            ranks = pd.Series(np.nan, index=local.index, dtype=float)
            ranks.loc[hit] = 1.0 / local.loc[hit, f"mrr__{model}"]
            rows.append(
                {
                    "condition": condition,
                    **base,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "users_n": int(len(local)),
                    "weighted_hit_rate": (
                        float(local.loc[hit, "population_weight"].sum()) / total_weight
                        if total_weight else float("nan")
                    ),
                    "hit_users_n": int(hit.sum()),
                    "first_relevant_rank_p25_among_hits": weighted_quantile(
                        ranks.loc[hit], local.loc[hit, "population_weight"], 0.25
                    ),
                    "first_relevant_rank_median_among_hits": weighted_quantile(
                        ranks.loc[hit], local.loc[hit, "population_weight"], 0.50
                    ),
                    "first_relevant_rank_p75_among_hits": weighted_quantile(
                        ranks.loc[hit], local.loc[hit, "population_weight"], 0.75
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.primary_results_dir / "checkpoints"
    data, inputs = load_relevant_rows(checkpoint_root, args.k)

    full = add_state(validate_and_pivot(data, "full_catalog"), "full")
    sampled = add_state(validate_and_pivot(data, "sampled"), "sampled")
    if full[["window", "user_id"]].duplicated().any():
        raise ValueError("Full-catalog users are duplicated")
    validate_primary_grid(full, sampled, args.k)

    detail = transition_detail(full, sampled)
    by_seed, summary = summarize_transitions(detail)
    ranks = pd.DataFrame(rank_rows(full, "full_catalog") + rank_rows(sampled, "sampled"))
    sampled_rank_summary = (
        ranks.loc[ranks["condition"].eq("sampled")]
        .groupby(["sampling_strategy", "requested_candidate_size", "model", "model_label"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            weighted_hit_rate_mean=("weighted_hit_rate", "mean"),
            weighted_hit_rate_sd=("weighted_hit_rate", "std"),
            first_relevant_rank_median_mean=("first_relevant_rank_median_among_hits", "mean"),
            first_relevant_rank_p25_mean=("first_relevant_rank_p25_among_hits", "mean"),
            first_relevant_rank_p75_mean=("first_relevant_rank_p75_among_hits", "mean"),
        )
    )

    detail.to_csv(args.output_dir / "careerbuilder_title_bm25_minilm_transition_detail.csv.gz", index=False)
    by_seed.to_csv(args.output_dir / "careerbuilder_title_bm25_minilm_transition_by_seed.csv", index=False)
    summary.to_csv(args.output_dir / "careerbuilder_title_bm25_minilm_transition_summary.csv", index=False)
    ranks.to_csv(args.output_dir / "careerbuilder_title_bm25_minilm_rank_by_seed.csv", index=False)
    sampled_rank_summary.to_csv(
        args.output_dir / "careerbuilder_title_bm25_minilm_rank_summary.csv", index=False
    )

    manifest_path = args.primary_results_dir / "checkpoint_manifest.json"
    audit = {
        "analysis": "post-decision field-matched title BM25 versus title MiniLM transition sensitivity",
        "k": args.k,
        "models": list(MODELS),
        "primary_users": int(len(full)),
        "windows": sorted(int(value) for value in full["window"].unique()),
        "sampled_user_conditions": int(len(detail)),
        "expected_grid": {
            "strategies": list(EXPECTED_STRATEGIES),
            "candidate_sizes": list(EXPECTED_SIZES),
            "seeds": list(EXPECTED_SEEDS),
        },
        "checkpoint_files": len(inputs),
        "checkpoint_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
        },
        "inputs": inputs,
        "rank_definition": "For hits at K=20, first relevant rank equals 1/MRR@20; misses have undefined rank.",
        "limitations": [
            "Hit transitions are relative to logged relevance in the protocol-defined eligible catalog.",
            "Sampled summaries average five deterministic seeds; they do not establish latent preference validity.",
        ],
    }
    (args.output_dir / "careerbuilder_title_bm25_minilm_transition_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"full users: {len(full):,}")
    print(f"sampled user-conditions: {len(detail):,}")
    print(f"checkpoint files: {len(inputs):,}")


if __name__ == "__main__":
    main()
