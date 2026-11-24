#!/usr/bin/env python3
"""Controlled CareerBuilder local-reference catalog-scale sensitivity.

The CareerBuilder primary audit used nested sampled candidate sets.  This
post-decision analysis treats each user's 5,000-candidate realization as a
*sampled local reference* and compares the nested 100/500/1,000 conditions
with that same strategy/seed reference.  Five thousand is the closest frozen
CareerBuilder grid point to the JTH catalog (median 5,808; mean 5,812.9).

The 5,000-candidate condition is not sampling-free and does not replace the
exact eligible-catalog results.  The purpose is to isolate catalog-scale
sensitivity while holding the CareerBuilder task, users, targets, models,
sampler, and seed fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_SIZE = 5000
SMALLER_SIZES = (100, 500, 1000)
SMALLER_SIZES_BY_METRIC = {
    "hit_rate": (100, 500, 1000),
    # K=150 is intentionally undefined when the nominal candidate size is 100.
    "map": (500, 1000),
}
MODELS = (
    "random_hash",
    "popularity_precut",
    "tfidf_combined_static",
    "bm25_full_static",
    "bm25_title_static",
    "minilm_title_static",
)
MODEL_LABELS = {
    "random_hash": "Random",
    "popularity_precut": "Popularity",
    "tfidf_combined_static": "Combined TF-IDF",
    "bm25_full_static": "Full-text BM25",
    "bm25_title_static": "Title BM25",
    "minilm_title_static": "Title MiniLM",
}
METRIC_SPECS = (("hit_rate", 20), ("map", 150))
USECOLS = (
    "window",
    "user_id",
    "population_weight",
    "candidate_regime",
    "sampling_strategy",
    "seed",
    "requested_candidate_size",
    "actual_candidate_count",
    "targets_injected",
    "model",
    "k",
    "hit_rate",
    "map",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_detail(checkpoint_root: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    files = sorted(checkpoint_root.glob("window_*/users_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No checkpoint files under {checkpoint_root}")
    pieces: list[pd.DataFrame] = []
    inputs: list[dict[str, object]] = []
    allowed_sizes = set(SMALLER_SIZES) | {REFERENCE_SIZE}
    for path in files:
        frame = pd.read_csv(path, usecols=list(USECOLS))
        frame = frame.loc[
            frame["candidate_regime"].eq("sampled")
            & frame["requested_candidate_size"].isin(allowed_sizes)
            & frame["model"].isin(MODELS)
        ].copy()
        metric_mask = pd.Series(False, index=frame.index)
        for _, k in METRIC_SPECS:
            metric_mask |= frame["k"].eq(k)
        frame = frame.loc[metric_mask]
        if frame.empty:
            raise ValueError(f"No local-reference rows in {path}")
        pieces.append(frame)
        inputs.append(
            {
                "file": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "selected_rows": int(len(frame)),
            }
        )
    return pd.concat(pieces, ignore_index=True), inputs


def long_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    identity = [
        "window",
        "user_id",
        "population_weight",
        "sampling_strategy",
        "seed",
        "requested_candidate_size",
        "actual_candidate_count",
        "targets_injected",
        "model",
        "k",
    ]
    for metric, k in METRIC_SPECS:
        part = frame.loc[frame["k"].eq(k), identity + [metric]].copy()
        part = part.rename(columns={metric: "value"})
        part["metric"] = metric
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True)
    result["requested_candidate_size"] = result["requested_candidate_size"].astype(int)
    return result


def validate_nested_conditions(data: pd.DataFrame) -> None:
    keys = ["window", "user_id", "sampling_strategy", "seed", "model", "metric"]
    for metric, smaller_sizes in SMALLER_SIZES_BY_METRIC.items():
        counts = data.loc[data["metric"].eq(metric)].groupby(keys)[
            "requested_candidate_size"
        ].nunique()
        if not counts.eq(len(smaller_sizes) + 1).all():
            raise ValueError(f"Incomplete local-reference size grid for {metric}")
    if data.duplicated(keys + ["requested_candidate_size"]).any():
        raise ValueError("Duplicate local-reference cells")
    if not data["targets_injected"].all():
        raise ValueError("Sampled conditions do not consistently retain target injection")
    if data["actual_candidate_count"].lt(data["requested_candidate_size"]).any():
        raise ValueError("Actual candidate count falls below its nominal size")


def attach_reference(data: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "window",
        "user_id",
        "population_weight",
        "sampling_strategy",
        "seed",
        "model",
        "metric",
        "k",
    ]
    reference = data.loc[
        data["requested_candidate_size"].eq(REFERENCE_SIZE),
        keys + ["value", "actual_candidate_count"],
    ]
    reference = reference.rename(
        columns={
            "value": "local_reference_value",
            "actual_candidate_count": "local_reference_candidate_count",
        }
    )
    smaller = data.loc[data["requested_candidate_size"].isin(SMALLER_SIZES)].copy()
    merged = smaller.merge(reference, on=keys, how="left", validate="many_to_one")
    if merged["local_reference_value"].isna().any():
        raise ValueError("A smaller condition lacks its 5,000-candidate local reference")
    if merged["actual_candidate_count"].gt(merged["local_reference_candidate_count"]).any():
        raise ValueError("A nested condition is larger than its local reference")
    merged["signed_difference"] = merged["value"] - merged["local_reference_value"]
    merged["actual_candidate_fraction_of_local_reference"] = (
        merged["actual_candidate_count"] / merged["local_reference_candidate_count"]
    )
    return merged


def weighted_summary(detail: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, local in detail.groupby(groups, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        weights = local["population_weight"].to_numpy(dtype=float)
        values = local["value"].to_numpy(dtype=float)
        references = local["local_reference_value"].to_numpy(dtype=float)
        differences = local["signed_difference"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(groups, group_values)),
                "users_n": int(len(local)),
                "weight_sum": float(weights.sum()),
                "sampled_estimate": float(np.average(values, weights=weights)),
                "local_reference_estimate": float(np.average(references, weights=weights)),
                "signed_difference": float(np.average(differences, weights=weights)),
                "mean_candidate_fraction_of_local_reference": float(
                    local["requested_candidate_size"].iloc[0] / REFERENCE_SIZE
                ),
                "weighted_mean_actual_candidate_fraction_of_local_reference": float(
                    np.average(
                        local["actual_candidate_fraction_of_local_reference"].to_numpy(dtype=float),
                        weights=weights,
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def pairwise_order(scores: pd.Series, tolerance: float = 1e-12) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for left, right in combinations(MODELS, 2):
        delta = float(scores[left] - scores[right])
        result[(left, right)] = 0 if abs(delta) <= tolerance else (1 if delta > 0 else -1)
    return result


def top_set(scores: pd.Series, tolerance: float = 1e-12) -> set[str]:
    maximum = float(scores.max())
    return {str(model) for model, value in scores.items() if abs(float(value) - maximum) <= tolerance}


def kendall_tau_b_from_orders(
    left: dict[tuple[str, str], int], right: dict[tuple[str, str], int]
) -> float:
    """Kendall tau-b for a small, fixed model set, with explicit ties."""
    pairs = list(left)
    concordant = sum(left[pair] * right[pair] > 0 for pair in pairs)
    discordant = sum(left[pair] * right[pair] < 0 for pair in pairs)
    ties_left = sum(left[pair] == 0 for pair in pairs)
    ties_right = sum(right[pair] == 0 for pair in pairs)
    denominator = np.sqrt((len(pairs) - ties_left) * (len(pairs) - ties_right))
    return float((concordant - discordant) / denominator) if denominator else float("nan")


def rank_fidelity(seed_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["sampling_strategy", "seed", "metric", "k"]
    rows: list[dict[str, object]] = []
    for condition, local in seed_summary.groupby(keys, sort=True):
        reference_consistency = local.groupby("model")["local_reference_estimate"].nunique()
        if not reference_consistency.eq(1).all():
            raise ValueError("The same local reference changes across nested sizes")
        reference_scores = local.groupby("model")["local_reference_estimate"].first()
        if set(reference_scores.index) != set(MODELS):
            raise ValueError("Incomplete model set in local reference")
        ref_top = top_set(reference_scores)
        ref_order = pairwise_order(reference_scores)
        for size, sized in local.groupby("requested_candidate_size", sort=True):
            sampled_scores = sized.set_index("model")["sampled_estimate"]
            sampled_top = top_set(sampled_scores)
            sampled_order = pairwise_order(sampled_scores)
            agreement = np.mean(
                [sampled_order[pair] == ref_order[pair] for pair in ref_order]
            )
            tau = kendall_tau_b_from_orders(ref_order, sampled_order)
            rows.append(
                {
                    **dict(zip(keys, condition)),
                    "requested_candidate_size": int(size),
                    "candidate_fraction_of_local_reference": float(size / REFERENCE_SIZE),
                    "kendall_tau_b": float(tau),
                    "pairwise_order_agreement": float(agreement),
                    "top_set_exact_agreement": bool(sampled_top == ref_top),
                    "sampled_top_set": "|".join(MODEL_LABELS[value] for value in MODELS if value in sampled_top),
                    "local_reference_top_set": "|".join(
                        MODEL_LABELS[value] for value in MODELS if value in ref_top
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    integrity_path = args.primary_results_dir / "INTEGRITY_AUDIT.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    required_integrity = {
        "target_inclusion_failures": 0,
        "nesting_failures": 0,
        "nested_candidate_failures": 0,
        "size100_k150_rows": 0,
        "invalid_cutoff_rows": 0,
        "all_targets_injected_in_sampled_candidates": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": integrity.get(key)}
        for key, expected in required_integrity.items()
        if integrity.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Authoritative integrity audit failed: {mismatches}")
    raw, inputs = load_detail(args.primary_results_dir / "checkpoints")
    data = long_metrics(raw)
    validate_nested_conditions(data)
    detail = attach_reference(data)

    groups = [
        "sampling_strategy",
        "seed",
        "requested_candidate_size",
        "model",
        "metric",
        "k",
    ]
    seed_summary = weighted_summary(detail, groups)
    per_window = weighted_summary(detail, ["window"] + groups)
    across_seed = (
        seed_summary.groupby(
            ["sampling_strategy", "requested_candidate_size", "model", "metric", "k"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            sampled_estimate_mean=("sampled_estimate", "mean"),
            local_reference_estimate_mean=("local_reference_estimate", "mean"),
            signed_difference_mean=("signed_difference", "mean"),
            signed_difference_sd=("signed_difference", "std"),
            signed_difference_min=("signed_difference", "min"),
            signed_difference_max=("signed_difference", "max"),
            candidate_fraction_of_local_reference=(
                "mean_candidate_fraction_of_local_reference",
                "first",
            ),
            actual_candidate_fraction_of_local_reference_mean=(
                "weighted_mean_actual_candidate_fraction_of_local_reference",
                "mean",
            ),
        )
    )
    ranks = rank_fidelity(seed_summary)
    rank_summary = (
        ranks.groupby(["sampling_strategy", "requested_candidate_size", "metric", "k"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            kendall_tau_b_mean=("kendall_tau_b", "mean"),
            kendall_tau_b_min=("kendall_tau_b", "min"),
            pairwise_order_agreement_mean=("pairwise_order_agreement", "mean"),
            pairwise_order_agreement_min=("pairwise_order_agreement", "min"),
            top_set_agreement_rate=("top_set_exact_agreement", "mean"),
            candidate_fraction_of_local_reference=("candidate_fraction_of_local_reference", "first"),
        )
    )

    detail.to_csv(args.output_dir / "careerbuilder_local_reference_difference_detail.csv.gz", index=False)
    seed_summary.to_csv(args.output_dir / "careerbuilder_local_reference_by_seed.csv", index=False)
    per_window.to_csv(args.output_dir / "careerbuilder_local_reference_by_window_seed.csv", index=False)
    across_seed.to_csv(args.output_dir / "careerbuilder_local_reference_summary.csv", index=False)
    ranks.to_csv(args.output_dir / "careerbuilder_local_reference_rank_fidelity_by_seed.csv", index=False)
    rank_summary.to_csv(args.output_dir / "careerbuilder_local_reference_rank_fidelity_summary.csv", index=False)

    manifest_path = args.primary_results_dir / "checkpoint_manifest.json"
    expanded = detail.loc[
        detail["actual_candidate_count"].gt(detail["requested_candidate_size"])
    ]
    expanded_conditions = expanded[
        [
            "window",
            "user_id",
            "sampling_strategy",
            "seed",
            "requested_candidate_size",
            "actual_candidate_count",
        ]
    ].drop_duplicates()
    audit = {
        "analysis": "post-decision controlled CareerBuilder local-reference catalog-scale sensitivity",
        "reference_size": REFERENCE_SIZE,
        "reference_status": "sampled local reference; not sampling-free",
        "jth_catalog_context": {"median": 5808, "mean": 5812.9},
        "smaller_sizes_by_metric": {
            metric: list(sizes) for metric, sizes in SMALLER_SIZES_BY_METRIC.items()
        },
        "models": list(MODELS),
        "metric_specs": [{"metric": metric, "k": k} for metric, k in METRIC_SPECS],
        "users": int(data["user_id"].nunique()),
        "windows": sorted(int(value) for value in data["window"].unique()),
        "checkpoint_files": len(inputs),
        "checkpoint_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
        },
        "authoritative_integrity_audit": {
            "path": str(integrity_path.resolve()),
            "sha256": sha256(integrity_path),
            "verified_fields": required_integrity,
            "target_count_exceeds_nominal_user_sizes": int(
                integrity["target_count_exceeds_nominal_user_sizes"]
            ),
        },
        "realized_size_expansion": {
            "selected_metric_detail_rows": int(len(expanded)),
            "unique_user_strategy_seed_size_conditions": int(len(expanded_conditions)),
            "nominal_actual_counts": [
                {
                    "requested_candidate_size": int(requested),
                    "actual_candidate_count": int(actual),
                    "conditions": int(len(group)),
                }
                for (requested, actual), group in expanded_conditions.groupby(
                    ["requested_candidate_size", "actual_candidate_count"], sort=True
                )
            ],
        },
        "inputs": inputs,
        "safeguards": [
            "Every comparison holds user, strategy, seed, model, metric, and cutoff fixed.",
            "Candidate sets are nested by the authoritative evaluator and include the same observed targets.",
            "When a user has more observed targets than a nominal small size, target injection can make the realized set larger; realized fractions are reported.",
            "The 5,000-candidate condition remains sampled and is never described as the exact catalog.",
        ],
    }
    (args.output_dir / "careerbuilder_local_reference_scale_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"users: {data['user_id'].nunique():,}")
    print(f"paired detail rows: {len(detail):,}")
    print(f"checkpoint files: {len(inputs):,}")


if __name__ == "__main__":
    main()
