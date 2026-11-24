#!/usr/bin/env python3
"""Leave-one-window-out sensitivity for the CareerBuilder primary audit.

This post-decision sensitivity analysis asks whether pooled conclusions depend
on any single provider-defined window.  It uses the already-completed,
authoritative aggregate output; it does not rerun recommenders or change the
candidate protocol.  For each omitted window, the script recomputes:

* pooled effectiveness for the audit-cohort and design-weighted estimands;
* sampled-minus-full metric bias;
* the five prespecified model contrasts; and
* sampled/full-catalog rank agreement across all six evaluated models.

The script fails closed unless the input is the complete seven-window primary
design recorded as authoritative production in ``RUN_REPORT.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


MODELS = (
    "random_hash",
    "popularity_precut",
    "tfidf_combined_static",
    "bm25_full_static",
    "bm25_title_static",
    "minilm_title_static",
)
WINDOWS = tuple(range(1, 8))
STRATEGIES = ("uniform", "popularity_weighted")
SEEDS = tuple(range(1, 6))
SIZES = (100, 500, 1000, 5000, 10000)
KS = (5, 10, 20, 150)
METRICS = ("precision", "recall", "hit_rate", "ndcg", "mrr", "map")
PRIMARY_OUTCOMES = (
    ("hit_rate", 20),
    ("ndcg", 20),
    ("map", 20),
    ("map", 150),
)
CONTRASTS = (
    ("bm25_title_static", "minilm_title_static", "BM25-title_minus_MiniLM-title"),
    ("bm25_full_static", "minilm_title_static", "BM25-full_minus_MiniLM-title"),
    ("bm25_full_static", "tfidf_combined_static", "BM25-full_minus_TFIDF-combined"),
    ("bm25_full_static", "popularity_precut", "BM25-full_minus_popularity"),
    ("bm25_full_static", "random_hash", "BM25-full_minus_random"),
)
TIE_RTOL = 1e-10
TIE_ATOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-results-dir",
        type=Path,
        required=True,
        help="Authoritative CareerBuilder primary run directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_authority(results_dir: Path) -> tuple[Path, Path, dict]:
    aggregate_path = results_dir / "aggregate_metrics.csv"
    report_path = results_dir / "RUN_REPORT.json"
    if not aggregate_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("aggregate_metrics.csv or RUN_REPORT.json is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    authority = report.get("authority", {})
    required_authority = {
        "authoritative": True,
        "production": True,
        "smoke": False,
        "study_role": "primary",
        "status": "completed",
    }
    if any(authority.get(key) != value for key, value in required_authority.items()):
        raise ValueError(f"Run is not completed authoritative production: {authority}")
    config = report.get("run_config", {})
    expected_config = {
        "protocol": "First7Last2",
        "windows": list(WINDOWS),
        "seeds": list(SEEDS),
        "strategies": list(STRATEGIES),
        "candidate_sizes": list(SIZES),
        "ks": list(KS),
        "models": list(MODELS),
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected RUN_REPORT {key}: {config.get(key)!r}")
    return aggregate_path, report_path, report


def validate_aggregate(frame: pd.DataFrame) -> None:
    required = {
        "protocol", "window", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k", "users", "weight_sum",
        "actual_candidate_fraction_mean", *METRICS,
        *(f"weighted_{metric}" for metric in METRICS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Aggregate input lacks columns: {missing}")
    keys = [
        "window", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k",
    ]
    if frame.duplicated(keys).any():
        raise ValueError("Aggregate input has duplicate design cells")
    if set(frame["window"].astype(int)) != set(WINDOWS):
        raise ValueError("Aggregate input does not contain exactly windows 1--7")
    if set(frame["model"].astype(str)) != set(MODELS):
        raise ValueError("Aggregate input has an unexpected model set")
    if set(frame["protocol"].astype(str)) != {"First7Last2"}:
        raise ValueError("Aggregate input has an unexpected protocol")
    full = frame[frame["candidate_regime"].eq("full_catalog")]
    sampled = frame[frame["candidate_regime"].eq("sampled")]
    if len(full) != len(WINDOWS) * len(MODELS) * len(KS):
        raise ValueError(f"Unexpected full-catalog cell count: {len(full)}")
    expected_sampled = len(WINDOWS) * len(STRATEGIES) * len(SEEDS) * len(MODELS) * 19
    if len(sampled) != expected_sampled:
        raise ValueError(f"Unexpected sampled cell count: {len(sampled)}")
    if set(sampled["sampling_strategy"].astype(str)) != set(STRATEGIES):
        raise ValueError("Unexpected sampling strategies")
    if set(sampled["seed"].astype(int)) != set(SEEDS):
        raise ValueError("Unexpected sampling seeds")
    if set(sampled["requested_candidate_size"].astype(int)) != set(SIZES):
        raise ValueError("Unexpected candidate sizes")
    invalid_100_150 = sampled[
        sampled["requested_candidate_size"].eq(100) & sampled["k"].eq(150)
    ]
    if not invalid_100_150.empty:
        raise ValueError("The excluded 100-candidate/K=150 cell is present")
    if not (frame["users"].astype(int) == 1000).all():
        raise ValueError("Expected 1,000 audit users per window and cell")
    numeric = [*METRICS, *(f"weighted_{metric}" for metric in METRICS)]
    values = frame[numeric].to_numpy(float)
    if not np.isfinite(values).all() or (values < -1e-12).any() or (values > 1 + 1e-12).any():
        raise ValueError("Metric values are non-finite or outside [0, 1]")


def pooled_without_window(frame: pd.DataFrame, omitted_window: int) -> pd.DataFrame:
    retained = frame[frame["window"].ne(omitted_window)].copy()
    condition = [
        "protocol", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k",
    ]
    records: list[dict] = []
    for values, cell in retained.groupby(condition, dropna=False, sort=False):
        common = dict(zip(condition, values))
        common.update({
            "omitted_window": omitted_window,
            "retained_windows": int(cell["window"].nunique()),
            "audit_users": int(cell["users"].sum()),
            "population_weight_sum": float(cell["weight_sum"].sum()),
            "actual_candidate_fraction_mean": float(np.average(
                cell["actual_candidate_fraction_mean"], weights=cell["users"]
            )),
        })
        audit = {
            metric: float(np.average(cell[metric], weights=cell["users"]))
            for metric in METRICS
        }
        weighted = {
            metric: float(np.average(cell[f"weighted_{metric}"], weights=cell["weight_sum"]))
            for metric in METRICS
        }
        records.append({**common, "estimand": "audit_cohort", **audit})
        records.append({
            **common, "estimand": "design_weighted_population", **weighted
        })
    return pd.DataFrame.from_records(records)


def make_sampling_bias(pooled: pd.DataFrame) -> pd.DataFrame:
    full = pooled[pooled["candidate_regime"].eq("full_catalog")]
    sampled = pooled[pooled["candidate_regime"].eq("sampled")]
    join = ["omitted_window", "estimand", "model", "k"]
    full_long = full.melt(
        id_vars=join,
        value_vars=list(METRICS),
        var_name="metric",
        value_name="full_value",
    )
    sampled_long = sampled.melt(
        id_vars=[
            *join, "sampling_strategy", "seed", "requested_candidate_size",
            "audit_users", "population_weight_sum", "actual_candidate_fraction_mean",
        ],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="sampled_value",
    )
    merged = sampled_long.merge(
        full_long, on=[*join, "metric"], how="left", validate="many_to_one"
    )
    if merged["full_value"].isna().any():
        raise ValueError("Sampling-bias join lost a full-catalog reference")
    merged["signed_bias_sampled_minus_full"] = (
        merged["sampled_value"] - merged["full_value"]
    )
    merged["absolute_bias"] = merged["signed_bias_sampled_minus_full"].abs()
    return merged.sort_values([
        "omitted_window", "estimand", "sampling_strategy",
        "requested_candidate_size", "model", "k", "metric", "seed",
    ]).reset_index(drop=True)


def make_contrasts(pooled: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for omitted_window in WINDOWS:
        for estimand in ("audit_cohort", "design_weighted_population"):
            local = pooled[
                pooled["omitted_window"].eq(omitted_window)
                & pooled["estimand"].eq(estimand)
            ]
            full = local[local["candidate_regime"].eq("full_catalog")]
            sampled = local[local["candidate_regime"].eq("sampled")]
            for metric, k in PRIMARY_OUTCOMES:
                full_cell = full[full["k"].eq(k)].set_index("model")
                for left, right, name in CONTRASTS:
                    full_contrast = float(full_cell.at[left, metric] - full_cell.at[right, metric])
                    records.append({
                        "omitted_window": omitted_window,
                        "retained_windows": 6,
                        "estimand": estimand,
                        "candidate_regime": "full_catalog",
                        "sampling_strategy": "full_catalog",
                        "seed_aggregation": "not_applicable",
                        "requested_candidate_size": np.nan,
                        "contrast": name,
                        "left_model": left,
                        "right_model": right,
                        "metric": metric,
                        "k": k,
                        "contrast_estimate": full_contrast,
                        "full_contrast_reference": full_contrast,
                        "effect_distortion_sampled_minus_full": 0.0,
                    })
                for strategy in STRATEGIES:
                    for size in SIZES:
                        if size == 100 and k == 150:
                            continue
                        cell = sampled[
                            sampled["sampling_strategy"].eq(strategy)
                            & sampled["requested_candidate_size"].eq(size)
                            & sampled["k"].eq(k)
                        ]
                        seed_means = cell.groupby("model", sort=False)[metric].mean()
                        if set(seed_means.index.astype(str)) != set(MODELS):
                            raise ValueError("Incomplete model vector in contrast analysis")
                        for left, right, name in CONTRASTS:
                            estimate = float(seed_means[left] - seed_means[right])
                            full_contrast = float(
                                full_cell.at[left, metric] - full_cell.at[right, metric]
                            )
                            records.append({
                                "omitted_window": omitted_window,
                                "retained_windows": 6,
                                "estimand": estimand,
                                "candidate_regime": "sampled",
                                "sampling_strategy": strategy,
                                "seed_aggregation": "mean_over_5_seeds",
                                "requested_candidate_size": size,
                                "contrast": name,
                                "left_model": left,
                                "right_model": right,
                                "metric": metric,
                                "k": k,
                                "contrast_estimate": estimate,
                                "full_contrast_reference": full_contrast,
                                "effect_distortion_sampled_minus_full": (
                                    estimate - full_contrast
                                ),
                            })
    return pd.DataFrame.from_records(records).sort_values([
        "omitted_window", "estimand", "candidate_regime", "sampling_strategy",
        "requested_candidate_size", "metric", "k", "contrast",
    ]).reset_index(drop=True)


def is_tie(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=TIE_RTOL, atol=TIE_ATOL))


def ordered_models(values: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda model: (-values[model], model)))


def tie_group_codes(values: Mapping[str, float]) -> np.ndarray:
    ordered = ordered_models(values)
    code_by_model: dict[str, int] = {}
    group = 0
    anchor = values[ordered[0]]
    code_by_model[ordered[0]] = group
    for model in ordered[1:]:
        if not is_tie(values[model], anchor):
            group += 1
            anchor = values[model]
        code_by_model[model] = group
    return np.asarray([code_by_model[model] for model in MODELS], dtype=float)


def winner_set(values: Mapping[str, float]) -> set[str]:
    maximum = max(values.values())
    return {model for model, value in values.items() if is_tie(value, maximum)}


def pairwise_state(left: float, right: float) -> int:
    if is_tie(left, right):
        return 0
    return 1 if left > right else -1


def kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float:
    """Return Kendall's tau-b for two small ordinal vectors.

    The implementation follows the pair-count definition and avoids adding a
    SciPy dependency to this lightweight revision audit.  Pairs tied on both
    vectors contribute to neither tie-only term.
    """
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Kendall inputs must be aligned one-dimensional vectors")
    concordant = 0
    discordant = 0
    tied_left_only = 0
    tied_right_only = 0
    for first, second in itertools.combinations(range(len(left)), 2):
        left_state = np.sign(left[first] - left[second])
        right_state = np.sign(right[first] - right[second])
        if left_state == 0 and right_state == 0:
            continue
        if left_state == 0:
            tied_left_only += 1
        elif right_state == 0:
            tied_right_only += 1
        elif left_state == right_state:
            concordant += 1
        else:
            discordant += 1
    numerator = concordant - discordant
    denominator = np.sqrt(
        (concordant + discordant + tied_left_only)
        * (concordant + discordant + tied_right_only)
    )
    return float(numerator / denominator) if denominator else float("nan")


def rank_record(
    full_values: Mapping[str, float], sampled_values: Mapping[str, float]
) -> dict[str, object]:
    tau = kendall_tau_b(
        tie_group_codes(full_values), tie_group_codes(sampled_values)
    )
    full_winners = winner_set(full_values)
    sampled_winners = winner_set(sampled_values)
    agreements = [
        pairwise_state(full_values[left], full_values[right])
        == pairwise_state(sampled_values[left], sampled_values[right])
        for left, right in itertools.combinations(MODELS, 2)
    ]
    full_maximum = max(full_values.values())
    sampled_winner_full_scores = [full_values[model] for model in sampled_winners]
    union = full_winners | sampled_winners
    intersection = full_winners & sampled_winners
    return {
        "full_model_order": "|".join(ordered_models(full_values)),
        "sampled_model_order": "|".join(ordered_models(sampled_values)),
        "kendall_tau_b": float(tau),
        "kendall_tau_b_defined": bool(np.isfinite(tau)),
        "pairwise_order_agreement": float(np.mean(agreements)),
        "full_winner_set": "|".join(sorted(full_winners)),
        "sampled_winner_set": "|".join(sorted(sampled_winners)),
        "winner_set_exact_match": full_winners == sampled_winners,
        "winner_set_overlap": bool(intersection),
        "winner_set_jaccard": len(intersection) / len(union),
        "sampled_winner_reference_loss_best": max(
            0.0, full_maximum - max(sampled_winner_full_scores)
        ),
        "sampled_winner_reference_loss_worst": max(
            0.0, full_maximum - min(sampled_winner_full_scores)
        ),
    }


def make_rank_fidelity(pooled: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for omitted_window in WINDOWS:
        for estimand in ("audit_cohort", "design_weighted_population"):
            local = pooled[
                pooled["omitted_window"].eq(omitted_window)
                & pooled["estimand"].eq(estimand)
            ]
            full = local[local["candidate_regime"].eq("full_catalog")]
            sampled = local[local["candidate_regime"].eq("sampled")]
            for metric, k in PRIMARY_OUTCOMES:
                full_cell = full[full["k"].eq(k)]
                full_values = dict(zip(full_cell["model"].astype(str), full_cell[metric]))
                if set(full_values) != set(MODELS):
                    raise ValueError("Incomplete full-catalog rank vector")
                for (strategy, seed, size), cell in sampled[
                    sampled["k"].eq(k)
                ].groupby(
                    ["sampling_strategy", "seed", "requested_candidate_size"],
                    dropna=False, sort=False,
                ):
                    sampled_values = dict(zip(cell["model"].astype(str), cell[metric]))
                    if set(sampled_values) != set(MODELS):
                        raise ValueError("Incomplete sampled rank vector")
                    records.append({
                        "omitted_window": omitted_window,
                        "retained_windows": 6,
                        "estimand": estimand,
                        "sampling_strategy": strategy,
                        "seed": int(seed),
                        "requested_candidate_size": int(size),
                        "metric": metric,
                        "k": k,
                        **rank_record(full_values, sampled_values),
                    })
    return pd.DataFrame.from_records(records).sort_values([
        "omitted_window", "estimand", "sampling_strategy",
        "requested_candidate_size", "metric", "k", "seed",
    ]).reset_index(drop=True)


def summarize_rank_fidelity(detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["estimand", "sampling_strategy", "requested_candidate_size", "metric", "k"]
    return detail.groupby(keys, as_index=False, sort=False).agg(
        omitted_windows=("omitted_window", "nunique"),
        seeds=("seed", "nunique"),
        units=("seed", "size"),
        kendall_tau_b_mean=("kendall_tau_b", "mean"),
        kendall_tau_b_min=("kendall_tau_b", "min"),
        kendall_tau_b_max=("kendall_tau_b", "max"),
        pairwise_order_agreement_mean=("pairwise_order_agreement", "mean"),
        pairwise_order_agreement_min=("pairwise_order_agreement", "min"),
        pairwise_order_agreement_max=("pairwise_order_agreement", "max"),
        winner_set_exact_match_rate=("winner_set_exact_match", "mean"),
        winner_set_overlap_rate=("winner_set_overlap", "mean"),
        sampled_winner_reference_loss_best_mean=(
            "sampled_winner_reference_loss_best", "mean"
        ),
        sampled_winner_reference_loss_best_max=(
            "sampled_winner_reference_loss_best", "max"
        ),
    )


def make_full_catalog_stability(pooled: pd.DataFrame) -> pd.DataFrame:
    full = pooled[pooled["candidate_regime"].eq("full_catalog")]
    records: list[dict] = []
    for omitted_window in WINDOWS:
        for estimand in ("audit_cohort", "design_weighted_population"):
            local = full[
                full["omitted_window"].eq(omitted_window)
                & full["estimand"].eq(estimand)
            ]
            for metric, k in PRIMARY_OUTCOMES:
                cell = local[local["k"].eq(k)]
                values = dict(zip(cell["model"].astype(str), cell[metric]))
                winners = winner_set(values)
                records.append({
                    "omitted_window": omitted_window,
                    "retained_windows": 6,
                    "estimand": estimand,
                    "metric": metric,
                    "k": k,
                    "full_model_order": "|".join(ordered_models(values)),
                    "full_winner_set": "|".join(sorted(winners)),
                    "winner_value": max(values.values()),
                    "runner_up_value": sorted(values.values(), reverse=True)[1],
                    "winner_margin": (
                        max(values.values()) - sorted(values.values(), reverse=True)[1]
                    ),
                })
    return pd.DataFrame.from_records(records).sort_values([
        "estimand", "metric", "k", "omitted_window"
    ]).reset_index(drop=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def main() -> None:
    args = parse_args()
    results_dir = args.primary_results_dir.resolve()
    output_dir = args.output_dir.resolve()
    aggregate_path, report_path, report = validate_authority(results_dir)
    aggregate = pd.read_csv(aggregate_path)
    validate_aggregate(aggregate)

    pooled = pd.concat(
        [pooled_without_window(aggregate, window) for window in WINDOWS],
        ignore_index=True,
    )
    pooled = pooled.sort_values([
        "omitted_window", "estimand", "candidate_regime", "sampling_strategy",
        "requested_candidate_size", "model", "k", "seed",
    ]).reset_index(drop=True)
    bias = make_sampling_bias(pooled)
    contrasts = make_contrasts(pooled)
    rank_detail = make_rank_fidelity(pooled)
    rank_summary = summarize_rank_fidelity(rank_detail)
    full_stability = make_full_catalog_stability(pooled)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "leave_one_window_out_pooled_estimates.csv": pooled,
        "leave_one_window_out_sampling_bias.csv": bias,
        "leave_one_window_out_primary_contrasts.csv": contrasts,
        "leave_one_window_out_rank_fidelity_detail.csv": rank_detail,
        "leave_one_window_out_rank_fidelity_summary.csv": rank_summary,
        "leave_one_window_out_full_catalog_stability.csv": full_stability,
    }
    for name, frame in outputs.items():
        write_csv(frame, output_dir / name)

    full_orders = full_stability.groupby(["estimand", "metric", "k"]).agg(
        unique_winner_sets=("full_winner_set", "nunique"),
        unique_model_orders=("full_model_order", "nunique"),
        minimum_winner_margin=("winner_margin", "min"),
    ).reset_index()
    primary_bias = bias[
        bias[["metric", "k"]].apply(tuple, axis=1).isin(PRIMARY_OUTCOMES)
    ]
    evidence = {
        "analysis": "CareerBuilder leave-one-provider-window-out sensitivity",
        "status": "completed",
        "post_decision_sensitivity": True,
        "does_not_rerun_recommenders": True,
        "source_authority": report["authority"],
        "inputs": {
            "aggregate_metrics": {
                "path": str(aggregate_path),
                "sha256": sha256(aggregate_path),
                "rows": int(len(aggregate)),
            },
            "run_report": {
                "path": str(report_path),
                "sha256": sha256(report_path),
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "design": {
            "omitted_windows": list(WINDOWS),
            "retained_windows_per_analysis": 6,
            "estimands": ["audit_cohort", "design_weighted_population"],
            "models": list(MODELS),
            "primary_outcomes": [f"{metric}@{k}" for metric, k in PRIMARY_OUTCOMES],
            "rank_tie_rtol": TIE_RTOL,
            "rank_tie_atol": TIE_ATOL,
        },
        "verified_findings": {
            "full_catalog_winner_set_stable_all_cells": bool(
                full_orders["unique_winner_sets"].eq(1).all()
            ),
            "full_catalog_top_model_bm25_full_all_56_leave_one_out_cells": bool(
                full_stability["full_winner_set"].eq("bm25_full_static").all()
            ),
            "full_catalog_outcome_estimand_groups_with_invariant_complete_order": int(
                full_orders["unique_model_orders"].eq(1).sum()
            ),
            "full_catalog_outcome_estimand_groups_total": int(len(full_orders)),
            "full_catalog_complete_order_stable_all_cells": bool(
                full_orders["unique_model_orders"].eq(1).all()
            ),
            "minimum_full_catalog_winner_margin": float(
                full_orders["minimum_winner_margin"].min()
            ),
            "primary_sampled_minus_full_bias_min": float(
                primary_bias["signed_bias_sampled_minus_full"].min()
            ),
            "primary_sampled_minus_full_bias_max": float(
                primary_bias["signed_bias_sampled_minus_full"].max()
            ),
            "primary_sampled_minus_full_bias_nonnegative_fraction": float(
                primary_bias["signed_bias_sampled_minus_full"].ge(-1e-12).mean()
            ),
            "rank_winner_exact_match_rate_min_across_conditions": float(
                rank_summary["winner_set_exact_match_rate"].min()
            ),
            "rank_winner_exact_match_rate_max_across_conditions": float(
                rank_summary["winner_set_exact_match_rate"].max()
            ),
        },
        "outputs": {
            name: {"rows": int(len(frame)), "sha256": sha256(output_dir / name)}
            for name, frame in outputs.items()
        },
        "interpretation_boundary": (
            "Leaving out one provider-defined window tests dependence on any one "
            "observed window. It does not establish independence from unmeasured "
            "platform-wide temporal or market effects, and inference remains "
            "conditional on the seven observed windows."
        ),
    }
    audit_path = output_dir / "leave_one_window_out_audit.json"
    audit_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "output_dir": str(output_dir),
        "pooled_rows": len(pooled),
        "bias_rows": len(bias),
        "contrast_rows": len(contrasts),
        "rank_detail_rows": len(rank_detail),
    }, indent=2))


if __name__ == "__main__":
    main()
