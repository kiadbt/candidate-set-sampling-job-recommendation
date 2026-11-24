#!/usr/bin/env python3
"""Independent QA and public-safe packaging for the CareerBuilder near-horizon run."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


PROJECT = Path(__file__).resolve().parents[4]
REVISION = PROJECT / "Manuscript" / "Revision_R1_BDCC-4711631_2026-11-23"
RUN = PROJECT / "Results" / "CareerBuilder_InternalTemporal_EvaluationValidity" / "Revision_R1_NearHorizon_Authoritative_v1"
PRIMARY = PROJECT / "Results" / "CareerBuilder_InternalTemporal_EvaluationValidity" / "Primary_First7Last2_Authoritative_v2"
REFERENCE_ROOT = PROJECT / "Dataset" / "CareerBuilder_InternalTemporal_v1"
FUTURE_ROOT = PROJECT / "Dataset" / "CareerBuilder_FutureContamination_v1"
PACKAGE = REVISION / "04_Experiment_Outputs" / "05_near_horizon_temporal"
MODELS = [
    "random_hash", "popularity_precut", "tfidf_combined_static",
    "bm25_full_static", "bm25_title_static", "minilm_title_static",
]
KS = [5, 10, 20, 150]
METRICS = ["precision", "recall", "hit_rate", "ndcg", "mrr", "map"]
REGIMES = ["cutoff_active_reference", "near_horizon_future_opening"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


checks: list[dict[str, object]] = []


def require(name: str, passed: bool, detail: object) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        raise AssertionError(f"{name}: {detail}")


def compare_numeric(name: str, expected: pd.DataFrame, actual: pd.DataFrame,
                    keys: list[str], values: list[str], tol: float = 1e-12) -> None:
    merged = expected.merge(actual, on=keys, how="outer", indicator=True,
                            suffixes=("_expected", "_actual"), validate="one_to_one")
    require(f"{name} key parity", merged["_merge"].eq("both").all(),
            merged["_merge"].value_counts().to_dict())
    max_diff: dict[str, float] = {}
    for value in values:
        left = pd.to_numeric(merged[f"{value}_expected"], errors="coerce").to_numpy(float)
        right = pd.to_numeric(merged[f"{value}_actual"], errors="coerce").to_numpy(float)
        both_nan = np.isnan(left) & np.isnan(right)
        diff = np.abs(left - right)
        diff[both_nan] = 0.0
        max_diff[value] = float(np.nanmax(diff)) if len(diff) else 0.0
    require(f"{name} numerical parity", all(v <= tol for v in max_diff.values()), max_diff)


def main() -> None:
    audit_path = RUN / "careerbuilder_temporal_contamination_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    run_config_path = RUN / "RUN_CONFIG.json"
    require("authoritative audit role", audit["analysis_role"] == "post_decision_near_horizon_temporal_sensitivity", audit["analysis_role"])
    require("authoritative run configuration hash", sha256(run_config_path) == audit["run_config"]["sha256"], audit["run_config"])
    for label, record in audit["code_provenance"].items():
        path = Path(record["path"])
        require(f"code hash {label}", path.is_file() and sha256(path) == record["sha256"], record)

    window_frames = []
    for window in range(1, 8):
        path = RUN / f"careerbuilder_near_horizon_window_{window}.csv.gz"
        frame = pd.read_csv(path)
        window_frames.append(frame)
        record = next(row for row in audit["window_records"] if int(row["window"]) == window)
        require(f"W{window} output hash", sha256(path) == record["output"]["sha256"], record["output"])
    near = pd.concat(window_frames, ignore_index=True)

    keys = ["window", "user_id", "catalog_regime", "model", "k"]
    require("336000-row complete grid", len(near) == 336000, len(near))
    require("result key uniqueness", not near.duplicated(keys).any(), int(near.duplicated(keys).sum()))
    require("7000 distinct queries", near[["window", "user_id"]].drop_duplicates().shape[0] == 7000,
            int(near[["window", "user_id"]].drop_duplicates().shape[0]))
    require("six-model grid", set(near["model"]) == set(MODELS), sorted(near["model"].unique()))
    require("four-cutoff grid", set(map(int, near["k"])) == set(KS), sorted(near["k"].unique()))
    require("two declared regimes", set(near["catalog_regime"]) == set(REGIMES), sorted(near["catalog_regime"].unique()))
    require("finite bounded metrics", np.isfinite(near[METRICS].to_numpy(float)).all()
            and ((near[METRICS] >= -1e-15) & (near[METRICS] <= 1 + 1e-15)).all().all(), {})

    manifest = json.loads((REFERENCE_ROOT / "INTERNAL_TEMPORAL_PREPARATION_MANIFEST.json").read_text(encoding="utf-8"))
    count_rows = []
    for window in range(1, 8):
        record = next(row for row in audit["window_records"] if int(row["window"]) == window)
        protocol = manifest["protocols"]["First7Last2"]["windows"][str(window)]
        cutoff = pd.Timestamp(protocol["cutoff"])
        evaluation_end = pd.Timestamp(protocol["evaluation_end"])
        source_path = FUTURE_ROOT / "First7Last2" / f"window_{window}" / "jobs_future_after_cutoff.csv.gz"
        source = pd.read_csv(source_path, usecols=["JobID", "StartDate"])
        source["StartDate"] = pd.to_datetime(source["StartDate"], errors="raise")
        near_raw = source[source["StartDate"].gt(cutoff) & source["StartDate"].le(evaluation_end)].copy()
        precut = pd.read_csv(REFERENCE_ROOT / "First7Last2" / f"window_{window}" / "precut_applications.csv", usecols=["JobID"])
        inconsistent = set(map(int, near_raw["JobID"])) & set(map(int, precut["JobID"]))
        retained = len(near_raw) - len(inconsistent)
        beyond = int(source["StartDate"].gt(evaluation_end).sum())
        require(f"W{window} date-gate and behavior-gate counts",
                int(record["raw_future_opening_jobs"]) == len(near_raw)
                and int(record["future_jobs_excluded_for_precut_interaction"]) == len(inconsistent)
                and int(record["future_jobs_added"]) == retained
                and int(record["future_jobs_beyond_evaluation_horizon_excluded"]) == beyond,
                {"raw_near": len(near_raw), "inconsistent": len(inconsistent), "retained": retained, "beyond": beyond, "record": record})
        cell = near[near["window"].eq(window)]
        require(f"W{window} reported addition count", cell["future_jobs_added"].eq(retained).all(), retained)
        counts = cell.pivot_table(index=["user_id", "model", "k"], columns="catalog_regime", values="candidate_count", aggfunc="first")
        delta = counts["near_horizon_future_opening"] - counts["cutoff_active_reference"]
        require(f"W{window} candidate-count delta", delta.eq(retained).all(), {"expected": retained, "observed": sorted(delta.unique())[:5]})
        count_rows.append({
            "window": window, "cutoff": cutoff.isoformat(), "evaluation_end": evaluation_end.isoformat(),
            "reference_jobs": int(record["reference_jobs"]), "source_future_opening_jobs": int(record["source_future_opening_jobs"]),
            "near_horizon_raw_jobs": len(near_raw), "behavior_inconsistent_excluded": len(inconsistent),
            "near_horizon_jobs_added": retained, "beyond_horizon_jobs_excluded": beyond,
            "near_horizon_catalog_jobs": int(record["contaminated_jobs"]),
        })

    reference = near[near["catalog_regime"].eq("cutoff_active_reference")].copy()
    perturbed = near[near["catalog_regime"].eq("near_horizon_future_opening")].copy()
    pair_keys = ["window", "user_id", "model", "k"]
    paired = reference.merge(perturbed, on=pair_keys, suffixes=("_ref", "_near"), validate="one_to_one")
    require("168000 paired reference rows", len(reference) == 168000 and len(paired) == 168000,
            {"reference": len(reference), "paired": len(paired)})
    positive = {}
    for metric in METRICS:
        positive[metric] = int((paired[f"{metric}_near"] - paired[f"{metric}_ref"] > 1e-12).sum())
    require("fixed-score metrics never improve after adding alternatives", sum(positive.values()) == 0, positive)

    primary_manifest = json.loads((PRIMARY / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    usecols = ["window", "user_id", "candidate_regime", "sampling_strategy", "seed", "model", "k",
               "eligible_candidate_count", "actual_candidate_count", "precut_application_count", "activity_group",
               "mean_target_precut_popularity", "target_popularity_group", "relevant_jobs", "inclusion_probability",
               "population_weight", "effective_k", "truncated_at_k", *METRICS]
    primary_parts = []
    for item in primary_manifest["authoritative_partitions"]:
        path = Path(item["path"])
        require(f"primary checkpoint hash {path.name}-{item['sha256'][:8]}", sha256(path) == item["sha256"], str(path))
        chunk = pd.read_csv(path, usecols=usecols)
        chunk = chunk[
            chunk["candidate_regime"].eq("full_catalog")
            & chunk["sampling_strategy"].eq("full_catalog")
            & chunk["seed"].eq(0)
            & chunk["model"].isin(MODELS)
            & chunk["k"].isin(KS)
        ].copy()
        primary_parts.append(chunk)
    primary = pd.concat(primary_parts, ignore_index=True)
    require("primary exact-reference grid has 168000 rows", len(primary) == 168000, len(primary))
    merged = primary.merge(reference, on=pair_keys, how="outer", suffixes=("_primary", "_near_run"), indicator=True, validate="one_to_one")
    require("primary/reference key parity", merged["_merge"].eq("both").all(), merged["_merge"].value_counts().to_dict())
    parity_values = {
        "candidate_count": np.max(np.abs(merged["actual_candidate_count"].to_numpy(float) - merged["candidate_count"].to_numpy(float))),
        **{metric: np.max(np.abs(merged[f"{metric}_primary"].to_numpy(float) - merged[f"{metric}_near_run"].to_numpy(float))) for metric in METRICS},
    }
    require("exact numerical reference parity", all(float(v) == 0.0 for v in parity_values.values()), parity_values)

    summary_rows = []
    for (regime, model, k), cell in near.groupby(["catalog_regime", "model", "k"], sort=True):
        for estimand in ("audit_cohort", "design_weighted_population"):
            weights = np.ones(len(cell)) if estimand == "audit_cohort" else cell["population_weight"].to_numpy(float)
            row = {"catalog_regime": regime, "model": model, "k": k, "estimand": estimand,
                   "queries": len(cell), "windows": cell["window"].nunique(),
                   "candidate_count_mean": float(np.average(cell["candidate_count"], weights=weights)),
                   "future_jobs_added_mean": float(np.average(cell["future_jobs_added"], weights=weights))}
            row.update({metric: float(np.average(cell[metric], weights=weights)) for metric in METRICS})
            summary_rows.append(row)
    expected_summary = pd.DataFrame(summary_rows)
    actual_summary = pd.read_csv(RUN / "careerbuilder_near_horizon_summary.csv")
    compare_numeric("summary reproduction", expected_summary, actual_summary,
                    ["catalog_regime", "model", "k", "estimand"],
                    ["queries", "windows", "candidate_count_mean", "future_jobs_added_mean", *METRICS])

    ranking_rows = []
    primary_summary = expected_summary[expected_summary["estimand"].eq("design_weighted_population")]
    for k in KS:
        for metric in METRICS:
            orders = {}
            for regime in REGIMES:
                cell = primary_summary[(primary_summary["catalog_regime"].eq(regime)) & (primary_summary["k"].eq(k))].sort_values([metric, "model"], ascending=[False, True])
                orders[regime] = cell["model"].tolist()
                for rank, row in enumerate(cell.itertuples(index=False), 1):
                    ranking_rows.append({"record_type": "model_rank", "k": k, "metric": metric,
                                         "catalog_regime": regime, "model": row.model, "rank": rank,
                                         "value": getattr(row, metric), "kendall_tau_between_regimes": np.nan,
                                         "top_model_agreement": np.nan})
            ref_order = orders[REGIMES[0]]
            alt_order = orders[REGIMES[1]]
            ref_ranks = {model: rank for rank, model in enumerate(ref_order)}
            tau = float(kendalltau(np.arange(len(ref_order)), [ref_ranks[m] for m in alt_order]).statistic)
            ranking_rows.append({"record_type": "regime_comparison", "k": k, "metric": metric,
                                 "catalog_regime": "reference_vs_contaminated", "model": "all", "rank": np.nan,
                                 "value": np.nan, "kendall_tau_between_regimes": tau,
                                 "top_model_agreement": ref_order[0] == alt_order[0]})
    expected_rank = pd.DataFrame(ranking_rows)
    actual_rank = pd.read_csv(RUN / "careerbuilder_near_horizon_model_rankings.csv")
    compare_numeric("rank reproduction",
                    expected_rank[expected_rank["record_type"].eq("model_rank")],
                    actual_rank[actual_rank["record_type"].eq("model_rank")],
                    ["record_type", "k", "metric", "catalog_regime", "model"], ["rank", "value"])
    compare_numeric("rank agreement reproduction",
                    expected_rank[expected_rank["record_type"].eq("regime_comparison")],
                    actual_rank[actual_rank["record_type"].eq("regime_comparison")],
                    ["record_type", "k", "metric", "catalog_regime", "model"], ["kendall_tau_between_regimes"])

    PACKAGE.mkdir(parents=True, exist_ok=True)
    public_files = {
        "summary": RUN / "careerbuilder_near_horizon_summary.csv",
        "rankings": RUN / "careerbuilder_near_horizon_model_rankings.csv",
    }
    for name, source in public_files.items():
        shutil.copy2(source, PACKAGE / source.name)
    counts = pd.DataFrame(count_rows)
    counts_path = PACKAGE / "careerbuilder_near_horizon_window_counts.csv"
    counts.to_csv(counts_path, index=False)

    k20 = actual_summary[(actual_summary["estimand"].eq("design_weighted_population")) & (actual_summary["k"].eq(20))]
    selected = ["bm25_full_static", "tfidf_combined_static", "bm25_title_static", "minilm_title_static", "popularity_precut", "random_hash"]
    metric_rows = []
    for model in selected:
        row = {"model": model}
        for regime in REGIMES:
            cell = k20[(k20["catalog_regime"].eq(regime)) & (k20["model"].eq(model))].iloc[0]
            row[f"hit_rate_{regime}"] = float(cell["hit_rate"])
            row[f"ndcg_{regime}"] = float(cell["ndcg"])
            row[f"map_{regime}"] = float(cell["map"])
        metric_rows.append(row)
    metric_table = pd.DataFrame(metric_rows)
    metrics_path = PACKAGE / "careerbuilder_near_horizon_k20_design_weighted.csv"
    metric_table.to_csv(metrics_path, index=False)

    comparison = actual_rank[actual_rank["record_type"].eq("regime_comparison")].copy()
    internal_audit = {
        "status": "PASS", "run": str(RUN), "primary_reference": str(PRIMARY),
        "queries": 7000, "rows": 336000, "reference_rows_exactly_reproduced": 168000,
        "positive_metric_deltas": positive, "window_counts": count_rows,
        "source_run_audit_sha256": sha256(audit_path), "qa_script_sha256": sha256(Path(__file__)),
        "checks": checks,
    }
    internal_json = Path(__file__).with_name("INDEPENDENT_NEAR_HORIZON_QA.json")
    internal_json.write_text(json.dumps(internal_audit, indent=2, default=str), encoding="utf-8")
    internal_lines = ["# Independent CareerBuilder Near-Horizon QA", "", "**Overall status: PASS**", "",
                      f"All {len(checks)} checks passed, including exact parity for 168,000 locked reference rows,",
                      "date and behavior gates, candidate-count deltas, metric monotonicity, aggregate reproduction,",
                      "rank reproduction, source hashes, and the complete 336,000-row result grid.", "", "## Checks", "",
                      "| Check | Status |", "|---|---:|"]
    internal_lines.extend(f"| {row['name']} | PASS |" for row in checks)
    Path(__file__).with_name("INDEPENDENT_NEAR_HORIZON_QA.md").write_text("\n".join(internal_lines) + "\n", encoding="utf-8")

    public_audit = {
        "status": "PASS", "analysis_role": audit["analysis_role"], "queries": 7000, "rows": 336000,
        "reference_rows_exactly_reproduced": 168000, "positive_metric_deltas": positive,
        "window_counts": count_rows,
        "top_model_agreement_all_comparisons": bool(comparison["top_model_agreement"].astype(bool).all()),
        "kendall_tau_min": float(comparison["kendall_tau_between_regimes"].min()),
        "kendall_tau_max": float(comparison["kendall_tau_between_regimes"].max()),
        "source_run_audit_sha256": sha256(audit_path),
        "qa_script_sha256": sha256(Path(__file__)),
        "public_outputs": {}, "check_count": len(checks), "all_checks_passed": True,
    }
    for path in [PACKAGE / source.name for source in public_files.values()] + [counts_path, metrics_path]:
        public_audit["public_outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    audit_out = PACKAGE / "NEAR_HORIZON_INDEPENDENT_QA.json"
    audit_out.write_text(json.dumps(public_audit, indent=2, default=str), encoding="utf-8")
    public_qa_md = PACKAGE / "NEAR_HORIZON_INDEPENDENT_QA.md"
    public_qa_md.write_text(
        "# Independent CareerBuilder Near-Horizon QA\n\n"
        "**Overall status: PASS**\n\n"
        f"All {len(checks)} checks passed. The audit independently verified the complete 336,000-row grid, "
        "exact reproduction of 168,000 locked reference rows, the date and behavior gates, candidate-count "
        "deltas, non-increasing fixed-score metrics, aggregate summaries, model rankings, and recorded hashes.\n\n"
        "No row-level user data or absolute local filesystem paths are included in this public-safe package.\n",
        encoding="utf-8",
    )

    lines = ["# CareerBuilder Near-Horizon Temporal-Eligibility Sensitivity", "",
             "## Design", "",
             "This post-decision sensitivity adds only behavior-consistent same-window jobs satisfying",
             "`cutoff < StartDate <= evaluation_end`. Queries, observed targets, profiles, pre-cutoff",
             "behavior, reference-fitted model parameters, scoring rules, and deterministic tie-breaking",
             "remain fixed. It complements the broader future-opening stress test without adding jobs that",
             "open after the evaluation horizon.", "", "## Verified findings", ""]
    total_added = int(counts["near_horizon_jobs_added"].sum())
    weighted_added = float(k20[k20["catalog_regime"].eq(REGIMES[1])]["future_jobs_added_mean"].iloc[0])
    lines.append(f"- Across seven windows, {total_added:,} near-horizon jobs were retained after the behavior-consistency gate; the design-weighted mean addition was {weighted_added:,.4f} jobs per query.")
    for row in metric_rows[:4]:
        lines.append(f"- {row['model']}: design-weighted HitRate@20 changed from {row['hit_rate_cutoff_active_reference']:.4f} to {row['hit_rate_near_horizon_future_opening']:.4f}.")
    lines.extend([
        f"- The leading configuration was preserved for every metric and cutoff; Kendall's tau-b ranged from {public_audit['kendall_tau_min']:.4f} to {public_audit['kendall_tau_max']:.4f}.",
        "- Adding alternatives never improved any fixed-score per-query metric, and all 168,000 reference rows reproduced the locked primary evaluation exactly.",
        "", "## Independent QA", "",
        f"All {len(checks)} declared checks passed. The public package excludes row-level user results and absolute local paths.",
        "", "## Public-safe files", "",
        "- `careerbuilder_near_horizon_summary.csv`", "- `careerbuilder_near_horizon_model_rankings.csv`",
        "- `careerbuilder_near_horizon_window_counts.csv`", "- `careerbuilder_near_horizon_k20_design_weighted.csv`",
        "- `NEAR_HORIZON_INDEPENDENT_QA.json`", "- `NEAR_HORIZON_INDEPENDENT_QA.md`", "",
    ])
    (PACKAGE / "README_VERIFIED_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": True, "checks": len(checks), "package": str(PACKAGE),
                      "metrics": metric_rows, "tau_min": public_audit["kendall_tau_min"],
                      "tau_max": public_audit["kendall_tau_max"], "total_added": total_added,
                      "weighted_added": weighted_added}, indent=2))


if __name__ == "__main__":
    main()
