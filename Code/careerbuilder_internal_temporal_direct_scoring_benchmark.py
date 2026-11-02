#!/usr/bin/env python3
"""Controlled RQ5 direct-scoring benchmark for CareerBuilder InternalTemporal.

This benchmark is deliberately separate from effectiveness evaluation.  It
compares full-catalog scoring with scoring *only* a frozen sampled candidate
set.  Model fitting, embedding encoding/loading, file I/O, and candidate
generation are outside every timed region.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

import careerbuilder_internal_temporal_evaluation as core


BENCHMARK_MODELS = (
    "tfidf_combined_static", "bm25_full_static", "minilm_title_static",
)
SELECTION_SEED = 20261020
SAMPLING_SEED = 1
SIZES = (100, 500, 1000, 5000, 10000)
KS = (5, 10, 20, 150)


class WorkingSetMonitor:
    """Poll process working set at 1 ms; numpy allocations are included."""

    def __init__(self, interval_seconds: float = 0.001):
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.baseline = 0
        self.peak = 0
        self.polls = 0
        self.thread: threading.Thread | None = None

    def __enter__(self):
        self.baseline = core.rss_bytes()
        self.peak = self.baseline

        def poll():
            while not self.stop_event.is_set():
                self.peak = max(self.peak, core.rss_bytes())
                self.polls += 1
                time.sleep(self.interval)

        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.peak = max(self.peak, core.rss_bytes())


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path,
                        default=root / "Dataset" / "CareerBuilder_InternalTemporal_v1")
    parser.add_argument("--protocol", default="First7Last2")
    parser.add_argument("--output-dir", type=Path, default=(
        root / "Results" / "CareerBuilder_InternalTemporal_EvaluationValidity"
        / "RQ5_DirectScoring_Benchmark"
    ))
    parser.add_argument("--windows", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--users-per-activity", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--semantic-model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument(
        "--authoritative-benchmark", action="store_true",
        help=(
            "Fail closed unless the complete frozen RQ5 benchmark design is "
            "requested and the output directory is fresh."
        ),
    )
    return parser.parse_args()


def select_queries(users: pd.DataFrame, window: int, n: int) -> pd.DataFrame:
    pieces = []
    for group_offset, group in enumerate((
        "cold_0", "one_1", "low_2_4", "high_5plus"
    )):
        available = users[users["activity_group"].eq(group)]
        if len(available) < n:
            raise AssertionError(f"W{window} {group}: only {len(available)} users")
        rng = np.random.default_rng(SELECTION_SEED + window * 100 + group_offset)
        chosen = rng.choice(available.index.to_numpy(), size=n, replace=False)
        pieces.append(available.loc[chosen])
    result = pd.concat(pieces).sort_values("UserID").reset_index(drop=True)
    if result["UserID"].duplicated().any() or len(result) != 4 * n:
        raise AssertionError("Benchmark query selection is not one-to-one")
    return result


def freeze_candidates(assets: dict, window: int) -> tuple[dict, dict]:
    frozen: dict[tuple[int, int | str], list[np.ndarray]] = {}
    audit = {
        "target_inclusion_failures": 0, "prior_inclusion_failures": 0,
        "nesting_failures": 0, "determinism_failures": 0,
    }
    for user in assets["users"].itertuples(index=False):
        uid = int(user.UserID)
        relevant = assets["truth"][uid]
        prior_idx = {
            assets["job_index"][job] for job in assets["prior"].get(uid, set())
            if job in assets["job_index"]
        }
        full_mask = np.ones(len(assets["job_ids"]), dtype=bool)
        if prior_idx:
            full_mask[np.fromiter(prior_idx, dtype=np.int64)] = False
        full_idx = np.flatnonzero(full_mask)
        if set(map(int, assets["job_ids"][full_idx])).intersection(
            assets["prior"].get(uid, set())
        ):
            audit["prior_inclusion_failures"] += 1
        frozen[(uid, "full")] = [full_idx]
        rel, negative, _ = core.candidate_indices(
            assets, uid, window, "uniform", SAMPLING_SEED, max(SIZES)
        )
        rel2, negative2, _ = core.candidate_indices(
            assets, uid, window, "uniform", SAMPLING_SEED, max(SIZES)
        )
        if not np.array_equal(rel, rel2) or not np.array_equal(negative, negative2):
            audit["determinism_failures"] += 1
        maximum = np.concatenate((rel, negative))
        previous: set[int] | None = None
        for size in SIZES:
            eligible = len(assets["job_ids"]) - len(prior_idx)
            actual = min(eligible, max(size, len(rel)))
            idx = maximum[:len(rel) + max(0, actual - len(rel))]
            ids = set(map(int, assets["job_ids"][idx]))
            if not relevant.issubset(ids):
                audit["target_inclusion_failures"] += 1
            prior_ids = assets["prior"].get(uid, set())
            if ids.intersection(prior_ids):
                audit["prior_inclusion_failures"] += 1
            if previous is not None and not previous.issubset(ids):
                audit["nesting_failures"] += 1
            previous = ids
            frozen[(uid, size)] = [idx]
    if any(audit.values()):
        raise AssertionError(f"Frozen benchmark candidate failure: {audit}")
    return frozen, audit


def condition_order() -> list[tuple[str, str, int | None]]:
    conditions = []
    for model in BENCHMARK_MODELS:
        conditions.append((model, "full_catalog", None))
        conditions.extend((model, "sampled_uniform", size) for size in SIZES)
    return conditions


def direct_scores(model: str, assets: dict, user_offset: int,
                  idx: np.ndarray) -> np.ndarray:
    if model == "tfidf_combined_static":
        query = assets["tfidf_users"][user_offset]
        matrix = assets["benchmark_tfidf_jobs"]
        return (query @ matrix[idx].T).toarray().ravel()
    if model == "bm25_full_static":
        query = assets["bm25_users"][user_offset]
        matrix = assets["benchmark_bm25_jobs"]
        return (query @ matrix[idx].T).toarray().ravel()
    if model == "minilm_title_static":
        query = np.asarray(assets["user_embeddings"][user_offset], dtype=np.float32)
        matrix = np.asarray(assets["job_embeddings"], dtype=np.float32)
        return np.asarray(matrix[idx]) @ query
    raise KeyError(model)


def run_condition(assets: dict, condition: tuple[str, str, int | None], frozen: dict,
                  monitor_memory: bool = True) -> dict[str, float | int]:
    model, regime, size = condition
    score_seconds = rank_seconds = metric_seconds = 0.0
    checksum = 0.0
    pairs = 0
    monitor = WorkingSetMonitor() if monitor_memory else None
    if monitor:
        monitor.__enter__()
    total_started = time.perf_counter()
    try:
        for offset, user in enumerate(assets["users"].itertuples(index=False)):
            uid = int(user.UserID)
            relevant = assets["truth"][uid]
            idx = frozen[(uid, "full")][0] if regime == "full_catalog" \
                else frozen[(uid, int(size))][0]
            score_started = time.perf_counter()
            raw_scores = direct_scores(model, assets, offset, idx)
            score_seconds += time.perf_counter() - score_started
            if len(raw_scores) != len(idx):
                raise AssertionError(
                    "Direct scorer did not return exactly one score per declared candidate"
                )
            # Both regimes score exactly their declared candidate indices.
            # The full regime is therefore catalog minus this user's prior jobs.
            scores = raw_scores
            pairs += len(idx)
            ids = assets["job_ids"][idx]
            valid_ks = tuple(k for k in KS if k <= len(idx))
            rank_started = time.perf_counter()
            ranked = core.top_with_deterministic_ties(ids, scores, max(valid_ks))
            rank_seconds += time.perf_counter() - rank_started
            metric_started = time.perf_counter()
            values = list(core.metric_rows(ranked, relevant, valid_ks))
            metric_seconds += time.perf_counter() - metric_started
            checksum += sum(float(row["map"]) for row in values)
    finally:
        total_seconds = time.perf_counter() - total_started
        if monitor:
            monitor.__exit__(None, None, None)
    return {
        "queries": len(assets["users"]), "scored_pairs": pairs,
        "score_seconds": score_seconds, "rank_seconds": rank_seconds,
        "metric_seconds": metric_seconds, "total_seconds": total_seconds,
        "working_set_baseline_bytes": monitor.baseline if monitor else 0,
        "working_set_peak_bytes": monitor.peak if monitor else 0,
        "working_set_peak_delta_bytes": (
            max(0, monitor.peak - monitor.baseline) if monitor else 0
        ),
        "working_set_poll_count": monitor.polls if monitor else 0,
        "working_set_requested_interval_seconds": (
            monitor.interval if monitor else 0.0
        ),
        "metric_checksum": checksum,
    }


def environment_metadata() -> dict[str, object]:
    torch_version = None
    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        pass
    return {
        "platform": platform.platform(), "processor": platform.processor(),
        "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "torch": torch_version,
        "thread_environment": {
            name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM",
            )
        },
        "memory_sampling": "Windows working set polled every 1 ms",
    }


def main() -> None:
    invocation_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    args = parse_args()
    args.windows = sorted(set(args.windows))
    if args.authoritative_benchmark:
        expected = {
            "protocol": "First7Last2",
            "windows": list(range(1, 8)),
            "users_per_activity": 5,
            "repeats": 5,
            "max_features": 50_000,
            "semantic_model": core.MINILM_MODEL,
            "embedding_batch_size": 512,
        }
        actual = {
            "protocol": args.protocol,
            "windows": args.windows,
            "users_per_activity": args.users_per_activity,
            "repeats": args.repeats,
            "max_features": args.max_features,
            "semantic_model": args.semantic_model,
            "embedding_batch_size": args.embedding_batch_size,
        }
        if actual != expected:
            raise RuntimeError(
                "--authoritative-benchmark requires the complete frozen RQ5 "
                f"design; actual={actual}"
            )
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise RuntimeError(
                "Authoritative benchmark output must be a fresh empty directory: "
                f"{args.output_dir}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semantic_snapshot_evidence = core.resolve_minilm_snapshot(args.semantic_model)
    script_path = Path(__file__).resolve()
    core_path = Path(core.__file__).resolve()
    preparation_manifest_path = (
        args.dataset_dir / "INTERNAL_TEMPORAL_PREPARATION_MANIFEST.json"
    )
    if not preparation_manifest_path.exists():
        raise FileNotFoundError(preparation_manifest_path)
    config = {
        "purpose": "RQ5 controlled direct-scoring benchmark",
        "not_effectiveness_runtime": True,
        "authority": {
            "authoritative": bool(args.authoritative_benchmark),
            "study_role": "direct_scoring_benchmark",
            "status": "configured",
        },
        "started_utc": started_utc,
        "protocol": args.protocol,
        "windows": args.windows,
        "selection_seed": SELECTION_SEED, "users_per_activity_per_window": args.users_per_activity,
        "sampling_strategy": "uniform", "sampling_seed": SAMPLING_SEED,
        "candidate_sizes": list(SIZES), "ks": list(KS),
        "models": list(BENCHMARK_MODELS), "primary_repeats": args.repeats,
        "semantic_model": args.semantic_model,
        "semantic_revision": core.MINILM_EXPECTED_REVISION,
        "semantic_snapshot_evidence": semantic_snapshot_evidence,
        "benchmark_script_path": str(script_path),
        "benchmark_script_sha256": core.sha256(script_path),
        "evaluator_module_path": str(core_path),
        "evaluator_module_sha256": core.sha256(core_path),
        "preparation_manifest_path": str(preparation_manifest_path.resolve()),
        "preparation_manifest_sha256": core.sha256(preparation_manifest_path),
        "execution_order": (
            "one labeled cold_first_diagnostic pass; one untimed warm-up; "
            "five primary warm-cache passes with cyclically rotated condition order"
        ),
        "timed_boundary": (
            "direct score, deterministic descending-score/ascending-JobID rank, and metrics; "
            "excludes I/O, model fitting, embedding encode/load, and candidate generation"
        ),
        "exact_set_rule": (
            "full scores only catalog-minus-user-prior indices; sampled scores only frozen "
            "sampled indices; one returned score and one counted pair per declared index"
        ),
        "environment": environment_metadata(),
        "command_equivalent_arguments": {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "protocol": args.protocol,
            "output_dir": str(args.output_dir.resolve()),
            "windows": args.windows,
            "users_per_activity": args.users_per_activity,
            "repeats": args.repeats,
            "max_features": args.max_features,
            "semantic_model": args.semantic_model,
            "embedding_batch_size": args.embedding_batch_size,
            "authoritative_benchmark": args.authoritative_benchmark,
        },
    }
    (args.output_dir / "BENCHMARK_CONFIG.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    rows = []
    selection_rows = []
    integrity_by_window = {}
    model_preparation_by_window = {}
    conditions = condition_order()
    for window in args.windows:
        loader_args = argparse.Namespace(
            dataset_dir=args.dataset_dir, protocol=args.protocol,
            smoke_users_per_activity=None, max_features=args.max_features,
            skip_semantic=False, semantic_model=args.semantic_model,
            embedding_batch_size=args.embedding_batch_size,
            semantic_revision=core.MINILM_EXPECTED_REVISION,
            semantic_snapshot_evidence=semantic_snapshot_evidence,
        )
        assets = core.load_window(loader_args, window, args.output_dir)
        assets["window"] = window
        selected_once = select_queries(
            assets["users"], window, args.users_per_activity
        )
        selected_twice = select_queries(
            assets["users"], window, args.users_per_activity
        )
        if not selected_once["UserID"].equals(selected_twice["UserID"]):
            raise AssertionError(f"W{window}: benchmark query selection is not deterministic")
        assets["users"] = selected_once
        selected_ids = set(assets["users"]["UserID"].astype(int))
        if not selected_ids.issubset(set(assets["population"]["UserID"].astype(int))):
            raise AssertionError("Benchmark user outside population")
        selection_rows.append(assets["users"][[
            "UserID", "activity_group", "precut_application_count",
            "inclusion_probability", "population_weight",
        ]].assign(window=window))
        frozen, candidate_audit = freeze_candidates(assets, window)
        integrity_by_window[str(window)] = candidate_audit
        model_preparation_by_window[str(window)] = core.prepare_models(
            loader_args, assets, args.output_dir
        )
        assets["benchmark_tfidf_jobs"] = assets["tfidf_jobs_t"].T.tocsr()
        assets["benchmark_bm25_jobs"] = assets["bm25_jobs_t"].T.tocsr()

        # Separately labeled first-touch diagnostic; excluded from summaries.
        for condition in conditions:
            result = run_condition(assets, condition, frozen)
            rows.append({
                "window": window, "phase": "cold_first_diagnostic", "repeat": 0,
                "model": condition[0], "candidate_regime": condition[1],
                "requested_candidate_size": condition[2], **result,
            })

        # Untimed warm-up after diagnostics, as prespecified.
        for condition in reversed(conditions):
            run_condition(assets, condition, frozen, monitor_memory=False)

        # Rotate order by one condition each repeat to reduce order confounding.
        for repeat in range(1, args.repeats + 1):
            offset = (repeat - 1) % len(conditions)
            rotated = conditions[offset:] + conditions[:offset]
            for order, condition in enumerate(rotated):
                result = run_condition(assets, condition, frozen)
                rows.append({
                    "window": window, "phase": "primary_warm_cache", "repeat": repeat,
                    "rotated_order": order, "model": condition[0],
                    "candidate_regime": condition[1],
                    "requested_candidate_size": condition[2], **result,
                })
        pd.DataFrame(rows).to_csv(
            args.output_dir / "benchmark_results.partial.csv", index=False
        )

    results = pd.DataFrame(rows)
    expected_diagnostic_rows = len(args.windows) * len(conditions)
    expected_primary_rows = expected_diagnostic_rows * args.repeats
    expected_total_rows = expected_diagnostic_rows + expected_primary_rows
    key_columns = [
        "window", "phase", "repeat", "model", "candidate_regime",
        "requested_candidate_size",
    ]
    duplicate_result_keys = int(results.duplicated(key_columns).sum())
    diagnostic_rows = int(results["phase"].eq("cold_first_diagnostic").sum())
    primary_rows = int(results["phase"].eq("primary_warm_cache").sum())
    if len(results) != expected_total_rows or diagnostic_rows != expected_diagnostic_rows \
            or primary_rows != expected_primary_rows or duplicate_result_keys:
        raise AssertionError({
            "rows": len(results), "expected_total_rows": expected_total_rows,
            "diagnostic_rows": diagnostic_rows,
            "expected_diagnostic_rows": expected_diagnostic_rows,
            "primary_rows": primary_rows,
            "expected_primary_rows": expected_primary_rows,
            "duplicate_result_keys": duplicate_result_keys,
        })
    invariant_groups = [
        "window", "model", "candidate_regime", "requested_candidate_size",
    ]
    invariants = results.groupby(invariant_groups, dropna=False).agg(
        scored_pairs_min=("scored_pairs", "min"),
        scored_pairs_max=("scored_pairs", "max"),
        checksum_min=("metric_checksum", "min"),
        checksum_max=("metric_checksum", "max"),
    ).reset_index()
    pair_invariance_failures = int((
        invariants["scored_pairs_min"] != invariants["scored_pairs_max"]
    ).sum())
    checksum_invariance_failures = int((
        (invariants["checksum_max"] - invariants["checksum_min"]).abs() > 1e-12
    ).sum())
    if pair_invariance_failures or checksum_invariance_failures:
        raise AssertionError({
            "pair_invariance_failures": pair_invariance_failures,
            "checksum_invariance_failures": checksum_invariance_failures,
        })
    results.to_csv(args.output_dir / "benchmark_results.csv", index=False)
    selection = pd.concat(selection_rows, ignore_index=True)
    selection.to_csv(args.output_dir / "benchmark_query_selection.csv", index=False)
    primary = results[results["phase"].eq("primary_warm_cache")]
    summary = primary.groupby([
        "model", "candidate_regime", "requested_candidate_size",
    ], dropna=False, as_index=False).agg(
        measurements=("total_seconds", "size"),
        total_seconds_mean=("total_seconds", "mean"),
        total_seconds_sd=("total_seconds", "std"),
        score_seconds_mean=("score_seconds", "mean"),
        rank_seconds_mean=("rank_seconds", "mean"),
        metric_seconds_mean=("metric_seconds", "mean"),
        scored_pairs_mean=("scored_pairs", "mean"),
        peak_delta_bytes_max=("working_set_peak_delta_bytes", "max"),
    )
    summary.to_csv(args.output_dir / "benchmark_primary_summary.csv", index=False)
    window_summary = primary.groupby([
        "window", "model", "candidate_regime", "requested_candidate_size",
    ], dropna=False, as_index=False).agg(
        measurements=("total_seconds", "size"),
        total_seconds_mean=("total_seconds", "mean"),
        total_seconds_sd=("total_seconds", "std"),
        score_seconds_mean=("score_seconds", "mean"),
        rank_seconds_mean=("rank_seconds", "mean"),
        metric_seconds_mean=("metric_seconds", "mean"),
        scored_pairs=("scored_pairs", "first"),
        peak_delta_bytes_max=("working_set_peak_delta_bytes", "max"),
        working_set_poll_count_min=("working_set_poll_count", "min"),
    )
    window_summary.to_csv(
        args.output_dir / "benchmark_primary_window_summary.csv", index=False
    )
    audit = {
        "selection_rows": len(selection),
        "expected_selection_rows": len(set(args.windows)) * args.users_per_activity * 4,
        "selection_activity_counts": {
            str(k): int(v) for k, v in selection["activity_group"].value_counts().items()
        },
        "expected_selection_per_activity": len(set(args.windows)) * args.users_per_activity,
        "candidate_integrity_by_window": integrity_by_window,
        "diagnostic_rows_excluded_from_primary_summary": True,
        "primary_repeats": args.repeats,
        "result_rows": len(results),
        "expected_result_rows": expected_total_rows,
        "diagnostic_rows": diagnostic_rows,
        "primary_rows": primary_rows,
        "duplicate_result_keys": duplicate_result_keys,
        "pair_invariance_failures": pair_invariance_failures,
        "checksum_invariance_failures": checksum_invariance_failures,
        "selection_is_deterministic": True,
        "same_frozen_candidate_sets_for_all_models": True,
        "full_scores_freshly_computed": True,
        "sampled_scores_direct_not_indexed_from_full": True,
    }
    if audit["selection_rows"] != audit["expected_selection_rows"] or any(
        audit["selection_activity_counts"].get(group, 0)
        != audit["expected_selection_per_activity"]
        for group in ("cold_0", "one_1", "low_2_4", "high_5plus")
    ):
        raise AssertionError(audit)
    (args.output_dir / "BENCHMARK_INTEGRITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    output_names = [
        "BENCHMARK_CONFIG.json", "benchmark_results.csv",
        "benchmark_query_selection.csv", "benchmark_primary_summary.csv",
        "benchmark_primary_window_summary.csv", "BENCHMARK_INTEGRITY_AUDIT.json",
    ]
    run_report = {
        "authority": {
            "authoritative": bool(args.authoritative_benchmark),
            "study_role": "direct_scoring_benchmark",
            "status": "completed",
        },
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - invocation_started,
        "config_sha256": core.sha256(args.output_dir / "BENCHMARK_CONFIG.json"),
        "model_preparation_by_window": model_preparation_by_window,
        "integrity": audit,
        "outputs": {
            name: {
                "bytes": (args.output_dir / name).stat().st_size,
                "sha256": core.sha256(args.output_dir / name),
            }
            for name in output_names
        },
    }
    (args.output_dir / "BENCHMARK_RUN_REPORT.json").write_text(
        json.dumps(run_report, indent=2), encoding="utf-8"
    )
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
