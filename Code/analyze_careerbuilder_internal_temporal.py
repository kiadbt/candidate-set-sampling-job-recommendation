#!/usr/bin/env python3
"""Fail-closed analysis of authoritative CareerBuilder internal-temporal runs.

This program analyzes only the internally reconstructed temporal task produced
by ``careerbuilder_internal_temporal_evaluation.py``.  It never treats those
outputs as official CareerBuilder competition results.  A run is unreadable
unless its completed report explicitly marks it as authoritative production
and all configuration, partition, integrity, coverage, and hash checks pass.

Examples
--------
Primary only::

    python Code/analyze_careerbuilder_internal_temporal.py \
      --run First7Last2=Results/CareerBuilder_InternalTemporal_EvaluationValidity/Primary_First7Last2_Authoritative_v2 \
      --primary-label First7Last2 \
      --output-dir Results/CareerBuilder_InternalTemporal_Authoritative_Analysis_v1

Primary plus independently authoritative sensitivity runs::

    python Code/analyze_careerbuilder_internal_temporal.py \
      --run First7Last2=Results/.../Primary_First7Last2_Audit1000_v1 \
      --run First6Last3=Results/.../Sensitivity_First6Last3_Audit250_v1 \
      --run First5Last4=Results/.../Sensitivity_First5Last4_Audit250_v1 \
      --primary-label First7Last2

The script intentionally has no smoke/development override.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
from typing import Iterable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "Results" / "CareerBuilder_InternalTemporal_Statistical_Analysis"
)

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
MODEL_COLORS = {
    "random_hash": "#7F7F7F",
    "popularity_precut": "#0072B2",
    "tfidf_combined_static": "#009E73",
    "bm25_full_static": "#D55E00",
    "bm25_title_static": "#E69F00",
    "minilm_title_static": "#CC79A7",
}
STRATEGIES = ("uniform", "popularity_weighted")
STRATEGY_LABELS = {
    "uniform": "Uniform negatives",
    "popularity_weighted": "Popularity-weighted negatives",
}
STRATEGY_STYLES = {"uniform": "-", "popularity_weighted": "--"}
SIZES = (100, 500, 1000, 5000, 10000)
KS = (5, 10, 20, 150)
SEEDS = (1, 2, 3, 4, 5)
WINDOWS = tuple(range(1, 8))
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
CONTRAST_LABELS = {
    "BM25-title_minus_MiniLM-title": "Title BM25 − title MiniLM",
    "BM25-full_minus_MiniLM-title": "Full-text BM25 − title MiniLM",
    "BM25-full_minus_TFIDF-combined": "Full-text BM25 − combined TF-IDF",
    "BM25-full_minus_popularity": "Full-text BM25 − popularity",
    "BM25-full_minus_random": "Full-text BM25 − random",
}
ACTIVITY_GROUPS = ("cold_0", "one_1", "low_2_4", "high_5plus")
TARGET_POPULARITY_GROUPS = (
    "mean_unseen_0",
    "mean_positive_le2",
    "mean_gt2",
)
SUBGROUP_LABELS = {
    "cold_0": "0 applications",
    "one_1": "1 application",
    "low_2_4": "2–4 applications",
    "high_5plus": "5+ applications",
    "mean_unseen_0": "Unseen (0)",
    "mean_positive_le2": "Rare (>0–2)",
    "mean_gt2": "Popular (>2)",
}

REQUIRED_COLUMNS = (
    "protocol", "window", "user_id", "precut_application_count",
    "activity_group", "mean_target_precut_popularity",
    "target_popularity_group", "relevant_jobs", "inclusion_probability",
    "population_weight", "eligible_candidate_count", "candidate_regime",
    "sampling_strategy", "seed", "requested_candidate_size",
    "actual_candidate_count", "actual_candidate_fraction", "targets_injected",
    "model", "k", "effective_k", "truncated_at_k", *METRICS,
)
TIE_RTOL = 1e-10
TIE_ATOL = 1e-12
METRIC_BOUND_ATOL = 1e-12
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261020


def _expected_design_keys() -> frozenset[tuple[str, str, int, int, str, int]]:
    """Return the exact per-user evaluation grid, normalizing full size to -1."""
    keys: set[tuple[str, str, int, int, str, int]] = set()
    for model in MODELS:
        for k in KS:
            keys.add(("full_catalog", "full_catalog", 0, -1, model, k))
    for strategy in STRATEGIES:
        for seed in SEEDS:
            for size in SIZES:
                for model in MODELS:
                    for k in KS:
                        if size == 100 and k == 150:
                            continue
                        keys.add(("sampled", strategy, seed, size, model, k))
    return frozenset(keys)


EXPECTED_DESIGN_KEYS = _expected_design_keys()
EXPECTED_ROWS_PER_USER = len(EXPECTED_DESIGN_KEYS)
PRIMARY_EXPECTED_USERS = 1000 * len(WINDOWS)
PRIMARY_EXPECTED_ROWS = PRIMARY_EXPECTED_USERS * EXPECTED_ROWS_PER_USER
EXPECTED_PRIMARY_MATRIX_COLUMNS = (
    len(MODELS) * len(PRIMARY_OUTCOMES)
    + len(STRATEGIES) * len(SEEDS) * len(MODELS)
    * sum(
        1
        for size in SIZES
        for _, k in PRIMARY_OUTCOMES
        if not (size == 100 and k == 150)
    )
)
VALID_SAMPLED_OUTCOME_SIZE_CELLS = sum(
    1
    for size in SIZES
    for _, k in PRIMARY_OUTCOMES
    if not (size == 100 and k == 150)
)
EXPECTED_CONTRAST_BOOTSTRAP_ROWS = 2 * (
    len(PRIMARY_OUTCOMES) * len(CONTRASTS)
    + len(STRATEGIES) * (len(SEEDS) + 1)
    * VALID_SAMPLED_OUTCOME_SIZE_CELLS * len(CONTRASTS) * 2
)


class AuthorityError(RuntimeError):
    """Raised before scientific data are loaded when authority checks fail."""


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path


@dataclass
class ValidatedRun:
    spec: RunSpec
    config: dict
    report: dict
    integrity: dict
    manifest: dict
    partitions: list[Path]
    validation: dict
    users_per_window: int


def parse_run(value: str) -> RunSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    label, raw = value.split("=", 1)
    label = label.strip()
    if not label or any(char in label for char in "/\\"):
        raise argparse.ArgumentTypeError(f"Invalid run label: {label!r}")
    path = Path(raw.strip()).expanduser().resolve()
    return RunSpec(label=label, path=path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze authoritative CareerBuilder internal-temporal production runs."
    )
    parser.add_argument(
        "--run", action="append", required=True, type=parse_run,
        help="Authoritative run as LABEL=PATH; repeat for sensitivity protocols.",
    )
    parser.add_argument(
        "--primary-label", required=True,
        help="LABEL of the prespecified primary run among --run values.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bootstrap-reps", type=int, choices=[BOOTSTRAP_REPS],
        default=BOOTSTRAP_REPS,
        help="Fixed by the analysis protocol.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--max-rss-gib", type=float, default=12.0,
        help="Fail safely if resident memory exceeds this ceiling (default: 12 GiB).",
    )
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


_PEAK_RSS_BYTES = 0


def rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        if os.name != "nt":
            return 0
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current = ctypes.windll.kernel32.GetCurrentProcess
        get_current.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory.restype = wintypes.BOOL
        handle = get_current()
        ok = get_memory(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else 0


def check_memory(max_rss_bytes: int) -> int:
    global _PEAK_RSS_BYTES
    current = rss_bytes()
    _PEAK_RSS_BYTES = max(_PEAK_RSS_BYTES, current)
    if current and current > max_rss_bytes:
        raise MemoryError(
            f"Fail-closed RSS ceiling exceeded: {current / 2**30:.2f} GiB > "
            f"{max_rss_bytes / 2**30:.2f} GiB"
        )
    return current


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise AuthorityError(f"Required authority file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthorityError(f"Invalid JSON authority file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuthorityError(f"Authority file must contain an object: {path}")
    return value


def require_equal(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise AuthorityError(f"{field}: expected {expected!r}, found {actual!r}")


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_authoritative_run(spec: RunSpec) -> ValidatedRun:
    """Authenticate a completed run before reading any scientific metric rows."""
    run_dir = spec.path
    if not run_dir.is_dir():
        raise AuthorityError(f"Run directory does not exist: {run_dir}")
    if any("smoke" in part.lower() for part in run_dir.parts):
        raise AuthorityError(f"Smoke-labelled paths are categorically rejected: {run_dir}")

    config_path = run_dir / "RUN_CONFIG.json"
    report_path = run_dir / "RUN_REPORT.json"
    integrity_path = run_dir / "INTEGRITY_AUDIT.json"
    manifest_path = run_dir / "checkpoint_manifest.json"
    aggregate_path = run_dir / "aggregate_metrics.csv"
    invocation_path = run_dir / "LAST_INVOCATION.json"
    for required_path in (
        config_path, report_path, integrity_path, manifest_path,
        aggregate_path, invocation_path,
    ):
        if not required_path.is_file():
            raise AuthorityError(f"Required production artifact is missing: {required_path}")

    config = read_json(config_path)
    report = read_json(report_path)
    integrity = read_json(integrity_path)
    manifest = read_json(manifest_path)
    invocation = read_json(invocation_path)

    report_authority = report.get("authority", {})
    if not isinstance(report_authority, dict):
        raise AuthorityError("RUN_REPORT.authority must be an object")
    for field, expected in {
        "authoritative": True,
        "production": True,
        "smoke": False,
        "status": "completed",
    }.items():
        require_equal(report_authority.get(field), expected, f"RUN_REPORT.authority.{field}")
    study_role = report_authority.get("study_role")
    if study_role == "primary":
        expected_protocols = {"First7Last2"}
        users_per_window = 1000
    elif study_role == "temporal_sensitivity":
        expected_protocols = {"First6Last3", "First5Last4"}
        users_per_window = 250
    else:
        raise AuthorityError(f"Unrecognized authoritative study_role: {study_role!r}")
    configured_authority = config.get("authority", {})
    require_equal(configured_authority.get("authoritative"), True, "config authority")
    require_equal(configured_authority.get("production"), True, "config production")
    require_equal(configured_authority.get("smoke"), False, "config smoke")
    require_equal(configured_authority.get("study_role"), study_role, "config study_role")
    require_equal(config.get("task"), "CareerBuilder_InternalTemporal", "task")
    require_equal(
        config.get("official_competition_evaluation"), False,
        "official_competition_evaluation",
    )
    require_equal(config.get("provider_split_used"), False, "provider_split_used")
    require_equal(config.get("windows"), list(WINDOWS), "windows")
    require_equal(config.get("seeds"), list(SEEDS), "seeds")
    require_equal(config.get("strategies"), list(STRATEGIES), "strategies")
    require_equal(config.get("candidate_sizes"), list(SIZES), "candidate_sizes")
    require_equal(config.get("ks"), list(KS), "ks")
    require_equal(config.get("models"), list(MODELS), "models")
    require_equal(
        config.get("size100_k150_policy"), "row excluded entirely",
        "size100_k150_policy",
    )
    require_equal(config.get("smoke_users_per_activity"), None, "smoke selector")
    command = config.get("command_equivalent_arguments", {})
    require_equal(command.get("skip_semantic"), False, "skip_semantic")
    if config.get("protocol") not in expected_protocols:
        raise AuthorityError(
            f"Protocol {config.get('protocol')!r} is invalid for study_role {study_role!r}"
        )

    require_equal(report.get("run_config"), config, "report/config identity")
    require_equal(report.get("integrity"), integrity, "report/integrity identity")
    require_equal(
        report.get("checkpoint_manifest"), manifest,
        "report/checkpoint manifest identity",
    )
    config_hash = sha256(config_path)
    require_equal(
        report.get("provenance", {}).get("run_config_sha256"), config_hash,
        "reported RUN_CONFIG hash",
    )
    require_equal(
        invocation.get("run_config_sha256"), config_hash,
        "LAST_INVOCATION RUN_CONFIG hash",
    )
    require_equal(
        report.get("provenance", {}).get("evaluator_sha256"),
        config.get("evaluator_sha256"), "evaluator provenance hash",
    )
    require_equal(
        report.get("provenance", {}).get("preparation_manifest_sha256"),
        config.get("preparation_manifest_sha256"), "preparation provenance hash",
    )

    zero_fields = (
        "target_inclusion_failures", "nesting_failures", "size100_k150_rows",
        "invalid_cutoff_rows", "nested_candidate_failures",
    )
    for field in zero_fields:
        require_equal(int(integrity.get(field, -1)), 0, f"integrity.{field}")
    true_fields = (
        "catalog_unique", "targets_in_catalog", "targets_novel_vs_precut",
        "sample_users_in_population", "all_targets_injected_in_sampled_candidates",
    )
    for field in true_fields:
        require_equal(integrity.get(field), True, f"integrity.{field}")
    require_equal(integrity.get("provider_split_read"), False, "provider split read")
    if int(integrity.get("determinism_checked", 0)) != 1:
        raise AuthorityError("Candidate-sampling determinism was not checked")

    window_reports = report.get("windows", [])
    require_equal(
        [int(item.get("window", -1)) for item in window_reports],
        list(WINDOWS), "window-report coverage",
    )
    for item in window_reports:
        window = int(item["window"])
        require_equal(
            int(item.get("audit_sample_queries", -1)), users_per_window,
            f"W{window} audit sample size",
        )
        require_equal(
            int(item.get("selected_run_queries", -1)), users_per_window,
            f"W{window} selected run size",
        )

    entries = manifest.get("authoritative_partitions")
    if not isinstance(entries, list) or not entries:
        raise AuthorityError("Checkpoint manifest has no authoritative partitions")
    require_equal(int(manifest.get("checkpoint_count", -1)), len(entries), "checkpoint count")
    checkpoint_users = int(config.get("checkpoint_users", 0))
    if checkpoint_users <= 0:
        raise AuthorityError("Invalid checkpoint_users")
    expected_partitions = len(WINDOWS) * math.ceil(users_per_window / checkpoint_users)
    require_equal(len(entries), expected_partitions, "partition completeness")

    partitions: list[Path] = []
    intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    total_rows = total_bytes = total_users = 0
    for entry in entries:
        raw_path = Path(str(entry.get("path", "")))
        path = raw_path if raw_path.is_absolute() else run_dir / raw_path
        path = path.resolve()
        if not path_is_within(path, run_dir / "checkpoints"):
            raise AuthorityError(f"Checkpoint escapes the run directory: {path}")
        if not path.is_file():
            raise AuthorityError(f"Checkpoint is missing: {path}")
        require_equal(sha256(path), entry.get("sha256"), f"hash {path.name}")
        require_equal(path.stat().st_size, int(entry.get("bytes", -1)), f"bytes {path.name}")
        metadata_path = path.with_suffix(".json")
        metadata = read_json(metadata_path)
        require_equal(metadata.get("sha256"), entry.get("sha256"), f"metadata hash {path.name}")
        require_equal(int(metadata.get("rows", -1)), int(entry.get("rows", -2)), f"rows {path.name}")
        require_equal(int(metadata.get("users", -1)), int(entry.get("users", -2)), f"users {path.name}")
        require_equal(
            int(entry.get("rows", -1)),
            int(entry.get("users", -1)) * EXPECTED_ROWS_PER_USER,
            f"complete row design {path.name}",
        )
        require_equal(int(metadata.get("bytes", -1)), int(entry.get("bytes", -2)), f"metadata bytes {path.name}")
        offsets = metadata.get("user_offsets")
        if not (
            isinstance(offsets, list) and len(offsets) == 2
            and int(offsets[1]) > int(offsets[0])
        ):
            raise AuthorityError(f"Invalid user offsets: {metadata_path}")
        window = int(metadata.get("window", -1))
        if window not in WINDOWS:
            raise AuthorityError(f"Invalid checkpoint window {window}: {path}")
        intervals[window].append((int(offsets[0]), int(offsets[1])))
        total_rows += int(entry["rows"])
        total_bytes += int(entry["bytes"])
        total_users += int(entry["users"])
        partitions.append(path)

    for window in WINDOWS:
        ordered = sorted(intervals[window])
        expected_start = 0
        for start, stop in ordered:
            if start != expected_start:
                raise AuthorityError(
                    f"W{window} checkpoint gap/overlap: expected {expected_start}, got {start}"
                )
            expected_start = stop
        require_equal(expected_start, users_per_window, f"W{window} user coverage")
    require_equal(total_rows, int(manifest.get("total_rows", -1)), "manifest total rows")
    require_equal(
        total_rows,
        users_per_window * len(WINDOWS) * EXPECTED_ROWS_PER_USER,
        "frozen-design total rows",
    )
    require_equal(
        total_bytes, int(manifest.get("total_compressed_bytes", -1)),
        "manifest total bytes",
    )
    require_equal(total_users, users_per_window * len(WINDOWS), "partition users")
    require_equal(int(manifest.get("forbidden_size100_k150_rows", -1)), 0, "forbidden rows")
    require_equal(int(manifest.get("invalid_cutoff_rows", -1)), 0, "invalid cutoff rows")

    validation = {
        "status": "passed",
        "authority": report_authority,
        "study_role": study_role,
        "users_per_window": users_per_window,
        "protocol": config["protocol"],
        "partitions": len(partitions),
        "partition_rows": total_rows,
        "partition_bytes": total_bytes,
        "users": total_users,
        "config_sha256": config_hash,
        "report_sha256": sha256(report_path),
        "integrity_sha256": sha256(integrity_path),
        "manifest_sha256": sha256(manifest_path),
        "aggregate_sha256": sha256(aggregate_path),
    }
    return ValidatedRun(
        spec=spec, config=config, report=report, integrity=integrity,
        manifest=manifest, partitions=sorted(partitions), validation=validation,
        users_per_window=users_per_window,
    )


def normalize_bool(series: pd.Series, field: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    normalized = series.astype(str).str.lower().map(mapping)
    if normalized.isna().any():
        raise ValueError(f"Unrecognized Boolean values in {field}")
    return normalized.astype(bool)


def _check_exact_design_grid(frame: pd.DataFrame, path: Path) -> None:
    """Reject row-count-preserving substitutions outside the frozen grid."""
    design = frame[[
        "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k",
    ]].copy()
    design["candidate_regime"] = design["candidate_regime"].astype(str)
    design["sampling_strategy"] = design["sampling_strategy"].astype(str)
    design["seed"] = design["seed"].astype(int)
    design["requested_candidate_size"] = (
        design["requested_candidate_size"].fillna(-1).astype(int)
    )
    design["model"] = design["model"].astype(str)
    design["k"] = design["k"].astype(int)
    observed = frozenset(design.drop_duplicates().itertuples(index=False, name=None))
    if observed != EXPECTED_DESIGN_KEYS:
        missing = sorted(EXPECTED_DESIGN_KEYS.difference(observed))
        unexpected = sorted(observed.difference(EXPECTED_DESIGN_KEYS))
        raise ValueError(
            f"{path}: exact evaluation-grid mismatch; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def _partition_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "protocol", "window", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k",
    ]
    work = frame.copy()
    work["weighted_denominator"] = work["population_weight"]
    for metric in METRICS:
        work[f"weighted_{metric}"] = work[metric] * work["population_weight"]
    return work.groupby(keys, dropna=False, as_index=False).agg(
        users=("user_id", "nunique"),
        weight_sum=("weighted_denominator", "sum"),
        actual_candidate_count_min=("actual_candidate_count", "min"),
        actual_candidate_count_max=("actual_candidate_count", "max"),
        actual_candidate_fraction_sum=("actual_candidate_fraction", "sum"),
        **{f"sum_{metric}": (metric, "sum") for metric in METRICS},
        **{
            f"weighted_sum_{metric}": (f"weighted_{metric}", "sum")
            for metric in METRICS
        },
    )


def _combine_partition_aggregates(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    keys = [
        "protocol", "window", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k",
    ]
    partial = pd.concat(parts, ignore_index=True)
    combined = partial.groupby(keys, dropna=False, as_index=False).agg(
        users=("users", "sum"),
        weight_sum=("weight_sum", "sum"),
        actual_candidate_count_min=("actual_candidate_count_min", "min"),
        actual_candidate_count_max=("actual_candidate_count_max", "max"),
        actual_candidate_fraction_sum=("actual_candidate_fraction_sum", "sum"),
        **{f"sum_{metric}": (f"sum_{metric}", "sum") for metric in METRICS},
        **{
            f"weighted_sum_{metric}": (f"weighted_sum_{metric}", "sum")
            for metric in METRICS
        },
    )
    combined["actual_candidate_fraction_mean"] = (
        combined["actual_candidate_fraction_sum"] / combined["users"]
    )
    for metric in METRICS:
        combined[metric] = combined[f"sum_{metric}"] / combined["users"]
        combined[f"weighted_{metric}"] = (
            combined[f"weighted_sum_{metric}"] / combined["weight_sum"]
        )
    return combined


def _validate_and_clip_metric_bounds(
    frame: pd.DataFrame,
    path: Path,
) -> dict[str, dict[str, float | int]]:
    """Reject substantive bound violations and normalize floating round-off.

    Metric accumulation can yield a value such as ``1.0000000000000002`` for
    a mathematically bounded measure. Values beyond the frozen absolute
    tolerance remain fatal. Accepted round-off is clipped to the exact interval
    and counted for the analysis integrity record.
    """
    audit: dict[str, dict[str, float | int]] = {}
    for metric in METRICS:
        values = frame[metric].to_numpy(float)
        if (
            (~np.isfinite(values)).any()
            or (values < -METRIC_BOUND_ATOL).any()
            or (values > 1.0 + METRIC_BOUND_ATOL).any()
        ):
            raise ValueError(f"{path}: invalid {metric}")
        outside = (values < 0.0) | (values > 1.0)
        excess = np.maximum(np.maximum(-values, values - 1.0), 0.0)
        audit[metric] = {
            "clipped_values": int(outside.sum()),
            "max_absolute_boundary_excess": float(excess.max(initial=0.0)),
        }
        if outside.any():
            frame.loc[:, metric] = np.clip(values, 0.0, 1.0)
    return audit


def _check_partition(
    frame: pd.DataFrame,
    validated: ValidatedRun,
    path: Path,
) -> dict[str, dict[str, float | int]]:
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"{path}: missing required columns {missing_columns}")
    if frame.empty:
        raise ValueError(f"Empty authoritative checkpoint: {path}")
    if set(frame["protocol"].astype(str)) != {validated.config["protocol"]}:
        raise ValueError(f"{path}: protocol mismatch")
    if set(frame["window"].astype(int)).difference(WINDOWS):
        raise ValueError(f"{path}: invalid window")
    if set(frame["model"].astype(str)) != set(MODELS):
        raise ValueError(f"{path}: incomplete or unexpected model set")
    if set(frame["activity_group"].astype(str)).difference(ACTIVITY_GROUPS):
        raise ValueError(f"{path}: unexpected activity group")
    if set(frame["target_popularity_group"].astype(str)).difference(
        TARGET_POPULARITY_GROUPS
    ):
        raise ValueError(f"{path}: unexpected target-popularity group")

    null_exempt = {"requested_candidate_size"}
    null_counts = frame[
        [column for column in REQUIRED_COLUMNS if column not in null_exempt]
    ].isna().sum()
    if int(null_counts.sum()):
        raise ValueError(f"{path}: missing required values {null_counts[null_counts > 0].to_dict()}")

    frame["targets_injected"] = normalize_bool(frame["targets_injected"], "targets_injected")
    frame["truncated_at_k"] = normalize_bool(frame["truncated_at_k"], "truncated_at_k")
    full = frame["candidate_regime"].eq("full_catalog")
    sampled = frame["candidate_regime"].eq("sampled")
    if not (full | sampled).all():
        raise ValueError(f"{path}: unexpected candidate regime")
    if not (
        frame.loc[full, "sampling_strategy"].eq("full_catalog").all()
        and frame.loc[full, "seed"].eq(0).all()
        and frame.loc[full, "requested_candidate_size"].isna().all()
        and (~frame.loc[full, "targets_injected"]).all()
        and frame.loc[full, "actual_candidate_fraction"].eq(1.0).all()
    ):
        raise ValueError(f"{path}: invalid full-catalog row semantics")
    if not (
        frame.loc[sampled, "sampling_strategy"].isin(STRATEGIES).all()
        and frame.loc[sampled, "seed"].isin(SEEDS).all()
        and frame.loc[sampled, "requested_candidate_size"].isin(SIZES).all()
        and frame.loc[sampled, "targets_injected"].all()
    ):
        raise ValueError(f"{path}: invalid sampled row semantics")
    if (
        (frame["k"] > frame["actual_candidate_count"]).any()
        or (frame["effective_k"] != frame["k"]).any()
        or frame["truncated_at_k"].any()
    ):
        raise ValueError(f"{path}: unsupported or truncated cutoff row")
    if (
        sampled
        & frame["requested_candidate_size"].eq(100)
        & frame["k"].eq(150)
    ).any():
        raise ValueError(f"{path}: forbidden size-100/k-150 row")
    if not np.allclose(
        frame["actual_candidate_fraction"].to_numpy(float),
        frame["actual_candidate_count"].to_numpy(float)
        / frame["eligible_candidate_count"].to_numpy(float),
        rtol=0.0, atol=2e-12,
    ):
        raise ValueError(f"{path}: realized candidate fraction mismatch")
    if (
        (frame["eligible_candidate_count"] <= 0).any()
        or (frame["actual_candidate_count"] <= 0).any()
        or (frame["actual_candidate_count"] > frame["eligible_candidate_count"]).any()
        or (frame["relevant_jobs"] <= 0).any()
        or (frame["inclusion_probability"] <= 0).any()
        or (frame["inclusion_probability"] > 1).any()
        or (frame["population_weight"] < 1).any()
    ):
        raise ValueError(f"{path}: invalid counts or design weights")
    if not np.allclose(
        frame["population_weight"] * frame["inclusion_probability"],
        1.0, rtol=1e-10, atol=1e-12,
    ):
        raise ValueError(f"{path}: population weight is not inverse inclusion probability")
    metric_bound_audit = _validate_and_clip_metric_bounds(frame, path)

    duplicate_keys = [
        "window", "user_id", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k",
    ]
    if frame.duplicated(duplicate_keys).any():
        raise ValueError(f"{path}: duplicate per-user metric keys")
    _check_exact_design_grid(frame, path)
    row_counts = frame.groupby(["window", "user_id"]).size()
    if not row_counts.eq(EXPECTED_ROWS_PER_USER).all():
        raise ValueError(
            f"{path}: expected {EXPECTED_ROWS_PER_USER} rows/user; "
            f"observed range {row_counts.min()}--{row_counts.max()}"
        )

    metadata_columns = [
        "precut_application_count", "activity_group",
        "mean_target_precut_popularity", "target_popularity_group",
        "relevant_jobs", "inclusion_probability", "population_weight",
        "eligible_candidate_count",
    ]
    for column in metadata_columns:
        if frame.groupby(["window", "user_id"])[column].nunique(dropna=False).max() != 1:
            raise ValueError(f"{path}: nonconstant user metadata {column}")
    return metric_bound_audit


def _primary_wide_partition(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce one checkpoint immediately to compact numeric inferential columns."""
    column_keys = [
        "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size_key", "model",
    ]
    work = frame.copy()
    work["requested_candidate_size_key"] = (
        work["requested_candidate_size"].fillna(-1).astype(int)
    )
    pieces: list[pd.DataFrame] = []
    for metric, k in PRIMARY_OUTCOMES:
        cell = work[work["k"].eq(k)]
        wide = cell.pivot(
            index=["window", "user_id"], columns=column_keys, values=metric
        )
        wide.columns = pd.MultiIndex.from_tuples(
            [(*tuple(column), metric, int(k)) for column in wide.columns],
            names=[
                "candidate_regime", "sampling_strategy", "seed",
                "requested_candidate_size", "model", "metric", "k",
            ],
        )
        pieces.append(wide.astype(np.float64))
    result = pd.concat(pieces, axis=1)
    if result.columns.duplicated().any() or result.isna().any().any():
        raise ValueError("Compact primary matrix has duplicate or missing cells")
    return result.sort_index(axis=1)


def load_and_verify_metrics(
    validated: ValidatedRun,
    max_rss_bytes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Stream authenticated partitions, verify them, and retain analysis rows."""
    aggregate_parts: list[pd.DataFrame] = []
    primary_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    user_parts: list[pd.DataFrame] = []
    total_rows = 0
    metric_boundary_adjustments = {metric: 0 for metric in METRICS}
    metric_max_boundary_excess = {metric: 0.0 for metric in METRICS}

    user_columns = [
        "window", "user_id", "precut_application_count", "activity_group",
        "mean_target_precut_popularity", "target_popularity_group",
        "relevant_jobs", "inclusion_probability", "population_weight",
        "eligible_candidate_count",
    ]
    candidate_columns = [
        "window", "user_id", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "eligible_candidate_count",
        "actual_candidate_count", "actual_candidate_fraction",
    ]

    for path in validated.partitions:
        frame = pd.read_csv(path, low_memory=False)
        check_memory(max_rss_bytes)
        partition_bound_audit = _check_partition(frame, validated, path)
        for metric, values in partition_bound_audit.items():
            metric_boundary_adjustments[metric] += int(values["clipped_values"])
            metric_max_boundary_excess[metric] = max(
                metric_max_boundary_excess[metric],
                float(values["max_absolute_boundary_excess"]),
            )
        total_rows += len(frame)
        aggregate_parts.append(_partition_aggregate(frame))
        primary_parts.append(_primary_wide_partition(frame))
        candidate_parts.append(
            frame[
                frame["model"].eq(MODELS[0]) & frame["k"].eq(5)
            ][candidate_columns].drop_duplicates().copy()
        )
        user_parts.append(frame[user_columns].drop_duplicates().copy())
        del frame
        check_memory(max_rss_bytes)

    require_equal(total_rows, int(validated.manifest["total_rows"]), "loaded rows")
    aggregate = _combine_partition_aggregates(aggregate_parts)
    expected_aggregate_rows = len(WINDOWS) * EXPECTED_ROWS_PER_USER
    if len(aggregate) != expected_aggregate_rows:
        raise ValueError(
            f"Expected {expected_aggregate_rows:,} aggregate cells; found {len(aggregate):,}"
        )

    published = pd.read_csv(validated.spec.path / "aggregate_metrics.csv")
    keys = [
        "protocol", "window", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k",
    ]
    compare_columns = [
        "users", "weight_sum", "actual_candidate_count_min",
        "actual_candidate_count_max", "actual_candidate_fraction_mean",
        *METRICS, *[f"weighted_{metric}" for metric in METRICS],
    ]
    comparison = published.merge(
        aggregate[keys + compare_columns], on=keys, how="outer",
        suffixes=("_published", "_recomputed"), indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError("Published and recomputed aggregate keys differ")
    aggregate_max_abs_difference = 0.0
    for column in compare_columns:
        left = comparison[f"{column}_published"].to_numpy(float)
        right = comparison[f"{column}_recomputed"].to_numpy(float)
        difference = float(np.max(np.abs(left - right)))
        aggregate_max_abs_difference = max(aggregate_max_abs_difference, difference)
        tolerance = 0.0 if column in {
            "users", "actual_candidate_count_min", "actual_candidate_count_max"
        } else 5e-12
        if difference > tolerance:
            raise ValueError(
                f"Aggregate reproduction failed for {column}: max abs diff {difference}"
            )

    primary = pd.concat(primary_parts, axis=0).sort_index()
    candidate = pd.concat(candidate_parts, ignore_index=True)
    users = pd.concat(user_parts, ignore_index=True).drop_duplicates()
    if users.duplicated(["window", "user_id"]).any():
        raise ValueError("User metadata conflicts across checkpoints")
    expected_users = validated.users_per_window * len(WINDOWS)
    if len(users) != expected_users:
        raise ValueError(f"Expected {expected_users:,} users; found {len(users):,}")
    if users.groupby("window").size().to_dict() != {
        window: validated.users_per_window for window in WINDOWS
    }:
        raise ValueError("Unequal or incomplete audit cohort by window")
    if set(users["activity_group"].astype(str)).difference(ACTIVITY_GROUPS):
        raise ValueError("Unexpected activity group after concatenation")
    if set(users["target_popularity_group"].astype(str)).difference(
        TARGET_POPULARITY_GROUPS
    ):
        raise ValueError("Unexpected target-popularity group after concatenation")

    if primary.index.duplicated().any():
        raise ValueError("Compact primary matrix has duplicate users")
    if primary.shape != (expected_users, EXPECTED_PRIMARY_MATRIX_COLUMNS):
        raise ValueError(
            f"Unexpected compact primary shape {primary.shape}; "
            f"expected ({expected_users}, {EXPECTED_PRIMARY_MATRIX_COLUMNS})"
        )
    if primary.isna().any().any():
        raise ValueError("Compact primary matrix contains missing values")
    check_memory(max_rss_bytes)
    audit = {
        "partition_rows_loaded": int(total_rows),
        "aggregate_cells": int(len(aggregate)),
        "aggregate_max_abs_difference": aggregate_max_abs_difference,
        "metric_boundary_normalization": {
            "absolute_tolerance": METRIC_BOUND_ATOL,
            "clipped_values_by_metric": metric_boundary_adjustments,
            "max_absolute_boundary_excess_by_metric": metric_max_boundary_excess,
        },
        "compact_primary_shape": [int(primary.shape[0]), int(primary.shape[1])],
        "compact_primary_bytes": int(primary.memory_usage(index=True, deep=True).sum()),
        "candidate_units_retained": int(len(candidate)),
        "unique_users": int(len(users)),
        "peak_rss_bytes_during_load": int(_PEAK_RSS_BYTES),
    }
    return aggregate, primary, candidate, users, audit


def _condition_columns(include_window: bool = False) -> list[str]:
    columns = [
        "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k",
    ]
    return (["window"] if include_window else []) + columns


def per_window_estimates(aggregate: pd.DataFrame, run: ValidatedRun) -> pd.DataFrame:
    records: list[dict] = []
    for row in aggregate.itertuples(index=False):
        common = {
            "run_label": run.spec.label,
            "protocol": run.config["protocol"],
            "window": int(row.window),
            "candidate_regime": row.candidate_regime,
            "sampling_strategy": row.sampling_strategy,
            "seed": int(row.seed),
            "requested_candidate_size": row.requested_candidate_size,
            "model": row.model,
            "k": int(row.k),
            "users": int(row.users),
            "population_weight_sum": float(row.weight_sum),
            "actual_candidate_fraction_mean": float(
                row.actual_candidate_fraction_mean
            ),
        }
        records.append({
            **common, "estimand": "audit_cohort",
            **{metric: float(getattr(row, metric)) for metric in METRICS},
        })
        records.append({
            **common, "estimand": "design_weighted_population",
            **{
                metric: float(getattr(row, f"weighted_{metric}"))
                for metric in METRICS
            },
        })
    return pd.DataFrame(records)


def pooled_estimates(aggregate: pd.DataFrame, run: ValidatedRun) -> pd.DataFrame:
    records: list[dict] = []
    for keys, cell in aggregate.groupby(_condition_columns(), dropna=False, sort=False):
        condition = dict(zip(_condition_columns(), keys))
        common = {
            "run_label": run.spec.label,
            "protocol": run.config["protocol"],
            **condition,
            "windows": int(cell["window"].nunique()),
            "audit_users": int(cell["users"].sum()),
            "population_weight_sum": float(cell["weight_sum"].sum()),
            "actual_candidate_fraction_mean": float(np.average(
                cell["actual_candidate_fraction_mean"], weights=cell["users"]
            )),
        }
        audit = {
            metric: float(
                cell[f"sum_{metric}"].sum() / cell["users"].sum()
            )
            for metric in METRICS
        }
        weighted = {
            metric: float(
                cell[f"weighted_sum_{metric}"].sum() / cell["weight_sum"].sum()
            )
            for metric in METRICS
        }
        records.append({**common, "estimand": "audit_cohort", **audit})
        records.append({
            **common, "estimand": "design_weighted_population", **weighted
        })
    pooled = pd.DataFrame(records)
    pooled["requested_candidate_size"] = pooled[
        "requested_candidate_size"
    ].astype(float)
    return pooled


def seed_summaries(pooled: pd.DataFrame) -> pd.DataFrame:
    sampled = pooled[pooled["candidate_regime"].eq("sampled")].copy()
    keys = [
        "run_label", "protocol", "estimand", "sampling_strategy",
        "requested_candidate_size", "model", "k",
    ]
    records: list[dict] = []
    for group_values, cell in sampled.groupby(keys, dropna=False, sort=False):
        if sorted(cell["seed"].astype(int).unique()) != list(SEEDS):
            raise ValueError(f"Incomplete seeds at {dict(zip(keys, group_values))}")
        common = dict(zip(keys, group_values))
        common.update({
            "seeds": int(cell["seed"].nunique()),
            "audit_users_per_seed": int(cell["audit_users"].iloc[0]),
            "actual_candidate_fraction_mean": float(
                cell["actual_candidate_fraction_mean"].mean()
            ),
        })
        for metric in METRICS:
            values = cell[metric].to_numpy(float)
            records.append({
                **common,
                "metric": metric,
                "seed_mean": float(values.mean()),
                "seed_sd": float(values.std(ddof=1)),
                "seed_min": float(values.min()),
                "seed_max": float(values.max()),
            })
    return pd.DataFrame(records)


def sampling_bias(pooled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pooled[pooled["candidate_regime"].eq("full_catalog")].copy()
    sampled = pooled[pooled["candidate_regime"].eq("sampled")].copy()
    join = ["run_label", "protocol", "estimand", "model", "k"]
    full_values = full[join + list(METRICS)].rename(
        columns={metric: f"full_{metric}" for metric in METRICS}
    )
    merged = sampled.merge(full_values, on=join, how="left", validate="many_to_one")
    if merged[[f"full_{metric}" for metric in METRICS]].isna().any().any():
        raise ValueError("Sampled/full bias join has missing full-catalog references")
    records: list[dict] = []
    id_columns = [
        "run_label", "protocol", "estimand", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k", "audit_users",
        "actual_candidate_fraction_mean",
    ]
    for row in merged.itertuples(index=False):
        common = {column: getattr(row, column) for column in id_columns}
        for metric in METRICS:
            sampled_value = float(getattr(row, metric))
            full_value = float(getattr(row, f"full_{metric}"))
            signed = sampled_value - full_value
            records.append({
                **common, "metric": metric,
                "sampled_value": sampled_value,
                "full_value": full_value,
                "signed_bias_sampled_minus_full": signed,
                "absolute_bias": abs(signed),
            })
    detail = pd.DataFrame(records)
    summary_keys = [
        "run_label", "protocol", "estimand", "sampling_strategy",
        "requested_candidate_size", "model", "k", "metric",
    ]
    summary = detail.groupby(summary_keys, dropna=False, as_index=False).agg(
        seeds=("seed", "nunique"),
        sampled_seed_mean=("sampled_value", "mean"),
        full_value=("full_value", "first"),
        mean_signed_bias=("signed_bias_sampled_minus_full", "mean"),
        sd_signed_bias=("signed_bias_sampled_minus_full", "std"),
        mean_absolute_bias=("absolute_bias", "mean"),
        max_absolute_bias=("absolute_bias", "max"),
        actual_candidate_fraction_mean=("actual_candidate_fraction_mean", "mean"),
    )
    summary["absolute_mean_bias"] = summary["mean_signed_bias"].abs()
    return detail, summary


def candidate_scale_summary(candidate: pd.DataFrame, run: ValidatedRun) -> pd.DataFrame:
    duplicate_keys = [
        "window", "user_id", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size",
    ]
    if candidate.duplicated(duplicate_keys).any():
        raise ValueError("Candidate-scale units are duplicated")
    records: list[dict] = []
    grouping = [
        "candidate_regime", "sampling_strategy", "seed", "requested_candidate_size"
    ]
    for keys, cell in candidate.groupby(grouping, dropna=False, sort=False):
        regime, strategy, seed, nominal = keys
        if len(cell) != run.users_per_window * len(WINDOWS):
            raise ValueError(f"Incomplete candidate-scale cell: {keys}")
        records.append({
            "run_label": run.spec.label,
            "protocol": run.config["protocol"],
            "candidate_regime": regime,
            "sampling_strategy": strategy,
            "seed": int(seed),
            "requested_candidate_size": nominal,
            "users": int(cell["user_id"].count()),
            "eligible_candidate_count_mean": float(
                cell["eligible_candidate_count"].mean()
            ),
            "eligible_candidate_count_median": float(
                cell["eligible_candidate_count"].median()
            ),
            "eligible_candidate_count_min": int(
                cell["eligible_candidate_count"].min()
            ),
            "eligible_candidate_count_max": int(
                cell["eligible_candidate_count"].max()
            ),
            "actual_candidate_count_mean": float(cell["actual_candidate_count"].mean()),
            "actual_candidate_count_median": float(
                cell["actual_candidate_count"].median()
            ),
            "actual_candidate_count_min": int(cell["actual_candidate_count"].min()),
            "actual_candidate_count_max": int(cell["actual_candidate_count"].max()),
            "actual_candidate_fraction_mean": float(
                cell["actual_candidate_fraction"].mean()
            ),
            "actual_candidate_fraction_median": float(
                cell["actual_candidate_fraction"].median()
            ),
            "users_actual_exceeds_nominal": (
                0 if pd.isna(nominal) else int(
                    (cell["actual_candidate_count"] > float(nominal)).sum()
                )
            ),
        })
    detail = pd.DataFrame(records)
    sampled = detail[detail["candidate_regime"].eq("sampled")]
    summary = sampled.groupby(
        [
            "run_label", "protocol", "sampling_strategy",
            "requested_candidate_size",
        ], as_index=False, dropna=False,
    ).agg(
        seeds=("seed", "nunique"),
        users_per_seed=("users", "first"),
        eligible_candidate_count_mean=("eligible_candidate_count_mean", "mean"),
        eligible_candidate_count_median=("eligible_candidate_count_median", "mean"),
        eligible_candidate_count_min=("eligible_candidate_count_min", "min"),
        eligible_candidate_count_max=("eligible_candidate_count_max", "max"),
        actual_candidate_count_mean=("actual_candidate_count_mean", "mean"),
        actual_candidate_count_median=("actual_candidate_count_median", "mean"),
        actual_candidate_count_min=("actual_candidate_count_min", "min"),
        actual_candidate_count_max=("actual_candidate_count_max", "max"),
        actual_candidate_fraction_mean=("actual_candidate_fraction_mean", "mean"),
        actual_candidate_fraction_median=("actual_candidate_fraction_median", "mean"),
        users_actual_exceeds_nominal_per_seed=("users_actual_exceeds_nominal", "mean"),
    )
    full = detail[detail["candidate_regime"].eq("full_catalog")].copy()
    if len(full) != 1:
        raise ValueError("Expected one pooled full-catalog scale cell")
    full_summary = pd.DataFrame([{
        "run_label": run.spec.label,
        "protocol": run.config["protocol"],
        "sampling_strategy": "full_catalog",
        "requested_candidate_size": np.nan,
        "seeds": 1,
        "users_per_seed": int(full.iloc[0]["users"]),
        "eligible_candidate_count_mean": full.iloc[0]["eligible_candidate_count_mean"],
        "eligible_candidate_count_median": full.iloc[0]["eligible_candidate_count_median"],
        "eligible_candidate_count_min": full.iloc[0]["eligible_candidate_count_min"],
        "eligible_candidate_count_max": full.iloc[0]["eligible_candidate_count_max"],
        "actual_candidate_count_mean": full.iloc[0]["actual_candidate_count_mean"],
        "actual_candidate_count_median": full.iloc[0]["actual_candidate_count_median"],
        "actual_candidate_count_min": full.iloc[0]["actual_candidate_count_min"],
        "actual_candidate_count_max": full.iloc[0]["actual_candidate_count_max"],
        "actual_candidate_fraction_mean": 1.0,
        "actual_candidate_fraction_median": 1.0,
        "users_actual_exceeds_nominal_per_seed": 0.0,
    }])
    return pd.concat([full_summary, summary], ignore_index=True)


def _is_tie(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=TIE_RTOL, atol=TIE_ATOL))


def _tie_group_codes(values: Mapping[str, float]) -> np.ndarray:
    """Return tolerance-defined ordinal groups in the fixed model order."""
    ordered = sorted(values, key=lambda model: (-values[model], model))
    code_by_model: dict[str, int] = {}
    group = 0
    anchor = values[ordered[0]]
    code_by_model[ordered[0]] = group
    for model in ordered[1:]:
        if not _is_tie(values[model], anchor):
            group += 1
            anchor = values[model]
        code_by_model[model] = group
    return np.asarray([code_by_model[model] for model in MODELS], dtype=float)


def _winner_set(values: Mapping[str, float]) -> set[str]:
    maximum = max(values.values())
    return {
        model for model, value in values.items()
        if _is_tie(value, maximum)
    }


def _pairwise_state(left: float, right: float) -> int:
    if _is_tie(left, right):
        return 0
    return 1 if left > right else -1


def _fidelity_record(
    full_values: Mapping[str, float], sampled_values: Mapping[str, float]
) -> dict[str, object]:
    full_codes = _tie_group_codes(full_values)
    sampled_codes = _tie_group_codes(sampled_values)
    tau_result = kendalltau(
        full_codes, sampled_codes, variant="b", method="auto",
        nan_policy="propagate",
    )
    full_winners = _winner_set(full_values)
    sampled_winners = _winner_set(sampled_values)
    pair_agreements = []
    for left, right in itertools.combinations(MODELS, 2):
        pair_agreements.append(
            _pairwise_state(full_values[left], full_values[right])
            == _pairwise_state(sampled_values[left], sampled_values[right])
        )
    full_maximum = max(full_values.values())
    sampled_winner_full_scores = [full_values[model] for model in sampled_winners]
    union = full_winners | sampled_winners
    intersection = full_winners & sampled_winners
    return {
        "kendall_tau_b": float(tau_result.statistic),
        "kendall_tau_b_defined": bool(np.isfinite(tau_result.statistic)),
        "pairwise_order_agreement": float(np.mean(pair_agreements)),
        "pairwise_pairs": len(pair_agreements),
        "full_winner_set": "|".join(sorted(full_winners)),
        "sampled_winner_set": "|".join(sorted(sampled_winners)),
        "winner_set_exact_match": full_winners == sampled_winners,
        "winner_set_overlap": bool(intersection),
        "winner_set_jaccard": len(intersection) / len(union),
        "sampled_winner_regret_best": max(
            0.0, full_maximum - max(sampled_winner_full_scores)
        ),
        "sampled_winner_regret_worst": max(
            0.0, full_maximum - min(sampled_winner_full_scores)
        ),
    }


def rank_fidelity(
    pooled: pd.DataFrame,
    per_window: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []

    def analyze(source: pd.DataFrame, scope: str, estimands: Iterable[str]) -> None:
        for estimand in estimands:
            scoped = source[source["estimand"].eq(estimand)]
            windows = [None] if scope == "pooled" else list(WINDOWS)
            for window in windows:
                local = scoped if window is None else scoped[scoped["window"].eq(window)]
                full = local[local["candidate_regime"].eq("full_catalog")]
                sampled = local[local["candidate_regime"].eq("sampled")]
                for metric, k in PRIMARY_OUTCOMES:
                    full_cell = full[full["k"].eq(k)]
                    if set(full_cell["model"].astype(str)) != set(MODELS):
                        raise ValueError(
                            f"Incomplete full rank vector at {scope}/{estimand}/W{window}/{metric}@{k}"
                        )
                    full_values = dict(zip(full_cell["model"].astype(str), full_cell[metric]))
                    grouping = ["sampling_strategy", "seed", "requested_candidate_size"]
                    for keys, cell in sampled[sampled["k"].eq(k)].groupby(
                        grouping, dropna=False, sort=False
                    ):
                        if set(cell["model"].astype(str)) != set(MODELS):
                            raise ValueError(f"Incomplete sampled rank vector: {keys}")
                        strategy, seed, size = keys
                        sampled_values = dict(zip(cell["model"].astype(str), cell[metric]))
                        records.append({
                            "run_label": str(local["run_label"].iloc[0]),
                            "protocol": str(local["protocol"].iloc[0]),
                            "rank_scope": scope,
                            "window": np.nan if window is None else int(window),
                            "estimand": estimand,
                            "sampling_strategy": strategy,
                            "seed": int(seed),
                            "requested_candidate_size": float(size),
                            "metric": metric,
                            "k": int(k),
                            **_fidelity_record(full_values, sampled_values),
                        })

    analyze(pooled, "pooled", ("audit_cohort", "design_weighted_population"))
    # Population weights are constant within a window; a second within-window
    # estimand would duplicate every row exactly.
    analyze(per_window, "window", ("audit_cohort",))
    detail = pd.DataFrame(records)
    summary_keys = [
        "run_label", "protocol", "rank_scope", "estimand",
        "sampling_strategy", "requested_candidate_size", "metric", "k",
    ]
    summary = detail.groupby(summary_keys, dropna=False, as_index=False).agg(
        units=("seed", "size"),
        seeds=("seed", "nunique"),
        windows=("window", "nunique"),
        kendall_tau_b_mean=("kendall_tau_b", "mean"),
        kendall_tau_b_sd=("kendall_tau_b", "std"),
        kendall_tau_b_defined_fraction=("kendall_tau_b_defined", "mean"),
        pairwise_order_agreement_mean=("pairwise_order_agreement", "mean"),
        pairwise_order_agreement_sd=("pairwise_order_agreement", "std"),
        winner_set_exact_match_rate=("winner_set_exact_match", "mean"),
        winner_set_overlap_rate=("winner_set_overlap", "mean"),
        winner_set_jaccard_mean=("winner_set_jaccard", "mean"),
        sampled_winner_regret_best_mean=("sampled_winner_regret_best", "mean"),
        sampled_winner_regret_best_max=("sampled_winner_regret_best", "max"),
        sampled_winner_regret_worst_mean=("sampled_winner_regret_worst", "mean"),
        sampled_winner_regret_worst_max=("sampled_winner_regret_worst", "max"),
    )
    return detail, summary


def _metric_vector(
    primary: pd.DataFrame,
    regime: str,
    strategy: str,
    seed: int,
    size: int,
    model: str,
    metric: str,
    k: int,
) -> np.ndarray:
    key = (regime, strategy, int(seed), int(size), model, metric, int(k))
    if key not in primary.columns:
        raise KeyError(f"Missing compact per-query metric column: {key}")
    return primary[key].to_numpy(np.float64, copy=False)


def _contrast_vector(
    primary: pd.DataFrame,
    regime: str,
    strategy: str,
    seed: int,
    size: int,
    left: str,
    right: str,
    metric: str,
    k: int,
) -> np.ndarray:
    return (
        _metric_vector(primary, regime, strategy, seed, size, left, metric, k)
        - _metric_vector(primary, regime, strategy, seed, size, right, metric, k)
    )


def _aligned_users(primary: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    metadata = users.set_index(["window", "user_id"]).sort_index()
    if not metadata.index.equals(primary.index):
        raise ValueError("Compact per-query matrix and user metadata are not aligned")
    for window, cell in metadata.groupby(level="window"):
        if cell["population_weight"].nunique() != 1:
            raise ValueError(f"W{window}: population weight must be constant by design")
    return metadata


def _stratified_bootstrap_batch(
    values: np.ndarray,
    metadata: pd.DataFrame,
    reps: int,
    seed: int,
    mask: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Paired query bootstrap, independently resampled within each window.

    Rows are the pairing unit.  A window contributes the same number of draws as
    observed in that window (or observed in the requested subgroup).  The same
    multinomial multiplicities are applied to every model/contrast column.
    """
    if values.ndim == 1:
        values = values[:, None]
    if mask is None:
        mask = np.ones(len(metadata), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(metadata) or not mask.any():
        raise ValueError("Bootstrap mask is empty or misaligned")
    if not np.isfinite(values[mask]).all():
        raise ValueError("Bootstrap values contain non-finite entries")

    rng = np.random.default_rng(seed)
    audit_numerator = np.zeros((reps, values.shape[1]), dtype=np.float64)
    weighted_numerator = np.zeros_like(audit_numerator)
    audit_denominator = 0.0
    weighted_denominator = 0.0
    window_values = metadata.index.get_level_values("window").to_numpy(int)
    weights = metadata["population_weight"].to_numpy(float)
    represented = 0
    for window in WINDOWS:
        indices = np.flatnonzero(mask & (window_values == window))
        if not len(indices):
            continue
        represented += 1
        window_weights = weights[indices]
        if not np.allclose(window_weights, window_weights[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"W{window}: nonconstant design weights in bootstrap stratum")
        # Multinomial counts are equivalent to sampling users with replacement.
        counts = rng.multinomial(
            len(indices), np.repeat(1.0 / len(indices), len(indices)), size=reps
        ).astype(np.float32)
        sampled_sums = counts @ values[indices]
        audit_numerator += sampled_sums
        weighted_numerator += sampled_sums * window_weights[0]
        audit_denominator += len(indices)
        weighted_denominator += len(indices) * window_weights[0]
    if not represented:
        raise ValueError("No bootstrap strata represented")
    return {
        "audit_cohort": (
            values[mask].mean(axis=0), audit_numerator / audit_denominator
        ),
        "design_weighted_population": (
            np.average(values[mask], axis=0, weights=weights[mask]),
            weighted_numerator / weighted_denominator,
        ),
    }


def _interval_record(point: float, replicates: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(replicates, [0.025, 0.975])
    return {
        "estimate": float(point),
        "bootstrap_se": float(replicates.std(ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
        "probability_gt_zero": float((np.sum(replicates > 0) + 1) / (len(replicates) + 1)),
        "probability_lt_zero": float((np.sum(replicates < 0) + 1) / (len(replicates) + 1)),
    }


def bootstrap_prespecified_contrasts(
    primary: pd.DataFrame,
    users: pd.DataFrame,
    run: ValidatedRun,
    reps: int,
    seed: int,
    max_rss_bytes: int,
) -> pd.DataFrame:
    metadata = _aligned_users(primary, users)
    descriptors: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    full_vectors: dict[tuple[str, str, int], np.ndarray] = {}

    for metric, k in PRIMARY_OUTCOMES:
        for left, right, contrast_name in CONTRASTS:
            vector = _contrast_vector(
                primary, "full_catalog", "full_catalog", 0, -1,
                left, right, metric, k,
            )
            full_vectors[(contrast_name, metric, k)] = vector
            descriptors.append({
                "effect_type": "contrast",
                "candidate_regime": "full_catalog",
                "sampling_strategy": "full_catalog",
                "seed": 0,
                "seed_aggregation": "not_applicable",
                "requested_candidate_size": np.nan,
                "contrast": contrast_name,
                "left_model": left,
                "right_model": right,
                "metric": metric,
                "k": k,
            })
            vectors.append(vector)

    for strategy in STRATEGIES:
        for sampling_seed in SEEDS:
            for size in SIZES:
                for metric, k in PRIMARY_OUTCOMES:
                    if size == 100 and k == 150:
                        continue
                    for left, right, contrast_name in CONTRASTS:
                        sampled_vector = _contrast_vector(
                            primary, "sampled", strategy, sampling_seed, size,
                            left, right, metric, k,
                        )
                        common = {
                            "candidate_regime": "sampled",
                            "sampling_strategy": strategy,
                            "seed": sampling_seed,
                            "seed_aggregation": "individual",
                            "requested_candidate_size": size,
                            "contrast": contrast_name,
                            "left_model": left,
                            "right_model": right,
                            "metric": metric,
                            "k": k,
                        }
                        descriptors.append({**common, "effect_type": "contrast"})
                        vectors.append(sampled_vector)
                        descriptors.append({
                            **common,
                            "effect_type": "effect_distortion_sampled_minus_full",
                        })
                        vectors.append(
                            sampled_vector - full_vectors[(contrast_name, metric, k)]
                        )

    # The five-seed mean is the prespecified summary of sampling variability.
    for strategy in STRATEGIES:
        for size in SIZES:
            for metric, k in PRIMARY_OUTCOMES:
                if size == 100 and k == 150:
                    continue
                for left, right, contrast_name in CONTRASTS:
                    seed_vectors = [
                        _contrast_vector(
                            primary, "sampled", strategy, sampling_seed, size,
                            left, right, metric, k,
                        )
                        for sampling_seed in SEEDS
                    ]
                    sampled_mean = np.mean(seed_vectors, axis=0)
                    common = {
                        "candidate_regime": "sampled",
                        "sampling_strategy": strategy,
                        "seed": 0,
                        "seed_aggregation": "mean_over_5_seeds",
                        "requested_candidate_size": size,
                        "contrast": contrast_name,
                        "left_model": left,
                        "right_model": right,
                        "metric": metric,
                        "k": k,
                    }
                    descriptors.append({**common, "effect_type": "contrast"})
                    vectors.append(sampled_mean)
                    descriptors.append({
                        **common,
                        "effect_type": "effect_distortion_sampled_minus_full",
                    })
                    vectors.append(
                        sampled_mean - full_vectors[(contrast_name, metric, k)]
                    )

    records: list[dict] = []
    # Column batching bounds both bootstrap output and BLAS temporary memory.
    for batch_start in range(0, len(vectors), 64):
        batch_descriptors = descriptors[batch_start:batch_start + 64]
        matrix = np.column_stack(vectors[batch_start:batch_start + 64])
        results = _stratified_bootstrap_batch(
            matrix, metadata, reps, seed
        )
        for estimand, (points, bootstrap_values) in results.items():
            for column, descriptor in enumerate(batch_descriptors):
                records.append({
                    "run_label": run.spec.label,
                    "protocol": run.config["protocol"],
                    "estimand": estimand,
                    **descriptor,
                    "n_users": int(len(metadata)),
                    "n_windows": len(WINDOWS),
                    "bootstrap_replicates": reps,
                    "bootstrap_seed": seed,
                    "bootstrap_method": "paired users resampled within each window",
                    **_interval_record(points[column], bootstrap_values[:, column]),
                })
        check_memory(max_rss_bytes)
    output = pd.DataFrame(records)
    if len(output) != EXPECTED_CONTRAST_BOOTSTRAP_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CONTRAST_BOOTSTRAP_ROWS:,} prespecified "
            f"contrast-bootstrap rows; found {len(output):,}"
        )
    if output["contrast"].nunique() != len(CONTRASTS):
        raise ValueError("Not all five prespecified contrasts were bootstrapped")
    contrast_keys = [
        "estimand", "effect_type", "candidate_regime",
        "sampling_strategy", "seed", "seed_aggregation",
        "requested_candidate_size", "contrast", "metric", "k",
    ]
    if output.duplicated(contrast_keys).any():
        raise ValueError("Duplicate prespecified contrast-bootstrap keys")
    return output


def bootstrap_subgroups(
    primary: pd.DataFrame,
    users: pd.DataFrame,
    run: ValidatedRun,
    reps: int,
    seed: int,
    max_rss_bytes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if run.validation["study_role"] != "primary":
        return (
            pd.DataFrame(columns=[
                "run_label", "protocol", "estimand", "subgroup_dimension",
                "subgroup_value", "subgroup_order", "n_queries", "n_windows",
                "effect_type", "model", "contrast", "left_model",
                "right_model", "metric", "k", "bootstrap_replicates",
                "bootstrap_seed", "bootstrap_method", "estimate",
                "bootstrap_se", "ci_low", "ci_high", "probability_gt_zero",
                "probability_lt_zero",
            ]),
            pd.DataFrame(columns=[
                "run_label", "protocol", "subgroup_dimension",
                "subgroup_value", "subgroup_order", "n_queries", "n_windows",
                "population_weight_sum", "interpretation",
            ]),
        )
    metadata = _aligned_users(primary, users)
    definitions = [
        ("training-interaction activity", "activity_group", ACTIVITY_GROUPS),
        (
            "query-mean target pre-cutoff popularity",
            "target_popularity_group", TARGET_POPULARITY_GROUPS,
        ),
    ]
    results: list[dict] = []
    count_records: list[dict] = []
    group_counter = 0
    for dimension, column, order in definitions:
        for order_index, group in enumerate(order, start=1):
            mask = metadata[column].astype(str).eq(group).to_numpy()
            n_queries = int(mask.sum())
            if not n_queries:
                raise ValueError(f"Empty prespecified subgroup: {dimension}/{group}")
            represented_windows = int(
                metadata.index.get_level_values("window")[mask].nunique()
            )
            count_records.append({
                "run_label": run.spec.label,
                "protocol": run.config["protocol"],
                "subgroup_dimension": dimension,
                "subgroup_value": group,
                "subgroup_order": order_index,
                "n_queries": n_queries,
                "n_windows": represented_windows,
                "population_weight_sum": float(
                    metadata.loc[mask, "population_weight"].sum()
                ),
                "interpretation": (
                    "descriptive heterogeneity only; target popularity is outcome-derived"
                    if column == "target_popularity_group"
                    else "descriptive heterogeneity only"
                ),
            })

            descriptors: list[dict] = []
            vectors: list[np.ndarray] = []
            for metric, k in PRIMARY_OUTCOMES:
                for model in MODELS:
                    descriptors.append({
                        "effect_type": "full_catalog_effectiveness",
                        "model": model,
                        "contrast": "",
                        "left_model": "",
                        "right_model": "",
                        "metric": metric,
                        "k": k,
                    })
                    vectors.append(_metric_vector(
                        primary, "full_catalog", "full_catalog", 0, -1,
                        model, metric, k,
                    ))
                for left, right, contrast_name in CONTRASTS:
                    descriptors.append({
                        "effect_type": "full_catalog_contrast",
                        "model": "",
                        "contrast": contrast_name,
                        "left_model": left,
                        "right_model": right,
                        "metric": metric,
                        "k": k,
                    })
                    vectors.append(_contrast_vector(
                        primary, "full_catalog", "full_catalog", 0, -1,
                        left, right, metric, k,
                    ))
            matrix = np.column_stack(vectors)
            bootstrap = _stratified_bootstrap_batch(
                matrix, metadata, reps, seed + group_counter, mask=mask
            )
            for estimand, (points, bootstrap_values) in bootstrap.items():
                for index, descriptor in enumerate(descriptors):
                    results.append({
                        "run_label": run.spec.label,
                        "protocol": run.config["protocol"],
                        "estimand": estimand,
                        "subgroup_dimension": dimension,
                        "subgroup_value": group,
                        "subgroup_order": order_index,
                        "n_queries": n_queries,
                        "n_windows": represented_windows,
                        **descriptor,
                        "bootstrap_replicates": reps,
                        "bootstrap_seed": seed + group_counter,
                        "bootstrap_method": (
                            "paired users resampled within each represented window subgroup"
                        ),
                        **_interval_record(points[index], bootstrap_values[:, index]),
                    })
            group_counter += 1
            check_memory(max_rss_bytes)
    return pd.DataFrame(results), pd.DataFrame(count_records)


SUBGROUP_BIAS_OUTCOMES = (("hit_rate", 20), ("map", 20))


def _empty_subgroup_bias_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "run_label", "protocol", "estimand", "effect_type",
        "subgroup_dimension", "subgroup_value", "subgroup_order",
        "n_queries", "n_windows", "model", "metric", "k",
        "candidate_regime", "sampling_strategy", "seed",
        "seed_aggregation", "requested_candidate_size",
        "full_catalog_value", "sampled_value",
        "bootstrap_replicates", "bootstrap_seed", "bootstrap_method",
        "estimate", "bootstrap_se", "ci_low", "ci_high",
        "probability_gt_zero", "probability_lt_zero", "interpretation",
    ])


def bootstrap_subgroup_evaluation_bias(
    primary: pd.DataFrame,
    users: pd.DataFrame,
    run: ValidatedRun,
    reps: int,
    seed: int,
    max_rss_bytes: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Paired sampled-minus-full subgroup distortion for the primary run.

    Each sampled value is averaged across seeds 1--5 within user before the
    same user's full-catalog value is subtracted. Users are then resampled with
    replacement independently within each represented provider window. The
    250-user/window temporal sensitivities deliberately return no subgroup
    distortion rows because they were not powered for fine subgroup claims.
    """
    role = run.validation["study_role"]
    if role != "primary":
        return _empty_subgroup_bias_frame(), {
            "status": "disabled_by_protocol",
            "study_role": role,
            "reason": (
                "Temporal sensitivities use 250 users per window and support "
                "protocol-level robustness only, not fine subgroup inference."
            ),
            "rows": 0,
        }

    metadata = _aligned_users(primary, users)
    definitions = [
        ("training-interaction activity", "activity_group", ACTIVITY_GROUPS),
        (
            "query-mean target pre-cutoff popularity",
            "target_popularity_group", TARGET_POPULARITY_GROUPS,
        ),
    ]
    descriptors: list[dict[str, object]] = []
    full_columns: list[np.ndarray] = []
    sampled_columns: list[np.ndarray] = []
    bias_columns: list[np.ndarray] = []
    for metric, k in SUBGROUP_BIAS_OUTCOMES:
        for model in MODELS:
            full = _metric_vector(
                primary, "full_catalog", "full_catalog", 0, -1,
                model, metric, k,
            )
            for strategy in STRATEGIES:
                for size in SIZES:
                    sampled = np.mean([
                        _metric_vector(
                            primary, "sampled", strategy, sampling_seed, size,
                            model, metric, k,
                        )
                        for sampling_seed in SEEDS
                    ], axis=0)
                    descriptors.append({
                        "effect_type": "sampled_minus_full_metric_bias",
                        "model": model,
                        "metric": metric,
                        "k": k,
                        "candidate_regime": "sampled",
                        "sampling_strategy": strategy,
                        "seed": 0,
                        "seed_aggregation": "per_user_mean_over_5_seeds",
                        "requested_candidate_size": size,
                    })
                    full_columns.append(full)
                    sampled_columns.append(sampled)
                    bias_columns.append(sampled - full)

    full_matrix = np.column_stack(full_columns)
    sampled_matrix = np.column_stack(sampled_columns)
    bias_matrix = np.column_stack(bias_columns)
    if (
        full_matrix.shape != (len(metadata), 120)
        or sampled_matrix.shape != full_matrix.shape
        or bias_matrix.shape != full_matrix.shape
        or not np.isfinite(bias_matrix).all()
    ):
        raise ValueError(
            f"Unexpected primary subgroup-bias matrix shape {bias_matrix.shape}"
        )

    weights = metadata["population_weight"].to_numpy(float)
    records: list[dict[str, object]] = []
    group_counter = 0
    for dimension, column, order in definitions:
        for order_index, group in enumerate(order, start=1):
            mask = metadata[column].astype(str).eq(group).to_numpy()
            n_queries = int(mask.sum())
            if not n_queries:
                raise ValueError(f"Empty prespecified subgroup: {dimension}/{group}")
            n_windows = int(
                metadata.index.get_level_values("window")[mask].nunique()
            )
            if n_windows != len(WINDOWS):
                raise ValueError(
                    f"Primary subgroup {dimension}/{group} represents "
                    f"{n_windows} rather than seven windows"
                )
            bootstrapped = _stratified_bootstrap_batch(
                bias_matrix, metadata, reps, seed + group_counter, mask=mask
            )
            for estimand, (points, replicate_values) in bootstrapped.items():
                if estimand == "audit_cohort":
                    full_points = full_matrix[mask].mean(axis=0)
                    sampled_points = sampled_matrix[mask].mean(axis=0)
                elif estimand == "design_weighted_population":
                    full_points = np.average(
                        full_matrix[mask], axis=0, weights=weights[mask]
                    )
                    sampled_points = np.average(
                        sampled_matrix[mask], axis=0, weights=weights[mask]
                    )
                else:
                    raise ValueError(f"Unknown subgroup-bias estimand: {estimand}")
                if not np.allclose(
                    sampled_points - full_points, points,
                    rtol=1e-11, atol=1e-12,
                ):
                    raise ValueError(
                        f"Subgroup-bias point decomposition failed for "
                        f"{dimension}/{group}/{estimand}"
                    )
                for index, descriptor in enumerate(descriptors):
                    records.append({
                        "run_label": run.spec.label,
                        "protocol": run.config["protocol"],
                        "estimand": estimand,
                        "subgroup_dimension": dimension,
                        "subgroup_value": group,
                        "subgroup_order": order_index,
                        "n_queries": n_queries,
                        "n_windows": n_windows,
                        **descriptor,
                        "full_catalog_value": float(full_points[index]),
                        "sampled_value": float(sampled_points[index]),
                        "bootstrap_replicates": reps,
                        "bootstrap_seed": seed + group_counter,
                        "bootstrap_method": (
                            "paired users resampled within each represented "
                            "window subgroup after per-user five-seed averaging"
                        ),
                        **_interval_record(
                            points[index], replicate_values[:, index]
                        ),
                        "interpretation": (
                            "descriptive distortion heterogeneity; target "
                            "popularity is outcome-derived; not fairness or causal inference"
                            if column == "target_popularity_group"
                            else "descriptive distortion heterogeneity; not fairness or causal inference"
                        ),
                    })
            group_counter += 1
            check_memory(max_rss_bytes)

    output = pd.DataFrame(records)
    expected_rows = (
        (len(ACTIVITY_GROUPS) + len(TARGET_POPULARITY_GROUPS))
        * len(MODELS) * len(SUBGROUP_BIAS_OUTCOMES)
        * len(STRATEGIES) * len(SIZES) * 2
    )
    if len(output) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} primary subgroup-bias rows; "
            f"found {len(output):,}"
        )
    duplicate_keys = [
        "estimand", "subgroup_dimension", "subgroup_value", "model",
        "metric", "k", "sampling_strategy", "requested_candidate_size",
    ]
    if output.duplicated(duplicate_keys).any():
        raise ValueError("Duplicate primary subgroup-bias result keys")
    return output, {
        "status": "completed",
        "study_role": role,
        "rows": int(len(output)),
        "descriptors_per_group": len(descriptors),
        "metrics": [f"{metric}@{k}" for metric, k in SUBGROUP_BIAS_OUTCOMES],
        "models": list(MODELS),
        "samplers": list(STRATEGIES),
        "candidate_sizes": list(SIZES),
        "seed_aggregation": "per_user_mean_over_5_seeds",
        "bootstrap_replicates": reps,
    }


def window_variability(per_window: pd.DataFrame) -> pd.DataFrame:
    """Equal-window descriptive sensitivity; not the primary pooled estimand."""
    keys = [
        "run_label", "protocol", "estimand", "candidate_regime",
        "sampling_strategy", "seed", "requested_candidate_size", "model", "k",
    ]
    records: list[dict] = []
    for group_values, cell in per_window.groupby(keys, dropna=False, sort=False):
        if set(cell["window"].astype(int)) != set(WINDOWS):
            raise ValueError(f"Incomplete seven-window cell: {dict(zip(keys, group_values))}")
        common = dict(zip(keys, group_values))
        for metric in METRICS:
            values = cell[metric].to_numpy(float)
            records.append({
                **common,
                "metric": metric,
                "windows": len(values),
                "equal_window_mean": float(values.mean()),
                "window_sd": float(values.std(ddof=1)),
                "window_min": float(values.min()),
                "window_max": float(values.max()),
                "summary_role": "descriptive equal-window sensitivity",
            })
    return pd.DataFrame(records)


def protocol_comparison(pooled_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    pooled = pd.concat(pooled_frames, ignore_index=True)
    selected = pooled[
        pooled["estimand"].eq("design_weighted_population")
        & (
            pooled["candidate_regime"].eq("full_catalog")
            | (
                pooled["candidate_regime"].eq("sampled")
                & pooled["seed"].eq(1)
                & pooled["requested_candidate_size"].isin([100, 10000])
            )
        )
    ].copy()
    return selected.sort_values([
        "model", "k", "candidate_regime", "sampling_strategy",
        "requested_candidate_size", "run_label",
    ])


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(
            directory / f"{stem}.{extension}", bbox_inches="tight",
            facecolor="white", **kwargs,
        )
    plt.close(fig)


def _add_figure_header(
    fig: plt.Figure,
    handles: Sequence[object],
    labels: Sequence[str],
    title: str,
    *,
    ncol: int,
    axes_top: float,
) -> None:
    """Place a title and external legend in a reserved, non-overlapping band."""
    fig.suptitle(title, y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=ncol,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.0,
    )
    fig.tight_layout(rect=[0, 0, 1, axes_top])


def make_figures(
    output_dir: Path,
    pooled: pd.DataFrame,
    per_window: pd.DataFrame,
    bias_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    contrast_bootstrap: pd.DataFrame,
    subgroup_bootstrap: pd.DataFrame,
    subgroup_evaluation_bias: pd.DataFrame,
) -> list[str]:
    matplotlib.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })
    stems: list[str] = []

    # Figure 1: signed (not ratio) bias at two central outcomes.
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True)
    outcomes = [("hit_rate", 20, "HitRate@20"), ("map", 150, "MAP@150")]
    for row, (metric, k, outcome_label) in enumerate(outcomes):
        for column, strategy in enumerate(STRATEGIES):
            ax = axes[row, column]
            cell = bias_summary[
                bias_summary["estimand"].eq("design_weighted_population")
                & bias_summary["metric"].eq(metric)
                & bias_summary["k"].eq(k)
                & bias_summary["sampling_strategy"].eq(strategy)
            ]
            for model in MODELS:
                model_cell = cell[cell["model"].eq(model)].sort_values(
                    "requested_candidate_size"
                )
                if model_cell.empty:
                    continue
                ax.plot(
                    model_cell["requested_candidate_size"],
                    model_cell["mean_signed_bias"],
                    color=MODEL_COLORS[model], marker="o", linewidth=1.4,
                    markersize=3.5, label=MODEL_LABELS[model],
                )
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_xscale("log")
            ax.grid(alpha=0.2)
            ax.set_title(f"{outcome_label}; {STRATEGY_LABELS[strategy]}")
            ax.set_ylabel("Sampled − full")
            ax.set_xlabel("Nominal candidate size")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_figure_header(
        fig, handles, labels, "CareerBuilder sampled-catalog metric bias",
        ncol=3, axes_top=0.84,
    )
    save_figure(fig, output_dir, "figure_1_signed_metric_bias")
    stems.append("figure_1_signed_metric_bias")

    # Figure 2: tie-aware rank and pairwise fidelity.
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.8), sharex=True, sharey="row")
    for column, strategy in enumerate(STRATEGIES):
        cell = rank_summary[
            rank_summary["rank_scope"].eq("pooled")
            & rank_summary["estimand"].eq("design_weighted_population")
            & rank_summary["metric"].eq("hit_rate")
            & rank_summary["k"].eq(20)
            & rank_summary["sampling_strategy"].eq(strategy)
        ].sort_values("requested_candidate_size")
        axes[0, column].plot(
            cell["requested_candidate_size"], cell["kendall_tau_b_mean"],
            color="#0072B2", marker="o",
        )
        axes[1, column].plot(
            cell["requested_candidate_size"],
            cell["pairwise_order_agreement_mean"],
            color="#D55E00", marker="s",
        )
        for ax in axes[:, column]:
            ax.set_xscale("log")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Nominal candidate size")
        axes[0, column].set_title(STRATEGY_LABELS[strategy])
    axes[0, 0].set_ylabel("Explicit-tie Kendall tau-b")
    axes[1, 0].set_ylabel("Pairwise order agreement")
    fig.suptitle("Full-versus-sampled HitRate@20 ranking fidelity")
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_2_rank_fidelity")
    stems.append("figure_2_rank_fidelity")

    # Figure 3: all seven windows, full catalog only.
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    full = per_window[
        per_window["candidate_regime"].eq("full_catalog")
        & per_window["estimand"].eq("audit_cohort")
        & per_window["k"].eq(20)
    ]
    for model in MODELS:
        cell = full[full["model"].eq(model)].sort_values("window")
        ax.plot(
            cell["window"], cell["hit_rate"], marker="o", linewidth=1.5,
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
        )
    ax.set_xticks(WINDOWS)
    ax.set_xlabel("CareerBuilder window")
    ax.set_ylabel("Sampling-free HitRate@20")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, frameon=False)
    ax.set_title("Sampling-free effectiveness across seven temporal windows")
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_3_seven_window_full_catalog")
    stems.append("figure_3_seven_window_full_catalog")

    # Figure 4: five prespecified contrast distortions, five-seed mean.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    contrast_colors = dict(zip(
        [name for _, _, name in CONTRASTS],
        ["#E69F00", "#D55E00", "#009E73", "#0072B2", "#7F7F7F"],
    ))
    for ax, strategy in zip(axes, STRATEGIES):
        cell = contrast_bootstrap[
            contrast_bootstrap["estimand"].eq("design_weighted_population")
            & contrast_bootstrap["effect_type"].eq(
                "effect_distortion_sampled_minus_full"
            )
            & contrast_bootstrap["seed_aggregation"].eq("mean_over_5_seeds")
            & contrast_bootstrap["metric"].eq("hit_rate")
            & contrast_bootstrap["k"].eq(20)
            & contrast_bootstrap["sampling_strategy"].eq(strategy)
        ]
        for contrast in contrast_colors:
            contrast_cell = cell[cell["contrast"].eq(contrast)].sort_values(
                "requested_candidate_size"
            )
            x = contrast_cell["requested_candidate_size"].to_numpy(float)
            y = contrast_cell["estimate"].to_numpy(float)
            ax.errorbar(
                x, y,
                yerr=np.vstack((
                    y - contrast_cell["ci_low"].to_numpy(float),
                    contrast_cell["ci_high"].to_numpy(float) - y,
                )),
                color=contrast_colors[contrast], marker="o", capsize=2,
                linewidth=1.2, label=CONTRAST_LABELS[contrast],
            )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Nominal candidate size")
        ax.set_title(STRATEGY_LABELS[strategy])
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(
        "Sampled − sampling-free\nmodel difference (HitRate@20)"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    _add_figure_header(
        fig, handles, labels,
        "Change in BM25 model differences under sampled evaluation",
        ncol=2, axes_top=0.74,
    )
    save_figure(fig, output_dir, "figure_4_contrast_distortion")
    stems.append("figure_4_contrast_distortion")

    # Figure 5: descriptive subgroup effectiveness for the primary cohort only.
    if not subgroup_bootstrap.empty:
        display_models = (
            "bm25_full_static", "bm25_title_static", "tfidf_combined_static",
            "minilm_title_static", "popularity_precut",
        )
        dimensions = [
            "training-interaction activity",
            "query-mean target pre-cutoff popularity",
        ]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
        for ax, dimension in zip(axes, dimensions):
            cell = subgroup_bootstrap[
                subgroup_bootstrap["estimand"].eq("design_weighted_population")
                & subgroup_bootstrap["effect_type"].eq("full_catalog_effectiveness")
                & subgroup_bootstrap["metric"].eq("hit_rate")
                & subgroup_bootstrap["k"].eq(20)
                & subgroup_bootstrap["subgroup_dimension"].eq(dimension)
            ]
            groups = (
                ACTIVITY_GROUPS if dimension == dimensions[0]
                else TARGET_POPULARITY_GROUPS
            )
            offsets = np.linspace(-0.24, 0.24, len(display_models))
            for offset, model in zip(offsets, display_models):
                model_cell = cell[cell["model"].eq(model)].set_index(
                    "subgroup_value"
                ).loc[list(groups)]
                y = model_cell["estimate"].to_numpy(float)
                x = np.arange(len(groups)) + offset
                ax.errorbar(
                    x, y,
                    yerr=np.vstack((
                        y - model_cell["ci_low"].to_numpy(float),
                        model_cell["ci_high"].to_numpy(float) - y,
                    )),
                    fmt="o", capsize=2, color=MODEL_COLORS[model],
                    label=MODEL_LABELS[model],
                )
            ax.set_xticks(
                range(len(groups)),
                [SUBGROUP_LABELS.get(group, group.replace("_", " ")) for group in groups],
            )
            ax.tick_params(axis="x", rotation=20)
            ax.set_title(
                "Pre-cutoff application activity"
                if dimension == "training-interaction activity"
                else "Mean target pre-cutoff popularity"
            )
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Sampling-free HitRate@20")
        handles, labels = axes[0].get_legend_handles_labels()
        _add_figure_header(
            fig, handles, labels,
            "Descriptive subgroup effectiveness with 95% bootstrap intervals",
            ncol=3, axes_top=0.80,
        )
        save_figure(fig, output_dir, "figure_5_subgroup_effectiveness")
        stems.append("figure_5_subgroup_effectiveness")

    # Figure 6: paired subgroup distortion for the primary 1,000/window cohort.
    if not subgroup_evaluation_bias.empty:
        dimensions = [
            "training-interaction activity",
            "query-mean target pre-cutoff popularity",
        ]
        fig, axes = plt.subplots(
            2, 2, figsize=(12.0, 8.0), sharey=True, squeeze=False
        )
        for row, dimension in enumerate(dimensions):
            groups = (
                ACTIVITY_GROUPS if dimension == dimensions[0]
                else TARGET_POPULARITY_GROUPS
            )
            for column, strategy in enumerate(STRATEGIES):
                ax = axes[row, column]
                cell = subgroup_evaluation_bias[
                    subgroup_evaluation_bias["estimand"].eq(
                        "design_weighted_population"
                    )
                    & subgroup_evaluation_bias["metric"].eq("hit_rate")
                    & subgroup_evaluation_bias["k"].eq(20)
                    & subgroup_evaluation_bias["sampling_strategy"].eq(strategy)
                    & subgroup_evaluation_bias["requested_candidate_size"].eq(100)
                    & subgroup_evaluation_bias["subgroup_dimension"].eq(dimension)
                ]
                offsets = np.linspace(-0.30, 0.30, len(MODELS))
                for offset, model in zip(offsets, MODELS):
                    model_cell = cell[cell["model"].eq(model)].set_index(
                        "subgroup_value"
                    ).loc[list(groups)]
                    x = np.arange(len(groups)) + offset
                    y = model_cell["estimate"].to_numpy(float)
                    ax.errorbar(
                        x, y,
                        yerr=np.vstack((
                            y - model_cell["ci_low"].to_numpy(float),
                            model_cell["ci_high"].to_numpy(float) - y,
                        )),
                        fmt="o", capsize=2, color=MODEL_COLORS[model],
                        label=MODEL_LABELS[model],
                    )
                ax.axhline(0, color="#333333", linewidth=0.8)
                ax.set_xticks(
                    range(len(groups)),
                    [
                        SUBGROUP_LABELS.get(group, group.replace("_", " "))
                        for group in groups
                    ],
                )
                ax.tick_params(axis="x", rotation=20)
                dimension_label = (
                    "Pre-cutoff application activity"
                    if dimension == "training-interaction activity"
                    else "Mean target pre-cutoff popularity"
                )
                ax.set_title(f"{dimension_label}; {STRATEGY_LABELS[strategy]}")
                ax.grid(axis="y", alpha=0.2)
        axes[0, 0].set_ylabel("100-candidate − sampling-free HitRate@20")
        axes[1, 0].set_ylabel("100-candidate − sampling-free HitRate@20")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        _add_figure_header(
            fig, handles, labels,
            "CareerBuilder HitRate@20 differences across descriptive subgroups",
            ncol=3, axes_top=0.86,
        )
        save_figure(fig, output_dir, "figure_6_subgroup_evaluation_bias")
        stems.append("figure_6_subgroup_evaluation_bias")
    return stems


def write_publication_tables(
    output_dir: Path,
    pooled: pd.DataFrame,
    per_window: pd.DataFrame,
    bias_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    contrast_bootstrap: pd.DataFrame,
    subgroup_bootstrap: pd.DataFrame,
    subgroup_counts: pd.DataFrame,
    candidate_scale: pd.DataFrame,
    subgroup_evaluation_bias: pd.DataFrame,
) -> None:
    full = pooled[
        pooled["candidate_regime"].eq("full_catalog")
        & pooled["estimand"].eq("design_weighted_population")
    ].copy()
    full.to_csv(output_dir / "publication_table_1_full_catalog.csv", index=False)
    bias_summary[
        bias_summary["estimand"].eq("design_weighted_population")
        & (
            (bias_summary["metric"].eq("hit_rate") & bias_summary["k"].eq(20))
            | (bias_summary["metric"].eq("ndcg") & bias_summary["k"].eq(20))
            | (bias_summary["metric"].eq("map") & bias_summary["k"].isin([20, 150]))
        )
    ].to_csv(output_dir / "publication_table_2_sampling_bias.csv", index=False)
    rank_summary[
        rank_summary["rank_scope"].eq("pooled")
        & rank_summary["estimand"].eq("design_weighted_population")
    ].to_csv(output_dir / "publication_table_3_rank_fidelity.csv", index=False)
    contrast_bootstrap[
        contrast_bootstrap["candidate_regime"].eq("full_catalog")
        | contrast_bootstrap["seed_aggregation"].eq("mean_over_5_seeds")
    ].to_csv(output_dir / "publication_table_4_contrast_bootstrap.csv", index=False)
    if not subgroup_bootstrap.empty:
        subgroup_bootstrap[
            subgroup_bootstrap["estimand"].eq("design_weighted_population")
        ].to_csv(output_dir / "publication_table_5_subgroups.csv", index=False)
        subgroup_counts.to_csv(
            output_dir / "publication_table_5b_subgroup_counts.csv", index=False
        )
    candidate_scale.to_csv(
        output_dir / "publication_table_6_candidate_scale.csv", index=False
    )
    per_window[
        per_window["candidate_regime"].eq("full_catalog")
        & per_window["estimand"].eq("audit_cohort")
    ].to_csv(output_dir / "publication_table_7_full_by_window.csv", index=False)
    if not subgroup_evaluation_bias.empty:
        publication_subgroup_bias = subgroup_evaluation_bias[
            subgroup_evaluation_bias["estimand"].eq(
                "design_weighted_population"
            )
        ].sort_values([
            "subgroup_dimension", "subgroup_order", "metric",
            "sampling_strategy", "model", "requested_candidate_size",
        ])
        expected_publication_rows = (
            (len(ACTIVITY_GROUPS) + len(TARGET_POPULARITY_GROUPS))
            * len(MODELS) * len(SUBGROUP_BIAS_OUTCOMES)
            * len(STRATEGIES) * len(SIZES)
        )
        if len(publication_subgroup_bias) != expected_publication_rows:
            raise ValueError(
                f"Expected {expected_publication_rows:,} publication subgroup-"
                f"distortion rows; found {len(publication_subgroup_bias):,}"
            )
        publication_subgroup_bias.to_csv(
            output_dir / "publication_table_9_subgroup_evaluation_bias.csv",
            index=False,
        )


def figure_source_map(
    include_subgroup_effectiveness: bool,
    include_subgroup_bias: bool,
) -> str:
    text = """# CareerBuilder figure source map

- **Figure 1 — sampled-catalog metric bias.** Signed sampled-minus-full
  differences (rather than unstable ratios) from `sampling_bias_seed_summary.csv`.
- **Figure 2 — ranking fidelity.** Tolerance-explicit Kendall tau-b and
  three-state pairwise order agreement from `rank_fidelity_summary.csv`.
- **Figure 3 — seven temporal windows.** Full-catalog HitRate@20 from
  `per_window_results.csv`; equal-window dispersion is descriptive only.
- **Figure 4 — BM25 contrast distortion.** Five-seed-mean, paired,
  within-window-stratified user-bootstrap estimates from
  `prespecified_contrast_bootstrap.csv`.
"""
    if include_subgroup_effectiveness:
        text += """
- **Figure 5 — subgroup effectiveness.** Descriptive full-catalog HitRate@20
  with within-window-stratified user-bootstrap intervals from
  `subgroup_bootstrap.csv`; query-mean target popularity is outcome-derived.
"""
    else:
        text += """
- **Subgroup-effectiveness figure intentionally omitted.** Fine subgroup
  inference is disabled for the 250-user-per-window temporal sensitivities.
"""
    if include_subgroup_bias:
        text += """
- **Figure 6 — subgroup evaluation bias.** Paired sampled-minus-full
  HitRate@20 after per-user averaging over seeds 1--5, with users resampled
  within each represented window, from
  `subgroup_evaluation_bias_cluster_bootstrap.csv`. This figure is produced
  only for the primary 1,000-user-per-window audit; target-popularity groups are
  outcome-derived and all subgroup patterns are descriptive.
"""
    else:
        text += """
- **Subgroup-distortion figure intentionally omitted.** The 250-user-per-window
  temporal-sensitivity cohort supports protocol-level robustness but not fine
  subgroup claims; see `SUBGROUP_EVALUATION_BIAS_STATUS.json`.
"""
    return text + "\nEvery emitted figure is a 300-dpi PNG plus vector PDF and SVG.\n"


def _json_default(value: object) -> object:
    """Convert NumPy scalars/arrays and paths without weakening JSON checks."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def analysis_protocol_markdown(
    run: ValidatedRun, reps: int, seed: int, max_rss_bytes: int
) -> str:
    return f"""# CareerBuilder internal-temporal statistical analysis protocol

## Scope and authority

This analysis concerns the internally reconstructed `{run.config['protocol']}`
task only. It is not the CareerBuilder competition test and is not official
competition MAP. Metric files were read only after `RUN_REPORT.json` explicitly
declared `authoritative=true`, `production=true`, `smoke=false`, and
`status=completed`, and after every checkpoint hash and coverage interval passed.

## Estimands

- `audit_cohort`: the unweighted query mean in the fixed
  {run.users_per_window:,}-query-per-window audit sample
  ({run.users_per_window * len(WINDOWS):,} queries).
- `design_weighted_population`: inverse-inclusion-weighted query mean using
  the evaluator's window-specific `N_w / n_w` weights. This is the primary
  finite-population descriptive estimand.
- Equal-window means and standard deviations are retained only as sensitivity
  descriptions; they are not substituted for query-level means.

## Bias and ranking fidelity

Bias is always reported as the signed difference `sampled - full` and its
absolute magnitude; ratios are not used. Candidate fraction is computed from
the realized per-query count divided by that query's exact eligible catalog.
The nominal-100 / k=150 cell is absent by design.

Kendall tau-b uses SciPy {scipy.__version__} with tolerance-defined tie groups
(`rtol={TIE_RTOL}`, `atol={TIE_ATOL}`) before `kendalltau(variant='b')`.
Winner sets contain every model tied with the maximum under the same tolerance.
Pairwise fidelity is exact agreement over the three states {{loss, tie, win}}.
Regret is the full-catalog maximum minus the best/worst full-catalog score among
the sampled winner set.

## Paired bootstrap

The five prespecified contrasts are title BM25 minus title MiniLM, full-text
BM25 minus title MiniLM, full-text BM25 minus combined TF-IDF, full-text BM25
minus popularity, and full-text BM25 minus random. The pairing unit is the user
query. In each of {reps:,} replicates, users are sampled with replacement
independently within each of the seven windows, while the same multiplicities
are applied to both models and—when estimating distortion—to sampled and full
conditions. The fixed primary seed is {seed}. Percentile 95% intervals,
bootstrap standard errors, and smoothed empirical probabilities above/below
zero are reported. Subgroups use the same algorithm within each represented
window subgroup. No interval is described as causal or as a fairness audit.

## Subgroups and computational boundary

Activity is the released pre-cutoff training-interaction count (`cold_0`,
`one_1`, `low_2_4`, `high_5plus`). Target popularity is the query mean of its
targets' pre-cutoff application counts; it is explicitly outcome-derived and
therefore descriptive. For the primary 1,000-user-per-window audit only,
HitRate@20 and MAP@20 sampled-minus-full subgroup distortion is estimated for
all six models, both samplers, and every candidate size. Each sampled value is
averaged over seeds 1--5 within user before subtracting that user's full-catalog
value; the paired differences use the same within-window stratified bootstrap.
Temporal sensitivities use 250 users per window and explicitly disable fine
subgroup-effectiveness and subgroup-distortion inference. The evaluator scores the full catalog before indexing
sampled candidates, so these results do not establish model-scoring or
end-to-end computational savings. The analyzer RSS ceiling is
{max_rss_bytes / 2**30:.2f} GiB.
"""


def verified_findings_markdown(
    run: ValidatedRun,
    pooled: pd.DataFrame,
    bias_summary: pd.DataFrame,
    rank_summary: pd.DataFrame,
    candidate_scale: pd.DataFrame,
    subgroup_bias_status: Mapping[str, object],
) -> str:
    full = pooled[
        pooled["candidate_regime"].eq("full_catalog")
        & pooled["estimand"].eq("design_weighted_population")
        & pooled["k"].eq(20)
    ].sort_values("hit_rate", ascending=False)
    full_lines = [
        f"- {MODEL_LABELS[row.model]}: HitRate@20={row.hit_rate:.6f}, "
        f"NDCG@20={row.ndcg:.6f}, MAP@20={row.map:.6f}."
        for row in full.itertuples(index=False)
    ]
    scale_full = candidate_scale[
        candidate_scale["sampling_strategy"].eq("full_catalog")
    ].iloc[0]
    rank_100 = rank_summary[
        rank_summary["rank_scope"].eq("pooled")
        & rank_summary["estimand"].eq("design_weighted_population")
        & rank_summary["metric"].eq("hit_rate")
        & rank_summary["k"].eq(20)
        & rank_summary["requested_candidate_size"].eq(100)
    ]
    rank_lines = [
        f"- {STRATEGY_LABELS[row.sampling_strategy]} at nominal 100: "
        f"mean tau-b={row.kendall_tau_b_mean:.6f}, mean pairwise agreement="
        f"{row.pairwise_order_agreement_mean:.6f}, exact winner-set rate="
        f"{row.winner_set_exact_match_rate:.6f}."
        for row in rank_100.itertuples(index=False)
    ]
    largest_bias = bias_summary[
        bias_summary["estimand"].eq("design_weighted_population")
        & bias_summary["metric"].eq("hit_rate")
        & bias_summary["k"].eq(20)
    ].sort_values("absolute_mean_bias", ascending=False).head(1)
    bias_line = ""
    if not largest_bias.empty:
        row = largest_bias.iloc[0]
        bias_line = (
            f"The largest mean signed HitRate@20 bias among the evaluated cells was "
            f"{row['mean_signed_bias']:.6f} for {MODEL_LABELS[row['model']]} under "
            f"{STRATEGY_LABELS[row['sampling_strategy']]} at nominal "
            f"{int(row['requested_candidate_size']):,}."
        )
    if subgroup_bias_status.get("status") == "completed":
        subgroup_bias_line = (
            "Paired sampled-minus-full HitRate@20 and MAP@20 subgroup "
            "distortion is reported in "
            "`publication_table_9_subgroup_evaluation_bias.csv`; sampled "
            "values are per-user means over all five seeds before subtraction."
        )
    else:
        subgroup_bias_line = (
            "Fine subgroup-distortion inference is disabled for this temporal "
            "sensitivity because its 250-user-per-window cohort supports only "
            "protocol-level robustness."
        )
    return f"""# Verified findings: {run.spec.label}

## Scope

These findings passed the authority, hash, completeness, aggregate-reproduction,
and statistical-output checks in `analysis_integrity.json`. They apply only to
the internally reconstructed `{run.config['protocol']}` task and are not official
CareerBuilder competition results.

## Full-catalog design-weighted results

{chr(10).join(full_lines)}

The exact eligible catalog averaged {scale_full['eligible_candidate_count_mean']:.2f}
jobs per query (median {scale_full['eligible_candidate_count_median']:.2f}; range
{int(scale_full['eligible_candidate_count_min']):,}--
{int(scale_full['eligible_candidate_count_max']):,}).

## Sampling distortion and ranking fidelity

{bias_line}

{chr(10).join(rank_lines)}

Interpret the paired contrast intervals in
`publication_table_4_contrast_bootstrap.csv`; do not infer support from mean
values alone. Activity and target-popularity results are descriptive
heterogeneity summaries, not causal or fairness claims.

{subgroup_bias_line}
"""


def _interval_violation_count(
    frame: pd.DataFrame,
    finite_columns: Sequence[str],
) -> int:
    """Count non-finite values, inverted intervals, and uncovered estimates."""
    nonfinite_rows = (~np.isfinite(
        frame[list(finite_columns)].to_numpy(float)
    )).any(axis=1)
    return int(
        nonfinite_rows.sum()
        + (frame["ci_low"] > frame["ci_high"]).sum()
        + (frame["estimate"] < frame["ci_low"]).sum()
        + (frame["estimate"] > frame["ci_high"]).sum()
    )


def _validate_analysis_outputs(
    contrast_bootstrap: pd.DataFrame,
    subgroup_bootstrap: pd.DataFrame,
    subgroup_evaluation_bias: pd.DataFrame,
    subgroup_bias_status: Mapping[str, object],
    subgroup_counts: pd.DataFrame,
    rank_detail: pd.DataFrame,
    expected_users: int,
    study_role: str,
) -> dict[str, object]:
    interval_frames = {
        "contrast": contrast_bootstrap,
        "subgroup": subgroup_bootstrap,
    }
    violations = {}
    for name, frame in interval_frames.items():
        violations[name] = _interval_violation_count(
            frame, ("estimate", "bootstrap_se", "ci_low", "ci_high")
        )
    if not subgroup_evaluation_bias.empty:
        violations["subgroup_evaluation_bias"] = _interval_violation_count(
            subgroup_evaluation_bias,
            (
                "estimate", "bootstrap_se", "ci_low", "ci_high",
                "full_catalog_value", "sampled_value",
            ),
        )
    else:
        violations["subgroup_evaluation_bias"] = 0
    subgroup_sums = (
        subgroup_counts.groupby("subgroup_dimension")["n_queries"].sum()
        .astype(int).to_dict()
    )
    tau = rank_detail.loc[
        rank_detail["kendall_tau_b_defined"], "kendall_tau_b"
    ]
    checks = {
        "interval_violations": violations,
        "subgroup_query_sums": subgroup_sums,
        "five_prespecified_contrasts_present": (
            set(contrast_bootstrap["contrast"])
            == {name for _, _, name in CONTRASTS}
        ),
        "size100_k150_absent": not (
            contrast_bootstrap["requested_candidate_size"].eq(100)
            & contrast_bootstrap["k"].eq(150)
        ).any(),
        "defined_tau_b_range": (
            [float(tau.min()), float(tau.max())] if len(tau) else []
        ),
        "pairwise_agreement_range": [
            float(rank_detail["pairwise_order_agreement"].min()),
            float(rank_detail["pairwise_order_agreement"].max()),
        ],
        "subgroup_evaluation_bias": dict(subgroup_bias_status),
    }
    if study_role == "primary":
        expected_subgroup_effectiveness_rows = (
            (len(ACTIVITY_GROUPS) + len(TARGET_POPULARITY_GROUPS))
            * (len(MODELS) + len(CONTRASTS))
            * len(PRIMARY_OUTCOMES) * 2
        )
        subgroup_effectiveness_valid = (
            len(subgroup_bootstrap) == expected_subgroup_effectiveness_rows
            and len(subgroup_counts)
            == len(ACTIVITY_GROUPS) + len(TARGET_POPULARITY_GROUPS)
            and set(subgroup_sums.values()) == {expected_users}
            and set(subgroup_bootstrap["estimand"])
            == {"audit_cohort", "design_weighted_population"}
            and subgroup_bootstrap["n_windows"].eq(len(WINDOWS)).all()
            and subgroup_counts["n_windows"].eq(len(WINDOWS)).all()
        )
        expected_bias_rows = (
            (len(ACTIVITY_GROUPS) + len(TARGET_POPULARITY_GROUPS))
            * len(MODELS) * len(SUBGROUP_BIAS_OUTCOMES)
            * len(STRATEGIES) * len(SIZES) * 2
        )
        subgroup_bias_valid = (
            len(subgroup_evaluation_bias) == expected_bias_rows
            and subgroup_bias_status.get("status") == "completed"
            and set(subgroup_evaluation_bias["estimand"])
            == {"audit_cohort", "design_weighted_population"}
            and set(subgroup_evaluation_bias["model"]) == set(MODELS)
            and set(zip(
                subgroup_evaluation_bias["metric"],
                subgroup_evaluation_bias["k"].astype(int),
            )) == set(SUBGROUP_BIAS_OUTCOMES)
            and set(subgroup_evaluation_bias["sampling_strategy"])
            == set(STRATEGIES)
            and set(subgroup_evaluation_bias["requested_candidate_size"].astype(int))
            == set(SIZES)
            and set(subgroup_evaluation_bias["seed_aggregation"])
            == {"per_user_mean_over_5_seeds"}
            and subgroup_evaluation_bias["n_windows"].eq(len(WINDOWS)).all()
        )
    else:
        subgroup_effectiveness_valid = (
            subgroup_bootstrap.empty and subgroup_counts.empty
        )
        subgroup_bias_valid = (
            subgroup_evaluation_bias.empty
            and subgroup_bias_status.get("status") == "disabled_by_protocol"
        )
    checks["subgroup_evaluation_bias_valid_for_role"] = subgroup_bias_valid
    checks["subgroup_effectiveness_valid_for_role"] = subgroup_effectiveness_valid
    if (
        any(violations.values())
        or not subgroup_effectiveness_valid
        or not checks["five_prespecified_contrasts_present"]
        or not checks["size100_k150_absent"]
        or not subgroup_bias_valid
        or (len(tau) and (tau.min() < -1 - 1e-12 or tau.max() > 1 + 1e-12))
        or not rank_detail["pairwise_order_agreement"].between(0, 1).all()
    ):
        raise ValueError(f"Post-analysis integrity checks failed: {checks}")
    return checks


def analyze_run(
    validated: ValidatedRun,
    output_root: Path,
    reps: int,
    seed: int,
    max_rss_bytes: int,
) -> dict[str, object]:
    aggregate, primary, candidate, users, load_audit = load_and_verify_metrics(
        validated, max_rss_bytes
    )
    per_window = per_window_estimates(aggregate, validated)
    pooled = pooled_estimates(aggregate, validated)
    seeds = seed_summaries(pooled)
    bias_detail, bias_summary = sampling_bias(pooled)
    candidate_scale = candidate_scale_summary(candidate, validated)
    rank_detail, rank_summary = rank_fidelity(pooled, per_window)
    contrast_bootstrap = bootstrap_prespecified_contrasts(
        primary, users, validated, reps, seed, max_rss_bytes
    )
    subgroup_bootstrap, subgroup_counts = bootstrap_subgroups(
        primary, users, validated, reps, seed + 10_000, max_rss_bytes
    )
    if validated.validation["study_role"] == "primary":
        subgroup_effectiveness_status = {
            "status": "completed",
            "study_role": "primary",
            "rows": int(len(subgroup_bootstrap)),
            "count_rows": int(len(subgroup_counts)),
        }
    else:
        subgroup_effectiveness_status = {
            "status": "disabled_by_protocol",
            "study_role": validated.validation["study_role"],
            "reason": (
                "Temporal sensitivities use 250 users per window and support "
                "protocol-level robustness only, not fine subgroup inference."
            ),
            "rows": 0,
            "count_rows": 0,
        }
    subgroup_evaluation_bias, subgroup_bias_status = (
        bootstrap_subgroup_evaluation_bias(
            primary, users, validated, reps, seed + 20_000, max_rss_bytes
        )
    )
    window_summary = window_variability(per_window)
    post_checks = _validate_analysis_outputs(
        contrast_bootstrap, subgroup_bootstrap, subgroup_evaluation_bias,
        subgroup_bias_status, subgroup_counts, rank_detail,
        validated.users_per_window * len(WINDOWS),
        validated.validation["study_role"],
    )
    check_memory(max_rss_bytes)

    output_dir = output_root / validated.spec.label
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite an existing nonempty analysis directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_window_results.csv": per_window,
        "pooled_estimates.csv": pooled,
        "sampled_seed_summary.csv": seeds,
        "sampling_bias_by_seed.csv": bias_detail,
        "sampling_bias_seed_summary.csv": bias_summary,
        "candidate_scale_and_realized_fraction.csv": candidate_scale,
        "rank_fidelity_detail.csv": rank_detail,
        "rank_fidelity_summary.csv": rank_summary,
        "prespecified_contrast_bootstrap.csv": contrast_bootstrap,
        "subgroup_bootstrap.csv": subgroup_bootstrap,
        "subgroup_evaluation_bias_cluster_bootstrap.csv": (
            subgroup_evaluation_bias
        ),
        "subgroup_definitions_and_counts.csv": subgroup_counts,
        "window_variability_equal_window_sensitivity.csv": window_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    write_publication_tables(
        output_dir, pooled, per_window, bias_summary, rank_summary,
        contrast_bootstrap, subgroup_bootstrap, subgroup_counts, candidate_scale,
        subgroup_evaluation_bias,
    )
    stems = make_figures(
        output_dir, pooled, per_window, bias_summary, rank_summary,
        contrast_bootstrap, subgroup_bootstrap, subgroup_evaluation_bias,
    )
    (output_dir / "STUDY_PROTOCOL.md").write_text(
        analysis_protocol_markdown(validated, reps, seed, max_rss_bytes),
        encoding="utf-8",
    )
    (output_dir / "FIGURE_SOURCE_MAP.md").write_text(
        figure_source_map(
            not subgroup_bootstrap.empty,
            not subgroup_evaluation_bias.empty,
        ),
        encoding="utf-8",
    )
    write_json(
        output_dir / "SUBGROUP_EVALUATION_BIAS_STATUS.json",
        subgroup_bias_status,
    )
    write_json(
        output_dir / "SUBGROUP_EFFECTIVENESS_STATUS.json",
        subgroup_effectiveness_status,
    )
    (output_dir / "VERIFIED_FINDINGS.md").write_text(
        verified_findings_markdown(
            validated, pooled, bias_summary, rank_summary, candidate_scale,
            subgroup_bias_status,
        ),
        encoding="utf-8",
    )

    figure_variants = {
        stem: {
            extension: (output_dir / f"{stem}.{extension}").stat().st_size
            for extension in ("png", "pdf", "svg")
        }
        for stem in stems
    }
    if any(
        size <= 0 for variants in figure_variants.values() for size in variants.values()
    ):
        raise ValueError("At least one publication figure variant is empty")

    generated = [
        path for path in output_dir.iterdir()
        if path.is_file() and path.name != "analysis_integrity.json"
    ]
    integrity_report = {
        "status": "passed",
        "scope": (
            "CareerBuilder internally reconstructed temporal task; not official "
            "competition evaluation"
        ),
        "run_label": validated.spec.label,
        "protocol": validated.config["protocol"],
        "source_run": str(validated.spec.path),
        "authority_validation": validated.validation,
        "load_and_aggregate_reproduction": load_audit,
        "analysis_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        },
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
        },
        "memory": {
            "configured_max_rss_bytes": int(max_rss_bytes),
            "observed_peak_rss_bytes": int(_PEAK_RSS_BYTES),
        },
        "bootstrap": {
            "replicates": reps,
            "primary_seed": seed,
            "subgroup_seed_base": seed + 10_000,
            "subgroup_evaluation_bias_seed_base": seed + 20_000,
            "unit": "paired user query",
            "stratification": "independent resampling within each window",
            "interval": "2.5th and 97.5th percentile",
            "contrasts": [name for _, _, name in CONTRASTS],
        },
        "rank_rules": {
            "kendall": (
                f"SciPy {scipy.__version__} kendalltau variant=b after tolerance-defined "
                "tie grouping"
            ),
            "tie_rtol": TIE_RTOL,
            "tie_atol": TIE_ATOL,
            "winner": "all models tolerance-tied with the maximum",
            "pairwise": "exact agreement over {-1, 0, +1}; tie is a third state",
            "regret": (
                "full maximum minus best/worst full score among sampled winner set"
            ),
        },
        "subgroups": {
            "activity": list(ACTIVITY_GROUPS),
            "target_popularity": list(TARGET_POPULARITY_GROUPS),
            "interpretation": "descriptive only; not causal or fairness inference",
            "full_catalog_effectiveness": subgroup_effectiveness_status,
            "sampled_minus_full_bias": subgroup_bias_status,
        },
        "post_analysis_checks": post_checks,
        "row_counts": {name: int(len(frame)) for name, frame in outputs.items()},
        "figures": figure_variants,
        "generated_sha256": {path.name: sha256(path) for path in generated},
    }
    write_json(output_dir / "analysis_integrity.json", integrity_report)
    summary = {
        "run_label": validated.spec.label,
        "protocol": validated.config["protocol"],
        "output_dir": str(output_dir.resolve()),
        "integrity_path": str((output_dir / "analysis_integrity.json").resolve()),
        "pooled": pooled.copy(),
        "authority_validation": validated.validation,
    }
    del aggregate, primary, candidate, users
    gc.collect()
    return summary


def combined_protocol_table(summaries: Sequence[dict[str, object]]) -> pd.DataFrame:
    frames = [summary["pooled"] for summary in summaries]
    pooled = pd.concat(frames, ignore_index=True)
    full = pooled[pooled["candidate_regime"].eq("full_catalog")].copy()
    sampled = pooled[pooled["candidate_regime"].eq("sampled")].copy()
    grouping = [
        "run_label", "protocol", "estimand", "candidate_regime",
        "sampling_strategy", "requested_candidate_size", "model", "k",
    ]
    sampled_mean = sampled.groupby(grouping, dropna=False, as_index=False).agg(
        seeds=("seed", "nunique"),
        audit_users=("audit_users", "first"),
        population_weight_sum=("population_weight_sum", "first"),
        actual_candidate_fraction_mean=("actual_candidate_fraction_mean", "mean"),
        **{metric: (metric, "mean") for metric in METRICS},
    )
    full["seeds"] = 1
    keep = grouping + [
        "seeds", "audit_users", "population_weight_sum",
        "actual_candidate_fraction_mean", *METRICS,
    ]
    return pd.concat([full[keep], sampled_mean[keep]], ignore_index=True).sort_values(
        ["estimand", "model", "k", "candidate_regime", "sampling_strategy",
         "requested_candidate_size", "run_label"]
    )


def main() -> None:
    args = parse_args()
    specs: list[RunSpec] = args.run
    labels = [spec.label for spec in specs]
    if len(labels) != len(set(labels)):
        raise ValueError("Each --run label must be unique")
    if args.primary_label not in labels:
        raise ValueError("--primary-label must identify one of the --run values")
    max_rss_bytes = int(args.max_rss_gib * 2**30)
    if max_rss_bytes <= 0:
        raise ValueError("--max-rss-gib must be positive")

    # Validate every run before creating any output directory or reading metrics.
    validated = [validate_authoritative_run(spec) for spec in specs]
    protocol_by_label = {
        run.spec.label: run.config["protocol"] for run in validated
    }
    role_by_label = {
        run.spec.label: run.validation["study_role"] for run in validated
    }
    protocols = list(protocol_by_label.values())
    if len(protocols) != len(set(protocols)):
        raise AuthorityError("Duplicate protocols under different run labels are forbidden")
    protocol_set = set(protocols)
    allowed_sets = [
        {"First7Last2"},
        {"First7Last2", "First6Last3", "First5Last4"},
    ]
    if protocol_set not in allowed_sets:
        raise AuthorityError(
            "Run collection must contain either the primary only or the exact primary "
            "plus both prespecified temporal sensitivities"
        )
    if (
        protocol_by_label[args.primary_label] != "First7Last2"
        or role_by_label[args.primary_label] != "primary"
    ):
        raise AuthorityError(
            "The declared primary label must be the authoritative primary First7Last2 run"
        )
    for run in validated:
        expected_role = (
            "primary" if run.config["protocol"] == "First7Last2"
            else "temporal_sensitivity"
        )
        require_equal(
            run.validation["study_role"], expected_role,
            f"combined role for {run.spec.label}",
        )
    output_root = args.output_dir.expanduser().resolve()
    if any(path_is_within(output_root, run.spec.path) for run in validated):
        raise ValueError("Analysis output cannot be nested inside a source run")
    for label in labels:
        destination = output_root / label
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing analysis output: {destination}"
            )
    for filename in (
        "protocol_sensitivity_comparison.csv", "combined_analysis_integrity.json"
    ):
        if (output_root / filename).exists():
            raise FileExistsError(
                f"Refusing to overwrite existing combined output: {output_root / filename}"
            )
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for run in validated:
        print(f"Analyzing authoritative run {run.spec.label}: {run.spec.path}", flush=True)
        summaries.append(analyze_run(
            run, output_root, args.bootstrap_reps, args.bootstrap_seed,
            max_rss_bytes,
        ))

    combined = combined_protocol_table(summaries)
    combined.to_csv(output_root / "protocol_sensitivity_comparison.csv", index=False)
    combined_integrity = {
        "status": "passed",
        "primary_label": args.primary_label,
        "runs": [
            {
                "label": summary["run_label"],
                "protocol": summary["protocol"],
                "output_dir": summary["output_dir"],
                "integrity_path": summary["integrity_path"],
                "authority_validation": summary["authority_validation"],
            }
            for summary in summaries
        ],
        "protocol_comparison_sha256": sha256(
            output_root / "protocol_sensitivity_comparison.csv"
        ),
        "observed_peak_rss_bytes": int(_PEAK_RSS_BYTES),
        "note": (
            "Every included run independently passed the explicit production-authority "
            "and checkpoint-hash gate. No smoke run was read."
        ),
    }
    write_json(output_root / "combined_analysis_integrity.json", combined_integrity)
    print(json.dumps({
        "status": "passed",
        "primary_label": args.primary_label,
        "runs": labels,
        "output_root": str(output_root),
        "peak_rss_gib": _PEAK_RSS_BYTES / 2**30,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()







