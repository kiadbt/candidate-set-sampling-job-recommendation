#!/usr/bin/env python3
"""Verify cross-window directional consistency from the locked CareerBuilder run.

This analysis is deliberately descriptive. It treats each window-by-seed result as
an audit cell and does not treat the 35 cells as independent inferential units.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
PROJECT = HERE.parents[2]
SOURCE = (
    PROJECT
    / "Results"
    / "Reproducibility_Lock_2026-11-02"
    / "CareerBuilder_CurrentScript_Rerun_v1"
    / "First7Last2"
    / "per_window_results.csv"
)

OUTPUT_DIR = (
    REPOSITORY / "aggregate_results" / "revision_r1"
    / "careerbuilder_cross_window_direction"
)
DETAIL = OUTPUT_DIR / "careerbuilder_size100_hitrate_directional_consistency_detail.csv"
SUMMARY = OUTPUT_DIR / "careerbuilder_size100_hitrate_directional_consistency_summary.csv"
AUDIT = OUTPUT_DIR / "careerbuilder_size100_hitrate_directional_consistency_audit.json"

MODELS = {
    "bm25_full_static": "Full-text BM25",
    "tfidf_combined_static": "Combined-profile/full-text TF-IDF",
    "minilm_title_static": "Combined-profile/title MiniLM",
}
STRATEGIES = ("uniform", "popularity_weighted")
WINDOWS = tuple(range(1, 8))
SEEDS = tuple(range(1, 6))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def dec(value: str) -> Decimal:
    return Decimal(value)


def fmt(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000000000001")), "f")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    full: dict[tuple[int, str], Decimal] = {}
    sampled: dict[tuple[int, str, int, str], Decimal] = {}
    duplicate_full: list[tuple[int, str]] = []
    duplicate_sampled: list[tuple[int, str, int, str]] = []

    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["protocol"] != "First7Last2":
                continue
            if row["estimand"] != "design_weighted_population":
                continue
            if row["model"] not in MODELS or row["k"] != "20":
                continue
            window = int(row["window"])
            model = row["model"]
            if row["candidate_regime"] == "full_catalog":
                key = (window, model)
                if key in full:
                    duplicate_full.append(key)
                full[key] = dec(row["hit_rate"])
                continue
            if row["candidate_regime"] != "sampled":
                continue
            if row["sampling_strategy"] not in STRATEGIES:
                continue
            if row["requested_candidate_size"] not in {"100", "100.0"}:
                continue
            seed = int(row["seed"])
            key = (window, row["sampling_strategy"], seed, model)
            if key in sampled:
                duplicate_sampled.append(key)
            sampled[key] = dec(row["hit_rate"])

    expected_full = {(w, m) for w in WINDOWS for m in MODELS}
    expected_sampled = {
        (w, strategy, seed, model)
        for w in WINDOWS
        for strategy in STRATEGIES
        for seed in SEEDS
        for model in MODELS
    }
    missing_full = sorted(expected_full - set(full))
    extra_full = sorted(set(full) - expected_full)
    missing_sampled = sorted(expected_sampled - set(sampled))
    extra_sampled = sorted(set(sampled) - expected_sampled)

    if duplicate_full or duplicate_sampled or missing_full or extra_full or missing_sampled or extra_sampled:
        raise RuntimeError(
            "Input-grid integrity failure: "
            f"duplicate_full={duplicate_full}, duplicate_sampled={duplicate_sampled}, "
            f"missing_full={missing_full}, extra_full={extra_full}, "
            f"missing_sampled={missing_sampled}, extra_sampled={extra_sampled}"
        )

    detail_rows: list[dict[str, str | int | bool]] = []
    grouped: dict[tuple[str, str], list[Decimal]] = defaultdict(list)
    grouped_window: dict[tuple[str, str, int], list[Decimal]] = defaultdict(list)

    for window in WINDOWS:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                for model, model_label in MODELS.items():
                    exact = full[(window, model)]
                    observed = sampled[(window, strategy, seed, model)]
                    inflation = observed - exact
                    detail_rows.append(
                        {
                            "window": window,
                            "sampling_strategy": strategy,
                            "seed": seed,
                            "requested_candidate_size": 100,
                            "estimand": "design_weighted_population",
                            "metric": "HitRate@20",
                            "model": model,
                            "model_label": model_label,
                            "sampled_hitrate20": fmt(observed),
                            "exact_hitrate20": fmt(exact),
                            "inflation_sampled_minus_exact": fmt(inflation),
                            "strictly_positive": inflation > 0,
                        }
                    )
                    grouped[(strategy, model)].append(inflation)
                    grouped_window[(strategy, model, window)].append(inflation)

    detail_fields = list(detail_rows[0])
    with DETAIL.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_rows: list[dict[str, str | int | bool]] = []
    for strategy in STRATEGIES:
        for model, model_label in MODELS.items():
            values = grouped[(strategy, model)]
            window_means = []
            positive_window_means = 0
            for window in WINDOWS:
                window_values = grouped_window[(strategy, model, window)]
                window_mean = sum(window_values) / Decimal(len(window_values))
                window_means.append(window_mean)
                positive_window_means += int(window_mean > 0)
            summary_rows.append(
                {
                    "sampling_strategy": strategy,
                    "requested_candidate_size": 100,
                    "estimand": "design_weighted_population",
                    "metric": "HitRate@20",
                    "model": model,
                    "model_label": model_label,
                    "window_seed_cells": len(values),
                    "strictly_positive_cells": sum(value > 0 for value in values),
                    "all_window_seed_cells_positive": all(value > 0 for value in values),
                    "minimum_inflation": fmt(min(values)),
                    "maximum_inflation": fmt(max(values)),
                    "mean_inflation": fmt(sum(values) / Decimal(len(values))),
                    "window_means": len(window_means),
                    "strictly_positive_window_means": positive_window_means,
                    "all_window_means_positive": positive_window_means == len(WINDOWS),
                    "minimum_window_mean_inflation": fmt(min(window_means)),
                    "maximum_window_mean_inflation": fmt(max(window_means)),
                }
            )

    summary_fields = list(summary_rows[0])
    with SUMMARY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    all_cells_positive = all(row["all_window_seed_cells_positive"] for row in summary_rows)
    all_window_means_positive = all(row["all_window_means_positive"] for row in summary_rows)
    checks = {
        "source_exists": SOURCE.exists(),
        "expected_full_reference_cells": len(expected_full),
        "observed_full_reference_cells": len(full),
        "expected_sampled_cells": len(expected_sampled),
        "observed_sampled_cells": len(sampled),
        "detail_rows": len(detail_rows),
        "summary_rows": len(summary_rows),
        "duplicate_full_reference_cells": len(duplicate_full),
        "duplicate_sampled_cells": len(duplicate_sampled),
        "missing_full_reference_cells": len(missing_full),
        "missing_sampled_cells": len(missing_sampled),
        "all_210_window_seed_cells_strictly_positive": all_cells_positive,
        "all_42_strategy_model_window_means_strictly_positive": all_window_means_positive,
    }
    passed = (
        len(full) == 21
        and len(sampled) == 210
        and len(detail_rows) == 210
        and len(summary_rows) == 6
        and not duplicate_full
        and not duplicate_sampled
        and not missing_full
        and not missing_sampled
        and all_cells_positive
        and all_window_means_positive
    )
    audit = {
        "analysis": "CareerBuilder size-100 HitRate@20 cross-window directional consistency",
        "scope": {
            "protocol": "First7Last2",
            "estimand": "design_weighted_population",
            "metric": "HitRate@20",
            "requested_candidate_size": 100,
            "windows": list(WINDOWS),
            "seeds": list(SEEDS),
            "strategies": list(STRATEGIES),
            "models": MODELS,
        },
        "interpretation_boundary": (
            "Descriptive robustness check only. Window-by-seed cells are not treated as "
            "independent inferential units."
        ),
        "source": SOURCE.relative_to(PROJECT).as_posix(),
        "source_sha256": sha256(SOURCE),
        "outputs": {
            DETAIL.name: sha256(DETAIL),
            SUMMARY.name: sha256(SUMMARY),
        },
        "checks": checks,
        "pass": passed,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("FAIL: directional-consistency audit did not pass")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
