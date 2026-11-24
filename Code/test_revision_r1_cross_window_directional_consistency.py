#!/usr/bin/env python3
"""Independent row-level QA for the cross-window consistency outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = (
    HERE.parent / "aggregate_results" / "revision_r1"
    / "careerbuilder_cross_window_direction"
)
DETAIL = OUTPUT_DIR / "careerbuilder_size100_hitrate_directional_consistency_detail.csv"
SUMMARY = OUTPUT_DIR / "careerbuilder_size100_hitrate_directional_consistency_summary.csv"
AUDIT = OUTPUT_DIR / "careerbuilder_size100_hitrate_directional_consistency_audit.json"
OUTPUT = OUTPUT_DIR / "INDEPENDENT_QA.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    detail = load_csv(DETAIL)
    summary = load_csv(SUMMARY)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    keys = [
        (row["window"], row["sampling_strategy"], row["seed"], row["model"])
        for row in detail
    ]
    arithmetic_failures = []
    positive_failures = []
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in detail:
        sampled = Decimal(row["sampled_hitrate20"])
        exact = Decimal(row["exact_hitrate20"])
        inflation = Decimal(row["inflation_sampled_minus_exact"])
        if sampled - exact != inflation:
            arithmetic_failures.append(row)
        if inflation <= 0 or row["strictly_positive"].lower() != "true":
            positive_failures.append(row)
        groups[(row["sampling_strategy"], row["model"])].append(row)

    summary_failures = []
    for row in summary:
        group = groups[(row["sampling_strategy"], row["model"])]
        values = [Decimal(item["inflation_sampled_minus_exact"]) for item in group]
        by_window: dict[str, list[Decimal]] = defaultdict(list)
        for item in group:
            by_window[item["window"]].append(
                Decimal(item["inflation_sampled_minus_exact"])
            )
        means = [sum(vals) / Decimal(len(vals)) for vals in by_window.values()]
        checks = {
            "window_seed_cells": int(row["window_seed_cells"]) == len(values) == 35,
            "strictly_positive_cells": int(row["strictly_positive_cells"])
            == sum(value > 0 for value in values)
            == 35,
            "minimum_inflation": Decimal(row["minimum_inflation"]) == min(values),
            "maximum_inflation": Decimal(row["maximum_inflation"]) == max(values),
            "mean_inflation": Decimal(row["mean_inflation"])
            == (sum(values) / Decimal(len(values))).quantize(Decimal("0.000000000001")),
            "window_means": int(row["window_means"]) == len(means) == 7,
            "strictly_positive_window_means": int(row["strictly_positive_window_means"])
            == sum(value > 0 for value in means)
            == 7,
            "minimum_window_mean": Decimal(row["minimum_window_mean_inflation"])
            == min(means).quantize(Decimal("0.000000000001")),
            "maximum_window_mean": Decimal(row["maximum_window_mean_inflation"])
            == max(means).quantize(Decimal("0.000000000001")),
        }
        if not all(checks.values()):
            summary_failures.append(
                {
                    "sampling_strategy": row["sampling_strategy"],
                    "model": row["model"],
                    "checks": checks,
                }
            )

    output_hashes_match_audit = {
        DETAIL.name: sha256(DETAIL) == audit["outputs"][DETAIL.name],
        SUMMARY.name: sha256(SUMMARY) == audit["outputs"][SUMMARY.name],
    }
    checks = {
        "detail_rows_equal_210": len(detail) == 210,
        "summary_rows_equal_6": len(summary) == 6,
        "detail_keys_unique": len(keys) == len(set(keys)) == 210,
        "six_strategy_model_groups": len(groups) == 6,
        "each_group_has_35_cells": all(len(rows) == 35 for rows in groups.values()),
        "zero_arithmetic_failures": not arithmetic_failures,
        "zero_positive_direction_failures": not positive_failures,
        "zero_summary_recalculation_failures": not summary_failures,
        "primary_audit_passed": audit.get("pass") is True,
        "output_hashes_match_primary_audit": all(output_hashes_match_audit.values()),
    }
    passed = all(checks.values())
    report = {
        "analysis": "Independent QA of CareerBuilder cross-window directional consistency",
        "checks": checks,
        "output_hashes_match_primary_audit": output_hashes_match_audit,
        "arithmetic_failures": len(arithmetic_failures),
        "positive_direction_failures": len(positive_failures),
        "summary_recalculation_failures": summary_failures,
        "pass": passed,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("FAIL: independent QA did not pass")


if __name__ == "__main__":
    main()
