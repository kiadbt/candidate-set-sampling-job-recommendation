#!/usr/bin/env python3
"""Audit cross-window identifier overlap for the CareerBuilder temporal task.

This sensitivity checks whether evaluation users, eligible
population users, cutoff-active jobs, or target jobs occur in more than one
provider window and whether the provider-defined calendar periods overlap. It
reads only identifier columns plus the provider's window-date table and writes
compact CSV/JSON evidence; it does not score recommendations or modify prepared
inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd


WINDOWS = tuple(range(1, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="CareerBuilder_InternalTemporal_v1 directory",
    )
    parser.add_argument("--protocol", default="First7Last2")
    parser.add_argument(
        "--primary-results-dir",
        type=Path,
        help=(
            "Optional primary evaluation directory containing "
            "checkpoints/window_*/users_*.csv.gz; when supplied, the exact "
            "audit-user overlap is reported."
        ),
    )
    parser.add_argument(
        "--window-dates",
        type=Path,
        required=True,
        help="Provider window_dates.tsv file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_csv(directory: Path, stem: str) -> Path:
    candidates = (directory / f"{stem}.csv", directory / f"{stem}.csv.gz")
    present = [path for path in candidates if path.exists()]
    if len(present) != 1:
        raise FileNotFoundError(
            f"Expected exactly one plain or gzip file for {stem} in {directory}; "
            f"found {present}"
        )
    return present[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path, column: str) -> set[int]:
    frame = pd.read_csv(path, usecols=[column])
    if frame[column].isna().any():
        raise AssertionError(f"Missing {column} in {path}")
    return set(frame[column].astype("int64").tolist())


def repeated_id_summary(window_sets: dict[int, set[int]]) -> dict[str, float | int]:
    membership = Counter(identifier for values in window_sets.values() for identifier in values)
    repeated = {identifier: count for identifier, count in membership.items() if count >= 2}
    return {
        "unique_ids": len(membership),
        "ids_in_two_or_more_windows": len(repeated),
        "share_unique_ids_in_two_or_more_windows": (
            len(repeated) / len(membership) if membership else 0.0
        ),
        "maximum_windows_per_id": max(membership.values(), default=0),
    }


def audit_periods(path: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frame = pd.read_csv(path, sep="\t")
    required = {"Window", "Train Start", "Train End / Test Start", "Test End"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"Missing window-date columns: {missing}")
    if frame["Window"].duplicated().any():
        raise AssertionError("Duplicate Window identifiers in window-date table")
    frame = frame.sort_values("Window").reset_index(drop=True)
    if tuple(frame["Window"].astype(int)) != WINDOWS:
        raise AssertionError(f"Expected windows {WINDOWS}; found {tuple(frame['Window'])}")
    for column in ("Train Start", "Train End / Test Start", "Test End"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if not (frame["Train Start"] < frame["Train End / Test Start"]).all():
        raise AssertionError("Every train start must precede train end/test start")
    if not (frame["Train End / Test Start"] < frame["Test End"]).all():
        raise AssertionError("Every train end/test start must precede test end")

    pairwise: list[dict[str, object]] = []
    for left, right in combinations(frame.to_dict("records"), 2):
        overlap_start = max(left["Train Start"], right["Train Start"])
        overlap_end = min(left["Test End"], right["Test End"])
        overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
        pairwise.append(
            {
                "window_left": int(left["Window"]),
                "window_right": int(right["Window"]),
                "left_start": left["Train Start"].isoformat(),
                "left_end_exclusive": left["Test End"].isoformat(),
                "right_start": right["Train Start"].isoformat(),
                "right_end_exclusive": right["Test End"].isoformat(),
                "overlap_seconds_half_open": overlap_seconds,
                "periods_overlap_half_open": bool(overlap_seconds > 0),
                "periods_touch_at_boundary": bool(
                    left["Test End"] == right["Train Start"]
                    or right["Test End"] == left["Train Start"]
                ),
            }
        )
    return frame, pairwise


def main() -> None:
    args = parse_args()
    protocol_root = args.dataset_dir / args.protocol
    args.output_dir.mkdir(parents=True, exist_ok=True)
    period_frame, period_rows = audit_periods(args.window_dates)

    set_specs = {
        "released_query_profiles": ("query_profiles", "UserID"),
        "eligible_population_users": ("cohort_population", "UserID"),
        "cutoff_active_jobs": ("jobs_active_at_cutoff", "JobID"),
        "target_users": ("valid_future_targets", "UserID"),
        "target_jobs": ("valid_future_targets", "JobID"),
    }
    sets: dict[str, dict[int, set[int]]] = {name: {} for name in set_specs}
    input_records: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    for window in WINDOWS:
        directory = protocol_root / f"window_{window}"
        if not directory.exists():
            raise FileNotFoundError(directory)
        row: dict[str, object] = {"window": window}
        loaded: dict[tuple[str, str], set[int]] = {}
        for set_name, (stem, column) in set_specs.items():
            key = (stem, column)
            path = resolve_csv(directory, stem)
            if key not in loaded:
                loaded[key] = read_ids(path, column)
                input_records.append(
                    {
                        "window": window,
                        "file": str(path.resolve()),
                        "column": column,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
            sets[set_name][window] = loaded[key]
            row[f"{set_name}_n"] = len(loaded[key])
        window_rows.append(row)

    if args.primary_results_dir is not None:
        checkpoint_root = args.primary_results_dir / "checkpoints"
        sets["primary_audit_users"] = {}
        for window in WINDOWS:
            files = sorted((checkpoint_root / f"window_{window}").glob("users_*.csv.gz"))
            if not files:
                raise FileNotFoundError(
                    f"No primary checkpoint files for window {window} under {checkpoint_root}"
                )
            identifiers: set[int] = set()
            for path in files:
                identifiers.update(read_ids(path, "user_id"))
                input_records.append(
                    {
                        "window": window,
                        "file": str(path.resolve()),
                        "column": "user_id",
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
            sets["primary_audit_users"][window] = identifiers
            window_rows[window - 1]["primary_audit_users_n"] = len(identifiers)

    pairwise_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for set_name, window_sets in sets.items():
        summary = repeated_id_summary(window_sets)
        summary_rows.append({"identifier_set": set_name, **summary})
        for left, right in combinations(WINDOWS, 2):
            intersection = window_sets[left] & window_sets[right]
            union = window_sets[left] | window_sets[right]
            pairwise_rows.append(
                {
                    "identifier_set": set_name,
                    "window_left": left,
                    "window_right": right,
                    "left_n": len(window_sets[left]),
                    "right_n": len(window_sets[right]),
                    "intersection_n": len(intersection),
                    "jaccard": len(intersection) / len(union) if union else 0.0,
                }
            )

    window_frame = pd.DataFrame(window_rows)
    summary_frame = pd.DataFrame(summary_rows)
    pairwise_frame = pd.DataFrame(pairwise_rows)
    period_pairwise_frame = pd.DataFrame(period_rows)
    window_frame.to_csv(args.output_dir / "careerbuilder_window_identifier_counts.csv", index=False)
    summary_frame.to_csv(args.output_dir / "careerbuilder_cross_window_overlap_summary.csv", index=False)
    pairwise_frame.to_csv(args.output_dir / "careerbuilder_cross_window_pairwise_overlap.csv", index=False)
    period_pairwise_frame.to_csv(
        args.output_dir / "careerbuilder_window_period_overlap.csv", index=False
    )

    period_records = []
    for row in period_frame.to_dict("records"):
        period_records.append(
            {
                "window": int(row["Window"]),
                "train_start": row["Train Start"].isoformat(),
                "train_end_test_start": row["Train End / Test Start"].isoformat(),
                "test_end_exclusive": row["Test End"].isoformat(),
            }
        )

    integrity = {
        "protocol": args.protocol,
        "windows": list(WINDOWS),
        "all_identifier_sets_disjoint": bool(
            (summary_frame["ids_in_two_or_more_windows"] == 0).all()
        ),
        "all_calendar_periods_nonoverlapping_half_open": bool(
            not period_pairwise_frame["periods_overlap_half_open"].any()
        ),
        "calendar_periods_are_contiguous_in_provider_order": bool(
            all(
                period_frame.loc[index, "Test End"]
                == period_frame.loc[index + 1, "Train Start"]
                for index in range(len(period_frame) - 1)
            )
        ),
        "calendar_periods": period_records,
        "window_dates_input": {
            "file": str(args.window_dates.resolve()),
            "bytes": args.window_dates.stat().st_size,
            "sha256": sha256(args.window_dates),
        },
        "summary": summary_rows,
        "inputs": input_records,
        "analysis": (
            "identifier and half-open provider-period overlap audit; "
            "no recommendation scoring"
        ),
    }
    (args.output_dir / "careerbuilder_cross_window_overlap_audit.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
