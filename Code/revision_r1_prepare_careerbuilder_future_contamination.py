#!/usr/bin/env python3
"""Prepare a future-opening CareerBuilder catalog-contamination sensitivity.

For each provider window, the reference catalog is the frozen set of jobs
active at the First7Last2 cutoff.  This preparer extracts same-window jobs with
a valid date interval whose ``StartDate`` is later than that cutoff.  Those jobs
are the only alternatives added by the subsequent sensitivity evaluator.

The procedure does not change queries, targets, profiles, relevance labels, or
the reference catalog.  It deliberately excludes already-expired jobs so that
the perturbation isolates future-information leakage rather than mixing two
different eligibility violations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from multiwindow_cbf_evaluation import clean_text
from preprocess_raw_careerbuilder import normalize_text


WINDOWS = tuple(range(1, 8))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=Path,
        default=root / "Dataset" / "source_tsv" / "jobs.tsv",
    )
    parser.add_argument(
        "--reference-root", type=Path,
        default=root / "Dataset" / "CareerBuilder_InternalTemporal_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "Dataset" / "CareerBuilder_FutureContamination_v1",
    )
    parser.add_argument("--protocol", default="First7Last2")
    parser.add_argument("--chunksize", type=int, default=20_000)
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def reference_catalog_path(reference_root: Path, protocol: str, window: int) -> Path:
    directory = reference_root / protocol / f"window_{window}"
    candidates = [
        directory / "jobs_active_at_cutoff.csv",
        directory / "jobs_active_at_cutoff.csv.gz",
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            f"Expected exactly one reference catalog for W{window}; found {present}"
        )
    return present[0]


def main() -> None:
    args = parse_args()
    started = time.time()
    if not args.jobs.is_file():
        raise FileNotFoundError(args.jobs)
    manifest_path = args.reference_root / "INTERNAL_TEMPORAL_PREPARATION_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.protocol not in manifest.get("protocols", {}):
        raise KeyError(f"Protocol absent from reference manifest: {args.protocol}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            "Future-contamination preparation requires a fresh empty output root: "
            f"{args.output.resolve()}"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    protocol_record = manifest["protocols"][args.protocol]["windows"]
    cutoffs = {
        window: pd.Timestamp(protocol_record[str(window)]["cutoff"])
        for window in WINDOWS
    }
    evaluation_ends = {
        window: pd.Timestamp(protocol_record[str(window)]["evaluation_end"])
        for window in WINDOWS
    }
    reference_paths = {
        window: reference_catalog_path(args.reference_root, args.protocol, window)
        for window in WINDOWS
    }
    reference_ids = {
        window: set(map(int, pd.read_csv(path, usecols=["JobID"])["JobID"]))
        for window, path in reference_paths.items()
    }
    for window in WINDOWS:
        expected = int(protocol_record[str(window)]["catalog_jobs"])
        if len(reference_ids[window]) != expected:
            raise AssertionError(
                f"W{window}: reference ID count {len(reference_ids[window])} != {expected}"
            )

    output_paths: dict[int, Path] = {}
    first_write: dict[int, bool] = {}
    for window in WINDOWS:
        directory = args.output / args.protocol / f"window_{window}"
        directory.mkdir(parents=True, exist_ok=True)
        output_paths[window] = directory / "jobs_future_after_cutoff.csv.gz"
        first_write[window] = True

    date_active_ids = {window: set() for window in WINDOWS}
    future_ids = {window: set() for window in WINDOWS}
    near_future_ids = {window: set() for window in WINDOWS}
    expired_ids = {window: set() for window in WINDOWS}
    valid_dated_ids = {window: set() for window in WINDOWS}
    all_job_ids: set[int] = set()
    duplicate_job_ids: set[int] = set()
    invalid_date_rows = {window: 0 for window in WINDOWS}
    rows_seen = 0

    for chunk_number, chunk in enumerate(pd.read_csv(
        args.jobs, sep="\t", chunksize=args.chunksize, low_memory=False,
        quoting=csv.QUOTE_NONE, on_bad_lines="error",
    ), 1):
        rows_seen += len(chunk)
        chunk["JobID"] = pd.to_numeric(chunk["JobID"], errors="coerce")
        chunk["WindowID"] = pd.to_numeric(chunk["WindowID"], errors="coerce")
        chunk["StartDate"] = pd.to_datetime(chunk["StartDate"], errors="coerce")
        chunk["EndDate"] = pd.to_datetime(chunk["EndDate"], errors="coerce")
        chunk = chunk.dropna(subset=["JobID", "WindowID"]).copy()
        chunk[["JobID", "WindowID"]] = chunk[["JobID", "WindowID"]].astype(np.int64)
        chunk_ids = set(map(int, chunk["JobID"]))
        duplicate_job_ids.update(all_job_ids.intersection(chunk_ids))
        duplicate_job_ids.update(map(
            int, chunk.loc[chunk["JobID"].duplicated(keep=False), "JobID"].unique()
        ))
        all_job_ids.update(chunk_ids)

        valid_dates = (
            chunk["StartDate"].notna() & chunk["EndDate"].notna()
            & chunk["StartDate"].le(chunk["EndDate"])
        )
        for window in WINDOWS:
            window_mask = chunk["WindowID"].eq(window)
            invalid_date_rows[window] += int((window_mask & ~valid_dates).sum())
            valid_window = window_mask & valid_dates
            valid_rows = chunk.loc[valid_window]
            valid_dated_ids[window].update(map(int, valid_rows["JobID"]))

            active_mask = (
                valid_window & chunk["StartDate"].le(cutoffs[window])
                & chunk["EndDate"].ge(cutoffs[window])
            )
            date_active_ids[window].update(map(int, chunk.loc[active_mask, "JobID"]))
            future_mask = valid_window & chunk["StartDate"].gt(cutoffs[window])
            future = chunk.loc[future_mask].copy().drop_duplicates("JobID")
            if not future.empty:
                future_ids[window].update(map(int, future["JobID"]))
                near_future_ids[window].update(map(
                    int, future.loc[
                        future["StartDate"].le(evaluation_ends[window]), "JobID"
                    ]
                ))
                for column in (
                    "Title", "Description", "Requirements", "City", "State", "Country",
                ):
                    future[column] = future[column].map(normalize_text)
                future["text_all"] = clean_text(
                    "title " + future["Title"]
                    + " requirements " + future["Requirements"]
                    + " description " + future["Description"]
                    + " city " + future["City"]
                    + " state " + future["State"]
                    + " country " + future["Country"]
                )
                future.to_csv(
                    output_paths[window], mode="w" if first_write[window] else "a",
                    header=first_write[window], index=False, compression="gzip",
                )
                first_write[window] = False
            expired_mask = valid_window & chunk["EndDate"].lt(cutoffs[window])
            expired_ids[window].update(map(int, chunk.loc[expired_mask, "JobID"]))
        if chunk_number % 10 == 0:
            print(f"Processed {rows_seen:,} raw job rows", flush=True)

    if duplicate_job_ids:
        raise AssertionError(
            f"Raw jobs source contains duplicate JobID values: {sorted(duplicate_job_ids)[:10]}"
        )

    summary_rows: list[dict[str, object]] = []
    output_records: dict[str, object] = {}
    for window in WINDOWS:
        if date_active_ids[window] != reference_ids[window]:
            missing = sorted(reference_ids[window] - date_active_ids[window])[:10]
            unexpected = sorted(date_active_ids[window] - reference_ids[window])[:10]
            raise AssertionError(
                f"W{window}: date-derived active catalog differs from frozen reference; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if future_ids[window].intersection(reference_ids[window]):
            raise AssertionError(f"W{window}: future and reference catalogs overlap")
        if not output_paths[window].is_file():
            raise AssertionError(f"W{window}: no future catalog was written")
        written = pd.read_csv(output_paths[window], usecols=["JobID"])["JobID"].astype(np.int64)
        if written.duplicated().any() or set(map(int, written)) != future_ids[window]:
            raise AssertionError(f"W{window}: written future catalog failed ID validation")
        target_path = (
            args.reference_root / args.protocol / f"window_{window}" / "valid_future_targets.csv"
        )
        target_ids = set(map(int, pd.read_csv(target_path, usecols=["JobID"])["JobID"]))
        if not target_ids.issubset(reference_ids[window]):
            raise AssertionError(f"W{window}: frozen targets are not a subset of the reference catalog")
        if target_ids.intersection(future_ids[window]):
            raise AssertionError(f"W{window}: a frozen target appears in the future-only additions")

        summary_rows.append({
            "window": window,
            "cutoff": cutoffs[window].isoformat(),
            "evaluation_end": evaluation_ends[window].isoformat(),
            "reference_active_jobs": len(reference_ids[window]),
            "future_opening_jobs_added": len(future_ids[window]),
            "near_horizon_future_jobs": len(near_future_ids[window]),
            "expired_jobs_not_added": len(expired_ids[window]),
            "valid_dated_same_window_jobs": len(valid_dated_ids[window]),
            "contaminated_catalog_jobs": len(reference_ids[window]) + len(future_ids[window]),
            "relative_catalog_increase": len(future_ids[window]) / len(reference_ids[window]),
            "invalid_date_rows": invalid_date_rows[window],
            "target_jobs": len(target_ids),
            "reference_equals_date_derived": True,
            "reference_future_overlap": 0,
        })
        output_records[str(window)] = {
            "path": str(output_paths[window].resolve()),
            "bytes": output_paths[window].stat().st_size,
            "sha256": sha256(output_paths[window]),
            "rows": len(written),
        }

    summary = pd.DataFrame(summary_rows)
    summary_path = args.output / "careerbuilder_future_catalog_summary.csv"
    summary.to_csv(summary_path, index=False)
    audit = {
        "analysis_role": "post_decision_sensitivity_preparation",
        "protocol": args.protocol,
        "definition": (
            "Reference jobs active at the cutoff plus same-window, valid-dated jobs "
            "whose StartDate is later than the cutoff. Expired jobs are not added."
        ),
        "invariants_for_subsequent_evaluation": {
            "queries_fixed": True,
            "targets_fixed": True,
            "profiles_fixed": True,
            "reference_catalog_fixed": True,
            "only_future_opening_alternatives_added": True,
        },
        "source_jobs": {
            "path": str(args.jobs.resolve()),
            "bytes": args.jobs.stat().st_size,
            "sha256": sha256(args.jobs),
            "rows": rows_seen,
            "unique_job_ids": len(all_job_ids),
        },
        "reference_manifest": {
            "path": str(manifest_path.resolve()),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "reference_catalogs": {
            str(window): {
                "path": str(reference_paths[window].resolve()),
                "bytes": reference_paths[window].stat().st_size,
                "sha256": sha256(reference_paths[window]),
                "rows": len(reference_ids[window]),
            }
            for window in WINDOWS
        },
        "future_catalogs": output_records,
        "summary": {
            "path": str(summary_path.resolve()),
            "bytes": summary_path.stat().st_size,
            "sha256": sha256(summary_path),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    audit_path = args.output / "careerbuilder_future_catalog_preparation_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {audit_path.resolve()}")


if __name__ == "__main__":
    main()
