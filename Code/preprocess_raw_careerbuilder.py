"""Build provenance-safe, window-specific analysis files from CareerBuilder TSVs.

The original ZIP and extracted TSV files are read-only inputs.  Outputs are
written under Dataset/processed_raw_v1.  Vacancy eligibility follows the
competition documentation: a job must be assigned to the window and overlap
at least part of that window's four-day test period.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd


CONTROL_RE = re.compile(r"\\[tnr]")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest().upper()


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = CONTROL_RE.sub(" ", text)
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = text.replace("\\\\", " ")
    return SPACE_RE.sub(" ", text).strip()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=root / "Dataset" / "source_tsv")
    parser.add_argument("--output", type=Path, default=root / "Dataset" / "processed_raw_v1")
    parser.add_argument("--zip", type=Path, default=root / "Dataset" / "raw" / "job-recommendation.zip")
    parser.add_argument("--chunksize", type=int, default=20_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dates = pd.read_csv(args.source / "window_dates.tsv", sep="\t")
    dates.columns = ["WindowID", "TrainStart", "TestStart", "TestEnd"]
    for column in ("TrainStart", "TestStart", "TestEnd"):
        dates[column] = pd.to_datetime(dates[column], errors="raise")
    write_csv(dates, args.output / "window_dates.csv")

    users = pd.read_csv(args.source / "users.tsv", sep="\t", low_memory=False)
    history = pd.read_csv(args.source / "user_history.tsv", sep="\t", low_memory=False)
    apps = pd.read_csv(args.source / "apps.tsv", sep="\t", low_memory=False)
    apps["ApplicationDate"] = pd.to_datetime(apps["ApplicationDate"], errors="coerce")
    history["JobTitle"] = history["JobTitle"].map(normalize_text)

    audit = {
        "source_zip": str(args.zip),
        "source_zip_sha256": sha256(args.zip),
        "source_files": {},
        "window_rule": "WindowID match and StartDate < TestEnd and EndDate > TestStart",
        "history_rule": "smaller Sequence indicates more recent employment; Sequence=1 is latest when present",
        "windows": {},
    }
    for name in ("window_dates.tsv", "users.tsv", "user_history.tsv", "apps.tsv", "jobs.tsv"):
        path = args.source / name
        audit["source_files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    for window_id in dates["WindowID"].astype(int):
        out = args.output / f"window_{window_id}"
        out.mkdir(exist_ok=True)
        window_users = users[users["WindowID"] == window_id].copy()
        window_history = history[history["WindowID"] == window_id].copy()
        window_apps = apps[apps["WindowID"] == window_id].copy()
        write_csv(window_users, out / "users.csv")
        write_csv(window_history, out / "user_history.csv")
        write_csv(window_apps, out / "apps.csv")

    job_counts = {int(w): {"assigned": 0, "test_available": 0, "invalid_dates": 0} for w in dates.WindowID}
    job_outputs = {int(w): args.output / f"window_{int(w)}" / "jobs_test_available.csv" for w in dates.WindowID}
    for path in job_outputs.values():
        if path.exists():
            path.unlink()

    first_write = {int(w): True for w in dates.WindowID}
    date_index = dates.set_index("WindowID")
    for chunk in pd.read_csv(
        args.source / "jobs.tsv", sep="\t", chunksize=args.chunksize,
        low_memory=False, quoting=3, on_bad_lines="error",
    ):
        chunk["StartDate"] = pd.to_datetime(chunk["StartDate"], errors="coerce")
        chunk["EndDate"] = pd.to_datetime(chunk["EndDate"], errors="coerce")
        for column in ("Title", "Description", "Requirements", "City", "State", "Country"):
            chunk[column] = chunk[column].map(normalize_text)
        chunk = chunk.drop_duplicates("JobID")
        for window_id, group in chunk.groupby("WindowID", sort=False):
            window_id = int(window_id)
            if window_id not in job_counts:
                continue
            job_counts[window_id]["assigned"] += len(group)
            invalid = group["StartDate"].isna() | group["EndDate"].isna()
            job_counts[window_id]["invalid_dates"] += int(invalid.sum())
            spec = date_index.loc[window_id]
            eligible = group[
                (~invalid)
                & (group["StartDate"] < spec["TestEnd"])
                & (group["EndDate"] > spec["TestStart"])
            ].copy()
            job_counts[window_id]["test_available"] += len(eligible)
            eligible.to_csv(
                job_outputs[window_id], mode="w" if first_write[window_id] else "a",
                header=first_write[window_id], index=False, encoding="utf-8",
            )
            first_write[window_id] = False

    for window_id in dates["WindowID"].astype(int):
        out = args.output / f"window_{window_id}"
        wu = pd.read_csv(out / "users.csv", low_memory=False)
        wh = pd.read_csv(out / "user_history.csv", low_memory=False)
        wa = pd.read_csv(out / "apps.csv", low_memory=False)
        wj = pd.read_csv(out / "jobs_test_available.csv", usecols=["JobID"])
        test_users = wu[wu["Split"].str.casefold() == "test"]
        test_apps = wa[wa["Split"].str.casefold() == "test"]
        valid_job_ids = set(wj["JobID"].astype(int))
        valid_test_apps = test_apps[test_apps["JobID"].isin(valid_job_ids)]
        audit["windows"][str(window_id)] = {
            **job_counts[window_id],
            "users_all": int(len(wu)),
            "test_users_official": int(len(test_users)),
            "history_rows": int(len(wh)),
            "applications_all": int(len(wa)),
            "test_applications_raw": int(len(test_apps)),
            "test_applications_to_available_jobs": int(len(valid_test_apps)),
            "test_users_with_valid_application": int(valid_test_apps["UserID"].nunique()),
            "test_users_with_history": int(test_users["UserID"].isin(wh["UserID"]).sum()),
        }

    (args.output / "preprocessing_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(args.output / "preprocessing_audit.json")


if __name__ == "__main__":
    main()
