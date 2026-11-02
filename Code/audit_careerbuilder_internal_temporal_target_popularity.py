#!/usr/bin/env python3
"""Audit pre-cutoff popularity cells for InternalTemporal future targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path,
                        default=root / "Dataset" / "CareerBuilder_InternalTemporal_v1")
    parser.add_argument("--protocol", default="First7Last2")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "Analysis" / "Journal_Manuscript_Planning")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_frames = []
    audit_user_frames = []
    for window in range(1, 8):
        directory = args.dataset_dir / args.protocol / f"window_{window}"
        targets = pd.read_csv(
            directory / "valid_future_targets.csv", usecols=["UserID", "JobID"]
        ).drop_duplicates(["UserID", "JobID"])
        precut = pd.read_csv(
            directory / "precut_applications.csv", usecols=["JobID"]
        )
        counts = precut["JobID"].value_counts()
        targets["precut_application_popularity"] = (
            targets["JobID"].map(counts).fillna(0).astype(int)
        )
        targets["popularity_group"] = pd.cut(
            targets["precut_application_popularity"],
            bins=[-1, 0, 2, float("inf")],
            labels=["unseen_0", "rare_1_2", "popular_3plus"],
        ).astype(str)
        targets["window"] = window
        pair_frames.append(targets)
        audit_users = pd.read_csv(
            directory / "cohort_audit_sample.csv", usecols=["UserID"]
        )
        audit_users["window"] = window
        audit_user_frames.append(audit_users)
    pairs = pd.concat(pair_frames, ignore_index=True)
    audit_users = pd.concat(audit_user_frames, ignore_index=True)

    rows = []
    for label, frame in [(str(w), pairs[pairs["window"].eq(w)]) for w in range(1, 8)] \
            + [("pooled", pairs)]:
        total = len(frame)
        for group in ("unseen_0", "rare_1_2", "popular_3plus"):
            selected = frame[frame["popularity_group"].eq(group)]
            rows.append({
                "protocol": args.protocol, "window": label,
                "popularity_group": group,
                "target_user_job_pairs": len(selected),
                "fraction_of_target_pairs": len(selected) / total if total else 0.0,
                "unique_target_jobs": int(selected["JobID"].nunique()),
                "users_with_at_least_one_target": int(selected["UserID"].nunique()),
                "all_target_pairs": total,
            })
    summary = pd.DataFrame(rows)
    path = args.output_dir / "careerbuilder_internal_temporal_target_popularity.csv"
    summary.to_csv(path, index=False)
    pairs[[
        "window", "UserID", "JobID", "precut_application_popularity",
        "popularity_group",
    ]].to_csv(
        args.output_dir / "careerbuilder_internal_temporal_target_popularity_pairs.csv.gz",
        index=False, compression="gzip",
    )
    query = pairs.groupby(["window", "UserID"], as_index=False).agg(
        mean_target_precut_popularity=("precut_application_popularity", "mean"),
        relevant_jobs=("JobID", "nunique"),
    )
    query["mean_popularity_group"] = pd.cut(
        query["mean_target_precut_popularity"],
        bins=[-1, 0, 2, float("inf")],
        labels=["mean_unseen_0", "mean_positive_le2", "mean_gt2"],
    ).astype(str)
    audit_keys = set(zip(audit_users["window"].astype(int), audit_users["UserID"].astype(int)))
    query["audit_sample"] = [
        (int(window), int(uid)) in audit_keys
        for window, uid in zip(query["window"], query["UserID"])
    ]
    query_rows = []
    for scope, scoped in (("population", query), ("audit_sample", query[query["audit_sample"]])):
        for label, frame in [(str(w), scoped[scoped["window"].eq(w)]) for w in range(1, 8)] \
                + [("pooled", scoped)]:
            for group in ("mean_unseen_0", "mean_positive_le2", "mean_gt2"):
                selected = frame[frame["mean_popularity_group"].eq(group)]
                query_rows.append({
                    "protocol": args.protocol, "scope": scope, "window": label,
                    "mean_popularity_group": group,
                    "queries": len(selected),
                    "fraction_of_queries": len(selected) / len(frame) if len(frame) else 0.0,
                    "all_queries": len(frame),
                })
    query_summary = pd.DataFrame(query_rows)
    query_summary.to_csv(
        args.output_dir / "careerbuilder_internal_temporal_query_mean_popularity.csv",
        index=False,
    )
    query[[
        "window", "UserID", "mean_target_precut_popularity", "mean_popularity_group",
        "relevant_jobs", "audit_sample",
    ]].to_csv(
        args.output_dir / "careerbuilder_internal_temporal_query_mean_popularity_assignments.csv.gz",
        index=False, compression="gzip",
    )
    pooled = summary[summary["window"].eq("pooled")]
    adequate = bool((pooled["target_user_job_pairs"] >= 500).all())
    query_pooled_audit = query_summary[
        query_summary["scope"].eq("audit_sample")
        & query_summary["window"].eq("pooled")
    ]
    query_adequate = bool((query_pooled_audit["queries"] >= 100).all())
    table_lines = [
        "| Group | Target user-job pairs | Fraction | Unique jobs | Users |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        table_lines.append(
            f"| {row.popularity_group} | {row.target_user_job_pairs:,} | "
            f"{row.fraction_of_target_pairs:.4f} | {row.unique_target_jobs:,} | "
            f"{row.users_with_at_least_one_target:,} |"
        )
    md = [
        "# CareerBuilder InternalTemporal target-popularity audit",
        "",
        f"Protocol: `{args.protocol}`. Popularity is the number of observed applications "
        "to the target job strictly before the window cutoff, using all released users "
        "in that window. Targets are unique user-job pairs after the frozen validity filters.",
        "",
        "Fixed groups: unseen = 0, rare = 1-2, popular = 3 or more.",
        "",
        *table_lines,
        "",
        f"Pooled minimum-cell rule (at least 500 target pairs per group): **{adequate}**.",
        "",
        "The inferential subgroup unit is one group per query, based on that query's "
        "mean target popularity: exactly 0, greater than 0 through 2, or greater than 2.",
        "",
        "| Audit-sample query group | Queries | Fraction |",
        "|---|---:|---:|",
        *[
            f"| {row.mean_popularity_group} | {row.queries:,} | {row.fraction_of_queries:.4f} |"
            for row in query_pooled_audit.itertuples(index=False)
        ],
        "",
        f"Pooled audit-sample query rule (at least 100 queries per group): **{query_adequate}**.",
        "Window-specific counts remain in the CSV and must be inspected before any "
        "within-window subgroup inference.",
    ]
    (args.output_dir / "CAREERBUILDER_INTERNAL_TEMPORAL_TARGET_POPULARITY.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(path)


if __name__ == "__main__":
    main()
