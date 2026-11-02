#!/usr/bin/env python3
"""Prepare defensible CareerBuilder InternalTemporal benchmarks.

This preparation ignores the provider ``Split`` field completely.  It creates
fixed-cutoff internal temporal tasks from observed nine-day applications and
full jobs.tsv metadata.  No recommender is trained or evaluated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from multiwindow_cbf_evaluation import clean_text
from preprocess_raw_careerbuilder import normalize_text


# First7Last2 is the frozen primary protocol.  The other entries are
# parameterized sensitivity protocols and are never run implicitly.
PROTOCOLS = {"First7Last2": 7, "First6Last3": 6, "First5Last4": 5}
MASTER_SAMPLE_SEED = 20261020
AUTHORITATIVE_PREPARATION_DESIGNS = {
    "primary": {
        "protocols": {"First7Last2"},
        "sample_cap_per_window": 1000,
    },
    "temporal_sensitivity": {
        "protocols": {"First6Last3", "First5Last4"},
        "sample_cap_per_window": 250,
    },
}


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def build_profile(users: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Static released content only; provider Split is neither read nor used."""
    history = history.copy()
    history["JobTitle"] = clean_text(history["JobTitle"])
    history = history.sort_values(["UserID", "Sequence"])
    history_all = history.groupby("UserID")["JobTitle"].agg(" ".join)
    users = users.set_index("UserID", drop=False).copy()
    users["history_all"] = users.index.map(history_all).fillna("")
    users["profile_structured"] = clean_text(
        "major " + clean_text(users["Major"])
        + " degree " + clean_text(users["DegreeType"])
        + " city " + clean_text(users["City"])
        + " state " + clean_text(users["State"])
        + " country " + clean_text(users["Country"])
        + " experience " + clean_text(users["TotalYearsExperience"])
        + " employed " + clean_text(users["CurrentlyEmployed"])
    )
    users["profile_combined"] = clean_text(
        users["profile_structured"] + " work history " + users["history_all"]
    )
    # At least one substantive released field. Literal labels alone do not count.
    structured_raw = pd.Series("", index=users.index, dtype=object)
    for column in ("Major", "DegreeType", "City", "State", "Country"):
        structured_raw = structured_raw + " " + clean_text(users[column])
    users["structured_content_present"] = structured_raw.str.strip().ne("")
    users["history_content_present"] = users["history_all"].str.strip().ne("")
    users["released_content_present"] = (
        users["structured_content_present"] | users["history_content_present"]
    )
    return users.reset_index(drop=True)[[
        "UserID", "profile_structured", "history_all", "profile_combined",
        "structured_content_present", "history_content_present",
        "released_content_present",
    ]]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path,
                        default=root / "Dataset" / "processed_raw_v1")
    parser.add_argument("--jobs", type=Path,
                        default=root / "Dataset" / "source_tsv" / "jobs.tsv")
    parser.add_argument("--output", type=Path,
                        default=root / "Dataset" / "CareerBuilder_InternalTemporal_v1")
    parser.add_argument("--sample-cap", type=int, default=1000)
    parser.add_argument("--chunksize", type=int, default=20_000)
    parser.add_argument("--protocols", nargs="+", choices=list(PROTOCOLS),
                        default=["First7Last2"])
    parser.add_argument("--compress-catalogs", action="store_true",
                        help="Write gzip CSV catalogs (recommended for sensitivity roots).")
    parser.add_argument(
        "--study-role", choices=list(AUTHORITATIVE_PREPARATION_DESIGNS),
        default="primary",
        help=(
            "Declare whether this prepared root is for the primary analysis or the "
            "prespecified temporal-boundary sensitivity analyses."
        ),
    )
    parser.add_argument(
        "--authoritative-preparation", action="store_true",
        help=(
            "Fail closed unless the complete frozen preparation design for "
            "--study-role is requested."
        ),
    )
    args = parser.parse_args()
    if args.authoritative_preparation:
        expected = AUTHORITATIVE_PREPARATION_DESIGNS[args.study_role]
        requested_protocols = set(args.protocols)
        duplicate_protocol = len(requested_protocols) != len(args.protocols)
        if (
            requested_protocols != expected["protocols"]
            or duplicate_protocol
            or args.sample_cap != expected["sample_cap_per_window"]
        ):
            raise RuntimeError(
                "--authoritative-preparation requires the complete frozen "
                f"{args.study_role} design; protocols={args.protocols}, "
                f"sample_cap={args.sample_cap}, expected_protocols="
                f"{sorted(expected['protocols'])}, expected_sample_cap="
                f"{expected['sample_cap_per_window']}"
            )
        if args.study_role == "temporal_sensitivity" and not args.compress_catalogs:
            raise RuntimeError(
                "Authoritative temporal-sensitivity preparation requires "
                "--compress-catalogs to keep its separate prepared-data root bounded."
            )
        primary_default = root / "Dataset" / "CareerBuilder_InternalTemporal_v1"
        if (
            args.study_role == "temporal_sensitivity"
            and args.output.resolve() == primary_default.resolve()
        ):
            raise RuntimeError(
                "Authoritative temporal sensitivities must use a prepared-data root "
                "separate from Dataset/CareerBuilder_InternalTemporal_v1."
            )
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(
                "Authoritative preparation refuses a nonempty output root. "
                "Use a fresh versioned directory so stale or partial files cannot "
                f"be mixed into the frozen snapshot: {args.output.resolve()}"
            )
    selected_protocols = {name: PROTOCOLS[name] for name in args.protocols}
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    dates = pd.read_csv(args.processed / "window_dates.csv")
    for column in ("TrainStart", "TestStart", "TestEnd"):
        dates[column] = pd.to_datetime(dates[column])
    date_index = dates.set_index("WindowID")

    apps_by_window: dict[int, pd.DataFrame] = {}
    profiles_by_window: dict[int, pd.DataFrame] = {}
    wanted_jobs: set[int] = set()
    for window in range(1, 8):
        # Deliberately omit Split from usecols.
        apps = pd.read_csv(
            args.processed / f"window_{window}" / "apps.csv",
            usecols=["UserID", "WindowID", "ApplicationDate", "JobID"],
        )
        apps["ApplicationDate"] = pd.to_datetime(apps["ApplicationDate"], errors="coerce")
        apps = apps.dropna(subset=["UserID", "JobID", "ApplicationDate"]).copy()
        apps[["UserID", "JobID"]] = apps[["UserID", "JobID"]].astype(np.int64)
        d = date_index.loc[window]
        apps = apps[
            apps["ApplicationDate"].between(
                d["TrainStart"], d["TestStart"], inclusive="left"
            )
        ].drop_duplicates()
        apps_by_window[window] = apps
        wanted_jobs.update(apps["JobID"].unique().tolist())
        # Deliberately omit Split here as well.
        users = pd.read_csv(
            args.processed / f"window_{window}" / "users.csv",
            usecols=lambda c: c != "Split", low_memory=False,
        )
        history = pd.read_csv(
            args.processed / f"window_{window}" / "user_history.csv",
            usecols=lambda c: c != "Split", low_memory=False,
        )
        profiles_by_window[window] = build_profile(users, history)

    output_files: dict[tuple[str, int], Path] = {}
    first_write: dict[tuple[str, int], bool] = {}
    catalog_ids: dict[tuple[str, int], set[int]] = {
        (protocol, window): set()
        for protocol in selected_protocols for window in range(1, 8)
    }
    target_records: list[pd.DataFrame] = []
    for protocol in selected_protocols:
        for window in range(1, 8):
            directory = args.output / protocol / f"window_{window}"
            directory.mkdir(parents=True, exist_ok=True)
            filename = (
                "jobs_active_at_cutoff.csv.gz" if args.compress_catalogs
                else "jobs_active_at_cutoff.csv"
            )
            path = directory / filename
            if path.exists():
                path.unlink()
            output_files[(protocol, window)] = path
            first_write[(protocol, window)] = True

    print("Streaming full jobs.tsv into InternalTemporal cutoff catalogs", flush=True)
    for chunk_no, chunk in enumerate(pd.read_csv(
        args.jobs, sep="\t", chunksize=args.chunksize, low_memory=False,
        quoting=3, on_bad_lines="error",
    ), 1):
        chunk["JobID"] = pd.to_numeric(chunk["JobID"], errors="coerce")
        chunk["WindowID"] = pd.to_numeric(chunk["WindowID"], errors="coerce")
        chunk["StartDate"] = pd.to_datetime(chunk["StartDate"], errors="coerce")
        chunk["EndDate"] = pd.to_datetime(chunk["EndDate"], errors="coerce")
        chunk = chunk.dropna(subset=["JobID", "WindowID"]).copy()
        chunk[["JobID", "WindowID"]] = chunk[["JobID", "WindowID"]].astype(np.int64)
        valid_dates = (
            chunk["StartDate"].notna() & chunk["EndDate"].notna()
            & chunk["StartDate"].le(chunk["EndDate"])
        )
        target_mask = chunk["JobID"].isin(wanted_jobs)
        if target_mask.any():
            target_records.append(chunk.loc[target_mask, [
                "JobID", "WindowID", "StartDate", "EndDate",
            ]].copy())
        for protocol, train_days in selected_protocols.items():
            for window in range(1, 8):
                cutoff = date_index.loc[window, "TrainStart"] + pd.Timedelta(days=train_days)
                mask = (
                    valid_dates & chunk["WindowID"].eq(window)
                    & chunk["StartDate"].le(cutoff)
                    & chunk["EndDate"].ge(cutoff)
                )
                if not mask.any():
                    continue
                selected = chunk.loc[mask].copy().drop_duplicates("JobID")
                for column in ("Title", "Description", "Requirements", "City", "State", "Country"):
                    selected[column] = selected[column].map(normalize_text)
                selected["text_all"] = clean_text(
                    "title " + selected["Title"]
                    + " requirements " + selected["Requirements"]
                    + " description " + selected["Description"]
                    + " city " + selected["City"]
                    + " state " + selected["State"]
                    + " country " + selected["Country"]
                )
                key = (protocol, window)
                selected.to_csv(
                    output_files[key], mode="w" if first_write[key] else "a",
                    header=first_write[key], index=False,
                    compression="gzip" if args.compress_catalogs else None,
                )
                first_write[key] = False
                catalog_ids[key].update(selected["JobID"].astype(int).tolist())
        if chunk_no % 10 == 0:
            print(f"  processed {chunk_no * args.chunksize:,} job rows", flush=True)

    job_metadata = (
        pd.concat(target_records, ignore_index=True)
        .sort_values("JobID").drop_duplicates("JobID", keep="first")
    )
    manifests: dict[str, object] = {
        "task_name": "CareerBuilder_InternalTemporal",
        "official_competition_evaluation": False,
        "organizer_split_used": False,
        "preparation_authority": {
            "authoritative": bool(args.authoritative_preparation),
            "study_role": args.study_role,
            "status": "completed",
        },
        "preparation_config": {
            "protocols": list(args.protocols),
            "sample_cap_per_window": int(args.sample_cap),
            "compress_catalogs": bool(args.compress_catalogs),
            "master_sample_seed": MASTER_SAMPLE_SEED,
            "processed_root": str(args.processed.resolve()),
            "jobs_path": str(args.jobs.resolve()),
            "output_root": str(args.output.resolve()),
        },
        "master_sample_seed": MASTER_SAMPLE_SEED,
        "sample_cap_per_window": args.sample_cap,
        "protocols": {},
        "preparer_path": str(Path(__file__).resolve()),
        "preparer_sha256": sha256(Path(__file__)),
        "source_window_dates_sha256": sha256(args.processed / "window_dates.csv"),
        "source_jobs_sha256": sha256(args.jobs),
    }
    for protocol, train_days in selected_protocols.items():
        protocol_manifest: dict[str, object] = {"train_days": train_days, "windows": {}}
        for window in range(1, 8):
            d = date_index.loc[window]
            cutoff = d["TrainStart"] + pd.Timedelta(days=train_days)
            events = apps_by_window[window]
            train = events[events["ApplicationDate"].lt(cutoff)].copy()
            future = events[
                events["ApplicationDate"].between(cutoff, d["TestStart"], inclusive="left")
            ].copy()
            pairs = (
                future.sort_values("ApplicationDate")
                .drop_duplicates(["UserID", "JobID"], keep="first")
                .merge(job_metadata, on="JobID", how="left", suffixes=("_app", "_job"),
                       validate="many_to_one")
            )
            prior_sets = train.groupby("UserID")["JobID"].agg(lambda x: set(map(int, x)))
            pairs["prior_applied"] = [
                int(jid) in prior_sets.get(int(uid), set())
                for uid, jid in zip(pairs["UserID"], pairs["JobID"])
            ]
            metadata_found = pairs["StartDate"].notna() & pairs["EndDate"].notna()
            same_window = pairs["WindowID_job"].eq(window)
            active_event = (
                metadata_found & pairs["StartDate"].le(pairs["ApplicationDate"])
                & pairs["EndDate"].ge(pairs["ApplicationDate"])
            )
            in_cutoff_catalog = pairs["JobID"].isin(catalog_ids[(protocol, window)])
            valid = same_window & active_event & in_cutoff_catalog & ~pairs["prior_applied"]
            valid_targets = pairs.loc[valid, [
                "UserID", "JobID", "ApplicationDate",
            ]].copy()

            profiles = profiles_by_window[window]
            content_users = set(
                profiles.loc[profiles["released_content_present"], "UserID"].astype(int)
            )
            valid_targets = valid_targets[valid_targets["UserID"].isin(content_users)]
            target_counts = valid_targets.groupby("UserID")["JobID"].nunique()
            population_users = np.sort(target_counts.index.to_numpy(dtype=np.int64))
            train_counts = train.groupby("UserID").size()
            population = pd.DataFrame({"UserID": population_users})
            population["precut_application_count"] = (
                population["UserID"].map(train_counts).fillna(0).astype(int)
            )
            population["activity_group"] = pd.cut(
                population["precut_application_count"],
                bins=[-1, 0, 1, 4, np.inf], labels=["cold_0", "one_1", "low_2_4", "high_5plus"],
            ).astype(str)
            population["relevant_jobs"] = population["UserID"].map(target_counts).astype(int)
            n_population = len(population)
            n_sample = min(args.sample_cap, n_population)
            if args.authoritative_preparation:
                required_sample_count = int(
                    AUTHORITATIVE_PREPARATION_DESIGNS[args.study_role][
                        "sample_cap_per_window"
                    ]
                )
                if n_sample != required_sample_count:
                    raise RuntimeError(
                        f"{protocol} W{window}: authoritative {args.study_role} "
                        f"requires {required_sample_count} audit users, but only "
                        f"{n_population} eligible users are available"
                    )
            rng = np.random.default_rng(MASTER_SAMPLE_SEED + window)
            sampled_ids = np.sort(rng.choice(population_users, size=n_sample, replace=False))
            inclusion_probability = n_sample / n_population if n_population else 0.0
            population["selected_for_audit"] = population["UserID"].isin(sampled_ids)
            population["inclusion_probability"] = inclusion_probability
            population["population_weight"] = (
                1.0 / inclusion_probability if inclusion_probability else 0.0
            )

            # Fail closed if any temporal, catalog, novelty, or cohort invariant is
            # violated.  These checks are intentionally redundant with the masks
            # above so future edits cannot silently relax the frozen protocol.
            if not train["ApplicationDate"].lt(cutoff).all():
                raise AssertionError(f"W{window}: pre-cutoff event at/after cutoff")
            if not (
                valid_targets["ApplicationDate"].ge(cutoff)
                & valid_targets["ApplicationDate"].lt(d["TestStart"])
            ).all():
                raise AssertionError(f"W{window}: target outside evaluation horizon")
            catalog_id_set = catalog_ids[(protocol, window)]
            if not set(valid_targets["JobID"].astype(int)).issubset(catalog_id_set):
                raise AssertionError(f"W{window}: target absent from cutoff catalog")
            if not set(valid_targets["UserID"].astype(int)).issubset(set(population_users)):
                raise AssertionError(f"W{window}: target user absent from cohort")
            prior_pair_set = set(zip(train["UserID"].astype(int), train["JobID"].astype(int)))
            target_pair_set = set(zip(
                valid_targets["UserID"].astype(int),
                valid_targets["JobID"].astype(int),
            ))
            if prior_pair_set.intersection(target_pair_set):
                raise AssertionError(f"W{window}: repeated prior-applied target retained")
            if not set(sampled_ids).issubset(set(population_users)):
                raise AssertionError(f"W{window}: sampled user absent from full cohort")

            directory = args.output / protocol / f"window_{window}"
            population.to_csv(directory / "cohort_population.csv", index=False)
            population[population["selected_for_audit"]].to_csv(
                directory / "cohort_audit_sample.csv", index=False
            )
            valid_targets.to_csv(directory / "valid_future_targets.csv", index=False)
            train[["UserID", "JobID", "ApplicationDate"]].to_csv(
                directory / "precut_applications.csv", index=False
            )
            profiles[profiles["UserID"].isin(population_users)].to_csv(
                directory / "query_profiles.csv", index=False
            )
            catalog_path = output_files[(protocol, window)]
            catalog_jobids = pd.read_csv(catalog_path, usecols=["JobID"])["JobID"]
            if catalog_jobids.duplicated().any():
                duplicate_count = int(catalog_jobids.duplicated().sum())
                raise AssertionError(
                    f"W{window}: {duplicate_count} duplicate JobID catalog rows"
                )
            activity_population = {
                str(k): int(v) for k, v in
                population["activity_group"].value_counts().sort_index().items()
            }
            sampled_population = population[population["selected_for_audit"]]
            activity_sample = {
                str(k): int(v) for k, v in
                sampled_population["activity_group"].value_counts().sort_index().items()
            }
            protocol_manifest["windows"][str(window)] = {
                "cutoff": str(cutoff),
                "evaluation_end": str(d["TestStart"]),
                "catalog_jobs": len(catalog_ids[(protocol, window)]),
                "catalog_file_bytes": catalog_path.stat().st_size,
                "catalog_file_sha256": sha256(catalog_path),
                "catalog_file": catalog_path.name,
                "catalog_compression": "gzip" if args.compress_catalogs else "none",
                "future_unique_pairs_before_filters": len(pairs),
                "excluded_cross_window": int((metadata_found & ~same_window).sum()),
                "excluded_date_inconsistent": int((metadata_found & ~active_event).sum()),
                "excluded_not_active_at_cutoff": int((metadata_found & ~in_cutoff_catalog).sum()),
                "excluded_repeated_prior_target_pairs": int(pairs["prior_applied"].sum()),
                "valid_pairs_before_content_filter": int(valid.sum()),
                "valid_pairs_after_content_filter": len(valid_targets),
                "population_queries": n_population,
                "cold_queries": int(population["precut_application_count"].eq(0).sum()),
                "warm_queries": int(population["precut_application_count"].gt(0).sum()),
                "population_activity_counts": activity_population,
                "audit_sample_queries": n_sample,
                "audit_sample_cold_queries": int(
                    sampled_population["precut_application_count"].eq(0).sum()
                ),
                "audit_sample_warm_queries": int(
                    sampled_population["precut_application_count"].gt(0).sum()
                ),
                "audit_sample_activity_counts": activity_sample,
                "inclusion_probability": inclusion_probability,
                "population_weight": 1.0 / inclusion_probability if inclusion_probability else 0.0,
                "sample_design": "uniform_without_replacement_within_window",
                "pooled_weight_definition": "N_w / n_w",
                "hard_gate_catalog_unique": True,
                "hard_gate_targets_in_catalog": True,
                "hard_gate_targets_novel_vs_precut": True,
                "hard_gate_time_boundaries": True,
            }
        manifests["protocols"][protocol] = protocol_manifest
    manifests["runtime_seconds"] = time.perf_counter() - started
    (args.output / "INTERNAL_TEMPORAL_PREPARATION_MANIFEST.json").write_text(
        json.dumps(manifests, indent=2), encoding="utf-8"
    )
    print(args.output / "INTERNAL_TEMPORAL_PREPARATION_MANIFEST.json", flush=True)


if __name__ == "__main__":
    main()
