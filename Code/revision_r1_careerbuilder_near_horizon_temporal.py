#!/usr/bin/env python3
"""CareerBuilder near-horizon future-opening sensitivity.

For each fixed First7Last2 query, this post-decision analysis compares the
protocol-defined cutoff-active catalog with that same catalog plus same-window
jobs whose StartDate is later than the cutoff but no later than the evaluation
horizon end.  Relevance, users, profiles, pre-cutoff behavior, model parameters,
and per-job scores are held fixed across the two catalog regimes.  Lexical models
are fitted only on the reference catalog; added jobs are transformed with those
frozen vocabularies and weights.  Nominally future jobs with any pre-cutoff
interaction are excluded globally so that released date/behavior contradictions
do not confound the perturbation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

import careerbuilder_internal_temporal_evaluation as base


MODELS = base.MODELS
DEFAULT_KS = (5, 10, 20, 150)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-root", type=Path,
        default=root / "Dataset" / "CareerBuilder_InternalTemporal_v1",
    )
    parser.add_argument(
        "--future-root", type=Path,
        default=root / "Dataset" / "CareerBuilder_FutureContamination_v1",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=root / "Results" / "CareerBuilder_InternalTemporal_EvaluationValidity",
    )
    parser.add_argument("--run-label", default="Revision_R1_NearHorizonFutureOpening_v1")
    parser.add_argument("--protocol", default="First7Last2")
    parser.add_argument("--windows", nargs="+", type=int, default=list(range(1, 8)))
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--semantic-model", default=base.MINILM_MODEL)
    parser.add_argument(
        "--max-users-per-window", type=int, default=None,
        help="Sorted audit-cohort prefix for engineering smoke tests only.",
    )
    return parser.parse_args()


def catalog_path(root: Path, protocol: str, window: int, stem: str) -> Path:
    directory = root / protocol / f"window_{window}"
    candidates = [directory / f"{stem}.csv", directory / f"{stem}.csv.gz"]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise FileNotFoundError(
            f"Expected exactly one catalog for W{window} stem={stem}; found {present}"
        )
    return present[0]


def weighted_bm25_assets(
    reference_text: pd.Series,
    future_text: pd.Series,
    query_text: pd.Series,
    max_features: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, object]]:
    vectorizer = CountVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2,
        max_df=0.98, max_features=max_features, dtype=np.float32,
    )
    reference_counts = sparse.csr_matrix(
        vectorizer.fit_transform(reference_text), dtype=np.float32,
    )
    future_counts = sparse.csr_matrix(
        vectorizer.transform(future_text), dtype=np.float32,
    )
    query_counts = sparse.csr_matrix(
        vectorizer.transform(query_text), dtype=np.float32,
    )
    query_counts.data[:] = 1.0
    document_frequency = np.asarray((reference_counts > 0).sum(axis=0)).ravel()
    inverse_document_frequency = np.log1p(
        (reference_counts.shape[0] - document_frequency + 0.5)
        / (document_frequency + 0.5)
    ).astype(np.float32)
    reference_lengths = np.asarray(reference_counts.sum(axis=1)).ravel()
    average_length = float(reference_lengths.mean())
    k1, b = 1.5, 0.75

    def weight(counts: sparse.csr_matrix) -> sparse.csr_matrix:
        counts = counts.copy().tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        row_indices = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        denominator = counts.data + k1 * (
            1.0 - b + b * lengths[row_indices] / average_length
        )
        counts.data = counts.data * (k1 + 1.0) / denominator
        return counts.multiply(inverse_document_frequency).tocsr()

    reference_weighted = weight(reference_counts)
    future_weighted = weight(future_counts)
    combined = sparse.vstack([reference_weighted, future_weighted], format="csr")
    return combined, query_counts, {
        "vocabulary": len(vectorizer.vocabulary_),
        "average_reference_document_length": average_length,
        "idf_fit_catalog": "reference_active_at_cutoff_only",
        "future_transform": "frozen_reference_vocabulary_idf_and_average_length",
        "k1": k1,
        "b": b,
    }


def load_window(args: argparse.Namespace, window: int) -> dict[str, object]:
    started = time.perf_counter()
    reference_directory = args.reference_root / args.protocol / f"window_{window}"
    reference_path = catalog_path(
        args.reference_root, args.protocol, window, "jobs_active_at_cutoff",
    )
    future_path = catalog_path(
        args.future_root, args.protocol, window, "jobs_future_after_cutoff",
    )
    preparation_manifest_path = (
        args.reference_root / "INTERNAL_TEMPORAL_PREPARATION_MANIFEST.json"
    )
    preparation_manifest = json.loads(
        preparation_manifest_path.read_text(encoding="utf-8")
    )
    reference_record = preparation_manifest["protocols"][args.protocol]["windows"][str(window)]
    if base.sha256(reference_path) != reference_record["catalog_file_sha256"]:
        raise AssertionError(f"W{window}: reference catalog hash mismatch")

    reference_jobs = pd.read_csv(reference_path, low_memory=False)
    future_jobs = pd.read_csv(future_path, low_memory=False)
    for frame in (reference_jobs, future_jobs):
        frame["JobID"] = frame["JobID"].astype(np.int64)
        frame["WindowID"] = frame["WindowID"].astype(np.int64)
        frame["StartDate"] = pd.to_datetime(frame["StartDate"], errors="raise")
        frame["EndDate"] = pd.to_datetime(frame["EndDate"], errors="raise")
        frame["Title"] = base.clean(frame["Title"])
        frame["text_all"] = base.clean(frame["text_all"])
        if frame["JobID"].duplicated().any():
            raise AssertionError(f"W{window}: duplicate catalog JobID")
    if len(reference_jobs) != int(reference_record["catalog_jobs"]):
        raise AssertionError(f"W{window}: unexpected reference row count")
    cutoff = pd.Timestamp(reference_record["cutoff"])
    evaluation_end = pd.Timestamp(reference_record["evaluation_end"])
    if not future_jobs["WindowID"].eq(window).all():
        raise AssertionError(f"W{window}: cross-window future job")
    if not future_jobs["StartDate"].gt(cutoff).all():
        raise AssertionError(f"W{window}: future catalog contains non-future StartDate")
    reference_ids = set(map(int, reference_jobs["JobID"]))
    source_future_count = len(future_jobs)
    beyond_horizon_count = int(future_jobs["StartDate"].gt(evaluation_end).sum())
    future_jobs = future_jobs[
        future_jobs["StartDate"].le(evaluation_end)
    ].reset_index(drop=True)
    if future_jobs.empty:
        raise AssertionError(f"W{window}: no near-horizon future-opening jobs")
    if not future_jobs["StartDate"].le(evaluation_end).all():
        raise AssertionError(f"W{window}: near-horizon catalog exceeds evaluation end")
    raw_future_ids = set(map(int, future_jobs["JobID"]))
    if reference_ids.intersection(raw_future_ids):
        raise AssertionError(f"W{window}: reference/future catalog overlap")

    required = [
        "cohort_population.csv", "cohort_audit_sample.csv", "valid_future_targets.csv",
        "precut_applications.csv", "query_profiles.csv",
    ]
    missing = [name for name in required if not (reference_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"W{window}: missing reference inputs {missing}")
    precut = pd.read_csv(reference_directory / "precut_applications.csv")
    precut[["UserID", "JobID"]] = precut[["UserID", "JobID"]].astype(np.int64)
    # A small subset of rows has an application timestamp before the released
    # job StartDate.  Adding those jobs would mix temporal-catalog leakage with
    # date-inconsistent behavioral evidence and give nominally future jobs
    # pre-cutoff popularity.  Exclude every such job globally, before looking
    # at model outcomes, to obtain a conservative future-unseen perturbation.
    inconsistent_future_ids = raw_future_ids.intersection(
        set(map(int, precut["JobID"]))
    )
    inconsistent_precut = precut[precut["JobID"].isin(inconsistent_future_ids)]
    future_jobs = future_jobs[
        ~future_jobs["JobID"].isin(inconsistent_future_ids)
    ].reset_index(drop=True)
    future_ids = set(map(int, future_jobs["JobID"]))
    if future_ids.intersection(set(map(int, precut["JobID"]))):
        raise AssertionError(f"W{window}: retained future job has pre-cutoff behavior")
    if not future_ids:
        raise AssertionError(f"W{window}: consistency gate removed every future job")
    jobs = pd.concat([reference_jobs, future_jobs], ignore_index=True)

    population = pd.read_csv(reference_directory / "cohort_population.csv")
    users = pd.read_csv(reference_directory / "cohort_audit_sample.csv")
    users = users.sort_values("UserID").reset_index(drop=True)
    if args.max_users_per_window is not None:
        users = users.head(args.max_users_per_window).copy()
    profiles = pd.read_csv(reference_directory / "query_profiles.csv")
    profiles["profile_combined"] = base.clean(profiles["profile_combined"])
    users = users.merge(
        profiles[["UserID", "profile_combined"]], on="UserID", how="left",
        validate="one_to_one",
    )
    if users["profile_combined"].fillna("").str.strip().eq("").any():
        raise AssertionError(f"W{window}: audit user without profile text")
    targets = pd.read_csv(reference_directory / "valid_future_targets.csv")
    targets = targets[targets["UserID"].isin(users["UserID"])].copy()
    truth = targets.groupby("UserID")["JobID"].agg(
        lambda values: set(map(int, values))
    ).to_dict()
    if set(map(int, users["UserID"])) != set(map(int, truth)):
        raise AssertionError(f"W{window}: user/target mismatch")
    if any(not relevant.issubset(reference_ids) for relevant in truth.values()):
        raise AssertionError(f"W{window}: target outside reference catalog")

    prior = {
        int(user_id): set(map(int, values))
        for user_id, values in precut[
            precut["UserID"].isin(users["UserID"])
        ].groupby("UserID")["JobID"]
    }
    for user_id, relevant in truth.items():
        if relevant.intersection(prior.get(user_id, set())):
            raise AssertionError(f"W{window} U{user_id}: repeated target")
    if any(future_ids.intersection(values) for values in prior.values()):
        raise AssertionError(f"W{window}: future job occurs in pre-cutoff behavior")

    popularity_counts = precut["JobID"].value_counts()
    popularity = jobs["JobID"].map(popularity_counts).fillna(0).to_numpy(np.float32)
    mean_target_popularity = {
        user_id: float(np.mean([popularity_counts.get(job_id, 0) for job_id in relevant]))
        for user_id, relevant in truth.items()
    }
    users["mean_target_precut_popularity"] = (
        users["UserID"].map(mean_target_popularity).astype(float)
    )
    users["target_popularity_group"] = pd.cut(
        users["mean_target_precut_popularity"], bins=[-1, 0, 2, np.inf],
        labels=["mean_unseen_0", "mean_positive_le2", "mean_gt2"],
    ).astype(str)
    return {
        "window": window,
        "reference_directory": reference_directory,
        "reference_path": reference_path,
        "future_path": future_path,
        "reference_jobs": reference_jobs,
        "future_jobs": future_jobs,
        "jobs": jobs,
        "reference_count": len(reference_jobs),
        "raw_future_count": len(raw_future_ids),
        "source_future_count": source_future_count,
        "beyond_horizon_count": beyond_horizon_count,
        "excluded_inconsistent_future_count": len(inconsistent_future_ids),
        "excluded_inconsistent_precut_rows": len(inconsistent_precut),
        "excluded_inconsistent_precut_users": int(inconsistent_precut["UserID"].nunique()),
        "future_count": len(future_jobs),
        "population": population,
        "users": users,
        "truth": truth,
        "prior": prior,
        "job_ids": jobs["JobID"].to_numpy(np.int64),
        "job_index": {int(job_id): index for index, job_id in enumerate(jobs["JobID"])},
        "popularity": popularity,
        "cutoff": cutoff,
        "evaluation_end": evaluation_end,
        "load_seconds": time.perf_counter() - started,
    }


def prepare_models(
    args: argparse.Namespace, assets: dict[str, object], output_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    models = set(args.models)
    reference_jobs = assets["reference_jobs"]
    future_jobs = assets["future_jobs"]
    users = assets["users"]
    metadata: dict[str, object] = {
        "lexical_fit_catalog": "reference_active_at_cutoff_only",
    }

    if "tfidf_combined_static" in models:
        vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2,
            max_df=0.98, max_features=args.max_features, sublinear_tf=True,
            dtype=np.float32,
        )
        reference_matrix = normalize(
            vectorizer.fit_transform(reference_jobs["text_all"]), copy=False,
        ).tocsr()
        future_matrix = normalize(
            vectorizer.transform(future_jobs["text_all"]), copy=False,
        ).tocsr()
        assets["tfidf_jobs_t"] = sparse.vstack(
            [reference_matrix, future_matrix], format="csr",
        ).T.tocsr()
        assets["tfidf_users"] = normalize(
            vectorizer.transform(users["profile_combined"]), copy=False,
        ).tocsr()
        metadata["tfidf"] = {
            "vocabulary": len(vectorizer.vocabulary_),
            "fit_catalog": "reference_active_at_cutoff_only",
            "future_transform": "frozen_reference_vectorizer_and_idf",
        }

    if "bm25_full_static" in models:
        jobs, queries, model_meta = weighted_bm25_assets(
            reference_jobs["text_all"], future_jobs["text_all"],
            users["profile_combined"], args.max_features,
        )
        assets["bm25_jobs_t"] = jobs.T.tocsr()
        assets["bm25_users"] = queries
        metadata["bm25_full"] = model_meta
    if "bm25_title_static" in models:
        jobs, queries, model_meta = weighted_bm25_assets(
            reference_jobs["Title"], future_jobs["Title"],
            users["profile_combined"], args.max_features,
        )
        assets["bm25_title_jobs_t"] = jobs.T.tocsr()
        assets["bm25_title_users"] = queries
        metadata["bm25_title"] = model_meta

    if "minilm_title_static" in models:
        reference_text = base.clean("job title: " + reference_jobs["Title"])
        future_text = base.clean("job title: " + future_jobs["Title"])
        user_text = base.clean("candidate profile: " + users["profile_combined"])
        reference_embeddings, reference_meta = base.load_or_encode(
            output_dir, reference_text,
            f"{args.protocol}_W{assets['window']}_jobs_title",
            args.semantic_model, base.MINILM_EXPECTED_REVISION,
            args.semantic_snapshot_evidence, args.embedding_batch_size,
        )
        future_embeddings, future_meta = base.load_or_encode(
            output_dir, future_text,
            f"{args.protocol}_W{assets['window']}_future_jobs_title",
            args.semantic_model, base.MINILM_EXPECTED_REVISION,
            args.semantic_snapshot_evidence, args.embedding_batch_size,
        )
        user_embeddings, user_meta = base.load_or_encode(
            output_dir, user_text,
            f"{args.protocol}_W{assets['window']}_users_static_"
            f"{base.sha256(assets['reference_directory'] / 'cohort_audit_sample.csv')[:12]}",
            args.semantic_model, base.MINILM_EXPECTED_REVISION,
            args.semantic_snapshot_evidence, args.embedding_batch_size,
        )
        assets["job_embeddings"] = np.vstack(
            [reference_embeddings, future_embeddings]
        ).astype(np.float32, copy=False)
        assets["user_embeddings"] = user_embeddings
        metadata["minilm"] = {
            "reference": reference_meta,
            "future": future_meta,
            "user": user_meta,
            "model": args.semantic_model,
            "revision": base.MINILM_EXPECTED_REVISION,
            "maximum_sequence_length": 256,
            "normalized": True,
            "similarity": "cosine via dot product of normalized embeddings",
        }
    metadata["elapsed_seconds"] = time.perf_counter() - started
    return metadata


def score_batch(
    args: argparse.Namespace, assets: dict[str, object], start: int, stop: int,
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    if "tfidf_combined_static" in args.models:
        scores["tfidf_combined_static"] = (
            assets["tfidf_users"][start:stop] @ assets["tfidf_jobs_t"]
        ).toarray()
    if "bm25_full_static" in args.models:
        scores["bm25_full_static"] = (
            assets["bm25_users"][start:stop] @ assets["bm25_jobs_t"]
        ).toarray()
    if "bm25_title_static" in args.models:
        scores["bm25_title_static"] = (
            assets["bm25_title_users"][start:stop] @ assets["bm25_title_jobs_t"]
        ).toarray()
    if "minilm_title_static" in args.models:
        scores["minilm_title_static"] = (
            np.asarray(assets["user_embeddings"][start:stop], dtype=np.float32)
            @ np.asarray(assets["job_embeddings"], dtype=np.float32).T
        )
    return scores


def evaluate_user(
    args: argparse.Namespace,
    assets: dict[str, object],
    user_offset: int,
    score_vectors: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    user = assets["users"].iloc[user_offset]
    user_id = int(user["UserID"])
    relevant = assets["truth"][user_id]
    job_ids = assets["job_ids"]
    prior_indices = np.asarray(sorted(
        assets["job_index"][job_id]
        for job_id in assets["prior"].get(user_id, set())
        if job_id in assets["job_index"]
    ), dtype=np.int64)
    union_mask = np.ones(len(job_ids), dtype=bool)
    union_mask[prior_indices] = False
    contaminated_indices = np.flatnonzero(union_mask)
    reference_mask = union_mask[:assets["reference_count"]]
    reference_indices = np.flatnonzero(reference_mask)
    if len(contaminated_indices) - len(reference_indices) != assets["future_count"]:
        raise AssertionError("Future candidate count changed after prior exclusion")
    common = {
        "protocol": args.protocol,
        "window": assets["window"],
        "user_id": user_id,
        "precut_application_count": int(user["precut_application_count"]),
        "activity_group": str(user["activity_group"]),
        "mean_target_precut_popularity": float(user["mean_target_precut_popularity"]),
        "target_popularity_group": str(user["target_popularity_group"]),
        "relevant_jobs": len(relevant),
        "inclusion_probability": float(user["inclusion_probability"]),
        "population_weight": float(user["population_weight"]),
        "future_jobs_added": assets["future_count"],
    }
    static_scores = {"popularity_precut": assets["popularity"], **score_vectors}
    rows: list[dict[str, object]] = []
    for model in args.models:
        if model == "random_hash":
            full_scores = base.splitmix64_scores(job_ids, user_id, assets["window"])
        else:
            full_scores = static_scores[model]
        for regime, indices in (
            ("cutoff_active_reference", reference_indices),
            ("near_horizon_future_opening", contaminated_indices),
        ):
            eligible_ids = job_ids[indices]
            ranked = base.top_with_deterministic_ties(
                eligible_ids, full_scores[indices], max(args.ks),
            )
            for values in base.metric_rows(ranked, relevant, tuple(args.ks)):
                rows.append({
                    **common,
                    "catalog_regime": regime,
                    "candidate_count": len(indices),
                    "model": model,
                    **values,
                })
    return rows


def aggregate_results(per_query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = list(base.METRICS)
    group_columns = ["catalog_regime", "model", "k"]
    summary_rows: list[dict[str, object]] = []
    for keys, cell in per_query.groupby(group_columns, sort=True):
        for estimand in ("audit_cohort", "design_weighted_population"):
            weights = (
                np.ones(len(cell), dtype=float)
                if estimand == "audit_cohort"
                else cell["population_weight"].to_numpy(dtype=float)
            )
            row: dict[str, object] = {
                **dict(zip(group_columns, keys)),
                "estimand": estimand,
                "queries": len(cell),
                "windows": cell["window"].nunique(),
                "candidate_count_mean": float(np.average(cell["candidate_count"], weights=weights)),
                "future_jobs_added_mean": float(np.average(cell["future_jobs_added"], weights=weights)),
            }
            row.update({
                metric: float(np.average(cell[metric], weights=weights))
                for metric in metrics
            })
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    paired = per_query.pivot(
        index=[
            "window", "user_id", "activity_group", "target_popularity_group",
            "population_weight", "model", "k",
        ],
        columns="catalog_regime", values=metrics,
    )
    paired.columns = [f"{metric}__{regime}" for metric, regime in paired.columns]
    paired = paired.reset_index()
    for metric in metrics:
        paired[f"{metric}_change_contaminated_minus_reference"] = (
            paired[f"{metric}__near_horizon_future_opening"]
            - paired[f"{metric}__cutoff_active_reference"]
        )

    ranking_rows: list[dict[str, object]] = []
    primary = summary[summary["estimand"].eq("design_weighted_population")]
    for k in sorted(primary["k"].unique()):
        for metric in metrics:
            orders: dict[str, list[str]] = {}
            for regime in ("cutoff_active_reference", "near_horizon_future_opening"):
                cell = primary[
                    primary["catalog_regime"].eq(regime) & primary["k"].eq(k)
                ].sort_values([metric, "model"], ascending=[False, True])
                orders[regime] = cell["model"].tolist()
                for rank, row in enumerate(cell.itertuples(index=False), 1):
                    ranking_rows.append({
                        "record_type": "model_rank", "k": k, "metric": metric,
                        "catalog_regime": regime, "model": row.model,
                        "rank": rank, "value": getattr(row, metric),
                        "kendall_tau_between_regimes": np.nan,
                        "top_model_agreement": np.nan,
                    })
            reference_order = orders["cutoff_active_reference"]
            contaminated_order = orders["near_horizon_future_opening"]
            reference_ranks = {model: rank for rank, model in enumerate(reference_order)}
            contaminated_ranks = [reference_ranks[model] for model in contaminated_order]
            ranking_rows.append({
                "record_type": "regime_comparison", "k": k, "metric": metric,
                "catalog_regime": "reference_vs_contaminated", "model": "all",
                "rank": np.nan, "value": np.nan,
                "kendall_tau_between_regimes": float(
                    kendalltau(np.arange(len(reference_order)), contaminated_ranks).statistic
                ),
                "top_model_agreement": reference_order[0] == contaminated_order[0],
            })
    return summary, paired, pd.DataFrame(ranking_rows)


def main() -> None:
    args = parse_args()
    started = time.time()
    args.windows = sorted(set(args.windows))
    args.ks = sorted(set(args.ks))
    args.models = list(dict.fromkeys(args.models))
    if any(window not in range(1, 8) for window in args.windows):
        raise ValueError("Windows must be in 1..7")
    output_dir = args.output_root / args.run_label
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh empty run directory: {output_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_path = Path(__file__).resolve()
    base_evaluator_path = Path(base.__file__).resolve()
    code_provenance = {
        "temporal_contamination_evaluator": {
            "path": str(evaluator_path),
            "sha256": base.sha256(evaluator_path),
        },
        "imported_base_evaluator": {
            "path": str(base_evaluator_path),
            "sha256": base.sha256(base_evaluator_path),
        },
    }
    if "minilm_title_static" in args.models:
        args.semantic_snapshot_evidence = base.resolve_minilm_snapshot(args.semantic_model)
    else:
        args.semantic_snapshot_evidence = {"skipped": True}
    run_config = {
        "analysis_role": "post_decision_sensitivity",
        "reference_root": str(args.reference_root.resolve()),
        "future_root": str(args.future_root.resolve()),
        "protocol": args.protocol,
        "windows": args.windows,
        "models": args.models,
        "cutoffs": args.ks,
        "max_features": args.max_features,
        "query_batch_size": args.query_batch_size,
        "embedding_batch_size": args.embedding_batch_size,
        "semantic_model": args.semantic_model,
        "max_users_per_window": args.max_users_per_window,
        "code_provenance": code_provenance,
    }
    run_config_path = output_dir / "RUN_CONFIG.json"
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    all_frames: list[pd.DataFrame] = []
    window_records: list[dict[str, object]] = []
    for window in args.windows:
        print(f"Loading W{window}", flush=True)
        assets = load_window(args, window)
        print(
            f"W{window}: {len(assets['users']):,} users, "
            f"{assets['reference_count']:,} reference jobs, "
            f"{assets['future_count']:,} retained future jobs "
            f"({assets['excluded_inconsistent_future_count']:,} date/behavior-"
            "inconsistent jobs excluded)",
            flush=True,
        )
        model_metadata = prepare_models(args, assets, output_dir)
        rows: list[dict[str, object]] = []
        eval_started = time.perf_counter()
        for start in range(0, len(assets["users"]), args.query_batch_size):
            stop = min(start + args.query_batch_size, len(assets["users"]))
            batch_scores = score_batch(args, assets, start, stop)
            for local_offset in range(stop - start):
                scores = {
                    model: values[local_offset]
                    for model, values in batch_scores.items()
                }
                rows.extend(evaluate_user(args, assets, start + local_offset, scores))
            if stop % 100 == 0 or stop == len(assets["users"]):
                print(f"  W{window}: evaluated {stop:,}/{len(assets['users']):,}", flush=True)
        frame = pd.DataFrame(rows)
        output_path = output_dir / f"careerbuilder_near_horizon_window_{window}.csv.gz"
        with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
        all_frames.append(frame)
        window_records.append({
            "window": window,
            "users": len(assets["users"]),
            "population_users": len(assets["population"]),
            "reference_jobs": assets["reference_count"],
            "source_future_opening_jobs": assets["source_future_count"],
            "future_jobs_beyond_evaluation_horizon_excluded": assets["beyond_horizon_count"],
            "raw_future_opening_jobs": assets["raw_future_count"],
            "future_jobs_excluded_for_precut_interaction": (
                assets["excluded_inconsistent_future_count"]
            ),
            "excluded_inconsistent_precut_rows": (
                assets["excluded_inconsistent_precut_rows"]
            ),
            "excluded_inconsistent_precut_users": (
                assets["excluded_inconsistent_precut_users"]
            ),
            "future_jobs_added": assets["future_count"],
            "contaminated_jobs": assets["reference_count"] + assets["future_count"],
            "cutoff": assets["cutoff"].isoformat(),
            "evaluation_end": assets["evaluation_end"].isoformat(),
            "reference_catalog_sha256": base.sha256(assets["reference_path"]),
            "future_catalog_sha256": base.sha256(assets["future_path"]),
            "model_metadata": model_metadata,
            "load_seconds": assets["load_seconds"],
            "evaluation_seconds": time.perf_counter() - eval_started,
            "output": {
                "path": str(output_path.resolve()),
                "bytes": output_path.stat().st_size,
                "sha256": base.sha256(output_path),
                "rows": len(frame),
            },
        })

    per_query = pd.concat(all_frames, ignore_index=True)
    summary, paired, rankings = aggregate_results(per_query)
    summary_path = output_dir / "careerbuilder_near_horizon_summary.csv"
    paired_path = output_dir / "careerbuilder_near_horizon_paired_changes.csv.gz"
    rankings_path = output_dir / "careerbuilder_near_horizon_model_rankings.csv"
    summary.to_csv(summary_path, index=False)
    with gzip.open(paired_path, "wt", encoding="utf-8", newline="") as handle:
        paired.to_csv(handle, index=False)
    rankings.to_csv(rankings_path, index=False)

    audit = {
        "analysis_role": "post_decision_near_horizon_temporal_sensitivity",
        "estimand": (
            "Change in logged-reference metrics when behavior-consistent same-window "
            "jobs opening after the First7Last2 cutoff but no later than the evaluation "
            "horizon end are added to the cutoff-active catalog."
        ),
        "invariants": {
            "queries_fixed": True,
            "targets_fixed": True,
            "profiles_fixed": True,
            "precut_behavior_fixed": True,
            "model_parameters_fixed_between_catalog_regimes": True,
            "per_job_scores_fixed_between_catalog_regimes": True,
            "tie_break_fixed": "descending score, then ascending JobID",
            "only_near_horizon_future_opening_alternatives_added": True,
            "added_job_start_date_rule": "cutoff < StartDate <= evaluation_end",
            "future_jobs_with_any_precut_interaction_excluded_globally": True,
        },
        "not_claimed": ["latent preference validity", "deployment utility", "causal effect"],
        "protocol": args.protocol,
        "windows": args.windows,
        "models": args.models,
        "cutoffs": args.ks,
        "max_users_per_window": args.max_users_per_window,
        "semantic_snapshot": args.semantic_snapshot_evidence,
        "code_provenance": code_provenance,
        "run_config": {
            "path": str(run_config_path.resolve()),
            "bytes": run_config_path.stat().st_size,
            "sha256": base.sha256(run_config_path),
        },
        "software": base.software_versions(),
        "window_records": window_records,
        "outputs": {
            "summary": {
                "path": str(summary_path.resolve()), "bytes": summary_path.stat().st_size,
                "sha256": base.sha256(summary_path),
            },
            "paired_changes": {
                "path": str(paired_path.resolve()), "bytes": paired_path.stat().st_size,
                "sha256": base.sha256(paired_path),
            },
            "rankings": {
                "path": str(rankings_path.resolve()), "bytes": rankings_path.stat().st_size,
                "sha256": base.sha256(rankings_path),
            },
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    audit_path = output_dir / "careerbuilder_temporal_contamination_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "completed": True,
        "queries": int(per_query[["window", "user_id"]].drop_duplicates().shape[0]),
        "rows": len(per_query),
        "elapsed_seconds": audit["elapsed_seconds"],
        "audit": str(audit_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
