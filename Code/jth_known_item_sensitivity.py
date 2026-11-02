#!/usr/bin/env python3
"""JTH known-item sensitivity with protocol-consistent scores and sampling.

The original JTH experiment allows jobs seen for a candidate in the released
training split to remain eligible at test time.  This script reproduces the
preselected full-catalog and 100-candidate K=20 conditions and adds a
novel-item estimand:

* remove every training-seen job from that candidate's eligible catalog;
* remove repeat targets from the relevance set;
* exclude a query only when no novel target remains;
* keep model scores unchanged; and
* construct the novel sampled pool by filtering the *original* seeded
  negative order and replenishing from later entries in that same order.

The latter rule isolates known-item removal from a new random draw.  Both the
original and novel-item rows are written so the original rows can be checked
against the independently retained production results.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from jth_sampling_evaluation import (
    SAFE_COMMON_FIELDS,
    join_fields,
    metric_row,
    query_seed,
    sampled_negative_order,
    semantic_embeddings,
)


MODELS = (
    "random", "popularity", "tfidf", "bm25", "recency_pop",
    "item_knn", "semantic",
)
STRATEGIES = ("uniform", "popularity_weighted")
SEEDS = (1, 2, 3, 4, 5)
K = 20
CANDIDATE_SIZE = 100
FIELDS = [
    "estimand", "query_index", "group", "n_relevant_original",
    "n_relevant_eval", "n_known_jobs_removed", "eligible_catalog_original",
    "eligible_catalog_eval", "model", "strategy", "seed", "candidate_size",
    "actual_candidate_size", "k", "precision", "recall", "hitrate",
    "ndcg", "map", "mrr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Dataset/external/JTH/data"))
    parser.add_argument(
        "--out", type=Path,
        default=Path("Results/JTH_EvaluationValidity_Strengthened"),
    )
    parser.add_argument(
        "--semantic-cache", type=Path, default=Path("Results/JTH_Semantic_Cache")
    )
    parser.add_argument(
        "--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def emit(
    writer: csv.DictWriter,
    *,
    estimand: str,
    query_index: int,
    group: str,
    n_relevant_original: int,
    n_relevant_eval: int,
    n_known_jobs_removed: int,
    eligible_original: int,
    eligible_eval: int,
    model: str,
    strategy: str,
    seed: int,
    candidate_size: int,
    actual_candidate_size: int,
    ranked_job_ids: np.ndarray,
    relevant: set[int],
) -> None:
    metrics = metric_row(ranked_job_ids, relevant, K)
    writer.writerow({
        "estimand": estimand,
        "query_index": query_index,
        "group": group,
        "n_relevant_original": n_relevant_original,
        "n_relevant_eval": n_relevant_eval,
        "n_known_jobs_removed": n_known_jobs_removed,
        "eligible_catalog_original": eligible_original,
        "eligible_catalog_eval": eligible_eval,
        "model": model,
        "strategy": strategy,
        "seed": seed,
        "candidate_size": candidate_size,
        "actual_candidate_size": actual_candidate_size,
        "k": K,
        **metrics,
    })


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "known_item_per_query_k20.csv.gz"

    jobs_path = args.data / "jobs.csv"
    candidates_path = args.data / "candidates.csv"
    train_path = args.data / "binary" / "train.csv"
    test_path = args.data / "binary" / "test.csv"

    jobs = pd.read_csv(jobs_path)
    candidates = pd.read_csv(candidates_path)
    train = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    jobs["create_date"] = pd.to_datetime(jobs["create_date"])
    candidates["create_date"] = pd.to_datetime(candidates["create_date"])
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test_raw["timestamp"] = pd.to_datetime(test_raw["timestamp"])

    test = test_raw.drop_duplicates().copy()
    grouped_all = (
        test.groupby(["candidate_id", "timestamp"], sort=True)["job_id"]
        .agg(lambda values: sorted(set(map(int, values))))
        .reset_index(name="relevant_jobs")
    )
    grouped_all.insert(0, "query_index", np.arange(len(grouped_all), dtype=np.int64))
    grouped = grouped_all if args.max_queries is None else grouped_all.head(args.max_queries).copy()

    jobs = jobs.sort_values("job_id").reset_index(drop=True)
    candidates = candidates.set_index("candidate_id", drop=False)
    job_ids = jobs.job_id.to_numpy(dtype=np.int64)
    jid_to_idx = {int(job_id): i for i, job_id in enumerate(job_ids)}
    job_dates = jobs.create_date.to_numpy(dtype="datetime64[ns]")

    train_pairs = train[["candidate_id", "job_id"]].drop_duplicates()
    seen_by_candidate = {
        int(candidate_id): set(map(int, part.job_id))
        for candidate_id, part in train_pairs.groupby("candidate_id", sort=False)
    }
    train_candidates = set(seen_by_candidate)
    train_popularity = train.job_id.value_counts().to_dict()
    pop_scores = np.array(
        [train_popularity.get(int(job_id), 0) for job_id in job_ids], dtype=np.float64
    )
    pop_weights = pop_scores + 1.0

    overlap = test.merge(train_pairs, on=["candidate_id", "job_id"], how="inner")
    overlap_pairs = int(overlap[["candidate_id", "job_id"]].drop_duplicates().shape[0])
    overlap_events = int(len(overlap))
    overlap_queries = int(overlap[["candidate_id", "timestamp"]].drop_duplicates().shape[0])

    unknown_targets = sorted({
        job_id for values in grouped.relevant_jobs for job_id in values
        if job_id not in jid_to_idx
    })
    if unknown_targets:
        raise ValueError(f"Test targets absent from jobs.csv: {unknown_targets[:10]}")

    candidate_rows = candidates.loc[grouped.candidate_id.unique()]
    candidate_vector_row = {
        int(candidate_id): row for row, candidate_id in enumerate(candidate_rows.index)
    }
    job_text = join_fields(jobs, SAFE_COMMON_FIELDS)
    candidate_text = join_fields(candidate_rows, SAFE_COMMON_FIELDS)

    tfidf_jobs = tfidf_candidates = None
    tfidf_vocabulary = None
    if "tfidf" in args.models:
        vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.98, sublinear_tf=True, ngram_range=(1, 2),
            max_features=100_000, norm="l2", dtype=np.float32,
        )
        tfidf_jobs = vectorizer.fit_transform(job_text)
        tfidf_candidates = sparse.csr_matrix(vectorizer.transform(candidate_text))
        tfidf_vocabulary = int(len(vectorizer.vocabulary_))

    bm25_jobs = bm25_candidates = None
    bm25_vocabulary = None
    bm25_avgdl = None
    if "bm25" in args.models:
        count_vectorizer = CountVectorizer(
            min_df=2, max_df=0.98, ngram_range=(1, 2), max_features=100_000,
            dtype=np.float32,
        )
        counts = sparse.csr_matrix(count_vectorizer.fit_transform(job_text), dtype=np.float32)
        query_counts = sparse.csr_matrix(
            count_vectorizer.transform(candidate_text), dtype=np.float32
        )
        query_counts.data[:] = 1.0
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        inverse_document_frequency = np.log1p(
            (counts.shape[0] - document_frequency + 0.5)
            / (document_frequency + 0.5)
        ).astype(np.float32)
        document_lengths = np.asarray(counts.sum(axis=1)).ravel()
        bm25_avgdl = float(document_lengths.mean())
        k1, b = 1.5, 0.75
        row_index = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        denominator = counts.data + k1 * (
            1 - b + b * document_lengths[row_index] / bm25_avgdl
        )
        counts.data = counts.data * (k1 + 1) / denominator
        bm25_jobs = counts.multiply(inverse_document_frequency).tocsr()
        bm25_candidates = query_counts
        bm25_vocabulary = int(len(count_vectorizer.vocabulary_))

    semantic_jobs = semantic_candidates = None
    semantic_meta: dict[str, object] | None = None
    if "semantic" in args.models:
        semantic_jobs, job_digest, job_cache_hit = semantic_embeddings(
            job_text, args.semantic_model, args.semantic_cache, "jobs"
        )
        semantic_candidates, candidate_digest, candidate_cache_hit = semantic_embeddings(
            candidate_text, args.semantic_model, args.semantic_cache, "candidates"
        )
        if semantic_jobs.shape[1] != semantic_candidates.shape[1]:
            raise ValueError("Semantic embedding dimensions differ")
        semantic_meta = {
            "model": args.semantic_model,
            "dimension": int(semantic_jobs.shape[1]),
            "normalized": True,
            "job_text_sha256": job_digest,
            "candidate_text_sha256": candidate_digest,
            "job_cache_hit": bool(job_cache_hit),
            "candidate_cache_hit": bool(candidate_cache_hit),
        }

    cf_seen = cf_item_similarity = None
    cf_candidate_row: dict[int, int] | None = None
    if "item_knn" in args.models:
        cf_candidates = np.sort(train_pairs.candidate_id.unique())
        cf_candidate_row = {
            int(candidate_id): row for row, candidate_id in enumerate(cf_candidates)
        }
        row_ids = train_pairs.candidate_id.map(cf_candidate_row).to_numpy()
        column_ids = train_pairs.job_id.map(jid_to_idx).to_numpy()
        valid = ~pd.isna(column_ids)
        interaction = sparse.csr_matrix(
            (
                np.ones(int(valid.sum()), dtype=np.float32),
                (row_ids[valid], column_ids[valid].astype(np.int32)),
            ),
            shape=(len(cf_candidates), len(job_ids)),
        )
        column_norm = np.sqrt(np.asarray(interaction.sum(axis=0)).ravel())
        inverse_norm = np.divide(
            1.0, column_norm, out=np.zeros_like(column_norm), where=column_norm > 0
        )
        normalized = interaction.multiply(inverse_norm).tocsr()
        cf_seen = interaction
        cf_item_similarity = (normalized.T @ normalized).tocsr()

    output_rows = 0
    original_rows = 0
    novel_rows = 0
    novel_queries = 0
    excluded_repeat_only_queries = 0
    removed_known_counts: list[int] = []
    original_target_events = 0
    novel_target_events = 0

    with gzip.open(output, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        for position, query in enumerate(grouped.itertuples(index=False), 1):
            query_index = int(query.query_index)
            candidate_id = int(query.candidate_id)
            timestamp = pd.Timestamp(query.timestamp)
            relevant_original = set(map(int, query.relevant_jobs))
            known_jobs = seen_by_candidate.get(candidate_id, set())
            relevant_novel = relevant_original.difference(known_jobs)
            original_target_events += len(relevant_original)
            novel_target_events += len(relevant_novel)
            if not relevant_novel:
                excluded_repeat_only_queries += 1
            else:
                novel_queries += 1

            eligible_original = np.flatnonzero(
                job_dates <= np.datetime64(timestamp)
            ).astype(np.int32)
            known_indices = np.fromiter(
                (
                    jid_to_idx[job_id] for job_id in known_jobs
                    if job_id in jid_to_idx and job_dates[jid_to_idx[job_id]] <= np.datetime64(timestamp)
                ),
                dtype=np.int32,
            )
            known_index_set = set(map(int, known_indices))
            if known_index_set:
                eligible_novel = np.fromiter(
                    (idx for idx in eligible_original if int(idx) not in known_index_set),
                    dtype=np.int32,
                )
            else:
                eligible_novel = eligible_original
            removed_known_counts.append(int(len(eligible_original) - len(eligible_novel)))

            relevant_idx_original = np.array(
                [jid_to_idx[job_id] for job_id in relevant_original], dtype=np.int32
            )
            relevant_idx_novel = np.array(
                [jid_to_idx[job_id] for job_id in relevant_novel], dtype=np.int32
            )
            group = "warm" if candidate_id in train_candidates else "cold"
            vector_row = candidate_vector_row[candidate_id]

            score_by_model: dict[str, np.ndarray] = {}
            if "random" in args.models:
                score_by_model["random"] = np.random.default_rng(
                    query_seed(0, candidate_id, timestamp, "model_random")
                ).random(len(job_ids))
            if "popularity" in args.models:
                score_by_model["popularity"] = pop_scores
            if "tfidf" in args.models:
                score_by_model["tfidf"] = (
                    tfidf_jobs @ tfidf_candidates[vector_row].T
                ).toarray().ravel()
            if "bm25" in args.models:
                score_by_model["bm25"] = (
                    bm25_jobs @ bm25_candidates[vector_row].T
                ).toarray().ravel()
            if "semantic" in args.models:
                score_by_model["semantic"] = (
                    semantic_jobs @ semantic_candidates[vector_row]
                )
            if "recency_pop" in args.models:
                age_days = np.maximum(
                    0.0,
                    (np.datetime64(timestamp) - job_dates) / np.timedelta64(1, "D"),
                )
                score_by_model["recency_pop"] = pop_scores * np.exp(-0.02 * age_days)
            if "item_knn" in args.models:
                cf_row = cf_candidate_row.get(candidate_id) if cf_candidate_row else None
                if cf_row is None:
                    score_by_model["item_knn"] = np.zeros(len(job_ids), dtype=np.float32)
                else:
                    seen_indices = cf_seen[cf_row].indices
                    score_by_model["item_knn"] = np.asarray(
                        cf_item_similarity[:, seen_indices].sum(axis=1)
                    ).ravel()

            negative_orders: dict[tuple[str, int], np.ndarray] = {}
            for strategy in STRATEGIES:
                for seed in SEEDS:
                    rng = np.random.default_rng(
                        query_seed(seed, candidate_id, timestamp, strategy)
                    )
                    negative_orders[(strategy, seed)] = sampled_negative_order(
                        eligible_original, relevant_idx_original, strategy, rng, pop_weights
                    )

            for model, scores in score_by_model.items():
                full_order_original = eligible_original[
                    np.lexsort((job_ids[eligible_original], -scores[eligible_original]))
                ]
                emit(
                    writer, estimand="original", query_index=query_index, group=group,
                    n_relevant_original=len(relevant_original),
                    n_relevant_eval=len(relevant_original),
                    n_known_jobs_removed=len(eligible_original) - len(eligible_novel),
                    eligible_original=len(eligible_original),
                    eligible_eval=len(eligible_original), model=model, strategy="full",
                    seed=0, candidate_size=-1,
                    actual_candidate_size=len(eligible_original),
                    ranked_job_ids=job_ids[full_order_original],
                    relevant=relevant_original,
                )
                original_rows += 1
                output_rows += 1

                if relevant_novel:
                    if known_index_set:
                        full_order_novel = np.fromiter(
                            (
                                idx for idx in full_order_original
                                if int(idx) not in known_index_set
                            ),
                            dtype=np.int32,
                        )
                    else:
                        full_order_novel = full_order_original
                    emit(
                        writer, estimand="novel_item", query_index=query_index,
                        group=group, n_relevant_original=len(relevant_original),
                        n_relevant_eval=len(relevant_novel),
                        n_known_jobs_removed=len(eligible_original) - len(eligible_novel),
                        eligible_original=len(eligible_original),
                        eligible_eval=len(eligible_novel), model=model, strategy="full",
                        seed=0, candidate_size=-1,
                        actual_candidate_size=len(eligible_novel),
                        ranked_job_ids=job_ids[full_order_novel],
                        relevant=relevant_novel,
                    )
                    novel_rows += 1
                    output_rows += 1

                for strategy in STRATEGIES:
                    for seed in SEEDS:
                        negative_order = negative_orders[(strategy, seed)]
                        original_take = min(
                            len(negative_order),
                            max(0, CANDIDATE_SIZE - len(relevant_idx_original)),
                        )
                        chosen_original = np.concatenate(
                            [relevant_idx_original, negative_order[:original_take]]
                        )
                        original_order = chosen_original[
                            np.lexsort((job_ids[chosen_original], -scores[chosen_original]))
                        ]
                        emit(
                            writer, estimand="original", query_index=query_index,
                            group=group, n_relevant_original=len(relevant_original),
                            n_relevant_eval=len(relevant_original),
                            n_known_jobs_removed=len(eligible_original) - len(eligible_novel),
                            eligible_original=len(eligible_original),
                            eligible_eval=len(eligible_original), model=model,
                            strategy=strategy, seed=seed,
                            candidate_size=CANDIDATE_SIZE,
                            actual_candidate_size=len(chosen_original),
                            ranked_job_ids=job_ids[original_order],
                            relevant=relevant_original,
                        )
                        original_rows += 1
                        output_rows += 1

                        if relevant_novel:
                            if known_index_set:
                                novel_negative_order = np.fromiter(
                                    (
                                        idx for idx in negative_order
                                        if int(idx) not in known_index_set
                                    ),
                                    dtype=np.int32,
                                )
                            else:
                                novel_negative_order = negative_order
                            novel_take = min(
                                len(novel_negative_order),
                                max(0, CANDIDATE_SIZE - len(relevant_idx_novel)),
                            )
                            chosen_novel = np.concatenate(
                                [relevant_idx_novel, novel_negative_order[:novel_take]]
                            )
                            novel_order = chosen_novel[
                                np.lexsort((job_ids[chosen_novel], -scores[chosen_novel]))
                            ]
                            emit(
                                writer, estimand="novel_item", query_index=query_index,
                                group=group, n_relevant_original=len(relevant_original),
                                n_relevant_eval=len(relevant_novel),
                                n_known_jobs_removed=len(eligible_original) - len(eligible_novel),
                                eligible_original=len(eligible_original),
                                eligible_eval=len(eligible_novel), model=model,
                                strategy=strategy, seed=seed,
                                candidate_size=CANDIDATE_SIZE,
                                actual_candidate_size=len(chosen_novel),
                                ranked_job_ids=job_ids[novel_order],
                                relevant=relevant_novel,
                            )
                            novel_rows += 1
                            output_rows += 1

            if args.progress_every and position % args.progress_every == 0:
                print(
                    f"Processed {position:,}/{len(grouped):,} queries; "
                    f"rows={output_rows:,}", flush=True,
                )

    expected_original = len(grouped) * len(args.models) * (1 + len(STRATEGIES) * len(SEEDS))
    expected_novel = novel_queries * len(args.models) * (1 + len(STRATEGIES) * len(SEEDS))
    if original_rows != expected_original or novel_rows != expected_novel:
        raise ValueError(
            f"Unexpected row counts: original={original_rows}/{expected_original}, "
            f"novel={novel_rows}/{expected_novel}"
        )

    audit = {
        "purpose": "known-item/repeat-target sensitivity at K=20",
        "input_files_sha256": {
            "jobs.csv": sha256(jobs_path),
            "candidates.csv": sha256(candidates_path),
            "binary/train.csv": sha256(train_path),
            "binary/test.csv": sha256(test_path),
        },
        "raw_test_rows": int(len(test_raw)),
        "deduplicated_test_events": int(len(test)),
        "all_queries": int(len(grouped_all)),
        "evaluated_queries": int(len(grouped)),
        "unique_candidates": int(grouped.candidate_id.nunique()),
        "train_test_overlap_unique_candidate_job_pairs": overlap_pairs,
        "train_test_overlap_target_events": overlap_events,
        "train_test_overlap_queries": overlap_queries,
        "novel_item_queries": int(novel_queries),
        "repeat_only_queries_excluded": int(excluded_repeat_only_queries),
        "original_target_events": int(original_target_events),
        "novel_target_events": int(novel_target_events),
        "known_jobs_removed_per_query": {
            "mean": float(np.mean(removed_known_counts)),
            "median": float(np.median(removed_known_counts)),
            "min": int(np.min(removed_known_counts)),
            "max": int(np.max(removed_known_counts)),
            "queries_with_any_removed": int(np.count_nonzero(removed_known_counts)),
        },
        "models": list(args.models),
        "strategies": list(STRATEGIES),
        "seeds": list(SEEDS),
        "candidate_size": CANDIDATE_SIZE,
        "cutoff": K,
        "original_rows": int(original_rows),
        "novel_item_rows": int(novel_rows),
        "output_rows": int(output_rows),
        "output_file": str(output),
        "sampling_preservation_rule": (
            "Generate the original seeded negative order, remove training-seen jobs "
            "from that order, then take its earliest remaining negatives to refill "
            "the novel-item candidate set."
        ),
        "score_preservation_rule": (
            "All model score vectors use the original production definitions; only "
            "ranking eligibility and repeat-target relevance change."
        ),
        "tfidf_vocabulary_size": tfidf_vocabulary,
        "bm25_vocabulary_size": bm25_vocabulary,
        "bm25_average_document_length": bm25_avgdl,
        "semantic": semantic_meta,
        "static_profile_limitation": (
            "JTH entity text is static and is not verified as an as-of-query snapshot."
        ),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (args.out / "known_item_sensitivity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
