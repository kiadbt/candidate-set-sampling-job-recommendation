#!/usr/bin/env python3
"""Post-decision JTH temporal-catalog contamination sensitivity.

The reference catalog for a candidate-date query contains jobs whose released
``create_date`` is no later than the query timestamp.  The contaminated
catalog adds every job created after the query timestamp.  Queries, observed
relevant jobs, model parameters, model scores, and tie handling are identical
between regimes; only the set of rankable alternatives changes.

This sensitivity analysis measures changes
relative to the protocol-defined logged catalog, not latent preference or
deployment utility.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


SAFE_COMMON_FIELDS = [
    "skills", "expertise_area", "job_category", "contract_type", "zipcode",
    "llm_seniority_level", "llm_hard_skills", "llm_soft_skills",
    "llm_programming_languages", "llm_tools_technologies", "llm_certifications",
]
DEFAULT_MODELS = (
    "random", "popularity", "tfidf", "bm25", "semantic", "recency_pop",
    "item_knn",
)
KS = (5, 10, 20)
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path,
        default=root / "Dataset" / "external" / "JTH" / "data",
    )
    parser.add_argument(
        "--out", type=Path,
        default=root / "Results" / "Revision_R1_JTH_Temporal_Contamination",
    )
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS,
                        default=list(DEFAULT_MODELS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(KS))
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--semantic-model", default=SEMANTIC_MODEL)
    parser.add_argument(
        "--semantic-cache", type=Path,
        default=root / "Results" / "JTH_Semantic_Cache",
    )
    return parser.parse_args()


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def join_fields(frame: pd.DataFrame, fields: list[str]) -> pd.Series:
    present = [field for field in fields if field in frame.columns]
    if not present:
        return pd.Series([""] * len(frame), index=frame.index)
    cleaned = [
        frame[field].fillna("").astype(str).str.replace(";", " ", regex=False)
        for field in present
    ]
    text = cleaned[0]
    for column in cleaned[1:]:
        text = text + " " + column
    return text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def text_digest(texts: pd.Series, model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for value in texts:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def semantic_embeddings(
    texts: pd.Series, model_name: str, cache_dir: Path, label: str,
) -> tuple[np.ndarray, str, bool, Path]:
    digest = text_digest(texts, model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{label}_{digest[:16]}.npy"
    cache_hit = path.exists()
    if cache_hit:
        array = np.load(path)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
        model.max_seq_length = 256
        array = model.encode(
            texts.tolist(), batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        array = np.asarray(array, dtype=np.float32)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, array)
        temporary.replace(path)
    if array.ndim != 2 or array.shape[0] != len(texts) or not np.isfinite(array).all():
        raise ValueError(f"Invalid semantic embedding cache: {path}")
    return np.asarray(array, dtype=np.float32), digest, cache_hit, path


def query_seed(candidate_id: int, timestamp: pd.Timestamp) -> int:
    raw = f"0|{candidate_id}|{timestamp.date()}|model_random".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def metric_values(ranked_job_ids: np.ndarray, relevant: set[int], k: int) -> dict[str, float]:
    top = ranked_job_ids[:k]
    hits = np.fromiter((int(job_id) in relevant for job_id in top), dtype=np.int8)
    n_hits = int(hits.sum())
    precision = n_hits / k
    recall = n_hits / len(relevant)
    hit_rate = float(n_hits > 0)
    dcg = float(sum(value / math.log2(position + 2) for position, value in enumerate(hits)))
    ideal = sum(1 / math.log2(position + 2) for position in range(min(len(relevant), k)))
    average_precision = 0.0
    seen = 0
    for position, value in enumerate(hits, 1):
        if value:
            seen += 1
            average_precision += seen / position
    average_precision /= min(len(relevant), k)
    first = np.flatnonzero(hits)
    reciprocal_rank = 1.0 / (int(first[0]) + 1) if len(first) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "ndcg": dcg / ideal if ideal else 0.0,
        "map": average_precision,
        "mrr": reciprocal_rank,
    }


def deterministic_order(
    indices: np.ndarray, scores: np.ndarray, job_ids: np.ndarray,
) -> np.ndarray:
    return indices[np.lexsort((job_ids[indices], -scores[indices]))]


def software_versions() -> dict[str, object]:
    versions: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import torch
        import sentence_transformers

        versions["torch"] = torch.__version__
        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        versions["torch"] = None
        versions["sentence_transformers"] = None
    return versions


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluator_path = Path(__file__).resolve()
    models = tuple(dict.fromkeys(args.models))
    ks = tuple(sorted(set(args.ks)))

    source_paths = {
        "jobs": args.data / "jobs.csv",
        "candidates": args.data / "candidates.csv",
        "train": args.data / "binary" / "train.csv",
        "test": args.data / "binary" / "test.csv",
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} source: {path}")

    jobs = pd.read_csv(source_paths["jobs"])
    candidates = pd.read_csv(source_paths["candidates"])
    train = pd.read_csv(source_paths["train"])
    raw_test = pd.read_csv(source_paths["test"])
    jobs["create_date"] = pd.to_datetime(jobs["create_date"], errors="raise")
    candidates["create_date"] = pd.to_datetime(candidates["create_date"], errors="raise")
    train["timestamp"] = pd.to_datetime(train["timestamp"], errors="raise")
    raw_test["timestamp"] = pd.to_datetime(raw_test["timestamp"], errors="raise")

    exact_test = raw_test.drop_duplicates().copy()
    queries = (
        exact_test.groupby(["candidate_id", "timestamp"], sort=True)["job_id"]
        .agg(lambda values: sorted(set(map(int, values))))
        .reset_index(name="relevant_jobs")
    )
    if args.max_queries is not None:
        queries = queries.head(args.max_queries).copy()
    queries.insert(0, "query_index", np.arange(len(queries), dtype=np.int64))

    jobs = jobs.sort_values("job_id").reset_index(drop=True)
    candidates = candidates.set_index("candidate_id", drop=False)
    job_ids = jobs["job_id"].to_numpy(dtype=np.int64)
    if len(np.unique(job_ids)) != len(job_ids):
        raise ValueError("jobs.csv contains duplicate job_id values")
    job_to_index = {int(job_id): index for index, job_id in enumerate(job_ids)}
    job_dates = jobs["create_date"].to_numpy(dtype="datetime64[ns]")
    missing_targets = sorted({
        job_id for relevant in queries["relevant_jobs"] for job_id in relevant
        if job_id not in job_to_index
    })
    if missing_targets:
        raise ValueError(f"Targets missing from jobs.csv: {missing_targets[:10]}")
    future_targets = sum(
        job_dates[job_to_index[job_id]] > np.datetime64(row.timestamp)
        for row in queries.itertuples(index=False)
        for job_id in row.relevant_jobs
    )
    if future_targets:
        raise ValueError(f"{future_targets} relevant jobs were created after their query")

    query_candidate_ids = queries["candidate_id"].drop_duplicates().to_numpy(dtype=np.int64)
    missing_candidates = sorted(set(map(int, query_candidate_ids)) - set(map(int, candidates.index)))
    if missing_candidates:
        raise ValueError(f"Query candidates missing from candidates.csv: {missing_candidates[:10]}")
    candidate_rows = candidates.loc[query_candidate_ids].copy()
    candidate_to_row = {
        int(candidate_id): row for row, candidate_id in enumerate(candidate_rows.index)
    }
    query_candidate_rows = queries["candidate_id"].map(candidate_to_row).to_numpy(dtype=np.int64)
    query_dates = queries["timestamp"].to_numpy(dtype="datetime64[ns]")

    job_text = join_fields(jobs, SAFE_COMMON_FIELDS)
    candidate_text = join_fields(candidate_rows, SAFE_COMMON_FIELDS)
    popularity_counts = train["job_id"].value_counts().to_dict()
    popularity_scores = np.asarray(
        [popularity_counts.get(int(job_id), 0) for job_id in job_ids], dtype=np.float64,
    )
    train_candidates = set(map(int, train["candidate_id"].unique()))

    tfidf_jobs = tfidf_candidates = None
    tfidf_vocabulary_size = None
    if "tfidf" in models:
        vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.98, sublinear_tf=True, ngram_range=(1, 2),
            max_features=100_000, norm="l2", dtype=np.float32,
        )
        tfidf_jobs = sparse.csr_matrix(vectorizer.fit_transform(job_text))
        tfidf_candidates = sparse.csr_matrix(vectorizer.transform(candidate_text))
        tfidf_vocabulary_size = len(vectorizer.vocabulary_)

    bm25_jobs = bm25_candidates = None
    bm25_audit = None
    if "bm25" in models:
        vectorizer = CountVectorizer(
            min_df=2, max_df=0.98, ngram_range=(1, 2), max_features=100_000,
            dtype=np.float32,
        )
        counts = sparse.csr_matrix(vectorizer.fit_transform(job_text), dtype=np.float32)
        query_counts = sparse.csr_matrix(vectorizer.transform(candidate_text), dtype=np.float32)
        query_counts.data[:] = 1.0
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        inverse_document_frequency = np.log1p(
            (counts.shape[0] - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = float(lengths.mean())
        k1, b = 1.5, 0.75
        row_indices = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        denominator = counts.data + k1 * (1 - b + b * lengths[row_indices] / average_length)
        counts.data = counts.data * (k1 + 1) / denominator
        bm25_jobs = counts.multiply(inverse_document_frequency).tocsr()
        bm25_candidates = query_counts
        bm25_audit = {
            "vocabulary_size": len(vectorizer.vocabulary_), "k1": k1, "b": b,
            "average_document_length": average_length,
        }

    semantic_jobs = semantic_candidates = None
    semantic_audit = None
    if "semantic" in models:
        semantic_jobs, job_digest, job_cache_hit, job_cache = semantic_embeddings(
            job_text, args.semantic_model, args.semantic_cache, "jobs",
        )
        semantic_candidates, candidate_digest, candidate_cache_hit, candidate_cache = semantic_embeddings(
            candidate_text, args.semantic_model, args.semantic_cache, "candidates",
        )
        if semantic_jobs.shape[1] != semantic_candidates.shape[1]:
            raise ValueError("Semantic job and candidate dimensions differ")
        semantic_audit = {
            "model": args.semantic_model,
            "dimension": int(semantic_jobs.shape[1]),
            "normalized": True,
            "device": "cpu",
            "max_sequence_length": 256,
            "job_text_sha256": job_digest,
            "candidate_text_sha256": candidate_digest,
            "job_cache_hit": job_cache_hit,
            "candidate_cache_hit": candidate_cache_hit,
            "job_cache": str(job_cache.resolve()),
            "candidate_cache": str(candidate_cache.resolve()),
        }

    item_seen = item_similarity = item_candidate_row = None
    if "item_knn" in models:
        unique_interactions = train[["candidate_id", "job_id"]].drop_duplicates()
        interaction_candidates = np.sort(unique_interactions["candidate_id"].unique())
        item_candidate_row = {
            int(candidate_id): row for row, candidate_id in enumerate(interaction_candidates)
        }
        rows = unique_interactions["candidate_id"].map(item_candidate_row).to_numpy()
        mapped_jobs = unique_interactions["job_id"].map(job_to_index)
        valid = mapped_jobs.notna().to_numpy()
        columns = mapped_jobs[valid].to_numpy(dtype=np.int32)
        item_seen = sparse.csr_matrix(
            (np.ones(int(valid.sum()), dtype=np.float32), (rows[valid], columns)),
            shape=(len(interaction_candidates), len(job_ids)),
        )
        column_norm = np.sqrt(np.asarray(item_seen.sum(axis=0)).ravel())
        inverse_norm = np.divide(
            1.0, column_norm, out=np.zeros_like(column_norm), where=column_norm > 0,
        )
        normalized = item_seen.multiply(inverse_norm).tocsr()
        item_similarity = (normalized.T @ normalized).tocsr()

    metric_rows: list[dict[str, object]] = []
    catalog_rows: list[dict[str, object]] = []
    all_indices = np.arange(len(job_ids), dtype=np.int32)
    max_k = max(ks)
    for batch_start in range(0, len(queries), args.batch_size):
        batch_end = min(len(queries), batch_start + args.batch_size)
        batch = queries.iloc[batch_start:batch_end]
        batch_candidate_rows = query_candidate_rows[batch_start:batch_end]

        batch_scores: dict[str, np.ndarray] = {}
        if "tfidf" in models:
            batch_scores["tfidf"] = (
                tfidf_candidates[batch_candidate_rows] @ tfidf_jobs.T
            ).toarray().astype(np.float32, copy=False)
        if "bm25" in models:
            batch_scores["bm25"] = (
                bm25_candidates[batch_candidate_rows] @ bm25_jobs.T
            ).toarray().astype(np.float32, copy=False)
        for local_index, query in enumerate(batch.itertuples(index=False)):
            query_index = int(query.query_index)
            candidate_id = int(query.candidate_id)
            timestamp = pd.Timestamp(query.timestamp)
            relevant = set(map(int, query.relevant_jobs))
            reference_indices = np.flatnonzero(job_dates <= query_dates[query_index]).astype(np.int32)
            future_count = len(job_ids) - len(reference_indices)
            catalog_rows.append({
                "query_index": query_index,
                "candidate_id": candidate_id,
                "timestamp": timestamp.date().isoformat(),
                "n_relevant": len(relevant),
                "reference_catalog_size": len(reference_indices),
                "future_jobs_added": future_count,
                "contaminated_catalog_size": len(job_ids),
                "relative_catalog_increase": future_count / len(reference_indices),
            })

            score_by_model: dict[str, np.ndarray] = {}
            if "random" in models:
                score_by_model["random"] = np.random.default_rng(
                    query_seed(candidate_id, timestamp)
                ).random(len(job_ids))
            if "popularity" in models:
                score_by_model["popularity"] = popularity_scores
            for model in ("tfidf", "bm25", "semantic"):
                if model in batch_scores:
                    score_by_model[model] = batch_scores[model][local_index]
            if "semantic" in models:
                # Preserve the submitted evaluator's exact floating-point
                # operation order.  The algebraically equivalent batched
                # candidate-by-job product can change a near-tied rank because
                # BLAS accumulation differs.  Job-matrix-by-candidate-vector
                # reproduces the locked reference metrics exactly.
                score_by_model["semantic"] = (
                    semantic_jobs @ semantic_candidates[batch_candidate_rows[local_index]]
                )
            if "recency_pop" in models:
                # Preserve the original non-negative age convention.  Future
                # jobs therefore receive age zero under contamination.
                ages = np.maximum(
                    0.0, (query_dates[query_index] - job_dates) / np.timedelta64(1, "D")
                )
                score_by_model["recency_pop"] = popularity_scores * np.exp(-0.02 * ages)
            if "item_knn" in models:
                candidate_row = item_candidate_row.get(candidate_id)
                if candidate_row is None:
                    score_by_model["item_knn"] = np.zeros(len(job_ids), dtype=np.float32)
                else:
                    seen_indices = item_seen[candidate_row].indices
                    score_by_model["item_knn"] = np.asarray(
                        item_similarity[:, seen_indices].sum(axis=1)
                    ).ravel()

            for model in models:
                scores = score_by_model[model]
                for regime, eligible in (
                    ("created_by_query_date", reference_indices),
                    ("future_contaminated_all_jobs", all_indices),
                ):
                    order = deterministic_order(eligible, scores, job_ids)
                    ranked = job_ids[order[:max_k]]
                    for k in ks:
                        values = metric_values(ranked, relevant, k)
                        metric_rows.append({
                            "query_index": query_index,
                            "candidate_id": candidate_id,
                            "timestamp": timestamp.date().isoformat(),
                            "group": "warm" if candidate_id in train_candidates else "cold",
                            "n_relevant": len(relevant),
                            "catalog_regime": regime,
                            "candidate_count": len(eligible),
                            "future_jobs_added": future_count if regime.startswith("future_") else 0,
                            "model": model,
                            "k": k,
                            **values,
                        })
        print(f"Processed {batch_end:,}/{len(queries):,} queries", flush=True)

    per_query = pd.DataFrame(metric_rows)
    catalog_delta = pd.DataFrame(catalog_rows)
    metric_columns = ["precision", "recall", "hit_rate", "ndcg", "map", "mrr"]
    summary = (
        pd.concat([per_query, per_query.assign(group="all")], ignore_index=True)
        .groupby(["group", "catalog_regime", "model", "k"], as_index=False)
        .agg(
            queries=("query_index", "size"),
            candidate_count_mean=("candidate_count", "mean"),
            future_jobs_added_mean=("future_jobs_added", "mean"),
            **{metric: (metric, "mean") for metric in metric_columns},
        )
    )

    paired = per_query.pivot(
        index=["query_index", "candidate_id", "timestamp", "group", "n_relevant", "model", "k"],
        columns="catalog_regime", values=metric_columns,
    )
    paired.columns = [f"{metric}__{regime}" for metric, regime in paired.columns]
    paired = paired.reset_index()
    for metric in metric_columns:
        paired[f"{metric}_change_contaminated_minus_reference"] = (
            paired[f"{metric}__future_contaminated_all_jobs"]
            - paired[f"{metric}__created_by_query_date"]
        )

    rankings: list[dict[str, object]] = []
    all_summary = summary[summary["group"].eq("all")]
    for k in ks:
        for metric in metric_columns:
            regime_orders: dict[str, list[str]] = {}
            for regime in ("created_by_query_date", "future_contaminated_all_jobs"):
                cell = all_summary[
                    all_summary["catalog_regime"].eq(regime) & all_summary["k"].eq(k)
                ].sort_values([metric, "model"], ascending=[False, True])
                regime_orders[regime] = cell["model"].tolist()
                for rank, row in enumerate(cell.itertuples(index=False), 1):
                    rankings.append({
                        "record_type": "model_rank",
                        "k": k,
                        "metric": metric,
                        "catalog_regime": regime,
                        "model": row.model,
                        "rank": rank,
                        "value": getattr(row, metric),
                        "kendall_tau_between_regimes": np.nan,
                        "top_model_agreement": np.nan,
                    })
            reference_order = regime_orders["created_by_query_date"]
            contaminated_order = regime_orders["future_contaminated_all_jobs"]
            reference_ranks = {model: rank for rank, model in enumerate(reference_order)}
            contaminated_ranks = [reference_ranks[model] for model in contaminated_order]
            tau = float(kendalltau(np.arange(len(models)), contaminated_ranks).statistic)
            rankings.append({
                "record_type": "regime_comparison",
                "k": k,
                "metric": metric,
                "catalog_regime": "reference_vs_contaminated",
                "model": "all",
                "rank": np.nan,
                "value": np.nan,
                "kendall_tau_between_regimes": tau,
                "top_model_agreement": reference_order[0] == contaminated_order[0],
            })

    output_paths = {
        "per_query": args.out / "jth_temporal_contamination_per_query.csv.gz",
        "catalog_delta": args.out / "jth_temporal_catalog_delta_by_query.csv.gz",
        "summary": args.out / "jth_temporal_contamination_summary.csv",
        "paired": args.out / "jth_temporal_contamination_paired_changes.csv.gz",
        "rankings": args.out / "jth_temporal_contamination_model_rankings.csv",
    }
    with gzip.open(output_paths["per_query"], "wt", encoding="utf-8", newline="") as handle:
        per_query.to_csv(handle, index=False)
    with gzip.open(output_paths["catalog_delta"], "wt", encoding="utf-8", newline="") as handle:
        catalog_delta.to_csv(handle, index=False)
    summary.to_csv(output_paths["summary"], index=False)
    with gzip.open(output_paths["paired"], "wt", encoding="utf-8", newline="") as handle:
        paired.to_csv(handle, index=False)
    pd.DataFrame(rankings).to_csv(output_paths["rankings"], index=False)

    audit = {
        "analysis_role": "post_decision_sensitivity",
        "code_provenance": {
            "path": str(evaluator_path),
            "bytes": evaluator_path.stat().st_size,
            "sha256": sha256(evaluator_path),
        },
        "estimand": (
            "Change in logged-reference ranking metrics when future-created jobs are added "
            "to the protocol-defined created-by-query-date catalog."
        ),
        "invariants": {
            "queries_fixed": True,
            "relevant_sets_fixed": True,
            "model_scores_fixed_between_catalog_regimes": True,
            "tie_break_fixed": "descending score, then ascending job_id",
            "only_candidate_eligibility_changes": True,
        },
        "not_claimed": ["latent preference validity", "deployment utility", "causal effect"],
        "source_files": {
            label: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for label, path in source_paths.items()
        },
        "raw_test_rows": int(len(raw_test)),
        "exact_deduplicated_test_rows": int(len(exact_test)),
        "queries": int(len(queries)),
        "multi_relevance_queries": int((queries["relevant_jobs"].str.len() > 1).sum()),
        "jobs": int(len(jobs)),
        "models": list(models),
        "cutoffs": list(ks),
        "safe_common_fields": SAFE_COMMON_FIELDS,
        "sensitive_fields_excluded": ["llm_sex", "llm_nationality", "llm_age_bucket"],
        "tfidf_vocabulary_size": tfidf_vocabulary_size,
        "bm25": bm25_audit,
        "semantic": semantic_audit,
        "catalog_delta": {
            "reference_size_mean": float(catalog_delta["reference_catalog_size"].mean()),
            "reference_size_median": float(catalog_delta["reference_catalog_size"].median()),
            "reference_size_min": int(catalog_delta["reference_catalog_size"].min()),
            "reference_size_max": int(catalog_delta["reference_catalog_size"].max()),
            "future_jobs_added_mean": float(catalog_delta["future_jobs_added"].mean()),
            "future_jobs_added_median": float(catalog_delta["future_jobs_added"].median()),
            "future_jobs_added_min": int(catalog_delta["future_jobs_added"].min()),
            "future_jobs_added_max": int(catalog_delta["future_jobs_added"].max()),
            "queries_with_future_jobs_added": int((catalog_delta["future_jobs_added"] > 0).sum()),
        },
        "software": software_versions(),
        "elapsed_seconds": round(time.time() - started, 3),
        "outputs": {
            label: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for label, path in output_paths.items()
        },
    }
    audit_path = args.out / "jth_temporal_contamination_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
