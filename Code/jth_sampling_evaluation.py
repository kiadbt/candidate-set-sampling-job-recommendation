#!/usr/bin/env python3
"""Candidate-set validity experiment for the JTH candidate-to-job task.

The official JTH evaluator keys predictions by (candidate_id, timestamp).  The
released test split contains repeated keys, so this implementation groups all
jobs observed for the same candidate and day into one multi-relevance query.
This prevents silent overwriting and makes MAP/recall well-defined.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


SAFE_COMMON_FIELDS = [
    "skills", "expertise_area", "job_category", "contract_type", "zipcode",
    "llm_seniority_level", "llm_hard_skills", "llm_soft_skills",
    "llm_programming_languages", "llm_tools_technologies", "llm_certifications",
]
KS = (1, 5, 10, 20)
SIZES = (100, 500, 1000, 2500, 5000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("Dataset/external/JTH/data"))
    p.add_argument("--out", type=Path, default=Path("Results/JTH_EvaluationValidity"))
    p.add_argument("--models", nargs="+", default=["random", "popularity", "tfidf"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    p.add_argument("--max-queries", type=int, default=None,
                   help="Chronological prefix for a smoke test; omit for full test.")
    p.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--semantic-cache", type=Path, default=Path("Results/JTH_Semantic_Cache"))
    return p.parse_args()


def join_fields(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    present = [f for f in fields if f in df.columns]
    cleaned = [df[f].fillna("").astype(str).str.replace(";", " ", regex=False)
               for f in present]
    if not cleaned:
        return pd.Series([""] * len(df), index=df.index)
    out = cleaned[0]
    for col in cleaned[1:]:
        out = out + " " + col
    return out.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def query_seed(seed: int, candidate_id: int, timestamp: pd.Timestamp, strategy: str) -> int:
    raw = f"{seed}|{candidate_id}|{timestamp.date()}|{strategy}".encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def text_digest(texts: pd.Series, model_name: str) -> str:
    h = hashlib.sha256(model_name.encode("utf-8"))
    for value in texts:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def semantic_embeddings(texts: pd.Series, model_name: str, cache_dir: Path,
                        label: str) -> tuple[np.ndarray, str, bool]:
    digest = text_digest(texts, model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{label}_{digest[:16]}.npy"
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] != len(texts) or arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError(f"Invalid semantic cache: {path}")
        return arr, digest, True
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
    arr = model.encode(texts.tolist(), batch_size=64, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] != len(texts) or not np.isfinite(arr).all():
        raise ValueError("Semantic encoder returned invalid embeddings")
    np.save(path, arr)
    return arr, digest, False


def metric_row(ranked: np.ndarray, relevant: set[int], k: int) -> dict[str, float]:
    top = ranked[:k]
    hits = np.fromiter((int(x) in relevant for x in top), dtype=np.int8)
    nh = int(hits.sum())
    precision = nh / k
    recall = nh / len(relevant)
    hit = float(nh > 0)
    dcg = float(sum(v / math.log2(i + 2) for i, v in enumerate(hits)))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    ndcg = dcg / ideal if ideal else 0.0
    ap_sum = 0.0
    seen = 0
    for i, v in enumerate(hits, 1):
        if v:
            seen += 1
            ap_sum += seen / i
    ap = ap_sum / min(len(relevant), k)
    rr = 1.0 / (int(np.flatnonzero(hits)[0]) + 1) if nh else 0.0
    return {"precision": precision, "recall": recall, "hitrate": hit,
            "ndcg": ndcg, "map": ap, "mrr": rr}


def sampled_negative_order(eligible: np.ndarray, relevant_idx: np.ndarray,
                           strategy: str, rng: np.random.Generator,
                           pop_weights: np.ndarray) -> np.ndarray:
    """Draw one order so candidate sets are nested within query/seed/strategy."""
    relset = set(relevant_idx.tolist())
    negatives = np.fromiter((x for x in eligible if int(x) not in relset), dtype=np.int32)
    if strategy == "uniform":
        return rng.permutation(negatives)
    # Plackett-Luce weighted ordering via exponential races. This is an exact
    # weighted-without-replacement ordering and avoids five independent draws.
    weights = pop_weights[negatives].astype(np.float64)
    keys = -np.log(np.maximum(rng.random(len(negatives)), np.finfo(float).tiny)) / weights
    return negatives[np.argsort(keys, kind="stable")]


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = pd.read_csv(args.data / "jobs.csv")
    candidates = pd.read_csv(args.data / "candidates.csv")
    train = pd.read_csv(args.data / "binary" / "train.csv")
    test_raw = pd.read_csv(args.data / "binary" / "test.csv")
    jobs["create_date"] = pd.to_datetime(jobs["create_date"])
    candidates["create_date"] = pd.to_datetime(candidates["create_date"])
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test_raw["timestamp"] = pd.to_datetime(test_raw["timestamp"])

    audit = {
        "raw_test_rows": int(len(test_raw)),
        "duplicate_exact_test_rows": int(test_raw.duplicated().sum()),
        "duplicate_candidate_date_rows": int(test_raw.duplicated(["candidate_id", "timestamp"]).sum()),
    }
    # Exact duplicated records provide no additional relevance information.
    test = test_raw.drop_duplicates().copy()
    grouped = (test.groupby(["candidate_id", "timestamp"], sort=True)["job_id"]
               .agg(lambda x: sorted(set(map(int, x)))).reset_index(name="relevant_jobs"))
    audit["all_unique_candidate_date_queries"] = int(len(grouped))
    audit["all_multi_relevance_queries"] = int((grouped.relevant_jobs.str.len() > 1).sum())
    if args.max_queries:
        grouped = grouped.head(args.max_queries).copy()
    audit["evaluated_candidate_date_queries"] = int(len(grouped))
    audit["unique_test_events_after_exact_deduplication"] = int(len(test))
    audit["multi_relevance_queries"] = int((grouped.relevant_jobs.str.len() > 1).sum())

    jobs = jobs.sort_values("job_id").reset_index(drop=True)
    candidates = candidates.set_index("candidate_id", drop=False)
    job_ids = jobs.job_id.to_numpy(dtype=np.int64)
    jid_to_idx = {int(j): i for i, j in enumerate(job_ids)}
    job_dates = jobs.create_date.to_numpy(dtype="datetime64[ns]")
    train_candidates = set(map(int, train.candidate_id.unique()))
    pop = train.job_id.value_counts().to_dict()
    pop_scores = np.array([pop.get(int(j), 0) for j in job_ids], dtype=np.float64)
    # Add-one smoothing gives every eligible job non-zero sampling probability.
    pop_weights = pop_scores + 1.0

    unknown_targets = sorted({j for xs in grouped.relevant_jobs for j in xs if j not in jid_to_idx})
    if unknown_targets:
        raise ValueError(f"Test targets absent from jobs.csv: {unknown_targets[:10]}")
    target_after_query = 0
    for row in grouped.itertuples(index=False):
        target_after_query += sum(job_dates[jid_to_idx[j]] > np.datetime64(row.timestamp)
                                  for j in row.relevant_jobs)
    if target_after_query:
        raise ValueError(f"{target_after_query} targets were created after their query date")

    candidate_rows = candidates.loc[grouped.candidate_id.unique()]
    candidate_vector_row = {int(cid): i for i, cid in enumerate(candidate_rows.index)}
    job_text = join_fields(jobs, SAFE_COMMON_FIELDS)
    candidate_text = join_fields(candidate_rows, SAFE_COMMON_FIELDS)

    tfidf_jobs = tfidf_candidates = None
    if "tfidf" in args.models:
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.98, sublinear_tf=True,
                                     ngram_range=(1, 2), max_features=100_000,
                                     norm="l2", dtype=np.float32)
        tfidf_jobs = vectorizer.fit_transform(job_text)
        tfidf_candidates = vectorizer.transform(candidate_text)
        tfidf_candidates = sparse.csr_matrix(tfidf_candidates)
        audit["tfidf_vocabulary_size"] = int(len(vectorizer.vocabulary_))

    bm25_jobs = bm25_candidates = None
    if "bm25" in args.models:
        cv = CountVectorizer(min_df=2, max_df=0.98, ngram_range=(1, 2),
                             max_features=100_000, dtype=np.float32)
        counts = sparse.csr_matrix(cv.fit_transform(job_text), dtype=np.float32)
        qcounts = sparse.csr_matrix(cv.transform(candidate_text), dtype=np.float32)
        qcounts.data[:] = 1.0
        n_docs = counts.shape[0]
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        avgdl = float(lengths.mean())
        k1, b = 1.5, 0.75
        rows_idx = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
        denom = counts.data + k1 * (1 - b + b * lengths[rows_idx] / avgdl)
        counts.data = counts.data * (k1 + 1) / denom
        bm25_jobs = counts.multiply(idf).tocsr()
        bm25_candidates = qcounts
        audit["bm25_vocabulary_size"] = int(len(cv.vocabulary_))
        audit["bm25_parameters"] = {"k1": k1, "b": b}

    semantic_jobs = semantic_candidates = None
    if "semantic" in args.models:
        semantic_jobs, job_digest, job_cache_hit = semantic_embeddings(
            job_text, args.semantic_model, args.semantic_cache, "jobs")
        semantic_candidates, candidate_digest, candidate_cache_hit = semantic_embeddings(
            candidate_text, args.semantic_model, args.semantic_cache, "candidates")
        if semantic_jobs.shape[1] != semantic_candidates.shape[1]:
            raise ValueError("Job and candidate semantic dimensions differ")
        audit["semantic"] = {
            "model": args.semantic_model,
            "dimension": int(semantic_jobs.shape[1]),
            "normalized": True,
            "device": "cpu",
            "job_text_sha256": job_digest,
            "candidate_text_sha256": candidate_digest,
            "job_cache_hit": job_cache_hit,
            "candidate_cache_hit": candidate_cache_hit,
        }

    cf_seen = cf_item_similarity = cf_candidate_row = None
    if "item_knn" in args.models:
        train_unique = train[["candidate_id", "job_id"]].drop_duplicates()
        cf_candidates = np.sort(train_unique.candidate_id.unique())
        cf_candidate_row = {int(cid): i for i, cid in enumerate(cf_candidates)}
        rr = train_unique.candidate_id.map(cf_candidate_row).to_numpy()
        cc = train_unique.job_id.map(jid_to_idx).to_numpy()
        valid = ~pd.isna(cc)
        interaction = sparse.csr_matrix(
            (np.ones(int(valid.sum()), dtype=np.float32),
             (rr[valid], cc[valid].astype(np.int32))),
            shape=(len(cf_candidates), len(job_ids)))
        col_norm = np.sqrt(np.asarray(interaction.sum(axis=0)).ravel())
        inv_norm = np.divide(1.0, col_norm, out=np.zeros_like(col_norm), where=col_norm > 0)
        normalized = interaction.multiply(inv_norm).tocsr()
        cf_seen = interaction
        cf_item_similarity = (normalized.T @ normalized).tocsr()
        audit["item_knn_train_candidates"] = int(len(cf_candidates))
        audit["item_knn_unique_interactions"] = int(interaction.nnz)
        audit["item_knn_similarity_nonzeros"] = int(cf_item_similarity.nnz)

    rows: list[dict] = []
    for qi, q in enumerate(grouped.itertuples(index=False)):
        cid, ts, relevant_jobs = int(q.candidate_id), q.timestamp, set(q.relevant_jobs)
        eligible = np.flatnonzero(job_dates <= np.datetime64(ts)).astype(np.int32)
        relevant_idx = np.array([jid_to_idx[j] for j in relevant_jobs], dtype=np.int32)
        group = "warm" if cid in train_candidates else "cold"
        score_by_model: dict[str, np.ndarray] = {}
        if "random" in args.models:
            score_by_model["random"] = np.random.default_rng(
                query_seed(0, cid, ts, "model_random")).random(len(job_ids))
        if "popularity" in args.models:
            score_by_model["popularity"] = pop_scores
        if "tfidf" in args.models:
            score_by_model["tfidf"] = (tfidf_jobs @ tfidf_candidates[candidate_vector_row[cid]].T).toarray().ravel()
        if "bm25" in args.models:
            score_by_model["bm25"] = (bm25_jobs @ bm25_candidates[candidate_vector_row[cid]].T).toarray().ravel()
        if "semantic" in args.models:
            score_by_model["semantic"] = semantic_jobs @ semantic_candidates[candidate_vector_row[cid]]
        if "recency_pop" in args.models:
            age_days = np.maximum(0.0, (np.datetime64(ts) - job_dates) / np.timedelta64(1, "D"))
            score_by_model["recency_pop"] = pop_scores * np.exp(-0.02 * age_days)
        if "item_knn" in args.models:
            row = cf_candidate_row.get(cid)
            if row is None:
                score_by_model["item_knn"] = np.zeros(len(job_ids), dtype=np.float32)
            else:
                seen_idx = cf_seen[row].indices
                score_by_model["item_knn"] = np.asarray(
                    cf_item_similarity[:, seen_idx].sum(axis=1)).ravel()

        for model, scores in score_by_model.items():
            # Stable job-id tie break via stable sort over job-id-ordered rows.
            full_order = eligible[np.lexsort((job_ids[eligible], -scores[eligible]))]
            for k in KS:
                vals = metric_row(job_ids[full_order], relevant_jobs, k)
                rows.append({"query_index": qi, "candidate_id": cid, "timestamp": ts.date().isoformat(),
                             "group": group, "n_relevant": len(relevant_jobs),
                             "eligible_catalog": len(eligible), "model": model,
                             "strategy": "full", "seed": 0, "candidate_size": -1,
                             "k": k, **vals})
            for strategy in ("uniform", "popularity_weighted"):
                for seed in args.seeds:
                    rng = np.random.default_rng(query_seed(seed, cid, ts, strategy))
                    negative_order = sampled_negative_order(
                        eligible, relevant_idx, strategy, rng, pop_weights)
                    for size in sorted(set(args.sizes)):
                        n_take = min(len(negative_order), max(0, size - len(relevant_idx)))
                        chosen = (eligible if size >= len(eligible) else
                                  np.concatenate([relevant_idx, negative_order[:n_take]]))
                        order = chosen[np.lexsort((job_ids[chosen], -scores[chosen]))]
                        for k in KS:
                            vals = metric_row(job_ids[order], relevant_jobs, k)
                            rows.append({"query_index": qi, "candidate_id": cid, "timestamp": ts.date().isoformat(),
                                         "group": group, "n_relevant": len(relevant_jobs),
                                         "eligible_catalog": len(eligible), "model": model,
                                         "strategy": strategy, "seed": seed,
                                         "candidate_size": min(size, len(eligible)), "k": k, **vals})
        if (qi + 1) % 250 == 0:
            print(f"Processed {qi + 1}/{len(grouped)} queries", flush=True)

    per_query = pd.DataFrame(rows)
    with gzip.open(args.out / "per_query_metrics.csv.gz", "wt", encoding="utf-8", newline="") as fh:
        per_query.to_csv(fh, index=False)
    metric_cols = ["precision", "recall", "hitrate", "ndcg", "map", "mrr"]
    summary_source = pd.concat(
        [per_query, per_query.assign(group="all")], ignore_index=True)
    summary = (summary_source.groupby(["group", "model", "strategy", "seed", "candidate_size", "k"],
                                 as_index=False)[metric_cols].mean())
    summary.to_csv(args.out / "summary_metrics.csv", index=False)
    audit.update({
        "models": list(args.models), "seeds": list(args.seeds), "candidate_sizes": list(args.sizes),
        "metrics_at": list(KS), "safe_profile_fields": SAFE_COMMON_FIELDS,
        "sensitive_fields_excluded": ["llm_sex", "llm_nationality", "llm_age_bucket"],
        "elapsed_seconds": round(time.time() - started, 3),
        "output_rows": int(len(per_query)),
    })
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
