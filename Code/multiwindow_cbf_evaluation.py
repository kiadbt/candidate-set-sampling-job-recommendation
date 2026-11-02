"""Dataset-faithful CareerBuilder content-based evaluation.

Each CareerBuilder challenge partition is evaluated separately. Test
applications are relevance judgments only; user representations contain profile
attributes and employment-title history, and no user/job identity features.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SEED = 42
KS = (5, 10, 20, 150)
TOP_N = max(KS)


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def load_sources(dataset_dir: Path, window_id: int):
    window = dataset_dir / f"window_{window_id}"
    jobs = pd.read_csv(window / "jobs_test_available.csv", low_memory=False)
    users = pd.read_csv(window / "users.csv", low_memory=False)
    history = pd.read_csv(window / "user_history.csv", low_memory=False)
    applications = pd.read_csv(window / "apps.csv", low_memory=False)
    return jobs, users, history, applications


def prepare_window(all_jobs, all_users, all_history, all_apps, window_id: int):
    # Jobs were already restricted to those overlapping the official test period.
    jobs = all_jobs[all_jobs["WindowID"] == window_id].copy()
    jobs = jobs.drop_duplicates("JobID").reset_index(drop=True)

    users = all_users[
        (all_users["WindowID"] == window_id)
        & (all_users["Split"].str.lower() == "test")
    ].copy()
    history = all_history[
        (all_history["WindowID"] == window_id)
        & all_history["UserID"].isin(users["UserID"])
    ].copy()
    apps = all_apps[
        (all_apps["WindowID"] == window_id)
        & (all_apps["Split"].str.lower() == "test")
        & all_apps["UserID"].isin(users["UserID"])
    ].copy()

    # Use a common content-available cohort for controlled profile ablations.
    eligible_users = set(history["UserID"]).intersection(apps["UserID"])
    users = users[users["UserID"].isin(eligible_users)].copy()
    valid_jobs = set(jobs["JobID"])
    apps = apps[apps["JobID"].isin(valid_jobs)].copy()
    truth_series = apps.groupby("UserID")["JobID"].agg(lambda x: set(map(int, x)))
    users = users[users["UserID"].isin(truth_series.index)].copy()

    history["JobTitle"] = clean_text(history["JobTitle"])
    history = history.sort_values(["UserID", "Sequence"])
    history_all = history.groupby("UserID")["JobTitle"].agg(" ".join)
    # Official documentation: smaller Sequence values indicate more recent jobs.
    history_latest = history.groupby("UserID").head(1).set_index("UserID")["JobTitle"]

    users = users.set_index("UserID", drop=False)
    users["history_all"] = users.index.map(history_all).fillna("")
    users["history_latest"] = users.index.map(history_latest).fillna("")
    users["profile_structured"] = clean_text(
        "major " + clean_text(users["Major"])
        + " degree " + clean_text(users["DegreeType"])
        + " city " + clean_text(users["City"])
        + " state " + clean_text(users["State"])
        + " country " + clean_text(users["Country"])
    )
    users["profile_combined"] = clean_text(
        users["history_all"] + " " + users["profile_structured"]
    )

    for column in ("Title", "Description", "Requirements", "City", "State", "Country"):
        jobs[column] = clean_text(jobs[column])
    jobs["text_all"] = clean_text(
        jobs["Title"] + " " + jobs["Description"] + " " + jobs["Requirements"]
    )
    truth = {int(uid): truth_series.loc[uid] for uid in users.index}
    return jobs, users, apps, truth


def per_user_metrics(recommendations, user_ids, truth, model, window_id):
    rows = []
    for uid, ranked in zip(user_ids, recommendations):
        relevant = truth[int(uid)]
        for k in KS:
            top = ranked[:k]
            hits = np.fromiter((int(j) in relevant for j in top), dtype=np.int8)
            n_hits = int(hits.sum())
            discounts = 1.0 / np.log2(np.arange(2, k + 2))
            dcg = float((hits * discounts).sum())
            ideal = float(discounts[: min(len(relevant), k)].sum())
            positions = np.flatnonzero(hits)
            ap = 0.0
            if n_hits:
                cumulative = np.cumsum(hits)
                ap = float(((cumulative / np.arange(1, k + 1)) * hits).sum() / min(len(relevant), k))
            rows.append({
                "window": window_id, "model": model, "user_id": int(uid), "k": k,
                "precision": n_hits / k, "recall": n_hits / len(relevant),
                "hit_rate": float(n_hits > 0), "ndcg": dcg / ideal if ideal else 0.0,
                "mrr": 1.0 / (positions[0] + 1) if len(positions) else 0.0,
                "map": ap, "relevant_jobs": len(relevant),
            })
    return rows


def random_recommendations(job_ids, user_ids, window_id):
    return np.asarray([
        np.random.default_rng(SEED + window_id * 1_000_000 + int(uid)).choice(
            job_ids, size=TOP_N, replace=False
        )
        for uid in user_ids
    ])


def popularity_recommendations(all_apps, jobs, n_users, window_id):
    counts = all_apps[
        (all_apps["WindowID"] == window_id)
        & (all_apps["Split"].str.lower() == "train")
    ]["JobID"].value_counts()
    ranked = sorted(
        jobs["JobID"].astype(int), key=lambda jid: (-int(counts.get(jid, 0)), jid)
    )[:TOP_N]
    return np.tile(np.asarray(ranked, dtype=np.int64), (n_users, 1))


def rank_sparse(job_matrix, user_matrix, job_ids):
    """Rank a sparse job matrix with larger BLAS-friendly user batches."""
    job_t = sparse.csr_matrix(job_matrix.T)
    recommendations = np.empty((user_matrix.shape[0], TOP_N), dtype=np.int64)
    for start in range(0, user_matrix.shape[0], 128):
        stop = min(start + 128, user_matrix.shape[0])
        scores = (user_matrix[start:stop] @ job_t).toarray()
        candidate_idx = np.argpartition(scores, -TOP_N, axis=1)[:, -TOP_N:]
        candidate_scores = np.take_along_axis(scores, candidate_idx, axis=1)
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        ranked_idx = np.take_along_axis(candidate_idx, order, axis=1)
        recommendations[start:stop] = job_ids[ranked_idx]
    return recommendations


def fit_tfidf(jobs, job_field, max_features):
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2,
        max_df=0.98, max_features=max_features, sublinear_tf=True, dtype=np.float32,
    )
    job_matrix = normalize(vectorizer.fit_transform(jobs[job_field]), copy=False)
    return vectorizer, job_matrix


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--dataset-dir", type=Path, default=root / "Dataset" / "processed_raw_v1")
    parser.add_argument("--output-dir", type=Path, default=root / "Results" / "RawFaithful_CBF")
    parser.add_argument("--windows", type=int, nargs="+", default=list(range(1, 8)))
    parser.add_argument("--max-features", type=int, default=50_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_all_fields = {
        "tfidf_all_latest": "history_latest",
        "tfidf_all_history": "history_all",
        "tfidf_all_structured": "profile_structured",
        "tfidf_all_combined": "profile_combined",
    }
    metric_rows, summaries = [], []
    for window_id in args.windows:
        print(f"Preparing Window {window_id}", flush=True)
        all_jobs, all_users, all_history, all_apps = load_sources(args.dataset_dir, window_id)
        jobs, users, apps, truth = prepare_window(
            all_jobs, all_users, all_history, all_apps, window_id
        )
        user_ids = users["UserID"].to_numpy(dtype=np.int64)
        experiments = {
            "random": random_recommendations(jobs["JobID"].to_numpy(dtype=np.int64), user_ids, window_id),
            "popularity": popularity_recommendations(all_apps, jobs, len(users), window_id),
        }
        runtimes = {"random": 0.0, "popularity": 0.0}
        vocabularies = {}
        job_ids = jobs["JobID"].to_numpy(dtype=np.int64)
        print(f"Window {window_id}: fitting title TF-IDF", flush=True)
        started = time.perf_counter()
        title_vectorizer, title_jobs = fit_tfidf(jobs, "Title", args.max_features)
        title_users = normalize(title_vectorizer.transform(users["history_all"]), copy=False)
        experiments["tfidf_title_history"] = rank_sparse(title_jobs, title_users, job_ids)
        vocabularies["tfidf_title_history"] = len(title_vectorizer.vocabulary_)
        runtimes["tfidf_title_history"] = time.perf_counter() - started

        print(f"Window {window_id}: fitting full-text TF-IDF once", flush=True)
        family_started = time.perf_counter()
        all_vectorizer, all_jobs_matrix = fit_tfidf(jobs, "text_all", args.max_features)
        vocabulary_size = len(all_vectorizer.vocabulary_)
        for name, user_field in text_all_fields.items():
            print(f"Window {window_id}: ranking {name}", flush=True)
            started = time.perf_counter()
            user_matrix = normalize(all_vectorizer.transform(users[user_field]), copy=False)
            experiments[name] = rank_sparse(all_jobs_matrix, user_matrix, job_ids)
            vocabularies[name] = vocabulary_size
            runtimes[name] = time.perf_counter() - started
        runtimes["tfidf_all_family_total"] = time.perf_counter() - family_started
        for name, recommendations in experiments.items():
            metric_rows.extend(per_user_metrics(recommendations, user_ids, truth, name, window_id))
        counts = apps.groupby("UserID").size()
        summaries.append({
            "window": window_id, "catalog_jobs": len(jobs), "test_users": len(users),
            "held_out_applications": len(apps), "unique_relevant_jobs": apps["JobID"].nunique(),
            "applications_mean": float(counts.mean()), "applications_median": float(counts.median()),
            "applications_min": int(counts.min()), "applications_max": int(counts.max()),
            "vocabularies": vocabularies, "runtime_seconds": runtimes,
        })
        pd.DataFrame(metric_rows).to_csv(args.output_dir / "per_user_metrics.partial.csv", index=False)
        (args.output_dir / "window_summaries.partial.json").write_text(
            json.dumps(summaries, indent=2), encoding="utf-8"
        )

    per_user = pd.DataFrame(metric_rows)
    aggregate = per_user.groupby(["window", "model", "k"], as_index=False).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        hit_rate=("hit_rate", "mean"), ndcg=("ndcg", "mean"),
        mrr=("mrr", "mean"), map=("map", "mean"), users=("user_id", "nunique"),
    )
    cross_window = aggregate.groupby(["model", "k"], as_index=False).agg(
        precision_mean=("precision", "mean"), precision_sd=("precision", "std"),
        recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
        hit_rate_mean=("hit_rate", "mean"), hit_rate_sd=("hit_rate", "std"),
        ndcg_mean=("ndcg", "mean"), ndcg_sd=("ndcg", "std"),
        mrr_mean=("mrr", "mean"), mrr_sd=("mrr", "std"),
        map_mean=("map", "mean"), map_sd=("map", "std"), windows=("window", "nunique"),
    )
    pooled = per_user.groupby(["model", "k"], as_index=False).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        hit_rate=("hit_rate", "mean"), ndcg=("ndcg", "mean"),
        mrr=("mrr", "mean"), map=("map", "mean"),
        users=("user_id", "count"), partitions=("window", "nunique"),
    )
    per_user.to_csv(args.output_dir / "per_user_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "window_metrics.csv", index=False)
    cross_window.to_csv(args.output_dir / "cross_window_summary.csv", index=False)
    pooled.to_csv(args.output_dir / "pooled_user_metrics.csv", index=False)
    (args.output_dir / "window_summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(cross_window.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
