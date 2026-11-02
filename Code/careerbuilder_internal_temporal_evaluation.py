#!/usr/bin/env python3
"""CareerBuilder internal-temporal full- and sampled-catalog evaluation.

This program evaluates an explicitly *internal* temporal task derived from the
released nine-day application log.  It is not the CareerBuilder competition
evaluation and must never be described as official MAP.  Prepared inputs are
created by ``prepare_careerbuilder_internal_temporal.py``.

The primary protocol is First7Last2.  The provider's ``Split`` field is absent
from every input read here.  User representations are released static content
only; pre-cutoff applications define popularity and candidate exclusions, not
lexical or semantic query enrichment.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


MODELS = (
    "random_hash",
    "popularity_precut",
    "tfidf_combined_static",
    "bm25_full_static",
    "bm25_title_static",
    "minilm_title_static",
)
STRATEGIES = ("uniform", "popularity_weighted")
DEFAULT_SIZES = (100, 500, 1000, 5000, 10000)
DEFAULT_KS = (5, 10, 20, 150)
MASTER_SAMPLING_SEED = 20261020
RANDOM_MODEL_SEED = np.uint64(0x6A09E667F3BCC909)
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
METRICS = ("precision", "recall", "hit_rate", "ndcg", "mrr", "map")
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_EXPECTED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path,
                   default=root / "Dataset" / "CareerBuilder_InternalTemporal_v1")
    p.add_argument("--protocol", default="First7Last2")
    p.add_argument("--output-root", type=Path,
                   default=root / "Results" / "CareerBuilder_InternalTemporal_EvaluationValidity")
    p.add_argument("--run-label", default="Primary_First7Last2")
    p.add_argument("--windows", type=int, nargs="+", default=list(range(1, 8)))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                   default=list(STRATEGIES))
    p.add_argument("--candidate-sizes", type=int, nargs="+",
                   default=list(DEFAULT_SIZES))
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--max-features", type=int, default=50_000)
    p.add_argument("--query-batch-size", type=int, default=16)
    p.add_argument("--checkpoint-users", type=int, default=50)
    p.add_argument("--semantic-model",
                   default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--embedding-batch-size", type=int, default=512)
    p.add_argument("--smoke-users-per-activity", type=int, default=None,
                   help="Select up to N audit users per activity stratum.")
    p.add_argument("--skip-semantic", action="store_true",
                   help="Engineering-only diagnostic; not valid for production.")
    p.add_argument("--authoritative-production", action="store_true",
                   help="Fail closed unless the complete frozen design for --study-role is requested.")
    p.add_argument("--study-role", choices=["primary", "temporal_sensitivity"],
                   default="primary")
    return p.parse_args()


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def stable_text_hash(texts: Iterable[str], role: str, model: str) -> str:
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(b"\0")
    digest.update(model.encode("utf-8"))
    digest.update(b"\0")
    for text in texts:
        encoded = str(text).encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def resolve_minilm_snapshot(model_name: str) -> dict[str, object]:
    """Fail closed unless the frozen local MiniLM snapshot is available."""
    if model_name != MINILM_MODEL:
        raise RuntimeError(
            f"Frozen evaluator requires {MINILM_MODEL}; received {model_name}"
        )
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repository = hf_home / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    ref_path = repository / "refs" / "main"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing frozen MiniLM ref: {ref_path}")
    resolved_ref = ref_path.read_text(encoding="utf-8").strip()
    if resolved_ref != MINILM_EXPECTED_REVISION:
        raise RuntimeError(
            f"MiniLM refs/main drift: expected {MINILM_EXPECTED_REVISION}, "
            f"found {resolved_ref}"
        )
    snapshot_path = repository / "snapshots" / resolved_ref
    if not snapshot_path.is_dir():
        raise FileNotFoundError(f"Missing frozen MiniLM snapshot: {snapshot_path}")
    digest = hashlib.sha256()
    files = [path for path in snapshot_path.rglob("*") if path.is_file()]
    total_bytes = 0
    for path in sorted(files, key=lambda value: value.as_posix()):
        relative = path.relative_to(snapshot_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        file_size = path.stat().st_size
        total_bytes += file_size
        digest.update(file_size.to_bytes(8, "little"))
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
    available = sorted(
        path.name for path in (repository / "snapshots").iterdir() if path.is_dir()
    )
    return {
        "model": model_name, "expected_revision": MINILM_EXPECTED_REVISION,
        "resolved_ref": resolved_ref, "ref_path": str(ref_path.resolve()),
        "ref_file_sha256": sha256(ref_path),
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_tree_sha256": digest.hexdigest(),
        "snapshot_files": len(files), "snapshot_bytes": total_bytes,
        "available_local_snapshots": available, "local_files_only": True,
    }


def clean(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        if os.name != "nt":
            return 0
        # Dependency-free Windows working-set measurement.
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        get_current = ctypes.windll.kernel32.GetCurrentProcess
        get_current.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD,
        ]
        get_memory.restype = wintypes.BOOL
        handle = get_current()
        ok = get_memory(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else 0


def software_versions() -> dict[str, object]:
    packages = {
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
    }
    for distribution in ("sentence-transformers", "transformers", "torch"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(), "packages": packages,
        "thread_environment": {
            key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TOKENIZERS_PARALLELISM",
            )
        },
    }


def splitmix64_scores(job_ids: np.ndarray, user_id: int, window: int) -> np.ndarray:
    """Stable content-independent score per (window, user, job)."""
    with np.errstate(over="ignore"):
        user_key = np.uint64(int(user_id)) * np.uint64(0xD1342543DE82EF95)
        window_key = np.uint64(window) * np.uint64(0x9E3779B97F4A7C15)
        x = job_ids.astype(np.uint64, copy=False) ^ user_key ^ window_key ^ RANDOM_MODEL_SEED
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & MASK64
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
        hashed = x ^ (x >> np.uint64(31))
        # Convert the high 53 bits to an exactly representable uniform float.
        # Returning uint64 would make unary negation wrap during descending sort.
        return (hashed >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def top_with_deterministic_ties(
    job_ids: np.ndarray, scores: np.ndarray, depth: int
) -> np.ndarray:
    depth = min(int(depth), len(job_ids))
    if depth <= 0:
        return np.empty(0, dtype=np.int64)
    if depth == len(job_ids):
        chosen = np.arange(len(job_ids))
    else:
        threshold = np.partition(scores, len(scores) - depth)[len(scores) - depth]
        above = np.flatnonzero(scores > threshold)
        ties = np.flatnonzero(scores == threshold)
        needed = depth - len(above)
        if needed < len(ties):
            ties = ties[np.argsort(job_ids[ties], kind="stable")[:needed]]
        chosen = np.concatenate((above, ties))
    order = np.lexsort((job_ids[chosen], -scores[chosen]))
    return job_ids[chosen[order]][:depth]


def metric_rows(ranked: np.ndarray, relevant: set[int], ks: tuple[int, ...]):
    for k in ks:
        effective = min(k, len(ranked))
        top = ranked[:effective]
        hits = np.fromiter((int(x) in relevant for x in top), dtype=np.int8,
                           count=effective)
        n_hits = int(hits.sum())
        discounts = 1.0 / np.log2(np.arange(2, effective + 2))
        dcg = float((hits * discounts).sum())
        ideal = float(discounts[:min(len(relevant), effective)].sum())
        positions = np.flatnonzero(hits)
        ap = 0.0
        if n_hits:
            cumulative = np.cumsum(hits)
            ap = float(
                ((cumulative / np.arange(1, effective + 1)) * hits).sum()
                / min(len(relevant), k)
            )
        yield {
            "k": k,
            "effective_k": effective,
            "truncated_at_k": bool(effective < k),
            "precision": n_hits / k,
            "recall": n_hits / len(relevant),
            "hit_rate": float(n_hits > 0),
            "ndcg": dcg / ideal if ideal else 0.0,
            "mrr": 1.0 / (positions[0] + 1) if len(positions) else 0.0,
            "map": ap,
        }


def supported_cutoffs(
    ks: Iterable[int],
    actual_candidate_count: int,
    requested_candidate_size: int | None = None,
) -> tuple[int, ...]:
    """Return cutoffs admitted by realized support and the frozen protocol.

    Target injection can make a relevance-preserving sample larger than its
    nominal size.  The nominal 100-candidate condition nevertheless excludes
    k=150, even when a query has more than 150 observed targets.
    """
    return tuple(
        int(k) for k in ks
        if int(k) <= int(actual_candidate_count)
        and not (
            requested_candidate_size is not None
            and int(requested_candidate_size) == 100
            and int(k) == 150
        )
    )


def fit_bm25(job_text: pd.Series, query_text: pd.Series, max_features: int):
    vectorizer = CountVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2,
        max_df=0.98, max_features=max_features, dtype=np.float32,
    )
    counts = sparse.csr_matrix(vectorizer.fit_transform(job_text), dtype=np.float32)
    queries = sparse.csr_matrix(vectorizer.transform(query_text), dtype=np.float32)
    queries.data[:] = 1.0
    n_docs = counts.shape[0]
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    avgdl = float(lengths.mean())
    k1, b = 1.5, 0.75
    row_idx = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    denom = counts.data + k1 * (1.0 - b + b * lengths[row_idx] / avgdl)
    counts.data = counts.data * (k1 + 1.0) / denom
    return (
        counts.multiply(idf).tocsr(), queries,
        {"vocabulary": len(vectorizer.vocabulary_), "average_document_length": avgdl,
         "k1": k1, "b": b},
    )


def load_or_encode(
    output_dir: Path,
    texts: pd.Series,
    role: str,
    model_name: str,
    revision: str,
    snapshot_evidence: dict[str, object],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Content-addressed ordered-text embedding cache."""
    normalized = clean(texts)
    content_hash = stable_text_hash(normalized, role, model_name)
    safe_model = model_name.replace("/", "_").replace("\\", "_")
    cache_dir = output_dir.parent / "Shared_InternalTemporal_EmbeddingCache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{role}_{safe_model}_{content_hash[:20]}.npy"
    meta_path = path.with_suffix(".json")
    started = time.perf_counter()
    created = not path.exists()
    unique_count = int(normalized.nunique(dropna=False))
    if created:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            model_name, revision=revision, device="cpu", local_files_only=True
        )
        model.max_seq_length = 256
        codes, unique = pd.factorize(normalized, sort=False)
        vectors_unique = model.encode(
            unique.tolist(), batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        vectors = vectors_unique[codes]
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, vectors)
        temporary.replace(path)
        meta_path.write_text(json.dumps({
            "role": role, "model": model_name, "ordered_text_sha256": content_hash,
            "model_revision": revision,
            "snapshot_tree_sha256": snapshot_evidence["snapshot_tree_sha256"],
            "rows": len(normalized), "unique_texts": unique_count,
            "dimensions": int(vectors.shape[1]),
        }, indent=2), encoding="utf-8")
        del model, vectors, vectors_unique
        gc.collect()
    # Certify legacy caches without re-encoding.  The cache's ordered-input hash
    # is checked by its content-addressed path, and the only permitted encoder is
    # the asserted offline local snapshot above.
    cache_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if cache_metadata.get("ordered_text_sha256") != content_hash \
            or cache_metadata.get("model") != model_name:
        raise AssertionError(f"Embedding-cache metadata mismatch: {path}")
    recorded_revision = cache_metadata.get("model_revision")
    if recorded_revision not in (None, revision):
        raise AssertionError(
            f"Embedding cache revision mismatch at {path}: {recorded_revision}"
        )
    if recorded_revision is None:
        cache_metadata.update({
            "model_revision": revision,
            "snapshot_tree_sha256": snapshot_evidence["snapshot_tree_sha256"],
            "legacy_cache_certification": {
                "upgraded_without_reencoding": True,
                "basis": (
                    "content-addressed ordered inputs; encoder code used "
                    "local_files_only; refs/main and sole permitted revision asserted"
                ),
                "certified_utc": datetime.now(timezone.utc).isoformat(),
            },
        })
        meta_path.write_text(json.dumps(cache_metadata, indent=2), encoding="utf-8")
    matrix = np.load(path, mmap_mode="r")
    if matrix.shape != (len(normalized), 384):
        raise AssertionError(f"Unexpected embedding shape {matrix.shape} at {path}")
    return matrix, {
        "path": str(path.resolve()), "sha256": sha256(path),
        "ordered_text_sha256": content_hash, "created": created,
        "model_revision": revision,
        "snapshot_tree_sha256": snapshot_evidence["snapshot_tree_sha256"],
        "rows": len(normalized), "unique_texts": unique_count,
        "bytes": path.stat().st_size,
        "load_or_encode_seconds": time.perf_counter() - started,
    }


def choose_users(cohort: pd.DataFrame, per_activity: int | None) -> pd.DataFrame:
    cohort = cohort.sort_values("UserID").copy()
    if per_activity is None:
        return cohort.reset_index(drop=True)
    pieces = []
    for group in ("cold_0", "one_1", "low_2_4", "high_5plus"):
        pieces.append(cohort[cohort["activity_group"].eq(group)].head(per_activity))
    selected = pd.concat(pieces, ignore_index=True).drop_duplicates("UserID")
    return selected.sort_values("UserID").reset_index(drop=True)


def load_window(args: argparse.Namespace, window: int, output_dir: Path):
    started = time.perf_counter()
    directory = args.dataset_dir / args.protocol / f"window_{window}"
    catalog_candidates = [
        directory / "jobs_active_at_cutoff.csv",
        directory / "jobs_active_at_cutoff.csv.gz",
    ]
    present_catalogs = [path for path in catalog_candidates if path.exists()]
    if len(present_catalogs) != 1:
        raise FileNotFoundError(
            f"Expected exactly one plain or gzip cutoff catalog for W{window}; "
            f"found {present_catalogs}"
        )
    catalog_path = present_catalogs[0]
    preparation_manifest_path = args.dataset_dir / "INTERNAL_TEMPORAL_PREPARATION_MANIFEST.json"
    preparation_manifest = json.loads(
        preparation_manifest_path.read_text(encoding="utf-8")
    )
    catalog_record = preparation_manifest["protocols"][args.protocol]["windows"][str(window)]
    actual_catalog_sha256 = sha256(catalog_path)
    if actual_catalog_sha256 != catalog_record["catalog_file_sha256"]:
        raise AssertionError(
            f"W{window}: cutoff-catalog SHA-256 mismatch; expected "
            f"{catalog_record['catalog_file_sha256']}, found {actual_catalog_sha256}"
        )
    required = [
        "cohort_population.csv", "cohort_audit_sample.csv", "valid_future_targets.csv",
        "precut_applications.csv", "query_profiles.csv",
    ]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing prepared W{window} inputs: {missing}")
    jobs = pd.read_csv(catalog_path, low_memory=False)
    if len(jobs) != int(catalog_record["catalog_jobs"]):
        raise AssertionError(
            f"W{window}: cutoff-catalog row mismatch; expected "
            f"{catalog_record['catalog_jobs']}, found {len(jobs)}"
        )
    jobs["JobID"] = jobs["JobID"].astype(np.int64)
    if jobs["JobID"].duplicated().any():
        raise AssertionError(f"W{window}: duplicate catalog JobID")
    jobs["Title"] = clean(jobs["Title"])
    jobs["text_all"] = clean(jobs["text_all"])

    population = pd.read_csv(directory / "cohort_population.csv")
    audit = pd.read_csv(directory / "cohort_audit_sample.csv")
    users = choose_users(audit, args.smoke_users_per_activity)
    profiles = pd.read_csv(directory / "query_profiles.csv")
    profiles["profile_combined"] = clean(profiles["profile_combined"])
    users = users.merge(
        profiles[["UserID", "profile_combined"]], on="UserID", how="left",
        validate="one_to_one",
    )
    if users["profile_combined"].fillna("").str.strip().eq("").any():
        raise AssertionError(f"W{window}: selected user without released content")

    targets = pd.read_csv(directory / "valid_future_targets.csv")
    targets = targets[targets["UserID"].isin(users["UserID"])].copy()
    truth = targets.groupby("UserID")["JobID"].agg(
        lambda values: set(map(int, values))
    ).to_dict()
    if set(users["UserID"].astype(int)) != set(truth):
        raise AssertionError(f"W{window}: selected cohort/target user mismatch")

    precut = pd.read_csv(directory / "precut_applications.csv")
    precut["UserID"] = precut["UserID"].astype(np.int64)
    precut["JobID"] = precut["JobID"].astype(np.int64)
    prior = {
        int(uid): set(map(int, values))
        for uid, values in precut[
            precut["UserID"].isin(users["UserID"])
        ].groupby("UserID")["JobID"]
    }
    catalog = set(jobs["JobID"].astype(int))
    for uid, relevant in truth.items():
        if not relevant.issubset(catalog):
            raise AssertionError(f"W{window} U{uid}: target absent from catalog")
        if relevant.intersection(prior.get(uid, set())):
            raise AssertionError(f"W{window} U{uid}: repeated prior target retained")
    if not set(users["UserID"]).issubset(set(population["UserID"])):
        raise AssertionError(f"W{window}: audit user absent from population")

    popularity_counts = precut["JobID"].value_counts()
    popularity = jobs["JobID"].map(popularity_counts).fillna(0).to_numpy(np.float32)
    mean_target_popularity = {
        uid: float(np.mean([popularity_counts.get(job, 0) for job in relevant]))
        for uid, relevant in truth.items()
    }
    users["mean_target_precut_popularity"] = (
        users["UserID"].map(mean_target_popularity).astype(float)
    )
    users["target_popularity_group"] = pd.cut(
        users["mean_target_precut_popularity"],
        bins=[-1, 0, 2, np.inf],
        labels=["mean_unseen_0", "mean_positive_le2", "mean_gt2"],
    ).astype(str)
    weights = np.power(popularity.astype(np.float64) + 1.0, 0.75)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return {
        "directory": directory, "catalog_path": catalog_path,
        "verified_catalog_sha256": actual_catalog_sha256,
        "verified_catalog_rows": len(jobs),
        "verified_catalog_bytes": catalog_path.stat().st_size,
        "jobs": jobs, "population": population,
        "users": users, "truth": truth, "prior": prior,
        "job_ids": jobs["JobID"].to_numpy(np.int64),
        "job_index": {int(j): i for i, j in enumerate(jobs["JobID"])},
        "popularity": popularity, "sampling_cdf": cdf,
        "load_seconds": time.perf_counter() - started,
    }


def prepare_models(args: argparse.Namespace, assets: dict, output_dir: Path):
    started = time.perf_counter()
    jobs = assets["jobs"]
    users = assets["users"]
    timing: dict[str, float] = {}

    phase = time.perf_counter()
    print(f"  W{assets['window']} fitting TF-IDF", flush=True)
    tfidf = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2,
        max_df=0.98, max_features=args.max_features, sublinear_tf=True,
        dtype=np.float32,
    )
    tfidf_jobs = normalize(tfidf.fit_transform(jobs["text_all"]), copy=False).tocsr()
    tfidf_users = normalize(tfidf.transform(users["profile_combined"]), copy=False).tocsr()
    timing["tfidf_fit_transform"] = time.perf_counter() - phase
    print(f"  TF-IDF ready in {timing['tfidf_fit_transform']:.1f}s", flush=True)

    phase = time.perf_counter()
    print(f"  W{assets['window']} fitting full-text BM25", flush=True)
    bm25_jobs, bm25_users, bm25_meta = fit_bm25(
        jobs["text_all"], users["profile_combined"], args.max_features
    )
    timing["bm25_full_fit_transform"] = time.perf_counter() - phase
    print(f"  full-text BM25 ready in {timing['bm25_full_fit_transform']:.1f}s", flush=True)

    phase = time.perf_counter()
    print(f"  W{assets['window']} fitting title BM25", flush=True)
    bm25_title_jobs, bm25_title_users, bm25_title_meta = fit_bm25(
        jobs["Title"], users["profile_combined"], args.max_features
    )
    timing["bm25_title_fit_transform"] = time.perf_counter() - phase
    print(f"  title BM25 ready in {timing['bm25_title_fit_transform']:.1f}s", flush=True)

    job_embeddings = user_embeddings = None
    embedding_meta: dict[str, object] = {"skipped": args.skip_semantic}
    if not args.skip_semantic:
        if not hasattr(args, "semantic_snapshot_evidence"):
            args.semantic_snapshot_evidence = resolve_minilm_snapshot(args.semantic_model)
            args.semantic_revision = MINILM_EXPECTED_REVISION
        print(f"  W{assets['window']} loading/encoding MiniLM embeddings", flush=True)
        job_text = clean("job title: " + jobs["Title"])
        user_text = clean("candidate profile: " + users["profile_combined"])
        job_embeddings, job_meta = load_or_encode(
            output_dir, job_text, f"{args.protocol}_W{int(assets['window'])}_jobs_title",
            args.semantic_model, args.semantic_revision,
            args.semantic_snapshot_evidence, args.embedding_batch_size,
        )
        user_embeddings, user_meta = load_or_encode(
            output_dir, user_text,
            f"{args.protocol}_W{int(assets['window'])}_users_static_"
            f"{sha256(assets['directory'] / 'cohort_audit_sample.csv')[:12]}",
            args.semantic_model, args.semantic_revision,
            args.semantic_snapshot_evidence, args.embedding_batch_size,
        )
        embedding_meta = {"job": job_meta, "user": user_meta, "skipped": False}
        timing["minilm_load_or_encode"] = (
            job_meta["load_or_encode_seconds"] + user_meta["load_or_encode_seconds"]
        )
        print(f"  MiniLM assets ready in {timing['minilm_load_or_encode']:.1f}s", flush=True)

    assets.update({
        "tfidf_jobs_t": tfidf_jobs.T.tocsr(), "tfidf_users": tfidf_users,
        "bm25_jobs_t": bm25_jobs.T.tocsr(), "bm25_users": bm25_users,
        "bm25_title_jobs_t": bm25_title_jobs.T.tocsr(),
        "bm25_title_users": bm25_title_users,
        "job_embeddings": job_embeddings, "user_embeddings": user_embeddings,
    })
    return {
        "tfidf_vocabulary": len(tfidf.vocabulary_),
        "bm25_full": bm25_meta, "bm25_title": bm25_title_meta,
        "embeddings": embedding_meta, "phase_seconds": timing,
        "total_seconds": time.perf_counter() - started,
    }


def sample_uniform(
    rng: np.random.Generator, n_jobs: int, banned: set[int], needed: int
) -> np.ndarray:
    if needed <= 0:
        return np.empty(0, dtype=np.int64)
    available = n_jobs - len(banned)
    if needed > available:
        raise AssertionError("Requested more uniform negatives than available")
    if needed > n_jobs // 4:
        mask = np.ones(n_jobs, dtype=bool)
        if banned:
            mask[np.fromiter(banned, dtype=np.int64)] = False
        return rng.choice(np.flatnonzero(mask), size=needed, replace=False)
    chosen: list[int] = []
    seen: set[int] = set()
    while len(chosen) < needed:
        remaining = needed - len(chosen)
        draw_count = min(n_jobs, max(256, int(remaining * 1.25)))
        for raw in rng.choice(n_jobs, size=draw_count, replace=False):
            idx = int(raw)
            if idx not in banned and idx not in seen:
                seen.add(idx)
                chosen.append(idx)
                if len(chosen) == needed:
                    break
    return np.asarray(chosen, dtype=np.int64)


def sample_weighted(
    rng: np.random.Generator, cdf: np.ndarray, banned: set[int], needed: int
) -> np.ndarray:
    if needed <= 0:
        return np.empty(0, dtype=np.int64)
    if needed > len(cdf) - len(banned):
        raise AssertionError("Requested more weighted negatives than available")
    chosen: list[int] = []
    seen: set[int] = set()
    while len(chosen) < needed:
        remaining = needed - len(chosen)
        draws = np.searchsorted(
            cdf, rng.random(max(512, int(remaining * 1.5))), side="right"
        )
        for raw in draws:
            idx = int(raw)
            if idx not in banned and idx not in seen:
                seen.add(idx)
                chosen.append(idx)
                if len(chosen) == needed:
                    break
    return np.asarray(chosen, dtype=np.int64)


def candidate_indices(
    assets: dict, uid: int, window: int, strategy: str, seed: int,
    max_nominal_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    relevant = assets["truth"][uid]
    relevant_idx = np.asarray(
        sorted(assets["job_index"][j] for j in relevant), dtype=np.int64
    )
    prior_idx = {
        assets["job_index"][j] for j in assets["prior"].get(uid, set())
        if j in assets["job_index"]
    }
    banned = prior_idx.union(map(int, relevant_idx))
    eligible_count = len(assets["job_ids"]) - len(prior_idx)
    needed = max(0, min(max_nominal_size, eligible_count) - len(relevant_idx))
    strategy_code = 0 if strategy == "uniform" else 500_000_000
    rng = np.random.default_rng(
        MASTER_SAMPLING_SEED + strategy_code + window * 10_000_000
        + seed * 100_000 + uid
    )
    if strategy == "uniform":
        negatives = sample_uniform(rng, len(assets["job_ids"]), banned, needed)
    else:
        negatives = sample_weighted(rng, assets["sampling_cdf"], banned, needed)
    return relevant_idx, negatives, eligible_count


def score_batch(args: argparse.Namespace, assets: dict, start: int, stop: int):
    scores = {
        "tfidf_combined_static": (
            assets["tfidf_users"][start:stop] @ assets["tfidf_jobs_t"]
        ).toarray(),
        "bm25_full_static": (
            assets["bm25_users"][start:stop] @ assets["bm25_jobs_t"]
        ).toarray(),
        "bm25_title_static": (
            assets["bm25_title_users"][start:stop] @ assets["bm25_title_jobs_t"]
        ).toarray(),
    }
    if not args.skip_semantic:
        scores["minilm_title_static"] = (
            np.asarray(assets["user_embeddings"][start:stop], dtype=np.float32)
            @ np.asarray(assets["job_embeddings"], dtype=np.float32).T
        )
    return scores


def evaluate_user(
    args: argparse.Namespace, assets: dict, local_offset: int,
    score_vectors: dict[str, np.ndarray], integrity: dict[str, int],
) -> list[dict[str, object]]:
    user = assets["users"].iloc[local_offset]
    uid = int(user["UserID"])
    relevant = assets["truth"][uid]
    job_ids = assets["job_ids"]
    prior_idx = np.asarray(sorted(
        assets["job_index"][j] for j in assets["prior"].get(uid, set())
        if j in assets["job_index"]
    ), dtype=np.int64)
    mask = np.ones(len(job_ids), dtype=bool)
    mask[prior_idx] = False
    eligible_idx = np.flatnonzero(mask)
    eligible_ids = job_ids[eligible_idx]
    eligible_count = len(eligible_idx)
    max_k = max(args.ks)
    common = {
        "protocol": args.protocol, "window": int(assets["window"]),
        "user_id": uid, "precut_application_count": int(user["precut_application_count"]),
        "activity_group": str(user["activity_group"]),
        "mean_target_precut_popularity": float(user["mean_target_precut_popularity"]),
        "target_popularity_group": str(user["target_popularity_group"]),
        "relevant_jobs": len(relevant),
        "inclusion_probability": float(user["inclusion_probability"]),
        "population_weight": float(user["population_weight"]),
        "eligible_candidate_count": eligible_count,
    }
    static_scores = {
        "popularity_precut": assets["popularity"],
        **score_vectors,
    }
    if args.skip_semantic:
        model_names = tuple(m for m in MODELS if m != "minilm_title_static")
    else:
        model_names = MODELS
    rows: list[dict[str, object]] = []

    # Full catalog after user-specific prior-job exclusion.
    integrity["full_catalog_ranked_candidate_model_pairs"] += (
        eligible_count * len(model_names)
    )
    full_random = splitmix64_scores(eligible_ids, uid, int(assets["window"]))
    for model in model_names:
        scores = full_random if model == "random_hash" else static_scores[model][eligible_idx]
        ranked = top_with_deterministic_ties(eligible_ids, scores, max_k)
        full_valid_ks = supported_cutoffs(args.ks, eligible_count)
        for values in metric_rows(ranked, relevant, full_valid_ks):
            rows.append({
                **common, "candidate_regime": "full_catalog",
                "sampling_strategy": "full_catalog", "seed": 0,
                "requested_candidate_size": np.nan,
                "actual_candidate_count": eligible_count,
                "actual_candidate_fraction": 1.0,
                "targets_injected": False, "model": model, **values,
            })

    sizes = tuple(sorted(set(args.candidate_sizes)))
    for strategy in args.strategies:
        for seed in sorted(set(args.seeds)):
            rel_idx, negatives, checked_eligible = candidate_indices(
                assets, uid, int(assets["window"]), strategy, seed, max(sizes)
            )
            if checked_eligible != eligible_count:
                raise AssertionError("Eligible candidate count mismatch")
            rel_ids = job_ids[rel_idx]
            max_idx = np.concatenate((rel_idx, negatives))
            # Sampling determinism audit for the first observed user/unit.
            if not integrity.get("determinism_checked", 0):
                rel2, neg2, eligible2 = candidate_indices(
                    assets, uid, int(assets["window"]), strategy, seed, max(sizes)
                )
                if eligible2 != eligible_count or not np.array_equal(rel_idx, rel2) \
                        or not np.array_equal(negatives, neg2):
                    raise AssertionError("Candidate sampling is not deterministic")
                integrity["determinism_checked"] = 1
            previous: set[int] | None = None
            for nominal in sizes:
                actual_target = min(eligible_count, max(nominal, len(rel_idx)))
                neg_needed = max(0, actual_target - len(rel_idx))
                candidate_idx = max_idx[:len(rel_idx) + neg_needed]
                candidate_ids = job_ids[candidate_idx]
                candidate_set = set(map(int, candidate_ids))
                if not relevant.issubset(candidate_set):
                    integrity["target_inclusion_failures"] += 1
                if previous is not None and not previous.issubset(candidate_set):
                    integrity["nesting_failures"] += 1
                previous = candidate_set
                if len(rel_idx) > nominal:
                    integrity["target_count_exceeds_nominal_user_sizes"] += 1
                integrity["sampled_ranked_candidate_model_pairs"] += (
                    len(candidate_ids) * len(model_names)
                )
                random_scores = splitmix64_scores(
                    candidate_ids, uid, int(assets["window"])
                )
                # Require realized support and the frozen nominal-size rule.
                # Target injection can enlarge a size-100 sample beyond 150,
                # but k=150 remains outside that nominal protocol condition.
                valid_ks = supported_cutoffs(
                    args.ks, len(candidate_ids), requested_candidate_size=nominal
                )
                if not valid_ks:
                    continue
                for model in model_names:
                    scores = (
                        random_scores if model == "random_hash"
                        else static_scores[model][candidate_idx]
                    )
                    ranked = top_with_deterministic_ties(candidate_ids, scores, max(valid_ks))
                    for values in metric_rows(ranked, relevant, valid_ks):
                        rows.append({
                            **common, "candidate_regime": "sampled",
                            "sampling_strategy": strategy, "seed": seed,
                            "requested_candidate_size": nominal,
                            "actual_candidate_count": len(candidate_ids),
                            "actual_candidate_fraction": len(candidate_ids) / eligible_count,
                            "targets_injected": True, "model": model, **values,
                        })
    return rows


def write_checkpoint(
    args: argparse.Namespace, assets: dict, output_dir: Path, start: int, stop: int,
    integrity: dict[str, int],
) -> tuple[Path, float, int]:
    path = output_dir / "checkpoints" / f"window_{assets['window']}" / (
        f"users_{start:04d}_{stop - 1:04d}.csv.gz"
    )
    if path.exists():
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            raise RuntimeError(f"Checkpoint lacks integrity metadata: {path}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        for key, value in metadata["integrity_delta"].items():
            integrity[key] += int(value)
        return path, 0.0, int(metadata["users"])
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    integrity_before = dict(integrity)
    batch_scores = score_batch(args, assets, start, stop)
    n_content_score_models = 3 + int(not args.skip_semantic)
    integrity["content_full_catalog_scored_pairs"] += (
        (stop - start) * len(assets["job_ids"]) * n_content_score_models
    )
    rows: list[dict[str, object]] = []
    for offset in range(start, stop):
        per_user_scores = {
            name: matrix[offset - start] for name, matrix in batch_scores.items()
        }
        rows.extend(evaluate_user(args, assets, offset, per_user_scores, integrity))
    frame = pd.DataFrame(rows)
    temporary = path.with_suffix(".tmp.gz")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)
    integrity_delta = {
        key: int(integrity[key] - integrity_before.get(key, 0))
        for key in integrity
    }
    metadata = {
        "window": int(assets["window"]), "user_offsets": [start, stop],
        "users": stop - start, "rows": len(frame), "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "score_boundary": (
            "all model scores computed once against full post-exclusion catalog; "
            "sampled effectiveness indexes those scores and is not an RQ5 speed benchmark"
        ),
        "integrity_delta": integrity_delta,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path, metadata["elapsed_seconds"], stop - start


def consolidate(paths: list[Path], output_dir: Path) -> dict[str, object]:
    aggregate_parts = []
    manifest = []
    forbidden_size100_k150_rows = 0
    invalid_cutoff_rows = 0
    for path in paths:
        frame = pd.read_csv(path)
        forbidden_size100_k150_rows += int((
            frame["candidate_regime"].eq("sampled")
            & frame["requested_candidate_size"].eq(100)
            & frame["k"].eq(150)
        ).sum())
        invalid_cutoff_rows += int((frame["k"] > frame["actual_candidate_count"]).sum())
        groups = [
            "protocol", "window", "candidate_regime", "sampling_strategy", "seed",
            "requested_candidate_size", "model", "k",
        ]
        frame["weighted_denominator"] = frame["population_weight"]
        for metric in METRICS:
            frame[f"weighted_{metric}"] = frame[metric] * frame["population_weight"]
        agg = frame.groupby(groups, dropna=False, as_index=False).agg(
            users=("user_id", "nunique"),
            weight_sum=("weighted_denominator", "sum"),
            actual_candidate_count_min=("actual_candidate_count", "min"),
            actual_candidate_count_max=("actual_candidate_count", "max"),
            actual_candidate_fraction_sum=("actual_candidate_fraction", "sum"),
            **{f"sum_{metric}": (metric, "sum") for metric in METRICS},
            **{f"weighted_sum_{metric}": (f"weighted_{metric}", "sum") for metric in METRICS},
        )
        aggregate_parts.append(agg)
        manifest.append({
            "path": str(path.resolve()), "rows": len(frame),
            "users": int(frame["user_id"].nunique()), "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    partial = pd.concat(aggregate_parts, ignore_index=True)
    groups = [
        "protocol", "window", "candidate_regime", "sampling_strategy", "seed",
        "requested_candidate_size", "model", "k",
    ]
    combined = partial.groupby(groups, dropna=False, as_index=False).agg(
        users=("users", "sum"), weight_sum=("weight_sum", "sum"),
        actual_candidate_count_min=("actual_candidate_count_min", "min"),
        actual_candidate_count_max=("actual_candidate_count_max", "max"),
        actual_candidate_fraction_sum=("actual_candidate_fraction_sum", "sum"),
        **{f"sum_{metric}": (f"sum_{metric}", "sum") for metric in METRICS},
        **{f"weighted_sum_{metric}": (f"weighted_sum_{metric}", "sum") for metric in METRICS},
    )
    for metric in METRICS:
        combined[metric] = combined[f"sum_{metric}"] / combined["users"]
        combined[f"weighted_{metric}"] = (
            combined[f"weighted_sum_{metric}"] / combined["weight_sum"]
        )
    combined["actual_candidate_fraction_mean"] = (
        combined["actual_candidate_fraction_sum"] / combined["users"]
    )
    keep = groups + [
        "users", "weight_sum", "actual_candidate_count_min",
        "actual_candidate_count_max", "actual_candidate_fraction_mean",
    ] + list(METRICS) + [f"weighted_{x}" for x in METRICS]
    combined[keep].to_csv(output_dir / "aggregate_metrics.csv", index=False)
    checkpoint_manifest = {
        "authoritative_partitions": manifest, "checkpoint_count": len(manifest),
        "total_rows": sum(x["rows"] for x in manifest),
        "total_compressed_bytes": sum(x["bytes"] for x in manifest),
        "monolithic_per_user_file": None,
        "forbidden_size100_k150_rows": forbidden_size100_k150_rows,
        "invalid_cutoff_rows": invalid_cutoff_rows,
    }
    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2), encoding="utf-8"
    )
    return checkpoint_manifest


def projected_resources(
    args: argparse.Namespace, all_manifest: dict, smoke_window: int,
    smoke_users: int, smoke_catalog: int, asset_seconds: float,
    evaluate_seconds: float, checkpoint_bytes: int, peak_rss: int,
) -> dict[str, object]:
    windows = all_manifest["protocols"][args.protocol]["windows"]
    total_pairs = 0
    total_users = 0
    largest_catalog = 0
    for raw in windows.values():
        n = int(raw["audit_sample_queries"])
        catalog = int(raw["catalog_jobs"])
        total_users += n
        total_pairs += n * catalog
        largest_catalog = max(largest_catalog, catalog)
    smoke_pairs = max(1, smoke_users * smoke_catalog)
    projected_eval_seconds = evaluate_seconds * total_pairs / smoke_pairs
    projected_asset_seconds = asset_seconds * sum(
        int(raw["catalog_jobs"]) for raw in windows.values()
    ) / max(1, smoke_catalog)
    rows_per_user = checkpoint_bytes / max(1, smoke_users)
    return {
        "basis": "linear engineering projection from one-window smoke; not an SLA",
        "audit_users_all_windows": total_users,
        "full_catalog_user_job_pairs_per_model": total_pairs,
        "full_catalog_user_job_pairs_all_six_models": total_pairs * len(MODELS),
        "projected_effectiveness_seconds": projected_eval_seconds,
        "projected_asset_fit_encode_seconds": projected_asset_seconds,
        "projected_total_hours": (projected_eval_seconds + projected_asset_seconds) / 3600,
        "projected_checkpoint_bytes": int(rows_per_user * total_users),
        "observed_smoke_peak_rss_bytes": peak_rss,
        "conservative_largest_window_rss_bytes": int(
            peak_rss * max(1.0, largest_catalog / max(1, smoke_catalog))
        ),
        "dense_full_score_cache_bytes": 0,
        "dense_full_score_cache_policy": (
            "none retained; query batches are scored once and discarded"
        ),
        "disk_safety_note": (
            "projection excludes prepared source catalogs and shared embedding caches, "
            "whose exact sizes are recorded separately"
        ),
    }


def main() -> None:
    invocation_started_perf = time.perf_counter()
    invocation_started_utc = datetime.now(timezone.utc).isoformat()
    args = parse_args()
    args.windows = sorted(set(args.windows))
    args.seeds = sorted(set(args.seeds))
    args.candidate_sizes = sorted(set(args.candidate_sizes))
    args.ks = sorted(set(args.ks))
    output_dir = args.output_root / args.run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset_dir / "INTERNAL_TEMPORAL_PREPARATION_MANIFEST.json"
    prep_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.protocol not in prep_manifest["protocols"]:
        raise KeyError(f"Protocol not prepared: {args.protocol}")
    if args.authoritative_production:
        if args.study_role == "primary":
            permitted_protocols = {"First7Last2"}
            required_sample_count = 1000
        else:
            permitted_protocols = {"First6Last3", "First5Last4"}
            required_sample_count = 250
        expected = {
            "windows": list(range(1, 8)), "seeds": [1, 2, 3, 4, 5],
            "strategies": list(STRATEGIES),
            "candidate_sizes": list(DEFAULT_SIZES), "ks": list(DEFAULT_KS),
        }
        actual = {
            "windows": args.windows, "seeds": args.seeds,
            "strategies": args.strategies, "candidate_sizes": args.candidate_sizes,
            "ks": args.ks,
        }
        if actual != expected or args.protocol not in permitted_protocols \
                or args.smoke_users_per_activity is not None or args.skip_semantic:
            raise RuntimeError(
                "--authoritative-production requires its complete frozen study-role design; "
                f"role={args.study_role}, protocol={args.protocol}, actual={actual}, "
                f"smoke={args.smoke_users_per_activity}, "
                f"skip_semantic={args.skip_semantic}"
            )
        sample_counts = [
            int(value["audit_sample_queries"])
            for value in prep_manifest["protocols"][args.protocol]["windows"].values()
        ]
        if sample_counts != [required_sample_count] * 7:
            raise RuntimeError(
                f"Authoritative {args.study_role} cohorts are not "
                f"{required_sample_count}/window: {sample_counts}"
            )
        if args.study_role == "temporal_sensitivity":
            prep_authority = prep_manifest.get("preparation_authority", {})
            prep_config = prep_manifest.get("preparation_config", {})
            required_protocols = {"First6Last3", "First5Last4"}
            sensitivity_prep_valid = (
                prep_authority == {
                    "authoritative": True,
                    "study_role": "temporal_sensitivity",
                    "status": "completed",
                }
                and set(prep_config.get("protocols", [])) == required_protocols
                and len(prep_config.get("protocols", [])) == 2
                and int(prep_config.get("sample_cap_per_window", -1)) == 250
                and prep_config.get("compress_catalogs") is True
                and int(prep_config.get("master_sample_seed", -1)) == MASTER_SAMPLING_SEED
                and int(prep_manifest.get("sample_cap_per_window", -1)) == 250
                and int(prep_manifest.get("master_sample_seed", -1)) == MASTER_SAMPLING_SEED
                and set(prep_manifest.get("protocols", {})) == required_protocols
            )
            if not sensitivity_prep_valid:
                raise RuntimeError(
                    "Authoritative sensitivity evaluation requires a completed, "
                    "authority-marked two-protocol sensitivity preparation manifest"
                )
    if args.skip_semantic:
        args.semantic_snapshot_evidence = {"skipped": True}
        args.semantic_revision = None
    else:
        args.semantic_snapshot_evidence = resolve_minilm_snapshot(args.semantic_model)
        args.semantic_revision = MINILM_EXPECTED_REVISION
    run_config = {
        "task": "CareerBuilder_InternalTemporal",
        "official_competition_evaluation": False,
        "authority": {
            "authoritative": bool(args.authoritative_production),
            "production": bool(args.authoritative_production),
            "smoke": args.smoke_users_per_activity is not None,
            "study_role": args.study_role,
            "status": "configured",
        },
        "protocol": args.protocol, "windows": args.windows,
        "seeds": args.seeds, "strategies": args.strategies,
        "candidate_sizes": args.candidate_sizes, "ks": args.ks,
        "size100_k150_policy": "row excluded entirely",
        "models": [m for m in MODELS if not (args.skip_semantic and m == "minilm_title_static")],
        "query_profile": "released static profile_combined only",
        "behavior_use": (
            "pre-cutoff applications only for popularity, candidate exclusion, and strata"
        ),
        "provider_split_used": False,
        "item_knn": (
            "omitted: requires a separately framed behavioral estimand and its cold-user "
            "fallback would duplicate popularity rather than preserve a common content query"
        ),
        "audit_sample_design": "uniform_without_replacement_within_window",
        "pooled_population_weight": "N_w / n_w",
        "activity_groups": ["cold_0", "one_1", "low_2_4", "high_5plus"],
        "target_popularity_groups": {
            "unit": "query mean of its targets' pre-cutoff application counts",
            "labels": ["mean_unseen_0", "mean_positive_le2", "mean_gt2"],
            "thresholds": ["mean=0", "0<mean<=2", "mean>2"],
        },
        "uniform_sampler": "nested simple random sampling without replacement",
        "popularity_weighted_sampler": (
            "nested successive PPS without replacement implemented by rejection "
            "from a fixed pre-cutoff-popularity CDF"
        ),
        "max_features": args.max_features,
        "semantic_model": args.semantic_model,
        "semantic_revision": args.semantic_revision,
        "semantic_snapshot_evidence": args.semantic_snapshot_evidence,
        "smoke_users_per_activity": args.smoke_users_per_activity,
        "query_batch_size": args.query_batch_size,
        "checkpoint_users": args.checkpoint_users,
        "embedding_batch_size": args.embedding_batch_size,
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256(Path(__file__)),
        "preparation_manifest_path": str(manifest_path.resolve()),
        "preparation_manifest_sha256": sha256(manifest_path),
        "software": software_versions(),
        "command_equivalent_arguments": {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "protocol": args.protocol, "output_root": str(args.output_root.resolve()),
            "run_label": args.run_label, "windows": args.windows,
            "seeds": args.seeds, "strategies": args.strategies,
            "candidate_sizes": args.candidate_sizes, "ks": args.ks,
            "max_features": args.max_features,
            "query_batch_size": args.query_batch_size,
            "checkpoint_users": args.checkpoint_users,
            "semantic_model": args.semantic_model,
            "embedding_batch_size": args.embedding_batch_size,
            "smoke_users_per_activity": args.smoke_users_per_activity,
            "skip_semantic": args.skip_semantic,
            "authoritative_production": args.authoritative_production,
            "study_role": args.study_role,
        },
    }
    config_path = output_dir / "RUN_CONFIG.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise RuntimeError(f"Run directory has a different configuration: {config_path}")
    else:
        config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    invocation_metadata = {
        "started_utc": invocation_started_utc,
        "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "run_config_sha256": sha256(config_path),
    }
    (output_dir / "LAST_INVOCATION.json").write_text(
        json.dumps(invocation_metadata, indent=2), encoding="utf-8"
    )

    all_paths: list[Path] = []
    window_reports: list[dict[str, object]] = []
    integrity = {
        "target_inclusion_failures": 0, "nesting_failures": 0,
        "target_count_exceeds_nominal_user_sizes": 0,
        "determinism_checked": 0,
        "full_catalog_ranked_candidate_model_pairs": 0,
        "sampled_ranked_candidate_model_pairs": 0,
        "content_full_catalog_scored_pairs": 0,
    }
    peak_rss = rss_bytes()
    total_eval_seconds = 0.0
    total_asset_seconds = 0.0
    first_window_catalog = first_window_users = 0
    for window in args.windows:
        assets = load_window(args, window, output_dir)
        assets["window"] = window
        selected = assets["users"]
        full = assets["population"]
        print(
            f"W{window}: {len(selected):,} audit queries / {len(full):,} population; "
            f"{len(assets['jobs']):,} jobs",
            flush=True,
        )
        model_meta = prepare_models(args, assets, output_dir)
        total_asset_seconds += assets["load_seconds"] + model_meta["total_seconds"]
        peak_rss = max(peak_rss, rss_bytes())
        window_paths = []
        window_eval_seconds = 0.0
        for start in range(0, len(selected), args.checkpoint_users):
            stop = min(start + args.checkpoint_users, len(selected))
            path, elapsed, _ = write_checkpoint(
                args, assets, output_dir, start, stop, integrity
            )
            window_paths.append(path)
            window_eval_seconds += elapsed
            peak_rss = max(peak_rss, rss_bytes())
            print(f"  W{window} checkpoint {start}:{stop} in {elapsed:.1f}s", flush=True)
        all_paths.extend(window_paths)
        total_eval_seconds += window_eval_seconds
        if not first_window_catalog:
            first_window_catalog = len(assets["jobs"])
            first_window_users = len(selected)
        window_prep = prep_manifest["protocols"][args.protocol]["windows"][str(window)]
        selected_counts = {
            str(k): int(v) for k, v in
            selected["activity_group"].value_counts().sort_index().items()
        }
        window_reports.append({
            "window": window,
            "population_queries": int(window_prep["population_queries"]),
            "population_warm_queries": int(window_prep["warm_queries"]),
            "population_cold_queries": int(window_prep["cold_queries"]),
            "population_activity_counts": window_prep["population_activity_counts"],
            "audit_sample_queries": int(window_prep["audit_sample_queries"]),
            "selected_run_queries": len(selected),
            "selected_run_activity_counts": selected_counts,
            "catalog_jobs": len(assets["jobs"]),
            "prepared_catalog_sha256": window_prep["catalog_file_sha256"],
            "verified_catalog_sha256": assets["verified_catalog_sha256"],
            "verified_catalog_rows": assets["verified_catalog_rows"],
            "verified_catalog_bytes": assets["verified_catalog_bytes"],
            "model_preparation": model_meta,
            "data_load_seconds": assets["load_seconds"],
            "effectiveness_seconds": window_eval_seconds,
            "full_catalog_user_job_pairs_per_model_before_prior_exclusion": (
                len(selected) * len(assets["jobs"])
            ),
            "dense_score_cache_bytes_retained": 0,
            "checkpoint_paths": [str(x.resolve()) for x in window_paths],
        })
        del assets
        gc.collect()

    checkpoint_manifest = consolidate(all_paths, output_dir)
    peak_rss = max(peak_rss, rss_bytes())
    integrity.update({
        "catalog_unique": True, "targets_in_catalog": True,
        "targets_novel_vs_precut": True, "sample_users_in_population": True,
        "provider_split_read": False,
        "size100_k150_rows": checkpoint_manifest["forbidden_size100_k150_rows"],
        "invalid_cutoff_rows": checkpoint_manifest["invalid_cutoff_rows"],
        "all_targets_injected_in_sampled_candidates": (
            integrity["target_inclusion_failures"] == 0
        ),
        "nested_candidate_failures": integrity["nesting_failures"],
    })
    if (
        integrity["target_inclusion_failures"]
        or integrity["nesting_failures"]
        or integrity["size100_k150_rows"]
        or integrity["invalid_cutoff_rows"]
    ):
        raise AssertionError(f"Candidate integrity failed: {integrity}")
    (output_dir / "INTEGRITY_AUDIT.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    projection = projected_resources(
        args, prep_manifest, args.windows[0], first_window_users,
        first_window_catalog, total_asset_seconds, total_eval_seconds,
        checkpoint_manifest["total_compressed_bytes"], peak_rss,
    )
    projection["free_disk_bytes_after_run"] = int(shutil.disk_usage(output_dir).free)
    (output_dir / "SMOKE_RESOURCE_PROJECTION.json").write_text(
        json.dumps(projection, indent=2), encoding="utf-8"
    )
    report = {
        "run_config": run_config,
        "authority": {
            **run_config["authority"],
            "status": "completed",
        },
        "windows": window_reports,
        "integrity": integrity, "resources": projection,
        "checkpoint_manifest": checkpoint_manifest,
        "timing_boundaries": {
            "effectiveness": (
                "full-catalog scores computed in query batches once; sampled metrics index "
                "these scores; this timing cannot support sampled-scoring savings claims"
            ),
            "rq5": "not measured here; requires separate direct-scoring benchmark",
        },
        "provenance": {
            "started_utc": invocation_started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "invocation_elapsed_seconds": time.perf_counter() - invocation_started_perf,
            "argv": invocation_metadata["argv"],
            "evaluator_sha256": run_config["evaluator_sha256"],
            "preparation_manifest_sha256": run_config["preparation_manifest_sha256"],
            "run_config_sha256": sha256(config_path),
            "embedding_cache_hashes": {
                str(window_report["window"]): window_report["model_preparation"]["embeddings"]
                for window_report in window_reports
            },
        },
    }
    (output_dir / "RUN_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Completed InternalTemporal run: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
