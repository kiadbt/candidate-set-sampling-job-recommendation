#!/usr/bin/env python3
"""Generate the publication-safe release manifest from repository contents."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "RELEASE_MANIFEST.json"
EXCLUDED_TOP_LEVEL = {".git", "Results"}
EXCLUDED_PATH_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def released_files() -> list[Path]:
    files: list[Path] = []
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        relative = path.relative_to(REPOSITORY_ROOT)
        if relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if any(part in EXCLUDED_PATH_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(REPOSITORY_ROOT).as_posix())


def main() -> None:
    records = [
        {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in released_files()
    ]
    manifest = {
        "manifest_version": 2,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        "scope": (
            "Publication-safe code, protocols, aggregate results, and SVG figure sources."
        ),
        "excluded": [
            "Raw and processed provider datasets",
            "Individual-level, per-user, per-query, and per-job outputs",
            "Candidate lists and paired-record details",
            "Cached embeddings, model caches, and checkpoints",
            "Temporary files, archives, submission documents, and local-path logs",
            "Row-level event timestamps, other disclosure-sensitive temporal records, and access credentials",
        ],
        "files": records,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {MANIFEST_PATH.name}: {len(records)} files")


if __name__ == "__main__":
    main()
