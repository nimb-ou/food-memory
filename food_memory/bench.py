"""Benchmark saved Food Memory indexes without re-embedding images."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .config import artifact_dir_for_mode
from .index import load_index
from .metadata import MetadataStore
from .metrics import percentiles_us


def bench_artifacts(artifact_dir: Path, n_queries: int, seed: int) -> dict:
    embeddings = np.load(artifact_dir / "train_embeddings.npy").astype(np.float32)
    store = MetadataStore(artifact_dir / "metadata.sqlite")
    rows = store.rows(split="train")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(embeddings), size=min(n_queries, len(embeddings)), replace=False)
    queries = embeddings[picks]

    exact = {row.image_hash: row.label_id for row in rows}
    exact_keys = [rows[i].image_hash for i in picks]
    exact_timings = []
    for key in exact_keys:
        t0 = time.perf_counter()
        _ = exact.get(key)
        exact_timings.append(time.perf_counter() - t0)

    indexes = {}
    for kind in ("flat", "hnsw"):
        path = artifact_dir / f"index_{kind}.faiss"
        if not path.exists():
            continue
        idx = load_index(path, kind=kind)
        timings = []
        correct_at_1 = 0
        for expected, q in zip(picks, queries):
            t0 = time.perf_counter()
            result = idx.search(q, k=5)
            timings.append(time.perf_counter() - t0)
            if int(result.indices[0]) == int(expected):
                correct_at_1 += 1
        indexes[kind] = {
            "latency": percentiles_us(timings),
            "self_recall_at_1": correct_at_1 / len(queries),
            "index_bytes": os.path.getsize(path),
        }

    out = {
        "artifact_dir": str(artifact_dir),
        "n_queries": len(queries),
        "exact_hash": {
            "latency": percentiles_us(exact_timings),
            "entries": len(exact),
        },
        "indexes": indexes,
    }
    out_path = artifact_dir / "bench_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark saved Food Memory artifacts.")
    ap.add_argument("--artifact-dir", type=Path, default=artifact_dir_for_mode("quick"))
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(bench_artifacts(args.artifact_dir, args.n_queries, args.seed), indent=2))


if __name__ == "__main__":
    main()

