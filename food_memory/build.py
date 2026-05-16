"""CLI for building Food Memory artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .config import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_ENCODER,
    FULL_TRAIN_PER_CLASS,
    QUICK_TRAIN_PER_CLASS,
    artifact_dir_for_mode,
)
from .datasets import iter_food101, load_food101, label_names_from_dataset
from .embedding import CLIPImageEmbedder
from .hashing import array_digest, image_digest
from .index import FAISSVectorIndex
from .metadata import ImageRecord, MetadataStore
from .metrics import l2_normalize


def build_artifacts(
    mode: str,
    artifact_dir: Path,
    samples_per_class: int | None,
    batch_size: int,
    encoder: str,
    device: str | None,
    seed: int,
) -> dict:
    started = time.perf_counter()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    store = MetadataStore(artifact_dir / "metadata.sqlite")
    store.initialize(clear=True)

    label_dataset = load_food101("train")
    labels = label_names_from_dataset(label_dataset)
    for label_id, label_name in enumerate(labels):
        store.insert_label(label_id, label_name)

    total_train = len(label_dataset) if samples_per_class is None else samples_per_class * len(labels)
    print(f"[1/4] Loading encoder: {encoder}")
    embedder = CLIPImageEmbedder(model_name=encoder, device=device)
    print(f"[2/4] Embedding {total_train:,} training images into CLIP memory")
    records: list[ImageRecord] = []
    embeddings: list[np.ndarray] = []
    images = []
    pending_rows = []
    row_id = 0

    with tqdm(total=total_train, unit="image") as progress:
        for sample in iter_food101("train", samples_per_class=samples_per_class, seed=seed):
            image = sample.image.convert("RGB")
            images.append(image)
            pending_rows.append((row_id, sample))
            row_id += 1
            if len(images) >= batch_size:
                _flush_batch(embedder, images, pending_rows, embeddings, records)
                progress.update(len(images))
                images, pending_rows = [], []

        if images:
            _flush_batch(embedder, images, pending_rows, embeddings, records)
            progress.update(len(images))

    print("[3/4] Writing SQLite metadata and embedding matrix")
    train_embeddings = l2_normalize(np.vstack(embeddings))
    np.save(artifact_dir / "train_embeddings.npy", train_embeddings)
    store.insert_records(records)
    store.set_metadata(
        "build",
        {
            "mode": mode,
            "encoder": encoder,
            "samples_per_class": samples_per_class,
            "seed": seed,
            "embedding_dim": int(train_embeddings.shape[1]),
            "train_rows": int(train_embeddings.shape[0]),
        },
    )

    print("[4/4] Building FAISS flat and HNSW indexes")
    for kind in ("flat", "hnsw"):
        idx = FAISSVectorIndex(kind=kind).build(train_embeddings)
        idx.save(artifact_dir / f"index_{kind}.faiss")

    summary = {
        "mode": mode,
        "artifact_dir": str(artifact_dir),
        "encoder": encoder,
        "train_rows": int(train_embeddings.shape[0]),
        "labels": len(labels),
        "embedding_dim": int(train_embeddings.shape[1]),
        "wall_s": time.perf_counter() - started,
        "indexes": ["flat", "hnsw"],
    }
    (artifact_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _flush_batch(embedder, images, pending_rows, embeddings, records):
    batch_embeddings = embedder.embed_images(images, batch_size=len(images))
    for i, (rid, sample) in enumerate(pending_rows):
        image = images[i]
        emb = batch_embeddings[i]
        embeddings.append(emb)
        records.append(
            ImageRecord(
                row_id=rid,
                source_id=sample.source_id,
                split="train",
                label_id=sample.label_id,
                label_name=sample.label_name,
                image_hash=image_digest(image),
                embedding_hash=array_digest(emb),
                width=image.width,
                height=image.height,
            )
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Food Memory embeddings and FAISS indexes.")
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--artifact-dir", type=Path, default=None)
    ap.add_argument("--samples-per-class", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--device", default=None, help="cpu, mps, cuda, or leave empty for auto")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    samples = args.samples_per_class
    if samples is None:
        samples = QUICK_TRAIN_PER_CLASS if args.mode == "quick" else FULL_TRAIN_PER_CLASS
    artifact_dir = args.artifact_dir or artifact_dir_for_mode(args.mode, DEFAULT_ARTIFACT_ROOT)
    summary = build_artifacts(
        mode=args.mode,
        artifact_dir=artifact_dir,
        samples_per_class=samples,
        batch_size=args.batch_size,
        encoder=args.encoder,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
