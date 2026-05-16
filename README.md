# Food Memory

Retrieval-first image classification for Food-101. The project applies the same fundamentals as Digit Memory to a larger, messier problem: remember the training examples, index them two ways, and measure whether lookup is already a strong baseline before training anything deep.

Food Memory stores CLIP image embeddings for Food-101 training images, checks an exact image hash first, then falls back to FAISS nearest-neighbor search over normalized 512-dimensional vectors.

## Core Idea

```
query image
    |
    v
canonical RGB hash -------------- hit ---> return memorized label
    |
   miss
    |
    v
CLIP image embedding -> FAISS vector search -> k-NN vote -> label + neighbors
```

The original Digit Memory project used raw 8x8 digit pixels, byte hashes, and KDTree/BallTree lookup. Food Memory keeps the same engineering pattern but moves the geometry into a pretrained visual embedding space and uses FAISS for scale.

## What It Builds

- `food_memory/`: Python package for dataset loading, embedding, indexing, metadata, querying, evaluation, and benchmarking.
- `metadata.sqlite`: SQLite metadata for labels, splits, hashes, row ids, dimensions, and build settings.
- `train_embeddings.npy`: normalized CLIP image embeddings.
- `index_flat.faiss` and `index_hnsw.faiss`: exact and approximate inner-product indexes.
- `course/`: reproducible notebook generator for a course-style walkthrough.
- `scripts/generate_assets.py`: reproducible local report/deck/PDF-style asset generator.
- `docs/`: portfolio write-ups and sample result schema.

Generated data and binary artifacts live under `artifacts/` or `docs/generated/` and are gitignored.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The full pipeline downloads Food-101 from Hugging Face and CLIP weights from `openai/clip-vit-base-patch32`.

## Quick Start

Build a small per-class index:

```bash
python -m food_memory.build --mode quick --batch-size 16
```

Evaluate retrieval on a small test slice:

```bash
python -m food_memory.evaluate --mode quick --index hnsw --k 1 3 5
```

Add perturbation checks when you want the robustness curve:

```bash
python -m food_memory.evaluate --mode quick --index hnsw --robustness
```

Query a local image:

```bash
python -m food_memory.query path/to/food.jpg --artifact-dir artifacts/quick
python -m food_memory.query path/to/food.jpg --reject-below 0.25
```

Benchmark saved indexes without re-embedding:

```bash
python -m food_memory.bench --artifact-dir artifacts/quick
```

Generate the learning notebook:

```bash
python course/build_notebook.py
```

Generate portfolio assets from benchmark JSON:

```bash
python scripts/generate_assets.py --results docs/sample_results.json --output docs/generated
```

## Quick vs Full

| Mode | Purpose | Default data |
|---|---|---:|
| `quick` | Fast local iteration and smoke checks | 8 train images/class, 3 test images/class |
| `full` | Final Food-101 evaluation | Full train/test split |

Food-101 contains 101 classes with 750 training images and 250 manually reviewed test images per class. The dataset license is listed as unknown on Hugging Face, so image data should not be committed.

## Evaluation Plan

The evaluation command writes `results_<index>.json` with:

- top-1, top-3, and top-5 retrieval accuracy
- macro F1 and per-class accuracy
- p50/p95/p99 query latency
- similarity-threshold rejection curve for "unknown" handling
- optional robustness checks for JPEG recompression, crop/resize, brightness, and noise
- nearest-neighbor examples
- CLIP zero-shot baseline
- logistic regression baseline on frozen embeddings

The benchmark command writes `bench_results.json` with exact-hash latency, FAISS search latency, self-recall, and index size.

## Design Notes

This is not a deep training project. The point is to make retrieval carry as much of the work as it can:

- exact cache for repeated or byte-identical inputs
- CLIP embedding space for perceptual similarity
- FAISS flat search as the exact-vector baseline
- FAISS HNSW as the approximate search path
- lightweight classifiers only as baselines over frozen embeddings

That makes the project bigger than Digit Memory without losing the original thesis: the fastest model is still the one you do not need.
