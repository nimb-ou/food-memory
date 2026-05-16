"""Evaluate Food Memory retrieval and lightweight baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import DEFAULT_ENCODER, QUICK_TEST_PER_CLASS, artifact_dir_for_mode
from .datasets import iter_food101
from .embedding import CLIPImageEmbedder, label_prompts
from .memory import FoodMemory
from .metrics import macro_f1, per_class_accuracy, ranked_labels_from_neighbors, topk_accuracy
from .robustness import PERTURBATIONS


def evaluate(
    artifact_dir: Path,
    index: str,
    k_values: list[int],
    samples_per_class: int | None,
    encoder: str,
    device: str | None,
    seed: int,
    skip_baselines: bool,
    robustness: bool,
) -> dict:
    embedder = CLIPImageEmbedder(model_name=encoder, device=device)
    memory = FoodMemory.from_artifact_dir(artifact_dir, index=index, embedder=embedder)
    max_k = max(k_values)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_pred_topk: list[list[int]] = []
    latencies_us: list[float] = []
    confidences: list[float] = []
    examples: list[dict] = []

    test_samples = list(iter_food101("test", samples_per_class=samples_per_class, seed=seed))
    for sample in test_samples:
        result = memory.query_image(sample.image, k=max_k, vote_k=max_k)
        ranked = ranked_labels_from_neighbors(
            [n.label_id for n in result.neighbors],
            [n.score for n in result.neighbors],
            max_k,
        )
        y_true.append(sample.label_id)
        y_pred.append(result.label_id)
        y_pred_topk.append(ranked)
        latencies_us.append(result.latency_us)
        confidences.append(result.confidence)
        if len(examples) < 8:
            examples.append(
                {
                    "truth": sample.label_name,
                    "prediction": result.label_name,
                    "confidence": result.confidence,
                    "neighbors": [n.label_name for n in result.neighbors[:3]],
                }
            )

    retrieval = {
        "topk": {f"top_{k}": topk_accuracy(y_true, y_pred_topk, k) for k in k_values},
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": [s.__dict__ | {"accuracy": s.accuracy} for s in per_class_accuracy(y_true, y_pred)],
        "latency_us": _latency_summary(latencies_us),
        "rejection_curve": _rejection_curve(y_true, y_pred, confidences),
        "examples": examples,
    }

    baselines = {} if skip_baselines else _run_baselines(memory, embedder, test_samples, max_k)
    robustness_results = _run_robustness(memory, test_samples, max_k) if robustness else {}
    out = {
        "artifact_dir": str(artifact_dir),
        "index": index,
        "n_test": len(y_true),
        "retrieval": retrieval,
        "baselines": baselines,
        "robustness": robustness_results,
    }
    out_path = artifact_dir / f"results_{index}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _run_baselines(memory: FoodMemory, embedder: CLIPImageEmbedder, test_samples, max_k: int) -> dict:
    images = [s.image for s in test_samples]
    y_true = [s.label_id for s in test_samples]
    test_embeddings = embedder.embed_images(images, batch_size=32)

    prompts = label_prompts([memory.labels[i] for i in sorted(memory.labels)])
    text_embeddings = embedder.embed_texts(prompts, batch_size=64)
    zero_scores = test_embeddings @ text_embeddings.T
    zero_order = np.argsort(-zero_scores, axis=1)[:, :max_k]
    zero_pred = zero_order[:, 0].astype(int).tolist()
    zero_topk = zero_order.astype(int).tolist()

    baselines = {
        "clip_zero_shot": {
            "top_1": topk_accuracy(y_true, zero_topk, 1),
            "top_3": topk_accuracy(y_true, zero_topk, min(3, max_k)),
            "top_5": topk_accuracy(y_true, zero_topk, min(5, max_k)),
            "macro_f1": macro_f1(y_true, zero_pred),
        }
    }

    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return baselines

    train_y = np.array([r.label_id for r in memory.rows], dtype=np.int64)
    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    t0 = time.perf_counter()
    clf.fit(memory.embeddings, train_y)
    fit_s = time.perf_counter() - t0
    probs = clf.predict_proba(test_embeddings)
    order = np.argsort(-probs, axis=1)[:, :max_k]
    pred = order[:, 0].astype(int).tolist()
    topk = order.astype(int).tolist()
    baselines["logistic_regression_on_embeddings"] = {
        "fit_s": fit_s,
        "top_1": topk_accuracy(y_true, topk, 1),
        "top_3": topk_accuracy(y_true, topk, min(3, max_k)),
        "top_5": topk_accuracy(y_true, topk, min(5, max_k)),
        "macro_f1": macro_f1(y_true, pred),
    }
    return baselines


def _latency_summary(latencies_us: list[float]) -> dict[str, float]:
    if not latencies_us:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    arr = np.asarray(latencies_us, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }


def _rejection_curve(y_true: list[int], y_pred: list[int], confidences: list[float]) -> list[dict]:
    curve = []
    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        accepted = [i for i, c in enumerate(confidences) if c >= threshold]
        if accepted:
            correct = sum(1 for i in accepted if int(y_true[i]) == int(y_pred[i]))
            accuracy = correct / len(accepted)
        else:
            accuracy = 0.0
        curve.append(
            {
                "threshold": threshold,
                "coverage": len(accepted) / len(y_true) if y_true else 0.0,
                "accepted_accuracy": accuracy,
            }
        )
    return curve


def _run_robustness(memory: FoodMemory, test_samples, max_k: int) -> dict:
    out = {}
    for name, fn in PERTURBATIONS.items():
        y_true = []
        y_pred = []
        latencies = []
        for sample in test_samples:
            result = memory.query_image(fn(sample.image), k=max_k, vote_k=max_k)
            y_true.append(sample.label_id)
            y_pred.append(result.label_id)
            latencies.append(result.latency_us)
        out[name] = {
            "top_1": topk_accuracy(y_true, [[p] for p in y_pred], 1),
            "macro_f1": macro_f1(y_true, y_pred),
            "latency_us": _latency_summary(latencies),
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a built Food Memory index.")
    ap.add_argument("--artifact-dir", type=Path, default=artifact_dir_for_mode("quick"))
    ap.add_argument("--index", choices=["hnsw", "flat"], default="hnsw")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--samples-per-class", type=int, default=None)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--robustness", action="store_true", help="Run JPEG/crop/brightness/noise perturbation checks")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    samples = args.samples_per_class
    if samples is None and args.mode == "quick":
        samples = QUICK_TEST_PER_CLASS
    out = evaluate(
        artifact_dir=args.artifact_dir,
        index=args.index,
        k_values=args.k,
        samples_per_class=samples,
        encoder=args.encoder,
        device=args.device,
        seed=args.seed,
        skip_baselines=args.skip_baselines,
        robustness=args.robustness,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
