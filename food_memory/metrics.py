"""Evaluation helpers for retrieval classifiers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return row-normalized float32 vectors."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        denom = max(float(np.linalg.norm(a)), eps)
        return (a / denom).astype(np.float32)
    denom = np.linalg.norm(a, axis=1, keepdims=True)
    return (a / np.maximum(denom, eps)).astype(np.float32)


def topk_accuracy(y_true: Sequence[int], y_pred_topk: Sequence[Sequence[int]], k: int) -> float:
    """Compute top-k accuracy from ranked predicted label ids."""
    if len(y_true) == 0:
        return 0.0
    correct = 0
    for truth, preds in zip(y_true, y_pred_topk):
        if int(truth) in [int(p) for p in preds[:k]]:
            correct += 1
    return correct / len(y_true)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute macro F1 without requiring sklearn at import time."""
    labels = sorted(set(map(int, y_true)) | set(map(int, y_pred)))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if int(a) == label and int(b) == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if int(a) != label and int(b) == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if int(a) == label and int(b) != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


def ranked_labels_from_neighbors(label_ids: Sequence[int], scores: Sequence[float], k: int) -> list[int]:
    """Vote labels from neighbors, breaking ties by total similarity then best rank."""
    totals: dict[int, float] = defaultdict(float)
    best_rank: dict[int, int] = {}
    for rank, (label, score) in enumerate(zip(label_ids[:k], scores[:k])):
        label = int(label)
        totals[label] += float(score)
        best_rank.setdefault(label, rank)
    return [
        label
        for label, _ in sorted(
            totals.items(),
            key=lambda item: (-item[1], best_rank[item[0]], item[0]),
        )
    ]


def percentiles_us(timings_s: Sequence[float]) -> dict[str, float]:
    """Latency summary in microseconds."""
    if not timings_s:
        return {"n": 0, "min_us": 0.0, "p50_us": 0.0, "mean_us": 0.0, "p95_us": 0.0, "p99_us": 0.0, "max_us": 0.0}
    arr = np.asarray(timings_s, dtype=np.float64) * 1e6
    return {
        "n": int(arr.size),
        "min_us": float(np.min(arr)),
        "p50_us": float(np.percentile(arr, 50)),
        "mean_us": float(np.mean(arr)),
        "p95_us": float(np.percentile(arr, 95)),
        "p99_us": float(np.percentile(arr, 99)),
        "max_us": float(np.max(arr)),
    }


@dataclass(frozen=True)
class ConfusionSummary:
    label_id: int
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def per_class_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> list[ConfusionSummary]:
    counts = Counter(map(int, y_true))
    correct = Counter(int(t) for t, p in zip(y_true, y_pred) if int(t) == int(p))
    return [ConfusionSummary(label, counts[label], correct[label]) for label in sorted(counts)]

