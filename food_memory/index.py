"""Vector index backends for Food Memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import l2_normalize


class MissingFaissError(RuntimeError):
    """Raised when FAISS-backed operations are requested without faiss-cpu."""


def _import_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise MissingFaissError(
            "FAISS is required for this operation. Install with `pip install faiss-cpu==1.13.2`."
        ) from exc
    return faiss


@dataclass
class SearchResult:
    scores: np.ndarray
    indices: np.ndarray


class FAISSVectorIndex:
    """Small wrapper around FAISS inner-product indexes."""

    def __init__(self, kind: str = "hnsw", dim: int | None = None, hnsw_m: int = 32):
        if kind not in {"flat", "hnsw"}:
            raise ValueError("kind must be 'flat' or 'hnsw'")
        self.kind = kind
        self.dim = dim
        self.hnsw_m = hnsw_m
        self.index = None

    def build(self, vectors: np.ndarray) -> "FAISSVectorIndex":
        faiss = _import_faiss()
        x = l2_normalize(vectors)
        self.dim = int(x.shape[1])
        if self.kind == "flat":
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 80
            self.index.hnsw.efSearch = 64
        self.index.add(np.ascontiguousarray(x, dtype=np.float32))
        return self

    def search(self, query: np.ndarray, k: int) -> SearchResult:
        if self.index is None:
            raise RuntimeError("index has not been built or loaded")
        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))
        scores, indices = self.index.search(q, int(k))
        return SearchResult(scores=scores[0], indices=indices[0])

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("index has not been built or loaded")
        faiss = _import_faiss()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path, kind: str = "hnsw") -> "FAISSVectorIndex":
        faiss = _import_faiss()
        obj = cls(kind=kind)
        obj.index = faiss.read_index(str(path))
        obj.dim = int(obj.index.d)
        return obj


class NumpyVectorIndex:
    """Tiny exact inner-product index used for smoke tests and fallbacks."""

    def __init__(self, vectors: np.ndarray | None = None):
        self.vectors = l2_normalize(vectors) if vectors is not None else None

    def build(self, vectors: np.ndarray) -> "NumpyVectorIndex":
        self.vectors = l2_normalize(vectors)
        return self

    def search(self, query: np.ndarray, k: int) -> SearchResult:
        if self.vectors is None:
            raise RuntimeError("index has not been built or loaded")
        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)[: int(k)]
        return SearchResult(scores=scores[order].astype(np.float32), indices=order.astype(np.int64))

    def save(self, path: str | Path) -> None:
        if self.vectors is None:
            raise RuntimeError("index has not been built or loaded")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vectors=self.vectors)

    @classmethod
    def load(cls, path: str | Path) -> "NumpyVectorIndex":
        data = np.load(path)
        return cls(vectors=data["vectors"])


def load_index(path: str | Path, kind: str | None = None):
    """Load a FAISS `.faiss` index or test `.npz` index."""
    path = Path(path)
    if path.suffix == ".npz":
        return NumpyVectorIndex.load(path)
    resolved_kind = kind or ("flat" if "flat" in path.stem else "hnsw")
    return FAISSVectorIndex.load(path, kind=resolved_kind)

