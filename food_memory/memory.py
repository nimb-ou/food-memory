"""Runtime query engine for Food Memory."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .embedding import ImageTextEmbedder
from .hashing import image_digest
from .index import load_index
from .metadata import ImageRecord, MetadataStore
from .metrics import ranked_labels_from_neighbors


@dataclass(frozen=True)
class Neighbor:
    row_id: int
    label_id: int
    label_name: str
    score: float
    source_id: str
    image_hash: str


@dataclass(frozen=True)
class QueryResult:
    label_id: int
    label_name: str
    confidence: float
    exact_hit: bool
    latency_us: float
    image_hash: str
    neighbors: list[Neighbor]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "label_name": self.label_name,
            "confidence": self.confidence,
            "exact_hit": self.exact_hit,
            "latency_us": self.latency_us,
            "image_hash": self.image_hash,
            "neighbors": [asdict(n) for n in self.neighbors],
        }


class FoodMemory:
    """Hash-first, vector-search-fallback Food-101 classifier."""

    def __init__(
        self,
        metadata_path: str | Path,
        embeddings_path: str | Path,
        index_path: str | Path,
        embedder: ImageTextEmbedder | None = None,
        index_kind: str | None = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.embeddings_path = Path(embeddings_path)
        self.index_path = Path(index_path)
        self.store = MetadataStore(self.metadata_path)
        self.rows = self.store.rows(split="train")
        self.labels = self.store.labels()
        self.embeddings = np.load(self.embeddings_path).astype(np.float32)
        self.index = load_index(self.index_path, kind=index_kind)
        self.embedder = embedder
        self._rows_by_id = {row.row_id: row for row in self.rows}
        self._exact = {row.image_hash: row for row in self.rows}

        if len(self.rows) != len(self.embeddings):
            raise RuntimeError(
                f"metadata rows ({len(self.rows)}) and embeddings ({len(self.embeddings)}) do not match"
            )

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        index: str = "hnsw",
        embedder: ImageTextEmbedder | None = None,
    ) -> "FoodMemory":
        artifact_dir = Path(artifact_dir)
        return cls(
            metadata_path=artifact_dir / "metadata.sqlite",
            embeddings_path=artifact_dir / "train_embeddings.npy",
            index_path=artifact_dir / f"index_{index}.faiss",
            embedder=embedder,
            index_kind=index,
        )

    def query_image(self, image_or_path: str | Path | Image.Image, k: int = 5, vote_k: int | None = None) -> QueryResult:
        image = _open_image(image_or_path)
        digest = image_digest(image)
        t0 = time.perf_counter()

        exact = self._exact.get(digest)
        if exact is not None:
            latency_us = (time.perf_counter() - t0) * 1e6
            neighbor = Neighbor(
                row_id=exact.row_id,
                label_id=exact.label_id,
                label_name=exact.label_name,
                score=1.0,
                source_id=exact.source_id,
                image_hash=exact.image_hash,
            )
            return QueryResult(
                label_id=exact.label_id,
                label_name=exact.label_name,
                confidence=1.0,
                exact_hit=True,
                latency_us=latency_us,
                image_hash=digest,
                neighbors=[neighbor],
            )

        if self.embedder is None:
            raise RuntimeError("query miss requires an embedder; pass CLIPImageEmbedder or another ImageTextEmbedder")
        vector = self.embedder.embed_images([image], batch_size=1)[0]
        result = self.query_embedding(vector, k=k, vote_k=vote_k, image_hash=digest, started_at=t0)
        return result

    def query_embedding(
        self,
        vector: np.ndarray,
        k: int = 5,
        vote_k: int | None = None,
        image_hash: str = "",
        started_at: float | None = None,
    ) -> QueryResult:
        t0 = started_at if started_at is not None else time.perf_counter()
        result = self.index.search(vector, k=k)
        neighbors = self._neighbors_from_search(result.indices, result.scores)
        vote_labels = ranked_labels_from_neighbors(
            [n.label_id for n in neighbors],
            [n.score for n in neighbors],
            vote_k or k,
        )
        label_id = vote_labels[0] if vote_labels else neighbors[0].label_id
        latency_us = (time.perf_counter() - t0) * 1e6
        return QueryResult(
            label_id=label_id,
            label_name=self.labels.get(label_id, str(label_id)),
            confidence=float(neighbors[0].score) if neighbors else 0.0,
            exact_hit=False,
            latency_us=latency_us,
            image_hash=image_hash,
            neighbors=neighbors,
        )

    def _neighbors_from_search(self, indices: Sequence[int], scores: Sequence[float]) -> list[Neighbor]:
        neighbors: list[Neighbor] = []
        for idx, score in zip(indices, scores):
            if int(idx) < 0:
                continue
            row = self._rows_by_id[int(idx)]
            neighbors.append(
                Neighbor(
                    row_id=row.row_id,
                    label_id=row.label_id,
                    label_name=row.label_name,
                    score=float(score),
                    source_id=row.source_id,
                    image_hash=row.image_hash,
                )
            )
        return neighbors


def _open_image(image_or_path: str | Path | Image.Image) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path.convert("RGB")
    return Image.open(image_or_path).convert("RGB")

