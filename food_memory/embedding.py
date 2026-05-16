"""Image and text embedding backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .config import DEFAULT_ENCODER, LABEL_PROMPT
from .metrics import l2_normalize


class ImageTextEmbedder(Protocol):
    """Protocol implemented by CLIP and test embedders."""

    dim: int

    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        ...

    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        ...


@dataclass
class CLIPImageEmbedder:
    """CLIP image/text embedder using Hugging Face Transformers."""

    model_name: str = DEFAULT_ENCODER
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP embedding requires `torch` and `transformers`. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

        self._torch = torch
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.projection_dim)

    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        chunks: list[np.ndarray] = []
        torch = self._torch
        batches = list(_batches(images, batch_size))
        iterator = tqdm(batches, desc="CLIP image batches", unit="batch", leave=False) if len(batches) > 1 else batches
        with torch.no_grad():
            for batch in iterator:
                rgb = [im.convert("RGB") for im in batch]
                inputs = self.processor(images=rgb, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_image_features(**inputs)
                chunks.append(_tensor_to_numpy(feats))
        return l2_normalize(np.vstack(chunks))

    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        chunks: list[np.ndarray] = []
        torch = self._torch
        batches = list(_batches(texts, batch_size))
        iterator = tqdm(batches, desc="CLIP text batches", unit="batch", leave=False) if len(batches) > 1 else batches
        with torch.no_grad():
            for batch in iterator:
                inputs = self.processor(text=list(batch), return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_text_features(**inputs)
                chunks.append(_tensor_to_numpy(feats))
        return l2_normalize(np.vstack(chunks))


def label_prompts(label_names: Iterable[str]) -> list[str]:
    """Convert Food-101 label names into CLIP prompts."""
    return [LABEL_PROMPT.format(name.replace("_", " ")) for name in label_names]


def _batches(items: Sequence, size: int):
    for i in range(0, len(items), int(size)):
        yield items[i : i + int(size)]


def _tensor_to_numpy(features) -> np.ndarray:
    """Handle tensor and model-output return types across Transformers versions."""
    if hasattr(features, "detach"):
        tensor = features
    elif hasattr(features, "pooler_output"):
        tensor = features.pooler_output
    elif isinstance(features, (tuple, list)) and features and hasattr(features[0], "detach"):
        tensor = features[0]
    else:
        raise TypeError(f"unsupported CLIP feature output type: {type(features)!r}")
    return tensor.detach().cpu().numpy().astype(np.float32)
