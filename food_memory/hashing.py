"""Stable hash keys for exact image and embedding memory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image


def _sha256(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def canonical_image_bytes(image: Image.Image) -> bytes:
    """Return deterministic bytes for exact image identity.

    The hash intentionally uses decoded RGB pixels plus dimensions rather than
    file bytes. The same visual image saved as PNG or JPEG will not collide
    unless decoding produces the same pixels, which is the desired exact-memory
    behavior for an image retrieval system.
    """
    rgb = image.convert("RGB")
    header = f"FMIMG1:{rgb.width}x{rgb.height}:RGB:".encode("ascii")
    return header + rgb.tobytes()


def image_digest(image: Image.Image) -> str:
    """Hash a PIL image after canonical RGB decoding."""
    return _sha256([canonical_image_bytes(image)])


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash raw file bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        _hash_stream(f, h, chunk_size)
    return h.hexdigest()


def _hash_stream(stream: BinaryIO, h: "hashlib._Hash", chunk_size: int) -> None:
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return
        h.update(chunk)


def array_digest(arr: np.ndarray, dtype: np.dtype | str = np.float32) -> str:
    """Hash an array with shape and dtype metadata included."""
    a = np.ascontiguousarray(arr, dtype=dtype)
    header = f"FMARR1:{a.dtype}:{','.join(map(str, a.shape))}:".encode("ascii")
    return _sha256([header, a.tobytes()])

