"""Image perturbations for robustness checks."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance


def jpeg_recompress(image: Image.Image, quality: int = 50) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def center_crop_resize(image: Image.Image, crop_fraction: float = 0.85) -> Image.Image:
    rgb = image.convert("RGB")
    w, h = rgb.size
    cw, ch = int(w * crop_fraction), int(h * crop_fraction)
    left = max((w - cw) // 2, 0)
    top = max((h - ch) // 2, 0)
    cropped = rgb.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.Resampling.BICUBIC)


def adjust_brightness(image: Image.Image, factor: float = 0.65) -> Image.Image:
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(float(factor))


def gaussian_noise(image: Image.Image, sigma: float = 8.0, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    arr = np.clip(arr + rng.normal(0, sigma, size=arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


PERTURBATIONS = {
    "jpeg_q50": jpeg_recompress,
    "center_crop_85": center_crop_resize,
    "brightness_65": adjust_brightness,
    "noise_sigma_8": gaussian_noise,
}

