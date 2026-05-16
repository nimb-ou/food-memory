"""Shared defaults for the Food Memory project."""

from __future__ import annotations

from pathlib import Path

DATASET_ID = "ethz/food101"
DEFAULT_ENCODER = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
DEFAULT_ARTIFACT_ROOT = Path("artifacts")

# Quick mode is intentionally small enough for iteration on a laptop.
QUICK_TRAIN_PER_CLASS = 8
QUICK_TEST_PER_CLASS = 3
FULL_TRAIN_PER_CLASS = None
FULL_TEST_PER_CLASS = None

LABEL_PROMPT = "a photo of {}"


def artifact_dir_for_mode(mode: str, root: Path | str = DEFAULT_ARTIFACT_ROOT) -> Path:
    """Return the conventional artifact directory for a build mode."""
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'")
    return Path(root) / mode

