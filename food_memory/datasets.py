"""Food-101 dataset loading helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator

from PIL import Image

from .config import DATASET_ID


@dataclass(frozen=True)
class DatasetImage:
    source_id: str
    split: str
    label_id: int
    label_name: str
    image: Image.Image


def load_food101(split: str):
    """Load a Food-101 split through Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Food-101 loading requires `datasets`. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return load_dataset(DATASET_ID, split=split)


def label_names_from_dataset(dataset) -> list[str]:
    """Return stable label names from a Hugging Face Dataset."""
    feature = dataset.features["label"]
    names = getattr(feature, "names", None)
    if not names:
        raise RuntimeError("dataset label feature does not expose class names")
    return list(names)


def iter_food101(
    split: str,
    samples_per_class: int | None = None,
    seed: int = 42,
) -> Iterator[DatasetImage]:
    """Yield Food-101 examples, optionally capped per class.

    The cap is applied after a deterministic shuffle so quick mode still sees a
    varied subset from each class.
    """
    dataset = load_food101(split)
    label_names = label_names_from_dataset(dataset)
    source = dataset.shuffle(seed=seed) if samples_per_class is not None else dataset
    counts: Counter[int] = Counter()
    yielded = 0
    target_total = None if samples_per_class is None else samples_per_class * len(label_names)

    for idx, row in enumerate(source):
        label_id = int(row["label"])
        if samples_per_class is not None and counts[label_id] >= samples_per_class:
            continue
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise RuntimeError(f"expected PIL image from dataset row, got {type(image)!r}")
        counts[label_id] += 1
        yielded += 1
        yield DatasetImage(
            source_id=f"{split}:{idx}",
            split=split,
            label_id=label_id,
            label_name=label_names[label_id],
            image=image,
        )
        if target_total is not None and yielded >= target_total:
            return

