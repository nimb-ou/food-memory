"""Build the Food Memory course notebook.

Run from the repository root:

    python course/build_notebook.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

NB_PATH = Path(__file__).parent / "food_memory_course.ipynb"
cells: list[dict] = []
counter = 0


def _id() -> str:
    global counter
    counter += 1
    return f"cell-{counter:03d}"


def _lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines(keepends=True)


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "id": _id(), "metadata": {}, "source": _lines(text)})


def code(text: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "id": _id(),
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _lines(text),
        }
    )


md(
    r"""
    # Food Memory - Retrieval-First Image Classification

    This notebook is generated from `course/build_notebook.py` so the learning
    path stays reproducible.

    The thesis is the same as Digit Memory: before training a model, build a
    memory and measure it. The larger Food-101 setting changes the representation
    from raw pixels to CLIP embeddings and the index from a small KDTree to FAISS.
    """
)

md(
    r"""
    ## 1. Architecture

    ```
    query image
        |
        v
    canonical RGB hash -> exact hit -> stored label
        |
       miss
        |
        v
    CLIP image embedding -> FAISS vector search -> k-NN vote
    ```

    Exact memory answers repeats. Vector search answers new but visually similar
    food images. Both paths report latency and neighbor evidence.
    """
)

code(
    r"""
    from pathlib import Path
    import json

    import numpy as np
    from PIL import Image

    from food_memory.hashing import image_digest, array_digest
    from food_memory.metrics import l2_normalize

    print("Food Memory imports loaded.")
    """
)

md(
    r"""
    ## 2. Exact Image Hashing

    The exact path uses decoded RGB pixels and dimensions, not raw file bytes.
    That makes the hash stable for the in-memory image object while still strict
    enough that recompression, crop, or color changes fall through to retrieval.
    """
)

code(
    r"""
    image = Image.new("RGB", (8, 8), color=(220, 80, 40))
    same = Image.new("RGB", (8, 8), color=(220, 80, 40))
    changed = Image.new("RGB", (8, 8), color=(220, 80, 41))

    print(image_digest(image) == image_digest(same))
    print(image_digest(image) == image_digest(changed))
    """
)

md(
    r"""
    ## 3. Embedding Space

    In the full project, images are embedded with
    `openai/clip-vit-base-patch32`. Embeddings are normalized so inner product is
    cosine similarity. This lets FAISS use fast maximum-inner-product search.
    """
)

code(
    r"""
    toy = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    print(l2_normalize(toy))
    print(array_digest(l2_normalize(toy)))
    """
)

md(
    r"""
    ## 4. Build And Evaluate

    Use quick mode first:

    ```bash
    python -m food_memory.build --mode quick --batch-size 16
    python -m food_memory.evaluate --mode quick --index hnsw --k 1 3 5
    python -m food_memory.bench --artifact-dir artifacts/quick
    ```

    Then run full mode when you are ready to spend the download and embedding
    time:

    ```bash
    python -m food_memory.build --mode full --batch-size 32
    python -m food_memory.evaluate --mode full --index hnsw --k 1 3 5
    ```
    """
)

md(
    r"""
    ## 5. Reading The Results

    Look for four things:

    1. Retrieval top-1/top-5 accuracy.
    2. HNSW speed versus flat FAISS search.
    3. CLIP zero-shot and logistic-regression baselines.
    4. Neighbor examples that make successes and failures inspectable.

    If retrieval is competitive, you have an explainable baseline with no deep
    fine-tuning. If it fails, the failure cases tell you what representation or
    model training needs to fix.
    """
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
