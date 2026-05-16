"""CLI for querying a built Food Memory index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_ENCODER, artifact_dir_for_mode
from .embedding import CLIPImageEmbedder
from .memory import FoodMemory


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Query Food Memory with an image.")
    ap.add_argument("image", type=Path)
    ap.add_argument("--artifact-dir", type=Path, default=artifact_dir_for_mode("quick"))
    ap.add_argument("--index", choices=["hnsw", "flat"], default="hnsw")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reject-below", type=float, default=None, help="Return unknown when similarity is below this threshold")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    embedder = CLIPImageEmbedder(model_name=args.encoder, device=args.device)
    mem = FoodMemory.from_artifact_dir(args.artifact_dir, index=args.index, embedder=embedder)
    result = mem.query_image(args.image, k=args.k)
    out = result.to_dict()
    if args.reject_below is not None and result.confidence < args.reject_below:
        out["accepted"] = False
        out["rejected_as_unknown"] = True
    else:
        out["accepted"] = True
        out["rejected_as_unknown"] = False
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
