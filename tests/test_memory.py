import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from food_memory.hashing import array_digest, image_digest
from food_memory.index import NumpyVectorIndex
from food_memory.memory import FoodMemory
from food_memory.metadata import ImageRecord, MetadataStore


class FakeEmbedder:
    dim = 3

    def embed_images(self, images, batch_size=32):
        vectors = []
        for image in images:
            digest = image_digest(image)
            if digest == image_digest(Image.new("RGB", (4, 4), (0, 0, 255))):
                vectors.append([0.0, 0.9, 0.1])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)

    def embed_texts(self, texts, batch_size=64):
        return np.zeros((len(texts), self.dim), dtype=np.float32)


class MemoryTests(unittest.TestCase):
    def test_exact_and_fallback_query(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = MetadataStore(root / "metadata.sqlite")
            store.initialize(clear=True)
            labels = {0: "apple_pie", 1: "ramen"}
            for label_id, label_name in labels.items():
                store.insert_label(label_id, label_name)

            red = Image.new("RGB", (4, 4), (255, 0, 0))
            green = Image.new("RGB", (4, 4), (0, 255, 0))
            vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
            rows = [
                ImageRecord(0, "train:0", "train", 0, "apple_pie", image_digest(red), array_digest(vectors[0]), 4, 4),
                ImageRecord(1, "train:1", "train", 1, "ramen", image_digest(green), array_digest(vectors[1]), 4, 4),
            ]
            store.insert_records(rows)
            np.save(root / "train_embeddings.npy", vectors)
            NumpyVectorIndex(vectors).save(root / "index_test.npz")

            memory = FoodMemory(
                metadata_path=root / "metadata.sqlite",
                embeddings_path=root / "train_embeddings.npy",
                index_path=root / "index_test.npz",
                embedder=FakeEmbedder(),
            )

            exact = memory.query_image(red, k=2)
            self.assertTrue(exact.exact_hit)
            self.assertEqual(exact.label_name, "apple_pie")

            blue = Image.new("RGB", (4, 4), (0, 0, 255))
            miss = memory.query_image(blue, k=2)
            self.assertFalse(miss.exact_hit)
            self.assertEqual(miss.label_name, "ramen")
            self.assertGreaterEqual(len(miss.neighbors), 1)


if __name__ == "__main__":
    unittest.main()

