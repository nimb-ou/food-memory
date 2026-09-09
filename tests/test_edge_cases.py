"""
Edge Case and Robustness Tests for Food Memory
Validates:
1. Degenerate image structures (1x1 single pixel, RGBA transparency, large monochrome).
2. Extreme neighbor count queries (k > n_samples) in NumpyVectorIndex.
3. Non-existent hash and label lookups in MetadataStore.
4. Bit-exact array and image digests on empty/edge inputs.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from food_memory.hashing import array_digest, image_digest
from food_memory.index import NumpyVectorIndex
from food_memory.metadata import MetadataStore, ImageRecord


class FoodMemoryEdgeCaseTests(unittest.TestCase):
    def test_degenerate_image_digests(self):
        """Verify image digests are deterministic and non-empty for unusual dimensions and color modes."""
        # 1x1 pixel image
        tiny = Image.new("RGB", (1, 1), (128, 64, 32))
        d_tiny = image_digest(tiny)
        self.assertEqual(len(d_tiny), 64)

        # RGBA image with alpha channel
        rgba = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        d_rgba = image_digest(rgba)
        self.assertEqual(len(d_rgba), 64)

        # Grayscale 'L' mode image
        gray = Image.new("L", (8, 8), 128)
        d_gray = image_digest(gray)
        self.assertEqual(len(d_gray), 64)

        # Identical pixels yield identical hash
        tiny_copy = Image.new("RGB", (1, 1), (128, 64, 32))
        self.assertEqual(d_tiny, image_digest(tiny_copy))

    def test_numpy_vector_index_excessive_k(self):
        """Verify requesting more neighbors than indexed vectors clamps safely without error."""
        vectors = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        idx = NumpyVectorIndex(vectors)

        # Request k = 10 when only 3 items exist
        res = idx.search(np.array([1.0, 0.0, 0.0]), k=10)
        self.assertEqual(len(res.indices), 3)
        self.assertEqual(len(res.scores), 3)
        self.assertEqual(res.indices[0], 0)
        self.assertAlmostEqual(res.scores[0], 1.0, places=5)

    def test_metadata_store_nonexistent_lookups(self):
        """Verify MetadataStore handles misses and non-existent lookups gracefully."""
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "meta.sqlite"
            store = MetadataStore(db_path)
            store.initialize(clear=True)

            # Query labels on empty store
            self.assertEqual(store.labels(), {})
            # Query non-existent image hash
            self.assertIsNone(store.get_by_image_hash("0000000000000000000000000000000000000000000000000000000000000000"))
            # Count records on empty store
            self.assertEqual(len(store.rows()), 0)

    def test_array_digest_distinction(self):
        """Verify array digest changes on small perturbations."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.001], dtype=np.float32)
        self.assertNotEqual(array_digest(a), array_digest(b))


if __name__ == "__main__":
    unittest.main()
