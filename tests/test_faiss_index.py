import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from food_memory.index import FAISSVectorIndex


@unittest.skipIf(importlib.util.find_spec("faiss") is None, "faiss-cpu is not installed")
class FaissIndexTests(unittest.TestCase):
    def test_faiss_save_load_search(self):
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "index_flat.faiss"
            FAISSVectorIndex("flat").build(vectors).save(path)
            loaded = FAISSVectorIndex.load(path, kind="flat")
            result = loaded.search(np.asarray([1.0, 0.0], dtype=np.float32), k=1)
            self.assertEqual(int(result.indices[0]), 0)


if __name__ == "__main__":
    unittest.main()

