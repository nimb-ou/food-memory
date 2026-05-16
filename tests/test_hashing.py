import unittest

import numpy as np
from PIL import Image

from food_memory.hashing import array_digest, image_digest


class HashingTests(unittest.TestCase):
    def test_image_digest_is_stable_and_strict(self):
        a = Image.new("RGB", (4, 4), (10, 20, 30))
        b = Image.new("RGB", (4, 4), (10, 20, 30))
        c = Image.new("RGB", (4, 4), (10, 20, 31))

        self.assertEqual(image_digest(a), image_digest(b))
        self.assertNotEqual(image_digest(a), image_digest(c))

    def test_array_digest_includes_shape(self):
        a = np.arange(6, dtype=np.float32)
        b = a.reshape(2, 3)

        self.assertNotEqual(array_digest(a), array_digest(b))
        self.assertEqual(array_digest(a), array_digest(np.arange(6, dtype=np.float64)))


if __name__ == "__main__":
    unittest.main()

