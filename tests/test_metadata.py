import tempfile
import unittest
from pathlib import Path

from food_memory.metadata import ImageRecord, MetadataStore


class MetadataTests(unittest.TestCase):
    def test_round_trip_record_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "metadata.sqlite"
            store = MetadataStore(path)
            store.initialize(clear=True)
            store.insert_label(7, "ramen")
            store.set_metadata("build", {"mode": "quick"})
            record = ImageRecord(
                row_id=0,
                source_id="train:0",
                split="train",
                label_id=7,
                label_name="ramen",
                image_hash="img",
                embedding_hash="emb",
                width=32,
                height=32,
            )
            store.insert_record(record)

            self.assertEqual(store.labels(), {7: "ramen"})
            self.assertEqual(store.get_metadata("build")["mode"], "quick")
            self.assertEqual(store.count("train"), 1)
            self.assertEqual(store.get_by_image_hash("img"), record)


if __name__ == "__main__":
    unittest.main()

