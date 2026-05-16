import unittest

from food_memory.metrics import macro_f1, ranked_labels_from_neighbors, topk_accuracy


class MetricsTests(unittest.TestCase):
    def test_topk_accuracy(self):
        self.assertEqual(topk_accuracy([1, 2], [[1, 3], [4, 2]], 2), 1.0)
        self.assertEqual(topk_accuracy([1, 2], [[3, 1], [4, 5]], 1), 0.0)

    def test_ranked_label_vote_uses_similarity(self):
        labels = [2, 1, 1, 2]
        scores = [0.5, 0.3, 0.3, 0.01]
        self.assertEqual(ranked_labels_from_neighbors(labels, scores, 4)[0], 1)

    def test_macro_f1_perfect(self):
        self.assertEqual(macro_f1([0, 1, 1], [0, 1, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()

