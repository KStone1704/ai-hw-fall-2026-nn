"""Sanity-check the saved experiment records without downloading any data."""

import json
from pathlib import Path
import unittest


class ResultsTests(unittest.TestCase):
    def test_saved_results_are_consistent(self):
        for name in ("mlp", "cnn", "transformer"):
            with self.subTest(model=name):
                root = Path(__file__).resolve().parent / "results"
                training = json.loads((root / f"{name}-training.json").read_text())
                result = json.loads((root / f"{name}-test.json").read_text())
                self.assertEqual(training["model"], result["model"])
                self.assertEqual(training["selected_epoch"], result["checkpoint_epoch"])
                self.assertEqual(len(training["history"]), training["epochs"])
                test = result["test"]
                matrix = test["confusion_matrix_true_rows_predicted_columns"]
                self.assertEqual(len(matrix), 10)
                self.assertTrue(all(len(row) == 10 for row in matrix))
                self.assertTrue(all(isinstance(n, int) and n >= 0 for row in matrix for n in row))
                self.assertEqual(sum(map(sum, matrix)), test["total"])
                self.assertEqual(sum(matrix[i][i] for i in range(10)), test["correct"])
                self.assertEqual(test["total"], 10000)
                self.assertAlmostEqual(test["correct"] / test["total"], test["accuracy"])


if __name__ == "__main__":
    unittest.main()
