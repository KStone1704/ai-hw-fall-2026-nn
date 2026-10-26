import hashlib
import json
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from models import PatchTransformer, build_model


class ExperimentTests(unittest.TestCase):
    def test_parameter_budget_and_backward(self):
        torch.set_num_threads(2)
        counts = []
        for name in ("mlp", "cnn", "transformer"):
            model = build_model(name)
            counts.append(sum(p.numel() for p in model.parameters()))
            logits = model(torch.randn(3, 1, 28, 28))
            self.assertEqual(logits.shape, (3, 10))
            F.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(counts, [100975, 101129, 100970])
        self.assertLess((max(counts) - min(counts)) / min(counts), 0.002)

    def test_shared_patch_projection(self):
        model = PatchTransformer().eval()
        x = torch.randn(2, 1, 28, 28)
        patches = F.unfold(x, kernel_size=7, stride=7).transpose(1, 2)
        linear = F.linear(patches, model.patch.weight.flatten(1), model.patch.bias)
        convolution = model.patch(x).flatten(2).transpose(1, 2)
        torch.testing.assert_close(convolution, linear)
        torch.testing.assert_close(model.patch(x[:1]), model.patch(x)[:1])
        with torch.inference_mode():
            torch.testing.assert_close(model(x[:1]), model(x)[:1], rtol=1e-4, atol=1e-5)
        self.assertEqual(model.position.shape, (1, 17, 64))
        self.assertTrue(model.position.requires_grad)

    def test_saved_protocol(self):
        root = Path(__file__).parent / "results"
        if not (root / "protocol.json").exists():
            self.skipTest("Prepare stage has not run")
        protocol = json.loads((root / "protocol.json").read_text())
        expected = torch.randperm(60000, generator=torch.Generator().manual_seed(42))
        if (root / "split.pt").exists():
            split = torch.load(root / "split.pt", weights_only=True)
            train, validation = split["train_indices"], split["validation_indices"]
        else:
            # Binary indices are regenerated rather than committed to GitHub.
            train, validation = expected[:55000], expected[55000:]
        self.assertEqual(len(train), 55000)
        self.assertEqual(len(validation), 5000)
        self.assertEqual(len(torch.cat((train, validation)).unique()), 60000)
        for name, idx in (("train", train), ("validation", validation)):
            self.assertEqual(hashlib.sha256(idx.numpy().tobytes()).hexdigest(), protocol[f"{name}_indices_sha256"])
        torch.testing.assert_close(torch.cat((train, validation)), expected)
        self.assertEqual(protocol["normalization"]["pixels"], 55000 * 28 * 28)
        self.assertEqual(protocol["normalization"]["correction"], 0)
        for name, expected_hash in protocol["source_sha256"].items():
            self.assertEqual(hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest(), expected_hash)

    def test_saved_final_results(self):
        root = Path(__file__).parent / "results"
        if not (root / "test-results.json").exists():
            self.skipTest("Final test has not run")
        selection_bytes = (root / "selection.json").read_bytes()
        selection = json.loads(selection_bytes)
        result = json.loads((root / "test-results.json").read_text())
        self.assertEqual(hashlib.sha256(selection_bytes).hexdigest(), result["selection_sha256"])
        for name, record in result["results"].items():
            winner = selection["winners"][name]
            trials = [json.loads(path.read_text()) for path in sorted((root / "trials").glob(f"{name}-*.json"))]
            self.assertEqual(len(trials), 6)
            def metric_key(metric):
                return metric["correct"], -metric["loss"]
            best_trial = max(trials, key=lambda trial: metric_key(trial["best_validation"]))
            self.assertEqual(best_trial["trial"], winner["trial"])
            for trial in trials:
                best_epoch = max(trial["history"], key=lambda epoch: metric_key(epoch["validation"]))
                self.assertEqual(best_epoch["epoch"], trial["selected_epoch"])
                self.assertEqual(best_epoch["validation"], trial["best_validation"])
                for epoch in trial["history"]:
                    self.assertEqual(epoch["validation"]["total"], 5000)
                    self.assertEqual(epoch["validation"]["correct"] / 5000, epoch["validation"]["accuracy"])
            self.assertEqual(record["trial"], winner["trial"])
            self.assertEqual(record["checkpoint_epoch"], winner["selected_epoch"])
            metrics = record["test"]
            matrix = metrics["confusion_matrix_true_rows_predicted_columns"]
            self.assertEqual(sum(map(sum, matrix)), 10000)
            self.assertEqual(sum(matrix[i][i] for i in range(10)), metrics["correct"])
            self.assertEqual(metrics["correct"] / 10000, metrics["accuracy"])


if __name__ == "__main__":
    unittest.main()
