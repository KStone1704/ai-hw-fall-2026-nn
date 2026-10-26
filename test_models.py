import unittest
import torch
from models import build_model


class ModelTests(unittest.TestCase):
    def test_output_shape_and_backward(self):
        torch.set_num_threads(2)
        for name in ("mlp", "cnn", "transformer"):
            with self.subTest(model=name):
                model = build_model(name)
                logits = model(torch.randn(3, 1, 28, 28))
                self.assertEqual(logits.shape, (3, 10))
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(any(p.grad is not None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
