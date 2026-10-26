"""Independently verify fitted statistics using a histogram of training bytes."""

import argparse
import json
from pathlib import Path
import torch
from torchvision.datasets import MNIST
from experiment import load_training, write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol, train, validation = load_training(args.data_dir, args.output_dir, torch.device("cpu"))
    source = MNIST(args.data_dir, train=True, download=False)
    split = torch.load(args.output_dir / "split.pt", weights_only=True)
    counts = torch.bincount(source.data[split["train_indices"]].flatten().long(), minlength=256).double()
    values = torch.arange(256, dtype=torch.float64) / 255
    mean = (counts * values).sum() / counts.sum()
    std = ((counts * (values - mean).square()).sum() / counts.sum()).sqrt()
    norm = protocol["normalization"]
    assert abs(mean.item() - norm["mean"]) < 1e-12
    assert abs(std.item() - norm["std"]) < 1e-10
    normalized_std, normalized_mean = torch.std_mean(train[0].double(), correction=0)
    assert abs(normalized_mean.item()) < 1e-6
    assert abs(normalized_std.item() - 1) < 1e-6
    assert train[0].shape == (55000, 1, 28, 28)
    assert validation[0].shape == (5000, 1, 28, 28)
    result = {"histogram_mean": mean.item(), "histogram_std": std.item(),
              "normalized_training_mean": normalized_mean.item(),
              "normalized_training_std": normalized_std.item(),
              "passed": True, "test_split_loaded": False}
    write_json(args.output_dir / "preprocessing-check.json", result)
    print(json.dumps(result), flush=True)
