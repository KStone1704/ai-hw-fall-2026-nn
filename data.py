"""Download the official MNIST splits, then make a train-only validation split."""

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST


def load_data(root, seed=42, batch_size=256, download=False):
    source = MNIST(root, train=True, download=download)
    test = MNIST(root, train=False, download=download)
    permutation = torch.randperm(60000, generator=torch.Generator().manual_seed(seed))
    train_indices, validation_indices = permutation[:55000], permutation[55000:]

    def dataset(images, labels):
        images = (images.unsqueeze(1).float() / 255.0 - 0.1307) / 0.3081
        return TensorDataset(images, labels)

    train = dataset(source.data[train_indices], source.targets[train_indices])
    validation = dataset(source.data[validation_indices], source.targets[validation_indices])
    held_out = dataset(test.data, test.targets)
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(validation, batch_size=batch_size, shuffle=False),
        DataLoader(held_out, batch_size=batch_size, shuffle=False),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    for split in (True, False):
        dataset = MNIST(args.data_dir, train=split, download=True)
        print("train" if split else "test", len(dataset), flush=True)
