"""Evaluate one already selected checkpoint on the official test split."""

import argparse
import json
from pathlib import Path
import torch
from torch import nn
from data import load_data
from models import build_model


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        loss_sum += criterion(logits, labels).item()
        correct += (predictions == labels).sum().item()
        total += labels.numel()
        bins = (10 * labels + predictions).cpu()
        confusion += torch.bincount(bins, minlength=100).reshape(10, 10)
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "correct": correct, "total": total,
            "confusion_matrix_true_rows_predicted_columns": confusion.tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["model"])
    model.load_state_dict(saved["state_dict"])
    _, _, test = load_data(args.data_dir, seed=saved["seed"])
    result = {"model": saved["model"], "checkpoint_epoch": saved["epoch"],
              "test": evaluate(model, test, "cpu")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
