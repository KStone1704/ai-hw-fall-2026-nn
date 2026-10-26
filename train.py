"""Train on 55,000 examples; select a checkpoint using 5,000 validation examples."""

import argparse
import json
import platform
import random
import time
from pathlib import Path
import torch
import torchvision
from torch import nn
from data import load_data
from evaluate import evaluate
from models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mlp", "cnn", "transformer"), required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    train, validation, _ = load_data(args.data_dir, args.seed, args.batch_size)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    output = Path(args.output_dir)
    checkpoints = Path(args.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints / f"{args.model}.pt"
    history, best_accuracy, best_epoch = [], -1.0, None
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            correct += (logits.argmax(1).cpu() == labels).sum().item()
            total += labels.numel()
        validation_metrics = evaluate(model, validation, device)
        record = {"epoch": epoch, "train_loss": loss_sum / total,
                  "train_accuracy": correct / total,
                  "validation_loss": validation_metrics["loss"],
                  "validation_accuracy": validation_metrics["accuracy"]}
        history.append(record)
        print(json.dumps({"model": args.model, **record}), flush=True)
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation_metrics["accuracy"], epoch
            torch.save({"model": args.model, "seed": args.seed, "epoch": epoch,
                        "state_dict": model.state_dict()}, checkpoint)
    result = {"model": args.model, "seed": args.seed,
              "split": {"train": 55000, "validation": 5000, "test": 10000},
              "epochs": args.epochs, "selected_epoch": best_epoch,
              "best_validation_accuracy": best_accuracy,
              "batch_size": args.batch_size, "optimizer": "Adam", "learning_rate": 0.001,
              "augmentation": False, "normalization_mean_std": [0.1307, 0.3081],
              "parameters": sum(p.numel() for p in model.parameters()),
              "training_seconds": time.perf_counter() - start,
              "device": str(device), "python": platform.python_version(),
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "platform": platform.platform(), "history": history}
    (output / f"{args.model}-training.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Selected checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    main()
