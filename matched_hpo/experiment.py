"""Separate prepare, validation-only tuning, and locked final-test stages."""

import argparse
import hashlib
import itertools
import json
import platform
import random
import time
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision.datasets import MNIST
from models import build_model

NAMES = ("mlp", "cnn", "transformer")
SEED = 42
LEARNING_RATES = (0.0003, 0.001, 0.003)
WEIGHT_DECAYS = (0.0, 0.0001)
EPOCHS = 8
PATIENCE = 3
BATCH_SIZE = 256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def seed_all():
    random.seed(SEED)
    torch.manual_seed(SEED)


def setup_device(name):
    torch.set_num_threads(4)
    # Use the ordinary attention path consistently in training and evaluation.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable. No silent CPU fallback is permitted.")
    return torch.device(name)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def prepare(root, output):
    if (output / "protocol.json").exists():
        raise FileExistsError("Protocol already exists; use a fresh output directory.")
    # Only the official TRAINING split is opened in this stage.
    source = MNIST(root, train=True, download=False)
    permutation = torch.randperm(len(source), generator=torch.Generator().manual_seed(SEED))
    train_idx, val_idx = permutation[:55000], permutation[55000:]
    # Population statistics of scaled training pixels, accumulated in float64.
    pixels = source.data[train_idx].to(torch.float64).div_(255)
    std, mean = torch.std_mean(pixels, correction=0)
    split = {"train_indices": train_idx, "validation_indices": val_idx}
    output.mkdir(parents=True, exist_ok=True)
    torch.save(split, output / "split.pt")
    protocol = {
        "seed": SEED, "split_sizes": {"train": 55000, "validation": 5000, "test": 10000},
        "split_method": "seed-42 random permutation of the official training split; no k-fold",
        "train_indices_sha256": digest(train_idx), "validation_indices_sha256": digest(val_idx),
        "class_counts": {
            "train": torch.bincount(source.targets[train_idx], minlength=10).tolist(),
            "validation": torch.bincount(source.targets[val_idx], minlength=10).tolist()},
        "normalization": {"mean": mean.item(), "std": std.item(), "correction": 0,
                          "fitted_on": "55000 training images only", "pixels": pixels.numel(),
                          "accumulation_dtype": "float64", "model_input_dtype": "float32"},
        "parameters": {name: sum(p.numel() for p in build_model(name).parameters()) for name in NAMES},
        "search": {"learning_rates": LEARNING_RATES, "weight_decays": WEIGHT_DECAYS,
                   "trials_per_model": 6, "max_epochs": EPOCHS, "patience": PATIENCE,
                   "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
                   "batch_size": BATCH_SIZE, "scheduler": None, "augmentation": False,
                   "selection": "highest validation correct count, then lowest validation loss; earlier trial/epoch on exact ties",
                   "architecture_search": False, "precision": "float32"},
        "test_policy": "Do not load test data during tuning. Lock all three winners before final evaluation. Baseline test results were seen previously.",
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("models.py", "experiment.py")}}
    write_json(output / "protocol.json", protocol)
    print(json.dumps({"prepared": protocol}), flush=True)


def load_training(root, output, device):
    protocol = json.loads((output / "protocol.json").read_text())
    split = torch.load(output / "split.pt", weights_only=True)
    source = MNIST(root, train=True, download=False)
    norm = protocol["normalization"]
    def subset(indices):
        x = source.data[indices].unsqueeze(1).float().div_(255)
        x.sub_(norm["mean"]).div_(norm["std"])
        return x.to(device), source.targets[indices].to(device)
    return protocol, subset(split["train_indices"]), subset(split["validation_indices"])


@torch.inference_mode()
def evaluate(model, dataset, batch_size=BATCH_SIZE, confusion=False):
    model.eval()
    images, labels = dataset
    loss_sum = torch.zeros((), device=images.device)
    correct = torch.zeros((), dtype=torch.int64, device=images.device)
    predictions = []
    for start in range(0, len(labels), batch_size):
        x, y = images[start:start + batch_size], labels[start:start + batch_size]
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum")
        pred = logits.argmax(1)
        correct += (pred == y).sum()
        if confusion:
            predictions.append(pred.cpu())
    count = int(correct.item())
    result = {"loss": loss_sum.item() / len(labels), "correct": count,
              "total": len(labels), "accuracy": count / len(labels)}
    if confusion:
        pred = torch.cat(predictions)
        bins = labels.cpu() * 10 + pred
        result["confusion_matrix_true_rows_predicted_columns"] = torch.bincount(bins, minlength=100).reshape(10, 10).tolist()
    return result


def key(metrics):
    return metrics["correct"], -metrics["loss"]


def tune(root, output, device):
    if (output / "selection.json").exists():
        raise FileExistsError("Selection is already locked; use a fresh output directory to retune.")
    protocol, train, validation = load_training(root, output, device)
    winners = {}
    for name in NAMES:
        records = []
        for trial, (lr, decay) in enumerate(itertools.product(LEARNING_RATES, WEIGHT_DECAYS), 1):
            record_path = output / "trials" / f"{name}-{trial}.json"
            checkpoint = output / "checkpoints" / f"{name}-{trial}.pt"
            if record_path.exists():
                records.append(json.loads(record_path.read_text()))
                continue
            seed_all()
            model = build_model(name).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
            generator = torch.Generator().manual_seed(SEED)
            best, best_epoch, stale, history = None, None, 0, []
            synchronize(device)
            started = time.perf_counter()
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = torch.randperm(len(train[1]), generator=generator).to(device)
                loss_sum = torch.zeros((), device=device)
                correct = torch.zeros((), dtype=torch.int64, device=device)
                for idx in order.split(BATCH_SIZE):
                    x, y = train[0][idx], train[1][idx]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.detach() * len(idx)
                    correct += (logits.detach().argmax(1) == y).sum()
                metrics = evaluate(model, validation)
                epoch_record = {"epoch": epoch, "online_train_loss": loss_sum.item() / len(train[1]),
                                "online_train_accuracy": correct.item() / len(train[1]), "validation": metrics}
                history.append(epoch_record)
                print(json.dumps({"model": name, "trial": trial, "lr": lr, "weight_decay": decay, **epoch_record}), flush=True)
                if best is None or key(metrics) > key(best):
                    best, best_epoch, stale = metrics, epoch, 0
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": name, "trial": trial, "epoch": epoch,
                                "normalization": protocol["normalization"],
                                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, checkpoint)
                else:
                    stale += 1
                    if stale >= PATIENCE:
                        break
            synchronize(device)
            record = {"model": name, "trial": trial, "learning_rate": lr, "weight_decay": decay,
                      "selected_epoch": best_epoch, "best_validation": best, "history": history,
                      "training_seconds": time.perf_counter() - started,
                      "checkpoint": str(checkpoint.relative_to(output)), "device": str(device)}
            write_json(record_path, record)
            records.append(record)
            del model, optimizer
        winner = max(records, key=lambda record: key(record["best_validation"]))
        winners[name] = winner
        print(json.dumps({"winner": name, "trial": winner["trial"], "validation": winner["best_validation"]}), flush=True)
    write_json(output / "selection.json", {"locked_before_test": True, "winners": winners,
               "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                               "torchvision": str(torchvision.__version__), "platform": platform.platform(),
                               "device": str(device), "mps_built": torch.backends.mps.is_built(),
                               "mps_available": torch.backends.mps.is_available(), "cpu_threads": 4,
                               "deterministic_algorithms": True, "mha_fastpath": False,
                               "mps_cpu_fallback_environment": __import__("os").environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset")}})


def final_test(root, output, device):
    if (output / "test-results.json").exists():
        raise FileExistsError("Final test results already exist; refusing a duplicate test run.")
    selection_bytes = (output / "selection.json").read_bytes()
    selection = json.loads(selection_bytes)
    if not selection["locked_before_test"] or set(selection["winners"]) != set(NAMES):
        raise RuntimeError("All model selections must be locked before opening the test split.")
    # This is the only stage that opens train=False.
    source = MNIST(root, train=False, download=False)
    protocol = json.loads((output / "protocol.json").read_text())
    norm = protocol["normalization"]
    x = source.data.unsqueeze(1).float().div_(255).sub_(norm["mean"]).div_(norm["std"])
    dataset = x.to(device), source.targets.to(device)
    results = {}
    for name, winner in selection["winners"].items():
        saved = torch.load(output / winner["checkpoint"], map_location="cpu", weights_only=True)
        model = build_model(name).to(device)
        model.load_state_dict(saved["state_dict"])
        synchronize(device)
        started = time.perf_counter()
        metrics = evaluate(model, dataset, confusion=True)
        synchronize(device)
        results[name] = {"trial": winner["trial"], "checkpoint_epoch": saved["epoch"],
                         "parameters": sum(p.numel() for p in model.parameters()), "test": metrics,
                         "evaluation_seconds": time.perf_counter() - started}
        print(json.dumps({"final_test": name, **results[name]}), flush=True)
    write_json(output / "test-results.json", {"selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
               "device": str(device), "results": results})


def smoke(device):
    seed_all()
    for name in NAMES:
        model = build_model(name).to(device)
        logits = model(torch.randn(16, 1, 28, 28, device=device))
        loss = nn.functional.cross_entropy(logits, torch.arange(16, device=device) % 10)
        loss.backward()
        synchronize(device)
        assert torch.isfinite(loss).item()
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.inference_mode():
            assert torch.isfinite(model(torch.randn(3, 1, 28, 28, device=device))).all().item()
        print(name, "forward/backward/eval OK", sum(p.numel() for p in model.parameters()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "smoke", "tune", "test"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    device = setup_device(args.device)
    if args.stage == "prepare":
        prepare(args.data_dir, args.output_dir)
    elif args.stage == "smoke":
        smoke(device)
    elif args.stage == "tune":
        tune(args.data_dir, args.output_dir, device)
    else:
        final_test(args.data_dir, args.output_dir, device)
