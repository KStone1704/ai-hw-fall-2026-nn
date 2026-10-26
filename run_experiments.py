"""Run a fixed three-epoch comparison, then test selected checkpoints."""

import argparse
import subprocess
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    for name in ("mlp", "cnn", "transformer"):
        subprocess.run([sys.executable, "train.py", "--model", name,
                        "--epochs", "3", "--data-dir", args.data_dir], check=True)
        subprocess.run([sys.executable, "evaluate.py", "--checkpoint",
                        f"checkpoints/{name}.pt", "--data-dir", args.data_dir,
                        "--output", f"results/{name}-test.json"], check=True)
