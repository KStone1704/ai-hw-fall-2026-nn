# MNIST Image Classification

A workspace for a small image-classification neural-network exercise using MNIST. Implementation and experiments are still pending; this repository does not yet contain trained models or measured results.

## Planned experiments

- A shallow multilayer perceptron.
- A convolutional neural network.
- A Transformer encoder.
- An optional comparison of training-time image augmentation.

Use the official training split for training and any validation split. Keep the test split separate for final evaluation; do not tune model choices on test accuracy.

## Implementation checklist

- [ ] Add model definitions.
- [ ] Add reproducible training and evaluation commands.
- [ ] Record the Python and package versions used.
- [ ] Run experiments and save actual test results.
- [ ] Document model settings, data splits, and limitations.

Suggested files are `models.py`, `train.py`, `evaluate.py`, and a `results/` directory with metrics and experiment notes. Add a requirements file after selecting and testing the environment.

## Data sources

- [MNIST on Hugging Face](https://huggingface.co/datasets/ylecun/mnist)
- [Torchvision MNIST dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

Download the dataset through the selected library rather than committing the raw dataset. Keep credentials, personal records, employer code, and proprietary data out of this repository.
