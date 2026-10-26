# MNIST Neural Network Comparison

This project compares a shallow MLP, a CNN, and a small ViT-style Transformer Encoder on MNIST. The main experiment matches the models at approximately 101000 parameters and tunes each model independently using the same fixed held-out validation split. All reported results were measured in local runs.

## Results

| Model | Parameters | Selected epoch | Validation accuracy | Test correct of 10000 | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 100975 | 8 | 97.56% | 9751 | 97.51% |
| CNN | 101129 | 5 | 99.00% | 9895 | 98.95% |
| Transformer Encoder | 100970 | 8 | 97.08% | 9728 | 97.28% |

The CNN performs best in this bounded, single-seed comparison. This does not establish a universal ranking of the architectures.

## Main experiment

- [Experiment details and tensor shapes](matched_hpo/README.md)
- [Experiment report](matched_hpo/report.md)
- [Model definitions](matched_hpo/models.py)
- [Training, validation tuning, and separate final evaluation](matched_hpo/experiment.py)
- [Recorded protocol](matched_hpo/results/protocol.json), [selected configurations](matched_hpo/results/selection.json), and [final test results](matched_hpo/results/test-results.json)
- [All eighteen trial histories](matched_hpo/results/trials/)

The official 60000 training images are split into 55000 training and 5000 validation examples using seed 42. Normalization statistics are fitted only on the training subset. Each model searches three learning rates and two AdamW weight-decay values, with a maximum of eight epochs per trial. All three model selections are locked before evaluation on the official 10000-image test split. Training and final model evaluation use the local Mac GPU through PyTorch MPS.

This is a single held-out split, not k-fold cross-validation. Only learning rate and weight decay are searched; model widths are chosen to match the parameter budget. The official test split was previously evaluated in the baseline experiment, so this follow-up is not a newly blind benchmark. Full methods and limitations are in the report.

## Reproduce

The recorded environment used Python 3.12.14, torch 2.14.1, and torchvision 0.29.1 on macOS arm64. Run the following from the repository root after installing `requirements.txt`:

```bash
python data.py --data-dir data
cd matched_hpo
python -B experiment.py smoke --device mps
python -B experiment.py prepare --device cpu --data-dir ../data --output-dir rerun
python -B check_preprocessing.py --data-dir ../data --output-dir rerun
python -B experiment.py tune --device mps --data-dir ../data --output-dir rerun
python -B experiment.py test --device mps --data-dir ../data --output-dir rerun
python -B -m unittest -v test_experiment.py
```

Use `--device cpu` on machines without MPS. Use a fresh output directory for a new run; recorded results are preserved. See the main experiment README for complete setup and stage behavior. Raw images, virtual environments, and model checkpoints are excluded from the repository. Split indices are regenerated from seed 42; their SHA256 hashes are recorded in the protocol.

## Historical baseline

The root-level `models.py`, `data.py`, `train.py`, `evaluate.py`, and `run_experiments.py` preserve the original three-epoch CPU experiment. Its six result files are in `results/`, and its interpretation is in [the baseline report](report.md). Baseline test accuracies were MLP 96.72%, CNN 97.99%, and Transformer 94.21%. The parameter-matched experiment above is the main revised comparison.

## Assistance

This project was completed with assistance from OpenAI Codex.
