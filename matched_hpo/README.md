# MNIST Matched Parameter Comparison

This follow-up compares an MLP, a CNN, and a small ViT-style Transformer Encoder with nearly equal parameter counts. Each model receives the same fixed training and validation split and the same six-configuration tuning budget. The earlier three-epoch CPU experiment remains unchanged in the parent directory as a historical baseline.

## Measured results

| Model | Parameters | Selected learning rate | Selected weight decay | Selected epoch | Validation accuracy | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 100975 | 0.003 | 0.0001 | 8 | 97.56% | 97.51% |
| CNN | 101129 | 0.003 | 0 | 5 | 99.00% | 98.95% |
| Transformer Encoder | 100970 | 0.003 | 0 | 8 | 97.08% | 97.28% |

All eighteen trials and the final selected-checkpoint evaluations actually ran on the local Mac GPU using MPS. The [report draft](report.md) contains the full comparison, tuning table, and limitations. `results/` preserves the protocol, eighteen trial histories, locked selections, preprocessing check, and final test counts/losses/confusion matrices. No test result was used to change a configuration in this run.

## Data preparation

The official 60000-image MNIST training split is divided using a seed-42 random permutation into 55000 training and 5000 validation images. This is one held-out validation split, not k-fold cross-validation. Preparation writes local indices to `results/split.pt`; their SHA256 hashes are recorded in the versioned `results/protocol.json`. The index file is regenerated rather than uploaded to GitHub.

Normalization statistics are fitted only on the 43120000 pixels in the 55000 training images after division by 255. Float64 accumulation and population standard deviation give:

- Mean: 0.13067043953945248
- Standard deviation: 0.30812228107625567

The same fitted transform is applied to training, validation, and final test images. Neither validation nor test pixels contribute to these statistics. Model inputs are float32 tensors of shape `[B, 1, 28, 28]`; labels are int64 class indices. No augmentation is used.

```python
x = pixels.unsqueeze(1).float() / 255.0
x = (x - training_mean) / training_std
```

These values are measured from the selected training subset, not hardcoded MNIST constants recalled from another source.

## Matched model sizes

| Model | Architecture | Trainable parameters |
| --- | --- | ---: |
| MLP | Flatten 784, Linear 127, ReLU, Linear 10 | 100975 |
| CNN | Conv 1 to 16, pool, Conv 16 to 32, pool, Linear 1568 to 61, ReLU, Linear 10 | 101129 |
| Transformer Encoder | 7 by 7 patches, embedding 64, 4 heads, 2 encoder layers, feed-forward width 240 | 100970 |

The largest difference is 159 parameters, approximately 0.1575% of the smallest model. Widths 127, 61, and 240 are chosen to match the parameter budget; they are not selected using validation or test accuracy. Equal parameter counts do not imply equal FLOPs, runtime, memory use, or inductive bias.

The Transformer maps sixteen non-overlapping patches to `[B, 16, 64]`, prepends a learned class token, adds learned absolute position embeddings of shape `[1, 17, 64]`, then classifies the final class-token representation. Every model returns ten unnormalized logits for cross-entropy loss.

This is ViT-style rather than an exact reproduction of the original ViT: it uses PyTorch's post-normalization encoder block (`norm_first=False`), not the original ViT's pre-normalization block. It is trained from scratch on MNIST with no pretrained weights.

## Hyperparameter search

Each fixed architecture is trained from the same seed for every combination of:

- Learning rate: 0.0003, 0.001, 0.003
- AdamW weight decay: 0, 0.0001

All trials use batch size 256, a maximum of eight epochs, and early stopping after three consecutive epochs without an improvement in the selection metric. The selection metric is highest validation correct count, with lowest validation cross-entropy as the tie-breaker. Exact ties keep the earlier epoch or trial. AdamW uses betas 0.9 and 0.999 and epsilon 1e-8; weight decay applies to all parameters. There is no scheduler, warmup, augmentation, mixed precision, architecture search, dropout search, batch-size search, or multi-seed search. Transformer dropout remains 0.1.

Every trial restarts from seed 42 and uses the same seeded training-order sequence. Each model selects its own best hyperparameters and checkpoint. All three choices are locked in `results/selection.json` before the test-loading stage can run. Test evaluation uses those checkpoints without retraining on the validation set.

The official 10000-image test split is not opened by the prepare or tuning stages. It is evaluated once per selected model in this follow-up. Since the earlier baseline already evaluated this official test set, it is not a newly unseen dataset; this is a follow-up comparison, not a fresh blind benchmark.

## Mac GPU execution

PyTorch 2.14.1 and torchvision 0.29.1 are already installed in the local coursework virtual environment. The Mac GPU is accessed through PyTorch's `mps` backend, not CUDA. A real MPS forward/backward diagnostic succeeded, followed by training/backward/evaluation smoke tests for all three models.

MPS requires host GPU access. The desktop filesystem sandbox reported MPS unavailable, while the approved host execution reported available and successfully ran the tensors. No operating-system change or global Python installation was needed. The experiment raises an error if MPS is requested but unavailable and does not silently switch to CPU. `PYTORCH_ENABLE_MPS_FALLBACK` is not enabled. Data tensors, models, gradients, and optimizer updates run on MPS; split creation, normalization-statistic fitting, checkpoint serialization, and final confusion-matrix counting use CPU.

All training uses float32 and `torch.use_deterministic_algorithms(True)`. The attention fast path is disabled consistently. Seeds and deterministic settings improve reproducibility but do not promise identical numerical results across different devices or library versions. Timers synchronize MPS before and after measured work.

## Reproduction

Run from this directory, using the dependencies specified by the parent `requirements.txt`. The local dataset has already been downloaded; another machine can use the parent `data.py` to download it first.

```bash
python -B experiment.py smoke --device mps
python -B experiment.py prepare --device cpu --data-dir ../data
python -B check_preprocessing.py --data-dir ../data
python -B experiment.py tune --device mps --data-dir ../data
python -B experiment.py test --device mps --data-dir ../data
python -B -m unittest -v test_experiment.py
```

Use `--device cpu` for machines without MPS. Each stage accepts `--output-dir` for a fresh run. Preparation refuses to overwrite an existing protocol. Tuning can resume completed trial JSON records, but it refuses to run after the final selections are locked. Final testing refuses to overwrite existing final results. To reproduce from scratch, use a fresh output directory rather than deleting historical evidence. Resume only an unchanged protocol and implementation on the same device.

`results/split.pt` contains only integer sample indices; it is not a model checkpoint. It is local and reproducibly regenerated by the prepare stage. Model checkpoints remain local under `results/checkpoints/` and are excluded from Git, along with raw data and virtual environments. `check_preprocessing.py` independently recomputes the fitted mean and standard deviation from a histogram of training pixel values and checks the normalized training moments without loading the test set. The saved-record tests can verify split hashes by regenerating the seeded permutation even when the binary index file is absent.

## Patch projection and position encoding

The patch embedding is a single `Conv2d(1, 64, kernel_size=7, stride=7)`. It contains 3136 weight values and 64 biases, shared across all sixteen patches, all images, and all batches. It is mathematically equivalent to flattening each 7 by 7 grayscale patch to 49 values and applying the same affine map `z = Wp + b`, with `W` shaped 64 by 49.

The optimizer updates this shared map during training. It is not freshly generated for each batch. At a fixed checkpoint, an identical patch has an identical patch embedding regardless of its batch. Later contextualized token representations depend on the other patches and position, and dropout makes training activations stochastic. The tests check convolution versus linear equivalence and evaluation-mode batch invariance.

Position embeddings here are learned absolute vectors, following the basic ViT approach. They are not the sinusoidal encoding chosen for the original Transformer. PyTorch's encoder does not insert position information automatically; the addition is explicit in `models.py`. No RoPE or relative-position bias is implemented. See [Vision Transformer](https://arxiv.org/abs/2010.11929) and [Attention Is All You Need](https://arxiv.org/html/1706.03762v7#S3.SS5).

## Modern multimodal image inputs

A common public vision-language design first resizes, crops, or tiles an RGB image, normalizes it according to the vision encoder's training recipe, then converts patches into visual feature vectors. A vision encoder contextualizes the vectors; a projector or merger maps or compresses them into the language model's representation space. Visual vectors and text embeddings then interact inside the multimodal model. These are continuous visual embeddings, not necessarily image patches converted into ordinary vocabulary IDs.

Implementations differ. [Qwen3-VL](https://arxiv.org/html/2511.21631v2#S2) uses a dynamic-resolution SigLIP-2 vision encoder, a two-layer MLP merger that compresses 2 by 2 visual features into one LLM-aligned token, multi-level feature injection, and multimodal position handling. [Gemma 3](https://arxiv.org/pdf/2503.19786) illustrates a different public design: a SigLIP vision encoder at 896 by 896, optional extra crops for difficult aspect ratios or high resolution, and pooling to 256 visual vectors per encoded image/crop. These are documented examples, not claims that every contemporary model follows the same architecture or that undisclosed commercial internals are known.

Our MNIST Transformer shares the patch-to-vector idea but is much smaller: grayscale 28 by 28, sixteen visual tokens plus a class token, no pretrained visual encoder, no language model, and a ten-class classifier.

## Assistance

This project was completed with assistance from OpenAI Codex.
