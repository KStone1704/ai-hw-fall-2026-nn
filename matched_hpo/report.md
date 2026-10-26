# MNIST Matched Parameter Experiment Report

## Objective

This experiment compares a shallow multilayer perceptron, a convolutional network, and a small ViT-style Transformer Encoder for MNIST classification. The models have almost equal parameter counts and receive separate hyperparameter searches on one common held-out validation split. In the measured run, the CNN achieved the highest final test accuracy at 98.95%.

## Data and preprocessing

A seed-42 permutation of the official 60000-image training split assigns 55000 images to training and 5000 to validation. The official 10000-image test split remains outside the tuning procedure. This is a single held-out design, not k-fold cross-validation.

Pixel values are divided by 255. Mean and population standard deviation are computed in float64 from only the 43120000 training pixels: 0.13067043953945248 and 0.30812228107625567. All splits use these fitted values and float32 model inputs. An independent histogram calculation reproduced the statistics; the normalized training mean was approximately -7.90e-9 and standard deviation 1.000000046. Validation and test pixels did not contribute to fitting. No image augmentation was used.

## Architectures and tensor shapes

| Model | Input processing and architecture | Parameters |
| --- | --- | ---: |
| MLP | `[B,1,28,28]` to `[B,784]`; 127-unit ReLU hidden layer; ten logits | 100975 |
| CNN | 3 by 3 convolutions with 16 then 32 channels, each followed by ReLU and 2 by 2 pooling; `[B,32,7,7]` to 1568 values; 61-unit ReLU layer; ten logits | 101129 |
| Transformer Encoder | Sixteen 7 by 7 patches mapped to `[B,16,64]`; class token and learned position embeddings produce `[B,17,64]`; two layers, four heads, feed-forward width 240, dropout 0.1; class-token classifier | 100970 |

The maximum parameter difference is 159, or about 0.1575% of the smallest model. The widths were selected to match a roughly 101000-parameter budget, not by inspecting validation or test results. Equal parameter counts do not equalize computational cost or architectural assumptions.

The Transformer's patch projection is one shared affine transformation from 49 pixel values to 64 features, implemented as a strided convolution. Its 3136 weights and 64 biases are reused for every patch, image, and batch. Optimization changes those shared weights across training steps; they are not regenerated per batch. Tests confirm equivalence to a shared linear projection and evaluation-mode batch invariance.

Position encoding is an explicitly added learned absolute embedding, not sinusoidal encoding and not an automatic PyTorch encoder feature. The model follows the patch-token idea of ViT but is not a full reproduction: it uses PyTorch's post-normalization encoder and is trained from scratch without pretraining.

## Tuning and checkpoint selection

Each architecture receives six AdamW trials: learning rates 0.0003, 0.001, and 0.003 crossed with weight decay 0 and 0.0001. Batch size is 256, the maximum duration is eight epochs, and early-stopping patience is three epochs. Each trial restarts from seed 42 with the same seeded training-order sequence. AdamW betas are 0.9 and 0.999, epsilon is 1e-8, and weight decay applies to all parameters. No scheduler, warmup, augmentation, or mixed precision is used.

The best validation correct count selects each checkpoint, with lower validation cross-entropy breaking a tie. The same rule selects the winning trial independently for each model. All three winners are saved in a locked selection record before the test stage opens the test dataset. Final evaluation uses those checkpoints directly, without retraining on validation images.

| Learning rate | Weight decay | MLP best validation | CNN best validation | Transformer best validation |
| ---: | ---: | ---: | ---: | ---: |
| 0.0003 | 0 | 95.44% | 98.38% | 92.92% |
| 0.0003 | 0.0001 | 95.46% | 98.38% | 92.92% |
| 0.001 | 0 | 97.12% | 98.74% | 96.52% |
| 0.001 | 0.0001 | 97.08% | 98.76% | 96.52% |
| 0.003 | 0 | 97.32% | 99.00% | 97.08% |
| 0.003 | 0.0001 | 97.56% | 98.82% | 96.52% |

## Measured final results

| Model | Selected learning rate | Selected weight decay | Selected epoch | Validation accuracy | Test correct of 10000 | Test accuracy | Test cross-entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 0.003 | 0.0001 | 8 | 97.56% | 9751 | 97.51% | 0.09345 |
| CNN | 0.003 | 0 | 5 | 99.00% | 9895 | 98.95% | 0.03146 |
| Transformer Encoder | 0.003 | 0 | 8 | 97.08% | 9728 | 97.28% | 0.08938 |

All training and final model evaluation ran locally on the Mac GPU through PyTorch MPS. Python was 3.12.14, PyTorch 2.14.1, and torchvision 0.29.1 on macOS arm64. A real GPU forward/backward diagnostic and all three model smoke tests passed. MPS CPU fallback was not enabled. Dataset preparation, normalization fitting, checkpoint serialization, and confusion-matrix counting used CPU.

The six-trial search times, including training, validation, and checkpoint saves, were approximately 16.36 seconds for the MLP, 85.63 seconds for the CNN, and 147.06 seconds for the Transformer, totaling 249.04 seconds. These are recorded local timings, not controlled device benchmarks; the earlier CPU baseline used a different protocol and cannot establish a GPU speedup factor.

## Interpretation and limitations

The CNN exceeded the MLP by 1.44 percentage points and the Transformer by 1.67 points on this test set. Local receptive fields and spatial weight sharing are plausible contributors to the CNN's performance on digit images, but the comparison does not isolate their causal effects. The MLP and Transformer differ by only 23 correctly classified images; one split and one seed cannot establish a general architectural ranking.

The Transformer improves substantially over the historical three-epoch baseline, but parameter count, normalization, optimizer, training duration, and hyperparameter selection all changed. The improvement cannot be attributed to one change alone. Several winning checkpoints reached the eight-epoch limit, and every model selected the largest searched learning rate; the bounded search does not establish convergence or a global optimum.

Only learning rate and weight decay were searched. Architecture, dropout, batch size, scheduler, augmentation, and seed were not tuned. The same 5000-image validation set selected both epochs and configurations, so its winning accuracy is optimistic relative to a fresh evaluation. Parameter matching does not equalize runtime or FLOPs. The official test set was evaluated once per winner in this follow-up, but it had already been seen in the historical baseline; this is not a newly blind benchmark. No further choices in this run were made using test outcomes.

The saved protocol, split hashes, eighteen trial histories, locked selections, preprocessing verification, and final confusion matrices provide the reproducible evidence. Future work could predefine a broader search and multiple seeds, retain a fresh final evaluation where possible, and compare measured compute budgets as well as parameter counts.

## References and assistance

- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html)

This project was completed with assistance from OpenAI Codex.
