# MNIST Experiment Report Draft

Historical baseline: the revised parameter-matched and tuned comparison is documented in `matched_hpo/README.md` and its accompanying report. The original results below remain evidence of the earlier run.

## Objective

The objective was to compare three small neural networks for classifying the ten handwritten digits in MNIST: a shallow multilayer perceptron, a convolutional network, and a patch-based Transformer Encoder. The comparison used a fixed training budget rather than attempting to reach state-of-the-art accuracy.

## Method

The official 60000-image training split was randomly divided into 55000 training and 5000 validation images using seed 42. The official 10000-image test split remained separate from optimization and checkpoint selection. Images were scaled to [0, 1] and normalized using fixed mean 0.1307 and standard deviation 0.3081. No augmentation was applied.

Each model trained for three epochs with Adam, learning rate 0.001, batch size 256, and cross-entropy loss. The checkpoint with the highest validation accuracy was selected independently for each model. All three selected the third epoch. Final test evaluation used those selected checkpoints, without retraining on validation data or tuning on test results.

## Results

The MLP classified 9672 of 10000 test images correctly, giving 96.72% accuracy. The CNN classified 9799 correctly, giving 97.99%. The Transformer Encoder classified 9421 correctly, giving 94.21%. The corresponding validation accuracies were 95.86%, 98.02%, and 93.50%.

The CNN improved on the MLP by 1.27 percentage points and on the Transformer by 3.78 percentage points. Its test confusion matrix shows that 25 examples of the true digit 2 were predicted as 3, the largest individual off-diagonal entry in that run. The matrix identifies an error pattern, but without visually examining those images it does not prove why the errors occurred.

## Interpretation

The MLP provides a fast baseline, but flattening the image does not explicitly exploit local spatial structure. The CNN introduces local receptive fields and shared convolution filters, which are a plausible reason for its strong performance on this small image task. This is an interpretation of the architecture and results, not a controlled causal finding.

The Transformer uses patch tokens and self-attention to combine information across the image. Its accuracy improved from 83.10% validation accuracy after the first epoch to 93.50% after the third, suggesting that the short training budget limits the conclusion. The result does not show that the architecture is inherently unsuitable for MNIST.

## Limitations and next steps

Only one seed and three epochs were used. Parameter counts differ, and a common optimizer configuration is not necessarily equally suitable for each model. No augmentation, learning-rate search, or statistical uncertainty analysis was performed. A stronger follow-up would compare several seeds, choose hyperparameters using only validation data, and test controlled training-only augmentation. Future model selection should not use this test set repeatedly; additional development should retain a genuinely independent final evaluation.

The measured results, environment, and confusion matrices are preserved in the accompanying JSON files.

This project was completed with assistance from OpenAI Codex.
