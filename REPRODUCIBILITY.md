# Reproducibility Manifest

This document outlines the exact computational environment and hyperparameter settings required to reproduce the Stage 3 experiment results.

## Environment & Hardware
- **Python Version**: 3.13
- **PyTorch Version**: 2.11.0
- **Hardware/Device**: Apple Silicon (MPS enabled) `device=mps`
- **Random Seed**: `42` (forced via `torch.manual_seed`, `np.random.seed`, and deterministic operations where possible).

## Model Architecture
- **Base Architecture**: EfficientNet-B0
- **Initialization**: ImageNet (`IMAGENET1K_V1`)
- **Input Channels**: 3 (Grayscale MRI repeated across RGB channels)
- **Classifier Head**: Linear(1280, 3)

## Preprocessing & Augmentation
- **Intensity Normalization**: Min-Max normalization to `[0,1]`.
- **ImageNet Normalization**: mean=`[0.485, 0.456, 0.406]`, std=`[0.229, 0.224, 0.225]`.
- **Resize**: `224x224`
- **Training Augmentation**: `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=15)`.
- **Validation/Test Augmentation**: None (only resize and normalization).

## Training Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: `1e-4`
- **Weight Decay**: `1e-4`
- **Batch Size**: `16`
- **Maximum Epochs**: `50`
- **Scheduler**: `ReduceLROnPlateau` (factor=`0.5`, patience=`5`, mode=`min`)
- **Early Stopping**: Triggered after `10` epochs of no improvement in validation loss.

## Calibration Methodology
- **Expected Calibration Error (ECE)**:
  - **Binning**: 15 equal-width bins between 0.0 and 1.0.
  - **Method**: The weighted sum of the absolute difference between mean accuracy and mean confidence within each bin.
- **Multiclass Brier Score**:
  - Computed as the mean squared error between the predicted probability distribution across all 3 classes and the true one-hot encoded label vector.
