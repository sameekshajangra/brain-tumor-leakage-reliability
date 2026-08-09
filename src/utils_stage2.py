"""
src/utils_stage2.py
====================
Shared utilities for Stage 2: dataset class, model factory, preprocessing.

Preprocessing rationale
------------------------
MRI images are stored as int16 with widely varying intensity windows
(observed global range: 0..10710 in a 20-image sample from Stage 1 inspection).
This variation reflects differences in MRI acquisition parameters between
patients and scanners rather than meaningful differences in pixel units.

Per-image min-max normalization to [0, 1] is the appropriate choice here
because:
  1. It preserves relative intra-image intensity structure.
  2. It removes inter-image intensity variation irrelevant to class.
  3. It is the standard approach used in medical imaging literature
     for MRI pre-processed with EfficientNet/ResNet.

After [0,1] normalization, the single-channel float image is replicated
to 3 channels (RGB copy) and ImageNet normalization is applied, because
EfficientNet-B0 was pretrained on ImageNet with mean=[0.485,0.456,0.406]
and std=[0.229,0.224,0.225].

This preprocessing is applied IDENTICALLY to training, validation, and test
images. Augmentation is applied to training images only.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, models


# ---------------------------------------------------------------------------
# Preprocessing transforms
# ---------------------------------------------------------------------------

class PerImageMinMaxNormalize:
    """Scale image to [0, 1] using per-image min and max."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        if mx > mn:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x, dtype=np.float32)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def get_train_transform():
    """Training transform: normalize + replicate channels + augment + ImageNet norm."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),               # -> [0,1] float tensor, single channel
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),   # grayscale -> 3-channel
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transform():
    """Validation/test transform: normalize + replicate channels + ImageNet norm. No augmentation."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class BrainTumorDataset(Dataset):
    """
    Loads brain-tumor MRI images from .mat (HDF5) files.

    Parameters
    ----------
    df        : DataFrame with columns [image_id, filename, PID, label, class_name]
    raw_dir   : Path to directory containing .mat files
    transform : torchvision transform to apply
    """
    def __init__(self, df, raw_dir: Path, transform=None):
        self.df        = df.reset_index(drop=True)
        self.raw_dir   = Path(raw_dir)
        self.transform = transform
        self._normalizer = PerImageMinMaxNormalize()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img_id  = int(row["image_id"])
        label   = int(row["label"]) - 1   # 0-indexed: Meningioma=0, Glioma=1, Pituitary=2

        mat_path = self.raw_dir / f"{img_id}.mat"
        with h5py.File(str(mat_path), "r") as f:
            img = f["cjdata"]["image"][()].astype(np.float32)

        # Per-image min-max normalization to [0, 1]
        img = self._normalizer(img)

        # Convert to uint8 for PIL compatibility (ToPILImage expects uint8 or float32)
        img_uint8 = (img * 255).astype(np.uint8)

        if self.transform:
            img_tensor = self.transform(img_uint8)
        else:
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

        return img_tensor, label, img_id


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_efficientnet_b0(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNet-B0 with ImageNet pretrained weights.
    Replaces the final Linear(1280, 1000) with Linear(1280, num_classes).
    """
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
