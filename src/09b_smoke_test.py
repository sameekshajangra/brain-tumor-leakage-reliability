"""
src/09b_smoke_test.py
=====================
Stage 2 — Smoke test: 2-epoch technical verification of Condition A pipeline.

Verifies
--------
1. Data loading from .mat (HDF5) files
2. Preprocessing pipeline (min-max norm → 3-channel → ImageNet norm)
3. EfficientNet-B0 forward pass
4. Cross-entropy loss calculation
5. Backpropagation
6. Checkpoint saving
7. Probability outputs sum ≈ 1 for all samples
8. Three-class output (shape [batch, 3])
9. Validation loop

IMPORTANT
---------
Smoke-test results must NOT be used in final analysis or paper.
This script creates a separate smoke-test checkpoint and deletes it on success.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils_stage2 import BrainTumorDataset, build_efficientnet_b0, get_train_transform, get_val_test_transform

SPLITS  = PROJECT_ROOT / "results" / "stage2" / "splits"
RAW_DIR = PROJECT_ROOT.parent / "data" / "raw"
SMOKE_CKPT = PROJECT_ROOT / "results" / "stage2" / "smoke_test_checkpoint.pt"

BATCH_SIZE  = 4    # small batch for smoke test
MAX_EPOCHS  = 2
SEED        = 42
NUM_CLASSES = 3

torch.manual_seed(SEED)
np.random.seed(SEED)

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

def fail(msg):
    log(f"\nSTOP — SMOKE TEST FAILED: {msg}")
    (PROJECT_ROOT / "results" / "stage2" / "smoke_test_report.txt").write_text("\n".join(lines))
    sys.exit(1)

log("=" * 70)
log("SMOKE TEST — 2-epoch Condition A pipeline verification")
log("=" * 70)

# ---------------------------------------------------------------------------
# 1. Load splits
# ---------------------------------------------------------------------------
log("\n[1] Loading Condition A splits ...")
train_df = pd.read_csv(SPLITS / "condition_A_train.csv")
val_df   = pd.read_csv(SPLITS / "condition_A_val.csv")
log(f"    Train: {len(train_df)} images")
log(f"    Val  : {len(val_df)} images")

# Use a tiny subset for speed
train_sub = train_df.sample(n=min(64, len(train_df)), random_state=SEED)
val_sub   = val_df.sample(n=min(32, len(val_df)), random_state=SEED)
log(f"    Using subset: train={len(train_sub)}, val={len(val_sub)}")

# ---------------------------------------------------------------------------
# 2. DataLoaders
# ---------------------------------------------------------------------------
log("\n[2] Building DataLoaders ...")
train_ds = BrainTumorDataset(train_sub, RAW_DIR, transform=get_train_transform())
val_ds   = BrainTumorDataset(val_sub,   RAW_DIR, transform=get_val_test_transform())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# Test first batch
first_imgs, first_labels, first_ids = next(iter(train_loader))
log(f"    First batch shape: {first_imgs.shape}  (expected [B,3,224,224])")
if first_imgs.shape[1:] != torch.Size([3, 224, 224]):
    fail(f"Unexpected image shape: {first_imgs.shape}")
log(f"    Label range: {first_labels.min()}..{first_labels.max()}")
log(f"    Image dtype: {first_imgs.dtype}")
log("    [OK] Data loading and preprocessing")

# ---------------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------------
log("\n[3] Building EfficientNet-B0 ...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
log(f"    Device: {device}")
model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True)
model = model.to(device)

# Forward pass
model.eval()
with torch.no_grad():
    test_out = model(first_imgs.to(device))
log(f"    Forward pass output shape: {test_out.shape}  (expected [B, 3])")
if test_out.shape[1] != NUM_CLASSES:
    fail(f"Model output has {test_out.shape[1]} classes, expected {NUM_CLASSES}")

# Probability check
probs = torch.softmax(test_out, dim=1)
prob_sums = probs.sum(dim=1)
log(f"    Probability sums: min={prob_sums.min():.6f}, max={prob_sums.max():.6f}")
if not (torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4)):
    fail(f"Probabilities don't sum to 1: {prob_sums}")
log("    [OK] Model architecture, forward pass, and probability output")

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
log("\n[4] Running 2-epoch training loop ...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

model.train()
for epoch in range(MAX_EPOCHS):
    t0 = time.time()
    epoch_loss = 0.0
    model.train()
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        if torch.isnan(loss):
            fail(f"NaN loss at epoch {epoch+1}")
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            val_loss += loss.item()

    log(f"    Epoch {epoch+1}: train_loss={epoch_loss/len(train_loader):.4f}, "
        f"val_loss={val_loss/len(val_loader):.4f}, "
        f"time={time.time()-t0:.1f}s")

log("    [OK] Training loop (2 epochs): forward pass, loss, backprop all verified")

# ---------------------------------------------------------------------------
# 5. Checkpoint save
# ---------------------------------------------------------------------------
log("\n[5] Testing checkpoint save ...")
torch.save({
    "epoch": 2,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "smoke_test": True,
}, str(SMOKE_CKPT))
log(f"    Saved to: {SMOKE_CKPT}")
log("    [OK] Checkpoint saving")

# ---------------------------------------------------------------------------
# 6. Probability output verification on val set
# ---------------------------------------------------------------------------
log("\n[6] Verifying probability outputs on val set ...")
model.eval()
all_probs = []
with torch.no_grad():
    for imgs, labels, _ in val_loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.append(probs)

all_probs = np.vstack(all_probs)
sums = all_probs.sum(axis=1)
log(f"    Prob shape: {all_probs.shape}")
log(f"    Sum range: {sums.min():.6f}..{sums.max():.6f}")
if not np.allclose(sums, 1.0, atol=1e-4):
    fail("Probability sums deviate from 1.0")
log("    [OK] All probability outputs sum to 1 (atol=1e-4)")

# ---------------------------------------------------------------------------
# Cleanup smoke test checkpoint (results must not be used in analysis)
# ---------------------------------------------------------------------------
SMOKE_CKPT.unlink(missing_ok=True)
log(f"\n    Smoke test checkpoint deleted (must not be used in analysis)")

log("\n" + "=" * 70)
log("SMOKE TEST PASSED — All 6 checks OK")
log("Safe to proceed to full training (script 10).")
log("=" * 70)

(PROJECT_ROOT / "results" / "stage2" / "smoke_test_report.txt").write_text("\n".join(lines) + "\n")
