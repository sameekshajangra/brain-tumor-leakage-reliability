"""
src/10_train_efficientnet.py
=============================
Stage 2 — Step 4: Train EfficientNet-B0 for one condition (A, B, or C).

Usage
-----
  python src/10_train_efficientnet.py --condition A
  python src/10_train_efficientnet.py --condition B
  python src/10_train_efficientnet.py --condition C

Identical hyperparameters are used for all conditions.
The test set is NEVER used during training.

Training procedure
------------------
- EfficientNet-B0, ImageNet pretrained, 3-class head
- AdamW optimizer (lr=1e-4, wd=1e-4)
- ReduceLROnPlateau scheduler (patience=5, factor=0.5)
- Early stopping: patience=10 on val_loss
- Max 50 epochs
- Checkpoint saved every epoch
- Best checkpoint = lowest val_loss epoch

Outputs (per condition X ∈ {A, B, C})
-------
  results/stage2/checkpoints/cond_X_best.pt
  results/stage2/checkpoints/cond_X_epoch_N.pt   (per epoch)
  results/stage2/training_log_cond_X.csv
"""
import argparse
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils_stage2 import (
    BrainTumorDataset, build_efficientnet_b0,
    get_train_transform, get_val_test_transform,
)

# ---------------------------------------------------------------------------
# Config (identical for all conditions)
# ---------------------------------------------------------------------------
SEED         = 42
BATCH_SIZE   = 16
MAX_EPOCHS   = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-4
ES_PATIENCE  = 10     # early stopping patience (val_loss)
SCHED_FACTOR = 0.5
SCHED_PAT    = 5
NUM_CLASSES  = 3

SPLITS  = PROJECT_ROOT / "results" / "stage2" / "splits"
RAW_DIR = PROJECT_ROOT.parent / "data" / "raw"
CKPT_DIR = PROJECT_ROOT / "results" / "stage2" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
TABLES2 = PROJECT_ROOT / "tables" / "stage2"
TABLES2.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * len(labels)
        preds          = out.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_n       += len(labels)
    return total_loss / total_n, total_correct / total_n


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss    += loss.item() * len(labels)
            preds          = out.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_n       += len(labels)
    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["A", "B", "C"], required=True)
    args = parser.parse_args()
    cond = args.condition

    print("=" * 70)
    print(f"TRAINING — Condition {cond}")
    print(f"  Seed={SEED}, lr={LR}, wd={WEIGHT_DECAY}, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}")
    print("=" * 70)

    set_seed(SEED)

    # ---- Load splits ----
    train_df = pd.read_csv(SPLITS / f"condition_{cond}_train.csv")
    val_df   = pd.read_csv(SPLITS / f"condition_{cond}_val.csv")

    # Verify test set NOT loaded during training
    print(f"\nTrain images: {len(train_df)}, Val images: {len(val_df)}")
    print(f"Train patients: {train_df['PID'].nunique()}, Val patients: {val_df['PID'].nunique()}")

    # ---- Datasets / DataLoaders ----
    train_ds = BrainTumorDataset(train_df, RAW_DIR, transform=get_train_transform())
    val_ds   = BrainTumorDataset(val_df,   RAW_DIR, transform=get_val_test_transform())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ---- Model ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True).to(device)

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHED_FACTOR, patience=SCHED_PAT, min_lr=1e-7)
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    log_rows  = []
    best_loss = float("inf")
    best_epoch = 0
    es_counter = 0
    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)
        ep_time = time.time() - t_ep
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.3f} | "
              f"lr={current_lr:.2e} | {ep_time:.1f}s")

        log_rows.append({
            "condition": cond, "epoch": epoch,
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "val_loss": round(vl_loss, 6), "val_acc": round(vl_acc, 6),
            "lr": current_lr, "epoch_time_s": round(ep_time, 1),
        })

        # Save per-epoch checkpoint
        epoch_ckpt = CKPT_DIR / f"cond_{cond}_epoch_{epoch:03d}.pt"
        torch.save({
            "condition": cond, "epoch": epoch,
            "val_loss": vl_loss, "val_acc": vl_acc,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, str(epoch_ckpt))

        # Best model
        if vl_loss < best_loss:
            best_loss  = vl_loss
            best_epoch = epoch
            es_counter = 0
            torch.save({
                "condition": cond, "epoch": epoch,
                "val_loss": vl_loss, "val_acc": vl_acc,
                "best_epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "SEED": SEED, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
                    "BATCH_SIZE": BATCH_SIZE, "MAX_EPOCHS": MAX_EPOCHS,
                },
            }, str(CKPT_DIR / f"cond_{cond}_best.pt"))
        else:
            es_counter += 1

        # Early stopping
        if es_counter >= ES_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (best epoch={best_epoch}, best_val_loss={best_loss:.4f})")
            break

    total_time = time.time() - t_start
    print(f"\nTraining complete. Best epoch: {best_epoch}, best_val_loss: {best_loss:.4f}")
    print(f"Total time: {total_time/60:.1f} min")

    # Save training log
    log_df = pd.DataFrame(log_rows)
    log_path = TABLES2 / f"training_log_cond_{cond}.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path}")
    print(f"Best checkpoint: {CKPT_DIR / f'cond_{cond}_best.pt'}")


if __name__ == "__main__":
    main()
