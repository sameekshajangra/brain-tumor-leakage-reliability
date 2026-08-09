"""
src/11_generate_predictions.py
===============================
Stage 2 — Step 5: Generate per-image predictions on the fixed test set.

Loads the best checkpoint for each condition and evaluates on the SAME 47
fixed test images. The test set is used ONLY here (never during training).

For every test image saves:
  image_id, filename, PID, true_label, true_class,
  predicted_label, predicted_class,
  probability_meningioma, probability_glioma, probability_pituitary,
  confidence, correct, condition

Outputs
-------
  results/stage2/predictions_condition_A.csv
  results/stage2/predictions_condition_B.csv
  results/stage2/predictions_condition_C.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils_stage2 import BrainTumorDataset, build_efficientnet_b0, get_val_test_transform

SPLITS   = PROJECT_ROOT / "results" / "stage2" / "splits"
CKPT_DIR = PROJECT_ROOT / "results" / "stage2" / "checkpoints"
RES2     = PROJECT_ROOT / "results" / "stage2"
TABLES2  = PROJECT_ROOT / "tables" / "stage2"
RAW_DIR  = PROJECT_ROOT.parent / "data" / "raw"

CLASS_NAMES = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}   # 0-indexed
LABEL_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}   # 1-indexed (original)
NUM_CLASSES = 3

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 70)
print("GENERATING PREDICTIONS ON FIXED TEST SET")
print(f"Device: {device}")
print("=" * 70)

# Load the fixed test split (identical for A, B, C — verified by script 09)
test_df = pd.read_csv(SPLITS / "condition_A_test.csv")   # same for all conditions
print(f"\nTest images: {len(test_df)}")
print(f"Test patients: {test_df['PID'].nunique()}")

test_ds = BrainTumorDataset(test_df, RAW_DIR, transform=get_val_test_transform())
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

all_predictions = {}

for cond in ["A", "B", "C"]:
    ckpt_path = CKPT_DIR / f"cond_{cond}_best.pt"
    if not ckpt_path.exists():
        print(f"\nSTOP — Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\n--- Condition {cond} ---")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    print(f"  Best epoch: {ckpt.get('best_epoch', ckpt.get('epoch', '?'))}")
    print(f"  Best val_loss: {ckpt.get('val_loss', '?'):.4f}")

    model = build_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Collect predictions
    all_img_ids, all_true, all_pred, all_probs = [], [], [], []

    with torch.no_grad():
        for imgs, labels, img_ids in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)

            all_img_ids.extend(img_ids.tolist())
            all_true.extend(labels.tolist())
            all_pred.extend(preds.tolist())
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    # Build prediction DataFrame
    rows = []
    for i, img_id in enumerate(all_img_ids):
        true_label_0 = all_true[i]       # 0-indexed
        pred_label_0 = all_pred[i]       # 0-indexed
        true_label_1 = true_label_0 + 1  # 1-indexed
        pred_label_1 = pred_label_0 + 1  # 1-indexed

        # Look up metadata
        meta_row = test_df[test_df["image_id"] == img_id].iloc[0]
        confidence = float(all_probs[i, pred_label_0])

        rows.append({
            "image_id"               : int(img_id),
            "filename"               : meta_row["filename"],
            "PID"                    : meta_row["PID"],
            "true_label"             : true_label_1,
            "true_class"             : LABEL_NAMES[true_label_1],
            "predicted_label"        : pred_label_1,
            "predicted_class"        : LABEL_NAMES[pred_label_1],
            "probability_meningioma" : round(float(all_probs[i, 0]), 6),
            "probability_glioma"     : round(float(all_probs[i, 1]), 6),
            "probability_pituitary"  : round(float(all_probs[i, 2]), 6),
            "confidence"             : round(confidence, 6),
            "correct"                : int(true_label_0 == pred_label_0),
            "condition"              : cond,
        })

    pred_df = pd.DataFrame(rows)
    acc = pred_df["correct"].mean()
    print(f"  Accuracy: {acc:.3f} ({pred_df['correct'].sum()}/{len(pred_df)})")
    print(f"  Mean confidence: {pred_df['confidence'].mean():.3f}")

    out_path = RES2 / f"predictions_condition_{cond}.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    all_predictions[cond] = pred_df

print("\n" + "=" * 70)
print("Predictions generated for all conditions.")
print("=" * 70)
