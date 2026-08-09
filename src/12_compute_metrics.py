"""
src/12_compute_metrics.py
==========================
Stage 2 — Step 6: Compute all performance and confidence metrics.

Performance metrics per condition:
  1. Accuracy
  2. Macro Precision
  3. Macro Recall
  4. Macro F1
  5. Balanced Accuracy
  6. Per-class Precision / Recall / F1
  7. Confusion Matrix

Confidence metrics per condition:
  1. Mean confidence
  2. Median confidence
  3. Mean confidence for correct predictions
  4. Mean confidence for incorrect predictions
  5. High-confidence error rate at >= 0.90
  6. High-confidence error rate at >= 0.95
  7. N incorrect with confidence >= 0.90
  8. N incorrect with confidence >= 0.95

Outputs
-------
  tables/stage2/pilot_performance_results.csv
  tables/stage2/pilot_confidence_results.csv
  tables/stage2/confidence_correctness.csv
  tables/stage2/confusion_matrix_{A,B,C}.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES2    = PROJECT_ROOT / "results" / "stage2"
TABLES2 = PROJECT_ROOT / "tables" / "stage2"
TABLES2.mkdir(parents=True, exist_ok=True)

CLASS_NAMES_0 = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}

print("=" * 70)
print("COMPUTING PERFORMANCE AND CONFIDENCE METRICS")
print("=" * 70)

# Load overlap metadata
integrity_df = pd.read_csv(TABLES2 / "split_integrity_report.csv")

perf_rows    = []
conf_rows    = []
cc_rows      = []  # confidence vs correctness

for cond in ["A", "B", "C"]:
    pred_path = RES2 / f"predictions_condition_{cond}.csv"
    if not pred_path.exists():
        print(f"STOP — Missing predictions: {pred_path}")
        sys.exit(1)

    df = pd.read_csv(pred_path)
    print(f"\n--- Condition {cond} ---")
    print(f"  N test images: {len(df)}")

    y_true = df["true_label"].values - 1   # 0-indexed
    y_pred = df["predicted_label"].values - 1
    conf   = df["confidence"].values
    correct = df["correct"].values.astype(bool)

    # ---- Performance ----
    acc    = accuracy_score(y_true, y_pred)
    mac_p  = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mac_r  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mac_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Per-class
    p_cls = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])
    r_cls = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])
    f_cls = f1_score(y_true, y_pred, average=None, zero_division=0, labels=[0,1,2])

    # Overlap metadata
    int_row = integrity_df[integrity_df["condition"] == cond].iloc[0]
    ov_pct  = float(int_row["overlap_percentage"])
    ov_pats = int(int_row["overlap_patients"])
    ov_imgs = int(int_row["overlap_images_in_train"])

    perf_rows.append({
        "condition"                : cond,
        "overlap_percentage"       : ov_pct,
        "overlapping_patients"     : ov_pats,
        "overlapping_train_images" : ov_imgs,
        "train_images"             : int(int_row["train_images"]),
        "test_images"              : len(df),
        "accuracy"                 : round(acc, 4),
        "macro_precision"          : round(mac_p, 4),
        "macro_recall"             : round(mac_r, 4),
        "macro_f1"                 : round(mac_f1, 4),
        "balanced_accuracy"        : round(bal_acc, 4),
        "precision_meningioma"     : round(p_cls[0], 4),
        "precision_glioma"         : round(p_cls[1], 4),
        "precision_pituitary"      : round(p_cls[2], 4),
        "recall_meningioma"        : round(r_cls[0], 4),
        "recall_glioma"            : round(r_cls[1], 4),
        "recall_pituitary"         : round(r_cls[2], 4),
        "f1_meningioma"            : round(f_cls[0], 4),
        "f1_glioma"                : round(f_cls[1], 4),
        "f1_pituitary"             : round(f_cls[2], 4),
    })

    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Macro F1:          {mac_f1:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cm_df = pd.DataFrame(cm,
                         index=[f"True_{n}" for n in CLASS_NAMES_0.values()],
                         columns=[f"Pred_{n}" for n in CLASS_NAMES_0.values()])
    cm_df.to_csv(TABLES2 / f"confusion_matrix_{cond}.csv")
    print(f"\n  Confusion matrix:\n{cm_df.to_string()}")

    # ---- Confidence ----
    mean_conf  = np.mean(conf)
    med_conf   = np.median(conf)
    mean_c_cor = np.mean(conf[correct]) if correct.any() else float("nan")
    mean_c_inc = np.mean(conf[~correct]) if (~correct).any() else float("nan")

    hi90_all   = conf >= 0.90
    hi95_all   = conf >= 0.95
    hi90_err   = hi90_all & ~correct
    hi95_err   = hi95_all & ~correct

    n_hi90     = hi90_all.sum()
    n_hi95     = hi95_all.sum()
    n_hi90_err = hi90_err.sum()
    n_hi95_err = hi95_err.sum()

    hcer90 = n_hi90_err / n_hi90 if n_hi90 > 0 else float("nan")
    hcer95 = n_hi95_err / n_hi95 if n_hi95 > 0 else float("nan")

    conf_rows.append({
        "condition"                       : cond,
        "mean_confidence"                 : round(mean_conf, 4),
        "median_confidence"               : round(med_conf, 4),
        "mean_confidence_correct"         : round(mean_c_cor, 4),
        "mean_confidence_incorrect"       : round(float(mean_c_inc) if not np.isnan(mean_c_inc) else float("nan"), 4),
        "confidence_gap"                  : round(float(mean_c_cor - mean_c_inc) if not np.isnan(mean_c_inc) else float("nan"), 4),
        "n_hi90"                          : int(n_hi90),
        "n_hi90_incorrect"                : int(n_hi90_err),
        "hcer_90"                         : round(hcer90, 4),
        "n_hi95"                          : int(n_hi95),
        "n_hi95_incorrect"                : int(n_hi95_err),
        "hcer_95"                         : round(hcer95, 4),
    })

    print(f"\n  Mean confidence:           {mean_conf:.4f}")
    print(f"  Mean conf (correct):       {mean_c_cor:.4f}")
    print(f"  Mean conf (incorrect):     {mean_c_inc:.4f}")
    print(f"  HCER >=0.90:               {hcer90:.4f}")
    print(f"  HCER >=0.95:               {hcer95:.4f}")

    # ---- Confidence vs Correctness ----
    for _, row in df.iterrows():
        cc_rows.append({
            "condition" : cond,
            "image_id"  : int(row["image_id"]),
            "true_class": row["true_class"],
            "pred_class": row["predicted_class"],
            "confidence": float(row["confidence"]),
            "correct"   : int(row["correct"]),
        })

# ---- Save tables ----
perf_df = pd.DataFrame(perf_rows)
conf_df = pd.DataFrame(conf_rows)
cc_df   = pd.DataFrame(cc_rows)

perf_df.to_csv(TABLES2 / "pilot_performance_results.csv", index=False)
conf_df.to_csv(TABLES2 / "pilot_confidence_results.csv", index=False)
cc_df.to_csv(TABLES2 / "confidence_correctness.csv", index=False)

print("\n" + "=" * 70)
print("Performance and confidence metrics computed.")
print(f"  pilot_performance_results.csv -> {TABLES2 / 'pilot_performance_results.csv'}")
print(f"  pilot_confidence_results.csv  -> {TABLES2 / 'pilot_confidence_results.csv'}")
print(f"  confidence_correctness.csv    -> {TABLES2 / 'confidence_correctness.csv'}")
print("=" * 70)
