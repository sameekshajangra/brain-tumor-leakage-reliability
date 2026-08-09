"""
src/13_calibration.py
======================
Stage 2 — Step 7: Calibration metrics and reliability diagrams.

Calibration metrics
-------------------
1. Expected Calibration Error (ECE)
   - 15 equal-width bins over [0, 1]
   - ECE = Σ_b (|B_b| / N) * |accuracy(B_b) - confidence(B_b)|
   - Where B_b = set of samples whose max probability falls in bin b,
     accuracy(B_b) = fraction correct in bin b,
     confidence(B_b) = mean max probability in bin b.
   - Identical implementation used for all three conditions.

2. Multiclass Brier Score
   - BS = (1/N) * Σ_i Σ_c (p_{ic} - y_{ic})^2
   - Where p_{ic} = predicted probability for class c for sample i,
         y_{ic} = 1 if true class is c else 0.
   - Lower is better; perfect prediction = 0.

Both metrics use the SAME implementation and parameters for A, B, C.
Bins and procedure are NOT changed after seeing results.

Outputs
-------
  tables/stage2/pilot_calibration_results.csv
  figures/stage2/reliability_condition_{A,B,C}.png
  figures/stage2/reliability_comparison.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES2    = PROJECT_ROOT / "results" / "stage2"
TABLES2 = PROJECT_ROOT / "tables" / "stage2"
FIGS2   = PROJECT_ROOT / "figures" / "stage2"
FIGS2.mkdir(parents=True, exist_ok=True)

N_BINS    = 15
BIN_EDGES = np.linspace(0, 1, N_BINS + 1)

COLORS = {"A": "#2196F3", "B": "#FF9800", "C": "#4CAF50"}
CONDITION_LABELS = {
    "A": "A — 0% overlap",
    "B": "B — 50% overlap",
    "C": "C — 100% overlap",
}


def compute_ece(confidences, correct, n_bins=15):
    """
    Compute Expected Calibration Error.

    Parameters
    ----------
    confidences : array of shape (N,) — max predicted probability per sample
    correct     : boolean array of shape (N,) — True if prediction correct
    n_bins      : number of equal-width bins

    Returns
    -------
    ece        : float
    bin_data   : list of dicts with per-bin statistics
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    N = len(confidences)

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        # Include right endpoint only for last bin
        if b < n_bins - 1:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)
        n_b = mask.sum()
        if n_b == 0:
            bin_data.append({"bin": b, "lo": lo, "hi": hi, "n": 0,
                             "acc": float("nan"), "conf": float("nan"), "gap": float("nan")})
            continue
        acc_b  = correct[mask].mean()
        conf_b = confidences[mask].mean()
        gap_b  = abs(acc_b - conf_b)
        ece    += (n_b / N) * gap_b
        bin_data.append({"bin": b, "lo": lo, "hi": hi, "n": int(n_b),
                         "acc": acc_b, "conf": conf_b, "gap": gap_b})

    return ece, bin_data


def compute_brier_multiclass(probs, y_true_1indexed, n_classes=3):
    """
    Compute multiclass Brier score.
    BS = (1/N) * Σ_i Σ_c (p_{ic} - y_{ic})^2

    Parameters
    ----------
    probs          : array (N, C) predicted probabilities
    y_true_1indexed: array (N,) true labels, 1-indexed
    n_classes      : int

    Returns
    -------
    brier_score : float
    """
    N = len(y_true_1indexed)
    y_onehot = np.zeros((N, n_classes))
    for i, lbl in enumerate(y_true_1indexed):
        y_onehot[i, lbl - 1] = 1.0
    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))
    return brier


print("=" * 70)
print("CALIBRATION METRICS AND RELIABILITY DIAGRAMS")
print(f"ECE: {N_BINS} equal-width bins, [0, 1]")
print("Brier: multiclass (sum of squared prob errors, mean over samples)")
print("=" * 70)

cal_rows = []
all_bin_data = {}

for cond in ["A", "B", "C"]:
    pred_path = RES2 / f"predictions_condition_{cond}.csv"
    if not pred_path.exists():
        print(f"STOP — Missing predictions: {pred_path}")
        sys.exit(1)

    df = pd.read_csv(pred_path)

    confidences = df["confidence"].values
    correct     = df["correct"].values.astype(bool)
    y_true      = df["true_label"].values       # 1-indexed
    probs       = df[["probability_meningioma",
                       "probability_glioma",
                       "probability_pituitary"]].values

    ece, bin_data = compute_ece(confidences, correct, n_bins=N_BINS)
    brier         = compute_brier_multiclass(probs, y_true)

    all_bin_data[cond] = bin_data

    print(f"\n  Condition {cond}: ECE={ece:.4f}, Brier={brier:.4f}")

    int_df = pd.read_csv(TABLES2 / "split_integrity_report.csv")
    int_row = int_df[int_df["condition"] == cond].iloc[0]

    cal_rows.append({
        "condition"           : cond,
        "overlap_percentage"  : float(int_row["overlap_percentage"]),
        "ece"                 : round(ece, 6),
        "brier_score"         : round(brier, 6),
        "ece_bins"            : N_BINS,
        "ece_scheme"          : "equal_width",
        "n_test_images"       : len(df),
    })

    # ---- Reliability diagram (single condition) ----
    fig, ax = plt.subplots(figsize=(5, 5))
    bin_lowers = [b["lo"] for b in bin_data if not np.isnan(b["acc"])]
    bin_uppers = [b["hi"] for b in bin_data if not np.isnan(b["acc"])]
    bin_accs   = [b["acc"] for b in bin_data if not np.isnan(b["acc"])]
    bin_confs  = [b["conf"] for b in bin_data if not np.isnan(b["acc"])]
    bin_widths = [u - l for l, u in zip(bin_lowers, bin_uppers)]
    bin_midpts = [(l + u) / 2 for l, u in zip(bin_lowers, bin_uppers)]

    ax.bar(bin_midpts, bin_accs, width=bin_widths,
           alpha=0.7, color=COLORS[cond], label="Fraction correct", edgecolor="gray", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted confidence", fontsize=11)
    ax.set_ylabel("Fraction of correct predictions", fontsize=11)
    ax.set_title(f"Reliability Diagram — Condition {CONDITION_LABELS[cond]}\n"
                 f"ECE = {ece:.4f}", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS2 / f"reliability_condition_{cond}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: reliability_condition_{cond}.png")

# ---- Combined reliability diagram ----
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, sharex=True)
cal_df = pd.DataFrame(cal_rows)

for ax, cond in zip(axes, ["A", "B", "C"]):
    ece_val  = cal_df[cal_df["condition"] == cond]["ece"].values[0]
    bd       = all_bin_data[cond]
    bin_lo   = [b["lo"] for b in bd if not np.isnan(b.get("acc", float("nan")))]
    bin_hi   = [b["hi"] for b in bd if not np.isnan(b.get("acc", float("nan")))]
    bin_accs = [b["acc"] for b in bd if not np.isnan(b.get("acc", float("nan")))]
    widths   = [u - l for l, u in zip(bin_lo, bin_hi)]
    midpts   = [(l + u) / 2 for l, u in zip(bin_lo, bin_hi)]

    ax.bar(midpts, bin_accs, width=widths, alpha=0.75,
           color=COLORS[cond], edgecolor="gray", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5)
    ax.set_title(f"Condition {cond}\n({CONDITION_LABELS[cond]})\nECE={ece_val:.4f}", fontsize=10)
    ax.set_xlabel("Confidence", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Fraction correct", fontsize=10)
fig.suptitle("Reliability Diagrams — Conditions A, B, C", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIGS2 / "reliability_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n  Saved: reliability_comparison.png")

# Save calibration table
cal_df.to_csv(TABLES2 / "pilot_calibration_results.csv", index=False)

print("\n" + "=" * 70)
print("Calibration complete.")
print(f"  pilot_calibration_results.csv -> {TABLES2 / 'pilot_calibration_results.csv'}")
print("=" * 70)
