"""
Post-hoc patient diversity and overlap-intensity sensitivity analysis.
READ-ONLY with respect to all existing Stage 3 results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
SPLITS_DIR = BASE_DIR / "results" / "stage3" / "splits"
TABLES_DIR = BASE_DIR / "tables" / "stage3"
FIG_DIR   = BASE_DIR / "figures" / "stage3"
RESULTS_DIR = BASE_DIR / "results" / "stage3"

# ─── helpers ──────────────────────────────────────────────────────────────────

def load_train(fold, cond):
    return pd.read_csv(SPLITS_DIR / f"fold{fold}_{cond}_train.csv")

def load_test(fold, cond):
    return pd.read_csv(SPLITS_DIR / f"fold{fold}_{cond}_test.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — TRAINING PATIENT DIVERSITY
# ═══════════════════════════════════════════════════════════════════════════════
print("PART 1 — training patient diversity")

div_rows = []
for fold in range(1, 6):
    for cond in ["A", "B", "C"]:
        tr = load_train(fold, cond)
        counts = tr.groupby("PID").size()
        div_rows.append({
            "fold": fold, "condition": cond,
            "training_images":            len(tr),
            "unique_training_patients":   len(counts),
            "mean_images_per_patient":    counts.mean(),
            "median_images_per_patient":  counts.median(),
            "min_images_per_patient":     counts.min(),
            "max_images_per_patient":     counts.max(),
            "patients_exactly_1":         (counts == 1).sum(),
            "patients_ge_2":              (counts >= 2).sum(),
            "patients_ge_5":              (counts >= 5).sum(),
            "patients_ge_10":             (counts >= 10).sum(),
        })

div_df = pd.DataFrame(div_rows)
div_df.to_csv(TABLES_DIR / "patient_diversity_by_fold_condition.csv", index=False)
print("  → saved patient_diversity_by_fold_condition.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — OVERLAP INTENSITY  (read overlap_selection_summary)
# ═══════════════════════════════════════════════════════════════════════════════
print("PART 2 — overlap intensity")

ov_src = pd.read_csv(TABLES_DIR / "overlap_selection_summary.csv")

intensity_rows = []
for fold in range(1, 6):
    for cond in ["A", "B", "C"]:
        sub = ov_src[(ov_src["outer_fold"] == fold) & (ov_src["condition"] == cond)]
        if sub.empty:
            continue
        row = sub.iloc[0]

        eligible     = int(row["eligible_test_patients"])
        overlapping  = int(row["overlapping_patients"])
        ov_pct       = float(row["overlap_percentage"])
        ov_images    = int(row["overlap_images"])
        train_total  = int(row["train_images"])

        pct_train_from_ov = (ov_images / train_total * 100) if train_total > 0 else 0.0

        # Per-patient stats for overlapping patients
        tr = load_train(fold, cond)
        te = load_test(fold, cond)
        test_pids = set(te["PID"].unique())
        ov_tr = tr[tr["PID"].isin(test_pids)]

        if len(ov_tr) > 0:
            per_pat = ov_tr.groupby("PID").size()
            mean_ov   = per_pat.mean()
            median_ov = per_pat.median()
            max_ov    = per_pat.max()
        else:
            mean_ov = median_ov = max_ov = 0.0

        intensity_rows.append({
            "fold": fold, "condition": cond,
            "eligible_test_patients":         eligible,
            "overlapping_test_patients":      overlapping,
            "pct_eligible_overlapping":       ov_pct,
            "overlap_training_images":        ov_images,
            "pct_training_from_overlap":      round(pct_train_from_ov, 2),
            "mean_nontest_imgs_per_overlap_patient":   round(mean_ov, 2),
            "median_nontest_imgs_per_overlap_patient": round(median_ov, 2),
            "max_nontest_imgs_per_overlap_patient":    int(max_ov),
        })

intensity_df = pd.DataFrame(intensity_rows)
intensity_df.to_csv(TABLES_DIR / "overlap_intensity_by_fold_condition.csv", index=False)
print("  → saved overlap_intensity_by_fold_condition.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — TRAINING SIZE & CLASS MATCHING INTEGRITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════
print("PART 3 — integrity check")

label_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}
integrity_rows = []
discrepancy_found = False

for fold in range(1, 6):
    for cond in ["A", "B", "C"]:
        tr = load_train(fold, cond)
        te = load_test(fold, cond)
        test_pids = set(te["PID"].unique())

        ov_sub = intensity_df[(intensity_df["fold"] == fold) & (intensity_df["condition"] == cond)]
        ov_pats = int(ov_sub["overlapping_test_patients"].iloc[0]) if not ov_sub.empty else 0
        ov_imgs = int(ov_sub["overlap_training_images"].iloc[0])   if not ov_sub.empty else 0

        # class column is named "label" in train files
        label_col = "label" if "label" in tr.columns else "class"
        class_counts = tr[label_col].value_counts().to_dict()

        integrity_rows.append({
            "fold": fold, "condition": cond,
            "training_images": len(tr),
            "meningioma": class_counts.get(1, 0),
            "glioma":     class_counts.get(2, 0),
            "pituitary":  class_counts.get(3, 0),
            "unique_training_patients":   tr["PID"].nunique(),
            "overlapping_test_patients":  ov_pats,
            "overlap_training_images":    ov_imgs,
        })

integrity_df = pd.DataFrame(integrity_rows)
integrity_df.to_csv(TABLES_DIR / "patient_diversity_integrity_check.csv", index=False)

# Verify identical image counts and class distributions within each fold
for fold in range(1, 6):
    fold_int = integrity_df[integrity_df["fold"] == fold]
    imgs = fold_int["training_images"].unique()
    for cls in ["meningioma", "glioma", "pituitary"]:
        vals = fold_int[cls].unique()
        if len(vals) > 1:
            print(f"  [DISCREPANCY] Fold {fold} {cls}: {vals}")
            discrepancy_found = True
    if len(imgs) > 1:
        print(f"  [DISCREPANCY] Fold {fold} training_images: {imgs}")
        discrepancy_found = True

if not discrepancy_found:
    print("  → Integrity check PASSED: all A/B/C have identical training image counts and class distributions")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — TEST-PATIENT FAMILIARITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("PART 4 — familiarity analysis")

pat_df = pd.read_csv(TABLES_DIR / "patient_overlap_prediction_analysis.csv")
print(f"  Columns: {list(pat_df.columns)}")

# images_in_training = number of training images from same PID under that condition
# Bin the variable
bins = [-1, 0, 5, 15, 30, 1000]
labels = ["0 (A patients)", "1–5", "6–15", "16–30", ">30"]
pat_df["training_bin"] = pd.cut(pat_df["images_in_training"], bins=bins, labels=labels)

fam_summary = pat_df.groupby("training_bin", observed=True).agg(
    n            = ("correct", "count"),
    accuracy     = ("correct", "mean"),
    mean_conf    = ("confidence", "mean"),
    median_conf  = ("confidence", "median"),
    hcer_90      = ("confidence", lambda x: ((x >= 0.90) & (pat_df.loc[x.index, "correct"] == 0)).mean()),
).reset_index()
fam_summary.to_csv(TABLES_DIR / "familiarity_bin_analysis.csv", index=False)
print("  → saved familiarity_bin_analysis.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — DOSE-RESPONSE SENSITIVITY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("PART 5 — dose-response sensitivity")

complete_df = pd.read_csv(TABLES_DIR / "complete_fold_condition_results.csv")
metrics = ["accuracy", "macro_f1", "balanced_accuracy", "mean_confidence", "ece", "brier_score", "hcer_90"]

dr_rows = []
for metric in metrics:
    for cond1, cond2, label in [("A", "B", "A_to_B"), ("B", "C", "B_to_C")]:
        fold_vals = []
        for fold in range(1, 6):
            v1 = complete_df[(complete_df["fold"]==fold) & (complete_df["condition"]==cond1)][metric].values[0]
            v2 = complete_df[(complete_df["fold"]==fold) & (complete_df["condition"]==cond2)][metric].values[0]
            fold_vals.append(v2 - v1)

        dr_rows.append({
            "metric": metric, "step": label,
            "fold1_diff": round(fold_vals[0], 4),
            "fold2_diff": round(fold_vals[1], 4),
            "fold3_diff": round(fold_vals[2], 4),
            "fold4_diff": round(fold_vals[3], 4),
            "fold5_diff": round(fold_vals[4], 4),
            "mean_diff":   round(np.mean(fold_vals), 4),
            "median_diff": round(np.median(fold_vals), 4),
            "n_increasing": sum(v > 0 for v in fold_vals),
            "n_decreasing": sum(v < 0 for v in fold_vals),
            "n_unchanged":  sum(v == 0 for v in fold_vals),
        })

dr_df = pd.DataFrame(dr_rows)
dr_df.to_csv(TABLES_DIR / "dose_response_sensitivity.csv", index=False)
print("  → saved dose_response_sensitivity.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 6 — EXPLORATORY ASSOCIATION AT FOLD-CONDITION LEVEL
# ═══════════════════════════════════════════════════════════════════════════════
print("PART 6 — exploratory fold-condition associations (n=15, descriptive only)")

merged = complete_df.merge(intensity_df[["fold","condition","overlap_training_images","pct_training_from_overlap"]],
                           on=["fold","condition"])
for metric in ["accuracy", "mean_confidence", "ece", "brier_score", "hcer_90"]:
    r = merged["overlap_training_images"].corr(merged[metric])
    print(f"  Pearson r (overlap_training_images vs {metric}): {r:.3f}  [exploratory, n=15]")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 11 — FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print("PART 11 — generating figures")

# Figure 1: unique training patients per condition across folds
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
cond_colors = {"A": "#4477AA", "B": "#EE7733", "C": "#228833"}
for ax, cond in zip(axes, ["A", "B", "C"]):
    sub = div_df[div_df["condition"] == cond]
    ax.bar(sub["fold"], sub["unique_training_patients"], color=cond_colors[cond], edgecolor="white")
    ax.set_title(f"Condition {cond}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Outer Fold", fontsize=11)
    ax.set_ylabel("Unique Training Patients", fontsize=11)
    ax.set_xticks(range(1, 6))
    ax.grid(axis="y", alpha=0.3)
    for i, row in sub.iterrows():
        ax.text(row["fold"], row["unique_training_patients"] + 0.5,
                str(int(row["unique_training_patients"])), ha="center", va="bottom", fontsize=9)
fig.suptitle("Unique Training Patients by Fold and Condition\n(matched training-image counts)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "training_patient_diversity.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 2: overlap intensity – overlap training images and pct
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for cond, marker, color in [("B","s","#EE7733"), ("C","^","#228833")]:
    sub = intensity_df[intensity_df["condition"] == cond]
    axes[0].plot(sub["fold"], sub["overlap_training_images"],
                 marker=marker, color=color, label=f"Cond {cond}", linewidth=2)
    axes[1].plot(sub["fold"], sub["pct_training_from_overlap"],
                 marker=marker, color=color, label=f"Cond {cond}", linewidth=2)

axes[0].axhline(0, linestyle="--", color="steelblue", label="Cond A (0)")
axes[1].axhline(0, linestyle="--", color="steelblue", label="Cond A (0%)")

for ax, ylabel, title in [
    (axes[0], "Overlap Training Images (count)", "Number of Training Images from Test-Fold Patients"),
    (axes[1], "% of Training Set from Test-Fold Patients", "Proportion of Training Set from Test-Fold Patients"),
]:
    ax.set_xlabel("Outer Fold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(1, 6))
    ax.legend()
    ax.grid(alpha=0.3)

plt.suptitle("Overlap Intensity by Fold and Condition", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "overlap_intensity.png", dpi=300, bbox_inches="tight")
plt.close()

# Figure 3: confidence vs training-image count (bin level)
if "training_bin" in pat_df.columns and pat_df["images_in_training"].max() > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fam_summary_plot = fam_summary.dropna(subset=["training_bin"])
    axes[0].bar(range(len(fam_summary_plot)), fam_summary_plot["mean_conf"],
                color="#4477AA", edgecolor="white")
    axes[0].set_xticks(range(len(fam_summary_plot)))
    axes[0].set_xticklabels(fam_summary_plot["training_bin"].astype(str), rotation=20, ha="right")
    axes[0].set_ylabel("Mean Confidence", fontsize=11)
    axes[0].set_title("Mean Confidence vs. Same-PID Training Images\n(exploratory, n=700 predictions)", fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(len(fam_summary_plot)), fam_summary_plot["accuracy"],
                color="#228833", edgecolor="white")
    axes[1].set_xticks(range(len(fam_summary_plot)))
    axes[1].set_xticklabels(fam_summary_plot["training_bin"].astype(str), rotation=20, ha="right")
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].set_title("Accuracy vs. Same-PID Training Images\n(exploratory, n=700 predictions)", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Exploratory Patient-Level Familiarity Analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "patient_overlap_vs_confidence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  → saved patient_overlap_vs_confidence.png")

print("\nAll analysis complete.")
