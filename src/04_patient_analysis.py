"""
src/04_patient_analysis.py
==========================
Stage 1 — Step 4: Patient-level distribution analysis.

For every PID calculates:
  - total number of images
  - number of images in each official fold (independent fold-disjoint verification)
  - tumor class
  - whether the patient appears in more than one fold (flag)

Also calculates global patient stats:
  - mean / median / min / max images per patient
  - number of patients with >1, >=5, >=10 images
  - patient counts broken down by tumor class

Outputs
-------
  tables/stage1/patient_distribution.csv
  figures/stage1/images_per_patient_distribution.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.stage1_config import (
    TABLES_DIR, FIGURES_DIR, CLASS_NAMES,
    EXPECTED_TOTAL_PATIENTS,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def stop(msg):
    print("\n" + "=" * 70 + "\nSTOP — ISSUE DETECTED\n" + "=" * 70)
    print(msg)
    print("=" * 70)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Load metadata
# ---------------------------------------------------------------------------
meta_csv = TABLES_DIR / "metadata.csv"
if not meta_csv.exists():
    stop(f"metadata.csv not found at {meta_csv}. Run 02_build_metadata.py first.")

meta = pd.read_csv(meta_csv)
meta["PID"]   = meta["PID"].astype(str)
meta["label"] = meta["label"].astype("Int64")
meta["fold"]  = meta["fold"].astype("Int64")

# Use only valid files with known PID/label/fold
valid = meta[meta["file_valid"] & meta["PID"].notna() & meta["label"].notna()].copy()

print(f"Valid records for patient analysis: {len(valid)} / {len(meta)}")


# ---------------------------------------------------------------------------
# 2. Build per-patient table
# ---------------------------------------------------------------------------
patient_rows = []

for pid, grp in valid.groupby("PID"):
    total_images = len(grp)
    label_val    = int(grp["label"].mode()[0])   # should be unique per PID
    class_name   = CLASS_NAMES.get(label_val, "Unknown")

    folds_present = sorted(grp["fold"].dropna().astype(int).unique().tolist())

    # Per-fold image counts
    fold_counts = {}
    for f in range(1, 6):
        fold_counts[f"images_fold_{f}"] = int((grp["fold"] == f).sum())

    # Flag: patient in more than one fold (should be False for official folds)
    in_multiple_folds = len(folds_present) > 1

    row = {
        "PID"               : pid,
        "total_images"      : total_images,
        "tumor_class"       : label_val,
        "class_name"        : class_name,
        "folds_present"     : str(folds_present),
        "n_folds"           : len(folds_present),
        "in_multiple_folds" : in_multiple_folds,
    }
    row.update(fold_counts)
    patient_rows.append(row)

patient_df = pd.DataFrame(patient_rows).sort_values("PID")

# Sanity: PIDs in multiple folds
multi_fold_pids = patient_df[patient_df["in_multiple_folds"]]
if len(multi_fold_pids) > 0:
    print(f"\nWARNING: {len(multi_fold_pids)} PID(s) appear in more than one fold:")
    print(multi_fold_pids[["PID", "folds_present", "total_images"]].to_string(index=False))
    # Note: this is already caught in script 03; here it's a secondary check
else:
    print("[OK] Independent check: no PID appears in more than one fold.")


# ---------------------------------------------------------------------------
# 3. Global patient statistics
# ---------------------------------------------------------------------------
images_per_patient = patient_df["total_images"].values

print("\n--- Global Patient Statistics ---")
print(f"  Total patients       : {len(patient_df)}")
print(f"  Mean images/patient  : {images_per_patient.mean():.2f}")
print(f"  Median               : {np.median(images_per_patient):.1f}")
print(f"  Min                  : {images_per_patient.min()}")
print(f"  Max                  : {images_per_patient.max()}")
print(f"  Std                  : {images_per_patient.std():.2f}")
print(f"  Patients with >1 img : {(images_per_patient > 1).sum()}")
print(f"  Patients with >=5    : {(images_per_patient >= 5).sum()}")
print(f"  Patients with >=10   : {(images_per_patient >= 10).sum()}")

if len(patient_df) != EXPECTED_TOTAL_PATIENTS:
    stop(
        f"Patient count mismatch: found {len(patient_df)}, "
        f"expected {EXPECTED_TOTAL_PATIENTS}."
    )

print("\n--- Patient Count by Tumor Class ---")
class_summary = patient_df.groupby("class_name").agg(
    n_patients=("PID", "count"),
    total_images=("total_images", "sum"),
    mean_images=("total_images", "mean"),
    min_images=("total_images", "min"),
    max_images=("total_images", "max"),
).reset_index()
print(class_summary.to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Save patient_distribution.csv
# ---------------------------------------------------------------------------
# Add aggregate stats as columns for convenience
patient_df["mean_images_per_patient"]   = images_per_patient.mean()
patient_df["median_images_per_patient"] = np.median(images_per_patient)
patient_df["min_images_per_patient"]    = images_per_patient.min()
patient_df["max_images_per_patient"]    = images_per_patient.max()

patient_csv = TABLES_DIR / "patient_distribution.csv"
patient_df.to_csv(patient_csv, index=False)
print(f"\n  Saved: {patient_csv}")


# ---------------------------------------------------------------------------
# 5. Plot: images per patient distribution
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Images per Patient — Cheng Brain Tumor Dataset", fontsize=15, fontweight="bold")

# Panel A: histogram of images per patient
ax = axes[0]
ax.hist(images_per_patient, bins=range(1, images_per_patient.max() + 2),
        color="#4C72B0", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Number of Images per Patient", fontsize=12)
ax.set_ylabel("Number of Patients", fontsize=12)
ax.set_title("Distribution of Images per Patient", fontsize=12)
ax.axvline(images_per_patient.mean(), color="#DD8452", linestyle="--",
           linewidth=1.5, label=f"Mean = {images_per_patient.mean():.1f}")
ax.axvline(np.median(images_per_patient), color="#55A868", linestyle=":",
           linewidth=1.5, label=f"Median = {np.median(images_per_patient):.1f}")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# Panel B: images per patient by tumor class
ax2 = axes[1]
class_colors = {"Meningioma": "#4C72B0", "Glioma": "#DD8452", "Pituitary": "#55A868"}
for cls_name, grp in patient_df.groupby("class_name"):
    vals = grp["total_images"].values
    ax2.hist(vals, bins=range(1, images_per_patient.max() + 2), alpha=0.65,
             label=f"{cls_name} (n={len(grp)})",
             color=class_colors.get(cls_name, "grey"), edgecolor="white", linewidth=0.5)
ax2.set_xlabel("Number of Images per Patient", fontsize=12)
ax2.set_ylabel("Number of Patients", fontsize=12)
ax2.set_title("Distribution by Tumor Class", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig_path = FIGURES_DIR / "images_per_patient_distribution.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")


# ---------------------------------------------------------------------------
# 6. Print summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Script 04 completed successfully.")
print(f"  patient_distribution.csv              -> {patient_csv}")
print(f"  images_per_patient_distribution.png   -> {fig_path}")
print("=" * 70)
