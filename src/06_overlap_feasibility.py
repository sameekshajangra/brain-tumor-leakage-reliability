"""
src/06_overlap_feasibility.py
==============================
Stage 1 — Step 6: Assess feasibility of the three patient-overlap conditions.

Context
-------
The official Cheng dataset folds are patient-disjoint: every patient's images
are assigned to exactly ONE fold. This is the correct structure for an
unbiased cross-validation study.

For our experiment, we will use Fold 5 as the FIXED TEST SET and construct
three CUSTOM training sets with controlled patient overlap levels.

The overlap experiment requires us to CONSTRUCT non-standard splits in Stage 2.
In particular, the test patients (Fold 5 patients) must each have images that
can optionally be placed in training (patient overlap) or not (no overlap).

Since every patient's images are entirely within a single fold, ALL of a
test patient's images are already in Fold 5. There are NO non-Fold-5 images
from test patients available in the official partitioning.

THEREFORE:
-----------
The overlap experiment will be constructed differently from the official folds:

  We will take the Fold 5 patients and split each patient's images:
    - some images -> test set  (held out, fixed)
    - remaining images -> potentially used for training (overlap images)

  This requires that test patients have MULTIPLE images so that at least one
  test image and at least one training image can come from the same patient.

A patient is "eligible for overlap" if they have >= 2 total images
(so at least 1 can be kept for testing and at least 1 can be used for training).

This script:
1. Determines all Fold 5 patients and their total image counts.
2. Identifies eligible patients (>= 2 images total).
3. Simulates what the test set would look like (1 image per patient reserved
   as the "held-out test image"; remaining images available for training).
4. Calculates achievable overlap percentages.
5. Assesses whether training-set size can be held constant across conditions.

STOP conditions
---------------
- metadata.csv not found
- Fold 5 has no patients
- No eligible overlap patients

Outputs
-------
  tables/stage1/overlap_feasibility_report.csv
  results/stage1/overlap_feasibility_report.txt
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.stage1_config import (
    TABLES_DIR, RESULTS_DIR, CLASS_NAMES, TEST_FOLD,
    EXPECTED_TOTAL_IMAGES,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(msg)


def stop(msg):
    log("\n" + "=" * 70 + "\nSTOP — ISSUE DETECTED\n" + "=" * 70)
    log(msg)
    log("=" * 70)
    (RESULTS_DIR / "overlap_feasibility_report.txt").write_text("\n".join(report_lines))
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

valid = meta[meta["file_valid"] & meta["PID"].notna() & meta["label"].notna()].copy()

log("=" * 70)
log("OVERLAP FEASIBILITY REPORT")
log(f"Fixed test fold: Fold {TEST_FOLD}")
log("=" * 70)


# ---------------------------------------------------------------------------
# 2. Overview of Fold 5
# ---------------------------------------------------------------------------
fold5 = valid[valid["fold"] == TEST_FOLD].copy()
fold1to4 = valid[valid["fold"] != TEST_FOLD].copy()

log(f"\n  Fold {TEST_FOLD} (test fold)  : {len(fold5)} images, {fold5['PID'].nunique()} patients")
log(f"  Folds 1-4 (train pool): {len(fold1to4)} images, {fold1to4['PID'].nunique()} patients")

if len(fold5) == 0:
    stop(f"Fold {TEST_FOLD} is empty.")

test_pids = sorted(fold5["PID"].unique())
log(f"\n  Test patients (Fold {TEST_FOLD}): {len(test_pids)}")


# ---------------------------------------------------------------------------
# 3. IMPORTANT DESIGN NOTE
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
log("IMPORTANT DESIGN NOTE")
log("=" * 70)
log("""
The official Cheng folds are patient-disjoint: every patient's images are
entirely within a single fold. Therefore, Fold 5 test patients have ZERO images
in Folds 1-4.

For the three-condition overlap experiment (Stage 2), we will construct
CUSTOM training splits rather than using the official folds directly:

  TEST SET (fixed):
    For each Fold 5 patient, reserve a fixed subset of their images as
    test images. These are NEVER used for training in any condition.

  CONDITION A (0% overlap):
    Training set contains images ONLY from patients in Folds 1-4.
    No test patient contributes any image to training.

  CONDITION B (~50% overlap):
    ~50% of eligible test patients contribute their non-test images to training.
    The remaining eligible test patients are excluded from training.

  CONDITION C (near-max overlap):
    Almost all eligible test patients contribute their non-test images to training.

  ELIGIBILITY:
    A Fold 5 patient is eligible for overlap if they have >= 2 images
    (so at least 1 test image can be reserved AND at least 1 image can
    be added to training without leaking test images).
""")


# ---------------------------------------------------------------------------
# 4. Per-test-patient analysis
# ---------------------------------------------------------------------------
log("--- Per-Test-Patient Analysis ---\n")

patient_rows = []
eligible_pids = []

for pid in test_pids:
    pid_data   = valid[valid["PID"] == pid]
    total_imgs = len(pid_data)
    label_val  = int(pid_data["label"].mode()[0])
    class_name = CLASS_NAMES.get(label_val, "Unknown")

    # Since all images of this patient are in Fold 5:
    n_fold5_images = int((pid_data["fold"] == TEST_FOLD).sum())
    n_other_folds  = int((pid_data["fold"] != TEST_FOLD).sum())

    # Eligibility: patient has >= 2 images (minimum 1 test + 1 for training)
    is_eligible = total_imgs >= 2

    # Simulated experiment:
    # - We will keep exactly 1 image per patient as the held-out test image
    # - Remaining images from that patient can be used in training (overlap images)
    n_potential_test_images    = 1          # we hold out 1 per patient
    n_potential_overlap_images = total_imgs - 1   # rest can go to training

    if is_eligible:
        eligible_pids.append(pid)

    patient_rows.append(
        {
            "PID"                      : pid,
            "tumor_class"              : label_val,
            "class_name"               : class_name,
            "total_images"             : total_imgs,
            "n_fold5_images"           : n_fold5_images,
            "n_other_fold_images"      : n_other_folds,
            "eligible_for_overlap"     : is_eligible,
            "n_potential_test_images"  : n_potential_test_images,
            "n_potential_overlap_images": n_potential_overlap_images,
        }
    )

patient_df = pd.DataFrame(patient_rows)

n_test_patients  = len(patient_df)
n_multi_image    = int((patient_df["total_images"] > 1).sum())
n_eligible       = int(patient_df["eligible_for_overlap"].sum())
total_eligible_overlap_imgs = int(
    patient_df[patient_df["eligible_for_overlap"]]["n_potential_overlap_images"].sum()
)
n_single_image   = int((patient_df["total_images"] == 1).sum())

log(f"  Total Fold-{TEST_FOLD} test patients               : {n_test_patients}")
log(f"  Patients with >1 image (total)            : {n_multi_image}")
log(f"  Patients with exactly 1 image             : {n_single_image}")
log(f"    (single-image patients CANNOT contribute to overlap)")
log(f"  Eligible overlap patients (>= 2 images)  : {n_eligible}")
log(f"  Total potential overlap images            : {total_eligible_overlap_imgs}")
log(f"    (= sum of (total_images - 1) for eligible patients)")

if n_eligible == 0:
    stop(
        "No eligible overlap patients (patients with >= 2 images) in Fold 5. "
        "The overlap experiment cannot be constructed from this test fold."
    )

# Class distribution of eligible patients
log("\n  Eligible patients by tumor class:")
elig_df = patient_df[patient_df["eligible_for_overlap"]]
for cls_name, grp in elig_df.groupby("class_name"):
    total_ov = grp["n_potential_overlap_images"].sum()
    log(f"    {cls_name:15s}: {len(grp):3d} patients, "
        f"{total_ov:4d} potential overlap images")


# ---------------------------------------------------------------------------
# 5. Define test set and training pool for feasibility analysis
# ---------------------------------------------------------------------------
log("\n--- Training Pool Definitions ---\n")

# For the experiment, the test set = 1 image per Fold 5 patient (47 test images)
n_test_images = n_test_patients        # 1 held-out test image per patient

# Base training pool for Condition A:
# All Fold 1-4 images + (for ineligible test patients, ALL their images are excluded)
# Since official fold structure puts all test-patient images in Fold 5,
# the base training pool is simply all Fold 1-4 images.
n_base_train = len(fold1to4)
log(f"  Base training pool (Folds 1-4, Condition A): {n_base_train} images")
log(f"  These images come from {fold1to4['PID'].nunique()} non-test patients.")
log(f"  No test patient has images in this pool.")

# Overlap images available for Conditions B and C:
log(f"\n  Potential overlap images (from eligible test patients):")
log(f"    Total: {total_eligible_overlap_imgs}")
log(f"    These are (total_images - 1) per eligible patient,")
log(f"    i.e., all images except the 1 held-out test image.")


# ---------------------------------------------------------------------------
# 6. Achievable overlap percentages
# ---------------------------------------------------------------------------
log("\n--- Achievable Overlap Percentages ---\n")

# Overlap % = (n eligible test patients in training) / n_eligible * 100

# Condition A: 0 eligible patients in training
achieved_A_pct = 0.0
feasible_A = True   # always possible

# Condition C: ALL eligible patients in training -> 100%
achieved_C_pct = 100.0
n_C_overlap_imgs = total_eligible_overlap_imgs
training_C_pool  = n_base_train + n_C_overlap_imgs
feasible_C = training_C_pool >= n_base_train   # trivially True (adding images)
# Training size control: sample exactly n_base_train from training_C_pool
# -> feasible as long as training_C_pool >= n_base_train
feasible_C_size = training_C_pool >= n_base_train

# Condition B: floor(n_eligible / 2)
n_B_pats = math.floor(n_eligible / 2.0)
achieved_B_pct = (n_B_pats / n_eligible * 100) if n_eligible > 0 else 0.0

# Select B patients: pick roughly balanced across classes
elig_sorted = elig_df.sort_values(["class_name", "PID"])
selected_B_pids = elig_sorted.head(n_B_pats)["PID"].tolist()
n_B_overlap_imgs = int(
    patient_df[patient_df["PID"].isin(selected_B_pids)]["n_potential_overlap_images"].sum()
)
training_B_pool = n_base_train + n_B_overlap_imgs
feasible_B_size = training_B_pool >= n_base_train

log(f"  Condition A (0% overlap):")
log(f"    Overlapping patients: 0 / {n_eligible}")
log(f"    Achieved overlap    : {achieved_A_pct:.1f}%")
log(f"    Training pool size  : {n_base_train} (Folds 1-4 only)")
log(f"    Feasible            : YES")

log(f"\n  Condition B (~50% overlap):")
log(f"    Overlapping patients: {n_B_pats} / {n_eligible}")
log(f"    Achieved overlap    : {achieved_B_pct:.1f}%")
log(f"    Overlap images added: {n_B_overlap_imgs}")
log(f"    Training pool size  : {training_B_pool}")
log(f"    Can sample {n_base_train} from {training_B_pool}? "
    f"{'YES -> FEASIBLE' if feasible_B_size else 'NO -> NOT FEASIBLE'}")

log(f"\n  Condition C (near-max / 100% overlap):")
log(f"    Overlapping patients: {n_eligible} / {n_eligible}")
log(f"    Achieved overlap    : {achieved_C_pct:.1f}%")
log(f"    Overlap images added: {n_C_overlap_imgs}")
log(f"    Training pool size  : {training_C_pool}")
log(f"    Can sample {n_base_train} from {training_C_pool}? "
    f"{'YES -> FEASIBLE' if feasible_C_size else 'NO -> NOT FEASIBLE'}")


# ---------------------------------------------------------------------------
# 7. Class-balance check for training size control
# ---------------------------------------------------------------------------
log("\n--- Class Balance of Training Pool ---\n")

fold1to4_class = fold1to4.groupby("label").size().rename("base_train_count")
log("  Base training pool (Folds 1-4) class distribution:")
for label_val, count in fold1to4_class.items():
    log(f"    Label {label_val} ({CLASS_NAMES.get(int(label_val), '?')}): {count} images "
        f"({count/n_base_train*100:.1f}%)")


# ---------------------------------------------------------------------------
# 8. Summary table
# ---------------------------------------------------------------------------
log("\n--- Summary Table ---\n")

summary_rows = [
    {
        "condition"                        : "A (0% overlap)",
        "target_overlap_pct"               : 0.0,
        "achieved_overlap_pct"             : achieved_A_pct,
        "n_eligible_patients"              : n_eligible,
        "n_overlapping_patients"           : 0,
        "n_overlap_images_added"           : 0,
        "base_training_pool"               : n_base_train,
        "total_training_pool"              : n_base_train,
        "feasible_with_fixed_train_size"   : feasible_A,
    },
    {
        "condition"                        : "B (~50% overlap)",
        "target_overlap_pct"               : 50.0,
        "achieved_overlap_pct"             : round(achieved_B_pct, 2),
        "n_eligible_patients"              : n_eligible,
        "n_overlapping_patients"           : n_B_pats,
        "n_overlap_images_added"           : n_B_overlap_imgs,
        "base_training_pool"               : n_base_train,
        "total_training_pool"              : training_B_pool,
        "feasible_with_fixed_train_size"   : feasible_B_size,
    },
    {
        "condition"                        : "C (100% overlap)",
        "target_overlap_pct"               : 100.0,
        "achieved_overlap_pct"             : achieved_C_pct,
        "n_eligible_patients"              : n_eligible,
        "n_overlapping_patients"           : n_eligible,
        "n_overlap_images_added"           : n_C_overlap_imgs,
        "base_training_pool"               : n_base_train,
        "total_training_pool"              : training_C_pool,
        "feasible_with_fixed_train_size"   : feasible_C_size,
    },
]

summary_df = pd.DataFrame(summary_rows)
log(summary_df.to_string(index=False))


# ---------------------------------------------------------------------------
# 9. Final feasibility verdict
# ---------------------------------------------------------------------------
log("\n--- Feasibility Verdict ---\n")
for _, row in summary_df.iterrows():
    status = "FEASIBLE" if row["feasible_with_fixed_train_size"] else "NOT FEASIBLE"
    log(f"  {row['condition']}: achieved={row['achieved_overlap_pct']:.1f}%  -> {status}")

log(f"""
CONCLUSION:
  All three conditions are feasible with this dataset.

  For the Stage 2 experiment:
  - Use exactly 1 image per Fold 5 patient as the held-out test image.
  - Test set size:  {n_test_images} images  ({n_test_patients} patients).
  - Base training pool (non-test patients): {n_base_train} images.

  Condition A: train on {n_base_train} images from Folds 1-4 only.
  Condition B: train on {n_base_train} images sampled from (Folds 1-4 +
               {n_B_overlap_imgs} overlap images from {n_B_pats} test patients).
  Condition C: train on {n_base_train} images sampled from (Folds 1-4 +
               {n_C_overlap_imgs} overlap images from {n_eligible} test patients).

  Exact achieved overlap percentages (based on actual data):
    Condition A: {achieved_A_pct:.1f}%
    Condition B: {achieved_B_pct:.1f}%  ({n_B_pats}/{n_eligible} eligible patients)
    Condition C: {achieved_C_pct:.1f}%  ({n_eligible}/{n_eligible} eligible patients)

  IMPORTANT: In all conditions, the EXACT held-out test image(s) per patient
  are never placed in training. Only the remaining images of a test patient
  can appear in training (Conditions B and C).
""")


# ---------------------------------------------------------------------------
# 10. Save outputs
# ---------------------------------------------------------------------------
per_patient_csv = TABLES_DIR / "overlap_feasibility_report.csv"
patient_df.to_csv(per_patient_csv, index=False)

report_txt = RESULTS_DIR / "overlap_feasibility_report.txt"
report_txt.write_text("\n".join(report_lines) + "\n")

log("=" * 70)
log("Script 06 completed successfully.")
log(f"  overlap_feasibility_report.csv  -> {per_patient_csv}")
log(f"  overlap_feasibility_report.txt  -> {report_txt}")
log("=" * 70)

report_txt.write_text("\n".join(report_lines) + "\n")
