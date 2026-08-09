"""
src/08_create_overlap_splits.py
================================
Stage 2 — Step 2: Build all three experimental splits (A, B, C).

Split design
------------
Validation (fixed, identical for A/B/C):
  - 20% patient-level stratified split from Folds 1-4 (seed=42)
  - No Fold-5 patient ever enters validation
  - Same val patient set reused for all three conditions

Condition A (0% overlap):
  - Train = remaining 80% of Folds 1-4 images (N_train)
  - No Fold-5 patient images in training

Condition B (~50% overlap):
  - 23/46 eligible Fold-5 patients selected (stratified by class, seed=42)
  - FORCED: all non-test images from these 23 patients must be in train pool
  - Pool = Train-A images + forced overlap images
  - Sample exactly N_train from pool, class-balanced to match Condition A distribution
  - Forced overlap images are always included (STOP if they exceed N_train)

Condition C (100% eligible-patient overlap):
  - All 46 eligible Fold-5 patients
  - FORCED: all non-test images from these 46 patients in train pool
  - Sample exactly N_train, same procedure as B

Patient overlap rate = overlapping_patients / 46 eligible

STOP conditions
---------------
- Fixed test images not found
- Required overlap images exceed N_train (cannot fit in budget)
- Test images appear in any training set

Outputs
-------
  results/stage2/splits/condition_{A,B,C}_{train,val,test}.csv
  tables/stage2/overlap_patients_50.csv
  tables/stage2/split_integrity_report.csv
  tables/stage2/training_class_distribution.csv
"""
import sys
import random
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.stage1_config import CLASS_NAMES, TEST_FOLD

TABLES2 = PROJECT_ROOT / "tables" / "stage2"
RES2    = PROJECT_ROOT / "results" / "stage2"
SPLITS  = RES2 / "splits"
TABLES2.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

def stop(msg):
    log("\nSTOP — " + msg)
    sys.exit(1)

def save_split(df, cond, role):
    """Save split CSV with standard columns."""
    out = df[["image_id", "filename", "PID", "label", "class_name"]].copy()
    out["condition"] = cond
    out["role"] = role
    path = SPLITS / f"condition_{cond}_{role}.csv"
    out.to_csv(path, index=False)
    return path

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
log("=" * 70)
log("STAGE 2 — OVERLAP SPLIT CONSTRUCTION")
log("=" * 70)

meta = pd.read_csv(PROJECT_ROOT / "tables" / "stage1" / "metadata.csv")
meta["PID"]   = meta["PID"].astype(str)
meta["label"] = meta["label"].astype("Int64")
meta["fold"]  = meta["fold"].astype("Int64")
valid = meta[meta["file_valid"] & meta["PID"].notna() & meta["label"].notna()].copy()

test_df = pd.read_csv(TABLES2 / "fixed_test_images.csv")
test_df["PID"] = test_df["PID"].astype(str)

test_image_ids = set(test_df["test_image_id"].astype(int))
test_pids      = set(test_df["PID"].unique())
eligible_pids  = set(test_df[test_df["eligible_for_overlap"]]["PID"].unique())
N_ELIGIBLE     = len(eligible_pids)

log(f"\nFixed test images: {len(test_image_ids)}")
log(f"Test patients:     {len(test_pids)}")
log(f"Eligible patients: {N_ELIGIBLE}")

# ---------------------------------------------------------------------------
# 2. Build validation set — once, from Folds 1-4, fixed for all conditions
# ---------------------------------------------------------------------------
log("\n--- Building validation set (Folds 1-4 patients, 20% stratified) ---")

folds14 = valid[valid["fold"] != TEST_FOLD].copy()

# Patient-level stratified split
patients14 = (folds14.groupby("PID")
              .agg(label=("label", "first"), n_images=("image_id", "count"))
              .reset_index())

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_pat_idx, val_pat_idx = next(sss.split(patients14["PID"], patients14["label"]))

val_pids   = set(patients14.iloc[val_pat_idx]["PID"].tolist())
train_pids = set(patients14.iloc[train_pat_idx]["PID"].tolist())

assert val_pids.isdisjoint(test_pids), "Val patients overlap with test patients!"
assert train_pids.isdisjoint(test_pids), "Train base patients overlap with test patients!"

val_images   = folds14[folds14["PID"].isin(val_pids)].copy()
train_base   = folds14[folds14["PID"].isin(train_pids)].copy()

N_VAL   = len(val_images)
N_TRAIN = len(train_base)   # this is the reference training size

log(f"  Val patients   : {len(val_pids)}")
log(f"  Val images     : {N_VAL}")
log(f"  Train patients : {len(train_pids)}")
log(f"  Train images   : {N_TRAIN}  <- N_train (reference for all conditions)")

# Condition A reference class distribution
cond_A_class_dist = Counter(train_base["label"].tolist())
cond_A_class_pct  = {k: v/N_TRAIN for k, v in cond_A_class_dist.items()}
log("\n  Condition A reference class distribution:")
for lbl in sorted(cond_A_class_dist):
    log(f"    Label {lbl} ({CLASS_NAMES[lbl]}): {cond_A_class_dist[lbl]} ({cond_A_class_pct[lbl]*100:.1f}%)")

# ---------------------------------------------------------------------------
# 3. Build Condition A split
# ---------------------------------------------------------------------------
log("\n--- Condition A (0% overlap) ---")

# Train = all of train_base
split_A_train = train_base.copy()
split_A_val   = val_images.copy()
split_A_test  = valid[valid["image_id"].isin(test_image_ids)].copy()

assert len(split_A_train) == N_TRAIN
log(f"  Train: {len(split_A_train)} images, {split_A_train['PID'].nunique()} patients")
log(f"  Val  : {len(split_A_val)} images,   {split_A_val['PID'].nunique()} patients")
log(f"  Test : {len(split_A_test)} images,   {split_A_test['PID'].nunique()} patients")
log(f"  Overlap patients: 0 / {N_ELIGIBLE} = 0.0%")
log(f"  Overlap images in train: 0")

# ---------------------------------------------------------------------------
# 4. Select 23/46 eligible patients for Condition B (stratified, seed=42)
# ---------------------------------------------------------------------------
log("\n--- Selecting Condition B patients (23/46, stratified) ---")

elig_df = test_df[test_df["eligible_for_overlap"]].copy()
elig_df["PID"] = elig_df["PID"].astype(str)

# Stratified selection of ~50% by class
N_B = math.floor(N_ELIGIBLE / 2)   # = 23
log(f"  Target: {N_B} / {N_ELIGIBLE} = {N_B/N_ELIGIBLE*100:.1f}%")

# Proportional class counts for B
class_counts_elig = Counter(elig_df["label"].tolist())
log(f"  Eligible class distribution: {dict(class_counts_elig)}")

rng_b = random.Random(SEED)
selected_B_pids = []
for lbl in sorted(class_counts_elig):
    class_pids = sorted(elig_df[elig_df["label"] == lbl]["PID"].tolist())
    target_n   = round(N_B * class_counts_elig[lbl] / N_ELIGIBLE)
    selected   = rng_b.sample(class_pids, min(target_n, len(class_pids)))
    selected_B_pids.extend(selected)
    log(f"    Label {lbl} ({CLASS_NAMES[lbl]}): selected {len(selected)} / {len(class_pids)}")

# Adjust to exactly N_B if rounding caused mismatch
all_elig_pids = sorted(set(elig_df["PID"].tolist()) - set(selected_B_pids))
while len(selected_B_pids) < N_B and all_elig_pids:
    selected_B_pids.append(all_elig_pids.pop(0))
while len(selected_B_pids) > N_B:
    selected_B_pids.pop()

selected_B_pids = set(selected_B_pids)
log(f"  Final selected B patients: {len(selected_B_pids)}")
achieved_B_pct = len(selected_B_pids) / N_ELIGIBLE * 100
log(f"  Achieved B overlap rate: {len(selected_B_pids)}/{N_ELIGIBLE} = {achieved_B_pct:.1f}%")

# Save overlap_patients_50.csv
overlap50_rows = []
for pid in sorted(elig_df["PID"].unique()):
    row = elig_df[elig_df["PID"] == pid].iloc[0]
    overlap50_rows.append({
        "PID": pid,
        "label": int(row["label"]),
        "class_name": row["class_name"],
        "total_images": int(row["total_patient_images"]),
        "n_test_images": 1,
        "n_non_test_images": int(row["n_non_test_images"]),
        "selected_for_overlap_B": pid in selected_B_pids,
    })
pd.DataFrame(overlap50_rows).to_csv(TABLES2 / "overlap_patients_50.csv", index=False)

# ---------------------------------------------------------------------------
# Helper: build Condition B or C training pool + sample to N_TRAIN
# ---------------------------------------------------------------------------
def build_overlap_train(overlap_pids, condition_name):
    """
    Build training set for an overlap condition.
    Forced: all non-test images from overlap_pids are included.
    Fill: sample from train_base (class-balanced, matching Condition A distribution).
    Final size must equal N_TRAIN.
    """
    # Collect non-test images from overlap patients
    overlap_imgs = valid[
        valid["PID"].isin(overlap_pids) &
        ~valid["image_id"].isin(test_image_ids) &
        ~valid["PID"].isin(val_pids)       # keep val patients out of train
    ].copy()

    n_forced = len(overlap_imgs)
    n_fill   = N_TRAIN - n_forced

    log(f"\n  {condition_name}: forced overlap images = {n_forced}, fill needed = {n_fill}")

    if n_forced > N_TRAIN:
        stop(
            f"{condition_name}: Required overlap images ({n_forced}) "
            f"exceed N_train ({N_TRAIN}). Cannot construct this condition."
        )

    # Class distribution of forced overlap images
    forced_class_ct = Counter(overlap_imgs["label"].tolist())
    log(f"  Overlap image class dist: {dict(forced_class_ct)}")

    # Determine how many to fill per class (target = A proportion * N_TRAIN - forced)
    fill_df_parts = []
    train_base_available = train_base.copy()  # all of Folds 1-4 train images

    for lbl in sorted(cond_A_class_dist):
        target_n  = round(cond_A_class_pct[lbl] * N_TRAIN)
        forced_n  = forced_class_ct.get(lbl, 0)
        fill_n    = max(0, target_n - forced_n)
        available = train_base_available[train_base_available["label"] == lbl]
        actual_fill = min(fill_n, len(available))
        sampled = available.sample(n=actual_fill, random_state=SEED)
        fill_df_parts.append(sampled)
        log(f"    Label {lbl}: target={target_n}, forced={forced_n}, fill={actual_fill}")

    fill_df = pd.concat(fill_df_parts, ignore_index=True)
    combined = pd.concat([overlap_imgs, fill_df], ignore_index=True)

    # Adjust to exactly N_TRAIN (rounding may produce +/-1)
    if len(combined) > N_TRAIN:
        # Drop excess from fill (non-overlap), randomly
        fill_ids   = set(fill_df["image_id"])
        excess_drop = len(combined) - N_TRAIN
        drop_cands  = combined[combined["image_id"].isin(fill_ids)].sample(
            n=excess_drop, random_state=SEED)
        combined = combined[~combined["image_id"].isin(drop_cands["image_id"])]
    elif len(combined) < N_TRAIN:
        # Try to add more fill images
        used_ids  = set(combined["image_id"])
        more      = train_base[~train_base["image_id"].isin(used_ids)].sample(
            n=N_TRAIN - len(combined), random_state=SEED)
        combined  = pd.concat([combined, more], ignore_index=True)

    assert len(combined) == N_TRAIN, f"Size mismatch: {len(combined)} != {N_TRAIN}"

    # Verify no duplicate image IDs
    assert combined["image_id"].nunique() == N_TRAIN

    # Verify no test images
    leaked = set(combined["image_id"]).intersection(test_image_ids)
    if leaked:
        stop(f"{condition_name}: Test images found in training set! IDs: {leaked}")

    return combined.copy()


# ---------------------------------------------------------------------------
# 5. Build Condition B split
# ---------------------------------------------------------------------------
log("\n--- Condition B (~50% overlap) ---")

split_B_train = build_overlap_train(selected_B_pids, "Condition B")
split_B_val   = val_images.copy()
split_B_test  = split_A_test.copy()

# Count actual overlapping patients in train
B_overlap_in_train = set(split_B_train["PID"]).intersection(test_pids)
n_B_overlap_pats   = len(B_overlap_in_train)
n_B_overlap_imgs   = len(split_B_train[split_B_train["PID"].isin(test_pids)])

log(f"  Train: {len(split_B_train)} images, {split_B_train['PID'].nunique()} patients")
log(f"  Overlapping patients in train: {n_B_overlap_pats} / {N_ELIGIBLE} = {n_B_overlap_pats/N_ELIGIBLE*100:.1f}%")
log(f"  Overlapping images in train: {n_B_overlap_imgs}")


# ---------------------------------------------------------------------------
# 6. Build Condition C split
# ---------------------------------------------------------------------------
log("\n--- Condition C (100% eligible-patient overlap) ---")

split_C_train = build_overlap_train(eligible_pids, "Condition C")
split_C_val   = val_images.copy()
split_C_test  = split_A_test.copy()

C_overlap_in_train = set(split_C_train["PID"]).intersection(test_pids)
n_C_overlap_pats   = len(C_overlap_in_train)
n_C_overlap_imgs   = len(split_C_train[split_C_train["PID"].isin(test_pids)])

log(f"  Train: {len(split_C_train)} images, {split_C_train['PID'].nunique()} patients")
log(f"  Overlapping patients in train: {n_C_overlap_pats} / {N_ELIGIBLE} = {n_C_overlap_pats/N_ELIGIBLE*100:.1f}%")
log(f"  Overlapping images in train: {n_C_overlap_imgs}")

# ---------------------------------------------------------------------------
# 7. Save all split files
# ---------------------------------------------------------------------------
log("\n--- Saving splits ---")

for cond, tr, vl, te in [
    ("A", split_A_train, split_A_val, split_A_test),
    ("B", split_B_train, split_B_val, split_B_test),
    ("C", split_C_train, split_C_val, split_C_test),
]:
    save_split(tr, cond, "train")
    save_split(vl, cond, "val")
    save_split(te, cond, "test")
    log(f"  Condition {cond}: train={len(tr)}, val={len(vl)}, test={len(te)}")

# Verify test sets are identical across conditions
assert set(split_A_test["image_id"]) == set(split_B_test["image_id"]) == set(split_C_test["image_id"]), \
    "Test sets differ across conditions!"
log("\n[OK] Test sets are identical across A, B, C.")

# ---------------------------------------------------------------------------
# 8. Training class distribution table
# ---------------------------------------------------------------------------
rows_class = []
for cond, tr in [("A", split_A_train), ("B", split_B_train), ("C", split_C_train)]:
    cc = Counter(tr["label"].tolist())
    rows_class.append({
        "condition"        : cond,
        "total_train"      : len(tr),
        "n_meningioma"     : cc.get(1, 0),
        "n_glioma"         : cc.get(2, 0),
        "n_pituitary"      : cc.get(3, 0),
        "pct_meningioma"   : round(cc.get(1, 0) / len(tr) * 100, 2),
        "pct_glioma"       : round(cc.get(2, 0) / len(tr) * 100, 2),
        "pct_pituitary"    : round(cc.get(3, 0) / len(tr) * 100, 2),
    })

class_dist_df = pd.DataFrame(rows_class)
class_dist_df.to_csv(TABLES2 / "training_class_distribution.csv", index=False)
log("\n  Training class distribution:")
log(class_dist_df.to_string(index=False))

# Check deviation from Condition A
ref_row = rows_class[0]
for row in rows_class[1:]:
    for cls in ["meningioma", "glioma", "pituitary"]:
        diff = abs(row[f"pct_{cls}"] - ref_row[f"pct_{cls}"])
        if diff > 2.0:
            log(f"\n  WARNING: {row['condition']} {cls} deviates {diff:.2f}pp from A — investigate.")

# ---------------------------------------------------------------------------
# 9. Split integrity summary table
# ---------------------------------------------------------------------------
integrity_rows = []
for cond, tr, vl, te, n_ov_pats, n_ov_imgs in [
    ("A", split_A_train, split_A_val, split_A_test, 0, 0),
    ("B", split_B_train, split_B_val, split_B_test, n_B_overlap_pats, n_B_overlap_imgs),
    ("C", split_C_train, split_C_val, split_C_test, n_C_overlap_pats, n_C_overlap_imgs),
]:
    exact_leak = len(set(tr["image_id"]).intersection(test_image_ids))
    integrity_rows.append({
        "condition"                  : cond,
        "train_images"               : len(tr),
        "val_images"                 : len(vl),
        "test_images"                : len(te),
        "train_patients"             : tr["PID"].nunique(),
        "val_patients"               : vl["PID"].nunique(),
        "test_patients"              : te["PID"].nunique(),
        "overlap_patients"           : n_ov_pats,
        "n_eligible_patients"        : N_ELIGIBLE,
        "overlap_percentage"         : round(n_ov_pats / N_ELIGIBLE * 100, 2),
        "overlap_images_in_train"    : n_ov_imgs,
        "exact_image_overlap_count"  : exact_leak,
        "test_identical_to_A"        : set(te["image_id"]) == test_image_ids,
    })

integrity_df = pd.DataFrame(integrity_rows)
integrity_df.to_csv(TABLES2 / "split_integrity_report.csv", index=False)
log("\n  Split integrity table:")
log(integrity_df[["condition","train_images","val_images","test_images",
                  "overlap_patients","overlap_percentage","exact_image_overlap_count"]].to_string(index=False))

# Critical: no test-image leakage
for row in integrity_rows:
    if row["exact_image_overlap_count"] > 0:
        stop(f"Condition {row['condition']}: {row['exact_image_overlap_count']} test images found in training!")

log("\n[OK] No test images found in any training set.")

log("\n" + "=" * 70)
log("Script 08 completed successfully.")
log(f"  Splits saved to:               {SPLITS}")
log(f"  overlap_patients_50.csv     -> {TABLES2 / 'overlap_patients_50.csv'}")
log(f"  split_integrity_report.csv  -> {TABLES2 / 'split_integrity_report.csv'}")
log(f"  training_class_distribution -> {TABLES2 / 'training_class_distribution.csv'}")
log("=" * 70)
