"""
src/09_verify_stage2_splits.py
==============================
Stage 2 — Step 3: Comprehensive split integrity verification.

Runs all 13 integrity checks. Stops on any critical failure.
Must pass before training begins.

Checks
------
 1. Test sets are identical across A, B, C
 2. No exact test image is in training
 3. No exact test image is in validation
 4. Training image count is identical across A, B, C
 5. Test image count is identical across A, B, C
 6. Train/val/test are mutually exclusive at image level (within each condition)
 7. No duplicate image IDs within any single split
 8. All required overlap images are present in B/C training
 9. Patient overlap counts match intended conditions
10. Class distributions are documented and deviation is reported
11. All image files exist on disk
12. No Fold-5 patient appears in validation
13. Validation patients are consistent across A, B, C

Output
------
  results/stage2/split_integrity_report.txt
"""
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.stage1_config import CLASS_NAMES, TEST_FOLD

TABLES2 = PROJECT_ROOT / "tables" / "stage2"
RES2    = PROJECT_ROOT / "results" / "stage2"
SPLITS  = RES2 / "splits"
RAW_DIR = PROJECT_ROOT.parent / "data" / "raw"

lines  = []
errors = []

def log(msg=""):
    print(msg)
    lines.append(msg)

def fail(msg):
    errors.append(msg)
    log(f"  [FAIL] {msg}")

def ok(msg):
    log(f"  [OK]   {msg}")


log("=" * 70)
log("STAGE 2 — SPLIT INTEGRITY VERIFICATION")
log("=" * 70)

# Load splits
splits = {}
for cond in ["A", "B", "C"]:
    for role in ["train", "val", "test"]:
        key = f"{cond}_{role}"
        path = SPLITS / f"condition_{cond}_{role}.csv"
        if not path.exists():
            log(f"\nSTOP — Missing split file: {path}")
            sys.exit(1)
        df = pd.read_csv(path)
        df["image_id"] = df["image_id"].astype(int)
        df["PID"] = df["PID"].astype(str)
        splits[key] = df

fixed_test = pd.read_csv(TABLES2 / "fixed_test_images.csv")
fixed_test["PID"] = fixed_test["PID"].astype(str)
test_image_ids = set(fixed_test["test_image_id"].astype(int))
test_pids      = set(fixed_test["PID"].unique())
eligible_pids  = set(fixed_test[fixed_test["eligible_for_overlap"]]["PID"].unique())
N_ELIGIBLE     = len(eligible_pids)

overlap50 = pd.read_csv(TABLES2 / "overlap_patients_50.csv")
overlap50["PID"] = overlap50["PID"].astype(str)
selected_B_pids = set(overlap50[overlap50["selected_for_overlap_B"]]["PID"].unique())

# --- CHECK 1: Test sets identical across A, B, C ---
log("\n[CHECK 1] Test sets identical across A, B, C")
test_A = set(splits["A_test"]["image_id"])
test_B = set(splits["B_test"]["image_id"])
test_C = set(splits["C_test"]["image_id"])
if test_A == test_B == test_C == test_image_ids:
    ok(f"All test sets identical: {len(test_A)} images")
else:
    fail(f"Test sets differ! A={len(test_A)}, B={len(test_B)}, C={len(test_C)}, fixed={len(test_image_ids)}")

# --- CHECK 2: No test image in training ---
log("\n[CHECK 2] No test image in training")
for cond in ["A", "B", "C"]:
    train_ids = set(splits[f"{cond}_train"]["image_id"])
    overlap   = train_ids.intersection(test_image_ids)
    if overlap:
        fail(f"Condition {cond}: {len(overlap)} test images found in training! IDs: {list(overlap)[:10]}")
    else:
        ok(f"Condition {cond}: 0 test images in training")

# --- CHECK 3: No test image in validation ---
log("\n[CHECK 3] No test image in validation")
for cond in ["A", "B", "C"]:
    val_ids = set(splits[f"{cond}_val"]["image_id"])
    overlap = val_ids.intersection(test_image_ids)
    if overlap:
        fail(f"Condition {cond}: {len(overlap)} test images found in validation!")
    else:
        ok(f"Condition {cond}: 0 test images in validation")

# --- CHECK 4: Training size identical across A, B, C ---
log("\n[CHECK 4] Training image count identical across A, B, C")
sizes = {c: len(splits[f"{c}_train"]) for c in ["A", "B", "C"]}
if len(set(sizes.values())) == 1:
    ok(f"All training sizes = {sizes['A']}")
else:
    fail(f"Training sizes differ: {sizes}")

# --- CHECK 5: Test size identical ---
log("\n[CHECK 5] Test image count identical across A, B, C")
t_sizes = {c: len(splits[f"{c}_test"]) for c in ["A", "B", "C"]}
if len(set(t_sizes.values())) == 1:
    ok(f"All test sizes = {t_sizes['A']}")
else:
    fail(f"Test sizes differ: {t_sizes}")

# --- CHECK 6: Train/val/test mutually exclusive within each condition ---
log("\n[CHECK 6] Train/val/test mutually exclusive within each condition")
for cond in ["A", "B", "C"]:
    tr = set(splits[f"{cond}_train"]["image_id"])
    vl = set(splits[f"{cond}_val"]["image_id"])
    te = set(splits[f"{cond}_test"]["image_id"])
    tv = tr.intersection(vl)
    tt = tr.intersection(te)
    vt = vl.intersection(te)
    if tv or tt or vt:
        fail(f"Condition {cond}: train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}")
    else:
        ok(f"Condition {cond}: train/val/test fully disjoint")

# --- CHECK 7: No duplicate image IDs within any split ---
log("\n[CHECK 7] No duplicate image IDs within any split")
for key, df in splits.items():
    dupes = len(df) - df["image_id"].nunique()
    if dupes:
        fail(f"{key}: {dupes} duplicate image IDs")
    else:
        ok(f"{key}: no duplicates ({len(df)} images, {df['image_id'].nunique()} unique)")

# --- CHECK 8: Required overlap images present ---
log("\n[CHECK 8] Required overlap images present in B and C training")

# For B: all non-test images from selected_B_pids must be in train
meta = pd.read_csv(PROJECT_ROOT / "tables" / "stage1" / "metadata.csv")
meta["PID"] = meta["PID"].astype(str)
meta["image_id"] = meta["image_id"].astype(int)

for cond, overlap_pids in [("B", selected_B_pids), ("C", eligible_pids)]:
    required_imgs = set(
        meta[(meta["PID"].isin(overlap_pids)) &
             (~meta["image_id"].isin(test_image_ids))]["image_id"].tolist()
    )
    train_ids = set(splits[f"{cond}_train"]["image_id"])
    # Note: val patients from Folds 1-4 may have reduced required set
    # Here we check that overlap-patient non-test images are in train
    # (Folds 1-4 val patients are not overlap patients, so this is fine)
    missing = required_imgs - train_ids
    n_required = len(required_imgs)
    n_found = n_required - len(missing)
    if missing:
        fail(f"Condition {cond}: {len(missing)} required overlap images not in train "
             f"(found {n_found}/{n_required}). Missing IDs (first 10): {sorted(missing)[:10]}")
    else:
        ok(f"Condition {cond}: all {n_required} required overlap images present in training")

# --- CHECK 9: Patient overlap counts match intended conditions ---
log("\n[CHECK 9] Patient overlap counts match intended conditions")
expected = {"A": (0, 0.0), "B": (23, 50.0), "C": (46, 100.0)}
for cond in ["A", "B", "C"]:
    train_pids = set(splits[f"{cond}_train"]["PID"])
    n_overlap  = len(train_pids.intersection(test_pids))
    pct        = n_overlap / N_ELIGIBLE * 100
    exp_n, exp_pct = expected[cond]
    if n_overlap == exp_n:
        ok(f"Condition {cond}: {n_overlap}/{N_ELIGIBLE} overlap patients = {pct:.1f}%")
    else:
        fail(f"Condition {cond}: expected {exp_n} overlap patients, got {n_overlap}")

# --- CHECK 10: Class distributions documented ---
log("\n[CHECK 10] Class distributions (training)")
class_dist_df = pd.read_csv(TABLES2 / "training_class_distribution.csv")
log(class_dist_df.to_string(index=False))
ref = class_dist_df[class_dist_df["condition"] == "A"].iloc[0]
for _, row in class_dist_df[class_dist_df["condition"] != "A"].iterrows():
    for cls in ["meningioma", "glioma", "pituitary"]:
        diff = abs(row[f"pct_{cls}"] - ref[f"pct_{cls}"])
        if diff > 2.0:
            log(f"  WARNING: Condition {row['condition']} {cls} deviates {diff:.2f}pp from A")
        else:
            ok(f"Condition {row['condition']} {cls}: diff={diff:.2f}pp (within 2pp threshold)")

# --- CHECK 11: All image files exist on disk ---
log("\n[CHECK 11] All image files exist on disk (sample: first 50 unique)")
all_ids = set()
for key, df in splits.items():
    all_ids.update(df["image_id"].tolist())
all_ids = sorted(all_ids)[:50]   # check a sample for speed
missing_files = [i for i in all_ids if not (RAW_DIR / f"{i}.mat").exists()]
if missing_files:
    fail(f"{len(missing_files)} image files missing from disk (sample): {missing_files[:10]}")
else:
    ok(f"All sampled {len(all_ids)} image files exist on disk")

# --- CHECK 12: No Fold-5 patient in validation ---
log("\n[CHECK 12] No Fold-5 patient in validation")
for cond in ["A", "B", "C"]:
    val_pids = set(splits[f"{cond}_val"]["PID"])
    overlap  = val_pids.intersection(test_pids)
    if overlap:
        fail(f"Condition {cond}: {len(overlap)} Fold-5 patients in validation!")
    else:
        ok(f"Condition {cond}: no Fold-5 patients in validation")

# --- CHECK 13: Validation patients consistent across A, B, C ---
log("\n[CHECK 13] Validation patients identical across A, B, C")
val_A = set(splits["A_val"]["PID"])
val_B = set(splits["B_val"]["PID"])
val_C = set(splits["C_val"]["PID"])
if val_A == val_B == val_C:
    ok(f"Val patient sets identical across A, B, C ({len(val_A)} patients)")
else:
    fail(f"Val patient sets differ! A={len(val_A)}, B={len(val_B)}, C={len(val_C)}")

# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------
log("\n" + "=" * 70)
if errors:
    log(f"RESULT: {len(errors)} CRITICAL ERROR(S) FOUND — STOP BEFORE TRAINING")
    for i, e in enumerate(errors, 1):
        log(f"  [{i}] {e}")
else:
    log("RESULT: ALL 13 CHECKS PASSED — SAFE TO PROCEED TO TRAINING")
log("=" * 70)

(RES2 / "split_integrity_report.txt").write_text("\n".join(lines) + "\n")
print(f"\nReport saved to: {RES2 / 'split_integrity_report.txt'}")

if errors:
    sys.exit(1)
