"""
src/03_verify_folds.py
======================
Stage 1 — Step 3: Verify the five cross-validation folds.

NOTE: cvind.mat is MATLAB v7.3 (HDF5) — loaded with h5py.

Verifies
--------
1. Exactly 5 folds.
2. Every image has exactly one fold assignment (no image missing, none duplicated).
3. Image count and patient count per fold.
4. Class distribution per fold.
5. Patient-disjointness across all 10 fold pairs.
   -> If ANY PID appears in more than one fold: STOP and report.
6. Images sorted by numeric stem correspond to the cvind index ordering —
   the mapping is independently re-confirmed here.

Outputs
-------
  tables/stage1/fold_summary.csv
  tables/stage1/fold_patient_overlap_report.csv
  results/stage1/dataset_verification_report.txt
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.stage1_config import (
    CVIND_PATH, TABLES_DIR, RESULTS_DIR,
    EXPECTED_TOTAL_IMAGES, EXPECTED_NUM_FOLDS,
    CLASS_NAMES, EXPECTED_CLASS_COUNTS,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)

def stop(msg: str):
    log("\n" + "=" * 70)
    log("STOP — ISSUE DETECTED")
    log("=" * 70)
    log(msg)
    log("=" * 70)
    # Save partial report before stopping
    (RESULTS_DIR / "dataset_verification_report.txt").write_text("\n".join(report_lines))
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Load metadata (built by script 02)
# ---------------------------------------------------------------------------
meta_csv = TABLES_DIR / "metadata.csv"
if not meta_csv.exists():
    stop(f"metadata.csv not found at {meta_csv}. Run 02_build_metadata.py first.")

meta = pd.read_csv(meta_csv)
meta["PID"]   = meta["PID"].astype(str)
meta["label"] = meta["label"].astype("Int64")
meta["fold"]  = meta["fold"].astype("Int64")

log("=" * 70)
log("DATASET FOLD VERIFICATION REPORT")
log("=" * 70)
log(f"\nTotal records loaded: {len(meta)}")
log(f"Columns: {list(meta.columns)}")


# ---------------------------------------------------------------------------
# 2. Verify fold coverage: every image has exactly one fold
# ---------------------------------------------------------------------------
log("\n--- Fold Coverage ---")

missing_fold = meta[meta["fold"].isna()]
if len(missing_fold) > 0:
    stop(
        f"{len(missing_fold)} images have no fold assignment.\n"
        f"image_ids: {missing_fold['image_id'].tolist()[:20]}"
    )
log(f"[OK] All {len(meta)} images have a fold assignment.")

# Verify fold values are in {1..5}
fold_vals = set(meta["fold"].dropna().astype(int).unique())
log(f"Fold values present: {sorted(fold_vals)}")
if fold_vals != set(range(1, EXPECTED_NUM_FOLDS + 1)):
    stop(
        f"Expected fold values {{1,2,3,4,5}}, got {sorted(fold_vals)}."
    )
log(f"[OK] Exactly {EXPECTED_NUM_FOLDS} folds present.")

# Verify no image assigned to multiple folds
# (since fold is a single column, a duplicate image_id in metadata would
# indicate a problem in script 02 — verify image_id is unique)
dup_ids = meta[meta.duplicated("image_id", keep=False)]
if len(dup_ids) > 0:
    stop(
        f"{len(dup_ids)} rows share an image_id (metadata has duplicates).\n"
        f"image_ids: {dup_ids['image_id'].unique()[:20].tolist()}"
    )
log("[OK] Every image_id appears exactly once.")


# ---------------------------------------------------------------------------
# 3. Re-confirm cvind mapping independently (using h5py for HDF5 format)
# ---------------------------------------------------------------------------
log("\n--- Re-confirming cvind mapping ---")

try:
    with h5py.File(str(CVIND_PATH), "r") as hf:
        if "cvind" not in hf:
            stop(f"'cvind' not found in cvind.mat. Keys: {list(hf.keys())}")
        cvind_flat = hf["cvind"][()].flatten().astype(int)
except Exception as e:
    stop(f"Could not open cvind.mat with h5py: {e}")

if len(cvind_flat) != EXPECTED_TOTAL_IMAGES:
    stop(f"cvind has {len(cvind_flat)} entries, expected {EXPECTED_TOTAL_IMAGES}.")

# For every image, check metadata fold vs. cvind entry
mismatches = 0
for _, mrow in meta.iterrows():
    img_num       = int(mrow["image_id"])
    expected_fold = int(cvind_flat[img_num - 1])   # 0-indexed cvind, 1-based image_id
    actual_fold   = int(mrow["fold"])
    if expected_fold != actual_fold:
        mismatches += 1

if mismatches > 0:
    stop(f"{mismatches} images have fold assignments inconsistent with cvind.mat.")
log("[OK] All fold assignments independently re-confirmed against cvind.mat.")


# ---------------------------------------------------------------------------
# 4. Per-fold statistics
# ---------------------------------------------------------------------------
log("\n--- Per-Fold Statistics ---")
log(f"\n{'Fold':>6} {'Images':>8} {'Patients':>10}  "
    f"{'Meningioma':>12} {'Glioma':>8} {'Pituitary':>11}")
log("-" * 65)

fold_summary_rows = []
for fold_id in sorted(fold_vals):
    fold_data = meta[meta["fold"] == fold_id]
    n_images  = len(fold_data)
    n_pats    = fold_data["PID"].nunique()
    n_mening  = int((fold_data["label"] == 1).sum())
    n_glioma  = int((fold_data["label"] == 2).sum())
    n_pituit  = int((fold_data["label"] == 3).sum())
    log(f"  {fold_id:>4} {n_images:>8} {n_pats:>10}  "
        f"{n_mening:>12} {n_glioma:>8} {n_pituit:>11}")
    fold_summary_rows.append(
        {
            "fold"          : fold_id,
            "n_images"      : n_images,
            "n_patients"    : n_pats,
            "n_meningioma"  : n_mening,
            "n_glioma"      : n_glioma,
            "n_pituitary"   : n_pituit,
        }
    )

fold_summary_df = pd.DataFrame(fold_summary_rows)


# ---------------------------------------------------------------------------
# 5. Patient-disjointness check — all 10 fold pairs
# ---------------------------------------------------------------------------
log("\n--- Patient-Disjointness Check (all 10 fold pairs) ---\n")

overlap_rows = []
any_overlap_found = False

fold_pids = {
    fold_id: set(meta[meta["fold"] == fold_id]["PID"].unique())
    for fold_id in sorted(fold_vals)
}

for fold_a, fold_b in combinations(sorted(fold_vals), 2):
    pids_a = fold_pids[fold_a]
    pids_b = fold_pids[fold_b]
    intersection = pids_a & pids_b
    n_overlap = len(intersection)
    status = "[OVERLAP DETECTED]" if n_overlap > 0 else "[OK]"
    log(f"  Fold {fold_a} vs Fold {fold_b}: {n_overlap} overlapping PID(s)  {status}")
    if n_overlap > 0:
        log(f"    Overlapping PIDs: {sorted(intersection)[:20]}")
        any_overlap_found = True
    overlap_rows.append(
        {
            "fold_a"          : fold_a,
            "fold_b"          : fold_b,
            "n_pids_fold_a"   : len(pids_a),
            "n_pids_fold_b"   : len(pids_b),
            "n_overlapping_pids": n_overlap,
            "overlapping_pids": str(sorted(intersection)) if intersection else "[]",
            "patient_disjoint": n_overlap == 0,
        }
    )

overlap_df = pd.DataFrame(overlap_rows)

if any_overlap_found:
    overlap_df.to_csv(TABLES_DIR / "fold_patient_overlap_report.csv", index=False)
    stop(
        "Patient overlap detected across official folds (see fold_patient_overlap_report.csv). "
        "The official folds are expected to be patient-disjoint. "
        "Manual review required before proceeding."
    )

log("\n[OK] All official folds are patient-disjoint.")


# ---------------------------------------------------------------------------
# 6. Global count summary
# ---------------------------------------------------------------------------
log("\n--- Global Dataset Summary ---")
log(f"  Total images   : {len(meta)}")
log(f"  Total patients : {meta['PID'].nunique()}")

for label_val, class_name in CLASS_NAMES.items():
    n = int((meta["label"] == label_val).sum())
    log(f"  {class_name:15s}: {n} images")


# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
fold_csv    = TABLES_DIR / "fold_summary.csv"
overlap_csv = TABLES_DIR / "fold_patient_overlap_report.csv"
report_txt  = RESULTS_DIR / "dataset_verification_report.txt"

fold_summary_df.to_csv(fold_csv, index=False)
overlap_df.to_csv(overlap_csv, index=False)
report_txt.write_text("\n".join(report_lines) + "\n")

log("\n" + "=" * 70)
log("Script 03 completed successfully.")
log(f"  fold_summary.csv                 -> {fold_csv}")
log(f"  fold_patient_overlap_report.csv  -> {overlap_csv}")
log(f"  dataset_verification_report.txt  -> {report_txt}")
log("=" * 70)

# Re-write report with final lines
report_txt.write_text("\n".join(report_lines) + "\n")
