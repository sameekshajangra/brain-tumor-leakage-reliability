"""
src/07_create_fixed_test.py
===========================
Stage 2 — Step 1: Select the 47 fixed test images.

Rule (deterministic):
  For each Fold-5 patient, select the image with the MINIMUM numeric image_id.
  This image is reserved permanently as the test image for Conditions A, B, C.
  It NEVER enters training or validation.

Outputs
-------
  tables/stage2/fixed_test_images.csv
  results/stage2/fixed_test_selection_report.txt
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.stage1_config import TABLES_DIR, RESULTS_DIR, CLASS_NAMES, TEST_FOLD

TABLES2 = PROJECT_ROOT / "tables" / "stage2"
RES2    = PROJECT_ROOT / "results" / "stage2"
TABLES2.mkdir(parents=True, exist_ok=True)
RES2.mkdir(parents=True, exist_ok=True)

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

def stop(msg):
    log("\nSTOP — " + msg)
    (RES2 / "fixed_test_selection_report.txt").write_text("\n".join(lines))
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load Stage 1 metadata
# ---------------------------------------------------------------------------
meta_csv = PROJECT_ROOT / "tables" / "stage1" / "metadata.csv"
meta = pd.read_csv(meta_csv)
meta["PID"]   = meta["PID"].astype(str)
meta["label"] = meta["label"].astype("Int64")
meta["fold"]  = meta["fold"].astype("Int64")

valid = meta[meta["file_valid"] & meta["PID"].notna() & meta["label"].notna()].copy()

log("=" * 70)
log("FIXED TEST IMAGE SELECTION")
log(f"Test fold: {TEST_FOLD}")
log(f"Rule: minimum image_id per Fold-{TEST_FOLD} patient")
log("=" * 70)

# ---------------------------------------------------------------------------
# Select test images
# ---------------------------------------------------------------------------
fold5 = valid[valid["fold"] == TEST_FOLD].copy()
fold5_pids = sorted(fold5["PID"].unique())

log(f"\nFold-{TEST_FOLD} patients: {len(fold5_pids)}")

test_rows = []
for pid in fold5_pids:
    pid_images = fold5[fold5["PID"] == pid].sort_values("image_id")
    test_img   = pid_images.iloc[0]   # minimum image_id
    total_imgs = len(pid_images)
    test_rows.append({
        "PID"          : pid,
        "test_image_id": int(test_img["image_id"]),
        "filename"     : test_img["filename"],
        "label"        : int(test_img["label"]),
        "class_name"   : test_img["class_name"],
        "fold"         : int(test_img["fold"]),
        "total_patient_images": total_imgs,
        "n_non_test_images"   : total_imgs - 1,
        "eligible_for_overlap": total_imgs >= 2,
    })

test_df = pd.DataFrame(test_rows)

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
n_test   = len(test_df)
n_elig   = int(test_df["eligible_for_overlap"].sum())
n_single = int((~test_df["eligible_for_overlap"]).sum())

log(f"\nFixed test images selected: {n_test}")
log(f"  Eligible overlap patients (>= 2 images): {n_elig}")
log(f"  Single-image patients (not eligible):    {n_single}")

single_pids = test_df[~test_df["eligible_for_overlap"]]["PID"].tolist()
log(f"  Single-image PID(s): {single_pids}")

log("\nClass distribution of test images:")
for label_val, class_name in CLASS_NAMES.items():
    n = int((test_df["label"] == label_val).sum())
    log(f"  Label {label_val} ({class_name}): {n} test images")

log("\nTest image IDs (first 10):")
log(f"  {test_df['test_image_id'].tolist()[:10]}")

# Verify every image is unique
if test_df["test_image_id"].nunique() != n_test:
    stop("Duplicate test image IDs detected — review selection logic.")
log("\n[OK] All test image IDs are unique.")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_csv = TABLES2 / "fixed_test_images.csv"
test_df.to_csv(out_csv, index=False)

report_txt = RES2 / "fixed_test_selection_report.txt"
report_txt.write_text("\n".join(lines) + "\n")

log("\n" + "=" * 70)
log("Script 07 completed successfully.")
log(f"  fixed_test_images.csv           -> {out_csv}")
log(f"  fixed_test_selection_report.txt -> {report_txt}")
log("=" * 70)

report_txt.write_text("\n".join(lines) + "\n")
