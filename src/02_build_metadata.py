"""
src/02_build_metadata.py
========================
Stage 1 — Step 2: Build metadata table and run data-quality checks.

NOTE: All .mat files are MATLAB v7.3 (HDF5) format — loaded with h5py.

For every .mat file in data/raw/:
  - Extract: image_id, filename, PID, label, class_name, height, width,
             fold (via verified cvind mapping), file_valid,
             has_nan, has_inf, image_empty, image_dtype
  - Run data-quality checks per file
  - Flag problematic files without modifying or deleting them

STOP conditions
---------------
- Verified image count != 3,064
- Unique PID count != 233
- Class counts differ from expected
- A PID is associated with more than one tumor class -> stop for manual review
- cvind.mat fold assignment cannot cover all images exactly once

Outputs
-------
  tables/stage1/metadata.csv
  tables/stage1/data_quality_report.csv
  tables/stage1/patient_class_consistency_report.csv
"""

import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.stage1_config import (
    RAW_MAT_DIR, CVIND_PATH, TABLES_DIR,
    EXPECTED_TOTAL_IMAGES, EXPECTED_TOTAL_PATIENTS,
    EXPECTED_CLASS_COUNTS, CLASS_NAMES, VALID_LABELS,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def stop(msg: str):
    print("\n" + "=" * 70)
    print("STOP — ISSUE DETECTED")
    print("=" * 70)
    print(msg)
    print("=" * 70 + "\n")
    sys.exit(1)


def decode_pid(raw_arr: np.ndarray) -> str:
    """Convert MATLAB uint16 char array to a Python string."""
    return "".join(chr(int(c)) for c in raw_arr.flatten() if c != 0).strip()


# ---------------------------------------------------------------------------
# 1. Locate all .mat files (sorted by numeric stem)
# ---------------------------------------------------------------------------
mat_files = sorted(RAW_MAT_DIR.glob("*.mat"), key=lambda p: int(p.stem))

if len(mat_files) != EXPECTED_TOTAL_IMAGES:
    stop(
        f"Expected {EXPECTED_TOTAL_IMAGES} .mat files in {RAW_MAT_DIR}, "
        f"found {len(mat_files)}. Run 01_inspect_dataset.py first."
    )

print(f"[OK] Found {len(mat_files)} .mat files in {RAW_MAT_DIR}")


# ---------------------------------------------------------------------------
# 2. Load cvind.mat (HDF5) and establish fold mapping
#    The mapping is verified, not merely assumed from sorted filenames.
# ---------------------------------------------------------------------------
print("\nLoading cvind.mat (HDF5) ...")

try:
    with h5py.File(str(CVIND_PATH), "r") as f:
        if "cvind" not in f:
            stop(
                f"'cvind' dataset not found in cvind.mat. "
                f"Available keys: {list(f.keys())}"
            )
        cvind_arr = f["cvind"][()].flatten().astype(int)
except Exception as e:
    stop(f"Could not open cvind.mat with h5py: {e}")

print(f"  cvind shape: {cvind_arr.shape}  dtype: {cvind_arr.dtype}")
print(f"  Total entries: {len(cvind_arr)}")

if len(cvind_arr) != EXPECTED_TOTAL_IMAGES:
    stop(
        f"cvind has {len(cvind_arr)} entries, expected {EXPECTED_TOTAL_IMAGES}."
    )

# Verify all .mat files have numeric stems 1..3064
stems = {int(p.stem) for p in mat_files}
expected_stems = set(range(1, EXPECTED_TOTAL_IMAGES + 1))
if stems != expected_stems:
    missing = sorted(expected_stems - stems)[:20]
    extra   = sorted(stems - expected_stems)[:20]
    stop(
        f"Cannot use index-based cvind mapping because .mat stems don't "
        f"cover 1..{EXPECTED_TOTAL_IMAGES} exactly.\n"
        f"Missing (up to 20): {missing}\nExtra (up to 20): {extra}"
    )

# Build fold_by_image_number: image_number (1-based) -> fold (1-based)
fold_by_image_number = {i + 1: int(cvind_arr[i]) for i in range(len(cvind_arr))}

# Verify fold values
fold_values_seen = set(fold_by_image_number.values())
print(f"  Fold values seen: {sorted(fold_values_seen)}")
if fold_values_seen != {1, 2, 3, 4, 5}:
    stop(f"Expected fold values {{1,2,3,4,5}}, got {sorted(fold_values_seen)}.")

print("[OK] Fold mapping established: entry i in cvind -> image (i+1).mat")
print("     This mapping will be independently re-confirmed in 03_verify_folds.py.")


# ---------------------------------------------------------------------------
# 3. Load every .mat file and build metadata + data-quality records
# ---------------------------------------------------------------------------
print("\nLoading all .mat files (HDF5) ...")

REQUIRED_FIELDS = {"image", "label", "PID", "tumorBorder", "tumorMask"}
metadata_rows = []
quality_rows  = []

for mat_path in tqdm(mat_files, desc="  Processing"):
    image_number = int(mat_path.stem)
    fold         = fold_by_image_number[image_number]

    row = {
        "image_id"    : image_number,
        "filename"    : mat_path.name,
        "PID"         : None,
        "label"       : None,
        "class_name"  : None,
        "height"      : None,
        "width"       : None,
        "fold"        : fold,
        "file_valid"  : True,
        "has_nan"     : False,
        "has_inf"     : False,
        "image_empty" : False,
        "image_dtype" : None,
    }

    qrow = {
        "image_id" : image_number,
        "filename" : mat_path.name,
        "issue"    : "none",
        "detail"   : "",
    }

    issues = []

    def flag(issue_str, detail_str=""):
        issues.append(issue_str)
        qrow["detail"] = (qrow["detail"] + " | " + detail_str).strip(" |")

    # ---- Try opening ----
    try:
        f = h5py.File(str(mat_path), "r")
    except Exception as e:
        issues.append("unreadable_file")
        qrow["detail"] = str(e)
        row["file_valid"] = False
        qrow["issue"] = "; ".join(issues)
        metadata_rows.append(row)
        quality_rows.append(qrow)
        continue

    with f:
        if "cjdata" not in f:
            issues.append("missing_cjdata")
            qrow["detail"] = f"Keys found: {list(f.keys())}"
            row["file_valid"] = False
            qrow["issue"] = "; ".join(issues)
            metadata_rows.append(row)
            quality_rows.append(qrow)
            continue

        cjdata = f["cjdata"]
        field_names = set(cjdata.keys())

        # ---- Extract label ----
        label = None
        if "label" not in field_names:
            flag("missing_label")
        else:
            try:
                label_val = int(float(cjdata["label"][()].flat[0]))
                if label_val not in VALID_LABELS:
                    flag(f"invalid_label:{label_val}")
                    label_val = None
                label = label_val
            except Exception as e:
                flag("label_read_error", str(e))

        # ---- Extract PID ----
        pid = None
        if "PID" not in field_names:
            flag("missing_PID")
        else:
            try:
                pid_raw = cjdata["PID"][()]
                pid = decode_pid(pid_raw)
                if pid == "":
                    flag("empty_PID")
                    pid = None
            except Exception as e:
                flag("PID_read_error", str(e))

        # ---- Extract image ----
        img_arr = None
        if "image" not in field_names:
            flag("missing_image")
        else:
            try:
                img_arr = cjdata["image"][()]
                if not isinstance(img_arr, np.ndarray) or img_arr.ndim < 2:
                    flag(f"invalid_image_shape:{getattr(img_arr,'shape','?')}")
                    img_arr = None
            except Exception as e:
                flag("image_read_error", str(e))

        if img_arr is not None:
            row["height"]      = img_arr.shape[0]
            row["width"]       = img_arr.shape[1]
            row["image_dtype"] = str(img_arr.dtype)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arr_float = img_arr.astype(np.float64)
                    if np.any(np.isnan(arr_float)):
                        row["has_nan"] = True
                        flag("has_nan")
                    if np.any(np.isinf(arr_float)):
                        row["has_inf"] = True
                        flag("has_inf")
                    if img_arr.size == 0 or np.all(arr_float == 0):
                        row["image_empty"] = True
                        flag("image_empty")
            except Exception as e:
                flag("quality_check_error", str(e))

    row["PID"]        = pid
    row["label"]      = label
    row["class_name"] = CLASS_NAMES.get(label, None) if label is not None else None

    if issues:
        row["file_valid"] = False
    qrow["issue"] = "; ".join(issues) if issues else "none"

    metadata_rows.append(row)
    quality_rows.append(qrow)


# ---------------------------------------------------------------------------
# 4. Build DataFrames
# ---------------------------------------------------------------------------
meta_df    = pd.DataFrame(metadata_rows)
quality_df = pd.DataFrame(quality_rows)

print(f"\n  Total records: {len(meta_df)}")


# ---------------------------------------------------------------------------
# 5. Automated verification: counts
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("AUTOMATED VERIFICATION")
print("=" * 70)

actual_total = len(meta_df)
print(f"  Total images  : {actual_total}  (expected {EXPECTED_TOTAL_IMAGES})")
if actual_total != EXPECTED_TOTAL_IMAGES:
    stop(f"Image count mismatch: found {actual_total}, expected {EXPECTED_TOTAL_IMAGES}.")
print("  [OK] Image count matches.")

valid_meta = meta_df[meta_df["file_valid"] & meta_df["PID"].notna()].copy()
actual_pids = valid_meta["PID"].nunique()
print(f"  Unique PIDs   : {actual_pids}  (expected {EXPECTED_TOTAL_PATIENTS})")
if actual_pids != EXPECTED_TOTAL_PATIENTS:
    stop(
        f"Unique patient count mismatch: found {actual_pids}, "
        f"expected {EXPECTED_TOTAL_PATIENTS}."
    )
print("  [OK] Patient count matches.")

print("  Class counts:")
for label_val, expected_count in sorted(EXPECTED_CLASS_COUNTS.items()):
    actual_count = int((valid_meta["label"] == label_val).sum())
    class_name   = CLASS_NAMES[label_val]
    status = "[OK]" if actual_count == expected_count else "[MISMATCH]"
    print(f"    {status} Label {label_val} ({class_name}): "
          f"{actual_count}  (expected {expected_count})")
    if actual_count != expected_count:
        stop(
            f"Class count mismatch for label {label_val} ({class_name}): "
            f"found {actual_count}, expected {expected_count}."
        )
print("  [OK] All class counts match.")


# ---------------------------------------------------------------------------
# 6. Patient-class consistency check
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PATIENT-CLASS CONSISTENCY CHECK")
print("=" * 70)

pid_classes = valid_meta.groupby("PID")["label"].apply(set)
inconsistent = pid_classes[pid_classes.apply(len) > 1]

consistency_rows = []
for pid, classes in pid_classes.items():
    affected_imgs = valid_meta[valid_meta["PID"] == pid]["image_id"].tolist()
    consistency_rows.append(
        {
            "PID"                : pid,
            "num_classes"        : len(classes),
            "classes"            : str(sorted(classes)),
            "is_consistent"      : len(classes) == 1,
            "num_images"         : len(affected_imgs),
            "affected_image_ids" : str(affected_imgs[:10]),
        }
    )

consistency_df = pd.DataFrame(consistency_rows).sort_values("PID")
consistency_df.to_csv(TABLES_DIR / "patient_class_consistency_report.csv", index=False)

if len(inconsistent) > 0:
    print(f"\n  WARNING: {len(inconsistent)} PID(s) with multiple tumor classes!")
    for pid, classes in inconsistent.items():
        imgs = valid_meta[valid_meta["PID"] == pid]["image_id"].tolist()
        print(f"    PID={pid}  classes={classes}  images={imgs}")
    stop(
        f"{len(inconsistent)} PID(s) have images from multiple tumor classes. "
        "Manual review required. See patient_class_consistency_report.csv."
    )
else:
    print("  [OK] Every PID is associated with exactly one tumor class.")


# ---------------------------------------------------------------------------
# 7. Save outputs
# ---------------------------------------------------------------------------
meta_csv    = TABLES_DIR / "metadata.csv"
quality_csv = TABLES_DIR / "data_quality_report.csv"

meta_df.to_csv(meta_csv, index=False)
quality_df.to_csv(quality_csv, index=False)

n_problematic = (quality_df["issue"] != "none").sum()
print(f"\n  Problematic files: {n_problematic}")
if n_problematic > 0:
    print("  Details:")
    print(quality_df[quality_df["issue"] != "none"].to_string(index=False))

print("\n" + "=" * 70)
print("Script 02 completed successfully.")
print(f"  metadata.csv                         -> {meta_csv}")
print(f"  data_quality_report.csv              -> {quality_csv}")
print(f"  patient_class_consistency_report.csv -> {TABLES_DIR / 'patient_class_consistency_report.csv'}")
print("=" * 70)
