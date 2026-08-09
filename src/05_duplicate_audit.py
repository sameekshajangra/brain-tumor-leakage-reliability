"""
src/05_duplicate_audit.py
=========================
Stage 1 — Step 5: Exact pixel-duplicate audit.

For every .mat image, computes a SHA-256 hash of the raw pixel-array bytes
(as a C-contiguous uint8 array after min-max normalization to 0-255 — using
the integer representation preserves exact pixel equality).

This is an EXACT-duplicate audit only.
Near-duplicate detection is out of scope for Stage 1.

DO NOT remove or modify any files.

Outputs
-------
  tables/stage1/exact_duplicate_audit.csv
  results/stage1/exact_duplicate_audit_report.txt
"""

import sys
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.stage1_config import (
    RAW_MAT_DIR, TABLES_DIR, RESULTS_DIR,
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
    (RESULTS_DIR / "exact_duplicate_audit_report.txt").write_text("\n".join(report_lines))
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Load metadata (for PID and label lookup)
# ---------------------------------------------------------------------------
meta_csv = TABLES_DIR / "metadata.csv"
if not meta_csv.exists():
    stop(f"metadata.csv not found. Run 02_build_metadata.py first.")

meta = pd.read_csv(meta_csv)
meta_lookup = meta.set_index("image_id")[["PID", "label", "class_name", "fold"]]

log("=" * 70)
log("EXACT PIXEL DUPLICATE AUDIT")
log("=" * 70)


# ---------------------------------------------------------------------------
# 2. Compute SHA-256 hash for every image
# ---------------------------------------------------------------------------
mat_files = sorted(RAW_MAT_DIR.glob("*.mat"), key=lambda p: int(p.stem))

if len(mat_files) != EXPECTED_TOTAL_IMAGES:
    stop(
        f"Expected {EXPECTED_TOTAL_IMAGES} .mat files, found {len(mat_files)}. "
        "Run 01_inspect_dataset.py first."
    )

log(f"\nComputing SHA-256 hashes for {len(mat_files)} images ...")

audit_rows = []
hash_to_images = defaultdict(list)   # hash -> list of image_ids

for mat_path in tqdm(mat_files, desc="  Hashing"):
    image_id = int(mat_path.stem)
    pid       = meta_lookup.loc[image_id, "PID"]       if image_id in meta_lookup.index else None
    label     = meta_lookup.loc[image_id, "label"]     if image_id in meta_lookup.index else None
    class_name= meta_lookup.loc[image_id, "class_name"]if image_id in meta_lookup.index else None
    fold      = meta_lookup.loc[image_id, "fold"]      if image_id in meta_lookup.index else None

    img_hash = None
    hash_error = None

    try:
        with h5py.File(str(mat_path), "r") as hf:
            img = hf["cjdata"]["image"][()]

            # Deterministic representation:
            # Hash the raw float64 bytes of the original pixel array.
            arr = np.ascontiguousarray(img.astype(np.float64))
            raw_bytes = arr.tobytes()
            img_hash  = hashlib.sha256(raw_bytes).hexdigest()

    except Exception as e:
        hash_error = str(e)

    audit_rows.append(
        {
            "image_id"  : image_id,
            "filename"  : mat_path.name,
            "PID"       : pid,
            "label"     : label,
            "class_name": class_name,
            "fold"      : fold,
            "sha256"    : img_hash,
            "hash_error": hash_error,
        }
    )

    if img_hash is not None:
        hash_to_images[img_hash].append(image_id)


# ---------------------------------------------------------------------------
# 3. Identify duplicate groups
# ---------------------------------------------------------------------------
log("\nAnalyzing duplicates ...")

duplicate_groups = {h: ids for h, ids in hash_to_images.items() if len(ids) > 1}
n_dup_groups     = len(duplicate_groups)
n_dup_images     = sum(len(ids) for ids in duplicate_groups.values())

log(f"  Total images hashed          : {len(audit_rows)}")
log(f"  Unique hashes                : {len(hash_to_images)}")
log(f"  Duplicate groups (exact)     : {n_dup_groups}")
log(f"  Images in duplicate groups   : {n_dup_images}")

# Add group membership to audit rows
hash_to_group = {}
for group_idx, (h, ids) in enumerate(duplicate_groups.items(), start=1):
    for img_id in ids:
        hash_to_group[img_id] = group_idx

audit_df = pd.DataFrame(audit_rows)
audit_df["duplicate_group"] = audit_df["image_id"].map(hash_to_group).fillna(-1).astype(int)
audit_df["is_duplicate"]    = audit_df["duplicate_group"] != -1


# ---------------------------------------------------------------------------
# 4. Analyse duplicates: intra-patient vs inter-patient
# ---------------------------------------------------------------------------
log("\n--- Duplicate Group Details ---")

intra_patient_groups = 0
inter_patient_groups = 0

for group_idx, (h, ids) in enumerate(duplicate_groups.items(), start=1):
    group_rows = audit_df[audit_df["image_id"].isin(ids)]
    pids       = group_rows["PID"].dropna().unique()
    folds      = group_rows["fold"].dropna().unique()

    is_intra = len(pids) == 1
    if is_intra:
        intra_patient_groups += 1
    else:
        inter_patient_groups += 1

    log(f"\n  Group {group_idx}: {len(ids)} images")
    log(f"    image_ids : {ids}")
    log(f"    PIDs      : {pids.tolist()}")
    log(f"    folds     : {sorted(folds.astype(int).tolist())}")
    log(f"    type      : {'intra-patient' if is_intra else 'INTER-PATIENT'}")
    if not is_intra:
        log(f"    WARNING: Exact duplicate images across different patients!")

log(f"\n  Intra-patient duplicate groups : {intra_patient_groups}")
log(f"  Inter-patient duplicate groups : {inter_patient_groups}")

if n_dup_groups == 0:
    log("\n[OK] No exact pixel duplicates found.")


# ---------------------------------------------------------------------------
# 5. Save outputs
# ---------------------------------------------------------------------------
audit_csv  = TABLES_DIR / "exact_duplicate_audit.csv"
report_txt = RESULTS_DIR / "exact_duplicate_audit_report.txt"

audit_df.to_csv(audit_csv, index=False)
report_txt.write_text("\n".join(report_lines) + "\n")

log("\n" + "=" * 70)
log("Script 05 completed successfully.")
log(f"  exact_duplicate_audit.csv       -> {audit_csv}")
log(f"  exact_duplicate_audit_report.txt-> {report_txt}")
log("=" * 70)

# Re-write with final lines
report_txt.write_text("\n".join(report_lines) + "\n")
