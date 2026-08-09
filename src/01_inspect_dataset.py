"""
src/01_inspect_dataset.py
=========================
Stage 1 — Step 1: Inspect the raw dataset.

NOTE: All .mat files in this dataset are MATLAB v7.3 (HDF5) format.
      scipy.io.loadmat cannot read them; h5py is used instead.

Actions
-------
1. List the contents of every ZIP file without extracting.
2. Verify that the four ZIPs collectively contain exactly 3,064 unique .mat
   files, with no duplicate filenames across ZIPs.
3. Extract the ZIPs into data/raw/ (skips files that already exist).
4. Load cvind.mat with h5py and print all variables, shapes, dtypes, and
   representative values so that the fold-to-image mapping can be understood.
5. Load 5 random .mat files with h5py and print all cjdata fields.
6. Save environment.txt with Python / library versions and GPU info.
7. Save tables/stage1/zip_contents_report.csv.

STOP conditions
---------------
- A ZIP file is missing.
- The four ZIPs do not contain exactly 3,064 unique .mat filenames.
- Duplicate .mat filenames exist across ZIPs.
- cvind.mat is missing or cannot be loaded.
- Any random sample .mat file cannot be loaded or is missing expected fields.
"""

import os
import sys
import zipfile
import random
import platform
import subprocess
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import h5py

# ---------------------------------------------------------------------------
# Allow running from project root or from src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from configs.stage1_config import (
    DATA_DIR, RAW_MAT_DIR, CVIND_PATH, TABLES_DIR, RESULTS_DIR,
    ZIP_FILENAMES, EXPECTED_TOTAL_IMAGES, SEED, CLASS_NAMES,
)

random.seed(SEED)
np.random.seed(SEED)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def stop(msg: str):
    """Print a STOP message and exit with error code 1."""
    print("\n" + "=" * 70)
    print("STOP — ISSUE DETECTED")
    print("=" * 70)
    print(msg)
    print("=" * 70 + "\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. List and verify ZIP contents
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1 — Verifying ZIP file contents")
print("=" * 70)

zip_records = []          # rows for zip_contents_report.csv
all_mat_names = []        # all .mat filenames across all ZIPs (may include dups)
mat_to_zip = {}           # filename -> list of ZIPs containing it

for zip_fname in ZIP_FILENAMES:
    zip_path = DATA_DIR / zip_fname
    if not zip_path.exists():
        stop(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        mat_members = sorted(
            m for m in members
            if m.endswith(".mat") and not m.startswith("__MACOSX")
        )
        first_mat = mat_members[0] if mat_members else "N/A"
        last_mat  = mat_members[-1] if mat_members else "N/A"

        print(f"\n  {zip_fname}")
        print(f"    Total entries : {len(members)}")
        print(f"    .mat files    : {len(mat_members)}")
        print(f"    First .mat    : {first_mat}")
        print(f"    Last .mat     : {last_mat}")

        for m in mat_members:
            all_mat_names.append(m)
            mat_to_zip.setdefault(m, []).append(zip_fname)

        zip_records.append(
            {
                "zip_filename"             : zip_fname,
                "total_entries"            : len(members),
                "mat_count"                : len(mat_members),
                "first_mat"                : first_mat,
                "last_mat"                 : last_mat,
                "duplicate_filenames_found": False,
            }
        )

# Check for duplicate filenames across ZIPs
duplicates = {name: zips for name, zips in mat_to_zip.items() if len(zips) > 1}
if duplicates:
    msg = "Duplicate .mat filenames found across ZIP files:\n"
    for name, zips in duplicates.items():
        msg += f"  {name}  ->  {zips}\n"
    stop(msg)

total_unique_mat = len(mat_to_zip)
print(f"\n  Total unique .mat files across all ZIPs: {total_unique_mat}")

if total_unique_mat != EXPECTED_TOTAL_IMAGES:
    stop(
        f"Expected {EXPECTED_TOTAL_IMAGES} unique .mat files across all ZIPs, "
        f"but found {total_unique_mat}."
    )
print(f"  [OK] Exactly {EXPECTED_TOTAL_IMAGES} unique .mat files confirmed.")

# Check whether all .mat filenames have numeric stems
non_numeric = [n for n in mat_to_zip if not Path(n).stem.isdigit()]
if non_numeric:
    print(f"\n  WARNING: Non-numeric .mat stems found (first 10): {non_numeric[:10]}")
else:
    print("  [OK] All .mat filenames have numeric stems.")

# Save zip_contents_report.csv
zip_df = pd.DataFrame(zip_records)
zip_csv = TABLES_DIR / "zip_contents_report.csv"
zip_df.to_csv(zip_csv, index=False)
print(f"\n  Saved: {zip_csv}")


# ---------------------------------------------------------------------------
# 2. Extract ZIPs into data/raw/
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2 — Extracting ZIPs into data/raw/")
print("=" * 70)

RAW_MAT_DIR.mkdir(parents=True, exist_ok=True)

for zip_fname in ZIP_FILENAMES:
    zip_path = DATA_DIR / zip_fname
    print(f"\n  Extracting {zip_fname} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        mat_members = [
            m for m in members
            if m.endswith(".mat") and not m.startswith("__MACOSX")
        ]
        extracted = 0
        skipped = 0
        for member in mat_members:
            member_path = Path(member)
            dest = RAW_MAT_DIR / member_path.name
            if dest.exists():
                skipped += 1
            else:
                source = zf.read(member)
                dest.write_bytes(source)
                extracted += 1
        print(f"    Extracted: {extracted}, Skipped (already exist): {skipped}")

# Verify post-extraction count
extracted_mats = sorted(RAW_MAT_DIR.glob("*.mat"))
print(f"\n  .mat files now in data/raw/: {len(extracted_mats)}")
if len(extracted_mats) != EXPECTED_TOTAL_IMAGES:
    stop(
        f"After extraction, expected {EXPECTED_TOTAL_IMAGES} .mat files in "
        f"{RAW_MAT_DIR}, but found {len(extracted_mats)}."
    )
print(f"  [OK] {EXPECTED_TOTAL_IMAGES} .mat files confirmed in data/raw/.")

# Check that all numeric stems from 1..3064 are present
stems = {Path(f).stem for f in extracted_mats}
expected_stems = {str(i) for i in range(1, EXPECTED_TOTAL_IMAGES + 1)}
missing_stems = expected_stems - stems
extra_stems   = stems - expected_stems
if missing_stems:
    stop(
        f"Missing .mat file numbers: {sorted(missing_stems, key=int)[:20]} "
        f"(and {max(0, len(missing_stems)-20)} more)"
    )
if extra_stems:
    print(f"  WARNING: Unexpected .mat file names: {list(extra_stems)[:20]}")
else:
    print("  [OK] Files 1.mat through 3064.mat all present.")


# ---------------------------------------------------------------------------
# 3. Inspect cvind.mat (MATLAB v7.3 / HDF5)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3 — Inspecting cvind.mat (HDF5/MATLAB v7.3)")
print("=" * 70)

if not CVIND_PATH.exists():
    stop(f"cvind.mat not found at: {CVIND_PATH}")

try:
    with h5py.File(str(CVIND_PATH), "r") as f:
        print(f"\n  Top-level keys: {list(f.keys())}")
        for key in f.keys():
            ds = f[key]
            print(f"\n  Variable '{key}':")
            print(f"    type  : {type(ds).__name__}")
            if isinstance(ds, h5py.Dataset):
                print(f"    shape : {ds.shape}")
                print(f"    dtype : {ds.dtype}")
                arr = ds[()]
                flat = arr.flatten()
                unique_vals = np.unique(flat)
                print(f"    unique values ({len(unique_vals)}): {unique_vals[:30].tolist()}")
                print(f"    first 10 : {flat[:10].tolist()}")
                print(f"    last  10 : {flat[-10:].tolist()}")
            elif isinstance(ds, h5py.Group):
                print(f"    (Group) sub-keys: {list(ds.keys())}")
except Exception as e:
    stop(f"Could not open cvind.mat with h5py: {e}")

# Load and summarise fold distribution
with h5py.File(str(CVIND_PATH), "r") as f:
    if "cvind" not in f:
        avail = list(f.keys())
        stop(
            f"Expected variable 'cvind' in cvind.mat but found: {avail}. "
            "Cannot establish fold mapping. Manual inspection required."
        )
    cvind_arr = f["cvind"][()].flatten().astype(int)

print(f"\n  cvind shape after flatten : {cvind_arr.shape}")
print(f"  Total entries            : {len(cvind_arr)}")
fold_counts = Counter(cvind_arr.tolist())
for fold_id in sorted(fold_counts):
    print(f"    Fold {fold_id}: {fold_counts[fold_id]} images")

if len(cvind_arr) != EXPECTED_TOTAL_IMAGES:
    stop(
        f"cvind has {len(cvind_arr)} entries but {EXPECTED_TOTAL_IMAGES} images "
        "are expected."
    )
print(f"\n  [OK] cvind contains {len(cvind_arr)} fold labels for {EXPECTED_TOTAL_IMAGES} images.")
print("  Mapping: entry i (0-indexed) -> image (i+1).mat")
print("  This mapping will be independently verified in 02_build_metadata.py and 03_verify_folds.py.")


# ---------------------------------------------------------------------------
# 4. Inspect 5 random .mat files (HDF5)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4 — Inspecting 5 random .mat files (h5py)")
print("=" * 70)

extracted_mat_list = sorted(
    [p for p in RAW_MAT_DIR.glob("*.mat")],
    key=lambda p: int(p.stem)
)
sample_files = random.sample(extracted_mat_list, min(5, len(extracted_mat_list)))
REQUIRED_FIELDS = {"image", "label", "PID", "tumorBorder", "tumorMask"}


def decode_pid(raw_arr: np.ndarray) -> str:
    """Convert uint16 character array (MATLAB string) to Python string."""
    return "".join(chr(c) for c in raw_arr.flatten() if c != 0)


for mat_path in sample_files:
    print(f"\n  File: {mat_path.name}")
    try:
        with h5py.File(str(mat_path), "r") as f:
            if "cjdata" not in f:
                print(f"    ERROR: 'cjdata' group not found. Keys: {list(f.keys())}")
                continue

            cjdata = f["cjdata"]
            field_names = list(cjdata.keys())
            print(f"    cjdata fields: {field_names}")

            missing = REQUIRED_FIELDS - set(field_names)
            if missing:
                print(f"    WARNING: Missing fields: {missing}")

            for field in field_names:
                ds = cjdata[field]
                if isinstance(ds, h5py.Dataset):
                    arr = ds[()]
                    print(f"      {field:15s}: shape={arr.shape}  dtype={arr.dtype}", end="")
                    if field == "PID":
                        pid_str = decode_pid(arr)
                        print(f"  decoded='{pid_str}'", end="")
                    elif field == "label":
                        print(f"  value={arr.flat[0]}", end="")
                    elif field == "image":
                        print(f"  min={arr.min():.1f}  max={arr.max():.1f}", end="")
                    print()
                elif isinstance(ds, h5py.Group):
                    print(f"      {field:15s}: (Group) sub-keys={list(ds.keys())}")
    except Exception as e:
        print(f"    ERROR loading file: {e}")


# ---------------------------------------------------------------------------
# 5. Save environment.txt
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 — Saving environment.txt")
print("=" * 70)

env_lines = []
env_lines.append("=== Environment ===")
env_lines.append(f"Python        : {sys.version}")
env_lines.append(f"Platform      : {platform.platform()}")
env_lines.append(f"Machine       : {platform.machine()}")


def pkg_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "not installed"


for pkg in ["numpy", "scipy", "pandas", "matplotlib", "seaborn",
            "scikit-learn", "Pillow", "tqdm", "h5py", "torch", "torchvision"]:
    env_lines.append(f"{pkg:15s}: {pkg_version(pkg)}")

# GPU info via nvidia-smi
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5,
    )
    gpu_info = result.stdout.strip() or "No NVIDIA GPU detected"
except Exception:
    gpu_info = "nvidia-smi not available"
env_lines.append(f"GPU           : {gpu_info}")

# PyTorch CUDA
try:
    import torch
    env_lines.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        env_lines.append(f"CUDA devices  : {torch.cuda.device_count()}")
        env_lines.append(f"CUDA device 0 : {torch.cuda.get_device_name(0)}")
except ImportError:
    env_lines.append("PyTorch       : not installed")

env_path = RESULTS_DIR / "environment.txt"
env_path.write_text("\n".join(env_lines) + "\n")
print(f"  Saved: {env_path}")
for line in env_lines:
    print(f"  {line}")

print("\n" + "=" * 70)
print("Script 01 completed successfully.")
print(f"  zip_contents_report.csv  -> {zip_csv}")
print(f"  environment.txt          -> {env_path}")
print("=" * 70)
