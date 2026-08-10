"""
Final training-composition transition audit.
READ-ONLY: no splits, predictions, or results are modified.
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR   = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
SPLITS_DIR = BASE_DIR / "results" / "stage3" / "splits"
TABLES_DIR = BASE_DIR / "tables" / "stage3"
META_PATH  = BASE_DIR / "tables" / "stage1" / "metadata.csv"
CVIND_PATH = BASE_DIR.parent / "cvind.mat"
RESULTS_DIR = BASE_DIR / "results" / "stage3"

# ── helpers ────────────────────────────────────────────────────────────────────
def select_patients_stratified(patients_df, frac, seed=42):
    n_total  = len(patients_df)
    n_select = int(round(n_total * frac))
    if n_select == 0:   return []
    if n_select == n_total: return patients_df["PID"].tolist()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_select, random_state=seed)
    for train_idx, _ in sss.split(patients_df, patients_df["label"]):
        return patients_df.iloc[train_idx]["PID"].tolist()

# ── load metadata ──────────────────────────────────────────────────────────────
meta = pd.read_csv(META_PATH)
with h5py.File(CVIND_PATH, "r") as f:
    cvind = np.array(f["cvind"]).flatten().astype(int)
meta["fold"] = cvind.flatten().astype(int)

# ── main audit ─────────────────────────────────────────────────────────────────
print("=" * 70)
print("TRAINING COMPOSITION TRANSITION AUDIT")
print("=" * 70)

audit_rows   = []
report_lines = []

for outer_fold in range(1, 6):
    test_fold_data = meta[meta["fold"] == outer_fold].copy()

    # ── reconstruct fixed test (min image_id per patient) ──────────────────────
    fixed_test = test_fold_data.loc[test_fold_data.groupby("PID")["image_id"].idxmin()]
    fixed_test_ids  = set(fixed_test["image_id"])
    test_fold_pids  = set(test_fold_data["PID"].unique())

    # ── reconstruct validation patients ────────────────────────────────────────
    non_test_data    = meta[meta["fold"] != outer_fold].copy()
    non_test_patients = non_test_data.groupby("PID")["label"].first().reset_index()
    val_pids          = set(select_patients_stratified(non_test_patients, 0.2, seed=42))

    # ── base training pool (= Condition A) ─────────────────────────────────────
    train_pool = non_test_data[~non_test_data["PID"].isin(val_pids)].copy()
    N_train    = len(train_pool)
    a_pids     = set(train_pool["PID"].unique())

    # ── overlap selection ───────────────────────────────────────────────────────
    patient_counts = test_fold_data.groupby("PID").size().reset_index(name="count")
    patient_info   = test_fold_data.groupby("PID")["label"].first().reset_index()
    patient_info   = patient_info.merge(patient_counts, on="PID")
    eligible       = patient_info[patient_info["count"] >= 2]

    b_overlap_pids_list = select_patients_stratified(eligible, 0.5, seed=42)
    c_overlap_pids_list = eligible["PID"].tolist()
    b_overlap_pids = set(b_overlap_pids_list)
    c_overlap_pids = set(c_overlap_pids_list)

    # ── forced overlap images ───────────────────────────────────────────────────
    test_candidates = test_fold_data[~test_fold_data["image_id"].isin(fixed_test_ids)]
    b_forced = test_candidates[test_candidates["PID"].isin(b_overlap_pids)]
    c_forced = test_candidates[test_candidates["PID"].isin(c_overlap_pids)]

    # ── READ actual saved manifests ─────────────────────────────────────────────
    a_train = pd.read_csv(SPLITS_DIR / f"fold{outer_fold}_A_train.csv")
    b_train = pd.read_csv(SPLITS_DIR / f"fold{outer_fold}_B_train.csv")
    c_train = pd.read_csv(SPLITS_DIR / f"fold{outer_fold}_C_train.csv")

    a_pids_actual = set(a_train["PID"].unique())
    b_pids_actual = set(b_train["PID"].unique())
    c_pids_actual = set(c_train["PID"].unique())

    a_ids = set(a_train["image_id"])
    b_ids = set(b_train["image_id"])
    c_ids = set(c_train["image_id"])

    # ── per-transition accounting ───────────────────────────────────────────────
    for cond, overlap_pids, forced_df, x_train, x_pids_actual, x_ids in [
        ("B", b_overlap_pids, b_forced, b_train, b_pids_actual, b_ids),
        ("C", c_overlap_pids, c_forced, c_train, c_pids_actual, c_ids),
    ]:
        # patients in A not in X  (displaced / removed entirely)
        pids_in_a_not_x = a_pids_actual - x_pids_actual
        # patients in X not in A  (newly introduced)
        pids_in_x_not_a = x_pids_actual - a_pids_actual

        # from the newly-introduced pids, how many are actual test-fold patients?
        overlap_pids_in_x_train = x_pids_actual & overlap_pids

        # images removed (in A but not in X)
        imgs_removed = a_ids - x_ids
        # images added (in X but not in A)
        imgs_added   = x_ids - a_ids

        # of imgs_added, how many belong to overlap patients?
        imgs_added_from_overlap = x_train[
            x_train["image_id"].isin(imgs_added) &
            x_train["PID"].isin(overlap_pids)
        ]

        # patients from train_pool who were completely dropped (their images fully removed)
        pool_pids_dropped = set()
        for pid in pids_in_a_not_x:
            # If this patient was in A but NOT in X and NOT a test-fold patient
            if pid not in test_fold_pids:
                pool_pids_dropped.add(pid)

        # pool images actually removed (images that were in A but not in X)
        pool_imgs_removed = len([
            img for img in imgs_removed
            if a_train[a_train["image_id"] == img]["PID"].values[0] not in test_fold_pids
        ])

        # Net change in unique patients
        delta_unique_patients = len(x_pids_actual) - len(a_pids_actual)

        # How many overlap-test patients appear in X training?
        n_overlap_test_in_x = len(overlap_pids_in_x_train)

        # Expected delta if NO displacement: +n_overlap_test_in_x
        # Actual delta: delta_unique_patients
        # Implied displaced patients = n_overlap_test_in_x - delta_unique_patients
        implied_displaced = n_overlap_test_in_x - delta_unique_patients

        row = {
            "fold": outer_fold,
            "transition": f"A→{cond}",
            "N_train":                        N_train,
            "unique_pts_A":                   len(a_pids_actual),
            "unique_pts_X":                   len(x_pids_actual),
            "delta_unique_patients":          delta_unique_patients,
            "n_overlap_test_patients_in_X_train":  n_overlap_test_in_x,
            "implied_non_test_pts_displaced": implied_displaced,
            "n_pool_patients_fully_dropped":  len(pool_pids_dropped),
            "overlap_images_forced_in":       len(forced_df),
            "pool_images_removed":            len(imgs_removed) - len(forced_df),
            "images_added_from_overlap":      len(imgs_added_from_overlap),
            "imgs_removed_total":             len(imgs_removed),
            "imgs_added_total":               len(imgs_added),
            "pids_in_A_not_X":               len(pids_in_a_not_x),
            "pids_in_X_not_A":               len(pids_in_x_not_a),
        }
        audit_rows.append(row)
        print(f"Fold {outer_fold} A→{cond}: "
              f"Δunique={delta_unique_patients:+d}, "
              f"overlap_pts_in_train={n_overlap_test_in_x}, "
              f"implied_displaced={implied_displaced}, "
              f"pool_fully_dropped={len(pool_pids_dropped)}")

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(TABLES_DIR / "training_composition_transition_audit.csv", index=False)
print("\n→ saved training_composition_transition_audit.csv")

# ── verify patient_overlap_prediction_analysis row count ──────────────────────
pat_df = pd.read_csv(TABLES_DIR / "patient_overlap_prediction_analysis.csv")
print(f"\npatient_overlap_prediction_analysis.csv: {len(pat_df)} rows (header excluded)")

# Print the full audit table for the report
print("\nFull audit table:")
print(audit_df.to_string(index=False))
