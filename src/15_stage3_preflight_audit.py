import pandas as pd
import h5py
import numpy as np
from pathlib import Path
import os

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
METADATA_PATH = BASE_DIR / "tables" / "stage1" / "metadata.csv"
CVIND_PATH = BASE_DIR.parent / "cvind.mat"
RESULTS_DIR = BASE_DIR / "results" / "stage3"
TABLES_DIR = BASE_DIR / "tables" / "stage3"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

def main():
    print("="*70)
    print("STAGE 3 PREFLIGHT AUDIT")
    print("="*70)
    
    meta = pd.read_csv(METADATA_PATH)
    total_images = len(meta)
    unique_pids = meta['PID'].nunique()
    
    assert total_images == 3064, "Image count mismatch"
    assert unique_pids == 233, "Patient count mismatch"
    
    invalid_labels = meta[~meta['label'].isin([1, 2, 3])]
    assert len(invalid_labels) == 0, "Invalid labels found"
    
    with h5py.File(CVIND_PATH, 'r') as f:
        cvind = np.array(f['cvind']).flatten().astype(int)
    
    assert len(cvind) == 3064, "cvind length mismatch"
    meta['fold'] = cvind.flatten().astype(int)
    
    pid_to_folds = meta.groupby('PID')['fold'].nunique()
    pids_multiple_folds = pid_to_folds[pid_to_folds > 1]
    assert len(pids_multiple_folds) == 0, "PID leakage across folds detected"
    
    assert meta['image_id'].is_unique, "Duplicate image IDs found"
    
    fold_stats = []
    for fold in range(1, 6):
        fold_data = meta[meta['fold'] == fold]
        n_images = len(fold_data)
        n_patients = fold_data['PID'].nunique()
        imgs_per_patient = fold_data.groupby('PID').size()
        
        fold_stats.append({
            'fold': fold,
            'patients': n_patients,
            'images': n_images,
            'mean_img_per_patient': imgs_per_patient.mean(),
            'median_img_per_patient': imgs_per_patient.median(),
            'min_img': imgs_per_patient.min(),
            'max_img': imgs_per_patient.max(),
            'patients_exactly_1_img': (imgs_per_patient == 1).sum(),
            'patients_ge_2_img': (imgs_per_patient >= 2).sum(),
            'patients_ge_5_img': (imgs_per_patient >= 5).sum(),
            'patients_ge_10_img': (imgs_per_patient >= 10).sum()
        })
        
    fold_df = pd.DataFrame(fold_stats)
    fold_df.to_csv(TABLES_DIR / "preflight_fold_summary.csv", index=False)
    
    report = [
        "STAGE 3 PREFLIGHT AUDIT REPORT",
        "==============================",
        f"Total Images: {total_images} (Expected: 3064)",
        f"Unique PIDs: {unique_pids} (Expected: 233)",
        "Labels: All in {1, 2, 3}",
        "PID multiple folds: 0",
        "",
        "FOLD DISTRIBUTIONS:",
        fold_df.to_string()
    ]
    
    with open(RESULTS_DIR / "preflight_audit_report.txt", "w") as f:
        f.write("\n".join(report) + "\n")
        
    print(fold_df)
    print("="*70)
    print("PREFLIGHT AUDIT PASSED")
    print("="*70)

if __name__ == "__main__":
    main()
