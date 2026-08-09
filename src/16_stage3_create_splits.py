import pandas as pd
import h5py
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
METADATA_PATH = BASE_DIR / "tables" / "stage1" / "metadata.csv"
CVIND_PATH = BASE_DIR.parent / "cvind.mat"
SPLITS_DIR = BASE_DIR / "results" / "stage3" / "splits"
TABLES_DIR = BASE_DIR / "tables" / "stage3"

os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

def load_data():
    meta = pd.read_csv(METADATA_PATH)
    with h5py.File(CVIND_PATH, 'r') as f:
        cvind = np.array(f['cvind']).flatten().astype(int)
    meta['fold'] = cvind.flatten().astype(int)
    return meta

def select_patients_stratified(patients_df, frac, seed=42):
    n_total = len(patients_df)
    n_select = int(round(n_total * frac))
    if n_select == 0:
        return []
    if n_select == n_total:
        return patients_df['PID'].tolist()
        
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_select, random_state=seed)
    for train_idx, _ in sss.split(patients_df, patients_df['label']):
        return patients_df.iloc[train_idx]['PID'].tolist()

def fill_remaining(forced_images, pool_images, target_n, target_dist):
    """
    Fills up to target_n images from pool_images, trying to match target_dist.
    target_dist is a Series of class proportions (e.g. {1: 0.2, 2: 0.5, 3: 0.3})
    """
    n_needed = target_n - len(forced_images)
    assert n_needed >= 0, "More forced images than target_n!"
    if n_needed == 0:
        return forced_images
        
    # Current counts in forced images
    curr_counts = forced_images['label'].value_counts().reindex([1,2,3], fill_value=0)
    
    # Target counts for the FULL dataset
    target_counts = (target_dist * target_n).round().astype(int)
    
    # How many more we need per class
    needed_per_class = target_counts - curr_counts
    # Handle negative needs (if forced overlap already exceeds target for a class)
    needed_per_class = needed_per_class.clip(lower=0)
    
    # If the sum of needed doesn't match n_needed perfectly due to rounding, adjust it
    diff = n_needed - needed_per_class.sum()
    if diff > 0:
        # Add to the most abundant class
        max_cls = target_dist.idxmax()
        needed_per_class[max_cls] += diff
    elif diff < 0:
        # Subtract from where we can
        for cls in [1, 2, 3]:
            if diff == 0: break
            if needed_per_class[cls] > 0:
                take = min(needed_per_class[cls], -diff)
                needed_per_class[cls] -= take
                diff += take
                
    # Now sample from pool_images
    selected_pool = []
    for cls in [1, 2, 3]:
        n_req = needed_per_class[cls]
        cls_pool = pool_images[pool_images['label'] == cls]
        
        # If we don't have enough in the pool, just take all we have, and we'll have to make it up elsewhere
        n_take = min(n_req, len(cls_pool))
        if n_take > 0:
            # We sort by image_id to be deterministic if we are sampling
            cls_pool = cls_pool.sort_values('image_id')
            selected_pool.append(cls_pool.head(n_take))
            
    selected_pool_df = pd.concat(selected_pool) if selected_pool else pd.DataFrame()
    
    # If we still haven't met n_needed (because some classes didn't have enough in pool)
    shortfall = n_needed - len(selected_pool_df)
    if shortfall > 0:
        remaining_pool = pool_images[~pool_images['image_id'].isin(selected_pool_df['image_id'])]
        remaining_pool = remaining_pool.sort_values('image_id')
        selected_pool_df = pd.concat([selected_pool_df, remaining_pool.head(shortfall)])
        
    final_df = pd.concat([forced_images, selected_pool_df]).sort_values('image_id')
    assert len(final_df) == target_n, f"Failed to reach exact N_train! Got {len(final_df)} vs {target_n}"
    return final_df

def main():
    print("="*70)
    print("STAGE 3 SPLIT CREATION")
    print("="*70)
    
    meta = load_data()
    summary_rows = []
    all_fixed_test = []
    
    for outer_fold in range(1, 6):
        print(f"\n--- OUTER FOLD {outer_fold} ---")
        
        # 1. FIXED TEST SET
        test_fold_data = meta[meta['fold'] == outer_fold].copy()
        
        # Determine Patient class (majority vote, though it's 100% consistent)
        patient_info = test_fold_data.groupby('PID')['label'].first().reset_index()
        patient_counts = test_fold_data.groupby('PID').size().reset_index(name='count')
        patient_info = patient_info.merge(patient_counts, on='PID')
        
        # Fixed test image: min image_id per patient
        fixed_test = test_fold_data.loc[test_fold_data.groupby('PID')['image_id'].idxmin()]
        fixed_test.to_csv(TABLES_DIR / f"fixed_test_images_fold{outer_fold}.csv", index=False)
        all_fixed_test.append(fixed_test)
        
        print(f"Fixed test images: {len(fixed_test)}")
        
        # 2. VALIDATION SET
        non_test_data = meta[meta['fold'] != outer_fold].copy()
        non_test_patients = non_test_data.groupby('PID')['label'].first().reset_index()
        
        val_pids = select_patients_stratified(non_test_patients, 0.2, seed=42)
        val_data = non_test_data[non_test_data['PID'].isin(val_pids)].copy()
        
        # 3. BASE TRAINING POOL (Non-test images minus Validation patients)
        train_pool = non_test_data[~non_test_data['PID'].isin(val_pids)].copy()
        N_train = len(train_pool)
        
        print(f"Validation images: {len(val_data)}")
        print(f"N_train defined as: {N_train}")
        
        # 4. OVERLAP FEASIBILITY & SELECTION
        eligible_patients = patient_info[patient_info['count'] >= 2]
        print(f"Eligible test-fold patients: {len(eligible_patients)} / {len(patient_info)}")
        
        # Select B overlap patients (approx 50% stratified)
        b_overlap_pids = select_patients_stratified(eligible_patients, 0.5, seed=42)
        c_overlap_pids = eligible_patients['PID'].tolist()
        
        # Extract candidate forced images (all test-fold images EXCEPT the fixed test images)
        test_fold_candidates = test_fold_data[~test_fold_data['image_id'].isin(fixed_test['image_id'])].copy()
        
        b_forced_images = test_fold_candidates[test_fold_candidates['PID'].isin(b_overlap_pids)].copy()
        c_forced_images = test_fold_candidates[test_fold_candidates['PID'].isin(c_overlap_pids)].copy()
        
        print(f"Forced overlap images B: {len(b_forced_images)}")
        print(f"Forced overlap images C: {len(c_forced_images)}")
        
        if len(b_forced_images) > N_train or len(c_forced_images) > N_train:
            print("FEASIBILITY CHECK FAILED: Forced overlap images exceed N_train!")
            exit(1)
            
        # 5. CONSTRUCT CONDITIONS
        
        # Condition A
        a_train = train_pool.copy().sort_values('image_id')
        a_dist = a_train['label'].value_counts(normalize=True).reindex([1,2,3], fill_value=0)
        
        # Condition B
        b_train = fill_remaining(b_forced_images, train_pool, N_train, a_dist)
        
        # Condition C
        c_train = fill_remaining(c_forced_images, train_pool, N_train, a_dist)
        
        # 6. SAVE SPLITS
        splits = {
            'A': a_train,
            'B': b_train,
            'C': c_train
        }
        
        for cond, df in splits.items():
            df.to_csv(SPLITS_DIR / f"fold{outer_fold}_{cond}_train.csv", index=False)
            val_data.to_csv(SPLITS_DIR / f"fold{outer_fold}_{cond}_val.csv", index=False)
            fixed_test.to_csv(SPLITS_DIR / f"fold{outer_fold}_{cond}_test.csv", index=False)
            
            # Log summary
            if cond == 'A':
                over_pids = 0
                forced = 0
            elif cond == 'B':
                over_pids = len(b_overlap_pids)
                forced = len(b_forced_images)
            else:
                over_pids = len(c_overlap_pids)
                forced = len(c_forced_images)
                
            dist = df['label'].value_counts().reindex([1,2,3], fill_value=0)
            
            summary_rows.append({
                'outer_fold': outer_fold,
                'condition': cond,
                'eligible_test_patients': len(eligible_patients),
                'overlapping_patients': over_pids,
                'overlap_percentage': (over_pids / len(eligible_patients) * 100) if len(eligible_patients)>0 else 0,
                'overlap_images': forced,
                'train_images': len(df),
                'val_images': len(val_data),
                'test_images': len(fixed_test),
                'meningioma_train': dist[1],
                'glioma_train': dist[2],
                'pituitary_train': dist[3]
            })
            
    pd.concat(all_fixed_test).to_csv(TABLES_DIR / "fixed_test_images_all_folds.csv", index=False)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "overlap_selection_summary.csv", index=False)
    
    print("\nOVERLAP SUMMARY:")
    print(summary_df.to_string())
    print("="*70)
    print("STAGE 3 SPLITS GENERATED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()
