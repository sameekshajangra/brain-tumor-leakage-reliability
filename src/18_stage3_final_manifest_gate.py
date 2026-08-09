import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
METADATA_PATH = BASE_DIR / "tables" / "stage1" / "metadata.csv"
SPLITS_DIR = BASE_DIR / "results" / "stage3" / "splits"
TABLES_DIR = BASE_DIR / "tables" / "stage3"
RESULTS_DIR = BASE_DIR / "results" / "stage3"

def main():
    print("="*70)
    print("STAGE 3 FINAL MANIFEST GATE")
    print("="*70)
    
    meta = pd.read_csv(METADATA_PATH)
    all_metadata_ids = set(meta['image_id'])
    summary_df = pd.read_csv(TABLES_DIR / "overlap_selection_summary.csv")
    
    report = []
    report.append("STAGE 3 FINAL MANIFEST GATE REPORT")
    report.append("==================================")
    
    all_passed = True
    
    for outer_fold in range(1, 6):
        report.append(f"\n--- OUTER FOLD {outer_fold} ---")
        
        # Load splits
        dfs = {}
        for cond in ['A', 'B', 'C']:
            for s in ['train', 'val', 'test']:
                dfs[f"{cond}_{s}"] = pd.read_csv(SPLITS_DIR / f"fold{outer_fold}_{cond}_{s}.csv")
                
        # Basic identicalness checks
        test_eq = dfs['A_test']['image_id'].equals(dfs['B_test']['image_id']) and dfs['A_test']['image_id'].equals(dfs['C_test']['image_id'])
        val_eq = dfs['A_val']['image_id'].equals(dfs['B_val']['image_id']) and dfs['A_val']['image_id'].equals(dfs['C_val']['image_id'])
        n_train_eq = (len(dfs['A_train']) == len(dfs['B_train'])) and (len(dfs['A_train']) == len(dfs['C_train']))
        
        a_counts = dfs['A_train']['label'].value_counts().sort_index()
        b_counts = dfs['B_train']['label'].value_counts().sort_index()
        c_counts = dfs['C_train']['label'].value_counts().sort_index()
        class_eq = a_counts.equals(b_counts) and a_counts.equals(c_counts)
        
        report.append(f"Identical test manifests: {'[OK]' if test_eq else '[FAIL]'}")
        report.append(f"Identical val manifests: {'[OK]' if val_eq else '[FAIL]'}")
        report.append(f"Identical N_train: {'[OK]' if n_train_eq else '[FAIL]'}")
        report.append(f"Identical class counts: {'[OK]' if class_eq else '[FAIL]'}")
        
        all_passed &= (test_eq and val_eq and n_train_eq and class_eq)
        if not all_passed:
            report.append("Basic equality checks failed.")
            break
            
        test_df = dfs['A_test']
        test_pids = set(test_df['PID'])
        test_images = set(test_df['image_id'])
        
        # Identify eligible/ineligible patients
        test_fold_data = meta[meta['PID'].isin(test_pids)]
        img_counts = test_fold_data.groupby('PID').size()
        eligible_pids = set(img_counts[img_counts >= 2].index)
        ineligible_pids = set(img_counts[img_counts < 2].index)
        
        val_df = dfs['A_val']
        val_pids = set(val_df['PID'])
        val_images = set(val_df['image_id'])
        
        for cond in ['A', 'B', 'C']:
            cond_passed = True
            train_df = dfs[f"{cond}_train"]
            train_pids = set(train_df['PID'])
            train_images = set(train_df['image_id'])
            
            # Validation disjointness (Rule 5)
            if len(val_pids.intersection(test_pids)) > 0:
                report.append(f"[FAIL] Fold {outer_fold} {cond}: Val/Test patient overlap")
                cond_passed = False
            if len(val_pids.intersection(train_pids)) > 0:
                report.append(f"[FAIL] Fold {outer_fold} {cond}: Val/Train patient overlap")
                cond_passed = False
            if len(val_images.intersection(train_images)) > 0 or len(val_images.intersection(test_images)) > 0:
                report.append(f"[FAIL] Fold {outer_fold} {cond}: Val image overlaps with Train/Test")
                cond_passed = False
                
            # Rule 8: No duplicate image_id
            if not train_df['image_id'].is_unique:
                report.append(f"[FAIL] Fold {outer_fold} {cond}: Duplicate image_id in train")
                cond_passed = False
            # Rule 9: Every image_id belongs to metadata
            if not train_images.issubset(all_metadata_ids):
                report.append(f"[FAIL] Fold {outer_fold} {cond}: Unknown image_id in train")
                cond_passed = False
                
            # Condition-specific patient rules
            test_pts_in_train = test_pids.intersection(train_pids)
            
            if cond == 'A':
                if len(test_pts_in_train) > 0:
                    report.append(f"[FAIL] Fold {outer_fold} Cond A: Found {len(test_pts_in_train)} test patients in training. Expected 0.")
                    cond_passed = False
                if len(test_images.intersection(train_images)) > 0:
                    report.append(f"[FAIL] Fold {outer_fold} Cond A: Fixed test image in training.")
                    cond_passed = False
                    
            elif cond == 'B':
                expected_pids = summary_df[(summary_df['outer_fold']==outer_fold) & (summary_df['condition']=='B')].iloc[0]['overlapping_patients']
                selected_overlap_pids = test_pts_in_train
                if len(selected_overlap_pids) != expected_pids:
                    report.append(f"[FAIL] Fold {outer_fold} Cond B: Expected {expected_pids} overlap patients, found {len(selected_overlap_pids)}")
                    cond_passed = False
                    
                unselected_pids = eligible_pids - selected_overlap_pids
                
                # Check selected overlap patients
                for pid in selected_overlap_pids:
                    # check fixed test image absent
                    pid_fixed_test = test_df[test_df['PID'] == pid]['image_id'].values[0]
                    if pid_fixed_test in train_images:
                        report.append(f"[FAIL] Fold {outer_fold} Cond B PID {pid}: Fixed test image {pid_fixed_test} in training.")
                        cond_passed = False
                    # check required non-test images present
                    pid_all_images = set(meta[meta['PID'] == pid]['image_id'])
                    required_train_images = pid_all_images - {pid_fixed_test}
                    if not required_train_images.issubset(train_images):
                        report.append(f"[FAIL] Fold {outer_fold} Cond B PID {pid}: Missing required non-test images in training.")
                        cond_passed = False
                        
                # Check unselected eligible
                if len(unselected_pids.intersection(train_pids)) > 0:
                    report.append(f"[FAIL] Fold {outer_fold} Cond B: Unselected eligible test patient has images in training.")
                    cond_passed = False
                    
                # Check ineligible
                if len(ineligible_pids.intersection(train_pids)) > 0:
                    report.append(f"[FAIL] Fold {outer_fold} Cond B: Ineligible test patient has images in training.")
                    cond_passed = False
                    
                # Check ALL fixed test images absent
                if len(test_images.intersection(train_images)) > 0:
                    report.append(f"[FAIL] Fold {outer_fold} Cond B: Fixed test image in training.")
                    cond_passed = False
                    
            elif cond == 'C':
                expected_pids = len(eligible_pids)
                if len(test_pts_in_train) != expected_pids:
                    report.append(f"[FAIL] Fold {outer_fold} Cond C: Expected {expected_pids} overlap patients, found {len(test_pts_in_train)}")
                    cond_passed = False
                
                for pid in eligible_pids:
                    pid_fixed_test = test_df[test_df['PID'] == pid]['image_id'].values[0]
                    if pid_fixed_test in train_images:
                        report.append(f"[FAIL] Fold {outer_fold} Cond C PID {pid}: Fixed test image in training.")
                        cond_passed = False
                    
                    pid_all_images = set(meta[meta['PID'] == pid]['image_id'])
                    required_train_images = pid_all_images - {pid_fixed_test}
                    if not required_train_images.issubset(train_images):
                        report.append(f"[FAIL] Fold {outer_fold} Cond C PID {pid}: Missing required non-test images in training.")
                        cond_passed = False
                        
                if len(ineligible_pids.intersection(train_pids)) > 0:
                    report.append(f"[FAIL] Fold {outer_fold} Cond C: Ineligible test patient has images in training.")
                    cond_passed = False
                    
                if len(test_images.intersection(train_images)) > 0:
                    report.append(f"[FAIL] Fold {outer_fold} Cond C: Fixed test image in training.")
                    cond_passed = False

            if cond_passed:
                report.append(f"[OK] Condition {cond} passed all patient/image level strict gating rules.")
            all_passed &= cond_passed
            
    if all_passed:
        print("FINAL MANIFEST GATE: PASSED")
        print("NO MODEL TRAINING PERFORMED")
        print("READY FOR PHASES 7–11")
        report.append("\n==================================")
        report.append("FINAL MANIFEST GATE: PASSED")
        report.append("NO MODEL TRAINING PERFORMED")
        report.append("READY FOR PHASES 7-11")
        report.append("==================================")
    else:
        print("FINAL MANIFEST GATE: FAILED")
        report.append("\n==================================")
        report.append("FINAL MANIFEST GATE: FAILED")
        report.append("==================================")
        
    with open(RESULTS_DIR / "final_manifest_gate_report.txt", "w") as f:
        f.write("\n".join(report) + "\n")

if __name__ == "__main__":
    main()
