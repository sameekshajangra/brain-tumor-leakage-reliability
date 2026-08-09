import pandas as pd
from pathlib import Path
import os

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
SPLITS_DIR = BASE_DIR / "results" / "stage3" / "splits"
TABLES_DIR = BASE_DIR / "tables" / "stage3"
RESULTS_DIR = BASE_DIR / "results" / "stage3"

def main():
    print("="*70)
    print("STAGE 3 SPLIT VERIFICATION")
    print("="*70)
    
    summary = pd.read_csv(TABLES_DIR / "overlap_selection_summary.csv")
    
    report = []
    report.append("STAGE 3 SPLIT INTEGRITY REPORT")
    report.append("==============================")
    
    all_passed = True
    
    for outer_fold in range(1, 6):
        report.append(f"\n--- OUTER FOLD {outer_fold} ---")
        
        dfs = {}
        for cond in ['A', 'B', 'C']:
            for s in ['train', 'val', 'test']:
                dfs[f"{cond}_{s}"] = pd.read_csv(SPLITS_DIR / f"fold{outer_fold}_{cond}_{s}.csv")
                
        # 1. Identical Test Sets
        test_eq = dfs['A_test']['image_id'].equals(dfs['B_test']['image_id']) and dfs['A_test']['image_id'].equals(dfs['C_test']['image_id'])
        report.append(f"[OK] A/B/C test sets identical" if test_eq else "[FAIL] A/B/C test sets differ")
        all_passed &= test_eq
        
        # 2. Identical Val Sets
        val_eq = dfs['A_val']['image_id'].equals(dfs['B_val']['image_id']) and dfs['A_val']['image_id'].equals(dfs['C_val']['image_id'])
        report.append(f"[OK] A/B/C val sets identical" if val_eq else "[FAIL] A/B/C val sets differ")
        all_passed &= val_eq
        
        # 3. Identical N_train
        n_train_eq = (len(dfs['A_train']) == len(dfs['B_train'])) and (len(dfs['A_train']) == len(dfs['C_train']))
        report.append(f"[OK] A/B/C train sizes identical ({len(dfs['A_train'])})" if n_train_eq else "[FAIL] A/B/C train sizes differ")
        all_passed &= n_train_eq
        
        # 4. Valid labels
        for split_name, df in dfs.items():
            valid_labels = df['label'].isin([1, 2, 3]).all()
            if not valid_labels:
                report.append(f"[FAIL] Invalid labels in {split_name}")
                all_passed = False
            
            # 5. No duplicates within a split
            if not df['image_id'].is_unique:
                report.append(f"[FAIL] Duplicate images in {split_name}")
                all_passed = False
                
        # For condition logic
        test_images = set(dfs['A_test']['image_id'])
        test_pids = set(dfs['A_test']['PID'])
        val_pids = set(dfs['A_val']['PID'])
        
        for cond in ['A', 'B', 'C']:
            train = dfs[f"{cond}_train"]
            val = dfs[f"{cond}_val"]
            train_images = set(train['image_id'])
            train_pids = set(train['PID'])
            
            # Basic invariant: No test image in train/val
            if len(test_images.intersection(train_images)) > 0:
                report.append(f"[FAIL] Cond {cond}: Test image leakage into training!")
                all_passed = False
            if len(test_images.intersection(set(val['image_id']))) > 0:
                report.append(f"[FAIL] Cond {cond}: Test image leakage into validation!")
                all_passed = False
                
            # Basic invariant: Val patients disjoint from train and test
            if len(val_pids.intersection(train_pids)) > 0:
                report.append(f"[FAIL] Cond {cond}: Val patient leakage into training!")
                all_passed = False
            if len(val_pids.intersection(test_pids)) > 0:
                report.append(f"[FAIL] Cond {cond}: Val patient leakage into test!")
                all_passed = False
                
            # Condition specific checks
            overlap_pids = test_pids.intersection(train_pids)
            if cond == 'A':
                if len(overlap_pids) > 0:
                    report.append(f"[FAIL] Cond A has {len(overlap_pids)} test patients in training!")
                    all_passed = False
                else:
                    report.append(f"[OK] Cond A has 0% test patient overlap")
            
            elif cond == 'B':
                # Check summary info
                cond_sum = summary[(summary['outer_fold'] == outer_fold) & (summary['condition'] == 'B')].iloc[0]
                expected_pids = cond_sum['overlapping_patients']
                if len(overlap_pids) == expected_pids:
                    report.append(f"[OK] Cond B has exactly {expected_pids} test patients in training (~50%)")
                else:
                    report.append(f"[FAIL] Cond B overlap mismatch! Got {len(overlap_pids)}, expected {expected_pids}")
                    all_passed = False
                    
            elif cond == 'C':
                cond_sum = summary[(summary['outer_fold'] == outer_fold) & (summary['condition'] == 'C')].iloc[0]
                expected_pids = cond_sum['overlapping_patients']
                if len(overlap_pids) == expected_pids:
                    report.append(f"[OK] Cond C has exactly {expected_pids} test patients in training (100% eligible)")
                else:
                    report.append(f"[FAIL] Cond C overlap mismatch! Got {len(overlap_pids)}, expected {expected_pids}")
                    all_passed = False
                    
    if all_passed:
        report.append("\n==============================")
        report.append("ALL INTEGRITY CHECKS PASSED ✅")
        report.append("==============================")
        print("ALL CHECKS PASSED")
    else:
        report.append("\n==============================")
        report.append("INTEGRITY CHECKS FAILED ❌")
        report.append("==============================")
        print("CHECKS FAILED")
        
    with open(RESULTS_DIR / "stage3_split_integrity_report.txt", "w") as f:
        f.write("\n".join(report) + "\n")

if __name__ == "__main__":
    main()
