import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
PRED_DIR = BASE_DIR / "results" / "stage3" / "predictions"
TABLES_DIR = BASE_DIR / "tables" / "stage3"

def compute_metrics(df):
    y_true = df['true_label']
    y_pred = df['predicted_label']
    
    acc = accuracy_score(y_true, y_pred)
    mac_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    mac_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    mac_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Per-class
    p_class = precision_score(y_true, y_pred, average=None, labels=[1,2,3], zero_division=0)
    r_class = recall_score(y_true, y_pred, average=None, labels=[1,2,3], zero_division=0)
    f1_class = f1_score(y_true, y_pred, average=None, labels=[1,2,3], zero_division=0)
    
    return {
        'accuracy': acc,
        'macro_precision': mac_p,
        'macro_recall': mac_r,
        'macro_f1': mac_f1,
        'balanced_accuracy': bal_acc,
        'precision_meningioma': p_class[0], 'precision_glioma': p_class[1], 'precision_pituitary': p_class[2],
        'recall_meningioma': r_class[0], 'recall_glioma': r_class[1], 'recall_pituitary': r_class[2],
        'f1_meningioma': f1_class[0], 'f1_glioma': f1_class[1], 'f1_pituitary': f1_class[2]
    }

def main():
    all_preds = []
    fold_results = []
    for fold in range(1, 6):
        for cond in ['A', 'B', 'C']:
            path = PRED_DIR / f"fold{fold}_condition{cond}_predictions.csv"
            df = pd.read_csv(path)
            all_preds.append(df)
            res = compute_metrics(df)
            res['fold'] = fold
            res['condition'] = cond
            fold_results.append(res)
            
    fold_df = pd.DataFrame(fold_results)
    cols = ['fold', 'condition'] + [c for c in fold_df.columns if c not in ['fold', 'condition']]
    fold_df = fold_df[cols]
    fold_df.to_csv(TABLES_DIR / "fold_condition_performance.csv", index=False)
    
    full_df = pd.concat(all_preds)
    pooled_results = []
    for cond in ['A', 'B', 'C']:
        cond_df = full_df[full_df['condition'] == cond]
        res = compute_metrics(cond_df)
        res['condition'] = cond
        res['test_images'] = len(cond_df)
        pooled_results.append(res)
        
    pooled_df = pd.DataFrame(pooled_results)
    cols = ['condition', 'test_images'] + [c for c in pooled_df.columns if c not in ['condition', 'test_images']]
    pooled_df = pooled_df[cols]
    pooled_df.to_csv(TABLES_DIR / "pooled_performance.csv", index=False)
    print("Performance metrics saved.")

if __name__ == "__main__":
    main()
