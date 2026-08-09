import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
PRED_DIR = BASE_DIR / "results" / "stage3" / "predictions"
TABLES_DIR = BASE_DIR / "tables" / "stage3"

def compute_ece(df, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        
        in_bin = df[(df['max_confidence'] > bin_lower) & (df['max_confidence'] <= bin_upper)]
        if i == 0:
            in_bin = df[(df['max_confidence'] >= bin_lower) & (df['max_confidence'] <= bin_upper)]
            
        n_in_bin = len(in_bin)
        if n_in_bin > 0:
            acc_in_bin = in_bin['correct'].mean()
            conf_in_bin = in_bin['max_confidence'].mean()
            ece += (n_in_bin / len(df)) * np.abs(acc_in_bin - conf_in_bin)
    return ece

def compute_brier(df):
    y_true = df['true_label']
    y_true_onehot = pd.get_dummies(y_true).reindex(columns=[1,2,3], fill_value=0).values
    probs = df[['probability_meningioma', 'probability_glioma', 'probability_pituitary']].values
    return np.mean(np.sum((probs - y_true_onehot)**2, axis=1))

def compute_calibration(df):
    mean_conf = df['max_confidence'].mean()
    median_conf = df['max_confidence'].median()
    
    correct_df = df[df['correct'] == 1]
    incorrect_df = df[df['correct'] == 0]
    
    conf_correct = correct_df['max_confidence'].mean() if len(correct_df)>0 else np.nan
    conf_incorrect = incorrect_df['max_confidence'].mean() if len(incorrect_df)>0 else np.nan
    conf_gap = conf_correct - conf_incorrect if not np.isnan(conf_correct) and not np.isnan(conf_incorrect) else np.nan
    
    hi90 = df[df['max_confidence'] >= 0.90]
    hcer_90 = (1 - hi90['correct'].mean()) if len(hi90)>0 else np.nan
    
    hi95 = df[df['max_confidence'] >= 0.95]
    hcer_95 = (1 - hi95['correct'].mean()) if len(hi95)>0 else np.nan
    
    ece = compute_ece(df, 15)
    brier = compute_brier(df)
    
    return {
        'mean_confidence': mean_conf,
        'median_confidence': median_conf,
        'conf_correct': conf_correct,
        'conf_incorrect': conf_incorrect,
        'confidence_gap': conf_gap,
        'hcer_90': hcer_90,
        'hcer_95': hcer_95,
        'ece': ece,
        'brier_score': brier
    }

def main():
    all_preds = []
    fold_results = []
    for fold in range(1, 6):
        for cond in ['A', 'B', 'C']:
            path = PRED_DIR / f"fold{fold}_condition{cond}_predictions.csv"
            df = pd.read_csv(path)
            all_preds.append(df)
            res = compute_calibration(df)
            res['fold'] = fold
            res['condition'] = cond
            fold_results.append(res)
            
    fold_df = pd.DataFrame(fold_results)
    cols = ['fold', 'condition'] + [c for c in fold_df.columns if c not in ['fold', 'condition']]
    fold_df = fold_df[cols]
    fold_df.to_csv(TABLES_DIR / "fold_condition_calibration.csv", index=False)
    
    full_df = pd.concat(all_preds)
    pooled_results = []
    for cond in ['A', 'B', 'C']:
        cond_df = full_df[full_df['condition'] == cond]
        res = compute_calibration(cond_df)
        res['condition'] = cond
        res['test_images'] = len(cond_df)
        pooled_results.append(res)
        
    pooled_df = pd.DataFrame(pooled_results)
    cols = ['condition', 'test_images'] + [c for c in pooled_df.columns if c not in ['condition', 'test_images']]
    pooled_df = pooled_df[cols]
    pooled_df.to_csv(TABLES_DIR / "pooled_calibration.csv", index=False)
    print("Calibration metrics saved.")

if __name__ == "__main__":
    main()
