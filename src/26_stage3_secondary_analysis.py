import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
PRED_DIR = BASE_DIR / "results" / "stage3" / "predictions"
SPLITS_DIR = BASE_DIR / "results" / "stage3" / "splits"
TABLES_DIR = BASE_DIR / "tables" / "stage3"
FIG_DIR = BASE_DIR / "figures" / "stage3"
RESULTS_DIR = BASE_DIR / "results" / "stage3"

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

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a), stats.sem(a, nan_policy='omit')
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def main():
    print("=" * 70)
    print("STAGE 3 — SECONDARY ANALYSIS")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1. RECOVER THE COMPLETE PER-FOLD RESULTS
    # -------------------------------------------------------------------------
    all_preds_list = []
    for fold in range(1, 6):
        for cond in ['A', 'B', 'C']:
            df = pd.read_csv(PRED_DIR / f"fold{fold}_condition{cond}_predictions.csv")
            all_preds_list.append(df)
            
    full_preds = pd.concat(all_preds_list)
    
    fold_cond_results = []
    for fold in range(1, 6):
        for cond in ['A', 'B', 'C']:
            df = full_preds[(full_preds['fold'] == fold) & (full_preds['condition'] == cond)]
            
            y_true = df['true_label']
            y_pred = df['predicted_label']
            
            acc = accuracy_score(y_true, y_pred)
            mac_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
            mac_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
            mac_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            
            mean_conf = df['max_confidence'].mean()
            median_conf = df['max_confidence'].median()
            
            conf_correct = df[df['correct'] == 1]['max_confidence'].mean()
            conf_incorrect = df[df['correct'] == 0]['max_confidence'].mean()
            
            ece = compute_ece(df)
            brier = compute_brier(df)
            
            hcer_90 = 1.0 - df[df['max_confidence'] >= 0.90]['correct'].mean() if len(df[df['max_confidence'] >= 0.90]) > 0 else np.nan
            hcer_95 = 1.0 - df[df['max_confidence'] >= 0.95]['correct'].mean() if len(df[df['max_confidence'] >= 0.95]) > 0 else np.nan
            
            fold_cond_results.append({
                'fold': fold, 'condition': cond,
                'test_patient_count': df['PID'].nunique(),
                'test_image_count': len(df),
                'accuracy': acc, 'macro_precision': mac_p, 'macro_recall': mac_r,
                'macro_f1': mac_f1, 'balanced_accuracy': bal_acc,
                'mean_confidence': mean_conf, 'median_confidence': median_conf,
                'conf_correct': conf_correct, 'conf_incorrect': conf_incorrect,
                'ece': ece, 'brier_score': brier,
                'hcer_90': hcer_90, 'hcer_95': hcer_95
            })
            
    complete_fold_cond_df = pd.DataFrame(fold_cond_results)
    complete_fold_cond_df.to_csv(TABLES_DIR / "complete_fold_condition_results.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 2. PAIRED FOLD-LEVEL EFFECTS
    # -------------------------------------------------------------------------
    metrics = [c for c in complete_fold_cond_df.columns if c not in ['fold', 'condition', 'test_patient_count', 'test_image_count']]
    
    paired_fold_diffs = []
    
    for fold in range(1, 6):
        fold_df = complete_fold_cond_df[complete_fold_cond_df['fold'] == fold].set_index('condition')
        for metric in metrics:
            valA = fold_df.loc['A', metric]
            valB = fold_df.loc['B', metric]
            valC = fold_df.loc['C', metric]
            
            paired_fold_diffs.append({
                'fold': fold,
                'metric': metric,
                'A': valA, 'B': valB, 'C': valC,
                'B_minus_A': valB - valA,
                'C_minus_A': valC - valA,
                'C_minus_B': valC - valB
            })
            
    paired_diff_df = pd.DataFrame(paired_fold_diffs)
    paired_diff_df.to_csv(TABLES_DIR / "paired_fold_differences.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 3. STATISTICAL COMPARISON RESULTS
    # -------------------------------------------------------------------------
    stats_results = []
    comparisons = [('A', 'C', 'C_minus_A'), ('A', 'B', 'B_minus_A'), ('B', 'C', 'C_minus_B')]
    
    for cond1, cond2, diff_col in comparisons:
        for metric in metrics:
            diffs = paired_diff_df[paired_diff_df['metric'] == metric][diff_col].values
            
            mean_d = np.nanmean(diffs)
            median_d = np.nanmedian(diffs)
            std_d = np.nanstd(diffs, ddof=1)
            
            try:
                t_stat, p_t = stats.ttest_1samp(diffs[~np.isnan(diffs)], 0.0)
            except:
                p_t = np.nan
                
            try:
                w_stat, p_w = stats.wilcoxon(diffs[~np.isnan(diffs)])
            except:
                p_w = np.nan
                
            pos_sign = np.sum(diffs > 0)
            neg_sign = np.sum(diffs < 0)
            exact_sign = f"+{pos_sign}, -{neg_sign}, = {np.sum(diffs == 0)}"
            
            _, ci_lower, ci_upper = mean_confidence_interval(diffs[~np.isnan(diffs)])
            
            effect_size = mean_d / std_d if std_d > 0 else np.nan # Cohen's d
            
            stats_results.append({
                'comparison': f"{cond1}_vs_{cond2}",
                'metric': metric,
                'mean_diff': mean_d, 'median_diff': median_d, 'std_diff': std_d,
                'ci_lower_95': ci_lower, 'ci_upper_95': ci_upper,
                'paired_t_pval': p_t, 'wilcoxon_pval': p_w,
                'exact_sign_direction': exact_sign,
                'cohens_d': effect_size
            })
            
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(TABLES_DIR / "statistical_comparison_results.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 6 & 7. PATIENT-OVERLAP EFFECT BY TEST PATIENT & INTENSITY
    # -------------------------------------------------------------------------
    patient_analysis = []
    
    for fold in range(1, 6):
        # Load train manifests to count images
        train_A = pd.read_csv(SPLITS_DIR / f"fold{fold}_A_train.csv")
        train_B = pd.read_csv(SPLITS_DIR / f"fold{fold}_B_train.csv")
        train_C = pd.read_csv(SPLITS_DIR / f"fold{fold}_C_train.csv")
        
        train_counts_A = train_A.groupby('PID').size().to_dict()
        train_counts_B = train_B.groupby('PID').size().to_dict()
        train_counts_C = train_C.groupby('PID').size().to_dict()
        
        preds_A = full_preds[(full_preds['fold'] == fold) & (full_preds['condition'] == 'A')]
        preds_B = full_preds[(full_preds['fold'] == fold) & (full_preds['condition'] == 'B')]
        preds_C = full_preds[(full_preds['fold'] == fold) & (full_preds['condition'] == 'C')]
        
        for df, cond, tc in [(preds_A, 'A', train_counts_A), (preds_B, 'B', train_counts_B), (preds_C, 'C', train_counts_C)]:
            for _, row in df.iterrows():
                pid = row['PID']
                patient_analysis.append({
                    'fold': fold,
                    'PID': pid,
                    'class': row['label'],
                    'condition': cond,
                    'images_in_training': tc.get(pid, 0),
                    'predicted_label': row['predicted_label'],
                    'true_label': row['true_label'],
                    'confidence': row['max_confidence'],
                    'correct': row['correct']
                })
                
    patient_df = pd.DataFrame(patient_analysis)
    patient_df.to_csv(TABLES_DIR / "patient_overlap_prediction_analysis.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 9. HIGH-CONFIDENCE ERROR ANALYSIS
    # -------------------------------------------------------------------------
    hce_list = []
    for cond in ['A', 'B', 'C']:
        cond_df = full_preds[full_preds['condition'] == cond]
        total = len(cond_df)
        
        errors = cond_df[cond_df['correct'] == 0]
        err_90 = errors[errors['max_confidence'] >= 0.90]
        err_95 = errors[errors['max_confidence'] >= 0.95]
        
        for _, r in err_90.iterrows():
            hce_list.append({
                'fold': r['fold'], 'condition': cond,
                'PID': r['PID'], 'image_id': r['image_id'],
                'true_class': r['true_label'], 'predicted_class': r['predicted_label'],
                'confidence': r['max_confidence']
            })
            
    hce_df = pd.DataFrame(hce_list)
    hce_df.sort_values(by=['condition', 'fold', 'confidence'], ascending=[True, True, False], inplace=True)
    hce_df.to_csv(TABLES_DIR / "high_confidence_errors.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 5 & 8. DOSE-RESPONSE FIGURES & RELIABILITY DIAGRAMS
    # -------------------------------------------------------------------------
    x = [0, 50, 100]
    labels = ['A (0%)', 'B (50%)', 'C (100%)']
    
    def plot_dose(metric_col, title, ylabel, fname):
        plt.figure(figsize=(8,6))
        for fold in range(1, 6):
            fold_data = complete_fold_cond_df[complete_fold_cond_df['fold'] == fold].set_index('condition').loc[['A', 'B', 'C']]
            plt.plot(x, fold_data[metric_col].values, color='gray', alpha=0.3, linestyle='--', marker='o')
        
        pooled_data = complete_fold_cond_df.groupby('condition')[metric_col].mean().loc[['A', 'B', 'C']]
        plt.plot(x, pooled_data.values, color='blue', linewidth=3, marker='s', markersize=8, label='Pooled Mean')
        
        plt.title(title, fontsize=14)
        plt.xlabel("Patient Overlap (%)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(x, labels)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(FIG_DIR / fname, dpi=300, bbox_inches='tight')
        plt.close()
        
    plot_dose('accuracy', 'Dose-Response: Accuracy vs Overlap', 'Accuracy', 'dose_response_accuracy.png')
    plot_dose('macro_f1', 'Dose-Response: Macro F1 vs Overlap', 'Macro F1', 'dose_response_f1.png')
    plot_dose('mean_confidence', 'Dose-Response: Mean Confidence vs Overlap', 'Confidence', 'dose_response_confidence.png')
    plot_dose('ece', 'Dose-Response: Expected Calibration Error vs Overlap', 'ECE', 'dose_response_ece.png')
    plot_dose('brier_score', 'Dose-Response: Brier Score vs Overlap', 'Brier Score', 'dose_response_brier.png')
    plot_dose('hcer_90', 'Dose-Response: HCER (>=0.90) vs Overlap', 'High-Confidence Error Rate', 'dose_response_hcer.png')
    
    # Reliability diagram
    plt.figure(figsize=(8,8))
    n_bins = 15
    for cond, color in zip(['A', 'B', 'C'], ['blue', 'orange', 'green']):
        df_cond = full_preds[full_preds['condition'] == cond]
        bins = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        for i in range(n_bins):
            in_bin = df_cond[(df_cond['max_confidence'] > bins[i]) & (df_cond['max_confidence'] <= bins[i+1])]
            if i == 0:
                in_bin = df_cond[(df_cond['max_confidence'] >= bins[i]) & (df_cond['max_confidence'] <= bins[i+1])]
            if len(in_bin) > 0:
                bin_accs.append(in_bin['correct'].mean())
                bin_confs.append(in_bin['max_confidence'].mean())
        plt.plot(bin_confs, bin_accs, marker='o', label=f'Cond {cond}', color=color)
        
    plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram (A vs B vs C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIG_DIR / "reliability_A_B_C_final.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Secondary analysis complete.")

if __name__ == "__main__":
    main()
