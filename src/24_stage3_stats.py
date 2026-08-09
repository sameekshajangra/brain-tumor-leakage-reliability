import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
TABLES_DIR = BASE_DIR / "tables" / "stage3"
FIG_DIR = BASE_DIR / "figures" / "stage3"
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("STAGE 3 — STATISTICAL COMPARISONS & FIGURES")
    print("=" * 70)
    
    perf_df = pd.read_csv(TABLES_DIR / "fold_condition_performance.csv")
    cal_df = pd.read_csv(TABLES_DIR / "fold_condition_calibration.csv")
    
    # Merge performance and calibration metrics
    df = pd.merge(perf_df, cal_df, on=['fold', 'condition'])
    
    metrics = [
        'accuracy', 'macro_f1', 'balanced_accuracy',
        'mean_confidence', 'conf_correct', 'conf_incorrect',
        'ece', 'brier_score', 'hcer_90'
    ]
    
    diff_rows = []
    
    # Extract data for A, B, C for paired comparisons
    condA = df[df['condition'] == 'A'].set_index('fold')
    condB = df[df['condition'] == 'B'].set_index('fold')
    condC = df[df['condition'] == 'C'].set_index('fold')
    
    comparisons = [('B', 'A', condB, condA), ('C', 'A', condC, condA), ('C', 'B', condC, condB)]
    
    for cond1_name, cond2_name, df1, df2 in comparisons:
        for metric in metrics:
            val1 = df1[metric].values
            val2 = df2[metric].values
            
            diff = val1 - val2
            
            mean_diff = np.nanmean(diff)
            std_diff = np.nanstd(diff, ddof=1)
            median_diff = np.nanmedian(diff)
            min_diff = np.nanmin(diff)
            max_diff = np.nanmax(diff)
            
            # Paired t-test
            try:
                t_stat, p_val_t = stats.ttest_rel(val1, val2, nan_policy='omit')
            except:
                t_stat, p_val_t = np.nan, np.nan
                
            # Wilcoxon signed-rank
            try:
                w_stat, p_val_w = stats.wilcoxon(val1, val2)
            except:
                w_stat, p_val_w = np.nan, np.nan
                
            diff_rows.append({
                'comparison': f"{cond1_name} - {cond2_name}",
                'metric': metric,
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'median_diff': median_diff,
                'min_diff': min_diff,
                'max_diff': max_diff,
                'paired_t_pval': p_val_t,
                'wilcoxon_pval': p_val_w
            })
            
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(TABLES_DIR / "paired_condition_differences.csv", index=False)
    
    print("Statistical differences calculated and saved.")
    
    # ---------------------------------------------------------
    # GENERATE FIGURES
    # ---------------------------------------------------------
    # Pooled performance for simple line plots
    pooled_perf = pd.read_csv(TABLES_DIR / "pooled_performance.csv")
    pooled_cal = pd.read_csv(TABLES_DIR / "pooled_calibration.csv")
    pooled = pd.merge(pooled_perf, pooled_cal, on=['condition', 'test_images'])
    
    cond_labels = ['A (0%)', 'B (50%)', 'C (100%)']
    x = [0, 50, 100]
    
    # Plot definitions
    plot_defs = [
        ('accuracy', 'Accuracy vs Overlap', 'Accuracy', 'accuracy_vs_overlap.png'),
        ('macro_f1', 'Macro F1 vs Overlap', 'Macro F1', 'macro_f1_vs_overlap.png'),
        ('balanced_accuracy', 'Balanced Accuracy vs Overlap', 'Balanced Accuracy', 'balanced_accuracy_vs_overlap.png'),
        ('mean_confidence', 'Mean Confidence vs Overlap', 'Mean Confidence', 'mean_confidence_vs_overlap.png'),
        ('conf_correct', 'Confidence (Correct) vs Overlap', 'Mean Confidence', 'conf_correct_vs_overlap.png'),
        ('conf_incorrect', 'Confidence (Incorrect) vs Overlap', 'Mean Confidence', 'conf_incorrect_vs_overlap.png'),
        ('ece', 'Expected Calibration Error (ECE) vs Overlap', 'ECE', 'ece_vs_overlap.png'),
        ('brier_score', 'Brier Score vs Overlap', 'Brier Score', 'brier_score_vs_overlap.png')
    ]
    
    for metric, title, ylabel, filename in plot_defs:
        plt.figure(figsize=(8, 6))
        
        # Plot Fold-level data as light lines
        for fold in range(1, 6):
            fold_data = df[df['fold'] == fold].set_index('condition').loc[['A', 'B', 'C']]
            plt.plot(x, fold_data[metric].values, color='gray', alpha=0.3, linestyle='--', marker='o')
            
        # Plot Pooled data as bold line
        pooled_data = pooled.set_index('condition').loc[['A', 'B', 'C']]
        plt.plot(x, pooled_data[metric].values, color='blue', linewidth=3, marker='s', markersize=8, label='Pooled (All Folds)')
        
        plt.title(title, fontsize=14)
        plt.xlabel("Patient Overlap (%)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(x, cond_labels)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(FIG_DIR / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
    print("Primary figures generated and saved.")

if __name__ == "__main__":
    main()
