import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
TABLES_DIR = BASE_DIR / "tables" / "stage3"
RESULTS_DIR = BASE_DIR / "results" / "stage3"

def main():
    print("=" * 70)
    print("STAGE 3 — GENERATING FINAL REPORT")
    print("=" * 70)
    
    pooled_perf = pd.read_csv(TABLES_DIR / "pooled_performance.csv").set_index('condition')
    pooled_cal = pd.read_csv(TABLES_DIR / "pooled_calibration.csv").set_index('condition')
    diff_df = pd.read_csv(TABLES_DIR / "paired_condition_differences.csv")
    
    # Format the table for the report
    perf_table = pooled_perf[['accuracy', 'macro_f1', 'balanced_accuracy']].round(4).to_markdown()
    cal_table = pooled_cal[['mean_confidence', 'ece', 'brier_score', 'hcer_90']].round(4).to_markdown()
    
    diff_table = diff_df.pivot(index='metric', columns='comparison', values='mean_diff').round(4).to_markdown()
    pval_table = diff_df.pivot(index='metric', columns='comparison', values='paired_t_pval').round(4).to_markdown()
    
    report_content = f"""# Stage 3 Final Research Report
    
## 1. Research Question
How does patient-level image overlap between training and testing data affect classification performance, predictive confidence, and calibration in brain tumor MRI classification?

## 2. Hypothesis
Increasing patient overlap between the training and testing sets will artificially inflate classification performance (accuracy, F1) while simultaneously increasing predictive confidence and worsening calibration (higher ECE, higher Brier score, and higher high-confidence error rates).

## 3. Dataset
The Cheng et al. Brain Tumor MRI Dataset containing 3,064 images from 233 patients.

## 4. Dataset Verification
All metadata mappings, labels, and PID assignments were strictly verified in Stage 1. 233 unique PIDs and 3,064 matching images were confirmed.

## 5. Experimental Design
A 5-fold cross-validation experiment was conducted. 
Three conditions were compared:
- **Condition A**: 0% patient overlap (Strict independence)
- **Condition B**: ~50% of eligible test-fold patients overlapped into training
- **Condition C**: 100% of eligible test-fold patients overlapped into training

## 6. Patient-overlap construction
The 20-rule split integrity protocol verified that all fixed test images were rigorously excluded from training and validation sets across all conditions. Overlap solely comprised the non-test images of eligible test-fold patients.

## 7. Model and training configuration
EfficientNet-B0 (ImageNet initialization), AdamW optimizer (LR=1e-4), CrossEntropyLoss. Trained for max 50 epochs with early stopping (patience=10) on validation loss. Device: Apple MPS.

## 8. Five-fold evaluation methodology
Each of the 15 models (5 folds × 3 conditions) was evaluated EXCLUSIVELY on its corresponding frozen outer-fold test set.

## 9. Performance Results (Pooled)
{perf_table}

## 10. Calibration Results (Pooled)
{cal_table}

## 11. Statistical Comparisons (Mean Differences)
{diff_table}

### Paired T-Test P-Values
{pval_table}

## 12. Figures
The following figures summarize the relationship between overlap and performance/calibration:
- ![Accuracy vs Overlap](../../figures/stage3/accuracy_vs_overlap.png)
- ![Macro F1 vs Overlap](../../figures/stage3/macro_f1_vs_overlap.png)
- ![Expected Calibration Error (ECE) vs Overlap](../../figures/stage3/ece_vs_overlap.png)
- ![Brier Score vs Overlap](../../figures/stage3/brier_score_vs_overlap.png)
- ![Mean Confidence vs Overlap](../../figures/stage3/mean_confidence_vs_overlap.png)

## 13. Limitations
- The dataset size is relatively small (233 patients), meaning fold-level variance can be high.
- The evaluation was conducted on a single architecture (EfficientNet-B0).
- The overlap simulation uses multiple slices from the same 2D MRI volumes, representing a specific type of data leakage.

## 14. Interpretation
(To be written by the user based on the empirical results above.)

## 15. Reproducibility
All deterministic split generation manifests, training logs, checkpoint references, and evaluation predictions have been saved. The pipeline can be fully reproduced by re-running Stages 1-3 sequentially.

## 16. Conclusion
This study empirically quantifies the impact of data leakage (patient overlap) on both accuracy and reliability metrics in medical imaging.
"""
    
    with open(RESULTS_DIR / "stage3_final_report.md", "w") as f:
        f.write(report_content)
        
    print("Final report generated successfully at results/stage3/stage3_final_report.md")

if __name__ == "__main__":
    main()
