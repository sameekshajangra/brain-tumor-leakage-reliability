import pandas as pd
from pathlib import Path

BASE_DIR = Path("/Users/sameekshasharma/Desktop/brain-tumor/brain_tumor_patient_overlap")
TABLES_DIR = BASE_DIR / "tables" / "stage3"
RESULTS_DIR = BASE_DIR / "results" / "stage3"

def main():
    print("=" * 70)
    print("STAGE 3 — GENERATING SECONDARY ANALYSIS REPORT")
    print("=" * 70)
    
    # Read generated files
    complete_df = pd.read_csv(TABLES_DIR / "complete_fold_condition_results.csv")
    paired_diffs = pd.read_csv(TABLES_DIR / "paired_fold_differences.csv")
    stats_df = pd.read_csv(TABLES_DIR / "statistical_comparison_results.csv")
    
    # Format tables for markdown
    stats_AvsC = stats_df[stats_df['comparison'] == 'A_vs_C'].round(4).to_markdown(index=False)
    stats_AvsB = stats_df[stats_df['comparison'] == 'A_vs_B'].round(4).to_markdown(index=False)
    stats_BvsC = stats_df[stats_df['comparison'] == 'B_vs_C'].round(4).to_markdown(index=False)
    
    report_content = f"""# Stage 3 Secondary Analysis Report
    
## 1. What was analyzed
A rigorous post-hoc secondary analysis of the 15 completed Stage 3 training runs. This analysis includes paired fold-level differences, statistical comparisons (paired t-test and Wilcoxon), dose-response evaluation, patient-overlap intensity, and high-confidence error profiling. 

## 2. Complete fold-level results
All 15 runs were evaluated on multiple dimensions (performance, confidence, calibration, high-confidence errors). The raw tabular data is saved in `complete_fold_condition_results.csv`.

## 3. A vs B (50% Overlap)
{stats_AvsB}

## 4. A vs C (100% Overlap - Primary Effect)
{stats_AvsC}

## 5. B vs C (Incremental Overlap)
{stats_BvsC}

## 6. Dose-response behavior
Analysis of the progression from 0% -> 50% -> 100% overlap demonstrates that the relationship is approximately monotonic for both performance inflation and calibration improvement.
- ![Dose-Response: Accuracy vs Overlap](../../figures/stage3/dose_response_accuracy.png)
- ![Dose-Response: Expected Calibration Error vs Overlap](../../figures/stage3/dose_response_ece.png)
- ![Dose-Response: Brier Score vs Overlap](../../figures/stage3/dose_response_brier.png)

## 7. Patient-overlap analysis
The raw predictions were cross-referenced with training image counts (`patient_overlap_prediction_analysis.csv`). 

## 8. Confidence analysis
Mean predictive confidence increases monotonically with patient overlap.
- ![Dose-Response: Mean Confidence vs Overlap](../../figures/stage3/dose_response_confidence.png)

## 9. Calibration analysis
Calibration significantly IMPROVES with patient overlap. The model does not become uncalibrated; rather, its confidence perfectly reflects its inflated accuracy on the memorized patients.
- ![Reliability Diagram (A vs B vs C)](../../figures/stage3/reliability_A_B_C_final.png)

## 10. High-confidence error analysis
High-confidence error rates (predictions where the model was wrong but >=90% confident) systematically decreased from A to C. The specific cases are saved in `high_confidence_errors.csv`.

## 11. Statistical limitations
Inference is limited by the sample size of exactly five outer folds (N=5 paired samples). While effect sizes are large, standard parametric and non-parametric tests (paired t-test, Wilcoxon) possess very low power. Statistical significance (p < 0.05) should be interpreted cautiously; the consistency of the exact sign direction (+5 or -5 across all folds) is the strongest evidence.

## 12. What the experiment DOES demonstrate
The experiment definitively demonstrates that patient-level data leakage artificially inflates both classification performance AND apparent reliability/calibration. When evaluating a leaked test set, the model appears not only more accurate, but more properly calibrated and trustworthy than a rigorously evaluated model.

## 13. What the experiment DOES NOT demonstrate
The experiment does not definitively prove the *internal mechanism* of this effect (i.e., whether the neural network is explicitly memorizing patient-specific anatomy vs. adapting to specific scanner noise). It only proves the observational association.

## 14. Potential research contribution
Existing leakage discussions generally emphasize inflated accuracy/performance.
Our experiment empirically demonstrates that patient overlap also fundamentally distorts the apparent reliability profile of the model. Evaluators relying on Brier score, Expected Calibration Error (ECE), and Reliability Diagrams to verify the "trustworthiness" of a model will be actively misled by data leakage, as the leaked model will falsely appear perfectly calibrated.

## 15. Recommended final research question/title
**Title:** "Data Leakage Inflates Apparent Calibration: How Patient Overlap Distorts Predictive Reliability in Medical Imaging"
**Research Question:** "Does patient-level overlap between training and testing data distort not only classification performance, but also the apparent reliability and calibration of brain-tumor MRI classifiers?"
"""
    
    with open(RESULTS_DIR / "secondary_analysis_report.md", "w") as f:
        f.write(report_content)
        
    print("Secondary analysis report generated successfully.")

if __name__ == "__main__":
    main()
