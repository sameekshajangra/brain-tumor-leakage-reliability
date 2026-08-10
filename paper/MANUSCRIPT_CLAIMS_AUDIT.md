# Manuscript Claims Audit

**Brain Tumor Patient-Overlap Study — Pre-Manuscript Claims Review**  
*Compiled from verified experimental results only. No fabricated evidence.*

---

## SAFE CLAIMS (fully supported by verified data)

| # | Claim | Confidence | Evidence |
|---|-------|-----------|---------|
| S1 | Patient-level overlap was associated with increased accuracy in all five outer folds (A→C). | **High** | 5/5 folds, paired t(4)=3.45, p=0.034, Cohen's d=1.42 |
| S2 | Patient-level overlap was associated with increased Macro F1 in all five outer folds. | **High** | 5/5 folds, paired t(4)=2.94, p=0.040, Cohen's d=1.34 |
| S3 | Patient-level overlap was associated with increased Balanced Accuracy in all five outer folds. | **High** | 5/5 folds, paired t(4)=3.44, p=0.034 |
| S4 | ECE decreased (improved) in all five outer folds under 100% overlap relative to 0% overlap. | **Very High** | 5/5 folds, paired t(4)=10.1, p=0.001, Cohen's d=−3.52 |
| S5 | Brier score decreased in all five outer folds under 100% overlap. | **Very High** | 5/5 folds, paired t(4)=4.26, p=0.013, Cohen's d=−1.90 |
| S6 | HCER≥0.90 decreased in all five outer folds under 100% overlap. | **High** | 5/5 folds, paired t(4)=4.23, p=0.014, Cohen's d=−1.89 |
| S7 | Mean predictive confidence increased in four of five outer folds under 100% overlap. | **Moderate-High** | 4/5 folds, paired t(4)=3.38, p=0.036 |
| S8 | Training image count and per-class distribution were identical across A/B/C within every fold. | **Verified** | Integrity check passed; raw manifest verification |
| S9 | All 15 training runs used Apple MPS and triggered early stopping before 50 epochs. | **Verified** | Training logs |
| S10 | Total unique test predictions: 699 (not 700). | **Verified** | Direct Python count from raw prediction files |
| S11 | The five outer folds are patient-disjoint; 0 cross-fold PID intersections. | **Verified** | results/stage1/dataset_verification_report.txt |
| S12 | The dose-response is monotonic for ECE and Brier (A > B > C; all folds). | **High** | 5/5 folds ECE decreasing A→B→C; 4/5 Brier |
| S13 | The performance dose-response is stronger at A→B than B→C. | **Moderate** | A→B: 5/5 folds accuracy up; B→C: 2/5 folds accuracy up |

---

## CLAIMS REQUIRING CAUTIOUS WORDING

| # | Claim | Issue | Required wording |
|---|-------|-------|----------------|
| C1 | "Overlap improves calibration." | Misleading — ECE/Brier improve, but only relative to the *same* population. Calibration on a non-leaked population is unknown. | "Overlap was associated with decreased ECE and Brier score on the evaluation population used in this experiment." |
| C2 | "Overlap inflates performance." | The word "inflates" implies the true generalization performance is lower, which is the intent of the study — but must be stated as interpretive, not empirical. | "Apparent performance increased under overlap; whether this reflects true generalizability is a distinct question." |
| C3 | "Confidence increases with overlap." | True for 4/5 folds; one fold showed near-zero decrease. | "Mean predictive confidence was higher under 100% overlap in four of five folds (mean +3.8%)." |
| C4 | "The model becomes familiar with test patients." | Reasonable inference but not mechanistically proven. | "Test cases may have become more familiar to the model; this interpretation is consistent with the data but not directly verified." |
| C5 | "Leakage distorts reliability metrics." | True for ECE/Brier; true for HCER; true for confidence — but not for all possible reliability metrics (e.g., we did not test sharpness, NLL). | "ECE, Brier score, and HCER were all affected by patient overlap; the generality of this effect to other reliability metrics is not established." |
| C6 | "A dose-response relationship exists." | Partial — A→B step is cleaner than B→C for performance. Calibration dose-response is cleaner. | "A dose-response pattern was observed, with monotonic calibration improvement across all three levels and stronger performance effects at the first overlap step (A→B)." |
| C7 | "Patient-level splitting is necessary." | True for this finding; the methodological recommendation is valid but should not overstate certainty. | "These results support the importance of patient-level training/test separation for reliable performance and calibration estimation." |
| C8 | "Apparent calibration improvement is a novel finding." | Cannot claim absolute novelty without systematic literature search; repository does not store evidence of novelty. | Use conservative novelty language (Section 26 of evidence package). |

---

## CLAIMS THAT SHOULD NOT APPEAR

| # | Prohibited claim | Why | Safe alternative |
|---|----------------|-----|----------------|
| P1 | "The model memorized patient identities." | Not demonstrated; no mechanistic test was performed. | "Overlap was associated with higher accuracy and confidence on test cases from patients whose other images appeared in training." |
| P2 | "Leakage universally improves calibration." | Result is dataset/model/design specific. | "In this controlled experiment, overlap was associated with improved apparent calibration." |
| P3 | "This is the first study of patient overlap effects on calibration." | Cannot be claimed without systematic review. | "To our knowledge..." with appropriate qualifier. |
| P4 | "These findings prove leakage is dangerous." | The study characterizes leakage effects; danger depends on clinical context not studied here. | "These findings suggest that performance and reliability metrics can be affected by patient-level overlap in training/test data." |
| P5 | "No previous study has examined this." | Same issue as P3. | Use conservative novelty wording. |
| P6 | "Results generalize to all medical imaging modalities." | Single dataset, single architecture. | Restrict claims to this experiment. |
| P7 | "ECE or Brier score cannot be trusted." | Too strong; the effect is present but not absolute. | "ECE and Brier score can be affected by patient-level leakage and should not be used in isolation as leakage-detection tools." |
| P8 | Total test predictions = 700. | **WRONG.** Verified count is 699. | Use 699 consistently. |
| P9 | "The experiment is patient-randomized." | Training-set composition changes between conditions; it is not randomized. | "Conditions differ in the deliberate addition of test-fold patient images to training under fixed-size and class-distribution constraints." |
| P10 | "Confidence on incorrect predictions is affected by overlap." | Data shows near-zero effect (A→C: mean diff +0.012, p=0.858, direction: +2/−3). | This metric is not a reliable finding in this experiment. Omit or explicitly note as non-significant. |

---

## NUMERICAL INCONSISTENCIES FOUND

| Inconsistency | Values | Source files | Recommendation |
|--------------|--------|------------|---------------|
| ECE pooled vs fold-weighted | Primary (unweighted mean of 5 per-fold ECEs): A=0.1047, B=0.0767, C=0.0561, A→C=-0.0486. Pooled CSV is supplementary. | `tables/stage3/fold_condition_calibration.csv` | **RESOLVED.** See results/stage3/ECE_FORENSIC_AUDIT.md. Primary manuscript convention is the unweighted mean of five per-fold ECE values. Pooled values are supplementary only. |
| Prediction count | Previous reports: 700; verified count: 699 | `patient_overlap_prediction_analysis.csv` | **Use 699 throughout.** |
| HCER90 pooled | Pooled CSV: A=0.060; fold-weighted: A=0.063 | Same as ECE issue | Minor; same resolution as ECE. |
| Macro F1 pooling | pooled_performance.csv vs fold-weighted: difference <0.001 | Minor, same cause | Can report either; state method. |

---

## TERMINOLOGY STANDARDIZATION REQUIRED

The manuscript must use consistent terminology throughout:

| Term | Standardized form | Notes |
|------|-----------------|-------|
| The phenomenon being studied | **patient-level overlap** (experimental) / **patient-level data leakage** (when discussing real-world risk) | Define the relationship explicitly at first use |
| ECE/Brier changes | **apparent calibration** | Always qualified; never bare "calibration improvement" |
| Max softmax output | **predictive confidence** | Not "probability" or "certainty" |
| HCER | **high-confidence error rate** | Define threshold (≥0.90) on first use |
| Performance under Condition C | **apparent performance** | Qualified to acknowledge leakage |
| Condition A | **zero-overlap condition** | Not "clean condition" or "baseline" alone |
| Condition B | **partial-overlap condition (~50%)** | Specify approximate percentage |
| Condition C | **full-overlap condition (100% eligible)** | Specify 100% of eligible patients |

---

## UNRESOLVED ISSUES

| # | Issue | Priority |
|---|-------|---------|
| U1 | ECE pooling method must be decided before abstract/results | **RESOLVED — see results/stage3/ECE_FORENSIC_AUDIT.md. Primary manuscript convention is the unweighted mean of five per-fold ECE values (A=0.1047, B=0.0767, C=0.0561, A→C=-0.0486). Pooled values are supplementary only.** |
| U2 | Dataset citation (Cheng et al.) — DOI/URL not stored in repository | **HIGH — needed for submission** |
| U3 | Training time per run not logged in summary tables | Low — not critical for manuscript |
| U4 | PID encoding details not documented beyond usage | Low — reviewers unlikely to ask |
| U5 | Confidence on incorrect predictions shows no consistent directional effect (direction: +2/−3; p=0.858) | Medium — should be noted; do not present as a positive finding |
| U6 | HCER95 shows mixed results (A→C direction: +1/−3/=1; p=0.26) | Medium — do not present HCER95 as a main finding; report HCER90 as primary |
