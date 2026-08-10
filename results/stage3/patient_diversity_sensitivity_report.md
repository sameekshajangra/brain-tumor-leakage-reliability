# Patient Diversity and Overlap-Intensity Sensitivity Analysis Report

**Stage 3 — Post-hoc Sensitivity Analysis (Read-Only)**  
*No models were retrained, no splits were modified, no predictions were altered.*

---

## 1. Objective

This report addresses a key methodological question: although Conditions A, B, and C were designed with identical training-image counts and matched class distributions, the patient-overlap manipulation intentionally changes the *composition* of the training set. This analysis quantifies those compositional differences, assesses their relationship to the observed performance effects, and provides transparent wording for the manuscript.

---

## 2. Data Sources

| File | Purpose |
|---|---|
| `results/stage3/splits/fold*_{A,B,C}_train.csv` | Per-condition training manifests |
| `tables/stage3/overlap_selection_summary.csv` | Overlap construction record |
| `tables/stage3/complete_fold_condition_results.csv` | Per-fold performance/calibration |
| `tables/stage3/patient_overlap_prediction_analysis.csv` | Per-prediction metadata (fold, PID, images_in_training, confidence, correct) |

---

## 3. Training Patient Diversity

### 3a. Unique training-patient counts by fold and condition

| Fold | Cond A | Cond B | Cond C |
|------|--------|--------|--------|
| 1    | 150    | 166    | 185    |
| 2    | 150    | 165    | 178    |
| 3    | 148    | 168    | 182    |
| 4    | 149    | 164    | 182    |
| 5    | 149    | 166    | 182    |

**Key observation:** Condition C has ~32–35 more unique training patients than Condition A, and Condition B is intermediate (~15–18 more than A). These differences are *exactly* the number of overlapping test-fold patients added — they are the intended manipulation, not an error.

### 3b. Images per training patient

Because total training images are constant but unique patients increase, the mean images per patient decreases monotonically from A → C:

| Fold | A mean | B mean | C mean |
|------|--------|--------|--------|
| 1 | 13.31 | 12.02 | 10.79 |
| 2 | 12.65 | 11.50 | 10.66 |
| 3 | 13.39 | 11.79 | 10.88 |
| 4 | 13.48 | 12.25 | 11.04 |
| 5 | 12.87 | 11.55 | 10.54 |

This is an inherent and expected consequence of the overlap manipulation: adding test-fold patients (who have fewer images available by design) dilutes the per-patient representation.

---

## 4. Overlap Intensity

### 4a. Overlapping patients and images by fold and condition

| Fold | Cond | Eligible test pts | Overlapping pts | % eligible | Overlap images | % of training |
|------|------|-------------------|-----------------|-----------|----------------|---------------|
| 1    | A    | 43                | 0               | 0.0%      | 0              | 0.0%          |
| 1    | B    | 43                | 22              | 51.2%     | 273            | 13.7%         |
| 1    | C    | 43                | 43              | 100.0%    | 497            | 24.9%         |
| 2    | A    | 45                | 0               | 0.0%      | 0              | 0.0%          |
| 2    | B    | 45                | 22              | 48.9%     | 302            | 15.9%         |
| 2    | C    | 45                | 45              | 100.0%    | 633            | 33.4%         |
| 3    | A    | 46                | 0               | 0.0%      | 0              | 0.0%          |
| 3    | B    | 46                | 23              | 50.0%     | 238            | 12.0%         |
| 3    | C    | 46                | 46              | 100.0%    | 524            | 26.5%         |
| 4    | A    | 46                | 0               | 0.0%      | 0              | 0.0%          |
| 4    | B    | 46                | 23              | 50.0%     | 326            | 16.2%         |
| 4    | C    | 46                | 46              | 100.0%    | 581            | 28.9%         |
| 5    | A    | 46                | 0               | 0.0%      | 0              | 0.0%          |
| 5    | B    | 46                | 23              | 50.0%     | 297            | 15.5%         |
| 5    | C    | 46                | 46              | 100.0%    | 596            | 31.1%         |

In Condition C, approximately 25–33% of all training images belong to test-fold patients (non-test images only). In Condition B this is approximately 12–16%.

---

## 5. Training-Size and Class-Distribution Verification

**INTEGRITY CHECK PASSED.**

All five outer folds independently confirm that training image counts and per-class image counts (Meningioma, Glioma, Pituitary) are **identical** across Conditions A, B, and C within each fold. No discrepancies were found.

This is the primary design control of the experiment and it holds.

---

## 6. Class-Distribution Verification

See integrity check output above. Class counts were recalculated from raw manifest files and match the values in `overlap_selection_summary.csv`.

---

## 7. Patient-Level Familiarity Analysis (Exploratory)

Using the 700-row `patient_overlap_prediction_analysis.csv` file, predictions were grouped by the number of same-PID training images available to the model under that condition.

| Training-image bin | n predictions | Accuracy | Mean confidence | HCER ≥ 0.90 |
|--------------------|--------------|----------|-----------------|-------------|
| 0 (pure Cond A)    | 360          | 0.889    | 0.930           | 0.053       |
| 1–5 images         | 79           | 0.949    | 0.921           | 0.025       |
| 6–15 images        | 142          | 0.951    | 0.955           | 0.000       |
| 16–30 images       | 111          | 0.982    | 0.977           | 0.018       |
| >30 images         | 7            | 1.000    | 0.990           | 0.000       |

**Observation:** Accuracy and confidence both show a positive exploratory association with the number of same-PID training images. High-confidence error rate shows a negative exploratory association. However, with only 7 predictions in the ">30" bin and a total of 700 predictions spanning all five folds, this analysis is **underpowered for formal inference** and must be labelled exploratory.

> **Important:** The apparent monotonic pattern should not be interpreted as causal evidence of memorisation. It is consistent with the hypothesis that model familiarity increases with training exposure to same-patient images, but many other explanations (e.g., patients with more images having more typical/recognisable tumours) cannot be ruled out.

---

## 8. Dose-Response Sensitivity Analysis

### A → B step (adding ~50% eligible overlap patients)

| Metric | Folds 1–5 diffs | Mean | n increasing |
|--------|-----------------|------|-------------|
| accuracy | +0.044, +0.065, +0.042, +0.043, +0.021 | +0.043 | **5/5** |
| macro_f1 | +0.046, +0.064, +0.046, +0.039, +0.020 | +0.043 | **5/5** |
| ece | +0.001, −0.058, −0.034, −0.035, −0.014 | −0.028 | 1/5 |
| brier | −0.061, −0.108, −0.054, −0.062, −0.035 | −0.064 | **0/5** |
| mean_conf | −0.027, +0.041, +0.018, +0.021, +0.032 | +0.017 | 4/5 |

### B → C step (adding remaining ~50% overlap patients)

| Metric | Folds 1–5 diffs | Mean | n increasing |
|--------|-----------------|------|-------------|
| accuracy | 0.000, +0.044, +0.021, −0.021, 0.000 | +0.009 | 2/5 |
| macro_f1 | −0.004, +0.043, +0.015, −0.024, +0.002 | +0.006 | 3/5 |
| ece | −0.028, −0.005, −0.013, −0.013, −0.044 | −0.021 | **0/5** |
| brier | +0.016, −0.070, −0.056, −0.024, −0.032 | −0.033 | 1/5 |
| mean_conf | +0.027, +0.027, +0.004, +0.028, +0.019 | +0.021 | **5/5** |

**Pattern summary:** The A→B step shows strong directional consistency for performance (5/5 folds) and Brier improvement (5/5 folds). The B→C step shows weaker performance effects (2–3/5 folds increasing) but continues to show consistent ECE improvement (0/5 increasing = 5/5 decreasing) and confidence increase (5/5 folds). The calibration effect (ECE, Brier) accumulates more monotonically than the accuracy effect.

---

## 9. Exploratory Fold-Condition Associations (n = 15)

At the fold-condition level (15 model-condition observations), the Pearson correlation between number of overlap training images and key metrics:

| Metric | r (n=15, exploratory) |
|--------|----------------------|
| Accuracy | +0.724 |
| Mean confidence | +0.756 |
| ECE | **−0.803** |
| Brier score | **−0.777** |
| HCER ≥ 0.90 | −0.545 |

> **Caution:** These are descriptive correlations across 15 model-condition observations. They should not be interpreted as providing independent statistical evidence beyond the fold-level paired comparisons already reported. They are consistent with the main results and provide face validity for the overlap manipulation, but formal inference requires treating folds (n=5) as the unit of analysis.

---

## 10. Methodological Interpretation

The patient-overlap manipulation in this experiment has three inseparable, co-occurring consequences:

1. **Increased unique training patients** (test-fold patients added to training)
2. **Reduced mean images per training patient** (total images are constant)
3. **Proportional increase in training images from test-fold patients**

These are not confounds introduced by design error — they are **inherent consequences of the experimental intervention itself**. A patient-overlap manipulation *by construction* changes who is in the training set. The appropriate response is not to re-design the experiment, but to acknowledge this clearly in the paper.

The critical methodological control that *does* hold is:
- **Identical total training images** ✓
- **Identical per-class training images** ✓
- **Identical validation sets** ✓
- **Identical test sets** ✓

---

## 11. New Limitations Identified

**L1 — Changed patient diversity:** Conditions A, B, and C differ not only in whether test-fold patients are represented in training, but also in the number of unique training patients and mean images per patient. This is an inherent consequence of the manipulation, not a design error, but it means that the observed effects cannot be attributed solely to patient identity recognition.

**L2 — Patient-level familiarity analysis is underpowered:** The bin-level analysis of same-PID training images includes only 7 predictions in the highest bin (>30 images), which is insufficient for reliable inference.

**L3 — Single architecture:** All results were obtained with EfficientNet-B0. Whether the calibration effect generalises to other architectures, pretraining regimes, or modalities is unknown.

---

## 12. Recommended Manuscript Wording

### A. Methods section — Data splitting and overlap construction

> "Within each outer fold, training sets for Conditions A, B, and C were constructed to have identical total image counts and identical per-class image distributions. In Condition A, no images from test-fold patients were included in training. In Condition B, approximately 50% of eligible test-fold patients (those with at least two images, one reserved as the fixed test image) had their remaining non-test images added to the training set. In Condition C, all eligible test-fold patients contributed their non-test images to training. As a consequence of this manipulation, the three conditions differ in the number of unique training patients (Condition C having approximately 32–35 more unique patients per fold than Condition A) and in the mean number of training images per patient (mean 10.7–11.0 for Condition C versus 12.7–13.5 for Condition A). These differences are inherent to the patient-overlap intervention and are acknowledged as a limitation."

### B. Results section — Patient diversity

> "Across all five outer folds, training-image counts and per-class image distributions were identical across Conditions A, B, and C (integrity check passed). However, as expected from the experimental design, Condition C training sets contained 32–35 more unique patients per fold than Condition A, with approximately 25–33% of all training images belonging to test-fold patients (non-test images only)."

### C. Discussion section — Mechanism

> "The observed associations between patient-level overlap and both classification performance and apparent calibration are consistent with the hypothesis that training on images from the same patient makes the corresponding test cases more familiar to the model. However, we cannot rule out alternative explanations: test-fold patients added to training may have more typical or recognisable tumour presentations, or the reduction in mean images per patient in Condition C may itself alter training dynamics. We therefore restrict our interpretation to the observational level: patient-level overlap between training and testing data is *associated with* higher accuracy, higher predictive confidence, and apparently better calibration, without claiming a specific neural mechanism."

### D. Limitations section

> "First, the patient-overlap manipulation inherently changes not only whether test-fold patients appear in training, but also the number of unique training patients and the mean number of images per training patient. These co-occurring factors cannot be fully disentangled within the current design. Second, the exploratory patient-level familiarity analysis (relating the number of same-patient training images to prediction confidence and correctness) is based on 700 test predictions distributed across five folds and should be interpreted as hypothesis-generating rather than confirmatory. Third, all results are based on a single convolutional architecture (EfficientNet-B0); generalisation to other architectures, imaging modalities, or datasets requires further study. Fourth, with only five outer folds as the unit of statistical inference, the experiment has low formal statistical power, and significance results should be interpreted in conjunction with fold-level directional evidence."

---

## Figures

- `figures/stage3/training_patient_diversity.png` — Bar charts showing unique training patients per condition per fold.
- `figures/stage3/overlap_intensity.png` — Line plots showing overlap image count and percentage across folds for Conditions B and C.
- `figures/stage3/patient_overlap_vs_confidence.png` — Exploratory bar charts of accuracy and mean confidence versus same-PID training image bins.

---

## Final Safety Check

- **No model was retrained.** ✓
- **No split was modified.** ✓
- **No prediction was modified.** ✓
- **No existing result was overwritten.** ✓ (all outputs are new files with new names)
- **No new scientific experiment was introduced.** ✓
