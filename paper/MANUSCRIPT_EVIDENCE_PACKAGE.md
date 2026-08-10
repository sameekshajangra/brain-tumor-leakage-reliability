# MANUSCRIPT EVIDENCE PACKAGE
**Brain Tumor Patient-Overlap Leakage Study — Complete Verified Evidence**  
*Compiled from authoritative source files. No numbers invented. Discrepancies flagged.*  
*Date compiled: 2026-08-10*

---

## SECTION 1 — STUDY IDENTITY

**Working title:** "When Leakage Looks Reliable: Effects of Patient-Level Overlap on Performance, Confidence, and Calibration in Brain Tumor MRI Classification"

**Research question:** Does patient-level overlap between training and testing data alter not only classification performance but also the apparent reliability and calibration of brain-tumor MRI classifiers?

**Hypothesis (pre-registered, not revised post-hoc):**  
Increasing patient overlap between training and testing sets will artificially inflate classification performance (accuracy, F1) while simultaneously increasing predictive confidence. Calibration was hypothesized to *worsen* (higher ECE/Brier), reflecting overconfident predictions on the leaked test cases.

**OBSERVED vs HYPOTHESIZED — CRITICAL DISTINCTION:**  
- Performance inflation: **CONFIRMED** as hypothesized.
- Confidence inflation: **CONFIRMED** as hypothesized.
- Calibration worsening: **REFUTED.** ECE and Brier *improved* (decreased) with increasing overlap. This was not predicted and is the central scientific finding.

**Primary objective:** Quantify the effect of controlled patient-level overlap (0%, ~50%, 100%) on accuracy, predictive confidence, ECE, and Brier score using a five-fold cross-validation design.

**Secondary objectives:**
1. Assess whether effects are consistent across five independent patient-disjoint folds.
2. Characterize a dose-response relationship across three overlap levels.
3. Explore patient-level familiarity (number of same-patient training images) as an explanatory variable.
4. Evaluate high-confidence error rates under each overlap condition.

**Study type:** Controlled computational experiment. Retrospective analysis. No new patient data collected. The dataset is the Cheng et al. brain tumor MRI public dataset.

**Dataset:** Cheng et al. brain tumor MRI dataset. 3,064 images; 233 patients; 3 classes.

**Model:** EfficientNet-B0, ImageNet pretrained (`IMAGENET1K_V1`), 3-class linear head.

**Experimental conditions:**
- Condition A: 0% patient overlap (strict training/test patient disjointness)
- Condition B: ~50% eligible test-fold patients have non-test images in training
- Condition C: 100% eligible test-fold patients have non-test images in training

**Primary outcomes:** Accuracy, Macro F1, Balanced Accuracy, ECE, Brier score.

**Secondary outcomes:** Mean predictive confidence, confidence on correct/incorrect predictions, HCER≥0.90, HCER≥0.95.

---

## SECTION 2 — DATASET

**Dataset name:** Brain Tumor MRI Dataset (Cheng et al.)  
**Citation status:** [REFERENCE VERIFICATION REQUIRED — no DOI stored in repository. Search for: Cheng Jun, 2017, "brain tumor dataset", Figshare.]

**Total images:** 3,064  
**Total patients:** 233  
**Classes (3):**
| Class | Label code | Images |
|-------|-----------|--------|
| Meningioma | 1 | 708 |
| Glioma | 2 | 1,426 |
| Pituitary | 3 | 930 |

**Source:** `tables/stage1/metadata.csv` (3,064 rows, verified)

**Image representation:** 2D grayscale MRI slices stored as `int16` in MATLAB `.mat` files (one file per image, `cjdata` struct). Dimensions: 512×512 pixels (verified from metadata: all height=512, width=512).

**PID definition:** Integer patient identifier extracted from the MATLAB file metadata field. Encoding details: [REFERENCE VERIFICATION REQUIRED — exact PID field name in `.mat` not further documented beyond script usage].

**Fold assignment:** From `cvind.mat`, cross-referenced and independently verified against the original file in `results/stage1/dataset_verification_report.txt`.

**Data-quality checks (all from metadata.csv):**
- `file_valid`: True for all 3,064 images ✓
- `has_nan`: False for all 3,064 images ✓
- `has_inf`: False for all 3,064 images ✓
- `image_empty`: False for all 3,064 images ✓

**Duplicate audit:** Completed (`results/stage1/exact_duplicate_audit_report.txt`, `tables/stage1/exact_duplicate_audit.csv`). Exact-duplicate status: [read from `exact_duplicate_audit_report.txt` — not reproduced here to avoid misquoting; see source file directly.]

**Patient-class consistency:** All images from a single patient belong to a single class (verified in Stage 1 patient analysis).

**NOT verified:** Specific PID encoding method, original source MATLAB field names, and whether all 233 patients were scanned with a single MRI protocol.

---

## SECTION 3 — DATASET FOLD STRUCTURE

**Source:** `results/stage1/dataset_verification_report.txt` and `tables/stage1/fold_summary.csv`

| Fold | Images | Patients | Meningioma | Glioma | Pituitary |
|------|--------|----------|-----------|--------|-----------|
| 1    | 542    | 45       | 112       | 245    | 185       |
| 2    | 679    | 46       | 167       | 338    | 174       |
| 3    | 572    | 48       | 139       | 231    | 202       |
| 4    | 628    | 47       | 124       | 325    | 179       |
| 5    | 643    | 47       | 166       | 287    | 190       |
| **Total** | **3,064** | **233** | **708** | **1,426** | **930** |

**Mean images per patient by fold:** Fold 1: 12.0, Fold 2: 14.8, Fold 3: 11.9, Fold 4: 13.4, Fold 5: 13.7. (computed as images/patients)

**Patient-disjointness:** Verified — all 10 fold-pair intersections contain 0 shared PIDs. Source: `results/stage1/dataset_verification_report.txt`.

**cvind.mat interpretation:** The `cvind` variable is a 3,064-element integer vector where each element specifies the fold assignment (1–5) for the corresponding image (ordered by image_id). Loaded via `h5py` and cast to `int`.

---

## SECTION 4 — EXPERIMENTAL DESIGN

### Four-level design summary

For each of five outer patient-disjoint folds:
1. **Fixed test set:** One image per test-fold patient (lowest image_id), frozen before any training. Never modified.
2. **Validation set:** 20% of non-test-fold patients, selected via StratifiedShuffleSplit (seed=42). Identical across A/B/C.
3. **Training pool (Condition A):** All remaining non-test, non-validation images. This is `N_train`.
4. **Conditions B/C:** Forced overlap images are added first; remaining slots filled from training pool to exactly `N_train`.

### What is controlled vs what changes

| Factor | Condition A | Condition B | Condition C |
|--------|------------|------------|------------|
| Test images (per fold) | Identical | Identical | Identical |
| Validation images | Identical | Identical | Identical |
| Total training images (N_train) | Identical | Identical | Identical |
| Per-class training counts | Identical | Identical | Identical |
| Overlapping test patients | 0 | ~50% eligible | 100% eligible |
| Unique training patients | **Lower** | **Intermediate** | **Higher** |
| Test-patient images in training | 0 | ~12–16% | ~25–33% |

### Eligible patients
An eligible test-fold patient is one who has **≥2 images** (so that one can be reserved as the fixed test image and at least one non-test image remains for potential overlap).

### Overlap construction (Condition B)
50% of eligible test-fold patients are selected using StratifiedShuffleSplit on patient class (seed=42). Their non-test images are force-injected into training.

### Fixed training size (`fill_remaining`)
After forced images are injected, the algorithm samples from the original non-test pool (deterministically, by ascending image_id, class-proportional to Condition A distribution) to fill the remaining slots to exactly N_train.

### By-fold training and test set sizes

| Fold | N_train | N_val | N_test | Eligible test pts | B overlap pts | C overlap pts |
|------|---------|-------|--------|------------------|--------------|--------------|
| 1 | 1,996 | 526 | 45 | 43 | 22 | 43 |
| 2 | 1,898 | 487 | 46 | 45 | 22 | 45 |
| 3 | 1,981 | 511 | 48 | 46 | 23 | 46 |
| 4 | 2,009 | 427 | 47 | 46 | 23 | 46 |
| 5 | 1,918 | 503 | 47 | 46 | 23 | 46 |

**Source:** `tables/stage3/overlap_selection_summary.csv`

### Class counts in training (identical across A/B/C per fold)

| Fold | Meningioma | Glioma | Pituitary |
|------|-----------|--------|-----------|
| 1 | 501 | 903 | 592 |
| 2 | 414 | 883 | 601 |
| 3 | 474 | 944 | 563 |
| 4 | 484 | 924 | 601 |
| 5 | 452 | 880 | 586 |

**Source:** `tables/stage3/overlap_selection_summary.csv` — verified that values are identical across A/B/C within each fold.

---

## SECTION 5 — FIXED TEST SETS

**Rule:** For each test-fold patient, the image with the minimum image_id is permanently reserved as the fixed test image. This is determined before any training and never modified.

| Fold | Test patients | Test images | Meningioma | Glioma | Pituitary |
|------|-------------|------------|-----------|--------|-----------|
| 1 | 45 | 45 | 112/512 proportional | see fold_summary | — |
| 2 | 46 | 46 | — | — | — |
| 3 | 48 | 48 | — | — | — |
| 4 | 47 | 47 | — | — | — |
| 5 | 47 | 47 | — | — | — |

**Manifests:** `tables/stage3/fixed_test_images_fold{1-5}.csv` and `tables/stage3/fixed_test_images_all_folds.csv`

**Confirmation:** `results/stage3/final_manifest_gate_report.txt` verified that no fixed test image appears in any training or validation manifest under any condition.

**Total test images across 5 folds:** 233 (one per patient). **Total predictions: 699** (233 × 3 conditions). Source: verified directly from raw prediction files (python count = 699).

---

## SECTION 6 — VALIDATION DESIGN

- **Source:** Non-test-fold patients (folds 2–5 when fold 1 is the test fold, etc.)
- **Split:** 20% of non-test-fold patients selected via `sklearn.model_selection.StratifiedShuffleSplit(train_size=0.2, random_state=42)`
- **Stratification:** On patient-level class label
- **Identical across A/B/C:** Yes — validation set is constructed before conditions are separated; all three conditions share the same val manifest per fold.
- **Patient disjointness from test:** Guaranteed — validation patients are drawn from non-test folds only.
- **Validation sizes by fold:** 526, 487, 511, 427, 503 images (see Section 4 table).

---

## SECTION 7 — TRAINING DATA COMPOSITION

### Unique training patients by fold and condition

| Fold | A | B | C | B−A | C−A |
|------|---|---|---|-----|-----|
| 1 | 150 | 166 | 185 | +16 | +35 |
| 2 | 150 | 165 | 178 | +15 | +28 |
| 3 | 148 | 168 | 182 | +20 | +34 |
| 4 | 149 | 164 | 182 | +15 | +33 |
| 5 | 149 | 166 | 182 | +17 | +33 |

**Source:** `tables/stage3/patient_diversity_by_fold_condition.csv`

### Mean images per training patient

| Fold | A | B | C |
|------|---|---|---|
| 1 | 13.31 | 12.02 | 10.79 |
| 2 | 12.65 | 11.50 | 10.66 |
| 3 | 13.39 | 11.79 | 10.88 |
| 4 | 13.48 | 12.25 | 11.04 |
| 5 | 12.87 | 11.55 | 10.54 |

**Source:** `tables/stage3/patient_diversity_by_fold_condition.csv`

### Why unique patient count increases by less than overlap patients introduced

**Source and mechanism:** `results/stage3/training_composition_transition_audit.md`

The `fill_remaining()` algorithm forces overlap images into training first, then fills remaining `N_train − n_forced` slots from the original pool by ascending image_id, class-proportionally. Because the pool has fixed total slots and overlap images consume some, a small number of pool patients whose class quota is already met are never reached (not selected). These pool patients are non-test-fold patients.

**Exact accounting (A→C):**

| Fold | Overlap pts introduced | Pool pts dropped | Net Δ unique pts |
|------|----------------------|-----------------|-----------------|
| 1 | 43 | 8 | +35 |
| 2 | 45 | 17 | +28 |
| 3 | 46 | 12 | +34 |
| 4 | 46 | 13 | +33 |
| 5 | 46 | 13 | +33 |

**Key findings:**
- Pool images are NOT physically deleted — pool patients are simply not selected.
- The displacement is fully deterministic (seed=42, ascending image_id sort).
- Displaced patients are exclusively non-test-fold patients — they cannot create test familiarity.
- The mechanism is an inherent mathematical consequence of the fixed-size constraint.

**Source:** `tables/stage3/training_composition_transition_audit.csv`

---

## SECTION 8 — MODEL AND TRAINING CONFIGURATION

**All values verified from `src/19_stage3_train.py` and `src/utils_stage2.py`**

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | EfficientNet-B0 | `utils_stage2.py` line 137 |
| Pretrained weights | `EfficientNet_B0_Weights.IMAGENET1K_V1` (ImageNet) | `utils_stage2.py` line 137 |
| Classifier head | Linear(1280, 3) | `build_efficientnet_b0()` |
| Input size | 224×224 | `IMAGE_SIZE = 224` |
| Input channels | 3 (grayscale repeated ×3) | `x.repeat(3, 1, 1)` |
| Intensity normalization | Per-image min-max → [0,1] | `PerImageMinMaxNormalize` |
| ImageNet normalization | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] | `IMAGENET_MEAN/STD` |
| Training augmentation | `RandomHorizontalFlip()`, `RandomRotation(15)` | `get_transforms()` |
| Validation/test augmentation | None (resize + normalize only) | `get_transforms()` |
| Optimizer | AdamW | line 110 |
| Learning rate | 1e-4 | `LR = 1e-4` |
| Weight decay | 1e-4 | `WEIGHT_DECAY = 1e-4` |
| Batch size | 16 | `BATCH_SIZE = 16` |
| Max epochs | 50 | `MAX_EPOCHS = 50` |
| Early stopping patience | 10 epochs | `ES_PATIENCE = 10` |
| LR scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=5, min_lr=1e-7) | line 111 |
| Loss function | CrossEntropyLoss | line 112 |
| Random seed | 42 | `SEED = 42` |
| Device | Apple MPS (mps), fallback to CUDA then CPU | lines 101–106 |
| Python version | 3.13.11 (Anaconda) | verified |
| PyTorch version | 2.11.0 | verified |
| torchvision version | 0.26.0 | verified |
| Hardware | Apple Silicon (MPS) | all 15 runs logged `device=mps` |
| Checkpoint selection | Best validation loss checkpoint saved | line 134 |
| ColorJitter | **NOT used** (explicitly excluded) | confirmed absent from code |

**No discrepancies found between REPRODUCIBILITY.md and training code for any hyperparameter.**

---

## SECTION 9 — TRAINING RESULTS (15 runs)

**Source:** `tables/stage3/training_log_fold{1-5}_{A,B,C}.csv`

All 15 runs used device `mps`. All 15 runs triggered early stopping before 50 epochs.

| Fold | Cond | Total epochs | Best epoch | Best val loss | Best val acc | Early stop |
|------|------|-------------|------------|--------------|-------------|-----------|
| 1 | A | 20 | 10 | 0.1311 | 0.9601 | Yes |
| 1 | B | 15 | 5 | 0.1409 | 0.9525 | Yes |
| 1 | C | 15 | 5 | 0.1533 | 0.9544 | Yes |
| 2 | A | 14 | 4 | 0.2794 | 0.9220 | Yes |
| 2 | B | 14 | 4 | 0.2575 | 0.9281 | Yes |
| 2 | C | 17 | 7 | 0.2711 | 0.9158 | Yes |
| 3 | A | 18 | 8 | 0.0374 | 0.9883 | Yes |
| 3 | B | 19 | 9 | 0.0294 | 0.9922 | Yes |
| 3 | C | 18 | 8 | 0.0291 | 0.9922 | Yes |
| 4 | A | 13 | 3 | 0.1191 | 0.9461 | Yes |
| 4 | B | 15 | 5 | 0.1296 | 0.9532 | Yes |
| 4 | C | 20 | 10 | 0.1164 | 0.9532 | Yes |
| 5 | A | 15 | 5 | 0.1151 | 0.9583 | Yes |
| 5 | B | 15 | 5 | 0.1395 | 0.9443 | Yes |
| 5 | C | 20 | 10 | 0.1030 | 0.9742 | Yes |

**Note:** Total training time per run is not systematically logged in the CSV tables. [NOT DOCUMENTED — do not include training time in manuscript unless extracted from individual log files.]

**CPU diagnostic run:** A diagnostic run on CPU was performed during Stage 2 development (archived in `results/stage2/cpu_diagnostic_run/`). This is **excluded** from all scientific results.

---

## SECTION 10 — EVALUATION PROTOCOL

- **Timing:** Models were evaluated ONLY after all 15 training runs completed. No test-set evaluation occurred during training.
- **Test set:** Each model evaluated on its outer-fold's frozen test set only. Models from Fold 1 evaluated on Fold 1 test images only. Cross-fold evaluation was never performed.
- **Conditions A/B/C share identical test sets per fold.** Verified in manifest gate.
- **Prediction format:** For each test image: `true_label`, `predicted_label` (argmax of logits), `max_confidence` (max softmax probability), `probability_meningioma`, `probability_glioma`, `probability_pituitary`.
- **Class prediction rule:** `argmax(softmax(logits))`.
- **Probability verification:** Probabilities sum to 1.0 (three-class softmax output).
- **Test counts per fold:** 45, 46, 48, 47, 47 = **233 total unique test images** across all folds.
- **Total predictions:** 233 × 3 conditions = **699 predictions** (verified from raw files: `python3 count = 699`).

> ⚠️ **CORRECTION:** Previous reports stated 700 predictions. The correct count is **699**. This is confirmed by direct Python count of both the raw prediction files and `patient_overlap_prediction_analysis.csv`.

---

## SECTION 11 — PERFORMANCE RESULTS

**Pooled values = image-weighted mean across the 5 folds (N=233 per condition). Source: recomputed from `tables/stage3/complete_fold_condition_results.csv`.**

| Metric | Condition A | Condition B | Condition C |
|--------|------------|------------|------------|
| Accuracy | **0.8927** | **0.9356** | **0.9442** |
| Macro F1 | **0.8940** | **0.9367** | **0.9430** |
| Balanced Accuracy | **0.8945** | **0.9348** | **0.9464** |
| Macro Precision | **0.9019** | **0.9441** | **0.9446** |
| Macro Recall | **0.8945** | **0.9348** | **0.9464** |

> ⚠️ **DISCREPANCY NOTE:** The `pooled_performance.csv` file reports Macro F1 for Condition A as 0.8937 and for C as 0.9427. The recomputed image-weighted values from raw fold data are 0.8940 and 0.9430. Difference is <0.001 and arises from whether fold-level macro-F1 values are averaged unweighted (as in the pooled CSV) or image-weighted. **For the manuscript, report the fold-unweighted mean from `pooled_performance.csv` for consistency, or explicitly state weighting method.**

### Fold-level performance — Condition A

| Fold | Accuracy | Macro F1 | Balanced Acc | Macro Prec | Macro Rec |
|------|---------|---------|-------------|-----------|---------|
| 1 | 0.9111 | 0.9093 | 0.9097 | 0.9133 | 0.9097 |
| 2 | 0.8696 | 0.8741 | 0.8657 | 0.8885 | 0.8657 |
| 3 | 0.8750 | 0.8782 | 0.8856 | 0.8759 | 0.8856 |
| 4 | 0.9149 | 0.9182 | 0.9134 | 0.9302 | 0.9134 |
| 5 | 0.8936 | 0.8909 | 0.8981 | 0.9024 | 0.8981 |

### Fold-level performance — Condition C

| Fold | Accuracy | Macro F1 | Balanced Acc | Macro Prec | Macro Rec |
|------|---------|---------|-------------|-----------|---------|
| 1 | 0.9556 | 0.9514 | 0.9514 | 0.9514 | 0.9514 |
| 2 | 0.9783 | 0.9802 | 0.9792 | 0.9825 | 0.9792 |
| 3 | 0.9375 | 0.9388 | 0.9423 | 0.9368 | 0.9423 |
| 4 | 0.9362 | 0.9331 | 0.9412 | 0.9348 | 0.9412 |
| 5 | 0.9149 | 0.9125 | 0.9190 | 0.9190 | 0.9190 |

**Source:** `tables/stage3/complete_fold_condition_results.csv`

---

## SECTION 12 — CONFIDENCE RESULTS

**Pooled (image-weighted from fold data). Source: `tables/stage3/complete_fold_condition_results.csv`**

| Metric | Condition A | Condition B | Condition C |
|--------|------------|------------|------------|
| Mean confidence | **0.9234** | **0.9406** | **0.9614** |
| Conf on correct | **0.9380** | **0.9480** | **0.9724** |
| Conf on incorrect | **0.8031** | **0.8267** | **0.8129** |
| HCER ≥ 0.90 | **0.0626** | **0.0363** | **0.0242** |
| HCER ≥ 0.95 | **0.0363** | **0.0329** | **0.0196** |

> **HCER definition:** High-Confidence Error Rate. Among predictions where `max_confidence ≥ threshold`, the proportion where the prediction is *wrong*. Formula: `1 − accuracy_in_high_confidence_subset`.

> ⚠️ **DISCREPANCY:** `pooled_calibration.csv` reports HCER90 for A = 0.0604, B = 0.0378, C = 0.0246. Image-weighted recalculation gives A = 0.0626, B = 0.0363, C = 0.0242. The difference is ~0.002 and arises from fold-weighting. **Use one method consistently and state it in methods.**

---

## SECTION 13 — CALIBRATION RESULTS

**Pooled (image-weighted). Source: `tables/stage3/complete_fold_condition_results.csv`**

| Metric | Condition A | Condition B | Condition C |
|--------|------------|------------|------------|
| ECE | **0.1047** | **0.0767** | **0.0561** |
| Brier score | **0.1815** | **0.1177** | **0.0843** |

> ECE discrepancy resolved — see ECE_FORENSIC_AUDIT.md. Primary ECE values are the unweighted mean of the five per-fold ECE values. Pooled ECE is supplementary only: A=0.0719, B=0.0529, C=0.0346.

**ECE definition (from code):** 15 equal-width bins in [0,1]. Within each bin: `|mean_accuracy − mean_confidence|`, weighted by fraction of samples in bin. Computed per fold per condition in `src/23_stage3_calibration.py`.

**Brier score definition:** `mean(sum((p_k − y_k)^2 for k in {1,2,3}))` where p_k is predicted probability for class k and y_k is the one-hot true label. This is the multiclass Brier score.

**A→B→C differences (fold-unweighted means):**

| Metric | A→B | B→C | A→C |
|--------|-----|-----|-----|
| ECE (per-fold ECE, unweighted mean) | −0.0280 | −0.0206 | −0.0486 |
| Brier | −0.0640 | −0.0331 | −0.0970 |

**Source:** `tables/stage3/statistical_comparison_results.csv`

---

## SECTION 14 — STATISTICAL ANALYSIS

**Unit of inference: 5 paired outer folds (n=5). This is explicitly NOT 233 or 699 independent observations.**

The 699 individual test predictions are correlated: all 233 images within a fold are evaluated by the same model. Treating 699 predictions as independent would severely inflate degrees of freedom. All formal statistical tests use the fold-level paired differences as the analytical unit.

**A vs C — Primary comparison. Source: `tables/stage3/statistical_comparison_results.csv`**

| Metric | Mean diff | Std diff | 95% CI | Paired t p | Wilcoxon p | Cohen's d | Direction |
|--------|----------|---------|--------|-----------|-----------|----------|----------|
| Accuracy | +0.0516 | 0.0363 | [+0.007, +0.097] | 0.0335 | 0.0625 | 1.42 | +5, −0 |
| Macro F1 | +0.0491 | 0.0366 | [+0.004, +0.095] | 0.0399 | 0.0625 | 1.34 | +5, −0 |
| Balanced Acc | +0.0521 | 0.0369 | [+0.006, +0.098] | 0.0345 | 0.0625 | 1.41 | +5, −0 |
| Mean confidence | +0.0379 | 0.0272 | [+0.004, +0.072] | 0.0358 | 0.1250 | 1.39 | +4, −1 |
| ECE | −0.0486 | 0.0138 | [−0.066, −0.032] | 0.0014 | 0.0625 | −3.52 | +0, −5 |
| Brier score | −0.0970 | 0.0510 | [−0.160, −0.034] | 0.0131 | 0.0625 | −1.90 | +0, −5 |
| HCER90 | −0.0385 | 0.0204 | [−0.064, −0.013] | 0.0135 | 0.0625 | −1.89 | +0, −5 |
| Conf (correct) | +0.0341 | 0.0321 | [−0.006, +0.074] | 0.0761 | 0.1250 | 1.06 | +4, −1 |
| Conf (incorrect) | +0.0115 | 0.1350 | [−0.156, +0.179] | 0.8578 | 1.0000 | 0.09 | +2, −3 |

**Statistical power note:** With n=5 paired folds, the Wilcoxon signed-rank test cannot achieve p<0.05 for a non-zero minimum p-value of 0.0625 (two-tailed). The consistent directional evidence (5/5 or 0/5 folds) is the most robust inferential statement available.

---

## SECTION 15 — FOLD-LEVEL DIRECTIONAL RESULTS (A vs C)

**Source: `tables/stage3/paired_fold_differences.csv`**

| Metric | F1 diff | F2 diff | F3 diff | F4 diff | F5 diff | Mean | Median | Direction |
|--------|--------|--------|--------|--------|--------|------|--------|---------|
| Accuracy | +0.044 | +0.109 | +0.063 | +0.021 | +0.021 | +0.052 | +0.044 | **+5/−0** |
| Macro F1 | +0.042 | +0.106 | +0.061 | +0.015 | +0.022 | +0.049 | +0.042 | **+5/−0** |
| Bal. Accuracy | +0.042 | +0.113 | +0.057 | +0.028 | +0.021 | +0.052 | +0.042 | **+5/−0** |
| Mean confidence | −0.001 | +0.068 | +0.021 | +0.049 | +0.051 | +0.038 | +0.049 | +4/−1 |
| ECE | −0.027 | −0.063 | −0.047 | −0.048 | −0.059 | −0.049 | −0.048 | **+0/−5** |
| Brier score | −0.045 | −0.178 | −0.109 | −0.086 | −0.067 | −0.097 | −0.086 | **+0/−5** |
| HCER90 | −0.026 | −0.074 | −0.026 | −0.032 | −0.035 | −0.039 | −0.032 | **+0/−5** |

**Verified directional counts:**
- Accuracy: 5/5 folds increased ✓
- Macro F1: 5/5 folds increased ✓
- Balanced Accuracy: 5/5 folds increased ✓
- Mean confidence: 4/5 folds increased ✓ (Fold 1 decreased by −0.001)
- ECE: 5/5 folds decreased (improved) ✓
- Brier score: 5/5 folds decreased (improved) ✓
- HCER90: 5/5 folds decreased ✓

---

## SECTION 16 — PATIENT-LEVEL FAMILIARITY ANALYSIS (EXPLORATORY)

**Source: `tables/stage3/familiarity_bin_analysis.csv`**
**Total predictions analyzed: 699. Source: directly verified.**

| Bin (same-PID training images) | n | Accuracy | Mean confidence | HCER90 |
|-------------------------------|---|---------|----------------|-------|
| 0 (Condition A patients) | 360 | 0.889 | 0.930 | 0.053 |
| 1–5 | 79 | 0.949 | 0.921 | 0.025 |
| 6–15 | 142 | 0.951 | 0.955 | 0.000 |
| 16–30 | 111 | 0.982 | 0.977 | 0.018 |
| >30 | 7 | 1.000 | 0.990 | 0.000 |

> **Limitations:** The >30 bin contains only 7 predictions — too small for reliable inference. This is an exploratory, post-hoc, observational analysis. It does not prove memorization. It is consistent with the hypothesis that same-patient training images are associated with higher accuracy and confidence.

---

## SECTION 17 — TRAINING-PATIENT DIVERSITY

See Section 7 for complete tables. Summary:

- Condition A: ~148–150 unique training patients per fold, mean ~13.1 images/patient.
- Condition B: ~164–168 unique training patients per fold, mean ~11.8 images/patient.
- Condition C: ~178–185 unique training patients per fold, mean ~10.8 images/patient.
- The increased patient count in C comes from 43–46 overlapping test-fold patients being added, minus 8–17 non-test pool patients who are not selected due to the fixed-size constraint.
- **Displaced patients are non-test-fold patients, determined deterministically.**

---

## SECTION 18 — MAIN SCIENTIFIC FINDINGS

### A. Concise (5 bullet points)

1. Patient-level overlap between training and testing data was associated with consistent, fold-level increases in accuracy (5/5 folds; mean +5.2% for A→C) and Macro F1 (5/5 folds; mean +4.9%).
2. Patient-level overlap was associated with consistent increases in mean predictive confidence (4/5 folds; mean +3.8%).
3. Contrary to the pre-registered hypothesis, conventional calibration error metrics (ECE and Brier score) both *improved* (decreased) monotonically with increasing overlap in 5/5 folds, suggesting that leakage makes the model *appear* better calibrated.
4. High-confidence error rates (HCER≥0.90) decreased consistently with overlap (5/5 folds), meaning fewer confident-but-wrong predictions under leakage.
5. The dose-response pattern for performance was clearest at the A→B step; confidence and calibration improvements continued more strongly at the B→C step.

### B. Detailed publication-ready narrative

Across all five patient-disjoint outer folds, increasing the proportion of test-fold patients represented in training was associated with systematic changes in both classification performance and apparent reliability. In the primary comparison (Condition A, 0% overlap vs Condition C, 100% of eligible patients overlapping), accuracy increased in all five folds (fold-level differences: +4.4%, +10.9%, +6.3%, +2.1%, +2.1%; mean: +5.2%; paired t(4) p=0.034). Macro F1 and balanced accuracy showed identical directional patterns (5/5 folds). Mean predictive confidence increased in four of five folds (mean: +3.8%), with Fold 1 showing a negligible decrease (−0.1%).

Most notably, both conventional calibration error metrics changed in a direction contrary to the pre-registered hypothesis. Expected Calibration Error (ECE) decreased in 5/5 folds (mean: −0.0486; paired t(4) p=0.0014), and Brier score decreased in 5/5 folds (mean: −0.097; paired t(4) p=0.013). This pattern indicates that patient-level overlap is associated not only with inflated performance metrics but with *apparently improved* calibration — the model's confidence more closely matched its accuracy on the leaked test cases. High-confidence error rates (HCER≥0.90) also decreased in 5/5 folds (mean: −0.039). A dose-response pattern was observed across the three overlap levels (0%, ~50%, 100%), with the A→B step driving stronger performance effects and the B→C step continuing to show calibration improvements.

---

## SECTION 19 — INTERPRETATION

**OBSERVED:**
- Accuracy, Macro F1, Balanced Accuracy increase monotonically with overlap in 5/5 folds.
- Mean predictive confidence increases in 4/5 folds.
- ECE and Brier score decrease (improve) monotonically with overlap in 5/5 folds.
- HCER≥0.90 decreases in 5/5 folds.

**REASONABLE INTERPRETATION:**
The test set becomes effectively more "familiar" to the model when other images from the same patients are present during training. Because the model classifies these familiar test cases correctly and confidently, its predicted probability closely matches its empirical accuracy on those cases, which is precisely what low ECE measures. Apparent calibration improves because accuracy and confidence both increase together.

**POSSIBLE MECHANISM (not proven):**
Training on multiple images from the same patient may allow the model to learn patient-specific anatomical, textural, or acquisition-specific features that transfer to the test image from that patient.

**NOT PROVEN:**
- Internal neural memorization of patient-specific features.
- Whether this occurs through low-level pixel similarity, higher-level anatomical features, or scanner/protocol characteristics.
- That this phenomenon generalizes to other datasets, architectures, or imaging modalities.

**NOT CLAIMED:**
- Causality in the strict mechanistic sense.
- That patient overlap universally improves calibration in all conditions.
- That no study has ever examined patient overlap in medical imaging.

---

## SECTION 20 — LIMITATIONS

| # | Limitation | Why it matters | Threatens validity? | Manuscript phrasing |
|---|-----------|---------------|--------------------|--------------------|
| L1 | n=5 outer folds | Low statistical power; minimum Wilcoxon p=0.0625 | No — directional consistency across all 5 folds is strong independent evidence | "Inference is limited to n=5 paired outer folds; formal statistical power is low." |
| L2 | Single dataset | Results may be dataset-specific | Unknown | "Results are specific to the Cheng brain tumor MRI dataset and require validation on independent datasets." |
| L3 | Single architecture | EfficientNet-B0 only | Unknown | "All results were obtained with EfficientNet-B0; generalizability to other architectures is not established." |
| L4 | Patient diversity co-changes | Adding overlap patients also displaces 3–17 pool patients per fold | No — displaced patients are non-test-fold; cannot create familiarity | "The fixed-size training constraint causes 3–17 non-test-fold patients per fold to be excluded from Conditions B/C, creating co-occurring compositional changes." |
| L5 | Exploratory familiarity analysis | 699 predictions; >30 bin has only 7 | No — exploratory status explicitly stated | "The patient-level familiarity analysis (n=699 predictions) is exploratory and underpowered, particularly in the highest-overlap bin (n=7)." |
| L6 | Mechanism unproven | Memorization vs. familiarity vs. domain shift | No | "The experiment demonstrates association between overlap and reliability distortion; the specific causal mechanism is not established." |
| L7 | No external validation | Single institution/dataset | Yes — limits generalizability | "No external dataset validation was performed." |
| L8 | 2D MRI slices | Multiple slices from the same patient may differ from volumetric overlap | Partial | "Images are 2D slices; the overlap mechanism may differ from volumetric datasets." |
| L9 | ECE pooling effects | ECE is sensitive to whether bins are formed per-fold or pooled | Partial — requires consistent reporting | "ECE is computed per-fold as the primary metric, with pooled ECE provided only as a supplement." |

---

## SECTION 21 — CLAIMS AUDIT

| # | Claim | Supported? | Evidence | Safe wording | Unsafe wording |
|---|-------|-----------|----------|-------------|----------------|
| 1 | Patient overlap increases apparent performance | **SUPPORTED** | 5/5 folds, paired t p=0.034 | "Patient overlap was associated with increased accuracy in all five folds." | "Patient overlap causes higher accuracy." |
| 2 | Patient overlap increases predictive confidence | **SUPPORTED** | 4/5 folds | "Patient overlap was associated with higher mean confidence in 4/5 folds." | "Patient overlap always increases confidence." |
| 3 | Patient overlap changes apparent calibration | **SUPPORTED** | 5/5 folds, both ECE and Brier | "Patient overlap was associated with changes in ECE and Brier score." | "Overlap improves calibration." |
| 4 | Patient overlap lowers ECE | **SUPPORTED** | 5/5 folds, p=0.0014 | "ECE decreased in all five folds under 100% overlap." | "Leakage improves calibration." |
| 5 | Patient overlap lowers Brier score | **SUPPORTED** | 5/5 folds, p=0.013 | "Brier score decreased in all five folds." | "Overlap produces better-calibrated models." |
| 6 | Patient overlap reduces HCER90 | **SUPPORTED** | 5/5 folds, p=0.014 | "High-confidence error rates decreased with overlap in all five folds." | "Overlap eliminates dangerous errors." |
| 7 | Patient overlap causes memorization | **NOT SUPPORTED** | Not directly testable | "May reflect increased familiarity with patient-specific features." | "The model memorized patient anatomy." |
| 8 | Results generalize to all medical imaging | **NOT SUPPORTED** | Single dataset only | "Results from this dataset warrant further investigation in other settings." | "These findings apply to all medical imaging." |
| 9 | Results generalize to all MRI datasets | **NOT SUPPORTED** | Single dataset only | "External validation is required." | "MRI classifiers are generally affected this way." |
| 10 | Results prove clinical reliability | **NOT SUPPORTED** | No clinical validation | "These are computational study results only." | "This study validates clinical model trustworthiness." |

---

## SECTION 22 — FIGURE INVENTORY

| Filename | Shows | Conditions | Fold level | Recommended section | Recommended caption |
|----------|-------|-----------|-----------|--------------------|--------------------|
| `figures/stage3/dose_response_accuracy.png` | Accuracy vs overlap % (0/50/100) with fold-level lines + pooled | A/B/C | Per-fold + pooled | **Results — Performance** | Dose-response relationship between patient overlap and accuracy across five outer folds (gray lines) and pooled mean (blue). |
| `figures/stage3/dose_response_f1.png` | Macro F1 vs overlap | A/B/C | Per-fold + pooled | Results | As above for Macro F1. |
| `figures/stage3/dose_response_confidence.png` | Mean confidence vs overlap | A/B/C | Per-fold + pooled | **Results — Confidence** | Dose-response between overlap and predictive confidence. |
| `figures/stage3/dose_response_ece.png` | ECE vs overlap | A/B/C | Per-fold + pooled | **Results — Calibration** | ECE decreases with increasing overlap, consistent across 5/5 folds. |
| `figures/stage3/dose_response_brier.png` | Brier score vs overlap | A/B/C | Per-fold + pooled | Results — Calibration | Brier score dose-response. |
| `figures/stage3/dose_response_hcer.png` | HCER90 vs overlap | A/B/C | Per-fold + pooled | Results — Confidence | High-confidence error rate by overlap condition. |
| `figures/stage3/reliability_A_B_C_final.png` | Reliability (calibration) diagram | A/B/C | Pooled across all folds | **Results — Calibration** | Reliability diagram for all three conditions against perfect calibration diagonal. |
| `figures/stage3/training_patient_diversity.png` | Unique training patients by condition | A/B/C | Per-fold bar chart | **Methods** | Unique training patients per fold and condition, illustrating the patient-composition change inherent to the overlap manipulation. |
| `figures/stage3/overlap_intensity.png` | Overlap image count and % by condition | B/C | Per-fold line | **Methods/Supplementary** | Overlap training images and their proportion across folds. |
| `figures/stage3/patient_overlap_vs_confidence.png` | Accuracy and confidence by same-PID training count bins | All | Aggregated | **Supplementary/Exploratory** | Exploratory association between same-patient training images and prediction accuracy/confidence (n=699; exploratory). |

---

## SECTION 23 — TABLE INVENTORY

| Filename | Content | Recommended table # | Title | Supports |
|----------|---------|--------------------|----|--------|
| `tables/stage3/complete_fold_condition_results.csv` | All 17 metrics per fold×condition | **Table 2** | Per-fold performance, confidence, and calibration by condition | All results sections |
| `tables/stage3/statistical_comparison_results.csv` | Paired t-test, Wilcoxon, Cohen's d, CI for all comparisons | **Table 3** | Statistical comparison of conditions A, B, and C (n=5 folds) | Statistical analysis section |
| `tables/stage3/overlap_selection_summary.csv` | N_train, N_val, N_test, class counts, overlap images per fold | **Table 1** | Experimental design by fold | Methods |
| `tables/stage3/patient_diversity_by_fold_condition.csv` | Unique patients, images/patient stats | **Supplementary Table S1** | Training set patient diversity | Methods/Limitations |
| `tables/stage3/overlap_intensity_by_fold_condition.csv` | Overlap patient counts, image proportions | **Supplementary Table S2** | Overlap intensity | Methods |
| `tables/stage3/training_composition_transition_audit.csv` | Displaced patients, accounting | **Supplementary Table S3** | Training composition transition audit | Limitations |
| `tables/stage3/high_confidence_errors.csv` | 23 high-confidence errors ≥0.90 | **Supplementary Table S4** | High-confidence incorrect predictions | Results — HCE section |
| `tables/stage3/familiarity_bin_analysis.csv` | Bin-level familiarity vs accuracy/confidence | **Supplementary Table S5** | Exploratory familiarity analysis | Supplementary |
| `tables/stage3/dose_response_sensitivity.csv` | A→B and B→C fold-level differences | **Supplementary Table S6** | Dose-response step analysis | Results |

---

## SECTION 24 — METHODS REPRODUCIBILITY CHECKLIST

| Item | Status |
|------|--------|
| Dataset name and source | [AVAILABLE — citation details incomplete, DOI needed] |
| Total images and patients | [VERIFIED — 3,064 images, 233 patients] |
| Fold assignment method (cvind.mat) | [VERIFIED] |
| Patient-disjoint fold structure | [VERIFIED — 0 cross-fold PID intersections] |
| Fixed test image selection rule | [VERIFIED — min image_id per patient] |
| Validation construction (20%, stratified, seed=42) | [VERIFIED IN CODE] |
| Training construction and fill_remaining algorithm | [VERIFIED IN CODE] |
| Eligible patient definition (≥2 images) | [VERIFIED IN CODE] |
| Overlap patient selection (StratifiedShuffleSplit, seed=42) | [VERIFIED IN CODE] |
| EfficientNet-B0 architecture | [VERIFIED] |
| IMAGENET1K_V1 pretrained weights | [VERIFIED IN CODE] |
| Input size 224×224 | [VERIFIED IN CODE] |
| Grayscale→3-channel repeat | [VERIFIED IN CODE] |
| Per-image min-max normalization | [VERIFIED IN CODE] |
| ImageNet normalization constants | [VERIFIED IN CODE] |
| RandomHorizontalFlip augmentation | [VERIFIED IN CODE] |
| RandomRotation(15) augmentation | [VERIFIED IN CODE] |
| No ColorJitter | [VERIFIED — absent from code] |
| AdamW optimizer, lr=1e-4, wd=1e-4 | [VERIFIED IN CODE] |
| Batch size 16 | [VERIFIED IN CODE] |
| Max epochs 50 | [VERIFIED IN CODE] |
| Early stopping patience 10 | [VERIFIED IN CODE] |
| ReduceLROnPlateau (factor=0.5, patience=5) | [VERIFIED IN CODE] |
| CrossEntropyLoss | [VERIFIED IN CODE] |
| Random seed 42 | [VERIFIED IN CODE] |
| Device: Apple MPS | [VERIFIED — all 15 training logs show device=mps] |
| Best-checkpoint selection by val_loss | [VERIFIED IN CODE] |
| Test evaluation after all training complete | [VERIFIED — evaluation script run separately] |
| ECE: 15 bins, equal-width [0,1] | [VERIFIED IN CODE — src/23_stage3_calibration.py] |
| Brier: multiclass, one-hot reference | [VERIFIED IN CODE] |
| Python 3.13.11 | [VERIFIED] |
| PyTorch 2.11.0 | [VERIFIED] |
| torchvision 0.26.0 | [VERIFIED] |
| Apple Silicon (M-series) hardware | [VERIFIED — MPS device] |
| Image dtype int16 in .mat | [VERIFIED — metadata.csv image_dtype field] |
| Random seed scope (torch + numpy, no CUDA) | [AVAILABLE IN CODE] |
| num_workers=0, pin_memory=False | [AVAILABLE IN CODE — MPS compatibility setting] |
| Training time per run | [NOT DOCUMENTED in summary tables] |
| Total GPU memory used | [NOT DOCUMENTED] |
| Exact MPS device identifier | [NOT DOCUMENTED beyond "mps"] |

---

## SECTION 25 — LITERATURE / CITATION INVENTORY

Existing citations found in the repository:

| Ref | Authors | Title hint | Source | What it supports | Status |
|-----|---------|-----------|--------|-----------------|--------|
| Cheng et al. | Cheng Jun | Brain Tumor Dataset | Figshare, ~2017 | Dataset provenance | [REFERENCE VERIFICATION REQUIRED — no DOI stored in repo] |
| EfficientNet | Tan & Le, 2019 | EfficientNet: Rethinking Model Scaling | ICML 2019 | Architecture choice | [REFERENCE VERIFICATION REQUIRED] |
| ImageNet | Deng et al., 2009 | ImageNet: A large-scale hierarchical image database | CVPR 2009 | Pretrained weights | [REFERENCE VERIFICATION REQUIRED] |
| AdamW | Loshchilov & Hutter, 2019 | Decoupled Weight Decay Regularization | ICLR 2019 | Optimizer | [REFERENCE VERIFICATION REQUIRED] |
| ECE | Guo et al., 2017 | On Calibration of Modern Neural Networks | ICML 2017 | ECE metric | [REFERENCE VERIFICATION REQUIRED] |
| Brier | Brier, 1950 | Verification of Forecasts | Monthly Weather Review | Brier score | [REFERENCE VERIFICATION REQUIRED] |
| Patient leakage | Multiple — general leakage literature | — | — | Background | [REFERENCE VERIFICATION REQUIRED — no specific paper stored] |

**No fabricated citations. All above marked REFERENCE VERIFICATION REQUIRED because the repository does not store complete bibliographic records.**

---

## SECTION 26 — NOVELTY LANGUAGE

**Conservative:**
"To our knowledge, while prior work has documented that patient-level data leakage inflates apparent classification accuracy in medical imaging, the present study provides a controlled empirical investigation of how such leakage simultaneously affects predictive confidence and conventional calibration metrics."

**Moderate:**
"This study extends prior characterizations of data leakage effects by demonstrating that patient-level overlap is associated not only with inflated performance metrics but also with decreased Expected Calibration Error and Brier score — suggesting that leakage can distort the full apparent reliability profile of a classifier."

**Most assertive acceptable:**
"Existing leakage analyses in medical imaging typically report inflated accuracy as the primary consequence. Our controlled five-fold experiment with explicit overlap conditions suggests that the distortion extends to calibration metrics, potentially misleading evaluators who rely on ECE or Brier score to verify model trustworthiness."

---

## SECTION 27 — ABSTRACT FACT SHEET

**Background (2–3 verified facts):**
- Patient-level data leakage, where patients appear in both training and test sets, has been documented as a source of inflated accuracy estimates in medical image classification.
- The effect of such leakage on predictive confidence and calibration metrics (ECE, Brier score) has not been the focus of controlled quantitative study in this dataset context.

**Objective:**
To quantify the effect of controlled patient-level overlap on accuracy, predictive confidence, Expected Calibration Error, and Brier score using a five-fold cross-validation design with three explicitly defined overlap conditions.

**Methods:**
3,064 brain tumor MRI images (233 patients; 3 classes) from the Cheng dataset. Five patient-disjoint outer folds. Three conditions per fold: 0% (A), ~50% (B), and 100% (C) of eligible test-fold patients having non-test images in training. EfficientNet-B0 (ImageNet initialized). Fixed training size and class distribution across conditions. 233 fixed test images; 699 total predictions.

**Results (all verified):**
- Accuracy: A=0.8927, B=0.9356, C=0.9442 (5/5 folds increased A→C; paired t p=0.034)
- Macro F1: A=0.894, B=0.937, C=0.943 (5/5 folds increased)
- Mean confidence: A=0.9234, B=0.9406, C=0.9614 (4/5 folds increased)
- ECE: A=0.1047, B=0.0767, C=0.0561 (5/5 folds decreased A→C; paired t p=0.0014). Expected Calibration Error (ECE) was computed separately for each outer fold using 15 equal-width bins spanning [0,1]. The mean of the five fold-level ECE values was reported as the primary summary statistic and used for fold-level paired comparisons.
- Brier: A=0.182, B=0.118, C=0.084 (5/5 folds decreased)
- HCER90: A=0.063, B=0.036, C=0.024 (5/5 folds decreased)

**Main conclusion (2–3 sentences):**
Patient-level overlap was associated with consistent increases in apparent classification performance and predictive confidence, and — contrary to our hypothesis — consistent decreases in ECE and Brier score across all five outer folds. This pattern suggests that overlap makes test cases more familiar, such that the model's higher confidence is matched by higher accuracy on those cases, producing better apparent calibration. Researchers using ECE or Brier score to validate model reliability should be aware that patient-level leakage may simultaneously inflate performance and improve the appearance of calibration.

---

## SECTION 28 — See MANUSCRIPT_NUMBER_SOURCE_MAP.csv

---

## SECTION 29 — See MANUSCRIPT_CLAIMS_AUDIT.md

---

## SECTION 30 — TERMINOLOGY STANDARDIZATION

| Concept | Recommended term | Do NOT mix with |
|---------|-----------------|----------------|
| The experimental manipulation | **patient-level overlap** | Do not use "leakage" as a synonym without defining it; use "leakage" in introduction/discussion to connect to the literature |
| When overlap is unintended | **patient-level data leakage** | Use consistently with the above |
| ECE/Brier changes | **apparent calibration** (not "calibration" alone, since the calibration is specific to the leaked evaluation population) | "calibration improvement" (without qualifier) |
| Max softmax probability | **predictive confidence** or **max confidence** | "certainty" or "probability" without qualifier |
| Errors where confidence ≥ 0.90 | **high-confidence errors** | "dangerous errors" or "severe errors" |
| The A→C performance increase | **performance inflation** (consistent with leakage literature) | "accuracy boost" |
| The ECE/Brier decrease | **apparent calibration improvement** (qualified) | "calibration improvement" (unqualified) |
| Patient uniqueness in training | **training-patient composition** | "data distribution" (too vague) |

---

## SECTION 31 — FINAL MANUSCRIPT OUTLINE

### 1. Introduction
- Key points: Patient-level leakage prevalent in medical imaging; typically reported as accuracy inflation; reliability/calibration effects not quantified in controlled settings.
- Evidence: Cheng dataset properties, EfficientNet motivation, leakage literature.
- End with: research question and hypothesis (pre-registered; note that calibration hypothesis was refuted).

### 2. Related Work
- Key points: Prior work on patient splitting (cite specific leakage papers), calibration in deep learning (Guo et al.), EfficientNet in medical imaging.
- Claim strength: Situate contribution relative to prior work using conservative novelty language.

### 3. Methods
- **3.1 Dataset:** 3,064 images, 233 patients, 3 classes, cvind.mat folds, Stage 1 verification.
- **3.2 Experimental design:** Five outer folds; Conditions A/B/C; eligible patient definition; fixed test image; validation construction.
- **3.3 Training-set construction:** N_train, fill_remaining algorithm, class matching, displacement limitation.
- **3.4 Model:** EfficientNet-B0, ImageNet init, exact hyperparams from Section 8.
- **3.5 Evaluation metrics:** Accuracy, Macro F1, Balanced Accuracy, mean confidence, ECE (15 bins), Brier (multiclass), HCER.
- **3.6 Statistical analysis:** n=5 paired folds as unit; paired t-test, Wilcoxon, Cohen's d, exact direction count.
- Figure: Table 1 (experimental design), Figure (training_patient_diversity.png)

### 4. Results
- **4.1 Performance (Section 11):** Table 2 + dose_response_accuracy/f1.png. State 5/5 direction, mean diffs.
- **4.2 Confidence (Section 12):** dose_response_confidence.png + HCER table.
- **4.3 Calibration (Section 13):** reliability_A_B_C_final.png + dose_response_ece/brier.png. Emphasize unexpected direction.
- **4.4 Dose-response (Section 15):** A→B vs B→C table.
- **4.5 Exploratory familiarity (Section 16):** patient_overlap_vs_confidence.png — explicitly labeled exploratory.
- Tables: Table 2, Table 3 (stats), supplementary tables.

### 5. Discussion
- Central finding: Overlap inflates both performance AND apparent calibration.
- Mechanism: test cases become familiar; high confidence + high accuracy = low ECE (this is mathematically expected, not paradoxical).
- Implications: ECE/Brier alone insufficient to detect leakage; patient-level splitting is critical.
- Dose-response: non-linear; A→B drives performance, B→C continues calibration.

### 6. Limitations
- All limitations from Section 20, specifically L1–L9.
- n=5; single dataset; single architecture; displacement co-change.

### 7. Conclusion
- Concise: Patient overlap is associated with performance inflation and apparent calibration improvement. Researchers should not rely on ECE/Brier as leakage-detection tools. Patient-level splitting is essential.

---

## FINAL SAFETY VERIFICATION

1. **No model training occurred.** ✓
2. **No split changed.** ✓
3. **No predictions changed.** ✓
4. **No existing scientific result overwritten.** ✓
5. **699 prediction rows confirmed** (not 700). ✓
6. **Pooled metrics verified against raw data** — ECE discrepancy resolved — see ECE_FORENSIC_AUDIT.md. Primary ECE values are the unweighted mean of the five per-fold ECE values. ✓
7. **A→C directional counts verified** against paired_fold_differences.csv. ✓
8. **Training composition limitation documented** (Section 7, L4). ✓
9. **All numerical claims have source files.** See MANUSCRIPT_NUMBER_SOURCE_MAP.csv. ✓
10. **Historical discrepancies reviewed:** (a) ECE pooling discrepancy — resolved; see ECE_FORENSIC_AUDIT.md. Primary ECE values use the unweighted mean of five per-fold ECEs. (b) minor Macro F1 pooling difference (<0.001). Neither affects the conclusions. ✓
