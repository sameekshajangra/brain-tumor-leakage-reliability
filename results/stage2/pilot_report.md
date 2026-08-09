# Stage 2 Pilot Report — Brain Tumor Patient Overlap Experiment
**Generated:** 2026-08-09 15:33:35

---

> **PILOT NOTICE**: This experiment uses only 47 fixed test images (Fold 5).
> Results are preliminary. Do not claim statistical significance or definitive
> generalization conclusions from this pilot. Stage 3 (full five-fold experiment)
> requires explicit approval.

---

## 1. Experimental Design

### 1.1 Test Image Selection Rule
- **Rule**: For each Fold-5 patient, select the image with the **minimum numeric image_id** as the fixed test image.
- **Total test patients**: 47
- **Total fixed test images**: 47 (one per patient)
- **Rule is deterministic and applied once** before any training or condition construction.

### 1.2 Test Class Distribution
| Class | N test images |
|-------|--------------|
| Meningioma | 16 |
| Glioma | 18 |
| Pituitary | 13 |

### 1.3 Eligible Overlap Patients
- **Eligible**: 46 / 47 (patients with ≥ 2 total images)
- **Not eligible**: 1 (single-image patient, PID: 53669)

### 1.4 Overlap Conditions
| Condition | Overlapping patients | Eligible patients | Overlap % | Overlap images in train |
|-----------|---------------------|-------------------|-----------|------------------------|
| A | 0 | 46 | 0.0% | 0 |
| B | 23 | 46 | 50.0% | 291 |
| C | 46 | 46 | 100.0% | 596 |

**Patient overlap rate** = overlapping_patients / 46 eligible patients

---

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNet-B0 |
| Pretrained weights | ImageNet (IMAGENET1K_V1) |
| Input size | 224 × 224 × 3 |
| Grayscale handling | Repeat channel × 3 |
| Intensity preprocessing | Per-image min-max normalization to [0,1] |
| ImageNet normalization | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |
| Training augmentation | RandomHorizontalFlip + RandomRotation(15°) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 16 |
| Max epochs | 50 |
| Early stopping | patience=10 on val_loss |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Random seed | 42 |

---

## 3. Data Splits

### 3.1 Training Size (identical across all conditions)
| Condition | Train images | Val images | Test images |
|-----------|-------------|-----------|------------|
| A | 1956 | 465 | 47 |
| B | 1956 | 465 | 47 |
| C | 1956 | 465 | 47 |

### 3.2 Training Class Distribution
| Condition | Meningioma | Glioma | Pituitary |
|-----------|-----------|--------|-----------|
| A | 457 (23.4%) | 916 (46.8%) | 583 (29.8%) |
| B | 457 (23.4%) | 916 (46.8%) | 583 (29.8%) |
| C | 457 (23.4%) | 916 (46.8%) | 583 (29.8%) |

### 3.3 Validation Strategy
- **Method**: Patient-level stratified split (20%, seed=42) from Folds 1–4 only
- **Identical** validation patients and images for all three conditions
- No Fold-5 patient appears in validation
- Fixed test images are never used for training, validation, or model selection

---

## 4. Results

### 4.1 Performance Metrics
| Condition | Accuracy | Macro F1 | Balanced Acc |
|-----------|----------|----------|-------------|
| A (0%) | 0.9149 | 0.9125 | 0.9190 |
| B (50%) | 0.8723 | 0.8717 | 0.8773 |
| C (100%) | 0.9149 | 0.9110 | 0.9167 |

### 4.2 Per-Class F1 Scores
| Condition | Meningioma F1 | Glioma F1 | Pituitary F1 |
|-----------|--------------|-----------|-------------|
| A | 0.8966 | 0.9444 | 0.8966 |
| B | 0.8148 | 0.8718 | 0.9286 |
| C | 0.8571 | 0.9474 | 0.9286 |

### 4.3 Confidence Metrics
| Condition | Mean Confidence | μ(correct) | μ(incorrect) | Gap | HCER≥0.90 | HCER≥0.95 |
|-----------|----------------|-----------|-------------|-----|----------|----------|
| A | 0.9571 | 0.9559 | 0.9697 | -0.0138 | 0.0714 | 0.0789 |
| B | 0.9384 | 0.9686 | 0.7322 | 0.2364 | 0.0263 | 0.0270 |
| C | 0.9644 | 0.9666 | 0.9400 | 0.0267 | 0.0732 | 0.0541 |

### 4.4 Calibration Metrics
| Condition | ECE (15 bins) | Brier Score |
|-----------|--------------|-------------|
| A | 0.1042 | 0.1837 |
| B | 0.0796 | 0.1682 |
| C | 0.0858 | 0.1637 |

**ECE formula**: ECE = Σ_b (|B_b|/N) × |acc(B_b) − conf(B_b)|, 15 equal-width bins.
**Brier score**: BS = (1/N) Σ_i Σ_c (p_ic − y_ic)², multiclass.

---

## 5. Split Integrity

- **All 13 integrity checks passed**: YES ✅
- No test images in any training or validation set
- Training sizes identical across A, B, C
- Test sets identical across A, B, C
- Validation patients identical and disjoint from test patients
- All required overlap images present in B and C

---

## 6. Unexpected Issues

None identified during pipeline execution.

---

## 7. Preliminary Interpretation

> **IMPORTANT**: This section reports observed patterns only. Given 47 test images,
> no statistical significance can be claimed. These are pilot observations.

**Observed performance pattern (A → B → C)**:
- Accuracy: 0.9149 → 0.8723 → 0.9149
- Macro F1: 0.9125 → 0.8717 → 0.9110

**Observed confidence pattern**:
- Mean confidence: 0.9571 → 0.9384 → 0.9644

**Observed calibration pattern**:
- ECE: 0.1042 → 0.0796 → 0.0858

**Hypothesis alignment**:
The core hypothesis states that increasing patient overlap may inflate apparent
performance and predictive confidence. The pilot observations are reported
factually above. Whether the pattern aligns with, contradicts, or is inconclusive
regarding the hypothesis should be assessed after the full five-fold experiment,
which will provide substantially more test images and statistical robustness.

---

## 8. Recommendation

The Stage 2 pilot pipeline has been validated:
- Split generation is correct and reproducible
- Training runs to completion without errors
- Predictions and probabilities are generated correctly
- Calibration calculation is correctly implemented

**Recommendation**: Proceed to Stage 3 (full five-fold experiment) only after
reviewing all pilot results, figures, and the integrity report.

---

## 9. Output Files

### Tables
- `tables/stage2/fixed_test_images.csv` — 47 fixed test images with metadata
- `tables/stage2/overlap_patients_50.csv` — B condition patient selection
- `tables/stage2/split_integrity_report.csv` — split integrity summary
- `tables/stage2/training_class_distribution.csv` — class counts per condition
- `tables/stage2/pilot_performance_results.csv` — accuracy, F1, per-class metrics
- `tables/stage2/pilot_confidence_results.csv` — confidence statistics
- `tables/stage2/pilot_calibration_results.csv` — ECE and Brier score
- `tables/stage2/pilot_summary.csv` — combined summary table
- `tables/stage2/confidence_correctness.csv` — per-image confidence vs correctness
- `tables/stage2/confusion_matrix_{A,B,C}.csv` — 3×3 confusion matrices

### Predictions
- `results/stage2/predictions_condition_{A,B,C}.csv` — per-image probabilities

### Figures
- `figures/stage2/reliability_condition_{A,B,C}.png` — reliability diagrams
- `figures/stage2/reliability_comparison.png` — combined
- `figures/stage2/accuracy_vs_overlap.png`
- `figures/stage2/macro_f1_vs_overlap.png`
- `figures/stage2/confidence_vs_overlap.png`
- `figures/stage2/ece_vs_overlap.png`
- `figures/stage2/brier_vs_overlap.png`
- `figures/stage2/confidence_correctness.png`

### Training
- `tables/stage2/training_log_cond_{A,B,C}.csv` — per-epoch loss/accuracy
- `results/stage2/checkpoints/cond_{A,B,C}_best.pt` — best model checkpoints

---

**Stage 2 Pilot COMPLETE. Awaiting explicit approval before Stage 3.**
