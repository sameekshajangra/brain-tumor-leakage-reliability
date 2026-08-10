# Training Composition Transition Audit Report

**Stage 3 — Final Pre-Manuscript Audit (Read-Only)**  
*No model was retrained. No split was modified. No prediction was altered.*

---

## Summary of Findings

The audit resolves the apparent discrepancy between the number of overlap patients *introduced* and the net change in unique training patients. The accounting is mathematically exact and reveals a clean, deterministic displacement mechanism that is an inherent consequence of the fixed-training-size constraint.

---

## Audit Table — Full Results

| Fold | Transition | N_train | Unique A | Unique X | Δ unique | Overlap pts in X_train | Displaced non-test pts | Pool imgs removed |
|------|-----------|---------|----------|----------|----------|------------------------|------------------------|------------------|
| 1    | A→B       | 1996    | 150      | 166      | +16      | 22                     | 6                      | 0                |
| 1    | A→C       | 1996    | 150      | 185      | +35      | 43                     | 8                      | 0                |
| 2    | A→B       | 1898    | 150      | 165      | +15      | 22                     | 7                      | 0                |
| 2    | A→C       | 1898    | 150      | 178      | +28      | 45                     | 17                     | 0                |
| 3    | A→B       | 1981    | 148      | 168      | +20      | 23                     | 3                      | 0                |
| 3    | A→C       | 1981    | 148      | 182      | +34      | 46                     | 12                     | 0                |
| 4    | A→B       | 2009    | 149      | 164      | +15      | 23                     | 8                      | 0                |
| 4    | A→C       | 2009    | 149      | 182      | +33      | 46                     | 13                     | 0                |
| 5    | A→B       | 1918    | 149      | 166      | +17      | 23                     | 6                      | 0                |
| 5    | A→C       | 1918    | 149      | 182      | +33      | 46                     | 13                     | 0                |

**Columns:** "Pool imgs removed" = number of pool (non-test-fold) patient images physically removed from a training slot. In all 10 cases this is **0**. See explanation below.

---

## A. Why does Δ unique patients < n overlap patients introduced?

The `fill_remaining()` function forces overlap images into the training set first, then fills the remaining `N_train - n_forced` slots from the original non-test pool (`train_pool`). The pool contains only non-test-fold patients.

Some pool patients in Condition A were represented by many images. When overlap images consume a fixed number of training slots, the `fill_remaining()` algorithm samples *fewer images per remaining pool patient* to hit the class-distribution target. If a pool patient's total images fall to zero after this resampling — because their class quota is already met by other pool patients — **they are dropped entirely** from the training set.

This is the displacement mechanism:
- **Overlap images consumed:** e.g., 497 images (Fold 1, A→C)
- **Pool images still needed:** `N_train − 497 = 1499`
- These 1499 slots are filled from pool patients who were never in the test fold
- Pool patients whose class was already adequately covered by other pool images are simply not selected
- Result: 8 non-test pool patients are dropped entirely (Fold 1, A→C)

The key finding: **no pool images are *forcibly removed* from training.** The pool is only ever *sampled down* to fill remaining slots. The displacement is a selection effect, not a deletion.

Formally: `Δ unique patients = n_overlap_patients_introduced − n_pool_patients_dropped`

Fold-level verification:

| Fold | A→C: overlap introduced | pool patients dropped | net Δ |
|------|------------------------|----------------------|-------|
| 1    | 43                     | 8                    | +35   |
| 2    | 45                     | 17                   | +28   |
| 3    | 46                     | 12                   | +34   |
| 4    | 46                     | 13                   | +33   |
| 5    | 46                     | 13                   | +33   |

The arithmetic closes exactly in every fold.

---

## B. Which patients are displaced?

The displaced patients are **non-test-fold pool patients whose images were not selected** by `fill_remaining()` because the class distribution target was already met by other pool patients. Specifically, `fill_remaining()` samples pool images sorted by `image_id` (ascending, deterministic), class by class. Once the per-class quota is met, remaining pool patients of that class are simply not drawn. These are pool patients with large image counts that "overflow" the quota once overlap images of the same class have consumed some slots.

The displacement is determined deterministically by: (1) the forced overlap images, (2) the class proportions of Condition A, (3) the `image_id`-sorted sampling order in `fill_remaining()`. It is fully reproducible from the fixed seed and the frozen split files.

---

## C. Is the displacement intentional and deterministic?

**Yes on both counts.**

The algorithm is designed to maintain exact training size (`N_train`) and exact class distribution. Displacement of some pool patients is a mathematical consequence of that constraint when overlap images are injected. It is deterministic because `fill_remaining()` uses a fixed sort order (`image_id` ascending) and the same class-distribution target (Condition A proportions) across all conditions.

---

## D. Does this reveal a split-integrity problem?

**No.**

The critical integrity controls all hold:
- ✓ Identical `N_train` across A/B/C within every fold
- ✓ Identical per-class image counts across A/B/C within every fold (integrity check passed in previous analysis)
- ✓ Fixed test images are never in training under any condition
- ✓ Validation patients are never in training under any condition
- ✓ All pool images removed from B/C were non-test-fold patients

The displacement is a *compositional* consequence of the manipulation, not an integrity failure. It does not constitute test leakage, class imbalance, or randomness.

---

## E. Does this materially change the interpretation of the experiment?

**No — but it must be disclosed precisely.**

The experiment demonstrates that patient-level overlap in training data is *associated with* higher accuracy, higher confidence, and apparently improved calibration. The displacement finding adds the following nuance: the improvement cannot be attributed to the **addition of test-patient images alone** because the training set composition also changes in a second way — some non-test pool patients are no longer represented. However:

1. The displaced patients are not test-fold patients. Their removal cannot create test-set familiarity.
2. The displacement is small (3–17 patients out of 148–150 total pool patients per fold).
3. The class distribution remains controlled, so the displacement is class-balanced.
4. The observed effects are consistent and monotonic across all five folds in the primary direction, which argues against the displaced patients being the primary driver of performance change.

The experiment remains a valid demonstration that *the combination of adding test-patient images under a fixed-size constraint* is associated with the observed effects. It correctly isolates the practical real-world scenario where a researcher inadvertently shares patients between train and test — which always involves both addition and implicit displacement.

---

## F. Exact Manuscript Wording

### Methods — Training composition

> "To maintain identical training-image counts across all three conditions, we employed a fixed-size training construction procedure. Condition A consisted of all available non-test, non-validation images (the full training pool). For Conditions B and C, overlap images (non-test images from selected test-fold patients) were forced into the training set first; remaining training slots were then filled from the original pool using a class-proportional deterministic sampling procedure (`fill_remaining`, seeded at 42, sampling pool images in ascending image-ID order). As a consequence of this fixed-size constraint, a small number of non-test-fold patients were not selected to fill remaining slots in Conditions B and C (range: 3–17 patients per fold for A→C, mean: 12.6). These displaced patients are non-test-fold individuals whose class quota was already met by other pool patients after overlap images were injected. The displacement is fully deterministic and does not introduce test-patient images or class imbalance."

### Methods — Acknowledgement of composition change

> "Within each outer fold, Conditions A, B, and C have identical total training-image counts (N_train range: 1,898–2,009) and identical per-class image distributions (verified from raw manifests). The three conditions differ in (1) the number of unique training patients, (2) whether any test-fold patients appear in training, and (3) which non-test-fold patients fill remaining training slots. These compositional differences are inherent to the overlap manipulation under a fixed-size constraint and are acknowledged as a limitation of the design."

### Limitations — Displacement limitation

> "Third, the fixed-training-size constraint required that introducing test-patient images (for Conditions B and C) be balanced by not selecting a small number of non-test-fold pool patients in those conditions (3–17 per fold). While the class distribution was preserved and the displaced patients were not test-fold individuals, this means the three conditions differ in training-patient composition in two ways simultaneously: addition of test-fold patients and displacement of a small number of pool patients. These effects cannot be fully separated within the current design."

---

## Prediction File Row Count — Correction

**`patient_overlap_prediction_analysis.csv` contains 699 data rows** (not 700 as previously stated). This was confirmed directly from the file. The discrepancy was an off-by-one error in previous reporting; the correct count is **699**.

---

## Final Safety Check

- **No model was retrained.** ✓
- **No split was modified.** ✓  
- **No prediction was modified.** ✓
- **No existing result was overwritten.** ✓
- **No new scientific experiment was introduced.** ✓
