# Final Stage 3 Results Audit

## 1. Experiment Reconstruction
- **Dataset:** Cheng brain tumor MRI dataset (3,064 images, 233 patients, 3 classes).
- **Outer evaluation:** 5 patient-disjoint folds.
- **Conditions:** A (0% overlap), B (~50% overlap), C (100% overlap).
- **Splits:** Identical validation and test sets across conditions. Overlap constructed solely by adding non-test images from test-fold patients to the training set.

## 2. 15-Run Completion Audit
|   Fold | A status   | B status   | C status   |   test images |
|-------:|:-----------|:-----------|:-----------|--------------:|
|      1 | Complete   | Complete   | Complete   |            45 |
|      2 | Complete   | Complete   | Complete   |            46 |
|      3 | Complete   | Complete   | Complete   |            48 |
|      4 | Complete   | Complete   | Complete   |            47 |
|      5 | Complete   | Complete   | Complete   |            47 |
All 15 runs successfully completed and saved checkpoints, training logs, predictions, and evaluation metrics. No missing data.

## 3. Data/split integrity
Integrity verified via `stage3_split_integrity_report.txt` and `final_manifest_gate_report.txt`. No test image leakage occurred.

## 4. Pooled Metric Verification
- Cond A accuracy: Verified (Actual: 0.8927, Expected: 0.8927)
- Cond A macro_f1: Verified (Actual: 0.8937, Expected: 0.8937)
- Cond A balanced_accuracy: Verified (Actual: 0.8950, Expected: 0.8950)
- Cond A mean_confidence: Verified (Actual: 0.9234, Expected: 0.9234)
- Cond A ece: Verified (Actual: 0.0719, Expected: 0.0719)
- Cond A brier_score: Verified (Actual: 0.1815, Expected: 0.1815)
- Cond A hcer_90: Verified (Actual: 0.0604, Expected: 0.0604)
- Cond B accuracy: Verified (Actual: 0.9356, Expected: 0.9356)
- Cond B macro_f1: Verified (Actual: 0.9371, Expected: 0.9371)
- Cond B balanced_accuracy: Verified (Actual: 0.9357, Expected: 0.9357)
- Cond B mean_confidence: Verified (Actual: 0.9406, Expected: 0.9406)
- Cond B ece: Verified (Actual: 0.0529, Expected: 0.0529)
- Cond B brier_score: Verified (Actual: 0.1177, Expected: 0.1177)
- Cond B hcer_90: Verified (Actual: 0.0378, Expected: 0.0378)
- Cond C accuracy: Verified (Actual: 0.9442, Expected: 0.9442)
- Cond C macro_f1: Verified (Actual: 0.9427, Expected: 0.9427)
- Cond C balanced_accuracy: Verified (Actual: 0.9465, Expected: 0.9465)
- Cond C mean_confidence: Verified (Actual: 0.9614, Expected: 0.9614)
- Cond C ece: Verified (Actual: 0.0346, Expected: 0.0346)
- Cond C brier_score: Verified (Actual: 0.0843, Expected: 0.0843)
- Cond C hcer_90: Verified (Actual: 0.0246, Expected: 0.0246)

## 5. Fold-level A/B/C results & 6. A vs C Analysis
**accuracy**:
  Folds 1-5 Diff: [0.0444, 0.1087, 0.0625, 0.0213, 0.0213]
  Mean/Median Diff: 0.0516 / 0.0444
  Direction: +5, -0
**macro_f1**:
  Folds 1-5 Diff: [0.0421, 0.1061, 0.0607, 0.0149, 0.0216]
  Mean/Median Diff: 0.0491 / 0.0421
  Direction: +5, -0
**balanced_accuracy**:
  Folds 1-5 Diff: [0.0417, 0.1134, 0.0566, 0.0278, 0.0208]
  Mean/Median Diff: 0.0521 / 0.0417
  Direction: +5, -0
**mean_confidence**:
  Folds 1-5 Diff: [-0.0005, 0.0682, 0.0214, 0.0494, 0.0507]
  Mean/Median Diff: 0.0379 / 0.0494
  Direction: +4, -1
**ece**:
  Folds 1-5 Diff: [-0.027, -0.0627, -0.0469, -0.0481, -0.0585]
  Mean/Median Diff: -0.0486 / -0.0481
  Direction: +0, -5
**brier_score**:
  Folds 1-5 Diff: [-0.0449, -0.1777, -0.1093, -0.0861, -0.0671]
  Mean/Median Diff: -0.0970 / -0.0861
  Direction: +0, -5
**hcer_90**:
  Folds 1-5 Diff: [-0.0256, -0.0744, -0.0262, -0.0317, -0.0346]
  Mean/Median Diff: -0.0385 / -0.0317
  Direction: +0, -5

## 7. Statistical Audit
- **Sample Size:** n = 5 paired outer folds.
- **Power:** Low statistical power due to n=5. 
- **Methodology:** The fold-level results correctly act as the independent experimental units. Image-level pooled observations were correctly omitted from formal parametric paired testing to avoid falsely treating 233 observations as independent models.
- **Conclusion:** The statistical analysis is appropriate. Emphasis on exact sign direction (+5/-5) across folds is the most robust metric.

## 8. Calibration Interpretation
**OBSERVATION:** Transitioning from 0% overlap (A) to 100% overlap (C) significantly improves apparent calibration (ECE drops from 0.0719 to 0.0346, Brier drops from 0.1815 to 0.0843) while simultaneously inflating accuracy (0.8927 to 0.9442) and confidence (0.9234 to 0.9614).

**POSSIBLE MECHANISM:** Overlap makes the test cases substantially more familiar to the model. Because the features are recognized, the model's high confidence becomes justified *on that specific leaked subset*, aligning prediction probability closely with accuracy, thus mathematically lowering ECE.

**PROVEN MECHANISM:** The experiment proves association (leakage -> better ECE), but does not intrinsically prove the neural network memorized specific anatomical structures versus generic noise. 

## 9. Patient-level Overlap Analysis
The `patient_overlap_prediction_analysis.csv` file maps prediction correctness/confidence to the number of training images for that specific PID. It supports exploratory analysis of dose-response intensity at the patient level but cannot make causal assertions.

## 10. High-Confidence Error Analysis
HCER_90 decreases sequentially: A (0.0604) > B (0.0378) > C (0.0246). 
The data strongly supports this trend; the model makes fewer confident-but-wrong predictions under leakage.

## 11. Methodological Risk Audit
No hidden issues found. Validation patients remained identical across conditions. Hyperparameters were fixed (EfficientNet-B0, LR=1e-4, 50 epochs). All probability outputs sum to 1. ECE and Brier were calculated correctly.

## 12. Supported vs Unsupported Claims
- **CLAIM 1 (Patient overlap increases apparent classification performance):** SUPPORTED.
- **CLAIM 2 (Patient overlap increases predictive confidence):** SUPPORTED.
- **CLAIM 3 (Patient overlap changes apparent calibration):** SUPPORTED.
- **CLAIM 4 (Patient overlap makes the model appear MORE calibrated according to ECE/Brier):** SUPPORTED.
- **CLAIM 5 (Patient overlap causes model memorization):** NOT FULLY SUPPORTED (The experiment proves association and familiarization, but "memorization" implies a specific internal cognitive mapping which cannot be strictly proven solely by metric outcomes).

## 13. Three Candidate Research Questions
1. "Does patient-level overlap between training and testing data distort not only classification performance but also the apparent reliability and calibration of brain-tumor MRI classifiers?" *(Strongest: perfectly encapsulates both the performance and the counter-intuitive calibration finding).*
2. "How does data leakage through patient overlap affect Expected Calibration Error and Brier scores in medical image classification?" *(Second: Good, but a bit too focused on the specific metrics rather than the broader concept of reliability).*
3. "Does patient overlap inflate predictive confidence in brain tumor MRI models?" *(Weakest: Misses the critical finding regarding calibration and ECE).*

## 14. Five Candidate Titles
1. "When Leakage Looks Reliable: Effects of Patient-Level Overlap on Performance, Confidence, and Calibration in Brain Tumor MRI Classification" *(Strongest: Accurate, captures the paradox).*
2. "The Illusion of Calibration: How Patient Overlap Distorts Predictive Reliability in Medical Imaging" *(Second: Catchy, but 'Illusion' is slightly less precise than 'Apparent').*
3. "Patient-Level Data Leakage Simultaneously Inflates Accuracy and Apparent Calibration in MRI Classification" *(Third: Very descriptive and scientifically precise).*
4. "Evaluating the Impact of Test-Set Leakage on Expected Calibration Error in Brain Tumor Models" *(Fourth: Too narrow).*
5. "Consequences of Improper Patient Splitting on the Reliability of Deep Learning for MRI" *(Fifth: A bit generic).*

## 15. Novelty Status
**Novelty status: NOT YET VERIFIED.**
To establish novelty, literature searches must specifically check for:
- Leakage effects explicitly measuring *confidence* and *calibration* (ECE/Brier) in medical imaging.
- Controlled 0/50/100% dose-response overlap experiments regarding reliability distortion.

## 16. Recommended next step
Conduct the targeted literature review to verify novelty status based on the specific calibration/leakage intersection, and then begin drafting the Methods and Results sections of the paper based on these verified metrics.
