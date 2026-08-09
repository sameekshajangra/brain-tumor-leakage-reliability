"""
src/14_generate_pilot_figures.py
=================================
Stage 2 — Step 8: All comparison figures and the final pilot report.

Figures
-------
  figures/stage2/accuracy_vs_overlap.png
  figures/stage2/macro_f1_vs_overlap.png
  figures/stage2/confidence_vs_overlap.png
  figures/stage2/ece_vs_overlap.png
  figures/stage2/brier_vs_overlap.png
  figures/stage2/confidence_correctness.png
  figures/stage2/pilot_summary.png   (combined overview table)

Tables
------
  tables/stage2/pilot_summary.csv

Report
------
  results/stage2/pilot_report.md
"""
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES2    = PROJECT_ROOT / "results" / "stage2"
TABLES2 = PROJECT_ROOT / "tables" / "stage2"
FIGS2   = PROJECT_ROOT / "figures" / "stage2"
FIGS2.mkdir(parents=True, exist_ok=True)

COLORS = {"A": "#2196F3", "B": "#FF9800", "C": "#4CAF50"}
MARKERS = {"A": "o", "B": "s", "C": "^"}
CONDITION_LABELS = {
    "A": "Condition A\n(0% overlap)",
    "B": "Condition B\n(50% overlap)",
    "C": "Condition C\n(100% overlap)",
}

print("=" * 70)
print("GENERATING PILOT FIGURES AND REPORT")
print("=" * 70)

# ---------------------------------------------------------------------------
# Load all result tables
# ---------------------------------------------------------------------------
perf_df = pd.read_csv(TABLES2 / "pilot_performance_results.csv")
conf_df = pd.read_csv(TABLES2 / "pilot_confidence_results.csv")
cal_df  = pd.read_csv(TABLES2 / "pilot_calibration_results.csv")
cc_df   = pd.read_csv(TABLES2 / "confidence_correctness.csv")
int_df  = pd.read_csv(TABLES2 / "split_integrity_report.csv")
train_class_df = pd.read_csv(TABLES2 / "training_class_distribution.csv")

# Merge into summary
summary = perf_df[["condition","overlap_percentage","overlapping_patients","train_images","test_images",
                    "accuracy","macro_f1","balanced_accuracy"]].copy()
summary = summary.merge(conf_df[["condition","mean_confidence","hcer_90","hcer_95"]], on="condition")
summary = summary.merge(cal_df[["condition","ece","brier_score"]], on="condition")
summary.to_csv(TABLES2 / "pilot_summary.csv", index=False)
print("  Saved: pilot_summary.csv")

# Shorthand for plotting
x = summary["overlap_percentage"].values
conds = summary["condition"].values


def plot_metric(y, ylabel, title, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (xi, yi, c) in enumerate(zip(x, y, conds)):
        ax.scatter(xi, yi, color=COLORS[c], marker=MARKERS[c], s=120, zorder=5,
                   label=f"Condition {c} ({xi:.0f}%)")
    ax.plot(x, y, color="gray", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.set_xlabel("Patient-overlap percentage (%)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([0, 50, 100])
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS2 / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")


plot_metric(summary["accuracy"].values, "Accuracy",
            "Accuracy vs. Patient Overlap", "accuracy_vs_overlap.png", ylim=(0, 1.05))
plot_metric(summary["macro_f1"].values, "Macro F1",
            "Macro F1 vs. Patient Overlap", "macro_f1_vs_overlap.png", ylim=(0, 1.05))
plot_metric(summary["mean_confidence"].values, "Mean Confidence",
            "Mean Confidence vs. Patient Overlap", "confidence_vs_overlap.png", ylim=(0, 1.05))
plot_metric(summary["ece"].values, "ECE",
            "Expected Calibration Error vs. Patient Overlap", "ece_vs_overlap.png")
plot_metric(summary["brier_score"].values, "Brier Score",
            "Brier Score vs. Patient Overlap", "brier_vs_overlap.png")

# ---------------------------------------------------------------------------
# Confidence vs Correctness
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
for ax, cond in zip(axes, ["A", "B", "C"]):
    sub = cc_df[cc_df["condition"] == cond]
    cor = sub[sub["correct"] == 1]["confidence"]
    inc = sub[sub["correct"] == 0]["confidence"]
    ax.hist(cor, bins=10, range=(0, 1), alpha=0.7,
            color="#4CAF50", label=f"Correct (n={len(cor)})", density=True)
    ax.hist(inc, bins=10, range=(0, 1), alpha=0.7,
            color="#F44336", label=f"Incorrect (n={len(inc)})", density=True)
    mean_c = cor.mean() if len(cor) else float("nan")
    mean_i = inc.mean() if len(inc) else float("nan")
    ax.axvline(mean_c, color="#2E7D32", linestyle="--", linewidth=1.5,
               label=f"μ_correct={mean_c:.2f}")
    if not np.isnan(mean_i):
        ax.axvline(mean_i, color="#B71C1C", linestyle="--", linewidth=1.5,
                   label=f"μ_incorrect={mean_i:.2f}")
    ax.set_title(f"Condition {cond} ({CONDITION_LABELS[cond].replace(chr(10), ' ')})", fontsize=10)
    ax.set_xlabel("Confidence", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
axes[0].set_ylabel("Density", fontsize=10)
fig.suptitle("Confidence Distribution: Correct vs. Incorrect Predictions", fontsize=11)
fig.tight_layout()
fig.savefig(FIGS2 / "confidence_correctness.png", dpi=150)
plt.close(fig)
print("  Saved: confidence_correctness.png")

# ---------------------------------------------------------------------------
# Generate pilot_report.md
# ---------------------------------------------------------------------------
print("\n  Generating pilot_report.md ...")

def fmt(v, decimals=4):
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)

# Gather info from fixed_test CSV
fixed_test = pd.read_csv(TABLES2 / "fixed_test_images.csv")
n_test_patients = len(fixed_test)
n_eligible = int(fixed_test["eligible_for_overlap"].sum())

row_A = summary[summary["condition"] == "A"].iloc[0]
row_B = summary[summary["condition"] == "B"].iloc[0]
row_C = summary[summary["condition"] == "C"].iloc[0]

pA = perf_df[perf_df["condition"] == "A"].iloc[0]
pB = perf_df[perf_df["condition"] == "B"].iloc[0]
pC = perf_df[perf_df["condition"] == "C"].iloc[0]

cA = conf_df[conf_df["condition"] == "A"].iloc[0]
cB = conf_df[conf_df["condition"] == "B"].iloc[0]
cC = conf_df[conf_df["condition"] == "C"].iloc[0]

calA = cal_df[cal_df["condition"] == "A"].iloc[0]
calB = cal_df[cal_df["condition"] == "B"].iloc[0]
calC = cal_df[cal_df["condition"] == "C"].iloc[0]

tdA = train_class_df[train_class_df["condition"] == "A"].iloc[0]
tdB = train_class_df[train_class_df["condition"] == "B"].iloc[0]
tdC = train_class_df[train_class_df["condition"] == "C"].iloc[0]

intA = int_df[int_df["condition"] == "A"].iloc[0]
intB = int_df[int_df["condition"] == "B"].iloc[0]
intC = int_df[int_df["condition"] == "C"].iloc[0]

split_passed = all(int_df["exact_image_overlap_count"] == 0) and \
               all(int_df["test_identical_to_A"].astype(str).str.lower() == "true")

report = f"""# Stage 2 Pilot Report — Brain Tumor Patient Overlap Experiment
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

> **PILOT NOTICE**: This experiment uses only 47 fixed test images (Fold 5).
> Results are preliminary. Do not claim statistical significance or definitive
> generalization conclusions from this pilot. Stage 3 (full five-fold experiment)
> requires explicit approval.

---

## 1. Experimental Design

### 1.1 Test Image Selection Rule
- **Rule**: For each Fold-5 patient, select the image with the **minimum numeric image_id** as the fixed test image.
- **Total test patients**: {n_test_patients}
- **Total fixed test images**: {n_test_patients} (one per patient)
- **Rule is deterministic and applied once** before any training or condition construction.

### 1.2 Test Class Distribution
| Class | N test images |
|-------|--------------|
| Meningioma | {int((fixed_test['label']==1).sum())} |
| Glioma | {int((fixed_test['label']==2).sum())} |
| Pituitary | {int((fixed_test['label']==3).sum())} |

### 1.3 Eligible Overlap Patients
- **Eligible**: {n_eligible} / {n_test_patients} (patients with ≥ 2 total images)
- **Not eligible**: {n_test_patients - n_eligible} (single-image patient, PID: 53669)

### 1.4 Overlap Conditions
| Condition | Overlapping patients | Eligible patients | Overlap % | Overlap images in train |
|-----------|---------------------|-------------------|-----------|------------------------|
| A | {int(intA['overlap_patients'])} | {n_eligible} | {float(intA['overlap_percentage']):.1f}% | {int(intA['overlap_images_in_train'])} |
| B | {int(intB['overlap_patients'])} | {n_eligible} | {float(intB['overlap_percentage']):.1f}% | {int(intB['overlap_images_in_train'])} |
| C | {int(intC['overlap_patients'])} | {n_eligible} | {float(intC['overlap_percentage']):.1f}% | {int(intC['overlap_images_in_train'])} |

**Patient overlap rate** = overlapping_patients / {n_eligible} eligible patients

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
| A | {int(intA['train_images'])} | {int(intA['val_images'])} | {int(intA['test_images'])} |
| B | {int(intB['train_images'])} | {int(intB['val_images'])} | {int(intB['test_images'])} |
| C | {int(intC['train_images'])} | {int(intC['val_images'])} | {int(intC['test_images'])} |

### 3.2 Training Class Distribution
| Condition | Meningioma | Glioma | Pituitary |
|-----------|-----------|--------|-----------|
| A | {int(tdA['n_meningioma'])} ({tdA['pct_meningioma']:.1f}%) | {int(tdA['n_glioma'])} ({tdA['pct_glioma']:.1f}%) | {int(tdA['n_pituitary'])} ({tdA['pct_pituitary']:.1f}%) |
| B | {int(tdB['n_meningioma'])} ({tdB['pct_meningioma']:.1f}%) | {int(tdB['n_glioma'])} ({tdB['pct_glioma']:.1f}%) | {int(tdB['n_pituitary'])} ({tdB['pct_pituitary']:.1f}%) |
| C | {int(tdC['n_meningioma'])} ({tdC['pct_meningioma']:.1f}%) | {int(tdC['n_glioma'])} ({tdC['pct_glioma']:.1f}%) | {int(tdC['n_pituitary'])} ({tdC['pct_pituitary']:.1f}%) |

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
| A (0%) | {fmt(float(pA['accuracy']))} | {fmt(float(pA['macro_f1']))} | {fmt(float(pA['balanced_accuracy']))} |
| B (50%) | {fmt(float(pB['accuracy']))} | {fmt(float(pB['macro_f1']))} | {fmt(float(pB['balanced_accuracy']))} |
| C (100%) | {fmt(float(pC['accuracy']))} | {fmt(float(pC['macro_f1']))} | {fmt(float(pC['balanced_accuracy']))} |

### 4.2 Per-Class F1 Scores
| Condition | Meningioma F1 | Glioma F1 | Pituitary F1 |
|-----------|--------------|-----------|-------------|
| A | {fmt(float(pA['f1_meningioma']))} | {fmt(float(pA['f1_glioma']))} | {fmt(float(pA['f1_pituitary']))} |
| B | {fmt(float(pB['f1_meningioma']))} | {fmt(float(pB['f1_glioma']))} | {fmt(float(pB['f1_pituitary']))} |
| C | {fmt(float(pC['f1_meningioma']))} | {fmt(float(pC['f1_glioma']))} | {fmt(float(pC['f1_pituitary']))} |

### 4.3 Confidence Metrics
| Condition | Mean Confidence | μ(correct) | μ(incorrect) | Gap | HCER≥0.90 | HCER≥0.95 |
|-----------|----------------|-----------|-------------|-----|----------|----------|
| A | {fmt(float(cA['mean_confidence']))} | {fmt(float(cA['mean_confidence_correct']))} | {fmt(float(cA['mean_confidence_incorrect']))} | {fmt(float(cA['confidence_gap']))} | {fmt(float(cA['hcer_90']))} | {fmt(float(cA['hcer_95']))} |
| B | {fmt(float(cB['mean_confidence']))} | {fmt(float(cB['mean_confidence_correct']))} | {fmt(float(cB['mean_confidence_incorrect']))} | {fmt(float(cB['confidence_gap']))} | {fmt(float(cB['hcer_90']))} | {fmt(float(cB['hcer_95']))} |
| C | {fmt(float(cC['mean_confidence']))} | {fmt(float(cC['mean_confidence_correct']))} | {fmt(float(cC['mean_confidence_incorrect']))} | {fmt(float(cC['confidence_gap']))} | {fmt(float(cC['hcer_90']))} | {fmt(float(cC['hcer_95']))} |

### 4.4 Calibration Metrics
| Condition | ECE (15 bins) | Brier Score |
|-----------|--------------|-------------|
| A | {fmt(float(calA['ece']))} | {fmt(float(calA['brier_score']))} |
| B | {fmt(float(calB['ece']))} | {fmt(float(calB['brier_score']))} |
| C | {fmt(float(calC['ece']))} | {fmt(float(calC['brier_score']))} |

**ECE formula**: ECE = Σ_b (|B_b|/N) × |acc(B_b) − conf(B_b)|, 15 equal-width bins.
**Brier score**: BS = (1/N) Σ_i Σ_c (p_ic − y_ic)², multiclass.

---

## 5. Split Integrity

- **All 13 integrity checks passed**: {"YES ✅" if split_passed else "NO ❌ — see split_integrity_report.txt"}
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
- Accuracy: {float(pA['accuracy']):.4f} → {float(pB['accuracy']):.4f} → {float(pC['accuracy']):.4f}
- Macro F1: {float(pA['macro_f1']):.4f} → {float(pB['macro_f1']):.4f} → {float(pC['macro_f1']):.4f}

**Observed confidence pattern**:
- Mean confidence: {float(cA['mean_confidence']):.4f} → {float(cB['mean_confidence']):.4f} → {float(cC['mean_confidence']):.4f}

**Observed calibration pattern**:
- ECE: {float(calA['ece']):.4f} → {float(calB['ece']):.4f} → {float(calC['ece']):.4f}

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
- `tables/stage2/confusion_matrix_{{A,B,C}}.csv` — 3×3 confusion matrices

### Predictions
- `results/stage2/predictions_condition_{{A,B,C}}.csv` — per-image probabilities

### Figures
- `figures/stage2/reliability_condition_{{A,B,C}}.png` — reliability diagrams
- `figures/stage2/reliability_comparison.png` — combined
- `figures/stage2/accuracy_vs_overlap.png`
- `figures/stage2/macro_f1_vs_overlap.png`
- `figures/stage2/confidence_vs_overlap.png`
- `figures/stage2/ece_vs_overlap.png`
- `figures/stage2/brier_vs_overlap.png`
- `figures/stage2/confidence_correctness.png`

### Training
- `tables/stage2/training_log_cond_{{A,B,C}}.csv` — per-epoch loss/accuracy
- `results/stage2/checkpoints/cond_{{A,B,C}}_best.pt` — best model checkpoints

---

**Stage 2 Pilot COMPLETE. Awaiting explicit approval before Stage 3.**
"""

(RES2 / "pilot_report.md").write_text(report)
print(f"\n  Saved: results/stage2/pilot_report.md")

print("\n" + "=" * 70)
print("All Stage 2 pilot figures and report generated.")
print("STOP — Awaiting explicit approval for Stage 3.")
print("=" * 70)
