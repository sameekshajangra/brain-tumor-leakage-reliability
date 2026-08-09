# Data Directory

The original MRI dataset is **not included** in this repository to protect
patient-sensitive data and due to file-size constraints.

## Obtaining the Dataset

The dataset is the official **Cheng Brain Tumor Dataset**:

> Jun Cheng, "brain tumor dataset," Figshare,
> DOI: [10.6084/m9.figshare.1512427.v8](https://doi.org/10.6084/m9.figshare.1512427.v8)

Download all four ZIP files and `cvind.mat` from the Figshare record:

```
brainTumorDataPublic_1766.zip
brainTumorDataPublic_7671532.zip
brainTumorDataPublic_15332298.zip
brainTumorDataPublic_22993064.zip
cvind.mat
```

Place them in a directory and set the environment variable:

```bash
export BRAIN_TUMOR_DATA_DIR=/path/to/your/dataset/folder
```

Then run the Stage 1 pipeline in order:

```bash
python src/01_inspect_dataset.py
python src/02_build_metadata.py
python src/03_verify_folds.py
python src/04_patient_analysis.py
python src/05_duplicate_audit.py
python src/06_overlap_feasibility.py
```

## Expected Dataset Characteristics

| Property | Expected Value |
|---|---|
| Total images | 3,064 |
| Total patients | 233 |
| Meningioma (label=1) | 708 |
| Glioma (label=2) | 1,426 |
| Pituitary (label=3) | 930 |
| Cross-validation folds | 5 (patient-disjoint) |

*These numbers are verified programmatically from the actual files;
the scripts will stop and report if any discrepancy is found.*
