# Zero-Shot Generalization Benchmark of Anatomical Territory-Dropout SE-ResNet1D on CPSC2018 Dataset

## Executive Summary
This document outlines the zero-shot external validation of **Model 3 (Anatomical Territory-Dropout SE-ResNet1D)** on the **CPSC2018 dataset** (`/home/awais/Desktop/Computing In Cardialogy/training/cpsc_2018`). The objective is to quantify out-of-distribution generalization capability, domain shift resilience, and multi-label classification fidelity without fine-tuning or re-training.

---

## 1. Dataset & Preprocessing Pipeline

### A. Signal Frequency & Length Normalization
- **Source Sampling Rate**: CPSC2018 records are sampled at **500 Hz**.
- **Target Sampling Rate**: Model 3 requires **100 Hz** input ($T = 1000$ time steps, $C = 12$ channels).
- **Downsampling Method**: Apply anti-aliasing polyphase resampling (`scipy.signal.resample_poly` with ratio $1:5$).
- **Duration Handling**:
  - **$> 10$ seconds** (64.72% / 4,451 records): Head/center crop to exactly 10.0 seconds (1000 points at 100 Hz).
  - **$= 10$ seconds** (35.13% / 2,416 records): Direct 1:1 mapping (1000 points).
  - **$< 10$ seconds** (0.15% / 10 records): Symmetric edge-padding to 1000 points to retain 100% of dataset records without data loss.
- **Lead Standardization**: Ensure 12-lead ordering matches standard PTB-XL format (`I`, `II`, `III`, `aVR`, `aVL`, `aVF`, `V1`, `V2`, `V3`, `V4`, `V5`, `V6`).
- **Amplitude Normalization**: Apply per-lead $z$-score standardization consistent with Model 3 training.

### B. Diagnostic Ontology & Mapping Strategy
The CPSC2018 dataset contains 6,877 records mapped via SNOMED CT codes. We establish a rigorous mapping to PTB-XL's 5 diagnostic superclasses ($NORM$, $MI$, $STTC$, $CD$, $HYP$) based on PhysioNet/CinC Challenge standards:

| CPSC2018 Condition | SNOMED Code | Count | PTB-XL Superclass |
| :--- | :---: | :---: | :---: |
| **Normal Sinus Rhythm** | `426783006` | 918 | **`NORM`** |
| **Right Bundle Branch Block (RBBB)** | `59118001` | 1,857 | **`CD`** |
| **Atrial Fibrillation (AF)** | `164889003` | 1,221 | **`CD`** |
| **ST-Segment Depression (STD)** | `429622005` | 869 | **`STTC`** |
| **First-Degree AV Block (1AVB)** | `270492004` | 722 | **`CD`** |
| **Premature Ventricular Contraction (PVC)** | `164884008` | 700 | **`CD`** |
| **Premature Atrial Contraction (PAC)** | `284470004` | 616 | **`CD`** |
| **Left Bundle Branch Block (LBBB)** | `164909002` | 236 | **`CD`** |
| **ST-Segment Elevation (STE)** | `164931005` | 220 | **`STTC`** |

---

## 2. Proposed System Architecture & Scripts

### A. Dedicated Execution Engine
#### [`evaluate_cpsc2018_zeroshot.py`](file:///home/awais/Desktop/PTB-XL/evaluate_cpsc2018_zeroshot.py)
- **Model Loader**: Load Model 3 (`checkpoints/se_resnet_anatomical_territory_dropout_fold_9_best.h5`).
- **Data Loader**: Iterate through all 6,877 `.hea` and `.mat` files in CPSC2018.
- **Batch Inference**: Perform GPU-accelerated batch prediction ($Batch = 128$).
- **Metric Computation**: Calculate Macro ROC-AUC, Per-Class ROC-AUC, F1-Score, Sensitivity, Specificity, and PR-AUC.
- **Reporting**: Save performance metrics to `experiments/cpsc2018_zeroshot_metrics.csv` and generate confusion matrices/ROC curves.

---

## 3. Verification Plan

### Automated Testing & Benchmark Validation
1. Execute `evaluate_cpsc2018_zeroshot.py`.
2. Verify zero-shot classification performance across 6,877 records.
3. Compare CPSC2018 zero-shot ROC-AUC against PTB-XL Fold 10 internal test set performance to quantify domain generalization decay.
4. Log all findings in `AUDIT_LOG_AND_SYSTEM_VERIFICATION.md` and `EXPERIMENT_CPSC2018_ZEROSHOT.md`.
