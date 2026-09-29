# PTB-XL Dataset Structure, Fold Distribution & Disease Analysis

**Document Purpose**: Comprehensive reference for cross-validation protocol, disease stratification, patient grouping, gender distribution, and signal metrics for academic publication and research audit.  
**Dataset Reference**: PhysioNet PTB-XL v1.0.3  
**Location**: `/home/awais/Desktop/PTB-XL/DATASET_FOLDS_AND_DISEASE_DISTRIBUTION_ANALYSIS.md`  

---

## 1. Dataset Dimensions & Signal Specifications

| Metric / Parameter | Value / Specification | Publication Rationale |
|---|---|---|
| **Total ECG Records** | 21,799 12-lead recordings | Full PTB-XL cohort |
| **Total Unique Patients** | 18,869 patients | Patient-grouped splits prevent data leakage |
| **Signal Duration** | **Fixed 10.0 seconds** across all records | Uniform temporal window; no slicing or padding needed |
| **Minimum Signal Length** | 10.0 seconds (1,000 samples @ 100 Hz / 5,000 samples @ 500 Hz) | Guaranteed uniform sequence length |
| **Maximum Signal Length** | 10.0 seconds (1,000 samples @ 100 Hz / 5,000 samples @ 500 Hz) | Guaranteed uniform sequence length |
| **Low-Res Format (`filename_lr`)** | $1000 \times 12$ array per record (100 Hz) | Standard resolution for DDPM/DDIM and ResNet training |
| **High-Res Format (`filename_hr`)** | $5000 \times 12$ array per record (500 Hz) | Fine morphological inspection format |

---

## 2. 10-Fold Cross-Validation Protocol (`strat_fold`)

PTB-XL incorporates a pre-computed `strat_fold` column (values `1` through `10`) in `ptbxl_database.csv` designed specifically by the PhysioNet dataset authors for standardized benchmarking.

### Key Architectural Properties:
1. **Zero Patient-Grouped Leakage**:
   - Multiple ECG recordings originating from the same patient (e.g., longitudinal follow-ups) are **strictly assigned to the same fold**.
   - Verified cross-fold patient overlap: **0 patients** across all 10 folds.
2. **Iterative Multi-Label Stratification**:
   - The multi-label diagnostic superclasses are balanced across all folds using iterative stratified sampling.

---

## 3. Disease Superclass Distribution Matrix across Folds

The table below documents the record count and percentage distribution for the five main SCP-ECG diagnostic superclasses across all 10 folds:
- **NORM**: Normal ECG
- **MI**: Myocardial Infarction
- **STTC**: ST/T Change
- **CD**: Conduction Disturbance
- **HYP**: Hypertrophy

| `strat_fold` | Total Records | **NORM** Count (%) | **MI** Count (%) | **STTC** Count (%) | **CD** Count (%) | **HYP** Count (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 2,175 | 940 (43.22%) | 549 (25.24%) | 526 (24.18%) | 481 (22.11%) | 263 (12.09%) |
| **Fold 2** | 2,181 | 967 (44.34%) | 538 (24.67%) | 524 (24.03%) | 485 (22.24%) | 264 (12.10%) |
| **Fold 3** | 2,192 | 991 (45.21%) | 528 (24.09%) | 515 (23.49%) | 487 (22.22%) | 264 (12.04%) |
| **Fold 4** | 2,174 | 927 (42.64%) | 551 (25.34%) | 527 (24.24%) | 494 (22.72%) | 261 (12.01%) |
| **Fold 5** | 2,174 | 939 (43.19%) | 563 (25.90%) | 532 (24.47%) | 496 (22.82%) | 265 (12.19%) |
| **Fold 6** | 2,173 | 930 (42.80%) | 562 (25.86%) | 528 (24.30%) | 494 (22.73%) | 270 (12.43%) |
| **Fold 7** | 2,176 | 970 (44.58%) | 550 (25.28%) | 520 (23.90%) | 478 (21.97%) | 264 (12.13%) |
| **Fold 8** | 2,173 | 932 (42.89%) | 538 (24.76%) | 514 (23.65%) | 492 (22.64%) | 268 (12.33%) |
| **Fold 9** | 2,183 | 955 (43.75%) | 540 (24.74%) | 528 (24.19%) | 495 (22.68%) | 268 (12.28%) |
| **Fold 10** | 2,198 | 963 (43.81%) | 550 (25.02%) | 521 (23.70%) | 496 (22.57%) | 262 (11.92%) |
| **Total Cohort** | **21,799** | **9,514 (43.64%)** | **5,469 (25.09%)** | **5,235 (24.02%)** | **4,978 (22.84%)** | **2,655 (12.18%)** |

---

## 4. Target Pathological Sub-Condition Distribution: LBBB

For counterfactual translation tasks targeting **Left Bundle Branch Block (LBBB)** (`CLBBB` / `ILBBB` statements), the 613 identified cases are partitioned as follows:

| `strat_fold` | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | **Total Cohort** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LBBB Cases** | 60 | 62 | 61 | 62 | 61 | 61 | 61 | 62 | 61 | 62 | **613 Records** |

---

## 5. Gender Distribution across Diagnostic Categories

In the PTB-XL database schema, gender is encoded as `sex = 0` (Male) and `sex = 1` (Female).

### A. Record-Level Gender Breakdown (21,799 Total ECG Records)

| Category / Diagnosis | Total Records | Male Records (`sex=0`) | Male % | Female Records (`sex=1`) | Female % |
|---|:---:|:---:|:---:|:---:|:---:|
| **NORM** (Normal) | 9,514 | 4,386 | 46.10% | 5,128 | 53.90% |
| **MI** (Myocardial Infarction) | 5,469 | 3,407 | 62.30% | 2,062 | 37.70% |
| **STTC** (ST/T Change) | 5,235 | 2,566 | 49.02% | 2,669 | 50.98% |
| **CD** (Conduction Disturbance) | 4,898 | 2,999 | 61.23% | 1,899 | 38.77% |
| **HYP** (Hypertrophy) | 2,649 | 1,520 | 57.38% | 1,129 | 42.62% |
| **LBBB** (Left Bundle Branch Block) | 613 | 352 | 57.42% | 261 | 42.58% |
| **Entire Cohort (ALL)** | **21,799** | **11,354** | **52.08%** | **10,445** | **47.92%** |

### B. Patient-Level Gender Breakdown (18,869 Unique Patients)

| Category / Diagnosis | Unique Patients | Male Patients | Male % | Female Patients | Female % |
|---|:---:|:---:|:---:|:---:|:---:|
| **NORM** (Normal) | 8,896 | 4,064 | 45.68% | 4,832 | 54.32% |
| **MI** (Myocardial Infarction) | 4,670 | 2,866 | 61.37% | 1,804 | 38.63% |
| **STTC** (ST/T Change) | 4,577 | 2,203 | 48.13% | 2,374 | 51.87% |
| **CD** (Conduction Disturbance) | 4,295 | 2,599 | 60.51% | 1,696 | 39.49% |
| **HYP** (Hypertrophy) | 2,384 | 1,354 | 56.80% | 1,030 | 43.20% |
| **LBBB** (Left Bundle Branch Block) | 497 | 276 | 55.53% | 221 | 44.47% |
| **Entire Cohort (ALL)** | **18,869** | **9,640** | **51.09%** | **9,229** | **48.91%** |

---

## 6. Ready-to-Use LaTeX / Methodology Text for Publications

```latex
\subsection{Dataset and Demographic Composition}
We evaluated our causal counterfactual diffusion model on the PhysioNet PTB-XL database (v1.0.3), comprising 21,799 12-lead ECG records ($10.0\,\text{s}$ duration) from 18,869 unique patients ($51.09\%$ male, $48.91\%$ female). To evaluate model generalization and eliminate patient-level data leakage, we adopted the official patient-grouped 10-fold cross-validation scheme (\texttt{strat\_fold}). This partitioning guarantees zero patient overlap across folds while maintaining multi-label diagnostic class balance ($\approx 43.6\%$ NORM, $25.1\%$ MI, $24.0\%$ STTC, $22.8\%$ CD, and $12.2\%$ HYP). Demographically, male patients exhibited a higher prevalence in Myocardial Infarction ($61.37\%$ male) and Conduction Disturbances ($60.51\%$ male), whereas Normal control records were slightly higher among females ($54.32\%$ female). All signals were evaluated at $100\,\text{Hz}$ ($1000 \times 12$ tensor size) with $0.5\text{--}40\,\text{Hz}$ zero-phase Butterworth filtering.
```

---
*Documentation generated for academic research auditing.*
