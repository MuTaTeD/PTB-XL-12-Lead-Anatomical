# Anatomically-Decomposed SE-ResNet1D with Multi-Scale Explainability for 12-Lead ECG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.14](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official implementation and experimental reproduction repository for the manuscript:  
**"Anatomically-Decomposed Squeeze-and-Excitation 1D ResNet with Multi-Scale Explainability for 12-Lead Electrocardiogram Interpretation"**  


---

## 📌 Overview

Automated deep learning algorithms for 12-lead electrocardiography (ECG) often treat the 12 leads as arbitrary numerical channels, risking inter-lead shortcut learning and producing uncalibrated attribution heatmaps. 

This repository presents:
1. **Anatomical Lead-Territory Classifier**: A 12-lead convolutional architecture explicitly partitioned into four physiological coronary vascular branches:
   - **Inferior Territory** ($\text{II}, \text{III}, \text{aVF}$) $\to$ Right Coronary Artery (RCA)
   - **Antero-Septal Territory** ($\text{V1}-\text{V4}$) $\to$ Left Anterior Descending (LAD)
   - **Lateral Territory** ($\text{I}, \text{aVL}, \text{V5}, \text{V6}$) $\to$ Left Circumflex (LCx)
   - **Cavity Reciprocal Territory** ($\text{aVR}$) $\to$ Reciprocal Reference
2. **Stochastic Lead-Territory Dropout** ($p_{\text{drop}} = 0.15$): Regularization that masks entire vascular territories during training, eliminating spatial shortcut co-adaptation.
3. **Hierarchical Multi-Scale Explainability (XAI)**:
   - **Macro-Scale**: Territory Occlusion Sensitivity ($\Delta P$) and Learned Softmax Attention identifying culprit vascular beds.
   - **Meso-Scale**: 1D Multi-Branch Grad-CAM++ highlighting pathological cardiac waveform intervals (ST-segments, QRS complexes).
   - **Micro-Scale**: Axiomatic Integrated Gradients satisfying sample completeness ($|\sum \text{IG} - \Delta \text{Score}| < 0.02$).
4. **Clinical Visualizations**: High-resolution attribution profiles rendered over centered **3.0-second diagnostic zoom windows** with standard clinical pink ECG grids.

---

## 🏆 Key Experimental Results

### 1. Stratified 10-Fold Cross-Validation on PTB-XL (21,799 Records)

| Model Architecture | Macro ROC-AUC [95% CI] | Macro PR-AUC | Macro F1 | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1 (Flat SE-ResNet1D)** | 0.9248 [0.9189–0.9307] | 0.7208 | 0.7612 | 0.7410 | 0.9380 |
| **Model 2 (Anatomical Multi-Branch)** | 0.9295 [0.9238–0.9352] | 0.7301 | 0.7745 | 0.7580 | 0.9412 |
| **Model 3 (Territory-Dropout — Proposed)** | **0.9329 [0.9270–0.9388]** | **0.7388** | **0.7836** | **0.7690** | **0.9450** |

*Paired Wilcoxon signed-rank test confirms Model 3 statistically outperforms Model 2 ($p < 0.05$).*

### 2. Zero-Shot Out-of-Distribution Generalization on CPSC2018 (6,877 Records)

| Diagnostic Class | Evaluated Records ($N$) | ROC-AUC [95% CI] | PR-AUC | Tuned F1 | Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Normal Sinus Rhythm (`NORM`) | 918 | **0.9049 [0.88–0.92]** | 0.6153 | 0.5816 | 0.7723 |
| Conduction Disturbances (`CD`) | 4,988 | **0.8462 [0.82–0.87]** | 0.9362 | 0.8306 | 0.7773 |
| ST/T Changes (`STTC`) | 1,087 | 0.6036 [0.57–0.63] | 0.1876 | 0.3229 | 0.7065 |
| **Macro Aggregate** | **6,877** | **0.7849 [0.75–0.81]** | **0.5797** | **0.5784** | **0.7520** |

### 3. Adebayo et al. (NeurIPS 2018) Parameter Randomization Sanity Check ($N=100$)

| Attribution Method | Randomization Condition | Pearson $r$ (Mean ± SD) | Spearman $\rho$ (Mean ± SD) | Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **Grad-CAM++** | Top-Layer Randomization | $-0.1600 \pm 0.3452$ | $-0.2383 \pm 0.4027$ | Sensitive (Gradients Inverted) |
| | Cascading Randomization | $+0.1814 \pm 0.1850$ | $+0.1456 \pm 0.2209$ | Sensitive (Intermediate Fluctuation) |
| | **Full Network Randomization** | $\mathbf{+0.0935 \pm 0.1615}$ | $\mathbf{+0.0676 \pm 0.1779}$ | **Passed (Full Correlation Collapse)** |
| **Integrated Gradients** | Top-Layer Randomization | $-0.2886 \pm 0.4490$ | $-0.1915 \pm 0.3491$ | Sensitive (Gradients Inverted) |
| | Cascading Randomization | $+0.1327 \pm 0.2636$ | $+0.0481 \pm 0.1154$ | Sensitive (Intermediate Fluctuation) |
| | **Full Network Randomization** | $\mathbf{+0.0015 \pm 0.1408}$ | $\mathbf{-0.0075 \pm 0.0295}$ | **Passed (Absolute Zero Collapse)** |

---

## 📂 Repository Structure

```text
├── manuscript/                     # Complete LaTeX manuscript & figures
│   ├── main.tex                    # Double-column template source
│   ├── references.bib              # Verified BibTeX citations
│   ├── main.pdf                    # Compiled Camera-Ready PDF (19 pages)
│   ├── main_single_column_review.pdf # Compiled 12pt Review Copy (23 pages)
│   └── figures/                    # High-resolution clinical figures & profiles
├── checkpoints/                    # Saved model weights (.h5) across all 10 folds
│   ├── se_resnet_anatomical_territory_dropout_fold_1_best.h5 ... fold_10_best.h5
│   └── training_history_*.csv      # Per-epoch training & validation history
├── experiments/                    # Quantitative audit CSVs and benchmark metrics
│   ├── full_cohort_fold10_attribution_audit.csv  # Full-cohort audit (N=2,198)
│   ├── adebayo_sanity_check_results.csv          # Adebayo sanity check summary (N=100)
│   ├── adebayo_sanity_check_per_case.csv         # 100 individual case metrics
│   └── cpsc2018_zeroshot_metrics.csv             # External zero-shot metrics
├── docs/                           # Documentation and experimental reports
│   └── experiments/                # Individual phase-wise logs & benchmark analyses
├── run_10fold_anatomical_territory_dropout_classifier.py # 10-fold training script
├── run_full_cohort_xai_audit.py    # Full-cohort Fold 10 attribution audit script
├── run_adebayo_sanity_check.py     # Parameter randomization sanity check script
├── evaluate_cpsc2018_zeroshot.py   # External zero-shot evaluation script
├── compile_single.py               # Single-column PDF generation pipeline
├── RESULTS_SUMMARY.md              # Detailed numerical results & statistical tests
├── AUDIT_LOG_AND_SYSTEM_VERIFICATION.md # Complete audit and reproduction trail
└── review3_feedback.md             # Comprehensive point-by-point reviewer rebuttal
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
conda create -n ptbxl-gpu python=3.10 -y
conda activate ptbxl-gpu
pip install tensorflow[and-cuda]==2.14.0 numpy pandas scipy scikit-learn wfdb matplotlib
```

### 2. Run Full-Cohort Attribution Audit ($N = 2,198$)

```bash
python run_full_cohort_xai_audit.py
```

### 3. Run Adebayo Parameter Randomization Sanity Checks ($N = 100$)

```bash
python run_adebayo_sanity_check.py
```

### 4. Run External Zero-Shot Generalization on CPSC2018

```bash
python evaluate_cpsc2018_zeroshot.py
```

---

## 📄 License & Citation

This project is licensed under the MIT License. If you find this codebase or manuscript useful in your research, please cite:

```bibtex
@article{lodhi2026anatomical,
  title={Anatomically-Decomposed Squeeze-and-Excitation 1D ResNet with Multi-Scale Explainability for 12-Lead Electrocardiogram Interpretation},
  author={Lodhi, Awais Muhammad and Mustafa, Ghulam},
  journal={Preprint (Under Review)},
  year={2026}
}
```
