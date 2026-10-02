# Anatomically-Decomposed SE-ResNet1D with Multi-Scale Explainability for 12-Lead ECG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.14](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official implementation and experimental reproduction repository for the manuscript:  
**"Anatomically-Decomposed Squeeze-and-Excitation 1D ResNet with Multi-Scale Explainability for 12-Lead Electrocardiogram Interpretation"**  

---

## 📌 Overview

Automated deep learning models for 12-lead electrocardiograms (ECGs) typically treat the 12 leads as an undifferentiated spatial matrix, risking inter-lead shortcut learning and producing uncalibrated attribution heatmaps. 

This repository presents:
1. **Anatomical Multi-Branch Architecture**: A 12-lead convolutional architecture structurally partitioned into four dedicated sub-networks corresponding to standard clinical regional cardiac walls:
   - **Inferior Leads** ($\text{II}, \text{III}, \text{aVF}$): Assesses the inferior ventricular wall.
   - **Antero-Septal Leads** ($\text{V1}-\text{V4}$): Evaluates the anterior myocardium and interventricular septum.
   - **Lateral Leads** ($\text{I}, \text{aVL}, \text{V5}, \text{V6}$): Evaluates the lateral ventricular wall.
   - **Cavity Reciprocal Lead** ($\text{aVR}$): Provides a reciprocal view of the ventricular cavity and basal septum.
2. **Dual-Regularization Training Protocol**:
   - **Runtime Input Territory Dropout** ($p_{\text{drop}} = 0.15$): Stochastically masks entire regional lead groups during training, eliminating shortcut co-adaptation and directly mirroring inference-time occlusion sensitivity.
   - **Runtime Temporal Time-Masking Cutout** ($p_{\text{cutout}} = 0.70$): Stochastically zeroes all 12 leads over a contiguous $0.50 - 1.00$\,s window ($50-100$ samples at 100\,Hz), forcing convolutional filters to capture global rhythmicity and complex morphologies rather than single-beat artifacts.
3. **Hierarchical Multi-Scale Explainability (XAI)**:
   - **Macro-Scale**: Territory Occlusion Sensitivity ($\Delta P$) and learned cross-territory Squeeze-and-Excitation Softmax Attention ($\vec{w}_{\text{attn}}$).
   - **Meso-Scale**: 1D Multi-Branch Grad-CAM++ highlighting pathological cardiac waveform segments (ST-elevations, inverted T-waves, pathological Q-waves).
   - **Micro-Scale**: Beat-synchronized Axiomatic Integrated Gradients satisfying sample completeness ($|\sum \text{IG} - \Delta \text{Score}| < 0.02$).
4. **Clinical Visualizations**: High-resolution attribution profiles rendered over centered **3.0-second diagnostic zoom windows** with standard clinical pink ECG grids ($0.20$\,s major, $0.04$\,s minor; $0.5$\,mV major, $0.1$\,mV minor).

---

## 🏆 Key Experimental Results

### 1. Standard Held-Out Fold 10 Benchmark Comparison (PTB-XL v1.0.3, $N=2,198$)

Evaluated on the exact 5-diagnostic superclass task (`NORM`, `MI`, `STTC`, `CD`, `HYP`) following Strodthoff et al. (*IEEE JBHI*, 2021):

| Model Architecture | Source / Reference | Number of Parameters | PTB-XL Fold 10 Macro AUC | Macro F1 (Val-Opt) | 10-Fold CV Macro AUC | Explainability Modality |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Panel A: High-Capacity Ensembles** | | | | | | |
| **Stacking Ensemble (Mamba+xLSTM+KAN+ECGFounder)** | Al-Mutawa et al. (2026) | Multi-Million (5 Models) | **0.9360** | --- | --- | Black-Box Ensemble |
| **Panel B: Individual Supervised & Foundation Baselines** | | | | | | |
| **ResNet1D-Wang** | Strodthoff et al. (2020) | ~500k | 0.9300 | 0.7300 | --- | Naive Saliency |
| **XResNet1D101** | Strodthoff et al. (2020) | ~2.5M | 0.9280 | 0.7240 | --- | Naive Saliency |
| **TolerantECG** | Nguyen et al. (ACM MM 2025) | Multi-Million | 0.9260 | --- | --- | Black-Box Embedding |
| **Inception1D** | Strodthoff et al. (2020) | ~450k | 0.9210 | 0.7180 | --- | Naive Saliency |
| **MIMIC-IV Foundation Tokenizer** | Hsu et al. (2026) | Multi-Million | 0.8945 | --- | --- | Black-Box Embedding |
| **Panel C: Proposed Framework and Controls** | | | | | | |
| **Model 1: Baseline Flat SE-ResNet1D** | Re-implemented Baseline | 763,629 | 0.9097 | 0.7214 | 0.9279 ± 0.0071 | Naive Grad-CAM |
| **Model 2: Anatomical Multi-Branch** | Intermediate Architecture | 492,185 | 0.9282 | 0.7580 | 0.9407 ± 0.0056 | Multi-Branch Grad-CAM++ |
| **Model 3: Territory-Dropout (Ours)** | Proposed Framework | **492,185** | **0.9306** | **0.7549** | **0.9407 ± 0.0051** | **Multi-Scale Anatomical XAI** |

- **Patient-Level Paired Bootstrap ($B=1,000$ iterations on Fold 10)**: Model 3 vs Model 1 $\Delta \text{AUC} = \mathbf{+0.0209}$ [95% CI: $\mathbf{+0.0160, +0.0260}$] ($p < 0.001$).
- **Efficiency**: Model 3 matches/exceeds baselines with **35.5% fewer parameters** than flat architectures (492k vs 764k).
- **Ensemble Context**: Approaches the performance of the 5-model stacking ensemble ($0.9306$ vs. $0.9360$) while preserving intrinsic multi-scale interpretability and 3.3s bedside latency.

### 2. Ablation Suite: Isolating Anatomical Inductive Bias vs. Regularization (Fold 10)

| Architecture / Configuration | Grouping Strategy | Parameters | Macro AUC | Macro F1 (Val-Opt) | $\Delta$ AUC vs Proposed |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Proposed Model 3** | Anatomical + Territory-Drop + Cutout | **492,185** | **0.9306** | **0.7549** | **Baseline** |
| **Control 1: Random Lead Groups** | Random [3, 4, 4, 1] + Branch-Drop + Cutout | 492,185 | 0.9234 | 0.7410 | **-0.0072** |
| **Control 2: Ordinary Lead Dropout** | Flat ResNet + Random Lead Drop + Cutout | 494,278 | 0.9185 | 0.7321 | **-0.0121** |
| **Control 3: Parameter-Matched Flat** | Flat ResNet (matched 494k params) | 494,278 | 0.9124 | 0.7248 | **-0.0182** |
| **Control 4: Ablation Without Cutout** | Anatomical + Territory-Drop (No Cutout) | 492,185 | 0.9258 | 0.7482 | **-0.0048** |
| **Control 5: Anatomical Alone (Model 2)** | Anatomical (No Dropout, No Cutout) | 492,185 | 0.9282 | 0.7580 | **-0.0024** |

### 3. Robustness Stress Tests, Calibration & Latency Suite

- **Missing Lead Resilience**: Under 1 to 6 missing leads, Model 3 retains a $+0.014$ to $+0.019$ AUC advantage over flat baselines.
- **Occluded Territories**: Under complete loss of the Lateral territory, Model 3 retains **0.8945 AUC** (vs. 0.7812 for Model 1, a **$+0.1133$ AUC advantage**).
- **Probability Calibration**: Expected Calibration Error (ECE) is reduced from **11.09%** to **8.93%** (**24.2% relative error reduction**; Brier score reduced from 0.1145 to 0.0977).
- **Clinical Latency**: Forward inference takes $140.6$\,ms; the full 3-tier XAI pipeline executes in **$3.33 \pm 0.28$\,s** on GPU ($3.36 \pm 0.37$\,s on CPU), well within bedside emergency windows.

### 4. External Out-of-Distribution Generalization on CPSC2018 ($N=6,877$)

Mapped via SNOMED CT ontology (6 classes: RBBB, AF, 1AVB, PVC, PAC, LBBB map to `CD`, explaining $N=4,988$ records):

| Diagnostic Class | Evaluated Records ($N$) | ROC-AUC [95% CI] | PR-AUC | Zero-Shot F1 (0.50) | Transferred F1 | Target Tuned F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Normal Sinus Rhythm (`NORM`) | 918 | **0.9049 [0.88–0.92]** | 0.6153 | 0.5214 | 0.5108 | 0.5816 |
| Conduction Disturbances (`CD`) | 4,988 | **0.8462 [0.82–0.87]** | 0.9362 | 0.7842 | 0.7491 | 0.8306 |
| ST/T Changes (`STTC`) | 1,087 | 0.6036 [0.57–0.63] | 0.1876 | 0.1623 | 0.1415 | 0.3229 |
| **Macro Aggregate** | **6,877** | **0.7849 [0.75–0.81]** | **0.5797** | **0.4893** | **0.4671** | **0.5784** |

*Note: STTC transfer (0.6036 AUC) reflects an ontological domain shift (PTB-XL non-specific repolarization vs CPSC2018 acute ischemic depression/elevation). MI and HYP are absent in CPSC2018.*

### 5. Adebayo et al. (NeurIPS 2018) Parameter Randomization Sanity Check ($N=100$)

| Attribution Method | Randomization Condition | Pearson $r$ (Mean ± SD) | Spearman $\rho$ (Mean ± SD) | Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **Grad-CAM++** | Top-Layer Randomization | $-0.1600 \pm 0.3452$ | $-0.2383 \pm 0.4027$ | Sensitive (Gradients Inverted) |
| | Cascading Randomization | $+0.1814 \pm 0.1850$ | $+0.1456 \pm 0.2209$ | Sensitive (Intermediate Fluctuation) |
| | **Full Network Randomization** | $\mathbf{+0.0935 \pm 0.1615}$ | $\mathbf{+0.0676 \pm 0.1779}$ | **Passed (Full Decorrelation Collapse)** |
| **Integrated Gradients** | Top-Layer Randomization | $-0.2886 \pm 0.4490$ | $-0.1915 \pm 0.3491$ | Sensitive (Gradients Inverted) |
| | Cascading Randomization | $+0.1327 \pm 0.2636$ | $+0.0481 \pm 0.1154$ | Sensitive (Intermediate Fluctuation) |
| | **Full Network Randomization** | $\mathbf{+0.0015 \pm 0.1408}$ | $\mathbf{-0.0075 \pm 0.0295}$ | **Passed (Absolute Zero Collapse)** |

---

## 📂 Repository Structure

```text
├── manuscript/                     # Complete LaTeX manuscript & figures
│   ├── main.tex                    # Two-column manuscript source
│   ├── main_single.tex             # Single-column review format source
│   ├── references.bib              # Verified BibTeX citations
│   ├── main.pdf                    # Compiled Two-Column PDF (22 pages)
│   ├── main_single.pdf             # Compiled Single-Column PDF (29 pages)
│   └── figures/                    # High-resolution clinical figures & profiles
│       ├── fig1_system_architecture.png
│       ├── fig2_confusion_matrices.png  # Evaluated on Fold 10 (N=2,198)
│       ├── fig3_xai_mi.png ... fig7_xai_norm.png
├── checkpoints/                    # Saved model weights (.h5) across all 10 folds
│   ├── se_resnet_anatomical_territory_dropout_fold_1_best.h5 ... fold_10_best.h5
│   └── training_history_*.csv      # Per-epoch training & validation history
├── experiments/                    # Quantitative audit CSVs, benchmarks & stress tests
│   ├── evaluate_fold10_benchmarks.py       # Standard Fold 10 benchmark & bootstrap test
│   ├── fold10_benchmark_evaluation.csv    # Fold 10 head-to-head metrics
│   ├── run_advisor_ablation_suite.py       # Full ablation suite (random groups, flat controls)
│   ├── advisor_ablation_results.csv        # Ablation numerical results
│   ├── run_robustness_calibration_latency.py # Stress tests, calibration & latency script
│   ├── robustness_stress_test_results.csv  # Missing leads/territories & noise results
│   ├── calibration_metrics.csv             # ECE and Brier scores
│   ├── latency_benchmark.csv               # Forward and XAI execution timings
│   ├── full_cohort_fold10_attribution_audit.csv # Full-cohort audit (N=2,198)
│   ├── adebayo_sanity_check_results.csv   # Adebayo sanity check summary (N=100)
│   └── cpsc2018_zeroshot_metrics.csv       # External zero-shot metrics
├── generate_confusion_matrices.py  # 3x5 Confusion matrix generator for Fold 10
├── run_10fold_anatomical_territory_dropout_classifier.py # 10-fold training script
├── run_full_cohort_xai_audit.py    # Full-cohort Fold 10 attribution audit script
├── run_adebayo_sanity_check.py     # Parameter randomization sanity check script
├── evaluate_cpsc2018_zeroshot.py   # External zero-shot evaluation script
├── ADVISOR_REVIEW_RESPONSE.md      # Comprehensive point-by-point rebuttal to advisor
├── RESULTS_SUMMARY.md              # Detailed numerical results & statistical tests
└── AUDIT_LOG_AND_SYSTEM_VERIFICATION.md # Complete audit and reproduction trail
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup

```bash
conda create -n ptbxl-gpu python=3.10 -y
conda activate ptbxl-gpu
pip install tensorflow[and-cuda]==2.14.0 numpy pandas scipy scikit-learn wfdb matplotlib
```

### 2. Run Standard Fold 10 Benchmark & Paired Bootstrap Test

```bash
python experiments/evaluate_fold10_benchmarks.py
```

### 3. Run Robustness Stress Tests, Calibration & Latency Benchmark

```bash
python experiments/run_robustness_calibration_latency.py
```

### 4. Run Full-Cohort Attribution Audit ($N = 2,198$)

```bash
python run_full_cohort_xai_audit.py
```

### 5. Run Adebayo Parameter Randomization Sanity Checks ($N = 100$)

```bash
python run_adebayo_sanity_check.py
```

### 6. Run External Zero-Shot Generalization on CPSC2018

```bash
python evaluate_cpsc2018_zeroshot.py
```

---

## 📄 License & References

This project is licensed under the MIT License.

### Key Comparative References:
- **Zhou & Chen (2024)**: *"Leadwise clustering multi-branch network for multi-label ECG classification"*, *Medical Engineering & Physics*, 130, 104196. [doi:10.1016/j.medengphy.2024.104196](https://doi.org/10.1016/j.medengphy.2024.104196)
- **Nguyen et al. (2025)**: *"TolerantECG: A Foundation Model for Imperfect Electrocardiogram"*, *ACM MM '25*, pp. 1–10. [doi:10.1145/3664647.3681423](https://doi.org/10.1145/3664647.3681423)
- **Liu et al. (2026)**: *"ACL-ECG: Anatomy-Aware Contrastive Learning for Multi-Lead Electrocardiograms"*, *Sensors*, 26(3), 772. [doi:10.3390/s26030772](https://doi.org/10.3390/s26030772)
- **Li et al. (2025)**: *"ECGFounder: A foundational AI model for electrocardiogram analysis"*, *arXiv preprint arXiv:2507.09887*.
- **Wagner et al. (2009)**: *"AHA/ACCF/HRS recommendations for the standardization and interpretation of the electrocardiogram: part VI: acute ischemia/infarction"*, *Circulation*, 119(10), e262–e270. [doi:10.1161/CIRCULATIONAHA.108.191093](https://doi.org/10.1161/CIRCULATIONAHA.108.191093)
- **Strodthoff et al. (2021)**: *"Deep Learning for ECG Analysis: Benchmarks and Knowledge Mapping on PTB-XL"*, *IEEE JBHI*, 25(5), 1519–1528. [doi:10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989)
- **Sundararajan et al. (2017)**: *"Axiomatic attribution for deep networks"*, *ICML*, pp. 3319–3328.

### Citation:
```bibtex
@article{lodhi2026anatomical,
  title={Anatomically-Decomposed Squeeze-and-Excitation 1D ResNet with Multi-Scale Explainability for 12-Lead Electrocardiogram Interpretation},
  author={Lodhi, Awais Muhammad and Mustafa, Ghulam},
  journal={Preprint (Under Review)},
  year={2026}
}
```
