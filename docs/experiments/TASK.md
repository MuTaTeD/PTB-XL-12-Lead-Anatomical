# Task Checklist: Anatomically-Decomposed ResNet & Multi-Scale Anatomical Explainability Framework

## Phase 1: Classification & Regularization Benchmarks
- `[x]` Task 1.1: Implement Model 1 (Baseline Global SE-ResNet1D).
- `[x]` Task 1.2: Implement Model 2 (Anatomical 4-Branch SE-ResNet1D: Inferior, Antero-Septal, Lateral, Cavity).
- `[x]` Task 1.3: Implement Model 3 (Anatomical Territory-Dropout SE-ResNet1D, $p=0.15$).
- `[x]` Task 1.4: Execute patient-stratified 10-fold cross-validation on PTB-XL (21,799 records).
- `[x]` Task 1.5: Validate zero-shot external generalization on CPSC2018 (6,877 records; Macro AUC 0.7849).

## Phase 2: Generative Counterfactual Exploration & Audit
- `[x]` Task 2.1: Implement 1D Conditional Diffusion U-Net Architecture (`causal_diffusion/unet_1d.py`).
- `[x]` Task 2.2: Implement DDPM Noise Schedule & Training Pipeline (`train_conditional_ddpm.py`).
- `[x]` Task 2.3: Train 1D Conditional DDPM U-Net on GPU (PTB-XL Folds 1–8).
- `[x]` Task 2.4: Implement Physics-Informed DDPM loss (Einthoven + Goldberger constraints) and audit results (`EXPERIMENT_PHYSICS_DDPM.md`).
  - *Finding*: Adding hard biophysical spatial penalties triggered severe temporal mode collapse (R-peak destruction, heart rate error $>80$ bpm). Generative counterfactuals also suffer from non-causal classifier guidance and lack of clinical ground truth. Decision made to pivot to Multi-Scale Post-Hoc XAI.

## Phase 3: Multi-Scale Anatomical Explainability (XAI) Suite
- `[x]` Task 3.1: Implement Multi-Scale Anatomical XAI Engine (`evaluate_anatomical_xai.py`):
  - **Macro-Tier**: Anatomical Territory Occlusion Sensitivity ($\Delta P$) + Learned Softmax Cross-Territory Attention ($\vec{w}_{\text{attn}}$).
  - **Meso-Tier**: 1D Multi-Branch Grad-CAM++ extracting gradient-weighted activation maps from final residual convolutional layers across each anatomical branch.
  - **Micro-Tier**: Axiomatic Integrated Gradients verifying point-by-point feature completeness ($|\sum \text{IG} - \Delta \text{Score}| < 0.02$).
- `[x]` Task 3.2: Conduct full quantitative audit across held-out Fold 10 test cohort (`experiments/xai_quantitative_cohort_audit.csv` & `experiments/XAI_QUANTITATIVE_AUDIT_SUMMARY.md`).
- `[x]` Task 3.3: Regenerate multi-panel publication figures with **3.0-second diagnostic zoom window** and authentic pink clinical ECG grid (0.20s major, 0.04s minor):
  - `manuscript/figures/fig3_xai_mi.png` (Myocardial Infarction — Record #15647)
  - `manuscript/figures/fig4_xai_sttc.png` (ST/T-Change — Record #10054)
  - `manuscript/figures/fig5_xai_cd.png` (Conduction Disturbance — Record #4893)
  - `manuscript/figures/fig6_xai_hyp.png` (Hypertrophy — Record #16182)
  - `manuscript/figures/fig7_xai_norm.png` (Normal Control — Record #2083)

## Phase 4: Project Documentation & Manuscript Publication Suite
- `[x]` Task 4.1: Update project log files (`EXPERIMENT_MULTI_SCALE_ANATOMICAL_XAI.md`, `RESULTS_SUMMARY.md`, `TASK.md`).
- `[x]` Task 4.2: Update architecture schematic diagram (`fig_xai_framework.png`).
- `[x]` Task 4.3: Overhaul research manuscript (`manuscript/main.tex` and `manuscript/main_single.tex`):
  - Update Title, Abstract, Highlights, Keywords.
  - Update Introduction and Table 1 (Traditional Naive Saliency vs. Multi-Scale Anatomical XAI).
  - Update Related Work (deep learning electrocardiology, XAI methods).
  - Update Methodology (equations for 1D Grad-CAM++, Integrated Gradients, and Territory Occlusion).
  - Add Section 3.3 for architectural specs (layer dimensions, parameter counts, training hyperparameters).
  - Update Results (Section 4: Quantitative Anatomical Attribution Audit, 3.0s multi-panel figures).
  - Update Discussion and Conclusion.
- `[x]` Task 4.4: Compile LaTeX manuscript with `pdflatex` to verify zero errors and generate clean publication PDFs (`main.pdf` and `main_single_column_review.pdf`).
- `[x]` Task 4.5: Document human cardiologist accuracy benchmarks for paper comparison and defense (`CARDIOLOGIST_BENCHMARK_REFERENCE.md`, `RESULTS_SUMMARY.md`).
