# Comprehensive Per-Class Clinical Metrics & Confusion Matrices Analysis

**Date**: September 13, 2026  
**Dataset**: PTB-XL v1.0.3 (100Hz, 12-Lead, 21,837 Records, Test Fold 10 $N=2,198$)  
**Evaluated Models**:
1. **Model 1: Calibrated SE-ResNet1D** (Flat 12-Lead Baseline)
2. **Model 2: Anatomical Multi-Branch SE-ResNet1D** (4 Cardiac Territories + Cross-SE Attention)
3. **Model 3: Anatomical Territory-Dropout SE-ResNet1D** (Regional Masking Regularization $p=0.15$)

---

## 1. Complete Per-Class Clinical Metrics Audit Table (N=2,198 Test Records)

The table below presents the exact per-class confusion matrix parameters (**TP, FP, TN, FN**) alongside clinical metrics (**Accuracy, Sensitivity, Specificity, PPV, NPV, F1-Score, ROC-AUC**) computed on held-out test fold 10:

| Superclass | Model Architecture | ROC-AUC | F1-Score | Accuracy | Sensitivity (Recall) | Specificity (TNR) | PPV (Precision) | NPV | Threshold | Confusion Matrix (TP / FP / TN / FN) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **NORM**<br>*(Normal Sinus)* | Model 1: Calibrated Baseline<br>**Model 2: Anatomical Multi-Branch**<br>Model 3: Territory-Dropout | 0.9626<br>🏆 **0.9647**<br>0.9469 | 0.8805<br>🏆 **0.8931**<br>0.8606 | 89.17%<br>🏆 **90.31%**<br>86.53% | 91.07%<br>92.42%<br>🏆 **94.91%** | 87.69%<br>🏆 **88.66%**<br>80.00% | 85.23%<br>🏆 **86.41%**<br>78.73% | 92.64%<br>93.75%<br>🏆 **95.27%** | 0.68<br>0.60<br>0.43 | `877 / 152 / 1083 / 86`<br>`890 / 140 / 1095 / 73`<br>`914 / 247 / 988 / 49` |
| **MI**<br>*(Myocardial Infarction)* | Model 1: Calibrated Baseline<br>**Model 2: Anatomical Multi-Branch**<br>Model 3: Territory-Dropout | 0.9442<br>🏆 **0.9563**<br>0.9244 | 0.7845<br>🏆 **0.8149**<br>0.7489 | 88.58%<br>🏆 **90.49%**<br>87.31% | 83.09%<br>🏆 **83.64%**<br>75.64% | 90.41%<br>🏆 **92.78%**<br>91.20% | 74.31%<br>🏆 **79.45%**<br>74.15% | 94.13%<br>🏆 **94.44%**<br>91.81% | 0.56<br>0.71<br>0.61 | `457 / 158 / 1490 / 93`<br>`460 / 119 / 1529 / 90`<br>`416 / 145 / 1503 / 134` |
| **STTC**<br>*(ST/T Changes)* | Model 1: Calibrated Baseline<br>**Model 2: Anatomical Multi-Branch**<br>Model 3: Territory-Dropout | 0.9565<br>0.9518<br>0.9347 | 0.8038<br>🏆 **0.8065**<br>0.7726 | 90.49%<br>🏆 **90.86%**<br>88.54% | 🏆 **82.15%**<br>80.42%<br>🏆 **82.15%** | 93.08%<br>🏆 **94.10%**<br>90.52% | 78.68%<br>🏆 **80.89%**<br>72.91% | 94.38%<br>93.93%<br>🏆 **94.23%** | 0.67<br>0.71<br>0.70 | `428 / 116 / 1561 / 93`<br>`419 / 99 / 1578 / 102`<br>`428 / 159 / 1518 / 93` |
| **CD**<br>*(Conduction Disturbance)*| Model 1: Calibrated Baseline<br>**Model 2: Anatomical Multi-Branch**<br>Model 3: Territory-Dropout | 0.9439<br>🏆 **0.9437**<br>0.9233 | 0.8054<br>🏆 **0.8120**<br>0.7728 | 91.49%<br>🏆 **91.77%**<br>89.35% | 78.02%<br>78.83%<br>🏆 **80.24%** | 95.42%<br>🏆 **95.53%**<br>92.01% | 83.23%<br>🏆 **83.73%**<br>74.53% | 93.71%<br>93.93%<br>🏆 **94.11%** | 0.65<br>0.69<br>0.48 | `387 / 78 / 1624 / 109`<br>`391 / 76 / 1626 / 105`<br>`398 / 136 / 1566 / 98` |
| **HYP**<br>*(Hypertrophy)* | Model 1: Calibrated Baseline<br>**Model 2: Anatomical Multi-Branch**<br>Model 3: Territory-Dropout | 0.9588<br>0.9464<br>0.9061 | 🏆 **0.7173**<br>0.7018<br>0.6233 | 🏆 **93.22%**<br>92.27%<br>89.72% | 72.14%<br>🏆 **76.34%**<br>71.37% | 🏆 **96.07%**<br>94.42%<br>92.20% | 🏆 **71.32%**<br>64.94%<br>55.33% | 96.22%<br>🏆 **96.72%**<br>95.97% | 0.71<br>0.67<br>0.61 | `189 / 76 / 1860 / 73`<br>`200 / 108 / 1828 / 62`<br>`187 / 151 / 1785 / 75` |

---

## 2. Key Cardiological & Technical Takeaways

1. **Model 2 (Anatomical Multi-Branch)** achieves the highest overall diagnostic **Precision (PPV)** and **Specificity (TNR)**:
   - **MI PPV**: Boosted to **79.45%** (reducing false positive alarm fatigue).
   - **NORM Accuracy**: Achieves **90.31%** overall binary accuracy.
2. **Model 3 (Anatomical Territory-Dropout)** yields the highest **Clinical Screening Sensitivity (TPR)**:
   - **NORM Sensitivity**: **94.91%** (ensuring healthy controls are reliably detected).
   - **CD Sensitivity**: **80.24%** (highest recall for bundle branch block anomalies).

---

## 3. Associated Visual & Code Artifacts

- **Detailed Metrics CSV**: [`experiments/comprehensive_diagnostic_metrics_3models.csv`](file:///home/awais/Desktop/PTB-XL/experiments/comprehensive_diagnostic_metrics_3models.csv)
- **Evaluation Script**: `generate_comprehensive_diagnostic_audit.py`
- **Visual Heatmap Plot**: [`experiments/confusion_matrices_comparison.png`](file:///home/awais/Desktop/PTB-XL/experiments/confusion_matrices_comparison.png)
- **Audit Log Entry**: Section 14 in [`AUDIT_LOG_AND_SYSTEM_VERIFICATION.md`](file:///home/awais/Desktop/PTB-XL/AUDIT_LOG_AND_SYSTEM_VERIFICATION.md)
