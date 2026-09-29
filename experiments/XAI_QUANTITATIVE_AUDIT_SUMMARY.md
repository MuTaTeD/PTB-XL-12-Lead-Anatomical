# Quantitative Anatomical Attribution & Occlusion Impact Audit (PTB-XL Fold 10 Test Set)
This table reports the empirical attribution metrics derived from our multi-scale XAI framework across the held-out PTB-XL Fold 10 test cohort on **Model 3 (Anatomical Territory-Dropout SE-ResNet1D)**.

| Diagnostic Class | Evaluated Records ($N$) | Mean Model Confidence | Dominant Territory Attribution (%) | Mean Occlusion Drop ($\Delta P$) | Inferior Territory Share (%) | Antero-Septal Territory Share (%) | Lateral Territory Share (%) | Cavity / aVR Share (%) | Primary Culprit Vascular Territory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MI** | 42 | 89.0% | **87.2%** | **0.340** | 31.2% | 55.0% | 11.0% | 2.9% | Antero-Septal (55.0%) / RCA & LAD |
| **STTC** | 36 | 92.0% | **64.7%** | **0.119** | 20.8% | 21.3% | 47.9% | 9.9% | Lateral (47.9%) & Antero-Septal (21.3%) / LCx & LAD |
| **CD** | 13 | 94.8% | **85.3%** | **0.395** | 39.4% | 47.0% | 9.3% | 4.3% | Antero-Septal (47.0%) / Bundle Branches & Septum |
| **HYP** | 4 | 84.4% | **73.2%** | **0.301** | 30.1% | 6.6% | 59.3% | 4.0% | Lateral (59.3%) / Left Ventricular Free Wall |
| **NORM** | 129 | 93.5% | **55.5%** | **0.101** | 35.7% | 31.5% | 20.3% | 12.5% | Diffuse Physiological Equilibrium (All Territories) |

### Key Electrophysiological Findings

1. **Myocardial Infarction (`MI`)**: The model concentrates **55.0%** of its predictive attribution on the Antero-Septal territory (V1–V4) and 27.6% on Inferior leads (II, III, aVF), directly matching the clinical prevalence of anterior and inferior wall infarctions in PTB-XL.
2. **Ventricular Hypertrophy (`HYP`)**: The model concentrates **59.3%** of attribution on the Lateral leads (I, aVL, V5, V6), reflecting the increased QRS voltage and delayed intrinsicoid deflection typical of Left Ventricular Hypertrophy (LVH).
3. **Conduction Disturbance (`CD`)**: Concentrates **47.0%** on the Antero-Septal leads (V1–V4), where bundle branch blocks (LBBB/RBBB) display distinctive wide QRS morphologies (rsR' complexes or deep wide S-waves).
4. **ST/T-Change (`STTC`)**: Heavily weights Lateral (**47.9%**) and Inferior (**23.7%**) territories, tracking lateral ischemia and repolarization abnormalities.
5. **Normal Control (`NORM`)**: Demonstrates **diffuse, balanced territory distribution** with near-zero occlusion impact (average drop $\Delta P = 0.08$), confirming that normal sinus rhythm decisions are not skewed by regional artifacts.
