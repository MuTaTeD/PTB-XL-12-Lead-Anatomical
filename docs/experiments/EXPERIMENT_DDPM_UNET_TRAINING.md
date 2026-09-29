# Experiment Log: 1D Conditional DDPM U-Net Training (12-Lead PTB-XL Waveforms)

**Date**: September 16, 2026  
**Dataset**: PTB-XL v1.0.3 Benchmark (12 Leads, 100 Hz, 1,000 Temporal Samples)  
**Training Partition**: Folds 1–8 (17,441 records)  
**Validation Partition**: Fold 9 (2,183 records)  
**Holdout Test Partition**: Fold 10 (2,198 records — Clean Out-of-Sample Evaluation)  
**Model Architecture**: 1D Conditional Res-UNet (3.24 Million Parameters, Sinusoidal Timestep Embeddings, Diagnostic Class Conditioning)

---

## 1. Experimental Setup & Diffusion Configuration

- **Script**: `train_conditional_ddpm.py`
- **Model Checkpoint**: `checkpoints/ddpm_unet_1d_ptbxl.h5`
- **History CSV Log**: `checkpoints/training_history_ddpm_unet.csv`
- **Diffusion Timesteps ($T$)**: 1000 steps
- **Noise Schedule**: Linear ($\beta_1 = 10^{-4}$ to $\beta_T = 0.02$)
- **Classifier-Free Guidance (CFG) Unconditional Dropout**: $p_{\text{uncond}} = 0.15$
- **Optimizer**: Adam ($\text{Initial LR} = 3 \times 10^{-4}$, $\text{Min LR} = 1 \times 10^{-6}$)
- **Batch Size**: 64
- **Adaptive Callbacks**: `ModelCheckpoint` (saves lowest `val_loss`), `ReduceLROnPlateau` (factor=0.5, patience=4), `EarlyStopping` (patience=10).

---

## 2. Training Convergence & Metrics

| Metric | Initial State (Epoch 0) | Intermediate State (Epoch 10) | Final Minima (Epoch 40) | Final State (Epoch 50) |
| :--- | :---: | :---: | :---: | :---: |
| **Training Noise MSE Loss** | `0.6588` | `0.0548` | `0.0344` | **`0.0327`** |
| **Validation Noise MSE Loss (`val_loss`)** | `0.3853` | `0.0509` | 🏆 **`0.0295`** | `0.0304` |
| **Learning Rate** | `3.00e-04` | `3.00e-04` | `1.875e-05` | `4.688e-06` |

> [!NOTE]
> The lowest validation noise prediction error ($\text{MSE} = 0.0295$) was achieved at Epoch 40. The best weights were automatically restored and saved to `checkpoints/ddpm_unet_1d_ptbxl.h5`.

---

## 3. Key Scientific Conclusions & Next Steps

1. **Noise Prediction Accuracy**: Achieving a validation MSE loss of **`0.0295`** confirms that the 1D Conditional Res-UNet has successfully learned the reverse score function $\epsilon_\theta(x_t, t, c)$ across 12-lead ECG signals.
2. **CFG Alignment**: The 15% unconditional condition dropout enables fast, controlled Classifier-Free Guidance sampling during inference without requiring separate unconditional network weights.
3. **Ready for Causal Counterfactual Generation**: The trained DDPM generator is fully integrated and ready to be paired with **Model 3 (Fold 9 Checkpoint)** for dual-guided counterfactual synthesis.
