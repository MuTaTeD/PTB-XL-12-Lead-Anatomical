# Implementation Plan: 1D Conditional DDPM U-Net Counterfactual Generation System

**Goal**: Develop a 1D Conditional Denoising Diffusion Probabilistic Model (DDPM) to generate minimal counterfactual edits on 12-lead ECGs, converting diseased signals (MI, STTC, CD, HYP) into Normal Sinus Rhythm (`NORM`) while strictly preserving individual patient identity (heart rate, QRS morphology, lead geometries, and baseline axis).

---

## 1. System Architecture & Workflow

```mermaid
flowchart TD
    subgraph A [Inputs & Target]
        PathECG["Diseased Patient ECG (1000 x 12)\ne.g., MI or STTC"]
        TargetClass["Target Condition: NORM [1,0,0,0,0]"]
    end

    subgraph B [Identity Preservation via Intermediate Noising]
        NoiseProcess["Forward Noise to t* (e.g. t* = 250)\nx_t* = sqrt(alpha_t*) * x_0 + sqrt(1-alpha_t*) * eps"]
    end

    subgraph C [Guided Reverse Sampling (t* -> 0)]
        UNet1D["1D Conditional Diffusion U-Net\n(Predicts Noise eps_theta(x_t, t, c))"]
        Classifier["Frozen Calibrated SE-ResNet1D\n(Classifier Guidance Gradient)"]
        CFG["Classifier-Free Guidance (CFG)\neps_hat = (1+w)*eps(c) - w*eps(empty)"]
    end

    subgraph D [Counterfactual Outputs & Identity Audit]
        CF_ECG["Generated Counterfactual ECG\n(Normal Sinus Rhythm)"]
        DiffMask["Minimal Edit Delta x = CF_ECG - PathECG"]
        Metrics["Identity Metrics:\n- HR Error < 2 bpm\n- Lead Cosine Similarity > 0.98\n- Classifier Target Prob > 90%"]
    end

    PathECG --> NoiseProcess
    NoiseProcess --> UNet1D
    TargetClass --> UNet1D
    UNet1D --> CFG
    Classifier --> CFG
    CFG --> CF_ECG
    PathECG --> DiffMask
    CF_ECG --> DiffMask
    CF_ECG --> Metrics
```

---

## 2. Proposed Modules & Core Components

### Component 1: 1D Conditional Diffusion U-Net Architecture
#### [NEW] [`causal_diffusion/unet_1d.py`](file:///home/awais/Desktop/PTB-XL/causal_diffusion/unet_1d.py)
- **1D Temporal U-Net with Skip Connections**:
  - Input shape: `(batch_size, 1000, 12)` (12-lead 10s ECG signal at 100 Hz).
  - Encoder: 4 Residual 1D Downsampling Blocks (`channels = 64, 128, 256, 512`) using `Conv1D(stride=2)` + Group Normalization + Swish activation.
  - Bottleneck: 1D Multi-Head Self-Attention layer over sequence time-steps.
  - Decoder: 4 Residual 1D Upsampling Blocks (`UpSampling1D` + `Conv1D`) with skip-connection concatenation.
- **Sinusoidal Positional Timestep Embedding**:
  - Projects scalar $t \in [1, 1000]$ into a Dense vector injected into every residual block.
- **Diagnostic Label Conditioning Vector**:
  - Embedding layer for the 5-superclass binary vector $c \in \{0, 1\}^5$. Supports unconditional dropout ($c \to \emptyset$) for Classifier-Free Guidance (CFG).

---

### Component 2: DDPM Training Pipeline & Noise Schedule
#### [NEW] [`train_conditional_ddpm.py`](file:///home/awais/Desktop/PTB-XL/train_conditional_ddpm.py)
- **Diffusion Noise Parameters**:
  - $T = 1000$ timesteps, linear/cosine $\beta$ schedule ($\beta_1 = 10^{-4}$ to $\beta_T = 0.02$).
  - Precalculated $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.
- **Classifier-Free Guidance Training**:
  - Randomly sets class condition $c = \emptyset$ (zero vector) with probability $p_{\text{uncond}} = 0.15$ during batch training.
- **Training Protocol**:
  - Loss: MSE on predicted noise $\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2$.
  - Data: Multi-fold PTB-XL training set (Folds 1–8).
  - Checkpoint: `checkpoints/ddpm_unet_1d_ptbxl.h5`.

---

### Component 3: Counterfactual Guidance Sampler & Identity Audit
#### [NEW] [`generate_ddpm_counterfactual.py`](file:///home/awais/Desktop/PTB-XL/generate_ddpm_counterfactual.py)
- **Intermediate Noising Sampler ($t^*$ trajectory)**:
  - Takes a diseased test ECG $x_0^{\text{pathology}}$ (e.g. from Fold 10).
  - Adds noise up to intermediate timestep $t^* \in [150, 350]$:
    $$x_{t^*} = \sqrt{\bar{\alpha}_{t^*}} x_0^{\text{pathology}} + \sqrt{1 - \bar{\alpha}_{t^*}} \epsilon$$
  - Preserves low-frequency patient identity (QRS duration, heart rate, baseline axis, lead geometries) while freeing high-frequency diagnostic regions for counterfactual editing.
- **Guided Reverse Generation ($t^* \to 0$)**:
  - Combines Classifier-Free Guidance ($w = 2.0$) with Classifier Guidance gradients from our frozen `se_resnet_calibrated_fold1_8_best.h5` model to steer predictions toward 100% `NORM`.
- **Identity & Counterfactual Quantitative Evaluation**:
  1. **Diagnostic Reversal**: Verifies target `NORM` probability $> 90\%$ on frozen classifier.
  2. **Heart Rate Preservation**: Computes R-R interval peak distance difference ($| \text{HR}_{\text{orig}} - \text{HR}_{\text{cf}} | < 2$ bpm).
  3. **Lead Geometry & Axis Cosine Similarity**: Computes lead-by-lead cosine similarity ($> 0.98$).
  4. **Minimal Edit Distance**: Computes $L_1$ and $L_2$ norm of the edit $\Delta x = x_{\text{cf}} - x_{\text{orig}}$.
- **Visualization Artifact**:
  - Saves 12-lead overlay plots showing original diseased signal in blue, counterfactual signal in green, and minimal edit mask in red to `experiments/ddpm_counterfactual_overlay.png`.

---

## 3. Verification Plan

### Automated Tests & Pipeline Validation
1. **U-Net Architecture Test**: Verify input/output shape consistency `(batch, 1000, 12)` and parameter count.
2. **Short Training Run**: Train DDPM U-Net for 5 epochs on GPU to verify loss convergence and CFG conditional embedding.
3. **Counterfactual Sampling Run**: Generate counterfactuals for 10 diseased Fold 10 test samples and evaluate:
   - Post-edit classifier `NORM` probability (Target: $> 90\%$).
   - Lead cosine similarity (Target: $> 0.98$).
   - Heart rate preservation error (Target: $< 2$ bpm).

### Manual & Visual Verification
- Review 12-lead overlay figure `experiments/ddpm_counterfactual_overlay.png` to ensure the edit is localized specifically to ST-T segments or QRS blocks (for MI/STTC) while preserving baseline identity across all leads.
