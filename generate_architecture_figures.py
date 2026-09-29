"""
generate_architecture_figures.py
Generates 3 publication-quality architecture figures (300 DPI):
1. Calibrated SE-ResNet1D Architecture
2. Anatomical Multi-Branch SE-ResNet1D Architecture
3. Causal DDPM Counterfactual Waveform Generation Pipeline
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global styles
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0


# -----------------------------------------------------------------------------
# FIGURE 1: CALIBRATED SE-RESNET1D ARCHITECTURE
# -----------------------------------------------------------------------------
def create_calibrated_architecture_figure():
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    ax.axis('off')

    # Color Palette
    c_input = '#e3f2fd'
    c_conv = '#bbdefb'
    c_resnet = '#90caf9'
    c_se = '#ffe082'
    c_head = '#c8e6c9'
    c_text = '#1a237e'

    # Box drawer helper
    def draw_box(x, y, w, h, text, color='#ffffff', border='#333333', fontsize=10, fontweight='bold'):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.5, edgecolor=border, facecolor=color, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color='#222222', zorder=3)

    # Drawer arrow helper
    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate(text, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#444444', edgecolor='#444444', arrowstyle="->", lw=1.8),
                    ha='center', va='bottom', fontsize=8, color='#333333', zorder=4)

    # Title
    plt.title("Calibrated SE-ResNet1D Classifier Architecture (Flat 12-Lead Input)",
              fontsize=14, fontweight='bold', pad=15, color='#111111')

    # Pipeline Blocks
    draw_box(0.5, 3.5, 2.0, 1.4, "Raw 12-Lead ECG\nShape: (1000, 12)\n100Hz, 10 Seconds", c_input, '#1565c0')
    draw_arrow(2.5, 4.2, 3.2, 4.2)

    draw_box(3.2, 3.5, 2.2, 1.4, "Initial Conv1D\n64 Filters, Kernel=7\n+ BatchNorm & ReLU\n+ Spatial Dropout (0.08)", c_conv, '#1976d2')
    draw_arrow(5.4, 4.2, 6.1, 4.2)

    # ResNet Blocks Stack
    draw_box(6.1, 3.2, 3.2, 2.0, "3x SE-ResNet1D Blocks\n--------------------------------\n• Conv1D (64 → 128 → 256)\n• Residual Skip Connection\n• Squeeze-and-Excitation (SE)\n  (Channel Attention r=16)", c_resnet, '#0d47a1')
    draw_arrow(9.3, 4.2, 10.0, 4.2)

    draw_box(10.0, 3.5, 2.2, 1.4, "Global Average\nPooling 1D\nShape: (Batch, 256)", c_se, '#f57f17')
    draw_arrow(12.2, 4.2, 12.9, 4.2)

    draw_box(12.9, 3.5, 2.4, 1.4, "Dense Dropout (0.30)\n+ Dense (5 Classes)\nSigmoid Activation", c_head, '#2e7d32')

    # Calibration & Loss Box (Bottom)
    draw_box(3.2, 0.8, 12.1, 1.8,
             "POST-TRAINING CALIBRATION & CLASS-WEIGHTED BCE LOSS\n"
             "• Class-Weighted Binary Cross-Entropy Loss: w_pos = N_neg / N_pos (HYP w_pos=7.2)\n"
             "• Per-Class F1-Maximizing Threshold Calibration (Validation Fold Grid Search): τ_c* ∈ [0.1, 0.9]\n"
             "• Outputs 5 Superclasses: NORM, MI, STTC, CD, HYP",
             '#f3e5f5', '#7b1fa2', fontsize=9.5)

    draw_arrow(14.1, 3.5, 14.1, 2.6)

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'architectural_diagram_calibrated_se_resnet1d.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 to '{path}'")


# -----------------------------------------------------------------------------
# FIGURE 2: ANATOMICAL MULTI-BRANCH SE-RESNET1D ARCHITECTURE
# -----------------------------------------------------------------------------
def create_anatomical_architecture_figure():
    fig, ax = plt.subplots(figsize=(17, 9), dpi=300)
    ax.axis('off')

    c_input = '#e3f2fd'
    c_inf = '#ffebee'
    c_sep = '#e8f5e9'
    c_lat = '#fff3e0'
    c_cav = '#f3e5f5'
    c_attn = '#fff9c4'
    c_head = '#c8e6c9'

    def draw_box(x, y, w, h, text, color='#ffffff', border='#333333', fontsize=9.5, fontweight='bold'):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.4, edgecolor=border, facecolor=color, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color='#222222', zorder=3)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#444444', edgecolor='#444444', arrowstyle="->", lw=1.5), zorder=4)

    plt.title("Anatomical Multi-Branch SE-ResNet1D Classifier with Cross-Territory Attention",
              fontsize=14, fontweight='bold', pad=15, color='#111111')

    # Input Box
    draw_box(0.4, 3.5, 2.2, 2.2, "Raw 12-Lead ECG\nShape: (1000, 12)\n------------------\n12 Standard Leads", c_input, '#1565c0')

    # Regional Split Lines
    draw_arrow(2.6, 5.2, 3.5, 7.3)
    draw_arrow(2.6, 4.8, 3.5, 5.3)
    draw_arrow(2.6, 4.4, 3.5, 3.3)
    draw_arrow(2.6, 4.0, 3.5, 1.3)

    # 4 Regional Branches
    draw_box(3.5, 6.6, 3.4, 1.4, "1. Inferior Wall Branch\nLeads: [II, III, aVF] (3 leads)\n• Dedicated ResNet1D Conv\nOutput: (Batch, 128)", c_inf, '#c62828')
    draw_box(3.5, 4.6, 3.4, 1.4, "2. Antero-Septal Branch\nLeads: [V1, V2, V3, V4] (4 leads)\n• Dedicated ResNet1D Conv\nOutput: (Batch, 128)", c_sep, '#2e7d32')
    draw_box(3.5, 2.6, 3.4, 1.4, "3. Lateral Wall Branch\nLeads: [I, aVL, V5, V6] (4 leads)\n• Dedicated ResNet1D Conv\nOutput: (Batch, 128)", c_lat, '#ef6c00')
    draw_box(3.5, 0.6, 3.4, 1.4, "4. Cavity Reciprocal Branch\nLead: [aVR] (1 lead)\n• Dedicated ResNet1D Conv\nOutput: (Batch, 128)", c_cav, '#6a1b9a')

    # Converge to Stacking
    draw_arrow(6.9, 7.3, 7.8, 4.8)
    draw_arrow(6.9, 5.3, 7.8, 4.6)
    draw_arrow(6.9, 3.3, 7.8, 4.4)
    draw_arrow(6.9, 1.3, 7.8, 4.2)

    # Territory Stacking
    draw_box(7.8, 3.8, 2.4, 1.6, "Territory Stacking\nLambda(tf.stack)\n--------------------\nShape: (Batch, 4, 128)", '#e0f2f1', '#00695c')
    draw_arrow(10.2, 4.6, 11.0, 4.6)

    # Cross-Territory SE Attention
    draw_box(11.0, 3.6, 3.2, 2.0, "Cross-Territory Attention Fusion\n---------------------------------------\n• Squeeze: GlobalAvgPool1D (Batch, 128)\n• Excitation: Dense(16, ReLU)\n• Softmax Weighting: Dense(4, Softmax)\n• Re-weighted Feature Multiply", c_attn, '#f57f17')
    draw_arrow(14.2, 4.6, 15.0, 4.6)

    # Head
    draw_box(15.0, 3.8, 2.4, 1.6, "Flatten + Dropout(0.30)\nDense (5 Classes)\nSigmoid Head\n--------------------\n[NORM, MI, STTC, CD, HYP]", c_head, '#2e7d32')

    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'architectural_diagram_anatomical_multibranch.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 to '{path}'")


# -----------------------------------------------------------------------------
# FIGURE 3: CAUSAL DDPM COUNTERFACTUAL WAVEFORM GENERATION PIPELINE
# -----------------------------------------------------------------------------
def create_ddpm_counterfactual_figure():
    fig, ax = plt.subplots(figsize=(17, 8), dpi=300)
    ax.axis('off')

    c_abnormal = '#ffebee'
    c_noise = '#eceff1'
    c_unet = '#bbdefb'
    c_guidance = '#fff9c4'
    c_cf = '#e8f5e9'

    def draw_box(x, y, w, h, text, color='#ffffff', border='#333333', fontsize=9.5, fontweight='bold'):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.4, edgecolor=border, facecolor=color, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color='#222222', zorder=3)

    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate(text, xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#444444', edgecolor='#444444', arrowstyle="->", lw=1.6),
                    ha='center', va='bottom', fontsize=8, color='#333333', zorder=4)

    plt.title("Causal 1D Conditional DDPM Counterfactual ECG Waveform Generation Pipeline",
              fontsize=14, fontweight='bold', pad=15, color='#111111')

    # Input Abnormal ECG
    draw_box(0.5, 3.2, 2.4, 1.8, "Abnormal ECG Input (x0)\nShape: (1000, 12)\n------------------------\nPathology: MI / STTC / CD\nPatient Identity Anchored", c_abnormal, '#c62828')
    draw_arrow(2.9, 4.1, 3.7, 4.1, "Forward Noising (t*=50)")

    # Noised Latent
    draw_box(3.7, 3.2, 2.4, 1.8, "Noised Latent (x_t*)\nt* = 50 / 1000 steps\n------------------------\nPreserves low-frequency\nidentity (Rhythm, Axis)", c_noise, '#37474f')
    draw_arrow(6.1, 4.1, 6.9, 4.1)

    # 1D Res-UNet Generator
    draw_box(6.9, 2.8, 3.2, 2.6, "1D Temporal Res-UNet\nGenerator (3.24M Params)\n-----------------------------------\n• 4 Encoder & Decoder Blocks\n• Sinusoidal Time Embedding t\n• Class Embedding c_target = NORM\n• Predicts Noise ε_θ(x_t, t, c)", c_unet, '#1565c0')
    draw_arrow(10.1, 4.1, 10.9, 4.1)

    # Dual Guidance Block (Above/Below Integration)
    draw_box(6.9, 5.7, 6.4, 1.7, "DUAL COUNTERFACTUAL GUIDANCE ENGINE\n--------------------------------------------------------------\n1. Classifier-Free Guidance (CFG s=1.0): ε_hat = ε_θ(x_t, t, ∅) + s·(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))\n2. Anatomical Classifier Guidance: x_t_grad = x_t + γ·∇_{x_t} log p_phi(c_NORM | x_t)", c_guidance, '#f57f17', fontsize=8.5)

    draw_arrow(10.1, 5.7, 10.1, 4.7)

    # Reverse Diffusion Loops
    draw_box(10.9, 3.2, 2.6, 1.8, "Reverse Diffusion Loop\n(t* → 0 Steps)\n------------------------\nSurgical Regional Editing\n(Local V1-V4 / II,III,aVF)", '#e0f2f1', '#00695c')
    draw_arrow(13.5, 4.1, 14.3, 4.1)

    # Generated Counterfactual Output
    draw_box(14.3, 3.2, 2.6, 1.8, "Counterfactual ECG (x0^CF)\nShape: (1000, 12)\n------------------------\nTarget: Normal Sinus Rhythm\nIdentity Preserved (L1=0.047)\nZero HR Deviation", c_cf, '#2e7d32')

    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'architectural_diagram_causal_ddpm_counterfactual.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 3 to '{path}'")


def main():
    create_calibrated_architecture_figure()
    create_anatomical_architecture_figure()
    create_ddpm_counterfactual_figure()
    print("\nAll 3 Architectural Diagrams rendered successfully!")


if __name__ == '__main__':
    main()
