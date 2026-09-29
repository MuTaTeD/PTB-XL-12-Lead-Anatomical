"""
causal_diffusion/visualize.py
Clinician Multi-Lead Visualization for Subtractive Waveform Counterfactuals (Delta-ECG).
"""

import matplotlib.pyplot as plt
import numpy as np
from causal_diffusion.physics import LEAD_NAMES


def plot_subtractive_counterfactual_12lead(
    x0,
    x_cf,
    delta_x,
    title="Subtractive Waveform Counterfactual (Delta-ECG)",
    save_path=None,
    fs=100.0,
    time_window_sec=5.0
):
    """
    Plots a 12-lead ECG comparison:
    - Lead by lead: Factual X_0 (Red), Counterfactual X_CF (Blue), Delta-ECG (Green).
    Args:
        x0: (1000, 12) numpy array
        x_cf: (1000, 12) numpy array
        delta_x: (1000, 12) numpy array
        title: Title of plot
        save_path: Optional path to save image
        fs: Sampling frequency (Hz)
        time_window_sec: Window length to display (e.g. 5 seconds)
    """
    num_samples = int(min(x0.shape[0], time_window_sec * fs))
    time_axis = np.arange(num_samples) / fs

    fig, axes = plt.subplots(6, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten(order='F')  # Column-major: Left col = Limb leads, Right col = Precordial leads

    for c in range(12):
        ax = axes[c]
        ax.plot(time_axis, x0[:num_samples, c], label='Factual X0 (Pathology)', color='#d9534f', linewidth=1.5)
        ax.plot(time_axis, x_cf[:num_samples, c], label='Counterfactual X_CF (Normal)', color='#0275d8', linewidth=1.5, linestyle='--')
        ax.plot(time_axis, delta_x[:num_samples, c], label=r'Subtractive $\Delta$-ECG ($X_0 - X_{CF}$)', color='#5cb85c', linewidth=1.8)

        ax.set_ylabel(f"Lead {LEAD_NAMES[c]} (mV)", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        if c == 0:
            ax.legend(loc='upper right', fontsize=8)
        if c in [5, 11]:
            ax.set_xlabel("Time (seconds)", fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
