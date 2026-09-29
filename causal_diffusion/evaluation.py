"""
causal_diffusion/evaluation.py
Clinical Explainability & Biophysical Metrics for Subtractive Counterfactuals (Delta-ECG).
"""

import numpy as np
from causal_diffusion.physics import evaluate_biophysical_residuals


def evaluate_counterfactual_quality(x0, x_cf, delta_x, qrs_mask, p_mask):
    """
    Computes clinician-grade quantitative evaluation metrics:
    1. PELS: Pathological Energy Localization Score inside QRS complex.
    2. PWLR: P-Wave Leakage Ratio (energy leaked into innocent P-waves).
    3. Residuals: Einthoven and Goldberger physical law discrepancies.
    4. Sparsity and L2 norm.
    """
    delta_sq = np.square(delta_x)
    total_energy = np.sum(delta_sq) + 1e-12

    # QRS energy
    qrs_energy = np.sum(delta_sq[qrs_mask])
    pels = float(qrs_energy / total_energy)

    # P-wave energy
    p_energy = np.sum(delta_sq[p_mask])
    pwlr = float(p_energy / total_energy)

    # Sparsity: fraction of samples with edit magnitude < 0.05 mV
    sparsity_frac = float(np.mean(np.abs(delta_x) < 0.05))

    # Biophysical residuals of counterfactual
    physio_metrics = evaluate_biophysical_residuals(x_cf)

    return {
        'pels_qrs_localization': pels,
        'pwlr_pwave_leakage': pwlr,
        'waveform_sparsity': sparsity_frac,
        'delta_l2_norm': float(np.linalg.norm(delta_x)),
        'delta_l1_norm': float(np.sum(np.abs(delta_x))),
        'einthoven_residual_mV': physio_metrics['einthoven_max_mV'],
        'goldberger_avr_residual_mV': physio_metrics['avr_max_mV'],
    }
