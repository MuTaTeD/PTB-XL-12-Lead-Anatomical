"""
causal_diffusion package
Causal Counterfactual Waveform Generation via Denoising Diffusion using PTB-XL
"""

from .physics import (
    compute_physics_violation_loss,
    project_signal_to_physics,
    evaluate_biophysical_residuals,
    LEAD_NAMES
)
from .classifier import build_ecg_classifier
from .diffusion_model import build_conditional_unet_1d, CausalDiffusionEngine, DiffusionSchedule
from .counterfactual_solver import CounterfactualOptimizer
from .evaluation import evaluate_counterfactual_quality
from .visualize import plot_subtractive_counterfactual_12lead
