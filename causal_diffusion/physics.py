"""
causal_diffusion/physics.py
Biophysical Circuit Laws and Constraint Projection for 12-Lead ECGs in TensorFlow.

Governing Laws:
1. Einthoven's Triangle Law:
   Lead II = Lead I + Lead III  <=>  Lead I - Lead II + Lead III = 0

2. Goldberger's Augmented Unipolar Limb Leads:
   aVR = -(Lead I + Lead II) / 2  <=>  0.5 * Lead I + 0.5 * Lead II + Lead aVR = 0
   aVL = (Lead I - Lead III) / 2  <=>  -Lead I + 0.5 * Lead II + Lead aVL = 0
   aVF = (Lead II + Lead III) / 2 <=>  -0.5 * Lead I + Lead II + Lead aVF = 0

Standard 12-lead index order:
0: Lead I
1: Lead II
2: Lead III
3: Lead aVR
4: Lead aVL
5: Lead aVF
6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
"""

import tensorflow as tf
import numpy as np


# Lead indices
LEAD_I = 0
LEAD_II = 1
LEAD_III = 2
LEAD_AVR = 3
LEAD_AVL = 4
LEAD_AVF = 5

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def build_physics_matrix():
    """
    Constructs the 4x12 linear biophysical constraint matrix M_physio such that:
    M_physio * x(t) = 0 for any physiologically valid 12-lead vector x(t).
    """
    M = np.zeros((4, 12), dtype=np.float32)

    # 1. Einthoven: I - II + III = 0
    M[0, LEAD_I] = 1.0
    M[0, LEAD_II] = -1.0
    M[0, LEAD_III] = 1.0

    # 2. Goldberger aVR: 0.5 * I + 0.5 * II + aVR = 0
    M[1, LEAD_I] = 0.5
    M[1, LEAD_II] = 0.5
    M[1, LEAD_AVR] = 1.0

    # 3. Goldberger aVL: -I + 0.5 * II + aVL = 0
    M[2, LEAD_I] = -1.0
    M[2, LEAD_II] = 0.5
    M[2, LEAD_AVL] = 1.0

    # 4. Goldberger aVF: -0.5 * I + II - aVF = 0 (since aVF = II - 0.5*I)
    M[3, LEAD_I] = -0.5
    M[3, LEAD_II] = 1.0
    M[3, LEAD_AVF] = -1.0

    return tf.constant(M, dtype=tf.float32)


def build_nullspace_projector():
    """
    Constructs the orthogonal projection matrix onto the null space of M_physio:
    Pi_physio = I - M^T (M M^T)^(-1) M = I - M^+ M
    Any signal multiplied by Pi_physio strictly satisfies Einthoven and Goldberger laws.
    """
    M = np.zeros((4, 12), dtype=np.float64)
    # Einthoven: I - II + III = 0
    M[0, LEAD_I] = 1.0
    M[0, LEAD_II] = -1.0
    M[0, LEAD_III] = 1.0

    # Goldberger aVR: aVR = -(I + II)/2 <=> 0.5*I + 0.5*II + aVR = 0
    M[1, LEAD_I] = 0.5
    M[1, LEAD_II] = 0.5
    M[1, LEAD_AVR] = 1.0

    # Goldberger aVL: aVL = I - II/2 <=> -I + 0.5*II + aVL = 0
    M[2, LEAD_I] = -1.0
    M[2, LEAD_II] = 0.5
    M[2, LEAD_AVL] = 1.0

    # Goldberger aVF: aVF = II - I/2 <=> -0.5*I + II - aVF = 0
    M[3, LEAD_I] = -0.5
    M[3, LEAD_II] = 1.0
    M[3, LEAD_AVF] = -1.0

    # Compute pseudo-inverse M^+ = M^T (M M^T)^(-1)
    M_pinv = np.linalg.pinv(M)
    Pi = np.eye(12, dtype=np.float64) - np.dot(M_pinv, M)

    return tf.constant(Pi, dtype=tf.float32)


# Precomputed global tensors
M_PHYSIO = build_physics_matrix()
PI_PHYSIO = build_nullspace_projector()


@tf.function
def compute_physics_violation_loss(x):
    """
    Computes D_physio(x) = mean squared violation of Einthoven and Goldberger laws.
    Args:
        x: Tensor of shape (batch_size, time_steps, 12) or (time_steps, 12)
    Returns:
        Scalar MSE violation loss.
    """
    x = tf.cast(x, tf.float32)
    if len(x.shape) == 2:
        x = tf.expand_dims(x, 0)  # (1, T, 12)

    lead_I = x[:, :, LEAD_I]
    lead_II = x[:, :, LEAD_II]
    lead_III = x[:, :, LEAD_III]
    lead_aVR = x[:, :, LEAD_AVR]
    lead_aVL = x[:, :, LEAD_AVL]
    lead_aVF = x[:, :, LEAD_AVF]

    # 1. Einthoven residual: II - (I + III)
    res_einthoven = lead_II - (lead_I + lead_III)

    # 2. Goldberger residuals
    res_avr = lead_aVR - (-(lead_I + lead_II) / 2.0)
    res_avl = lead_aVL - ((lead_I - lead_III) / 2.0)
    res_avf = lead_aVF - ((lead_II + lead_III) / 2.0)

    # 3. Wilson Central Terminal null-sum: aVR + aVL + aVF = 0
    res_wct = lead_aVR + lead_aVL + lead_aVF

    loss = (
        tf.reduce_mean(tf.square(res_einthoven)) +
        tf.reduce_mean(tf.square(res_avr)) +
        tf.reduce_mean(tf.square(res_avl)) +
        tf.reduce_mean(tf.square(res_avf)) +
        tf.reduce_mean(tf.square(res_wct))
    )
    return loss


@tf.function
def project_signal_to_physics(x):
    """
    Strictly projects a 12-lead ECG signal onto the null-space of Einthoven/Goldberger laws.
    Args:
        x: Tensor of shape (batch_size, time_steps, 12) or (time_steps, 12)
    Returns:
        x_proj: Tensor of same shape satisfying biophysical circuit laws to machine precision.
    """
    x = tf.cast(x, tf.float32)
    return tf.tensordot(x, PI_PHYSIO, axes=[[-1], [0]])


def evaluate_biophysical_residuals(x_numpy):
    """
    Evaluates absolute mean and max residuals for clinician verification.
    Args:
        x_numpy: numpy array of shape (T, 12) or (N, T, 12)
    """
    if len(x_numpy.shape) == 2:
        x_numpy = np.expand_dims(x_numpy, 0)

    lead_I = x_numpy[:, :, LEAD_I]
    lead_II = x_numpy[:, :, LEAD_II]
    lead_III = x_numpy[:, :, LEAD_III]
    lead_aVR = x_numpy[:, :, LEAD_AVR]
    lead_aVL = x_numpy[:, :, LEAD_AVL]
    lead_aVF = x_numpy[:, :, LEAD_AVF]

    res_einthoven = np.abs(lead_II - (lead_I + lead_III))
    res_avr = np.abs(lead_aVR - (-(lead_I + lead_II) / 2.0))
    res_avl = np.abs(lead_aVL - ((lead_I - lead_III) / 2.0))
    res_avf = np.abs(lead_aVF - ((lead_II + lead_III) / 2.0))

    return {
        'einthoven_mean_mV': float(np.mean(res_einthoven)),
        'einthoven_max_mV': float(np.max(res_einthoven)),
        'avr_mean_mV': float(np.mean(res_avr)),
        'avr_max_mV': float(np.max(res_avr)),
        'avl_mean_mV': float(np.mean(res_avl)),
        'avl_max_mV': float(np.max(res_avl)),
        'avf_mean_mV': float(np.mean(res_avf)),
        'avf_max_mV': float(np.max(res_avf)),
    }
