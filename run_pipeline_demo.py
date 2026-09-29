"""
run_pipeline_demo.py
End-to-End Pipeline Execution for Causal Counterfactual Waveform Generation on PTB-XL v1.0.3.
Demonstrates:
1. Loading genuine LBBB abnormal ECG from PTB-XL v1.0.3.
2. Delineating QRS complex vs innocent P-waves/baseline.
3. Instantiating Keras 1D Classifier and Conditional Diffusion Model.
4. Solving for minimal counterfactual perturbation Delta_X to flip diagnosis to NORM.
5. Projecting onto Einthoven & Goldberger biophysical null-space.
6. Generating Subtractive Waveform Delta (Delta-ECG) and clinical metrics.
7. Rendering and saving 12-lead multi-channel visualization.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

from causal_diffusion.dataset import load_ptbxl_metadata, load_ecg_record, estimate_qrs_and_p_masks
from causal_diffusion.physics import project_signal_to_physics, evaluate_biophysical_residuals
from causal_diffusion.classifier import build_ecg_classifier
from causal_diffusion.diffusion_model import build_conditional_unet_1d, CausalDiffusionEngine
from causal_diffusion.counterfactual_solver import CounterfactualOptimizer
from causal_diffusion.evaluation import evaluate_counterfactual_quality
from causal_diffusion.visualize import plot_subtractive_counterfactual_12lead


def main():
    data_dir = 'dataset-1.0.3/'
    print("=" * 70)
    print("CAUSAL COUNTERFACTUAL ECG WAVEFORM GENERATION PIPELINE (PTB-XL v1.0.3)")
    print("=" * 70)

    # 1. Load Metadata
    print("\n[Step 1] Loading PTB-XL database...")
    df = load_ptbxl_metadata(data_dir)
    print(f"Total ECG records indexed: {len(df)}")

    # Find a representative LBBB record
    lbbb_records = df[df['primary_class'] == 'LBBB']
    print(f"Total Left Bundle Branch Block (LBBB) records available: {len(lbbb_records)}")
    sample_ecg_id = lbbb_records.index[0]
    sample_filename = lbbb_records.loc[sample_ecg_id, 'filename_lr']
    sample_patient_id = lbbb_records.loc[sample_ecg_id, 'patient_id']
    sample_age = lbbb_records.loc[sample_ecg_id, 'age']
    sample_sex = 'Male' if lbbb_records.loc[sample_ecg_id, 'sex'] == 0 else 'Female'
    print(f"Selected Factual Subject: ECG ID {sample_ecg_id}, Patient ID {sample_patient_id}, Age {sample_age}, Sex {sample_sex}")

    # 2. Load and preprocess factual ECG
    print("\n[Step 2] Loading factual 12-lead waveform (10s @ 100Hz)...")
    x0_raw = load_ecg_record(data_dir, sample_filename, apply_filter=True, fs=100.0)  # (1000, 12)
    # Strictly enforce biophysical laws on factual baseline
    x0_physio = project_signal_to_physics(x0_raw).numpy()

    # Delineate QRS and P-wave masks
    qrs_mask, anat_mask = estimate_qrs_and_p_masks(x0_physio, fs=100.0)
    p_mask = anat_mask.astype(bool)
    print(f"Segmented QRS complex samples: {np.sum(qrs_mask[:, 0])} / 1000 ({np.sum(qrs_mask[:, 0])/10:.1f}%)")
    print(f"Segmented innocent P-wave / baseline samples: {np.sum(p_mask[:, 0])} / 1000 ({np.sum(p_mask[:, 0])/10:.1f}%)")

    # 3. Initialize Models in TensorFlow/Keras
    print("\n[Step 3] Initializing Keras 1D ResNet Classifier & Conditional DDPM...")
    classifier = build_ecg_classifier(input_shape=(1000, 12), num_classes=5)
    ddpm_unet = build_conditional_unet_1d(seq_len=1000, in_channels=12, num_classes=5)
    print("1D ResNet Classifier parameters:", classifier.count_params())
    print("Conditional 1D U-Net parameters:", ddpm_unet.count_params())

    # 4. Minimal Counterfactual Waveform Optimization
    print("\n[Step 4] Solving for Minimal Counterfactual Perturbation (Delta-X)...")
    print("Minimizing: ||Delta_X||_2^2 + lambda_1 ||Delta_X||_1 + gamma*L_clf + beta*D_physio + kappa*D_anat")
    optimizer = CounterfactualOptimizer(
        classifier_model=classifier,
        target_class_idx=0,   # Class 0: NORM
        gamma=25.0,
        beta=20.0,
        lambda_l2=1.0,
        lambda_sparse=0.5,
        lambda_tv=0.3,
        kappa_anat=15.0,
        learning_rate=0.015,
        num_iterations=120
    )

    x0_tensor = tf.convert_to_tensor(x0_physio[np.newaxis, ...], dtype=tf.float32)
    anat_tensor = tf.convert_to_tensor(anat_mask[np.newaxis, ...], dtype=tf.float32)

    x_cf_tensor, delta_x_tensor, history = optimizer.optimize_perturbation(
        x0=x0_tensor,
        anat_mask=anat_tensor
    )

    x_cf = x_cf_tensor.numpy()[0]
    delta_x = delta_x_tensor.numpy()[0]

    # 5. Evaluate Clinical and Biophysical Metrics
    print("\n[Step 5] Evaluating Clinical & Biophysical Metrics...")
    metrics = evaluate_counterfactual_quality(x0_physio, x_cf, delta_x, qrs_mask, p_mask)

    print("-" * 55)
    print(f"Target Classification Flip (P(NORM)): {history['prob_norm'][-1]:.4f}")
    print(f"Pathological Energy Localization (PELS_QRS): {metrics['pels_qrs_localization'] * 100:.2f}% (Target: > 90%)")
    print(f"P-Wave Invariance Leakage (PWLR):         {metrics['pwlr_pwave_leakage'] * 100:.3f}% (Target: < 1.0%)")
    print(f"Waveform Sparsity (|Delta_X| < 0.05 mV):  {metrics['waveform_sparsity'] * 100:.2f}%")
    print(f"Delta-X L2 Norm:                          {metrics['delta_l2_norm']:.4f}")
    print(f"Max Einthoven Residual (Circuit Law):    {metrics['einthoven_residual_mV']:.2e} mV")
    print(f"Max Goldberger aVR Residual:             {metrics['goldberger_avr_residual_mV']:.2e} mV")
    print("-" * 55)

    # 6. Generate 12-Lead Subtractive Visualization
    output_img_path = "output_subtractive_counterfactual.png"
    print(f"\n[Step 6] Rendering 12-Lead Subtractive Delta-ECG to '{output_img_path}'...")
    plot_subtractive_counterfactual_12lead(
        x0=x0_physio,
        x_cf=x_cf,
        delta_x=delta_x,
        title=f"Subtractive Waveform Counterfactual (Delta-ECG) | PTB-XL ECG ID {sample_ecg_id} (LBBB -> NORM)",
        save_path=output_img_path,
        fs=100.0,
        time_window_sec=4.0
    )
    print(f"Visualization successfully saved at: {os.path.abspath(output_img_path)}")
    print("\nPipeline execution completed successfully!")


if __name__ == '__main__':
    main()
