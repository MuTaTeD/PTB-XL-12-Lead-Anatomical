"""
run_phase3_counterfactual_cohort.py

Phase 3: Large-Scale Causal Counterfactual ECG Waveform Generation Benchmark
Evaluates 100 unseen Fold 10 test patients across MI, STTC, CD, HYP, and NORM.
Measures counterfactual diagnostic flips, rhythm invariance, and signal edit metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import find_peaks

from causal_diffusion.unet_1d import build_conditional_unet_1d
from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
from causal_diffusion.classifier import build_calibrated_se_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data
from train_conditional_ddpm import diff_schedule, TIMESTEPS, CACHE_PATH, DATA_DIR

CHECKPOINT_DIR = 'checkpoints'
DDPM_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'old training', 'ddpm_unet_1d_ptbxl.h5')
GUIDE_CLASSIFIER_PATH = os.path.join(CHECKPOINT_DIR, 'se_resnet_anatomical_territory_dropout_fold_9_best.h5')
EVAL_CLASSIFIER_PATH = os.path.join(CHECKPOINT_DIR, 'se_resnet_calibrated_fold_9_best.h5')
EXPERIMENTS_DIR = 'experiments'
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CFG_WEIGHT = 1.5


def estimate_heart_rate(ecg_signal, fs=100):
    lead_ii = ecg_signal[:, 1]
    peaks, _ = find_peaks(lead_ii, distance=int(fs * 0.4), prominence=0.3)
    if len(peaks) < 2:
        return 60.0
    rr_intervals = np.diff(peaks) / fs
    mean_rr = np.mean(rr_intervals)
    return round(60.0 / mean_rr, 1)


def compute_lead_cosine_similarity(orig_ecg, cf_ecg):
    sims = []
    for l in range(12):
        u = orig_ecg[:, l]
        v = cf_ecg[:, l]
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u > 0 and norm_v > 0:
            sims.append(np.dot(u, v) / (norm_u * norm_v))
    return np.mean(sims)


def generate_ddpm_counterfactual(unet_model, classifier_model, x_orig, t_start=150, classifier_scale=2.5):
    x_orig_tensor = tf.cast(x_orig[None, :, :], tf.float32)
    target_c_tensor = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
    uncond_c_tensor = tf.zeros_like(target_c_tensor)

    noise_init = tf.random.normal(shape=x_orig_tensor.shape)
    sqrt_alpha_t = diff_schedule.sqrt_alphas_cumprod[t_start]
    sqrt_one_minus_alpha_t = diff_schedule.sqrt_one_minus_alphas_cumprod[t_start]
    x_t = sqrt_alpha_t * x_orig_tensor + sqrt_one_minus_alpha_t * noise_init

    for t_idx in reversed(range(1, t_start + 1)):
        t_batch = tf.constant([t_idx], dtype=tf.int32)
        eps_cond = unet_model([x_t, t_batch, target_c_tensor], training=False)
        eps_uncond = unet_model([x_t, t_batch, uncond_c_tensor], training=False)
        eps_cfg = (1.0 + CFG_WEIGHT) * eps_cond - CFG_WEIGHT * eps_uncond

        with tf.GradientTape() as tape:
            tape.watch(x_t)
            probs = classifier_model(x_t, training=False)
            log_p_target = tf.math.log(probs[:, 0] + 1e-7)
        grad = tape.gradient(log_p_target, x_t)

        beta_t = diff_schedule.betas[t_idx]
        alpha_t = diff_schedule.alphas[t_idx]
        sqrt_one_minus_alpha_bar_t = diff_schedule.sqrt_one_minus_alphas_cumprod[t_idx]

        base_mean = (1.0 / np.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_cfg)
        guided_mean = base_mean + classifier_scale * sqrt_one_minus_alpha_bar_t * grad
        x_t = guided_mean

    return x_t[0].numpy()


def load_cached_fold10_test_data():
    npz = np.load(CACHE_PATH)
    all_X = npz['X']
    cached_ecg_ids = npz['ecg_ids']

    filenames, targets = load_ptbxl_superclass_data(DATA_DIR, folds=[10])
    db_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
    fold10_df = db_df[db_df['strat_fold'] == 10]
    target_ecg_ids = fold10_df['ecg_id'].values

    id_to_idx = {ecg_id: idx for idx, ecg_id in enumerate(cached_ecg_ids)}
    valid_indices = [id_to_idx[ecg_id] for ecg_id in target_ecg_ids if ecg_id in id_to_idx]

    X10 = all_X[valid_indices]
    Y10 = targets
    ecg_ids = target_ecg_ids
    return X10, Y10, ecg_ids


def main():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    print("=" * 80)
    print("PHASE 3: CAUSAL DDPM COUNTERFACTUAL BENCHMARK (100 FOLD-10 TEST PATIENTS)")
    print("=" * 80)

    print(f"[Loading Models] Guide Classifier (Territory Dropout): '{GUIDE_CLASSIFIER_PATH}'...")
    guide_classifier = build_anatomical_se_ecg_classifier()
    guide_classifier.load_weights(GUIDE_CLASSIFIER_PATH)

    print(f"[Loading Models] Independent Eval Classifier (Baseline 1D): '{EVAL_CLASSIFIER_PATH}'...")
    eval_classifier = build_calibrated_se_ecg_classifier()
    eval_classifier.load_weights(EVAL_CLASSIFIER_PATH)

    print(f"[Loading Models] Generative DDPM U-Net: '{DDPM_MODEL_PATH}'...")
    unet = build_conditional_unet_1d(base_filters=32)
    unet.load_weights(DDPM_MODEL_PATH)

    X10, Y10, ecg_ids = load_cached_fold10_test_data()

    # Select 20 representative patients per superclass (20 * 5 = 100 total)
    cohort_indices = []
    class_counts = {sc: 0 for sc in SUPERCLASSES}

    for idx in range(len(X10)):
        y = Y10[idx]
        x_orig = X10[idx]

        for sc_idx, sc_name in enumerate(SUPERCLASSES):
            if class_counts[sc_name] < 20 and y[sc_idx] == 1:
                # 1. Purity Filter: Exclude multi-morbid pathological patients
                if np.sum(y[1:]) > 1:
                    continue
                
                # 2. Evaluator Pre-Verification: The independent judge must agree with the label
                prob_orig = eval_classifier.predict(x_orig[None, :, :], verbose=0)[0]
                
                if sc_name == 'NORM':
                    # Evaluator must confidently predict NORM
                    if prob_orig[0] < 0.8:
                        continue
                else:
                    # Evaluator must confidently predict the pathology AND reject NORM
                    if prob_orig[sc_idx] < 0.5 or prob_orig[0] >= 0.5:
                        continue

                cohort_indices.append((sc_name, idx))
                class_counts[sc_name] += 1
                break

    print(f"[Cohort Selection] Selected {len(cohort_indices)} patients across superclasses:")
    for sc_name, cnt in class_counts.items():
        print(f"  - {sc_name:5s}: {cnt} patients")

    results = []

    print("\n" + "=" * 80)
    print("PROCESSING COUNTERFACTUAL GENERATION & METRIC AUDIT")
    print("=" * 80)

    for step_num, (primary_sc, idx) in enumerate(cohort_indices, 1):
        ecg_id = ecg_ids[idx]
        x_orig = X10[idx]
        y_orig = Y10[idx]
        orig_diseases = [SUPERCLASSES[j] for j in range(5) if y_orig[j] == 1]

        prob_orig = eval_classifier.predict(x_orig[None, :, :], verbose=0)[0]
        orig_norm_prob = prob_orig[0]

        # Pathology-specific parameters
        if orig_norm_prob >= 0.80:
            adaptive_t_start = 10
            scale_param = 0.5
        elif primary_sc == 'HYP':
            adaptive_t_start = 200
            scale_param = 4.0
        elif primary_sc == 'STTC':
            adaptive_t_start = 180
            scale_param = 3.5
        elif primary_sc in ['MI', 'CD']:
            adaptive_t_start = 160
            scale_param = 2.5
        else:
            adaptive_t_start = 150
            scale_param = 2.5

        x_cf = generate_ddpm_counterfactual(unet, guide_classifier, x_orig, t_start=adaptive_t_start, classifier_scale=scale_param)
        prob_cf = eval_classifier.predict(x_cf[None, :, :], verbose=0)[0]
        cf_norm_prob = prob_cf[0]

        # Metric Calculations
        hr_orig = estimate_heart_rate(x_orig)
        hr_cf = estimate_heart_rate(x_cf)
        hr_err = abs(hr_orig - hr_cf)

        delta_x = x_cf - x_orig
        l1_norm = float(np.mean(np.abs(delta_x)))
        l2_norm = float(np.sqrt(np.mean(delta_x ** 2)))
        cos_sim = float(compute_lead_cosine_similarity(x_orig, x_cf))
        sparsity = float(np.mean(np.abs(delta_x) < 0.05)) # % of unedited points (<0.05 mV edit)

        flip_success = bool(cf_norm_prob >= 0.80)

        results.append({
            'Patient_Index': step_num,
            'ECG_ID': int(ecg_id),
            'Primary_Superclass': primary_sc,
            'Original_Diagnoses': ", ".join(orig_diseases),
            'Orig_NORM_Prob': float(orig_norm_prob),
            'CF_NORM_Prob': float(cf_norm_prob),
            'Diagnostic_Flip_Success': flip_success,
            'Orig_HR_bpm': float(hr_orig),
            'CF_HR_bpm': float(hr_cf),
            'HR_Error_bpm': float(hr_err),
            'L1_Edit_Norm': l1_norm,
            'L2_Edit_Norm': l2_norm,
            'Lead_Cosine_Sim': cos_sim,
            'Sparsity_Ratio': sparsity
        })

        # Save CSV incrementally after every patient so file exists immediately!
        df_temp = pd.DataFrame(results)
        out_csv = os.path.join(EXPERIMENTS_DIR, 'phase3_counterfactual_cohort_100_patients.csv')
        df_temp.to_csv(out_csv, index=False)

        print(f"[{step_num:3d}/{len(cohort_indices)}] ECG ID {ecg_id:5d} ({primary_sc:5s}) | NORM Prob: {orig_norm_prob:.4f} -> {cf_norm_prob:.4f} | HR Err: {hr_err:.1f} bpm | L1: {l1_norm:.4f}")

    print(f"\n[Saved Results] Final 100-patient audit CSV updated at '{out_csv}'")

    df_res = pd.DataFrame(results)

    # Cohort Summary Statistics
    print("\n" + "=" * 80)
    print("PHASE 3 COHORT BENCHMARK SUMMARY STATISTICS")
    print("=" * 80)
    for sc in SUPERCLASSES:
        sc_df = df_res[df_res['Primary_Superclass'] == sc]
        flip_rate = sc_df['Diagnostic_Flip_Success'].mean() * 100
        mean_l1 = sc_df['L1_Edit_Norm'].mean()
        mean_hr_err = sc_df['HR_Error_bpm'].mean()
        mean_cos = sc_df['Lead_Cosine_Sim'].mean()
        mean_cf_norm = sc_df['CF_NORM_Prob'].mean()
        print(f"Class: {sc:5s} | Flip Rate: {flip_rate:5.1f}% | Avg CF NORM Prob: {mean_cf_norm:.4f} | Avg HR Err: {mean_hr_err:.2f} bpm | Avg CosSim: {mean_cos:.4f} | Avg L1: {mean_l1:.4f}")

    total_flip_rate = df_res[df_res['Primary_Superclass'] != 'NORM']['Diagnostic_Flip_Success'].mean() * 100
    avg_hr_err_overall = df_res['HR_Error_bpm'].mean()
    avg_cos_overall = df_res['Lead_Cosine_Sim'].mean()
    print("-" * 80)
    print(f"Overall Pathological Flip Rate (MI, STTC, CD, HYP -> NORM): {total_flip_rate:.1f}%")
    print(f"Overall Heart Rate Error: {avg_hr_err_overall:.2f} bpm")
    print(f"Overall Lead Cosine Similarity: {avg_cos_overall:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
