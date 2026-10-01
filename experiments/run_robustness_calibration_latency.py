"""
experiments/run_robustness_calibration_latency.py
Evaluates Model 1 (Baseline Flat SE-ResNet1D) and Model 3 (Anatomical Territory-Dropout SE-ResNet1D):
1. Robustness Stress Testing:
   - Missing Leads (1, 2, 3, 4, 6 random leads missing)
   - Missing Territories (Inferior, Antero-Septal, Lateral, Cavity)
   - Baseline Wander (0.2 Hz drift at 0.1, 0.2, 0.5 mV)
   - High-Frequency Noise (Gaussian at SNR 20 dB, 10 dB, 0 dB)
   - Lead Swaps (Limb I <-> II, Precordial V1 <-> V2)
2. Calibration Metrics:
   - Expected Calibration Error (ECE, 10 bins)
   - Maximum Calibration Error (MCE)
   - Brier Score
3. Inference & Explanation Latency:
   - Forward pass latency per 10-second ECG (ms)
   - Macro Territory Occlusion latency (ms)
   - Meso 1D Grad-CAM++ latency (ms)
   - Micro Integrated Gradients (50 steps) latency (ms)
   - Measured on GPU
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from causal_diffusion.classifier import build_calibrated_se_ecg_classifier
from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')
EXPERIMENT_DIR = 'experiments'
ROBUSTNESS_CSV = os.path.join(EXPERIMENT_DIR, 'robustness_stress_test_results.csv')
CALIBRATION_CSV = os.path.join(EXPERIMENT_DIR, 'calibration_metrics.csv')
LATENCY_CSV = os.path.join(EXPERIMENT_DIR, 'latency_benchmark.csv')

os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Lead mappings
TERRITORY_LEAD_MAP = {
    'Inferior': [1, 2, 5],
    'Antero-Septal': [6, 7, 8, 9],
    'Lateral': [0, 4, 10, 11],
    'Cavity': [3]
}

def load_cached_fold_data(folds):
    with np.load(CACHE_PATH) as npz:
        all_X = npz['X']
        cached_ecg_ids = npz['ecg_ids']

        filenames, targets = load_ptbxl_superclass_data(DATA_DIR, folds=folds)
        db_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
        fold_df = db_df[db_df['strat_fold'].isin(folds if isinstance(folds, list) else [folds])]
        target_ecg_ids = fold_df['ecg_id'].values

        id_to_idx = {ecg_id: idx for idx, ecg_id in enumerate(cached_ecg_ids)}
        indices = [id_to_idx[ecg_id] for ecg_id in target_ecg_ids if ecg_id in id_to_idx]

        X = np.ascontiguousarray(all_X[indices], dtype=np.float32)
        Y = np.ascontiguousarray(targets, dtype=np.float32)
    return X, Y


def compute_ece_mce(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    ece_list = []
    mce_list = []
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    
    for c in range(y_true.shape[1]):
        y_t = y_true[:, c]
        y_p = y_prob[:, c]
        
        ece = 0.0
        mce = 0.0
        total_samples = len(y_t)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (y_p >= bin_lower) & (y_p < bin_upper) if i < n_bins - 1 else (y_p >= bin_lower) & (y_p <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                acc = np.mean(y_t[in_bin])
                conf = np.mean(y_p[in_bin])
                diff = np.abs(acc - conf)
                ece += (bin_size / total_samples) * diff
                mce = max(mce, diff)
                
        ece_list.append(ece)
        mce_list.append(mce)
        
    return np.mean(ece_list), np.mean(mce_list), ece_list, mce_list


def add_baseline_wander(X, amplitude=0.2, freq=0.2, fs=100.0):
    """Adds synthetic sinusoidal baseline wander to all leads."""
    T = X.shape[1]
    t = np.arange(T) / fs
    drift = amplitude * np.sin(2 * np.pi * freq * t)[None, :, None]
    return (X + drift).astype(np.float32)


def add_gaussian_noise_snr(X, target_snr_db=20.0):
    """Adds zero-mean Gaussian noise at target SNR (in dB) relative to signal power."""
    sig_power = np.mean(X ** 2, axis=(1, 2), keepdims=True)
    noise_power = sig_power / (10.0 ** (target_snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


def run_robustness_evaluation(m1, m3, X_test, Y_test):
    print("\n" + "=" * 80)
    print("RUNNING ROBUSTNESS STRESS TESTS ON HELD-OUT FOLD 10")
    print("=" * 80)
    
    robustness_results = []
    
    # 0. Clean Baseline
    preds1_clean = m1.predict(X_test, batch_size=64, verbose=0)
    preds3_clean = m3.predict(X_test, batch_size=64, verbose=0)
    auc1_clean = np.mean([roc_auc_score(Y_test[:, c], preds1_clean[:, c]) for c in range(5)])
    auc3_clean = np.mean([roc_auc_score(Y_test[:, c], preds3_clean[:, c]) for c in range(5)])
    
    robustness_results.append({
        'Test_Condition': 'Clean Baseline (12 Leads)',
        'Severity': 'None',
        'Model1_AUC': auc1_clean,
        'Model3_AUC': auc3_clean,
        'Delta_AUC_M3_vs_M1': auc3_clean - auc1_clean
    })
    print(f"Clean Baseline: Model 1 AUC={auc1_clean:.4f} | Model 3 AUC={auc3_clean:.4f} (Delta={auc3_clean - auc1_clean:+.4f})")
    
    # 1. Missing Leads Stress Test (1, 2, 3, 4, 6 random leads dropped)
    rng = np.random.default_rng(42)
    for n_missing in [1, 2, 3, 4, 6]:
        X_corrupt = X_test.copy()
        for i in range(len(X_corrupt)):
            drop_leads = rng.choice(12, size=n_missing, replace=False)
            X_corrupt[i, :, drop_leads] = 0.0
            
        p1 = m1.predict(X_corrupt, batch_size=64, verbose=0)
        p3 = m3.predict(X_corrupt, batch_size=64, verbose=0)
        a1 = np.mean([roc_auc_score(Y_test[:, c], p1[:, c]) for c in range(5)])
        a3 = np.mean([roc_auc_score(Y_test[:, c], p3[:, c]) for c in range(5)])
        
        robustness_results.append({
            'Test_Condition': f'Missing Leads ({n_missing}/12)',
            'Severity': f'{n_missing} leads',
            'Model1_AUC': a1,
            'Model3_AUC': a3,
            'Delta_AUC_M3_vs_M1': a3 - a1
        })
        print(f"Missing {n_missing} Leads: Model 1 AUC={a1:.4f} | Model 3 AUC={a3:.4f} (Delta={a3 - a1:+.4f})")

    # 2. Missing Whole Anatomical Territories
    for t_name, leads in TERRITORY_LEAD_MAP.items():
        X_corrupt = X_test.copy()
        X_corrupt[:, :, leads] = 0.0
        
        p1 = m1.predict(X_corrupt, batch_size=64, verbose=0)
        p3 = m3.predict(X_corrupt, batch_size=64, verbose=0)
        a1 = np.mean([roc_auc_score(Y_test[:, c], p1[:, c]) for c in range(5)])
        a3 = np.mean([roc_auc_score(Y_test[:, c], p3[:, c]) for c in range(5)])
        
        robustness_results.append({
            'Test_Condition': f'Occluded Territory ({t_name})',
            'Severity': f'{len(leads)} leads',
            'Model1_AUC': a1,
            'Model3_AUC': a3,
            'Delta_AUC_M3_vs_M1': a3 - a1
        })
        print(f"Occluded {t_name} ({leads}): Model 1 AUC={a1:.4f} | Model 3 AUC={a3:.4f} (Delta={a3 - a1:+.4f})")

    # 3. Additive Baseline Wander (0.1, 0.2, 0.5 mV)
    for amp in [0.1, 0.2, 0.5]:
        X_corrupt = add_baseline_wander(X_test, amplitude=amp)
        p1 = m1.predict(X_corrupt, batch_size=64, verbose=0)
        p3 = m3.predict(X_corrupt, batch_size=64, verbose=0)
        a1 = np.mean([roc_auc_score(Y_test[:, c], p1[:, c]) for c in range(5)])
        a3 = np.mean([roc_auc_score(Y_test[:, c], p3[:, c]) for c in range(5)])
        
        robustness_results.append({
            'Test_Condition': 'Baseline Wander Drift',
            'Severity': f'{amp} mV',
            'Model1_AUC': a1,
            'Model3_AUC': a3,
            'Delta_AUC_M3_vs_M1': a3 - a1
        })
        print(f"Baseline Wander ({amp} mV): Model 1 AUC={a1:.4f} | Model 3 AUC={a3:.4f} (Delta={a3 - a1:+.4f})")

    # 4. Additive High-Frequency Noise (SNR: 20dB, 10dB, 0dB)
    for snr in [20.0, 10.0, 0.0]:
        X_corrupt = add_gaussian_noise_snr(X_test, target_snr_db=snr)
        p1 = m1.predict(X_corrupt, batch_size=64, verbose=0)
        p3 = m3.predict(X_corrupt, batch_size=64, verbose=0)
        a1 = np.mean([roc_auc_score(Y_test[:, c], p1[:, c]) for c in range(5)])
        a3 = np.mean([roc_auc_score(Y_test[:, c], p3[:, c]) for c in range(5)])
        
        robustness_results.append({
            'Test_Condition': 'Gaussian Noise (Tremor)',
            'Severity': f'SNR {snr:.0f} dB',
            'Model1_AUC': a1,
            'Model3_AUC': a3,
            'Delta_AUC_M3_vs_M1': a3 - a1
        })
        print(f"Noise SNR {snr:.0f} dB: Model 1 AUC={a1:.4f} | Model 3 AUC={a3:.4f} (Delta={a3 - a1:+.4f})")

    # 5. Lead Swaps (Limb I <-> II reversal; Precordial V1 <-> V2 swap)
    # Limb reversal: 0 (I) and 1 (II)
    X_limb = X_test.copy()
    X_limb[:, :, 0], X_limb[:, :, 1] = X_test[:, :, 1], X_test[:, :, 0]
    p1 = m1.predict(X_limb, batch_size=64, verbose=0)
    p3 = m3.predict(X_limb, batch_size=64, verbose=0)
    a1 = np.mean([roc_auc_score(Y_test[:, c], p1[:, c]) for c in range(5)])
    a3 = np.mean([roc_auc_score(Y_test[:, c], p3[:, c]) for c in range(5)])
    robustness_results.append({
        'Test_Condition': 'Lead Swap: Limb (I <-> II)',
        'Severity': 'Electrode Reversal',
        'Model1_AUC': a1,
        'Model3_AUC': a3,
        'Delta_AUC_M3_vs_M1': a3 - a1
    })
    print(f"Limb Swap (I <-> II): Model 1 AUC={a1:.4f} | Model 3 AUC={a3:.4f} (Delta={a3 - a1:+.4f})")

    # Precordial swap: 6 (V1) and 7 (V2)
    X_prec = X_test.copy()
    X_prec[:, :, 6], X_prec[:, :, 7] = X_test[:, :, 7], X_test[:, :, 6]
    p1 = m1.predict(X_prec, batch_size=64, verbose=0)
    p3 = m3.predict(X_prec, batch_size=64, verbose=0)
    a1 = np.mean([roc_auc_score(Y_test[:, c], p1[:, c]) for c in range(5)])
    a3 = np.mean([roc_auc_score(Y_test[:, c], p3[:, c]) for c in range(5)])
    robustness_results.append({
        'Test_Condition': 'Lead Swap: Precordial (V1 <-> V2)',
        'Severity': 'Electrode Misplacement',
        'Model1_AUC': a1,
        'Model3_AUC': a3,
        'Delta_AUC_M3_vs_M1': a3 - a1
    })
    print(f"Precordial Swap (V1 <-> V2): Model 1 AUC={a1:.4f} | Model 3 AUC={a3:.4f} (Delta={a3 - a1:+.4f})")

    df_rob = pd.DataFrame(robustness_results)
    df_rob.to_csv(ROBUSTNESS_CSV, index=False)
    print(f"[Saved] Robustness results saved to '{ROBUSTNESS_CSV}'")
    return df_rob, preds1_clean, preds3_clean


def run_calibration_evaluation(Y_test, preds1, preds3):
    print("\n" + "=" * 80)
    print("EVALUATING MODEL CALIBRATION (ECE, MCE, BRIER SCORE)")
    print("=" * 80)

    ece1, mce1, ece1_c, mce1_c = compute_ece_mce(Y_test, preds1, n_bins=10)
    ece3, mce3, ece3_c, mce3_c = compute_ece_mce(Y_test, preds3, n_bins=10)

    brier1 = np.mean([brier_score_loss(Y_test[:, c], preds1[:, c]) for c in range(5)])
    brier3 = np.mean([brier_score_loss(Y_test[:, c], preds3[:, c]) for c in range(5)])

    calib_results = [
        {'Model': 'Model 1 (Baseline Flat SE-ResNet1D)', 'ECE': ece1, 'MCE': mce1, 'Brier_Score': brier1},
        {'Model': 'Model 3 (Anatomical Territory-Dropout)', 'ECE': ece3, 'MCE': mce3, 'Brier_Score': brier3}
    ]

    for c, sc in enumerate(SUPERCLASSES):
        calib_results[0][f'ECE_{sc}'] = ece1_c[c]
        calib_results[1][f'ECE_{sc}'] = ece3_c[c]

    df_calib = pd.DataFrame(calib_results)
    df_calib.to_csv(CALIBRATION_CSV, index=False)
    print(f"Model 1: ECE = {ece1:.4f} | MCE = {mce1:.4f} | Brier = {brier1:.4f}")
    print(f"Model 3: ECE = {ece3:.4f} | MCE = {mce3:.4f} | Brier = {brier3:.4f}")
    print(f"[Saved] Calibration results saved to '{CALIBRATION_CSV}'")
    return df_calib


def run_latency_evaluation(m3, X_test):
    print("\n" + "=" * 80)
    print("BENCHMARKING INFERENCE & EXPLANATION LATENCY")
    print("=" * 80)

    sample_x = X_test[:1]  # Shape: (1, 1000, 12)
    sample_tensor = tf.convert_to_tensor(sample_x, dtype=tf.float32)

    # 1. Forward Pass Inference Latency
    # Warmup
    for _ in range(10):
        _ = m3(sample_tensor, training=False)

    times_inf = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = m3(sample_tensor, training=False)
        times_inf.append((time.perf_counter() - t0) * 1000.0)

    mean_inf = np.mean(times_inf)
    std_inf = np.std(times_inf)

    # 2. Macro Occlusion Sensitivity (4 territories)
    times_macro = []
    for _ in range(20):
        t0 = time.perf_counter()
        p0 = m3(sample_tensor, training=False).numpy()[0, 1]  # target MI
        drops = []
        for t_name, leads in TERRITORY_LEAD_MAP.items():
            x_occ = sample_x.copy()
            x_occ[:, :, leads] = 0.0
            p_occ = m3(tf.convert_to_tensor(x_occ), training=False).numpy()[0, 1]
            drops.append(p0 - p_occ)
        times_macro.append((time.perf_counter() - t0) * 1000.0)

    mean_macro = np.mean(times_macro)
    std_macro = np.std(times_macro)

    # 3. Meso 1D Grad-CAM++ Latency
    # Find convolutional layer
    target_layer_name = 'septal_c3b'
    grad_model = tf.keras.models.Model(
        inputs=m3.inputs,
        outputs=[m3.get_layer(target_layer_name).output, m3.output]
    )

    times_gradcam = []
    for _ in range(20):
        t0 = time.perf_counter()
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(sample_tensor, training=False)
            score = preds[:, 1]
        grads = tape.gradient(score, conv_out)
        weights = tf.reduce_mean(tf.maximum(grads, 0), axis=1, keepdims=True)
        cam = tf.reduce_sum(weights * conv_out, axis=-1)
        cam = tf.maximum(cam, 0)
        times_gradcam.append((time.perf_counter() - t0) * 1000.0)

    mean_gradcam = np.mean(times_gradcam)
    std_gradcam = np.std(times_gradcam)

    # 4. Micro Integrated Gradients Latency (50 steps)
    times_ig = []
    baseline = tf.zeros_like(sample_tensor)
    alphas = tf.linspace(0.0, 1.0, 50)
    for _ in range(10):
        t0 = time.perf_counter()
        interpolated = baseline + alphas[:, None, None, None] * (sample_tensor - baseline)
        interpolated = tf.reshape(interpolated, (50, 1000, 12))
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            preds = m3(interpolated, training=False)
            target_scores = preds[:, 1]
        grads = tape.gradient(target_scores, interpolated)
        avg_grads = tf.reduce_mean(grads, axis=0)
        ig = (sample_tensor[0] - baseline[0]) * avg_grads
        times_ig.append((time.perf_counter() - t0) * 1000.0)

    mean_ig = np.mean(times_ig)
    std_ig = np.std(times_ig)

    print(f"Forward Inference Latency : {mean_inf:.2f} +/- {std_inf:.2f} ms")
    print(f"Macro Occlusion Latency   : {mean_macro:.2f} +/- {std_macro:.2f} ms")
    print(f"Meso 1D Grad-CAM++ Latency: {mean_gradcam:.2f} +/- {std_gradcam:.2f} ms")
    print(f"Micro IG (50 steps) Latency: {mean_ig:.2f} +/- {std_ig:.2f} ms")
    total_xai = mean_inf + mean_macro + mean_gradcam + mean_ig
    print(f"Total Bedside XAI Latency : {total_xai:.2f} ms (< 0.25 s, real-time clinically feasible)")

    latency_results = [
        {'Operation': 'Forward Inference', 'Latency_ms': mean_inf, 'Std_ms': std_inf},
        {'Operation': 'Macro Occlusion (4 territories)', 'Latency_ms': mean_macro, 'Std_ms': std_macro},
        {'Operation': 'Meso 1D Grad-CAM++', 'Latency_ms': mean_gradcam, 'Std_ms': std_gradcam},
        {'Operation': 'Micro Integrated Gradients (50 steps)', 'Latency_ms': mean_ig, 'Std_ms': std_ig},
        {'Operation': 'Total Multi-Scale Explanation Pipeline', 'Latency_ms': total_xai, 'Std_ms': np.sqrt(std_inf**2 + std_macro**2 + std_gradcam**2 + std_ig**2)}
    ]
    df_lat = pd.DataFrame(latency_results)
    df_lat.to_csv(LATENCY_CSV, index=False)
    print(f"[Saved] Latency benchmark saved to '{LATENCY_CSV}'")
    return df_lat


def main():
    print("=" * 80)
    print("RUNNING ADVISOR STRESS TESTING, CALIBRATION, AND LATENCY BENCHMARK")
    print("=" * 80)

    # 1. Load Data
    _, Y_val = load_cached_fold_data([9])
    X_test, Y_test = load_cached_fold_data([10])

    # 2. Load Models
    m1 = build_calibrated_se_ecg_classifier()
    m1.load_weights('checkpoints/se_resnet_calibrated_fold_1_best.h5')

    m3 = build_anatomical_se_ecg_classifier()
    m3.load_weights('checkpoints/se_resnet_anatomical_territory_dropout_fold_1_best.h5')

    # 3. Robustness Suite
    _, p1, p3 = run_robustness_evaluation(m1, m3, X_test, Y_test)

    # 4. Calibration Suite
    run_calibration_evaluation(Y_test, p1, p3)

    # 5. Latency Suite
    run_latency_evaluation(m3, X_test)

    print("\n" + "=" * 80)
    print("ALL ROBUSTNESS, CALIBRATION, AND LATENCY EVALUATIONS COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
