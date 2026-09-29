"""
evaluate_cpsc2018_zeroshot.py
Zero-Shot External Generalization Evaluation of Model 3
(Anatomical Territory-Dropout SE-ResNet1D) on CPSC2018 Dataset (6,877 records)
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss
)

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU Setup] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"[GPU Setup Error] {e}")

from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier

# Paths & Settings
CPSC_DIR = '/home/awais/Desktop/Computing In Cardialogy/training/cpsc_2018'
CHECKPOINT_PATH = 'checkpoints/se_resnet_anatomical_territory_dropout_fold_9_best.h5'
EXPERIMENTS_DIR = 'experiments'
OUT_CSV = os.path.join(EXPERIMENTS_DIR, 'cpsc2018_zeroshot_metrics.csv')
OUT_PREDS = os.path.join(EXPERIMENTS_DIR, 'cpsc2018_zeroshot_predictions.npz')

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
NUM_CLASSES = len(SUPERCLASSES)

# SNOMED CT Mapping to PTB-XL 5 Superclasses
SNOMED_MAP = {
    '426783006': ['NORM'],  # Normal sinus rhythm
    '59118001':  ['CD'],    # Right bundle branch block (RBBB)
    '164889003': ['CD'],    # Atrial fibrillation (AF)
    '429622005': ['STTC'],  # ST segment depression (STD)
    '270492004': ['CD'],    # First degree AV block (1AVB)
    '164884008': ['CD'],    # Ventricular premature beats (PVC)
    '284470004': ['CD'],    # Premature atrial contraction (PAC)
    '164909002': ['CD'],    # Left bundle branch block (LBBB)
    '164931005': ['STTC'],  # ST segment elevation (STE)
}


def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=500.0, order=3):
    """
    3rd order Butterworth bandpass filter applied at 500 Hz.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)


def load_and_preprocess_cpsc_record(hea_path):
    """
    Loads a CPSC2018 record, applies bandpass filter, downsamples 500Hz -> 100Hz,
    crops/pads to exactly 1000 steps (10.0s), and extracts multi-label target.
    """
    # 1. Parse .hea file for SNOMED codes
    targets = np.zeros(NUM_CLASSES, dtype=np.float32)
    with open(hea_path, 'r') as fp:
        lines = fp.readlines()

    snomed_codes = []
    for line in lines:
        if line.startswith('# Dx:'):
            codes = line.strip().split(':')[1].strip().split(',')
            snomed_codes = [c.strip() for c in codes]
            break

    for code in snomed_codes:
        if code in SNOMED_MAP:
            for sc in SNOMED_MAP[code]:
                idx = SUPERCLASSES.index(sc)
                targets[idx] = 1.0

    # 2. Load .mat file
    mat_path = hea_path.replace('.hea', '.mat')
    mat_data = sio.loadmat(mat_path)
    val = mat_data['val'].astype(np.float32) / 1000.0  # Scale gain to mV

    # Transpose to shape (T, 12) if shape is (12, T)
    if val.shape[0] == 12:
        val = val.T

    # 3. Apply 0.5-40Hz Bandpass Filter at native 500 Hz
    val_filtered = butter_bandpass_filter(val, lowcut=0.5, highcut=40.0, fs=500.0)

    # 4. Downsample 500 Hz -> 100 Hz (ratio 1:5)
    val_100hz = signal.resample_poly(val_filtered, 1, 5, axis=0).astype(np.float32)

    # 5. Fixed length windowing (1000 steps = 10s at 100 Hz)
    T = len(val_100hz)
    if T >= 1000:
        val_final = val_100hz[:1000]
    else:
        pad_len = 1000 - T
        val_final = np.pad(val_100hz, ((0, pad_len), (0, 0)), mode='edge')

    return val_final, targets


def main():
    print("=" * 80)
    print("ZERO-SHOT EXTERNAL GENERALIZATION EVALUATION: MODEL 3 ON CPSC2018")
    print("=" * 80)

    # Search all .hea files
    hea_files = sorted(glob.glob(os.path.join(CPSC_DIR, '**/*.hea'), recursive=True))
    total_records = len(hea_files)
    print(f"[Dataset] Found {total_records} CPSC2018 records in '{CPSC_DIR}'.")

    # Load Model 3 Architecture & Weights
    print(f"[Model Setup] Loading Model 3 Checkpoint: '{CHECKPOINT_PATH}'...")
    model = build_anatomical_se_ecg_classifier(input_shape=(1000, 12), num_classes=NUM_CLASSES)
    model.load_weights(CHECKPOINT_PATH)
    print("[Model Setup] Model 3 loaded successfully.")

    # Preprocess all records
    print("\n[Preprocessing] Filtering, downsampling 500Hz -> 100Hz, and extracting labels...")
    t0 = time.time()
    
    X_list = []
    Y_list = []
    
    for i, hea_file in enumerate(hea_files):
        x_sig, y_tgt = load_and_preprocess_cpsc_record(hea_file)
        X_list.append(x_sig)
        Y_list.append(y_tgt)
        
        if (i + 1) % 1000 == 0 or (i + 1) == total_records:
            elapsed = time.time() - t0
            print(f"  Processed [{i+1:5d}/{total_records:5d}] records ({elapsed:.1f}s elapsed)")

    X_data = np.array(X_list, dtype=np.float32)
    Y_data = np.array(Y_list, dtype=np.float32)
    
    print(f"\n[Dataset Shapes] X_data: {X_data.shape}, Y_data: {Y_data.shape}")
    print(f"[Class Distributions] Positive counts per superclass:")
    for idx, sc in enumerate(SUPERCLASSES):
        pos_cnt = int(Y_data[:, idx].sum())
        print(f"  - {sc:5s}: {pos_cnt:4d} positive samples ({pos_cnt/total_records*100:5.2f}%)")

    # Batch GPU Inference
    print("\n[Inference] Running GPU Batch Inference (Batch Size = 128)...")
    t_inf = time.time()
    Y_pred_probs = model.predict(X_data, batch_size=128, verbose=1)
    inf_duration = time.time() - t_inf
    print(f"[Inference Complete] Processed {total_records} records in {inf_duration:.2f}s ({inf_duration/total_records*1000:.2f} ms/sample)")

    # Save raw predictions and ground truth
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    np.savez_compressed(OUT_PREDS, Y_true=Y_data, Y_prob=Y_pred_probs, classes=SUPERCLASSES)
    print(f"[Predictions Saved] Saved raw outputs to '{OUT_PREDS}'")

    # Evaluate Evaluated Superclasses (NORM, STTC, CD)
    # Note: CPSC2018 does not contain MI or HYP explicit labels, so we evaluate on present classes + overall macro
    eval_indices = [idx for idx, sc in enumerate(SUPERCLASSES) if Y_data[:, idx].sum() > 0]
    eval_classes = [SUPERCLASSES[idx] for idx in eval_indices]

    print("\n" + "=" * 80)
    print("ZERO-SHOT BENCHMARK METRICS SUMMARY (CPSC2018)")
    print("=" * 80)

    class_metrics = []
    
    for idx in range(NUM_CLASSES):
        sc = SUPERCLASSES[idx]
        y_t = Y_data[:, idx]
        y_p = Y_pred_probs[:, idx]
        pos_count = int(y_t.sum())

        if pos_count == 0:
            print(f"Class {sc:5s}: No positive samples in CPSC2018 (N/A)")
            continue

        auc_roc = roc_auc_score(y_t, y_p)
        auc_pr = average_precision_score(y_t, y_p)
        
        # Default 0.5 threshold
        y_bin_50 = (y_p >= 0.5).astype(int)
        f1_50 = f1_score(y_t, y_bin_50, zero_division=0)
        rec_50 = recall_score(y_t, y_bin_50, zero_division=0)
        prec_50 = precision_score(y_t, y_bin_50, zero_division=0)

        # Optimal F1 threshold tuning
        best_thresh = 0.5
        best_f1 = f1_50
        for thresh in np.arange(0.1, 0.9, 0.02):
            y_b = (y_p >= thresh).astype(int)
            score = f1_score(y_t, y_b, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh

        y_bin_opt = (y_p >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_bin_opt).ravel()
        spec_opt = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rec_opt = recall_score(y_t, y_bin_opt, zero_division=0)
        acc_opt = (tp + tn) / (tp + tn + fp + fn)

        class_metrics.append({
            'Superclass': sc,
            'Positive_Count': pos_count,
            'ROC_AUC': auc_roc,
            'PR_AUC': auc_pr,
            'F1_0.5': f1_50,
            'F1_Tuned': best_f1,
            'Optimal_Thresh': best_thresh,
            'Sensitivity_Tuned': rec_opt,
            'Specificity_Tuned': spec_opt,
            'Accuracy_Tuned': acc_opt
        })

        print(f"Class: {sc:5s} | Pos: {pos_count:4d} | ROC-AUC: {auc_roc:.4f} | PR-AUC: {auc_pr:.4f} | F1 (0.5): {f1_50:.4f} | F1 (Tuned @ {best_thresh:.2f}): {best_f1:.4f} | Sens: {rec_opt:.4f} | Spec: {spec_opt:.4f}")

    df_metrics = pd.DataFrame(class_metrics)
    df_metrics.to_csv(OUT_CSV, index=False)
    print(f"\n[Metrics CSV Saved] Summary metrics saved to '{OUT_CSV}'")

    # Macro & Micro Aggregate Metrics
    macro_auc = df_metrics['ROC_AUC'].mean()
    macro_pr = df_metrics['PR_AUC'].mean()
    macro_f1_50 = df_metrics['F1_0.5'].mean()
    macro_f1_tuned = df_metrics['F1_Tuned'].mean()
    macro_sens = df_metrics['Sensitivity_Tuned'].mean()
    macro_spec = df_metrics['Specificity_Tuned'].mean()

    # Binary Cross-Entropy Loss
    bce_loss = log_loss(Y_data[:, eval_indices].ravel(), Y_pred_probs[:, eval_indices].ravel())

    # Overall Micro F1-Score across evaluated classes
    Y_true_eval = Y_data[:, eval_indices]
    Y_pred_eval_bin = (Y_pred_probs[:, eval_indices] >= 0.5).astype(int)
    micro_f1 = f1_score(Y_true_eval, Y_pred_eval_bin, average='micro')

    print("-" * 80)
    print(f"ZERO-SHOT MACRO ROC-AUC (Evaluated Classes):  🏆 {macro_auc:.4f}")
    print(f"ZERO-SHOT MACRO PR-AUC:                       🌟 {macro_pr:.4f}")
    print(f"ZERO-SHOT MACRO F1-SCORE (Threshold Tuned):  🌟 {macro_f1_tuned:.4f}")
    print(f"ZERO-SHOT MACRO F1-SCORE (Fixed 0.5):         {macro_f1_50:.4f}")
    print(f"ZERO-SHOT MACRO SENSITIVITY (Recall):         {macro_sens:.4f}")
    print(f"ZERO-SHOT MACRO SPECIFICITY:                  {macro_spec:.4f}")
    print(f"ZERO-SHOT MICRO F1-SCORE:                     {micro_f1:.4f}")
    print(f"ZERO-SHOT BINARY CROSS-ENTROPY LOSS:          {bce_loss:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
