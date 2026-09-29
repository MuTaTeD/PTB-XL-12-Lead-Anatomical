"""
generate_confusion_matrices.py
Computes and visualizes Per-Class & Multilabel Confusion Matrices on held-out test sets for:
1. Best Calibrated SE-ResNet Fold (Fold 8: Train 8-5, Val 6, Test 7)
2. Anatomical Multi-Branch SE-ResNet Model (Fold 1: Train 1-8, Val 9, Test 10)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, f1_score

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        pass

from causal_diffusion.classifier import build_calibrated_se_ecg_classifier
from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')
EXPERIMENT_DIR = 'experiments'
PLOT_PATH = os.path.join(EXPERIMENT_DIR, 'confusion_matrices_comparison.png')


class BatchECGSequence(tf.keras.utils.Sequence):
    def __init__(self, X_data, Y_data, batch_size=64):
        self.X_data = np.ascontiguousarray(X_data, dtype=np.float32)
        self.Y_data = np.ascontiguousarray(Y_data, dtype=np.float32)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.X_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.Y_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


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


def find_optimal_thresholds(val_preds, Y_val):
    best_thresholds = np.full(5, 0.5)
    for c in range(5):
        best_f1 = 0.0
        for th in np.linspace(0.1, 0.9, 81):
            pred_bin = (val_preds[:, c] >= th).astype(int)
            f1 = f1_score(Y_val[:, c], pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = th
    return best_thresholds


def main():
    print("=" * 80)
    print("EVALUATING CONFUSION MATRICES: CALIBRATED VS ANATOMICAL MULTI-BRANCH CLASSIFIERS")
    print("=" * 80)

    # 1. EVALUATE BEST CALIBRATED MODEL (Fold 8: Test Fold 7)
    print("\n[1/2] Evaluating Best Calibrated Model (Fold 8, Test Fold 7)...")
    calib_val_fold = 6
    calib_test_fold = 7

    X_val_c, Y_val_c = load_cached_fold_data([calib_val_fold])
    X_test_c, Y_test_c = load_cached_fold_data([calib_test_fold])

    val_gen_c = BatchECGSequence(X_val_c, Y_val_c, batch_size=64)
    test_gen_c = BatchECGSequence(X_test_c, Y_test_c, batch_size=64)

    model_calib = build_calibrated_se_ecg_classifier(input_shape=(1000, 12), num_classes=5)
    model_calib.load_weights('checkpoints/se_resnet_calibrated_fold_8_best.h5')

    val_preds_c = model_calib.predict(val_gen_c, verbose=0)
    th_calib = find_optimal_thresholds(val_preds_c, Y_val_c)

    test_preds_c_prob = model_calib.predict(test_gen_c, verbose=0)
    test_preds_c_bin = np.zeros_like(test_preds_c_prob)
    for c in range(5):
        test_preds_c_bin[:, c] = (test_preds_c_prob[:, c] >= th_calib[c]).astype(int)

    mcm_calib = multilabel_confusion_matrix(Y_test_c, test_preds_c_bin)

    tf.keras.backend.clear_session()
    del model_calib, X_val_c, Y_val_c, X_test_c

    # 2. EVALUATE ANATOMICAL MODEL (Fold 1: Test Fold 10)
    print("\n[2/2] Evaluating Anatomical Multi-Branch Model (Fold 1, Test Fold 10)...")
    anat_val_fold = 9
    anat_test_fold = 10

    X_val_a, Y_val_a = load_cached_fold_data([anat_val_fold])
    X_test_a, Y_test_a = load_cached_fold_data([anat_test_fold])

    val_gen_a = BatchECGSequence(X_val_a, Y_val_a, batch_size=64)
    test_gen_a = BatchECGSequence(X_test_a, Y_test_a, batch_size=64)

    model_anat = build_anatomical_se_ecg_classifier(input_shape=(1000, 12), num_classes=5)
    model_anat.load_weights('checkpoints/se_resnet_anatomical_fold1_8_best.h5')

    val_preds_a = model_anat.predict(val_gen_a, verbose=0)
    th_anat = find_optimal_thresholds(val_preds_a, Y_val_a)

    test_preds_a_prob = model_anat.predict(test_gen_a, verbose=0)
    test_preds_a_bin = np.zeros_like(test_preds_a_prob)
    for c in range(5):
        test_preds_a_bin[:, c] = (test_preds_a_prob[:, c] >= th_anat[c]).astype(int)

    mcm_anat = multilabel_confusion_matrix(Y_test_a, test_preds_a_bin)

    tf.keras.backend.clear_session()
    del model_anat, X_val_a, Y_val_a, X_test_a

    # PRINT PER-CLASS CONFUSION MATRICES (2x2 FOR EACH CLASS)
    print("\n" + "=" * 80)
    print("PER-CLASS 2x2 CONFUSION MATRICES (TP, FP, TN, FN Breakdown)")
    print("=" * 80)

    for i, cls in enumerate(SUPERCLASSES):
        tn_c, fp_c, fn_c, tp_c = mcm_calib[i].ravel()
        sens_c = tp_c / (tp_c + fn_c + 1e-5)
        spec_c = tn_c / (tn_c + fp_c + 1e-5)
        ppv_c = tp_c / (tp_c + fp_c + 1e-5)
        f1_c = f1_score(Y_test_c[:, i], test_preds_c_bin[:, i])

        tn_a, fp_a, fn_a, tp_a = mcm_anat[i].ravel()
        sens_a = tp_a / (tp_a + fn_a + 1e-5)
        spec_a = tn_a / (tn_a + fp_a + 1e-5)
        ppv_a = tp_a / (tp_a + fp_a + 1e-5)
        f1_a = f1_score(Y_test_a[:, i], test_preds_a_bin[:, i])

        print(f"\n--- CLASS: {cls} ---")
        print(f"  [Best Calibrated SE-ResNet Model (Fold 8)]:")
        print(f"    [[TN={tn_c:<4}, FP={fp_c:<4}], [FN={fn_c:<4}, TP={tp_c:<4}]]")
        print(f"    Sensitivity (Recall) = {sens_c*100:.2f}% | Specificity = {spec_c*100:.2f}% | Precision = {ppv_c*100:.2f}% | F1 = {f1_c:.4f}")
        print(f"  [Anatomical Multi-Branch SE-ResNet Model]:")
        print(f"    [[TN={tn_a:<4}, FP={fp_a:<4}], [FN={fn_a:<4}, TP={tp_a:<4}]]")
        print(f"    Sensitivity (Recall) = {sens_a*100:.2f}% | Specificity = {spec_a*100:.2f}% | Precision = {ppv_a*100:.2f}% | F1 = {f1_a:.4f}")

    # PLOT VISUAL COMPARISON CHART
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))

    for i, cls in enumerate(SUPERCLASSES):
        # Calibrated
        sns.heatmap(mcm_calib[i], annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred Neg', 'Pred Pos'], yticklabels=['True Neg', 'True Pos'], ax=axes[0, i])
        axes[0, i].set_title(f'Calibrated SE-ResNet (Fold 8)\nClass: {cls}', fontweight='bold', fontsize=11)

        # Anatomical
        sns.heatmap(mcm_anat[i], annot=True, fmt='d', cmap='Greens', cbar=False,
                    xticklabels=['Pred Neg', 'Pred Pos'], yticklabels=['True Neg', 'True Pos'], ax=axes[1, i])
        axes[1, i].set_title(f'Anatomical Multi-Branch\nClass: {cls}', fontweight='bold', fontsize=11)

    plt.suptitle('Per-Class Confusion Matrices Comparison: Calibrated SE-ResNet vs Anatomical Multi-Branch',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 80)
    print(f"Visual Confusion Matrix plot saved to '{PLOT_PATH}'")
    print("=" * 80)


if __name__ == '__main__':
    main()
