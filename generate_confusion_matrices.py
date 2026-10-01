"""
generate_confusion_matrices.py
Computes and visualizes Per-Class Multilabel Confusion Matrices strictly on held-out Test Fold 10 (N=2,198)
for ALL THREE MODELS:
Row 1: Model 1 (Baseline Flat SE-ResNet1D)
Row 2: Model 2 (Anatomical Multi-Branch SE-ResNet1D)
Row 3: Model 3 (Anatomical Territory-Dropout SE-ResNet1D, Ours)

All matrices sum to exactly N=2,198, resolving the advisor's review comment on Figure 3.
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
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from causal_diffusion.classifier import build_calibrated_se_ecg_classifier
from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')
PLOT_PATH = 'manuscript/figures/fig2_confusion_matrices.png'
EXPERIMENT_PLOT_PATH = 'experiments/confusion_matrices_comparison.png'

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

def optimize_thresholds(y_true, y_pred_prob):
    best_th = np.full(5, 0.5)
    for c in range(5):
        best_f1 = 0.0
        for th in np.linspace(0.1, 0.9, 81):
            pred = (y_pred_prob[:, c] >= th).astype(int)
            f = f1_score(y_true[:, c], pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_th[c] = th
    return best_th

def main():
    print("=" * 80)
    print("GENERATING 3-MODEL FOLD 10 CONFUSION MATRICES (N=2,198)")
    print("=" * 80)

    # 1. Load Data
    X_val, Y_val = load_cached_fold_data([9])
    X_test, Y_test = load_cached_fold_data([10])
    N_test = len(Y_test)
    assert N_test == 2198, f"Expected 2198 test records, got {N_test}"

    # 2. Build and Load Models
    print("Loading checkpoints for Models 1, 2, and 3...")
    m1 = build_calibrated_se_ecg_classifier()
    m1.load_weights('checkpoints/se_resnet_calibrated_fold_1_best.h5')

    m2 = build_anatomical_se_ecg_classifier()
    m2.load_weights('checkpoints/se_resnet_anatomical_fold_1_best.h5')

    m3 = build_anatomical_se_ecg_classifier()
    m3.load_weights('checkpoints/se_resnet_anatomical_territory_dropout_fold_1_best.h5')

    # 3. Predict
    print("Inference on validation and test sets...")
    val_p1 = m1.predict(X_val, batch_size=64, verbose=0)
    test_p1 = m1.predict(X_test, batch_size=64, verbose=0)
    th1 = optimize_thresholds(Y_val, val_p1)

    val_p2 = m2.predict(X_val, batch_size=64, verbose=0)
    test_p2 = m2.predict(X_test, batch_size=64, verbose=0)
    th2 = optimize_thresholds(Y_val, val_p2)

    val_p3 = m3.predict(X_val, batch_size=64, verbose=0)
    test_p3 = m3.predict(X_test, batch_size=64, verbose=0)
    th3 = optimize_thresholds(Y_val, val_p3)

    # Binarize with validation-derived thresholds
    bin1 = (test_p1 >= th1).astype(int)
    bin2 = (test_p2 >= th2).astype(int)
    bin3 = (test_p3 >= th3).astype(int)

    cm1 = multilabel_confusion_matrix(Y_test, bin1)
    cm2 = multilabel_confusion_matrix(Y_test, bin2)
    cm3 = multilabel_confusion_matrix(Y_test, bin3)

    # Verify that all matrices sum to 2198
    for c in range(5):
        assert cm1[c].sum() == 2198
        assert cm2[c].sum() == 2198
        assert cm3[c].sum() == 2198

    # 4. Plot 3x5 Grid Figure
    print("Rendering publication-grade 3x5 Confusion Matrix comparison figure...")
    sns.set_theme(style='white', font_scale=1.0)
    fig, axes = plt.subplots(3, 5, figsize=(18, 11), dpi=300)

    model_data = [
        ("Model 1: Baseline Flat SE-ResNet1D (Fold 10, N=2,198)", cm1, th1, 'Blues'),
        ("Model 2: Anatomical Multi-Branch (Fold 10, N=2,198)", cm2, th2, 'Purples'),
        ("Model 3: Anatomical Territory-Dropout (Ours, Fold 10, N=2,198)", cm3, th3, 'Blues')
    ]

    for row_idx, (model_name, cm, thresholds, cmap) in enumerate(model_data):
        for col_idx, class_name in enumerate(SUPERCLASSES):
            ax = axes[row_idx, col_idx]
            matrix = cm[col_idx]
            tn, fp, fn, tp = matrix.ravel()

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            sns.heatmap(
                matrix,
                annot=True,
                fmt='d',
                cmap=cmap,
                cbar=False,
                ax=ax,
                linewidths=1.5,
                linecolor='#e2e8f0',
                annot_kws={'fontsize': 11, 'fontweight': 'bold'}
            )

            # Titles and Labels
            if row_idx == 0:
                ax.set_title(f"{class_name}\nPos: {tp+fn} | Neg: {tn+fp}", fontsize=12, fontweight='bold', pad=10)
            
            if col_idx == 0:
                ax.set_ylabel(f"{model_name}\n\nActual", fontsize=11, fontweight='bold')
            else:
                ax.set_ylabel("")

            if row_idx == 2:
                ax.set_xlabel("Predicted", fontsize=11, fontweight='bold')
            else:
                ax.set_xlabel("")

            ax.set_xticklabels(['Neg (0)', 'Pos (1)'], fontsize=10)
            ax.set_yticklabels(['Neg (0)', 'Pos (1)'], fontsize=10, rotation=0)

            # Metrics text box in each cell
            metrics_str = f"F1: {f1:.3f} | Th: {thresholds[col_idx]:.2f}\nSens: {sens:.1%} | Spec: {spec:.1%}"
            ax.text(
                0.5, -0.22, metrics_str,
                transform=ax.transAxes,
                fontsize=9,
                ha='center',
                va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.9)
            )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45, wspace=0.3)
    
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
    fig.savefig(EXPERIMENT_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[Done] Updated confusion matrices successfully saved to:\n  - {PLOT_PATH}\n  - {EXPERIMENT_PLOT_PATH}")

if __name__ == '__main__':
    main()
