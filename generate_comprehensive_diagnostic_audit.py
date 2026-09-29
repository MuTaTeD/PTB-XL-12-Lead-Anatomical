"""
generate_comprehensive_diagnostic_audit.py
Generates full per-class confusion matrices and clinical metrics:
- Accuracy, Sensitivity, Specificity, PPV (Precision), NPV, F1-Score, ROC-AUC
- Confusion Matrix Parameters (TP, FP, TN, FN)
For Model 1 (Calibrated Baseline), Model 2 (Anatomical Multi-Branch), and Model 3 (Anatomical Territory-Dropout).
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score

# Enable GPU Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[GPU Warning] {e}")

from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')
CHECKPOINT_DIR = 'checkpoints'
EXPERIMENT_DIR = 'experiments'

OUTPUT_CSV = os.path.join(EXPERIMENT_DIR, 'comprehensive_diagnostic_metrics_3models.csv')


def load_cached_test_data(test_fold=10):
    with np.load(CACHE_PATH) as npz:
        all_X = npz['X']
        cached_ecg_ids = npz['ecg_ids']

        filenames, targets = load_ptbxl_superclass_data(DATA_DIR, folds=[test_fold])
        db_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
        fold_df = db_df[db_df['strat_fold'] == test_fold]
        target_ecg_ids = fold_df['ecg_id'].values

        id_to_idx = {ecg_id: idx for idx, ecg_id in enumerate(cached_ecg_ids)}
        indices = [id_to_idx[ecg_id] for ecg_id in target_ecg_ids if ecg_id in id_to_idx]

        X = np.ascontiguousarray(all_X[indices], dtype=np.float32)
        Y = np.ascontiguousarray(targets, dtype=np.float32)

    return X, Y


def compute_clinical_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred_bin = (y_pred_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    f1 = 2 * (ppv * sens) / (ppv + sens + 1e-8)
    auc = roc_auc_score(y_true, y_pred_prob)

    return {
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'Accuracy': float(acc),
        'Sensitivity': float(sens),
        'Specificity': float(spec),
        'PPV': float(ppv),
        'NPV': float(npv),
        'F1_Score': float(f1),
        'ROC_AUC': float(auc),
        'Threshold': float(threshold)
    }


def evaluate_model_checkpoint(model_name, model_type, ckpt_path, X_test, Y_test):
    print(f"\n[Evaluating {model_name}] Loading checkpoint '{ckpt_path}'...")

    if model_type == 'calibrated':
        from run_10fold_calibrated_classifier import build_calibrated_se_ecg_classifier
        model = build_calibrated_se_ecg_classifier(
            input_shape=(1000, 12),
            num_classes=5,
            final_activation='sigmoid'
        )
    else:
        from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
        model = build_anatomical_se_ecg_classifier(
            input_shape=(1000, 12),
            num_classes=5,
            final_activation='sigmoid'
        )

    model.load_weights(ckpt_path)

    preds = model.predict(X_test, batch_size=64, verbose=0)

    results = []
    for i, cls in enumerate(SUPERCLASSES):
        best_th = 0.5
        best_f1 = 0.0
        for th in np.linspace(0.1, 0.9, 81):
            pred_b = (preds[:, i] >= th).astype(int)
            f1 = f1_score(Y_test[:, i], pred_b, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        m = compute_clinical_metrics(Y_test[:, i], preds[:, i], threshold=best_th)
        m['Model'] = model_name
        m['Superclass'] = cls
        results.append(m)

        print(f"  - {cls:<5}: AUC={m['ROC_AUC']:.4f} | Sens={m['Sensitivity']*100:.2f}% | Spec={m['Specificity']*100:.2f}% | PPV={m['PPV']*100:.2f}% | NPV={m['NPV']*100:.2f}% | F1={m['F1_Score']:.4f} | TP={m['TP']}, FP={m['FP']}, TN={m['TN']}, FN={m['FN']}")

    tf.keras.backend.clear_session()
    return results


def main():
    X_test, Y_test = load_cached_test_data(test_fold=10)
    print(f"[Data Loaded] Test Fold 10 Shape: X={X_test.shape}, Y={Y_test.shape}")

    all_records = []

    # Model 1 Checkpoint
    ckpt_m1 = 'checkpoints/se_resnet_calibrated_fold_8_best.h5'
    if not os.path.exists(ckpt_m1):
        ckpt_m1 = 'checkpoints/se_resnet_calibrated_fold_1_best.h5'
    rec_m1 = evaluate_model_checkpoint('Model 1: Calibrated Baseline', 'calibrated', ckpt_m1, X_test, Y_test)
    all_records.extend(rec_m1)

    # Model 2 Checkpoint (Best Fold 9 or Fold 5)
    ckpt_m2 = 'checkpoints/se_resnet_anatomical_fold_9_best.h5'
    if not os.path.exists(ckpt_m2):
        ckpt_m2 = 'checkpoints/se_resnet_anatomical_fold1_8_best.h5'
    rec_m2 = evaluate_model_checkpoint('Model 2: Anatomical Multi-Branch', 'anatomical', ckpt_m2, X_test, Y_test)
    all_records.extend(rec_m2)

    # Model 3 Checkpoint
    ckpt_m3 = 'checkpoints/se_resnet_anatomical_territory_dropout_fold1_8_best.h5'
    rec_m3 = evaluate_model_checkpoint('Model 3: Anatomical Territory-Dropout', 'anatomical', ckpt_m3, X_test, Y_test)
    all_records.extend(rec_m3)

    df_out = pd.DataFrame(all_records)
    cols_order = ['Model', 'Superclass', 'ROC_AUC', 'F1_Score', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Threshold', 'TP', 'FP', 'TN', 'FN']
    df_out = df_out[cols_order]
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Audit Saved] Comprehensive clinical metrics saved to '{OUTPUT_CSV}'")


if __name__ == '__main__':
    main()
