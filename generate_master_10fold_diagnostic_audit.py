"""
generate_master_10fold_diagnostic_audit.py

Master 10-Fold Performance & Diagnostic Confusion Matrix Evaluator:
1. Model 1: Calibrated Model (Global SE-ResNet1D)
2. Model 2: Anatomically Guided Model (4-Branch SE-ResNet1D)
3. Model 3: Anatomical Territory-Dropout Model (4-Branch SE-ResNet1D + Territory Dropout p=0.15)

Parses pre-computed 10-fold result CSVs for exact out-of-sample metrics, parses training history CSVs for early stopping & overfitting metrics, and runs inference on the best fold of each model to produce exact diagnostic confusion matrices (TP, FP, TN, FN, Sens, Spec, PPV, NPV).
"""

import os
import gc
import numpy as np
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES
from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
from run_10fold_calibrated_classifier import build_calibrated_se_ecg_classifier

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')
CHECKPOINT_DIR = 'checkpoints'
EXPERIMENT_DIR = 'experiments'

MASTER_AUDIT_CSV = os.path.join(EXPERIMENT_DIR, 'master_10fold_diagnostic_audit_metrics.csv')
MASTER_SUMMARY_TXT = os.path.join(EXPERIMENT_DIR, 'master_10fold_diagnostic_audit_summary.txt')
CONFUSION_MATRICES_TXT = os.path.join(EXPERIMENT_DIR, 'best_fold_confusion_matrices.txt')


def get_fold_split_indices(fold_num):
    test_fold = ((fold_num + 8) % 10) + 1
    val_fold = ((fold_num + 7) % 10) + 1
    all_folds = list(range(1, 11))
    train_folds = [f for f in all_folds if f not in [val_fold, test_fold]]
    return train_folds, val_fold, test_fold


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


def calculate_binary_class_metrics(y_true, y_pred_bin):
    metrics_per_class = {}
    for c_idx, class_name in enumerate(SUPERCLASSES):
        yt = y_true[:, c_idx]
        yp = y_pred_bin[:, c_idx]

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))

        sens = tp / (tp + fn + 1e-7)
        spec = tn / (tn + fp + 1e-7)
        ppv = tp / (tp + fp + 1e-7)
        npv = tn / (tn + fn + 1e-7)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        f1 = 2 * (ppv * sens) / (ppv + sens + 1e-7)

        metrics_per_class[class_name] = {
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Sensitivity': sens, 'Specificity': spec,
            'PPV': ppv, 'NPV': npv, 'Accuracy': acc, 'F1': f1
        }
    return metrics_per_class


def analyze_history_convergence(history_prefix):
    epochs_list = []
    train_losses = []
    val_losses = []
    loss_gaps = []

    for f in range(1, 11):
        history_csv = os.path.join(CHECKPOINT_DIR, f"{history_prefix}_fold_{f}.csv")
        if not os.path.exists(history_csv):
            continue
        df = pd.read_csv(history_csv)
        total_epochs = len(df)
        best_epoch_idx = df['val_loss'].idxmin()
        best_row = df.iloc[best_epoch_idx]

        train_loss = best_row['loss']
        val_loss = best_row['val_loss']

        epochs_list.append(total_epochs)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        loss_gaps.append(val_loss - train_loss)

    return {
        'avg_epochs': float(np.mean(epochs_list)),
        'avg_train_loss': float(np.mean(train_losses)),
        'avg_val_loss': float(np.mean(val_losses)),
        'avg_loss_gap': float(np.mean(loss_gaps))
    }


def parse_csv_results(csv_path, history_prefix):
    df = pd.read_csv(csv_path)

    macro_auc = df['macro_auc'].values
    macro_f1_opt = df['macro_f1_opt'].values
    macro_f1_fixed = df['macro_f1_fixed'].values
    micro_f1_opt = df['micro_f1_opt'].values
    test_acc = df['test_acc'].values
    test_loss = df['test_loss'].values

    # Convergence stats
    conv_stats = analyze_history_convergence(history_prefix)

    # Class stats
    class_stats = {}
    for cls in SUPERCLASSES:
        auc_col = f'auc_{cls}'
        f1_col = f'f1_{cls}'
        auc_vals = df[auc_col] if auc_col in df.columns else np.zeros(len(df))
        f1_vals = df[f1_col] if f1_col in df.columns else np.zeros(len(df))
        
        class_stats[cls] = {
            'auc_mean': auc_vals.mean(), 'auc_std': auc_vals.std(),
            'auc_min': auc_vals.min(), 'auc_max': auc_vals.max(),
            'f1_mean': f1_vals.mean(), 'f1_std': f1_vals.std(),
            'f1_min': f1_vals.min(), 'f1_max': f1_vals.max(),
        }

    best_fold_idx = int(np.argmax(macro_auc)) + 1  # 1-indexed fold

    return {
        'df': df,
        'macro_auc_mean': np.mean(macro_auc), 'macro_auc_std': np.std(macro_auc), 'macro_auc_min': np.min(macro_auc), 'macro_auc_max': np.max(macro_auc),
        'macro_f1_opt_mean': np.mean(macro_f1_opt), 'macro_f1_opt_std': np.std(macro_f1_opt), 'macro_f1_opt_min': np.min(macro_f1_opt), 'macro_f1_opt_max': np.max(macro_f1_opt),
        'macro_f1_fixed_mean': np.mean(macro_f1_fixed), 'macro_f1_fixed_std': np.std(macro_f1_fixed), 'macro_f1_fixed_min': np.min(macro_f1_fixed), 'macro_f1_fixed_max': np.max(macro_f1_fixed),
        'micro_f1_opt_mean': np.mean(micro_f1_opt), 'micro_f1_opt_std': np.std(micro_f1_opt), 'micro_f1_opt_min': np.min(micro_f1_opt), 'micro_f1_opt_max': np.max(micro_f1_opt),
        'test_acc_mean': np.mean(test_acc), 'test_acc_std': np.std(test_acc), 'test_acc_min': np.min(test_acc), 'test_acc_max': np.max(test_acc),
        'test_loss_mean': np.mean(test_loss), 'test_loss_std': np.std(test_loss), 'test_loss_min': np.min(test_loss), 'test_loss_max': np.max(test_loss),
        'conv_stats': conv_stats,
        'class_stats': class_stats,
        'best_fold': best_fold_idx,
        'best_fold_macro_auc': macro_auc[best_fold_idx - 1],
        'best_fold_macro_f1': macro_f1_opt[best_fold_idx - 1],
    }


def evaluate_best_fold(model_name, build_fn, ckpt_prefix, fold_num, opt_th):
    train_folds, val_fold, test_fold = get_fold_split_indices(fold_num)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_prefix}_fold_{fold_num}_best.h5")

    X_test, Y_test = load_cached_fold_data([test_fold])

    model = build_fn(
        input_shape=(1000, 12),
        num_classes=5,
        final_activation='sigmoid',
        dense_dropout=0.3,
        spatial_dropout=0.08,
        l2_weight=2e-5
    )
    model.load_weights(ckpt_path)

    test_preds = model(X_test, training=False).numpy()

    test_bin_opt = np.zeros_like(test_preds)
    for c in range(5):
        test_bin_opt[:, c] = (test_preds[:, c] >= opt_th[c]).astype(int)

    class_metrics = calculate_binary_class_metrics(Y_test, test_bin_opt)

    del model, X_test, Y_test
    return fold_num, test_fold, opt_th, class_metrics


def main():
    print("=" * 80)
    print("MASTER 10-FOLD DIAGNOSTIC PERFORMANCE & CONFUSION MATRIX AUDIT")
    print("=" * 80)

    # 1. Model 1 Stats
    stats_m1 = parse_csv_results("experiments/10fold_classifier_results.csv", "training_history_se_calibrated")
    # 2. Model 2 Stats
    stats_m2 = parse_csv_results("experiments/10fold_anatomical_classifier_results.csv", "training_history_se_anatomical")
    # 3. Model 3 Stats
    stats_m3 = parse_csv_results("experiments/10fold_anatomical_territory_dropout_classifier_results.csv", "training_history_se_anatomical_territory_dropout")

    # Extract best fold threshold array
    def get_opt_th(stats):
        bf_idx = stats['best_fold'] - 1
        row = stats['df'].iloc[bf_idx]
        th_cols = [f'th_{cls}' if f'th_{cls}' in row else f'opt_th_{cls}' for cls in SUPERCLASSES]
        return np.array([row[col] for col in th_cols], dtype=np.float32)

    th_m1 = get_opt_th(stats_m1)
    th_m2 = get_opt_th(stats_m2)
    th_m3 = get_opt_th(stats_m3)

    # Evaluate best fold checkpoints for detailed diagnostic metrics & confusion matrices
    print(f"\n[Best Fold Evaluation] Model 1 Best Fold: {stats_m1['best_fold']}")
    fold_m1, test_m1, th_m1, cm_m1 = evaluate_best_fold("Model 1: Calibrated Model", build_calibrated_se_ecg_classifier, "se_resnet_calibrated", stats_m1['best_fold'], th_m1)

    print(f"[Best Fold Evaluation] Model 2 Best Fold: {stats_m2['best_fold']}")
    fold_m2, test_m2, th_m2, cm_m2 = evaluate_best_fold("Model 2: Anatomically Guided Model", build_anatomical_se_ecg_classifier, "se_resnet_anatomical", stats_m2['best_fold'], th_m2)

    print(f"[Best Fold Evaluation] Model 3 Best Fold: {stats_m3['best_fold']}")
    fold_m3, test_m3, th_m3, cm_m3 = evaluate_best_fold("Model 3: Anatomical Territory-Dropout Model", build_anatomical_se_ecg_classifier, "se_resnet_anatomical_territory_dropout", stats_m3['best_fold'], th_m3)

    # Build Master Summary Text
    dash_90 = "=" * 90
    dash_line = "-" * 90

    report = []
    report.append(dash_90)
    report.append("MASTER 10-FOLD DIAGNOSTIC PERFORMANCE & CONVERGENCE AUDIT REPORT")
    report.append(dash_90)

    models_data = [
        ("Model 1: Calibrated Model (Global SE-ResNet1D)", stats_m1, fold_m1, test_m1, cm_m1),
        ("Model 2: Anatomically Guided Model (4-Branch SE-ResNet1D)", stats_m2, fold_m2, test_m2, cm_m2),
        ("Model 3: Anatomical Territory-Dropout Model (4-Branch + Territory Dropout p=0.15)", stats_m3, fold_m3, test_m3, cm_m3),
    ]

    for title, st, f_num, t_num, cm_dict in models_data:
        report.append("\n" + dash_line)
        report.append(f"OVERALL 10-FOLD BENCHMARK RESULTS: {title}")
        report.append(dash_line)
        report.append(f"  Macro ROC-AUC        : {st['macro_auc_mean']:.4f} ± {st['macro_auc_std']:.4f}  [Min: {st['macro_auc_min']:.4f}, Max: {st['macro_auc_max']:.4f}]")
        report.append(f"  Macro F1 (Opt Thresh): {st['macro_f1_opt_mean']:.4f} ± {st['macro_f1_opt_std']:.4f}  [Min: {st['macro_f1_opt_min']:.4f}, Max: {st['macro_f1_opt_max']:.4f}]")
        report.append(f"  Macro F1 (Fixed 0.5) : {st['macro_f1_fixed_mean']:.4f} ± {st['macro_f1_fixed_std']:.4f}  [Min: {st['macro_f1_fixed_min']:.4f}, Max: {st['macro_f1_fixed_max']:.4f}]")
        report.append(f"  Micro F1 (Opt Thresh): {st['micro_f1_opt_mean']:.4f} ± {st['micro_f1_opt_std']:.4f}  [Min: {st['micro_f1_opt_min']:.4f}, Max: {st['micro_f1_opt_max']:.4f}]")
        report.append(f"  Binary Test Accuracy : {st['test_acc_mean']*100:.2f}% ± {st['test_acc_std']*100:.2f}%  [Min: {st['test_acc_min']*100:.2f}%, Max: {st['test_acc_max']*100:.2f}%]")
        report.append(f"  Binary Test Loss     : {st['test_loss_mean']:.4f} ± {st['test_loss_std']:.4f}  [Min: {st['test_loss_min']:.4f}, Max: {st['test_loss_max']:.4f}]")

        cst = st['conv_stats']
        report.append(f"\n  [Training Convergence & Overfitting Audit]")
        report.append(f"  Average Early Stop Epoch : {cst['avg_epochs']:.1f} Epochs")
        report.append(f"  Average Training Loss    : {cst['avg_train_loss']:.4f}")
        report.append(f"  Average Validation Loss  : {cst['avg_val_loss']:.4f}")
        report.append(f"  Average Loss Gap (Val-Tr): {cst['avg_loss_gap']:.4f} (Low gap indicates strong generalizability)")

        report.append(f"\n  [Per-Class 10-Fold ROC-AUC & F1-Score (Mean ± Std)]")
        report.append(f"  Class  | 10-Fold ROC-AUC [Min, Max]           | 10-Fold F1-Score [Min, Max]")
        report.append(f"  -------|--------------------------------------|-------------------------------------")
        for cls in SUPERCLASSES:
            cs = st['class_stats'][cls]
            report.append(f"  {cls:<6} | {cs['auc_mean']:.4f} ± {cs['auc_std']:.4f} [{cs['auc_min']:.4f}, {cs['auc_max']:.4f}] | {cs['f1_mean']:.4f} ± {cs['f1_std']:.4f} [{cs['f1_min']:.4f}, {cs['f1_max']:.4f}]")

        report.append(f"\n  [BEST FOLD SELECTION & CLINICAL DIAGNOSTIC METRICS (Fold {f_num}, Test Fold {t_num})]")
        report.append(f"  Class  | TP   | FP   | TN   | FN   | Sensitivity (Recall) | Specificity        | PPV (Precision)    | NPV")
        report.append(f"  -------|------|------|------|------|----------------------|--------------------|--------------------|--------------------")
        for cls in SUPERCLASSES:
            cm = cm_dict[cls]
            report.append(f"  {cls:<6} | {cm['TP']:<4} | {cm['FP']:<4} | {cm['TN']:<4} | {cm['FN']:<4} | {cm['Sensitivity']*100:.2f}%               | {cm['Specificity']*100:.2f}%             | {cm['PPV']*100:.2f}%               | {cm['NPV']*100:.2f}%")

    report_str = "\n".join(report)
    print("\n" + report_str)

    with open(MASTER_SUMMARY_TXT, 'w') as f:
        f.write(report_str)

    # Confusion Matrices Output Text
    cm_report = []
    cm_report.append(dash_90)
    cm_report.append("PER-CLASS DIAGNOSTIC 2x2 CONFUSION MATRICES FOR BEST FOLD OF EACH MODEL")
    cm_report.append(dash_90)

    for title, st, f_num, t_num, cm_dict in models_data:
        cm_report.append("\n" + dash_line)
        cm_report.append(f"{title} (Best Fold {f_num}, Evaluated on Test Fold {t_num})")
        cm_report.append(dash_line)
        for cls in SUPERCLASSES:
            cm = cm_dict[cls]
            cm_report.append(f"\n  --- Superclass: {cls} ---")
            cm_report.append(f"                     Predicted Negative    Predicted Positive")
            cm_report.append(f"  Actual Negative    {cm['TN']:<20} {cm['FP']:<20} (Specificity: {cm['Specificity']*100:.2f}%)")
            cm_report.append(f"  Actual Positive    {cm['FN']:<20} {cm['TP']:<20} (Sensitivity: {cm['Sensitivity']*100:.2f}%)")
            cm_report.append(f"  Precision (PPV): {cm['PPV']*100:.2f}% | NPV: {cm['NPV']*100:.2f}% | Class F1: {cm['F1']:.4f}")

    cm_str = "\n".join(cm_report)
    with open(CONFUSION_MATRICES_TXT, 'w') as f:
        f.write(cm_str)

    print(f"\n[Audit Completed Successfully]")
    print(f"  - Master Summary Text: '{MASTER_SUMMARY_TXT}'")
    print(f"  - Confusion Matrices Text: '{CONFUSION_MATRICES_TXT}'")


if __name__ == '__main__':
    main()
