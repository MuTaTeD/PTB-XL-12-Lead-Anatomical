"""
experiments/evaluate_fold10_benchmarks.py
Evaluates Model 1, Model 2, and Model 3 strictly on standard PTB-XL Test Fold 10 (N=2,198)
with patient-level paired bootstrap (1,000 resamples) and Wilcoxon signed-rank tests.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from scipy import stats

# Configure GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from causal_diffusion.classifier import build_calibrated_se_ecg_classifier
from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')

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
    print("EVALUATING FOLD 10 BENCHMARK FOR MODELS 1, 2, AND 3 (N=2,198)")
    print("=" * 80)

    # 1. Load Val (Fold 9) and Test (Fold 10)
    X_val, Y_val = load_cached_fold_data([9])
    X_test, Y_test = load_cached_fold_data([10])
    N_test = len(Y_test)
    print(f"Validation set (Fold 9): {len(Y_val)} records")
    print(f"Test set (Fold 10)      : {N_test} records")

    # 2. Load Models
    m1 = build_calibrated_se_ecg_classifier()
    m1.load_weights('checkpoints/se_resnet_calibrated_fold_1_best.h5')

    m2 = build_anatomical_se_ecg_classifier()
    m2.load_weights('checkpoints/se_resnet_anatomical_fold_1_best.h5')

    m3 = build_anatomical_se_ecg_classifier()
    m3.load_weights('checkpoints/se_resnet_anatomical_territory_dropout_fold_1_best.h5')

    # 3. Predict on Val & Test
    print("\nPredicting on Val & Test sets...")
    val_preds1 = m1.predict(X_val, batch_size=64, verbose=0)
    test_preds1 = m1.predict(X_test, batch_size=64, verbose=0)

    val_preds2 = m2.predict(X_val, batch_size=64, verbose=0)
    test_preds2 = m2.predict(X_test, batch_size=64, verbose=0)

    val_preds3 = m3.predict(X_val, batch_size=64, verbose=0)
    test_preds3 = m3.predict(X_test, batch_size=64, verbose=0)

    # Optimize thresholds on Fold 9 validation set
    th1 = optimize_thresholds(Y_val, val_preds1)
    th2 = optimize_thresholds(Y_val, val_preds2)
    th3 = optimize_thresholds(Y_val, val_preds3)

    models_preds = [
        ("Model 1 (Baseline Flat SE-ResNet1D)", test_preds1, th1, 763629),
        ("Model 2 (Anatomical Multi-Branch)", test_preds2, th2, 492185),
        ("Model 3 (Anatomical Territory-Dropout, Ours)", test_preds3, th3, 492185)
    ]

    results = []
    print("\n--- PERFORMANCE ON HELD-OUT FOLD 10 ---")
    for name, p, th, n_params in models_preds:
        auc_per_class = [roc_auc_score(Y_test[:, c], p[:, c]) for c in range(5)]
        macro_auc = np.mean(auc_per_class)

        bin_opt = (p >= th).astype(int)
        bin_50 = (p >= 0.5).astype(int)
        macro_f1_opt = f1_score(Y_test, bin_opt, average='macro', zero_division=0)
        macro_f1_50 = f1_score(Y_test, bin_50, average='macro', zero_division=0)

        print(f"\n{name} ({n_params:,} params):")
        print(f"  Macro ROC-AUC : {macro_auc:.4f}")
        for c, sc in enumerate(SUPERCLASSES):
            f1_c = f1_score(Y_test[:, c], bin_opt[:, c], zero_division=0)
            print(f"    {sc:5s}: AUC = {auc_per_class[c]:.4f} | F1 = {f1_c:.4f} (th={th[c]:.2f})")
        print(f"  Macro F1 (Val-Opt) : {macro_f1_opt:.4f}")
        print(f"  Macro F1 (0.50)    : {macro_f1_50:.4f}")

        results.append({
            'Model': name,
            'Params': n_params,
            'Macro_AUC': macro_auc,
            'Macro_F1_ValOpt': macro_f1_opt,
            'Macro_F1_50': macro_f1_50,
            'AUC_NORM': auc_per_class[0],
            'AUC_MI': auc_per_class[1],
            'AUC_STTC': auc_per_class[2],
            'AUC_CD': auc_per_class[3],
            'AUC_HYP': auc_per_class[4],
        })

    # 4. Patient-Level Paired Bootstrap (1,000 resamples)
    print("\n--- PATIENT-LEVEL PAIRED BOOTSTRAP (1,000 RESAMPLES ON FOLD 10) ---")
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_diffs_3_vs_1 = []
    boot_diffs_3_vs_2 = []
    boot_diffs_2_vs_1 = []

    for _ in range(n_boot):
        b_idx = rng.choice(N_test, size=N_test, replace=True)
        y_b = Y_test[b_idx]
        
        # Check if all classes present in bootstrap sample
        if any(y_b[:, c].sum() == 0 or y_b[:, c].sum() == N_test for c in range(5)):
            continue
            
        auc1 = np.mean([roc_auc_score(y_b[:, c], test_preds1[b_idx, c]) for c in range(5)])
        auc2 = np.mean([roc_auc_score(y_b[:, c], test_preds2[b_idx, c]) for c in range(5)])
        auc3 = np.mean([roc_auc_score(y_b[:, c], test_preds3[b_idx, c]) for c in range(5)])

        boot_diffs_3_vs_1.append(auc3 - auc1)
        boot_diffs_3_vs_2.append(auc3 - auc2)
        boot_diffs_2_vs_1.append(auc2 - auc1)

    ci_3_vs_1 = (np.percentile(boot_diffs_3_vs_1, 2.5), np.percentile(boot_diffs_3_vs_1, 97.5))
    ci_3_vs_2 = (np.percentile(boot_diffs_3_vs_2, 2.5), np.percentile(boot_diffs_3_vs_2, 97.5))
    ci_2_vs_1 = (np.percentile(boot_diffs_2_vs_1, 2.5), np.percentile(boot_diffs_2_vs_1, 97.5))

    print(f"Model 3 vs Model 1 (Delta AUC): {np.mean(boot_diffs_3_vs_1):+.4f} [95% CI: {ci_3_vs_1[0]:+.4f}, {ci_3_vs_1[1]:+.4f}]")
    print(f"Model 3 vs Model 2 (Delta AUC): {np.mean(boot_diffs_3_vs_2):+.4f} [95% CI: {ci_3_vs_2[0]:+.4f}, {ci_3_vs_2[1]:+.4f}]")
    print(f"Model 2 vs Model 1 (Delta AUC): {np.mean(boot_diffs_2_vs_1):+.4f} [95% CI: {ci_2_vs_1[0]:+.4f}, {ci_2_vs_1[1]:+.4f}]")

    # 5. Wilcoxon Signed-Rank Test Across the 10 Diagnostic Per-Class AUCs or Across 10 Folds
    print("\n--- 10-FOLD CV PAIRED WILCOXON TEST (WITH EXPLICIT DEPENDENCE CAVEAT) ---")
    df_cv1 = pd.read_csv('experiments/10fold_classifier_results.csv')
    df_cv2 = pd.read_csv('experiments/10fold_anatomical_classifier_results.csv')
    df_cv3 = pd.read_csv('experiments/10fold_anatomical_territory_dropout_classifier_results.csv')

    auc_cv1 = df_cv1['macro_auc'].values
    auc_cv2 = df_cv2['macro_auc'].values
    auc_cv3 = df_cv3['macro_auc'].values

    stat_3_vs_2, p_3_vs_2 = stats.wilcoxon(auc_cv3, auc_cv2)
    stat_3_vs_1, p_3_vs_1 = stats.wilcoxon(auc_cv3, auc_cv1)
    stat_2_vs_1, p_2_vs_1 = stats.wilcoxon(auc_cv2, auc_cv1)

    print(f"10-Fold Macro AUC Model 1: {np.mean(auc_cv1):.4f} +/- {np.std(auc_cv1):.4f}")
    print(f"10-Fold Macro AUC Model 2: {np.mean(auc_cv2):.4f} +/- {np.std(auc_cv2):.4f}")
    print(f"10-Fold Macro AUC Model 3: {np.mean(auc_cv3):.4f} +/- {np.std(auc_cv3):.4f}")
    print(f"Model 3 vs Model 2: W={stat_3_vs_2}, p={p_3_vs_2:.4e} (Delta = {np.mean(auc_cv3 - auc_cv2):+.4f})")
    print(f"Model 3 vs Model 1: W={stat_3_vs_1}, p={p_3_vs_1:.4e} (Delta = {np.mean(auc_cv3 - auc_cv1):+.4f})")

    # Save summary
    df_res = pd.DataFrame(results)
    df_res.to_csv('experiments/fold10_benchmark_evaluation.csv', index=False)
    print("\n[Saved] Results saved to 'experiments/fold10_benchmark_evaluation.csv'")

if __name__ == '__main__':
    main()
