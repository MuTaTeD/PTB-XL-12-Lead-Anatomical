"""
run_10fold_anatomical_classifier.py
Complete 10-Fold Cross-Validation for Model 2: Anatomical Multi-Branch SE-ResNet1D Classifier

Sliding Window Split Protocol (1-based fold indexing):
  Fold 1 : train=[1..8], val=[9], test=[10]
  Fold 2 : train=[2..9], val=[10], test=[1]
  Fold 3 : train=[3..10], val=[1], test=[2]
  ...
  Fold 10: train=[10,1..7], val=[8], test=[9]
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

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
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')

CHECKPOINT_DIR = 'checkpoints'
EXPERIMENT_DIR = 'experiments'
SUMMARY_CSV_PATH = os.path.join(EXPERIMENT_DIR, '10fold_anatomical_classifier_results.csv')
SUMMARY_PLOT_PATH = os.path.join(EXPERIMENT_DIR, '10fold_anatomical_classifier_metrics_summary.png')

BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 12
INITIAL_LR = 3e-4


class AnatomicalECGSequence(tf.keras.utils.Sequence):
    """
    Memory-Efficient Keras Sequence Generator for Anatomical Multi-Branch SE-ResNet.
    """
    def __init__(self, X_data, Y_data, batch_size=64, shuffle=True, augment=False):
        self.X_data = np.ascontiguousarray(X_data, dtype=np.float32)
        self.Y_data = np.ascontiguousarray(Y_data, dtype=np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.X_data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.X_data[batch_indices].copy()
        batch_y = self.Y_data[batch_indices]

        if self.augment:
            noise = np.random.normal(0.0, 0.005, size=batch_x.shape).astype(np.float32)
            batch_x += noise

            b_size, seq_len, num_leads = batch_x.shape
            mask_len = np.random.randint(50, 100)
            for i in range(b_size):
                if np.random.rand() > 0.3:
                    start_idx = np.random.randint(0, seq_len - mask_len)
                    batch_x[i, start_idx:start_idx + mask_len, :] = 0.0

        return batch_x, batch_y


def get_weighted_bce_loss(pos_weights):
    pos_weights = tf.constant(pos_weights, dtype=tf.float32)

    def weighted_bce(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        loss = - (pos_weights * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(loss)

    return weighted_bce


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


def get_fold_split_indices(fold_num):
    """
    Returns (train_folds, val_fold, test_fold) for 1-based fold_num (1 to 10).
    Fold 1 : train=[1..8], val=9, test=10
    Fold 2 : train=[2..9], val=10, test=1
    Fold 3 : train=[3..10], val=1, test=2
    ...
    Fold 10: train=[10,1..7], val=8, test=9
    """
    test_fold = ((fold_num + 8) % 10) + 1
    val_fold = ((fold_num + 7) % 10) + 1

    train_folds = []
    for offset in range(8):
        f = ((fold_num - 1 + offset) % 10) + 1
        train_folds.append(f)

    return train_folds, val_fold, test_fold


def train_and_eval_fold(fold_num):
    train_folds, val_fold, test_fold = get_fold_split_indices(fold_num)
    print("\n" + "=" * 80)
    print(f"STARTING ANATOMICAL MULTI-BRANCH CROSS-VALIDATION FOLD {fold_num}/10")
    print(f"  Train Folds : {train_folds}")
    print(f"  Val Fold   : [{val_fold}]")
    print(f"  Test Fold  : [{test_fold}]")
    print("=" * 80)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f'se_resnet_anatomical_fold_{fold_num}_best.h5')
    csv_log_path = os.path.join(CHECKPOINT_DIR, f'training_history_se_anatomical_fold_{fold_num}.csv')

    # Load Data
    X_train, Y_train = load_cached_fold_data(train_folds)
    X_val, Y_val = load_cached_fold_data([val_fold])
    X_test, Y_test = load_cached_fold_data([test_fold])

    pos_counts = np.sum(Y_train, axis=0)
    neg_counts = len(Y_train) - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-5)

    train_gen = AnatomicalECGSequence(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_gen = AnatomicalECGSequence(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    test_gen = AnatomicalECGSequence(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False, augment=False)

    model = build_anatomical_se_ecg_classifier(
        input_shape=(1000, 12),
        num_classes=5,
        final_activation='sigmoid',
        dense_dropout=0.3,
        spatial_dropout=0.08,
        l2_weight=2e-5
    )
    loss_fn = get_weighted_bce_loss(pos_weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )

    initial_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"\n[Checkpoint Recovery] Found existing model checkpoint for Fold {fold_num} at '{ckpt_path}'.")
        try:
            model.load_weights(ckpt_path)
            if os.path.exists(csv_log_path):
                log_df = pd.read_csv(csv_log_path)
                if len(log_df) > 0:
                    initial_epoch = int(log_df['epoch'].iloc[-1]) + 1
                    print(f"[Checkpoint Recovery] Resuming Fold {fold_num} from Epoch {initial_epoch + 1}/{EPOCHS}")
        except Exception as e:
            print(f"[Checkpoint Recovery Error] {e}. Training Fold {fold_num} fresh.")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=csv_log_path,
            separator=',',
            append=True
        )
    ]

    if initial_epoch < EPOCHS:
        print(f"\n[Training Fold {fold_num}] Training Anatomical Model (Epoch {initial_epoch + 1} to {EPOCHS})...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            workers=1,
            use_multiprocessing=False,
            verbose=1
        )

    # Load Best Model for Evaluation
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    # Optimal Decision Threshold Search on Validation Fold
    val_preds = model.predict(val_gen, verbose=0)
    optimal_thresholds = find_optimal_thresholds(val_preds, Y_val)

    # Test Set Evaluation
    test_results = model.evaluate(test_gen, verbose=0)
    test_loss = test_results[0]
    test_acc = test_results[1]

    test_preds = model.predict(test_gen, verbose=0)
    test_preds_fixed = (test_preds >= 0.5).astype(int)

    test_preds_opt = np.zeros_like(test_preds)
    for c in range(5):
        test_preds_opt[:, c] = (test_preds[:, c] >= optimal_thresholds[c]).astype(int)

    auc_per_class = {}
    f1_per_class_opt = {}
    for i, cls_name in enumerate(SUPERCLASSES):
        auc_per_class[cls_name] = float(roc_auc_score(Y_test[:, i], test_preds[:, i]))
        f1_per_class_opt[cls_name] = float(f1_score(Y_test[:, i], test_preds_opt[:, i], zero_division=0))

    macro_auc = float(np.mean(list(auc_per_class.values())))
    macro_f1_fixed = float(f1_score(Y_test, test_preds_fixed, average='macro'))
    macro_f1_opt = float(f1_score(Y_test, test_preds_opt, average='macro'))
    micro_f1_opt = float(f1_score(Y_test, test_preds_opt, average='micro'))

    print(f"\nFOLD {fold_num} COMPLETED SUMMARY:")
    print(f"  Test Loss          : {test_loss:.4f}")
    print(f"  Binary Accuracy     : {test_acc * 100:.2f}%")
    print(f"  Macro ROC-AUC       : {macro_auc:.4f}")
    print(f"  Macro F1 (Fixed)    : {macro_f1_fixed:.4f}")
    print(f"  Macro F1 (Tuned)    : {macro_f1_opt:.4f}")
    print(f"  Micro F1 (Tuned)    : {micro_f1_opt:.4f}")

    fold_metrics = {
        'fold': fold_num,
        'train_folds': str(train_folds),
        'val_fold': val_fold,
        'test_fold': test_fold,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'macro_auc': macro_auc,
        'macro_f1_fixed': macro_f1_fixed,
        'macro_f1_opt': macro_f1_opt,
        'micro_f1_opt': micro_f1_opt,
        'opt_th_NORM': optimal_thresholds[0],
        'opt_th_MI': optimal_thresholds[1],
        'opt_th_STTC': optimal_thresholds[2],
        'opt_th_CD': optimal_thresholds[3],
        'opt_th_HYP': optimal_thresholds[4],
        'auc_NORM': auc_per_class['NORM'],
        'auc_MI': auc_per_class['MI'],
        'auc_STTC': auc_per_class['STTC'],
        'auc_CD': auc_per_class['CD'],
        'auc_HYP': auc_per_class['HYP'],
        'f1_NORM': f1_per_class_opt['NORM'],
        'f1_MI': f1_per_class_opt['MI'],
        'f1_STTC': f1_per_class_opt['STTC'],
        'f1_CD': f1_per_class_opt['CD'],
        'f1_HYP': f1_per_class_opt['HYP']
    }

    # Clean Memory
    tf.keras.backend.clear_session()
    del model, X_train, Y_train, X_val, Y_val, X_test, Y_test, train_gen, val_gen, test_gen
    gc.collect()

    return fold_metrics


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print("=" * 80)
    print("STARTING FULL 10-FOLD CROSS-VALIDATION FOR ANATOMICAL MULTI-BRANCH MODEL")
    print("=" * 80)
    print("Architecture        : Anatomical Multi-Branch SE-ResNet1D (4 Regional Branches)")
    print("Cardiac Territories : Inferior [II,III,aVF], Septal [V1-V4], Lateral [I,aVL,V5,V6], Cavity [aVR]")
    print("Folds               : 10 Folds (Sliding Window Train 8, Val 1, Test 1)")
    print("Results Directory   : " + EXPERIMENT_DIR)
    print("=" * 80)

    results = []
    completed_folds = set()
    if os.path.exists(SUMMARY_CSV_PATH):
        try:
            existing_df = pd.read_csv(SUMMARY_CSV_PATH)
            if len(existing_df) > 0:
                results = existing_df.to_dict('records')
                completed_folds = set(existing_df['fold'].astype(int).values)
                print(f"[Resume Engine] Loaded {len(completed_folds)} completed fold(s) from '{SUMMARY_CSV_PATH}': {sorted(list(completed_folds))}")
        except Exception as e:
            print(f"[Resume Engine Error] {e}")

    for fold in range(1, 11):
        if fold in completed_folds:
            print(f"\n[Resume Engine] Fold {fold}/10 already completed & logged. Skipping to next fold.")
            continue

        fold_res = train_and_eval_fold(fold)
        results.append(fold_res)

        # Save Intermediary Results CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv(SUMMARY_CSV_PATH, index=False)
        print(f"\n[Progress Saved] Intermediary 10-fold results updated in '{SUMMARY_CSV_PATH}'")

    # Generate Final 10-Fold Summary
    print("\n" + "=" * 80)
    print("FINAL 10-FOLD CROSS-VALIDATION RESULTS FOR ANATOMICAL MULTI-BRANCH MODEL")
    print("=" * 80)
    df_results = pd.read_csv(SUMMARY_CSV_PATH)

    macro_auc_mean = df_results['macro_auc'].mean()
    macro_auc_std = df_results['macro_auc'].std()
    macro_f1_mean = df_results['macro_f1_opt'].mean()
    macro_f1_std = df_results['macro_f1_opt'].std()
    acc_mean = df_results['test_acc'].mean()
    acc_std = df_results['test_acc'].std()
    loss_mean = df_results['test_loss'].mean()
    loss_std = df_results['test_loss'].std()

    print(f"Overall 10-Fold Test Loss          : {loss_mean:.4f} +/- {loss_std:.4f}")
    print(f"Overall 10-Fold Binary Accuracy     : {acc_mean * 100:.2f}% +/- {acc_std * 100:.2f}%")
    print(f"Overall 10-Fold Macro ROC-AUC       : {macro_auc_mean:.4f} +/- {macro_auc_std:.4f}")
    print(f"Overall 10-Fold Macro F1 (Tuned)    : {macro_f1_mean:.4f} +/- {macro_f1_std:.4f}")
    print("-" * 80)
    print("Per-Class 10-Fold Average ROC-AUC & F1-Scores:")
    for cls in SUPERCLASSES:
        auc_m = df_results[f'auc_{cls}'].mean()
        auc_s = df_results[f'auc_{cls}'].std()
        f1_m = df_results[f'f1_{cls}'].mean()
        f1_s = df_results[f'f1_{cls}'].std()
        print(f"  - {cls:<5}: ROC-AUC = {auc_m:.4f} +/- {auc_s:.4f} | F1 = {f1_m:.4f} +/- {f1_s:.4f}")
    print("=" * 80)

    # Plot 10-Fold Metrics
    plt.figure(figsize=(12, 6))
    folds_arr = df_results['fold'].values

    plt.plot(folds_arr, df_results['macro_auc'], marker='o', color='navy', linewidth=2.5, label=f'Macro ROC-AUC (Mean: {macro_auc_mean:.4f})')
    plt.plot(folds_arr, df_results['macro_f1_opt'], marker='s', color='darkgreen', linewidth=2.5, label=f'Macro F1 Tuned (Mean: {macro_f1_mean:.4f})')
    plt.plot(folds_arr, df_results['test_acc'], marker='^', color='crimson', linewidth=2, linestyle='--', label=f'Binary Acc (Mean: {acc_mean:.4f})')

    plt.axhline(macro_auc_mean, color='navy', linestyle=':', alpha=0.6)
    plt.axhline(macro_f1_mean, color='darkgreen', linestyle=':', alpha=0.6)

    plt.title('10-Fold Cross-Validation Performance (Anatomical Multi-Branch SE-ResNet1D)', fontsize=14, fontweight='bold')
    plt.xlabel('Cross-Validation Fold Number', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.xticks(folds_arr)
    plt.ylim(0.70, 0.96)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PATH, dpi=300)
    plt.close()
    print(f"\n[Visualization Saved] 10-fold summary plot saved to '{SUMMARY_PLOT_PATH}'")


if __name__ == '__main__':
    main()
