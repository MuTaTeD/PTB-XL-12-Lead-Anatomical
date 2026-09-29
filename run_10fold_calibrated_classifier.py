"""
run_10fold_calibrated_classifier.py
Sliding Window 10-Fold Cross-Validation for Calibrated SE-ResNet1D Classifier.

Fold Allocation Scheme (Sliding Window):
- Fold 1 : Train [1,2,3,4,5,6,7,8], Val [9], Test [10]
- Fold 2 : Train [2,3,4,5,6,7,8,9], Val [10], Test [1]
- Fold 3 : Train [3,4,5,6,7,8,9,10], Val [1], Test [2]
- Fold 4 : Train [4,5,6,7,8,9,10,1], Val [2], Test [3]
- Fold 5 : Train [5,6,7,8,9,10,1,2], Val [3], Test [4]
- Fold 6 : Train [6,7,8,9,10,1,2,3], Val [4], Test [5]
- Fold 7 : Train [7,8,9,10,1,2,3,4], Val [5], Test [6]
- Fold 8 : Train [8,9,10,1,2,3,4,5], Val [6], Test [7]
- Fold 9 : Train [9,10,1,2,3,4,5,6], Val [7], Test [8]
- Fold 10: Train [10,1,2,3,4,5,6,7], Val [8], Test [9]

Memory Optimized: Explicit GC, clear_session between folds, float32 pre-casting.
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

# Enable GPU Memory Growth & Strict VRAM Management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU Setup] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"[GPU Setup Error] {e}")

from causal_diffusion.classifier import build_calibrated_se_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')

CHECKPOINT_DIR = 'checkpoints'
EXPERIMENT_DIR = 'experiments'
SUMMARY_CSV_PATH = os.path.join(EXPERIMENT_DIR, '10fold_classifier_results.csv')
SUMMARY_PLOT_PATH = os.path.join(EXPERIMENT_DIR, '10fold_classifier_metrics_summary.png')

BATCH_SIZE = 64  # Reduced batch size to eliminate OOM risks on limited RAM/VRAM
EPOCHS = 100
PATIENCE = 12
INITIAL_LR = 3e-4


class CalibratedECGSequence(tf.keras.utils.Sequence):
    """
    Memory-Efficient Keras Sequence Generator with Time-Masking Cutout & Noise Jittering.
    """
    def __init__(self, X_data, Y_data, batch_size=64, shuffle=True, augment=False):
        # Ensure float32 contiguous arrays to avoid per-batch allocation overhead
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
            # 1. Noise Jittering
            noise = np.random.normal(0.0, 0.005, size=batch_x.shape).astype(np.float32)
            batch_x += noise

            # 2. Time-Masking Cutout (50 to 100 samples)
            b_size, seq_len, num_leads = batch_x.shape
            mask_len = np.random.randint(50, 100)
            for i in range(b_size):
                if np.random.rand() > 0.3:  # 70% chance
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
    print("\n" + "=" * 75)
    print(f"STARTING SLIDING WINDOW CROSS-VALIDATION FOLD {fold_num}/10")
    print(f"  Train Folds : {train_folds}")
    print(f"  Val Fold   : [{val_fold}]")
    print(f"  Test Fold  : [{test_fold}]")
    print("=" * 75)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f'se_resnet_calibrated_fold_{fold_num}_best.h5')
    csv_log_path = os.path.join(CHECKPOINT_DIR, f'training_history_se_calibrated_fold_{fold_num}.csv')

    # Load Data
    X_train, Y_train = load_cached_fold_data(train_folds)
    X_val, Y_val = load_cached_fold_data([val_fold])
    X_test, Y_test = load_cached_fold_data([test_fold])

    pos_counts = np.sum(Y_train, axis=0)
    neg_counts = len(Y_train) - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-5)

    train_gen = CalibratedECGSequence(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_gen = CalibratedECGSequence(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    test_gen = CalibratedECGSequence(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False, augment=False)

    model = build_calibrated_se_ecg_classifier(
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
        print(f"[Checkpoint Recovery] Found existing model at '{ckpt_path}'. Loading...")
        try:
            model.load_weights(ckpt_path)
            if os.path.exists(csv_log_path):
                log_df = pd.read_csv(csv_log_path)
                if len(log_df) > 0:
                    initial_epoch = int(log_df['epoch'].iloc[-1]) + 1
                    print(f"[Checkpoint Recovery] Resuming from Epoch {initial_epoch + 1}/{EPOCHS}")
        except Exception as e:
            print(f"[Checkpoint Recovery Error] {e}. Training from scratch.")

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

    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    # Threshold Search on Val Fold
    val_preds = model.predict(val_gen, verbose=0)
    optimal_thresholds = find_optimal_thresholds(val_preds, Y_val)

    # Evaluation on Test Fold
    test_results = model.evaluate(test_gen, verbose=0)
    test_loss = test_results[0]
    test_acc = test_results[1]

    test_preds = model.predict(test_gen, verbose=0)
    test_preds_fixed = (test_preds >= 0.5).astype(int)
    test_preds_opt = np.zeros_like(test_preds)
    for c in range(5):
        test_preds_opt[:, c] = (test_preds[:, c] >= optimal_thresholds[c]).astype(int)

    auc_scores = [roc_auc_score(Y_test[:, i], test_preds[:, i]) for i in range(5)]
    macro_auc = np.mean(auc_scores)
    macro_f1_fixed = f1_score(Y_test, test_preds_fixed, average='macro')
    macro_f1_opt = f1_score(Y_test, test_preds_opt, average='macro')
    micro_f1_opt = f1_score(Y_test, test_preds_opt, average='micro')

    print(f"\n---> Fold {fold_num} Results (Test Fold {test_fold}):")
    print(f"     Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}% | Macro AUC: {macro_auc:.4f} | Macro F1 (Opt): {macro_f1_opt:.4f}")

    fold_result = {
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
        'auc_NORM': auc_scores[0],
        'auc_MI': auc_scores[1],
        'auc_STTC': auc_scores[2],
        'auc_CD': auc_scores[3],
        'auc_HYP': auc_scores[4],
        'th_NORM': optimal_thresholds[0],
        'th_MI': optimal_thresholds[1],
        'th_STTC': optimal_thresholds[2],
        'th_CD': optimal_thresholds[3],
        'th_HYP': optimal_thresholds[4]
    }

    # Strict Memory Cleanup
    del X_train, Y_train, X_val, Y_val, X_test, Y_test
    del train_gen, val_gen, test_gen, model
    tf.keras.backend.clear_session()
    gc.collect()

    return fold_result


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print("=" * 75)
    print("SLIDING WINDOW 10-FOLD CROSS-VALIDATION FOR CALIBRATED SE-RESNET1D (MEMORY OPTIMIZED)")
    print("=" * 75)

    results = []
    # Check if existing CSV results exist to resume
    if os.path.exists(SUMMARY_CSV_PATH):
        try:
            prev_df = pd.read_csv(SUMMARY_CSV_PATH)
            results = prev_df.to_dict('records')
            print(f"[Resume] Found {len(results)} previously completed folds in '{SUMMARY_CSV_PATH}'.")
        except Exception as e:
            print(f"[Resume Error] {e}")

    completed_folds = {r['fold'] for r in results}

    for fold_num in range(1, 11):
        if fold_num in completed_folds:
            print(f"\n[Skip] Fold {fold_num} already completed. Moving to next fold...")
            continue

        res = train_and_eval_fold(fold_num)
        results.append(res)

        # Save progress after each fold
        res_df = pd.DataFrame(results)
        res_df.to_csv(SUMMARY_CSV_PATH, index=False)

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 75)
    print("10-FOLD CROSS-VALIDATION SUMMARY RESULTS")
    print("=" * 75)
    print(res_df[['fold', 'test_fold', 'macro_auc', 'macro_f1_fixed', 'macro_f1_opt', 'auc_HYP']].to_string(index=False))

    mean_auc = res_df['macro_auc'].mean()
    std_auc = res_df['macro_auc'].std()
    mean_f1_opt = res_df['macro_f1_opt'].mean()
    std_f1_opt = res_df['macro_f1_opt'].std()

    print("-" * 75)
    print(f"Mean Macro ROC-AUC across 10 Folds : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Mean Macro F1 (Opt) across 10 Folds: {mean_f1_opt:.4f} ± {std_f1_opt:.4f}")
    print("=" * 75)

    # Plot 10-Fold Summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    folds = res_df['fold'].values

    axes[0].bar(folds - 0.2, res_df['macro_auc'], width=0.4, label='Macro ROC-AUC', color='navy')
    axes[0].axhline(mean_auc, color='crimson', linestyle='--', label=f'Mean AUC ({mean_auc:.4f})')
    axes[0].set_xticks(folds)
    axes[0].set_xlabel('Cross-Validation Fold')
    axes[0].set_ylabel('ROC-AUC')
    axes[0].set_title('10-Fold Cross-Validation Macro ROC-AUC', fontweight='bold')
    axes[0].set_ylim(0.85, 0.95)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(folds - 0.2, res_df['macro_f1_opt'], width=0.4, label='Macro F1 (Optimized)', color='teal')
    axes[1].axhline(mean_f1_opt, color='crimson', linestyle='--', label=f'Mean F1 ({mean_f1_opt:.4f})')
    axes[1].set_xticks(folds)
    axes[1].set_xlabel('Cross-Validation Fold')
    axes[1].set_ylabel('Macro F1 Score')
    axes[1].set_title('10-Fold Cross-Validation Macro F1 Score', fontweight='bold')
    axes[1].set_ylim(0.65, 0.78)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Summary Chart Saved] 10-Fold Cross-Validation summary plot saved to '{SUMMARY_PLOT_PATH}'")


if __name__ == '__main__':
    main()
