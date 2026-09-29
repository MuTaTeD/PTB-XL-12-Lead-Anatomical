"""
run_10fold_anatomical_territory_dropout_classifier.py
Full 10-Fold Cross-Validation for Model 3: Anatomical Territory-Dropout SE-ResNet1D
- Incorporates Regional Lead-Territory Masking Regularization (p=0.15)
- Uses Weighted Binary Cross Entropy Loss & BinaryAccuracy (Consistent with Model 1 & Model 2)
- Automated Resume Engine tracking 'experiments/10fold_anatomical_territory_dropout_classifier_results.csv'
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

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

RESULTS_CSV_PATH = os.path.join(EXPERIMENT_DIR, '10fold_anatomical_territory_dropout_classifier_results.csv')
SUMMARY_PLOT_PATH = os.path.join(EXPERIMENT_DIR, '10fold_anatomical_territory_dropout_metrics_summary.png')

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 12
INITIAL_LR = 3e-4
TERRITORY_DROPOUT_PROB = 0.15

# Lead Indices for 4 Cardiac Territories in Standard WFDB Order:
TERRITORY_LEAD_MAP = [
    [1, 2, 5],        # Inferior Wall (II, III, aVF)
    [6, 7, 8, 9],     # Antero-Septal Wall (V1, V2, V3, V4)
    [0, 4, 10, 11],   # Lateral Wall (I, aVL, V5, V6)
    [3]               # Cavity Reciprocal (aVR)
]


class AnatomicalTerritoryDropoutECGSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence Generator with:
    1. Gaussian Voltage Jittering
    2. Temporal Time-Masking Cutout (50-100ms)
    3. Anatomical Territory Dropout (Zeroing out 1 of 4 cardiac lead territories with p=0.15)
    """
    def __init__(self, X_data, Y_data, batch_size=64, shuffle=True, augment=False, territory_dropout_prob=0.15):
        self.X_data = np.ascontiguousarray(X_data, dtype=np.float32)
        self.Y_data = np.ascontiguousarray(Y_data, dtype=np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.territory_dropout_prob = territory_dropout_prob
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

            # 2. Temporal Time-Masking Cutout
            b_size, seq_len, num_leads = batch_x.shape
            mask_len = np.random.randint(50, 100)
            for i in range(b_size):
                if np.random.rand() > 0.3:
                    start_idx = np.random.randint(0, seq_len - mask_len)
                    batch_x[i, start_idx:start_idx + mask_len, :] = 0.0

            # 3. Anatomical Territory Dropout (Regional Lead Masking)
            for i in range(b_size):
                if np.random.rand() < self.territory_dropout_prob:
                    dropped_territory = TERRITORY_LEAD_MAP[np.random.randint(0, 4)]
                    batch_x[i, :, dropped_territory] = 0.0

        return batch_x, batch_y


def get_weighted_bce_loss(pos_weights):
    pos_weights = tf.constant(pos_weights, dtype=tf.float32)

    def weighted_bce(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        loss = - (pos_weights * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(loss)

    return weighted_bce


def load_ptbxl_cached_data_by_folds(folds_list):
    """Fast contig array loader for specified folds."""
    with np.load(CACHE_PATH) as npz:
        all_X = npz['X']
        cached_ecg_ids = npz['ecg_ids']

        filenames, targets = load_ptbxl_superclass_data(DATA_DIR, folds=folds_list)
        db_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
        fold_df = db_df[db_df['strat_fold'].isin(folds_list)]
        target_ecg_ids = fold_df['ecg_id'].values

        id_to_idx = {ecg_id: idx for idx, ecg_id in enumerate(cached_ecg_ids)}
        indices = [id_to_idx[ecg_id] for ecg_id in target_ecg_ids if ecg_id in id_to_idx]

        X_fold = np.ascontiguousarray(all_X[indices], dtype=np.float32)
        Y_fold = np.ascontiguousarray(targets, dtype=np.float32)

    return X_fold, Y_fold


def find_optimal_thresholds(y_true, y_pred_prob):
    """Grid search optimal decision thresholds for each class to maximize F1-score."""
    opt_thresholds = []
    opt_f1s = []
    for i in range(y_true.shape[1]):
        best_th = 0.5
        best_f1 = 0.0
        for th in np.linspace(0.1, 0.9, 81):
            pred_bin = (y_pred_prob[:, i] >= th).astype(int)
            score = f1_score(y_true[:, i], pred_bin, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_th = th
        opt_thresholds.append(best_th)
        opt_f1s.append(best_f1)
    return np.array(opt_thresholds), np.array(opt_f1s)


def run_10fold_cross_validation():
    print("=" * 80)
    print("STARTING FULL 10-FOLD CROSS-VALIDATION FOR ANATOMICAL TERRITORY-DROPOUT MODEL")
    print("=" * 80)
    print("Architecture        : Anatomical Territory-Dropout SE-ResNet1D (4 Regional Branches)")
    print("Cardiac Territories : Inferior [II,III,aVF], Septal [V1-V4], Lateral [I,aVL,V5,V6], Cavity [aVR]")
    print(f"Territory Dropout p : {TERRITORY_DROPOUT_PROB} (Regional Lead Masking)")
    print("Loss & Metrics      : Weighted BCE Loss + BinaryAccuracy (Consistent with Models 1 & 2)")
    print("Folds               : 10 Folds (Sliding Window Train 8, Val 1, Test 1)")
    print("=" * 80)

    completed_folds = []
    if os.path.exists(RESULTS_CSV_PATH):
        df_existing = pd.read_csv(RESULTS_CSV_PATH)
        if 'fold' in df_existing.columns:
            completed_folds = df_existing['fold'].tolist()
            print(f"[Resume Engine] Loaded {len(completed_folds)} completed fold(s) from '{RESULTS_CSV_PATH}': {completed_folds}")

    for fold_num in range(1, 11):
        if fold_num in completed_folds:
            print(f"\n[Resume Engine] Fold {fold_num}/10 already completed & logged. Skipping to next fold.")
            continue

        val_fold = (fold_num + 7) % 10 + 1
        test_fold = (fold_num + 8) % 10 + 1
        all_folds = list(range(1, 11))
        train_folds = [f for f in all_folds if f not in [val_fold, test_fold]]

        print("\n" + "=" * 80)
        print(f"STARTING ANATOMICAL TERRITORY-DROPOUT CROSS-VALIDATION FOLD {fold_num}/10")
        print(f"  Train Folds : {train_folds}")
        print(f"  Val Fold    : [{val_fold}]")
        print(f"  Test Fold   : [{test_fold}]")
        print("=" * 80)

        # 1. Load Data
        X_train, Y_train = load_ptbxl_cached_data_by_folds(train_folds)
        X_val, Y_val = load_ptbxl_cached_data_by_folds([val_fold])
        X_test, Y_test = load_ptbxl_cached_data_by_folds([test_fold])

        train_seq = AnatomicalTerritoryDropoutECGSequence(
            X_train, Y_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            augment=True,
            territory_dropout_prob=TERRITORY_DROPOUT_PROB
        )
        val_seq = AnatomicalTerritoryDropoutECGSequence(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False, augment=False)
        test_seq = AnatomicalTerritoryDropoutECGSequence(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False, augment=False)

        # 2. Build & Compile Model with Consistent Weighted Loss & BinaryAccuracy
        model = build_anatomical_se_ecg_classifier(
            input_shape=(1000, 12),
            num_classes=5,
            final_activation='sigmoid',
            dense_dropout=0.3,
            spatial_dropout=0.08,
            l2_weight=2e-5
        )

        pos_counts = np.sum(Y_train, axis=0)
        neg_counts = len(Y_train) - pos_counts
        pos_weights = neg_counts / (pos_counts + 1e-5)
        loss_fn = get_weighted_bce_loss(pos_weights)

        optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.AUC(name='auc', multi_label=True)
            ]
        )

        fold_ckpt_path = os.path.join(CHECKPOINT_DIR, f'se_resnet_anatomical_territory_dropout_fold_{fold_num}_best.h5')
        fold_csv_path = os.path.join(CHECKPOINT_DIR, f'training_history_se_anatomical_territory_dropout_fold_{fold_num}.csv')

        initial_epoch = 0
        if os.path.exists(fold_ckpt_path):
            try:
                print(f"[Checkpoint Recovery] Found existing model checkpoint for Fold {fold_num} at '{fold_ckpt_path}'.")
                model.load_weights(fold_ckpt_path)
                if os.path.exists(fold_csv_path):
                    df_h = pd.read_csv(fold_csv_path)
                    initial_epoch = len(df_h)
                    print(f"[Checkpoint Recovery] Resuming Fold {fold_num} from Epoch {initial_epoch}/{EPOCHS}")
            except Exception as ex:
                print(f"[Checkpoint Recovery Warning] Could not load checkpoint: {ex}. Training from scratch.")

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=fold_ckpt_path,
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(fold_csv_path, append=True)
        ]

        # 3. Train
        if initial_epoch < EPOCHS:
            print(f"\n[Training Fold {fold_num}] Training Anatomical Territory-Dropout Model (Epoch {initial_epoch+1} to {EPOCHS})...")
            history = model.fit(
                train_seq,
                validation_data=val_seq,
                epochs=EPOCHS,
                initial_epoch=initial_epoch,
                callbacks=callbacks,
                verbose=1
            )

        # 4. Evaluate Fold Model on Held-out Test Fold
        if os.path.exists(fold_ckpt_path):
            model.load_weights(fold_ckpt_path)

        val_preds = model.predict(val_seq, verbose=0)
        test_preds = model.predict(test_seq, verbose=0)

        opt_th, val_opt_f1s = find_optimal_thresholds(Y_val, val_preds)

        test_eval = model.evaluate(test_seq, verbose=0)
        test_loss = test_eval[0]
        test_acc = test_eval[1]

        test_auc_per_class = [roc_auc_score(Y_test[:, i], test_preds[:, i]) for i in range(5)]
        macro_auc = np.mean(test_auc_per_class)

        test_f1_fixed = f1_score(Y_test, (test_preds >= 0.5).astype(int), average='macro', zero_division=0)

        test_bin_opt = np.zeros_like(test_preds)
        for i in range(5):
            test_bin_opt[:, i] = (test_preds[:, i] >= opt_th[i]).astype(int)

        macro_f1_opt = f1_score(Y_test, test_bin_opt, average='macro', zero_division=0)
        micro_f1_opt = f1_score(Y_test, test_bin_opt, average='micro', zero_division=0)

        per_class_f1_opt = [f1_score(Y_test[:, i], test_bin_opt[:, i], zero_division=0) for i in range(5)]

        print(f"\n" + "-" * 60)
        print(f"FOLD {fold_num}/10 TEST RESULTS (Held-out Test Fold {test_fold}):")
        print(f"  Test Loss            : {test_loss:.4f}")
        print(f"  Test Binary Accuracy : {test_acc*100:.2f}%")
        print(f"  Macro ROC-AUC        : {macro_auc:.4f}")
        print(f"  Macro F1 (Fixed 0.5) : {test_f1_fixed:.4f}")
        print(f"  Macro F1 (Opt Thresh): {macro_f1_opt:.4f}")
        print(f"  Micro F1 (Opt Thresh): {micro_f1_opt:.4f}")
        print("-" * 60)
        for i, cls in enumerate(SUPERCLASSES):
            print(f"  - {cls:<5}: ROC-AUC = {test_auc_per_class[i]:.4f} | Opt Thresh = {opt_th[i]:.2f} | F1 = {per_class_f1_opt[i]:.4f}")
        print("-" * 60)

        # 5. Log Fold Result
        fold_record = {
            'fold': fold_num,
            'train_folds': str(train_folds),
            'val_fold': val_fold,
            'test_fold': test_fold,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'macro_auc': macro_auc,
            'macro_f1_fixed': test_f1_fixed,
            'macro_f1_opt': macro_f1_opt,
            'micro_f1_opt': micro_f1_opt
        }

        for i, cls in enumerate(SUPERCLASSES):
            fold_record[f'opt_th_{cls}'] = opt_th[i]
            fold_record[f'auc_{cls}'] = test_auc_per_class[i]
            fold_record[f'f1_{cls}'] = per_class_f1_opt[i]

        df_fold = pd.DataFrame([fold_record])
        if not os.path.exists(RESULTS_CSV_PATH):
            df_fold.to_csv(RESULTS_CSV_PATH, index=False)
        else:
            df_fold.to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)

        print(f"[Resume Engine] Fold {fold_num} results appended to '{RESULTS_CSV_PATH}'.")

        # Memory Cleanup
        del model, X_train, Y_train, X_val, Y_val, X_test, Y_test
        tf.keras.backend.clear_session()
        gc.collect()

    print("\n" + "=" * 80)
    print("ALL 10 FOLDS COMPLETED! COMPUTING FINAL CROSS-VALIDATION SUMMARY METRICS...")
    print("=" * 80)

    df_results = pd.read_csv(RESULTS_CSV_PATH)
    mean_auc = df_results['macro_auc'].mean()
    std_auc = df_results['macro_auc'].std()
    mean_f1_opt = df_results['macro_f1_opt'].mean()
    std_f1_opt = df_results['macro_f1_opt'].std()
    mean_acc = df_results['test_acc'].mean()
    std_acc = df_results['test_acc'].std()

    print(f"10-Fold Mean Macro ROC-AUC  : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"10-Fold Mean Macro F1-Score : {mean_f1_opt:.4f} +/- {std_f1_opt:.4f}")
    print(f"10-Fold Mean Binary Accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    folds_range = df_results['fold'].values

    axes[0].plot(folds_range, df_results['macro_auc'], 'o-', color='purple', linewidth=2, label='Macro ROC-AUC')
    axes[0].axhline(mean_auc, color='purple', linestyle='--', label=f'Mean: {mean_auc:.4f}')
    axes[0].set_title('10-Fold Macro ROC-AUC (Territory Dropout)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Fold Index')
    axes[0].set_ylabel('ROC-AUC')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    axes[1].plot(folds_range, df_results['macro_f1_opt'], 's-', color='teal', linewidth=2, label='Macro F1 (Tuned)')
    axes[1].axhline(mean_f1_opt, color='teal', linestyle='--', label=f'Mean: {mean_f1_opt:.4f}')
    axes[1].set_title('10-Fold Macro F1-Score (Territory Dropout)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Fold Index')
    axes[1].set_ylabel('F1-Score')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend()

    axes[2].plot(folds_range, df_results['test_acc'] * 100, '^--', color='darkorange', linewidth=2, label='Binary Accuracy (%)')
    axes[2].axhline(mean_acc * 100, color='darkorange', linestyle='--', label=f'Mean: {mean_acc*100:.2f}%')
    axes[2].set_title('10-Fold Binary Accuracy (Territory Dropout)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Fold Index')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].grid(True, linestyle=':', alpha=0.6)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PATH, dpi=300)
    plt.close()
    print(f"[Plot Saved] 10-Fold summary visual plot saved to '{SUMMARY_PLOT_PATH}'.")


if __name__ == '__main__':
    run_10fold_cross_validation()
