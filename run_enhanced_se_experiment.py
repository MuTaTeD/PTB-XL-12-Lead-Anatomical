"""
run_enhanced_se_experiment.py
Enhanced SE-ResNet1D Trial Experiment with Lead Attention, Class Weighting,
Learning Rate Scheduling, Runtime Batch Generator, and Threshold Tuning.

Features:
- Architecture: 1D ResNet with Squeeze-and-Excitation Lead Attention (build_se_ecg_classifier)
- Data Yielding: Runtime Batch Generator (CachedECGSequence)
- Loss: Weighted Binary Cross-Entropy (Class-imbalance aware)
- Learning Rate Schedule: ReduceLROnPlateau (factor=0.5, patience=4)
- Early Stopping: Increased patience (30 epochs)
- Evaluation: Optimal Class Threshold Search on Val Fold 9 -> Applied to Test Fold 10
- Isolated Storage: Saved independently to avoid overwriting previous baseline trial results.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU Setup] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"[GPU Setup Error] {e}")

from causal_diffusion.classifier import build_se_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')

CHECKPOINT_DIR = 'checkpoints'
EXPERIMENT_DIR = 'experiments'
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'se_resnet_fold1_8_best.h5')
CSV_LOG_PATH = os.path.join(CHECKPOINT_DIR, 'training_history_se_fold1_8.csv')
PLOT_PATH = os.path.join(EXPERIMENT_DIR, 'fold_1_8_se_training_curves.png')

BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 30
INITIAL_LR = 1e-3


class CachedECGSequence(tf.keras.utils.Sequence):
    """
    Custom runtime Keras Sequence generator yielding batches dynamically at runtime.
    Features optional on-the-fly random Gaussian noise augmentation.
    """
    def __init__(self, X_data, Y_data, batch_size=64, shuffle=True, augment=False):
        self.X_data = X_data
        self.Y_data = Y_data
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
        batch_x = np.array(self.X_data[batch_indices], dtype=np.float32)
        batch_y = np.array(self.Y_data[batch_indices], dtype=np.float32)

        # Dynamic On-The-Fly Augmentation during Training
        if self.augment:
            noise = np.random.normal(0.0, 0.005, size=batch_x.shape).astype(np.float32)
            batch_x = batch_x + noise

        return batch_x, batch_y


def get_weighted_bce_loss(pos_weights):
    """
    Computes class-weighted Binary Cross-Entropy loss for imbalanced multi-label targets.
    """
    pos_weights = tf.constant(pos_weights, dtype=tf.float32)

    def weighted_bce(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        loss = - (pos_weights * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(loss)

    return weighted_bce


def load_cached_fold_data(folds):
    npz = np.load(CACHE_PATH)
    all_X = npz['X']
    cached_ecg_ids = npz['ecg_ids']

    filenames, targets = load_ptbxl_superclass_data(DATA_DIR, folds=folds)
    db_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
    fold_df = db_df[db_df['strat_fold'].isin(folds if isinstance(folds, list) else [folds])]
    target_ecg_ids = fold_df['ecg_id'].values

    id_to_idx = {ecg_id: idx for idx, ecg_id in enumerate(cached_ecg_ids)}
    indices = [id_to_idx[ecg_id] for ecg_id in target_ecg_ids if ecg_id in id_to_idx]

    X = all_X[indices]
    Y = targets
    return X, Y


def find_optimal_thresholds(val_preds, Y_val):
    """
    Finds optimal class decision thresholds on validation set to maximize F1-score per class.
    """
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
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print("=" * 75)
    print("ENHANCED SE-RESNET1D EXPERIMENT WITH LEAD ATTENTION & SCHEDULING")
    print("=" * 75)
    print(f"Training Folds   : 1–8")
    print(f"Validation Fold  : 9")
    print(f"Test Fold        : 10")
    print(f"Batch Size       : {BATCH_SIZE}")
    print(f"Max Epochs       : {EPOCHS}")
    print(f"Early Stopping   : Patience = {PATIENCE} epochs")
    print(f"Model File       : {MODEL_CHECKPOINT_PATH}")
    print("=" * 75)

    # 1. Load Data
    X_train, Y_train = load_cached_fold_data(list(range(1, 9)))
    X_val, Y_val = load_cached_fold_data([9])
    X_test, Y_test = load_cached_fold_data([10])

    # 2. Compute Class Pos-Weights for Loss Function
    pos_counts = np.sum(Y_train, axis=0)
    neg_counts = len(Y_train) - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-5)
    print("\nCalculated Class Positive Loss Weights (Folds 1-8):")
    for cls_name, w, p in zip(SUPERCLASSES, pos_weights, pos_counts):
        print(f"  - {cls_name:<6}: {p:>5} positive samples | Loss Weight = {w:.3f}")

    # 3. Create Custom Runtime Generators
    train_gen = CachedECGSequence(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_gen = CachedECGSequence(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False, augment=False)
    test_gen = CachedECGSequence(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False, augment=False)

    # 4. Build SE-ResNet Classifier with Lead-wise Attention
    model = build_se_ecg_classifier(input_shape=(1000, 12), num_classes=5, final_activation='sigmoid', dropout_rate=0.4)
    loss_fn = get_weighted_bce_loss(pos_weights)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )

    # 5. Checkpoint Resumption Check
    initial_epoch = 0
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"\n[Checkpoint Recovery] Found existing model at '{MODEL_CHECKPOINT_PATH}'. Loading weights...")
        try:
            model.load_weights(MODEL_CHECKPOINT_PATH)
            if os.path.exists(CSV_LOG_PATH):
                log_df = pd.read_csv(CSV_LOG_PATH)
                if len(log_df) > 0:
                    initial_epoch = int(log_df['epoch'].iloc[-1]) + 1
                    print(f"[Checkpoint Recovery] Resuming SE-ResNet training from Epoch {initial_epoch + 1}/{EPOCHS}")
        except Exception as e:
            print(f"[Checkpoint Recovery Error] {e}. Starting fresh.")

    # 6. Callbacks Setup with Learning Rate Scheduler
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
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
            filename=CSV_LOG_PATH,
            separator=',',
            append=True
        )
    ]

    # 7. Execute Training via Runtime Generator
    if initial_epoch < EPOCHS:
        print(f"\n[Training Started] Training SE-ResNet1D via Runtime Batch Generator (Epoch {initial_epoch + 1} to {EPOCHS})...")
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print(f"\n[Training Complete] Already completed all {EPOCHS} epochs.")

    # Load Best Model Weights
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        model.load_weights(MODEL_CHECKPOINT_PATH)

    # 8. Optimal Threshold Tuning on Validation Fold 9
    print("\n" + "=" * 75)
    print("OPTIMAL DECISION THRESHOLD TUNING (ON VALIDATION FOLD 9)")
    print("=" * 75)
    val_preds = model.predict(val_gen, verbose=0)
    optimal_thresholds = find_optimal_thresholds(val_preds, Y_val)
    print("Optimal Class Decision Thresholds (Val Fold 9):")
    for cls_name, th in zip(SUPERCLASSES, optimal_thresholds):
        print(f"  - {cls_name:<6}: Optimal Threshold = {th:.2f}")

    # 9. Final Evaluation on Held-Out Test Set (Fold 10)
    print("\n" + "=" * 75)
    print("FINAL EVALUATION ON HELD-OUT TEST SET (FOLD 10)")
    print("=" * 75)
    test_results = model.evaluate(test_gen, verbose=1)
    test_loss = test_results[0]
    test_acc = test_results[1]

    test_preds = model.predict(test_gen, verbose=0)

    # Comparison: Fixed 0.5 Threshold vs Optimal Thresholds
    test_preds_fixed = (test_preds >= 0.5).astype(int)
    test_preds_opt = np.zeros_like(test_preds)
    for c in range(5):
        test_preds_opt[:, c] = (test_preds[:, c] >= optimal_thresholds[c]).astype(int)

    auc_scores = []
    print("\nPer-Class ROC-AUC & F1 Scores on Fold 10:")
    print("-" * 65)
    print(f"{'Class':<8} {'ROC-AUC':<10} {'Fixed F1 (0.5)':<16} {'Optimized F1':<15}")
    print("-" * 65)
    for i, cls_name in enumerate(SUPERCLASSES):
        auc = roc_auc_score(Y_test[:, i], test_preds[:, i])
        auc_scores.append(auc)
        f1_fix = f1_score(Y_test[:, i], test_preds_fixed[:, i], zero_division=0)
        f1_opt = f1_score(Y_test[:, i], test_preds_opt[:, i], zero_division=0)
        print(f"{cls_name:<8} {auc:<10.4f} {f1_fix:<16.4f} {f1_opt:<15.4f}")

    macro_auc = np.mean(auc_scores)
    macro_f1_fixed = f1_score(Y_test, test_preds_fixed, average='macro')
    macro_f1_opt = f1_score(Y_test, test_preds_opt, average='macro')
    micro_f1_opt = f1_score(Y_test, test_preds_opt, average='micro')

    print("-" * 65)
    print(f"Overall Fold 10 Test Loss          : {test_loss:.4f}")
    print(f"Overall Fold 10 Binary Accuracy     : {test_acc * 100:.2f}%")
    print(f"Overall Fold 10 Macro ROC-AUC       : {macro_auc:.4f}")
    print(f"Overall Fold 10 Macro F1 (Fixed)    : {macro_f1_fixed:.4f}")
    print(f"Overall Fold 10 Macro F1 (Optimized): {macro_f1_opt:.4f}")
    print(f"Overall Fold 10 Micro F1 (Optimized): {micro_f1_opt:.4f}")
    print("=" * 75)

    # 10. Render & Save Training Curves Plot
    if os.path.exists(CSV_LOG_PATH):
        log_df = pd.read_csv(CSV_LOG_PATH)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(log_df['epoch'] + 1, log_df['loss'], label='Train Loss', color='navy', linewidth=2)
        axes[0].plot(log_df['epoch'] + 1, log_df['val_loss'], label='Val Loss', color='crimson', linewidth=2, linestyle='--')
        axes[0].set_title('Weighted Cross-Entropy Loss', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(log_df['epoch'] + 1, log_df['accuracy'], label='Train Accuracy', color='navy', linewidth=2)
        axes[1].plot(log_df['epoch'] + 1, log_df['val_accuracy'], label='Val Accuracy', color='crimson', linewidth=2, linestyle='--')
        axes[1].set_title('Binary Accuracy', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # AUC
        if 'auc' in log_df.columns:
            axes[2].plot(log_df['epoch'] + 1, log_df['auc'], label='Train AUC', color='navy', linewidth=2)
            axes[2].plot(log_df['epoch'] + 1, log_df['val_auc'], label='Val AUC', color='crimson', linewidth=2, linestyle='--')
            axes[2].set_title('Multi-Label ROC-AUC', fontsize=13, fontweight='bold')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('ROC-AUC')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.suptitle('Enhanced SE-ResNet1D Training History (Lead Attention + LR Scheduling)', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[Visualization Saved] Enhanced training curves saved to '{PLOT_PATH}'")


if __name__ == '__main__':
    main()
