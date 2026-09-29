"""
run_single_fold_experiment.py
Single-Fold Experiment Trial for 1D ResNet ECG Classifier on PTB-XL v1.0.3.

Setup:
- Training Set: Folds 1-8 (17,418 records)
- Validation Set: Fold 9 (2,183 records)
- Test Set: Fold 10 (2,198 records)
- Epochs: 100 max
- Batch Size: 64
- Optimizer: Adam (lr=1e-3)
- Callbacks: Early Stopping (patience=15), Model Checkpoint (save_best_only=True), CSVLogger
- Recovery: Auto-resume from checkpoint if interrupted
"""

import os
import sys
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

from causal_diffusion.classifier import build_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')
CHECKPOINT_DIR = 'checkpoints'
EXPERIMENT_DIR = 'experiments'
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'resnet_fold1_8_best.h5')
CSV_LOG_PATH = os.path.join(CHECKPOINT_DIR, 'training_history_fold1_8.csv')
PLOT_PATH = os.path.join(EXPERIMENT_DIR, 'fold_1_8_training_curves.png')

BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3


def load_cached_fold_data(folds):
    """
    Instantly loads dataset splits from compressed binary .npz cache.
    """
    print(f"[Fast Loader] Loading preprocessed signals from '{CACHE_PATH}' for Folds {folds}...")
    npz = np.load(CACHE_PATH)
    all_X = npz['X']
    cached_ecg_ids = npz['ecg_ids']

    # Load targets and folds mapping
    filenames, targets = load_ptbxl_superclass_data(DATA_DIR, folds=folds)

    db_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
    fold_df = db_df[db_df['strat_fold'].isin(folds if isinstance(folds, list) else [folds])]
    target_ecg_ids = fold_df['ecg_id'].values

    # Mask cached array by ecg_ids
    id_to_idx = {ecg_id: idx for idx, ecg_id in enumerate(cached_ecg_ids)}
    indices = [id_to_idx[ecg_id] for ecg_id in target_ecg_ids if ecg_id in id_to_idx]

    X = all_X[indices]
    Y = targets

    print(f"[Fast Loader] Folds {folds} loaded instantly into RAM: X shape {X.shape}, Y shape {Y.shape}")
    return X, Y


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    print("=" * 70)
    print("PTB-XL SINGLE-FOLD TRIAL EXPERIMENT (ResNet 1D Classifier)")
    print("=" * 70)
    print(f"Training Folds   : 1, 2, 3, 4, 5, 6, 7, 8")
    print(f"Validation Fold  : 9")
    print(f"Test Fold        : 10")
    print(f"Batch Size       : {BATCH_SIZE}")
    print(f"Max Epochs       : {EPOCHS}")
    print(f"Early Stopping   : Patience = {PATIENCE} epochs")
    print(f"Checkpoint File  : {MODEL_CHECKPOINT_PATH}")
    print("=" * 70)

    # 1. Load Preprocessed Data Instantly from Binary Cache
    X_train, Y_train = load_cached_fold_data(list(range(1, 9)))
    X_val, Y_val = load_cached_fold_data([9])
    X_test, Y_test = load_cached_fold_data([10])

    # 2. Build 1D ResNet Classifier
    model = build_ecg_classifier(input_shape=(1000, 12), num_classes=5, final_activation='sigmoid')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )

    # 3. Checkpoint & Resumption Logic
    initial_epoch = 0
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"\n[Checkpoint Recovery] Existing model checkpoint found at '{MODEL_CHECKPOINT_PATH}'. Loading weights...")
        try:
            model.load_weights(MODEL_CHECKPOINT_PATH)
            if os.path.exists(CSV_LOG_PATH):
                history_df = pd.read_csv(CSV_LOG_PATH)
                if len(history_df) > 0:
                    initial_epoch = int(history_df['epoch'].iloc[-1]) + 1
                    print(f"[Checkpoint Recovery] Resuming training from Epoch {initial_epoch + 1}/{EPOCHS}")
        except Exception as e:
            print(f"[Checkpoint Warning] Could not load previous weights ({e}). Starting fresh.")

    # 4. Callbacks Setup
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
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

    # 5. Execute GPU Model Training
    if initial_epoch < EPOCHS:
        print(f"\n[Training Started] Running fit from epoch {initial_epoch + 1} to {EPOCHS}...")
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print(f"\n[Training Complete] Model already completed all {EPOCHS} epochs.")

    # Load best checkpoint weights for final evaluation
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        model.load_weights(MODEL_CHECKPOINT_PATH)

    # 6. Evaluate Model on Held-Out Fold 10 Test Set
    print("\n" + "=" * 70)
    print("EVALUATION ON HELD-OUT TEST SET (FOLD 10)")
    print("=" * 70)
    test_results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=1)
    test_loss = test_results[0]
    test_acc = test_results[1]

    # Predict Probabilities
    test_preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    test_pred_binary = (test_preds >= 0.5).astype(int)

    auc_scores = []
    print("\nPer-Class ROC-AUC Scores on Fold 10:")
    print("-" * 50)
    for i, cls_name in enumerate(SUPERCLASSES):
        try:
            auc = roc_auc_score(Y_test[:, i], test_preds[:, i])
            auc_scores.append(auc)
            print(f"  - {cls_name:<6}: ROC-AUC = {auc:.4f}")
        except ValueError:
            print(f"  - {cls_name:<6}: ROC-AUC = N/A")

    macro_auc = np.mean(auc_scores)
    macro_f1 = f1_score(Y_test, test_pred_binary, average='macro')
    micro_f1 = f1_score(Y_test, test_pred_binary, average='micro')

    print("-" * 50)
    print(f"Overall Fold 10 Test Loss     : {test_loss:.4f}")
    print(f"Overall Fold 10 Binary Accuracy: {test_acc * 100:.2f}%")
    print(f"Overall Fold 10 Macro ROC-AUC  : {macro_auc:.4f}")
    print(f"Overall Fold 10 Macro F1-Score : {macro_f1:.4f}")
    print(f"Overall Fold 10 Micro F1-Score : {micro_f1:.4f}")
    print("=" * 70)

    # 7. Render Training Curves Plot
    if os.path.exists(CSV_LOG_PATH):
        log_df = pd.read_csv(CSV_LOG_PATH)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss Plot
        axes[0].plot(log_df['epoch'] + 1, log_df['loss'], label='Train Loss', color='navy', linewidth=2)
        axes[0].plot(log_df['epoch'] + 1, log_df['val_loss'], label='Val Loss', color='crimson', linewidth=2, linestyle='--')
        axes[0].set_title('Cross-Entropy Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy Plot
        axes[1].plot(log_df['epoch'] + 1, log_df['accuracy'], label='Train Accuracy', color='navy', linewidth=2)
        axes[1].plot(log_df['epoch'] + 1, log_df['val_accuracy'], label='Val Accuracy', color='crimson', linewidth=2, linestyle='--')
        axes[1].set_title('Binary Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # AUC Plot
        if 'auc' in log_df.columns:
            axes[2].plot(log_df['epoch'] + 1, log_df['auc'], label='Train AUC', color='navy', linewidth=2)
            axes[2].plot(log_df['epoch'] + 1, log_df['val_auc'], label='Val AUC', color='crimson', linewidth=2, linestyle='--')
            axes[2].set_title('Multi-Label ROC-AUC', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('ROC-AUC')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.suptitle('PTB-XL ResNet 1D Classifier Training History (Folds 1-8 Train, Fold 9 Val)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[Visualization Saved] Training history plot saved to '{PLOT_PATH}'")


if __name__ == '__main__':
    main()
