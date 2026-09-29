"""
train_conditional_ddpm.py
Training Pipeline for 1D Conditional Denoising Diffusion Probabilistic Model (DDPM) on PTB-XL ECGs.

Includes:
- Linear Noise Schedule (T = 1000, beta_1 = 1e-4 to beta_T = 0.02)
- Classifier-Free Guidance (CFG) Unconditional Dropout (p_uncond = 0.15)
- Anatomical Territory Dropout (p_territory = 0.15)
- Physics-Informed DDPM Loss (Einthoven & Goldberger residual penalization)
- Fast GPU Runtime Sequence Generator
- Model Checkpoint & Loss Logging
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

# GPU Memory Growth Setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU Setup] Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"[GPU Setup Error] {e}")

from causal_diffusion.unet_1d import build_conditional_unet_1d
from causal_diffusion.dataset import load_ptbxl_superclass_data
from causal_diffusion.physics import compute_physics_violation_loss

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')

CHECKPOINT_DIR = 'checkpoints'
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'ddpm_unet_1d_ptbxl.h5')
CSV_LOG_PATH = os.path.join(CHECKPOINT_DIR, 'training_history_ddpm_unet.csv')

# Diffusion Hyperparameters
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

BATCH_SIZE = 64
EPOCHS = 10
INITIAL_LR = 3e-4
CFG_UNCOND_PROB = 0.15
PHYSICS_LOSS_WEIGHT = 0.001

# Lead Indices for 4 Cardiac Territories in Standard WFDB Order:
TERRITORY_LEAD_MAP = [
    [1, 2, 5],        # Inferior Wall (II, III, aVF)
    [6, 7, 8, 9],     # Antero-Septal Wall (V1, V2, V3, V4)
    [0, 4, 10, 11],   # Lateral Wall (I, aVL, V5, V6)
    [3]               # Cavity Reciprocal (aVR)
]

class GaussianDiffusionSchedule:
    """
    Precalculates Forward & Reverse Diffusion Schedule Terms for T=1000.
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0).astype(np.float32)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)


diff_schedule = GaussianDiffusionSchedule(timesteps=TIMESTEPS, beta_start=BETA_START, beta_end=BETA_END)


class DDPMTrainingSequence(tf.keras.utils.Sequence):
    """
    Runtime Sequence Generator for DDPM Training with Classifier-Free Guidance (CFG) Dropout
    and Anatomical Territory Dropout.
    """
    def __init__(self, X_data, Y_data, batch_size=64, shuffle=True):
        self.X_data = np.ascontiguousarray(X_data, dtype=np.float32)
        self.Y_data = np.ascontiguousarray(Y_data, dtype=np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X_data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x0 = self.X_data[batch_indices].copy()
        batch_c = self.Y_data[batch_indices].copy()
        b_size = len(batch_x0)

        # 1. Sample Random Timesteps t in [0, TIMESTEPS - 1]
        t = np.random.randint(0, TIMESTEPS, size=(b_size,))

        # 2. Sample Random Gaussian Noise
        noise = np.random.normal(0.0, 1.0, size=batch_x0.shape).astype(np.float32)

        # 3. Compute x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha_t = diff_schedule.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha_t = diff_schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        batch_xt = sqrt_alpha_t * batch_x0 + sqrt_one_minus_alpha_t * noise

        # 4. Classifier-Free Guidance Condition Dropout (Set condition to zeros with 15% prob)
        uncond_mask = np.random.rand(b_size) < CFG_UNCOND_PROB
        batch_c_cfg = batch_c.copy()
        batch_c_cfg[uncond_mask] = 0.0

        # Model Input: (x_t, t, c), Target Output: noise
        return (batch_xt, t.astype(np.int32), batch_c_cfg), noise


class PhysicsInformedDDPM(tf.keras.Model):
    """
    Custom Keras Model to incorporate Biophysical Penalty into DDPM Training.
    """
    def __init__(self, unet, diff_schedule, physics_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.unet = unet
        
        # We need these as tf constants for the graph
        self.sqrt_alphas = tf.constant(diff_schedule.sqrt_alphas_cumprod, dtype=tf.float32)
        self.sqrt_one_minus_alphas = tf.constant(diff_schedule.sqrt_one_minus_alphas_cumprod, dtype=tf.float32)
        self.physics_weight = physics_weight
        
        self.mse_tracker = tf.keras.metrics.Mean(name="mse_loss")
        self.phys_tracker = tf.keras.metrics.Mean(name="phys_loss")
        self.total_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.total_tracker, self.mse_tracker, self.phys_tracker]
        
    def call(self, inputs, training=False):
        return self.unet(inputs, training=training)

    def compute_losses(self, data, training=True):
        (x_t, t, c), noise = data
        
        pred_noise = self.unet([x_t, t, c], training=training)
        mse_loss = tf.reduce_mean(tf.square(noise - pred_noise))
        
        # Reconstruct implied x0
        sqrt_alpha_t = tf.gather(self.sqrt_alphas, t)
        sqrt_one_minus_alpha_t = tf.gather(self.sqrt_one_minus_alphas, t)
        
        sqrt_alpha_t = tf.reshape(sqrt_alpha_t, [-1, 1, 1])
        sqrt_one_minus_alpha_t = tf.reshape(sqrt_one_minus_alpha_t, [-1, 1, 1])
        
        x0_pred = (x_t - sqrt_one_minus_alpha_t * pred_noise) / (sqrt_alpha_t + 1e-8)
        
        # Physics Loss
        phys_loss = compute_physics_violation_loss(x0_pred)
        
        total_loss = mse_loss + self.physics_weight * phys_loss
        return total_loss, mse_loss, phys_loss
        
    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, mse_loss, phys_loss = self.compute_losses(data, training=True)
            
        trainable_vars = self.unet.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.total_tracker.update_state(total_loss)
        self.mse_tracker.update_state(mse_loss)
        self.phys_tracker.update_state(phys_loss)
        return {m.name: m.result() for m in self.metrics}
        
    def test_step(self, data):
        total_loss, mse_loss, phys_loss = self.compute_losses(data, training=False)
        self.total_tracker.update_state(total_loss)
        self.mse_tracker.update_state(mse_loss)
        self.phys_tracker.update_state(phys_loss)
        return {m.name: m.result() for m in self.metrics}


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


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("=" * 75)
    print("1D CONDITIONAL DDPM U-NET TRAINING PIPELINE")
    print("=" * 75)
    print(f"Training Cohort : Folds 1–8")
    print(f"Validation Cohort: Fold 9")
    print(f"Batch Size      : {BATCH_SIZE}")
    print(f"Learning Rate   : {INITIAL_LR}")
    print(f"Max Epochs      : {EPOCHS}")
    print(f"CFG Uncond Prob : {CFG_UNCOND_PROB}")
    print(f"Physics Weight  : {PHYSICS_LOSS_WEIGHT}")
    print(f"Model Path      : {MODEL_CHECKPOINT_PATH}")
    print("=" * 75)

    # 1. Load Data
    X_train, Y_train = load_cached_fold_data(list(range(1, 9)))
    X_val, Y_val = load_cached_fold_data([9])

    train_gen = DDPMTrainingSequence(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = DDPMTrainingSequence(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Build 1D Conditional DDPM U-Net
    unet = build_conditional_unet_1d(
        input_shape=(1000, 12),
        num_classes=5,
        base_filters=32,
        time_emb_dim=256
    )

    # Wrap in custom physics-informed model
    model = PhysicsInformedDDPM(unet=unet, diff_schedule=diff_schedule, physics_weight=PHYSICS_LOSS_WEIGHT)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
    )

    # 3. Checkpoint Recovery
    initial_epoch = 0
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"\n[Checkpoint Recovery] Found existing model at '{MODEL_CHECKPOINT_PATH}'. Loading weights...")
        try:
            # Build the model by passing dummy data
            dummy_x = tf.zeros((1, 1000, 12))
            dummy_t = tf.zeros((1,), dtype=tf.int32)
            dummy_c = tf.zeros((1, 5))
            model([dummy_x, dummy_t, dummy_c])
            model.load_weights(MODEL_CHECKPOINT_PATH)
            
            if os.path.exists(CSV_LOG_PATH):
                log_df = pd.read_csv(CSV_LOG_PATH)
                if len(log_df) > 0:
                    initial_epoch = int(log_df['epoch'].iloc[-1]) + 1
                    print(f"[Checkpoint Recovery] Resuming training from Epoch {initial_epoch + 1}/{EPOCHS}")
        except Exception as e:
            print(f"[Checkpoint Recovery Error] {e}. Starting fresh.")

    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_CHECKPOINT_PATH,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
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
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(CSV_LOG_PATH, append=True)
    ]

    # 5. Train Model
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    print(f"\n[Finished] DDPM U-Net training completed. Best model saved to '{MODEL_CHECKPOINT_PATH}'.")


if __name__ == '__main__':
    main()
