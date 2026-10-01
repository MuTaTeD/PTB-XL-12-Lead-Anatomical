"""
experiments/run_advisor_ablation_suite.py
Ablation Suite to address Reviewer / Advisor Comment 5:
Isolating whether anatomical inductive bias causes improvement.

Evaluates on standard PTB-XL split: Train (Folds 1-8), Val (Fold 9), Test (Fold 10, N=2,198).
Controls:
1. Proposed Model 3: Anatomical Lead Groups + Territory Dropout (p=0.15) + Temporal Cutout (p=0.7)
2. Control 1: Random Lead Groups (matched sizes 3, 4, 4, 1; matched parameters 492k; branch dropout p=0.15; cutout p=0.7)
3. Control 2: Ordinary Lead Dropout (flat model + random lead dropout p=0.15 + cutout p=0.7)
4. Control 3: Parameter-Matched Flat Model (~494k parameters, matching Model 3's 492k)
5. Control 4: Model 3 Ablation WITHOUT Temporal Cutout (isolating the blanking contribution)
6. Multi-Seed Stability: Model 3 evaluated with Seeds 42, 123, 456
"""

import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score

# Enable GPU Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from causal_diffusion.classifier import se_resnet_block_1d_calibrated, squeeze_excitation_1d
from causal_diffusion.anatomical_classifier import anatomical_branch
from causal_diffusion.dataset import load_ptbxl_superclass_data, SUPERCLASSES

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')
EXPERIMENT_DIR = 'experiments'
CHECKPOINT_DIR = 'checkpoints'
RESULTS_CSV = os.path.join(EXPERIMENT_DIR, 'advisor_ablation_results.csv')

os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 10
INITIAL_LR = 3e-4

# Fixed Random Lead Partition for Control 1 (matched sizes: 3, 4, 4, 1):
RANDOM_LEAD_GROUPS = [
    [0, 5, 8],        # Group A (3 leads)
    [1, 3, 7, 10],    # Group B (4 leads)
    [2, 4, 6, 11],    # Group C (4 leads)
    [9]               # Group D (1 lead)
]

# Anatomical Lead Groups for Model 3:
ANATOMICAL_LEAD_GROUPS = [
    [1, 2, 5],        # Inferior (II, III, aVF)
    [6, 7, 8, 9],     # Antero-Septal (V1-V4)
    [0, 4, 10, 11],   # Lateral (I, aVL, V5, V6)
    [3]               # Cavity Reciprocal (aVR)
]


class AblationECGSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_data,
        Y_data,
        batch_size=64,
        shuffle=True,
        augment=True,
        lead_groups=None,
        group_dropout_prob=0.0,
        ordinary_lead_dropout_prob=0.0,
        use_cutout=True,
        cutout_prob=0.7
    ):
        self.X_data = np.ascontiguousarray(X_data, dtype=np.float32)
        self.Y_data = np.ascontiguousarray(Y_data, dtype=np.float32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.lead_groups = lead_groups
        self.group_dropout_prob = group_dropout_prob
        self.ordinary_lead_dropout_prob = ordinary_lead_dropout_prob
        self.use_cutout = use_cutout
        self.cutout_prob = cutout_prob
        self.indices = np.arange(len(self.X_data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        b_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.X_data[b_idx].copy()
        batch_y = self.Y_data[b_idx]

        if self.augment:
            # 1. Subtle jitter
            batch_x += np.random.normal(0.0, 0.005, size=batch_x.shape).astype(np.float32)

            # 2. Temporal Time-Masking Cutout (Signal Blanking)
            if self.use_cutout:
                b_size, seq_len, _ = batch_x.shape
                mask_len = np.random.randint(50, 100)
                for i in range(b_size):
                    if np.random.rand() < self.cutout_prob:
                        start_idx = np.random.randint(0, seq_len - mask_len)
                        batch_x[i, start_idx:start_idx + mask_len, :] = 0.0

            # 3. Group / Territory Dropout
            if self.lead_groups is not None and self.group_dropout_prob > 0.0:
                b_size = len(batch_x)
                for i in range(b_size):
                    if np.random.rand() < self.group_dropout_prob:
                        dropped_group = self.lead_groups[np.random.randint(0, len(self.lead_groups))]
                        batch_x[i, :, dropped_group] = 0.0

            # 4. Ordinary Lead Dropout (random individual leads)
            if self.ordinary_lead_dropout_prob > 0.0:
                b_size = len(batch_x)
                for i in range(b_size):
                    if np.random.rand() < self.ordinary_lead_dropout_prob:
                        # Drop 1 to 4 random leads
                        num_drop = np.random.randint(1, 5)
                        drop_leads = np.random.choice(12, size=num_drop, replace=False)
                        batch_x[i, :, drop_leads] = 0.0

        return batch_x, batch_y


def build_grouped_multibranch_classifier(lead_groups, name="Grouped_MultiBranch"):
    """Builds a 4-branch SE-ResNet1D with specified lead groups and SE fusion."""
    inputs = tf.keras.layers.Input(shape=(1000, 12), name="ecg_input")
    branch_features = []

    for b_idx, group in enumerate(lead_groups):
        group_leads = tf.gather(inputs, group, axis=-1, name=f"gather_branch_{b_idx}")
        feat = anatomical_branch(group_leads, f"branch_{b_idx}", spatial_dropout=0.08, l2_weight=2e-5)
        branch_features.append(feat)

    stacked = tf.keras.layers.Lambda(lambda f: tf.stack(f, axis=1), name="stack_branches")(branch_features)

    # Cross-Branch SE Attention Fusion
    sq = tf.keras.layers.GlobalAveragePooling1D(name="branch_squeeze")(stacked)
    ex = tf.keras.layers.Dense(16, activation='relu', name="branch_ex1")(sq)
    ex = tf.keras.layers.Dense(len(lead_groups), activation='softmax', name="branch_attention")(ex)
    weights = tf.keras.layers.Reshape((len(lead_groups), 1), name="weights_reshape")(ex)

    weighted = tf.keras.layers.Multiply(name="apply_attention")([stacked, weights])
    flat = tf.keras.layers.Flatten(name="flatten_branches")(weighted)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(flat)
    outputs = tf.keras.layers.Dense(5, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(2e-5), name="diag_head")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_param_matched_flat_classifier():
    """Flat 12-lead SE-ResNet1D parameter-matched to ~494k parameters."""
    inputs = tf.keras.layers.Input(shape=(1000, 12), name="ecg_input")
    stem_attn = squeeze_excitation_1d(inputs, reduction_ratio=4)
    x = tf.keras.layers.Conv1D(25, kernel_size=7, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(2e-5))(stem_attn)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = se_resnet_block_1d_calibrated(x, filters=25, stride=1, l2_weight=2e-5)
    x = se_resnet_block_1d_calibrated(x, filters=51, stride=2, downsample=True, l2_weight=2e-5)
    x = se_resnet_block_1d_calibrated(x, filters=103, stride=2, downsample=True, l2_weight=2e-5)
    x = se_resnet_block_1d_calibrated(x, filters=206, stride=2, downsample=True, l2_weight=2e-5)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(103, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(2e-5))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(5, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(2e-5), name="diag_head")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ParamMatched_Flat_SE_ResNet1D")


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


def get_weighted_bce_loss(pos_weights):
    pos_weights = tf.constant(pos_weights, dtype=tf.float32)
    def weighted_bce(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss = - (pos_weights * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(loss)
    return weighted_bce


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


def train_and_evaluate_model(
    config_name,
    model_builder,
    lead_groups=None,
    group_dropout_prob=0.0,
    ordinary_lead_dropout_prob=0.0,
    use_cutout=True,
    seed=42,
    X_train=None, Y_train=None,
    X_val=None, Y_val=None,
    X_test=None, Y_test=None,
    pos_weights=None
):
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {config_name} (Seed: {seed})")
    print("=" * 80)

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = model_builder()
    params = model.count_params()
    print(f"Total Parameters: {params:,}")

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"ablation_{config_name.lower().replace(' ', '_').replace('-', '_')}_seed{seed}.h5")
    if config_name.startswith('Proposed Model 3') and os.path.exists('checkpoints/se_resnet_anatomical_territory_dropout_fold_1_best.h5'):
        ckpt_path = 'checkpoints/se_resnet_anatomical_territory_dropout_fold_1_best.h5'

    # If checkpoint exists, skip training
    if os.path.exists(ckpt_path):
        print(f"[Resume] Found existing checkpoint '{ckpt_path}'. Loading weights directly...")
        model.load_weights(ckpt_path)
    else:
        loss_fn = get_weighted_bce_loss(pos_weights)
        optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[tf.keras.metrics.AUC(multi_label=True, name='auc')]
        )

        train_gen = AblationECGSequence(
            X_train, Y_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            augment=True,
            lead_groups=lead_groups,
            group_dropout_prob=group_dropout_prob,
            ordinary_lead_dropout_prob=ordinary_lead_dropout_prob,
            use_cutout=use_cutout
        )
        val_gen = AblationECGSequence(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=False, augment=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min', restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', mode='min', save_best_only=True, verbose=0)
        ]

        t0 = time.time()
        print(f"Training up to {EPOCHS} epochs (Patience {PATIENCE})...")
        model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)
        print(f"Training completed in {(time.time() - t0)/60:.1f} minutes.")

    # Evaluate on Fold 9 (Validation) to optimize thresholds
    val_preds = model.predict(X_val, batch_size=64, verbose=0)
    opt_th = optimize_thresholds(Y_val, val_preds)

    # Evaluate on Fold 10 (Held-Out Test Set)
    test_preds = model.predict(X_test, batch_size=64, verbose=0)

    auc_per_class = [roc_auc_score(Y_test[:, c], test_preds[:, c]) for c in range(5)]
    macro_auc = np.mean(auc_per_class)

    bin_opt = (test_preds >= opt_th).astype(int)
    bin_50 = (test_preds >= 0.5).astype(int)
    macro_f1_opt = f1_score(Y_test, bin_opt, average='macro', zero_division=0)
    macro_f1_50 = f1_score(Y_test, bin_50, average='macro', zero_division=0)

    print(f"\nResults for {config_name}:")
    print(f"  Macro ROC-AUC : {macro_auc:.4f}")
    print(f"  Macro F1 (Val-Opt) : {macro_f1_opt:.4f}")
    print(f"  Macro F1 (0.50)    : {macro_f1_50:.4f}")
    for c, sc in enumerate(SUPERCLASSES):
        f1_c = f1_score(Y_test[:, c], bin_opt[:, c], zero_division=0)
        print(f"    {sc:5s}: AUC = {auc_per_class[c]:.4f} | F1 = {f1_c:.4f} (th={opt_th[c]:.2f})")

    result_dict = {
        'Config': config_name,
        'Seed': seed,
        'Params': params,
        'Macro_AUC': macro_auc,
        'Macro_F1_ValOpt': macro_f1_opt,
        'Macro_F1_50': macro_f1_50,
        'AUC_NORM': auc_per_class[0],
        'AUC_MI': auc_per_class[1],
        'AUC_STTC': auc_per_class[2],
        'AUC_CD': auc_per_class[3],
        'AUC_HYP': auc_per_class[4],
    }

    # Clean memory
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return result_dict, test_preds


def main():
    print("=" * 80)
    print("STARTING ADVISOR ABLATION SUITE (STANDARD FOLD 10 BENCHMARK)")
    print("=" * 80)

    # 1. Load Data
    train_folds = list(range(1, 9))
    val_fold = 9
    test_fold = 10

    print(f"Train Folds: {train_folds}")
    print(f"Val Fold   : [{val_fold}]")
    print(f"Test Fold  : [{test_fold}]")

    X_train, Y_train = load_cached_fold_data(train_folds)
    X_val, Y_val = load_cached_fold_data([val_fold])
    X_test, Y_test = load_cached_fold_data([test_fold])

    # Class imbalance weights
    N_pos = Y_train.sum(axis=0)
    N_neg = len(Y_train) - N_pos
    pos_weights = N_neg / (N_pos + 1e-5)

    all_results = []
    saved_preds = {}

    # Define Configurations:
    configs = [
        {
            'name': 'Proposed Model 3 (Anatomical + Dropout + Cutout)',
            'builder': lambda: build_grouped_multibranch_classifier(ANATOMICAL_LEAD_GROUPS, name="Model3"),
            'lead_groups': ANATOMICAL_LEAD_GROUPS,
            'group_dropout_prob': 0.15,
            'ordinary_lead_dropout_prob': 0.0,
            'use_cutout': True,
            'seed': 42
        },
        {
            'name': 'Control 1 (Random Lead Groups)',
            'builder': lambda: build_grouped_multibranch_classifier(RANDOM_LEAD_GROUPS, name="RandomGroups"),
            'lead_groups': RANDOM_LEAD_GROUPS,
            'group_dropout_prob': 0.15,
            'ordinary_lead_dropout_prob': 0.0,
            'use_cutout': True,
            'seed': 42
        },
        {
            'name': 'Control 2 (Ordinary Lead Dropout on Flat Model)',
            'builder': lambda: build_param_matched_flat_classifier(),
            'lead_groups': None,
            'group_dropout_prob': 0.0,
            'ordinary_lead_dropout_prob': 0.15,
            'use_cutout': True,
            'seed': 42
        },
        {
            'name': 'Control 3 (Parameter-Matched Flat Model)',
            'builder': lambda: build_param_matched_flat_classifier(),
            'lead_groups': None,
            'group_dropout_prob': 0.0,
            'ordinary_lead_dropout_prob': 0.0,
            'use_cutout': True,
            'seed': 42
        },
        {
            'name': 'Control 4 (Model 3 Without Temporal Cutout)',
            'builder': lambda: build_grouped_multibranch_classifier(ANATOMICAL_LEAD_GROUPS, name="NoCutout"),
            'lead_groups': ANATOMICAL_LEAD_GROUPS,
            'group_dropout_prob': 0.15,
            'ordinary_lead_dropout_prob': 0.0,
            'use_cutout': False,
            'seed': 42
        },
        {
            'name': 'Model 3 Seed 123',
            'builder': lambda: build_grouped_multibranch_classifier(ANATOMICAL_LEAD_GROUPS, name="Model3_s123"),
            'lead_groups': ANATOMICAL_LEAD_GROUPS,
            'group_dropout_prob': 0.15,
            'ordinary_lead_dropout_prob': 0.0,
            'use_cutout': True,
            'seed': 123
        },
        {
            'name': 'Model 3 Seed 456',
            'builder': lambda: build_grouped_multibranch_classifier(ANATOMICAL_LEAD_GROUPS, name="Model3_s456"),
            'lead_groups': ANATOMICAL_LEAD_GROUPS,
            'group_dropout_prob': 0.15,
            'ordinary_lead_dropout_prob': 0.0,
            'use_cutout': True,
            'seed': 456
        }
    ]

    for cfg in configs:
        res, preds = train_and_evaluate_model(
            config_name=cfg['name'],
            model_builder=cfg['builder'],
            lead_groups=cfg['lead_groups'],
            group_dropout_prob=cfg['group_dropout_prob'],
            ordinary_lead_dropout_prob=cfg['ordinary_lead_dropout_prob'],
            use_cutout=cfg['use_cutout'],
            seed=cfg['seed'],
            X_train=X_train, Y_train=Y_train,
            X_val=X_val, Y_val=Y_val,
            X_test=X_test, Y_test=Y_test,
            pos_weights=pos_weights
        )
        all_results.append(res)
        saved_preds[cfg['name']] = preds

        # Save after each run
        df_out = pd.DataFrame(all_results)
        df_out.to_csv(RESULTS_CSV, index=False)
        print(f"[Checkpoint Saved] Progress logged to '{RESULTS_CSV}'")

    # Save test predictions
    np.savez_compressed(
        os.path.join(EXPERIMENT_DIR, 'advisor_ablation_predictions.npz'),
        Y_test=Y_test,
        **{k.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus'): v for k, v in saved_preds.items()}
    )

    print("\n" + "=" * 80)
    print("ALL ABLATION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()
