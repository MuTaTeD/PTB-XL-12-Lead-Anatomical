#!/usr/bin/env python3
"""
run_adebayo_sanity_check.py
Executes the Adebayo et al. (NeurIPS 2018) Model Parameter Randomization Test:
"Sanity Checks for Saliency Maps" on the trained SE-ResNet1D model.

Tests whether the multi-scale explainability stack (Grad-CAM++ and Integrated Gradients)
is sensitive to model parameters by performing:
1. Cascading layer randomization (classification head, attention layers)
2. Complete network weight randomization (independent Glorot re-initialization)

Computes Pearson correlation (r) and Spearman rank correlation (rho) between
trained attribution maps and randomized attribution maps across test samples.
"""

import os
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr
import pandas as pd

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
from causal_diffusion.dataset import load_ptbxl_superclass_data
from evaluate_anatomical_xai import AnatomicalXAIExplainer

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')
CHECKPOINT_PATH = "checkpoints/se_resnet_anatomical_territory_dropout_fold_9_best.h5"
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = 'checkpoints/se_resnet_anatomical_territory_dropout_fold1_8_best.h5'

def main():
    print("="*70)
    print("ADEBAYO ET AL. (2018) MODEL PARAMETER RANDOMIZATION SANITY CHECK")
    print("="*70)

    # 1. Load Data
    with np.load(CACHE_PATH) as npz:
        all_X = npz['X']
        cached_ecg_ids = npz['ecg_ids']
    filenames, targets = load_ptbxl_superclass_data(DATA_DIR, folds=[10])
    db_df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
    fold10_df = db_df[db_df['strat_fold'] == 10].copy()
    target_ecg_ids = fold10_df['ecg_id'].values

    id_to_idx = {ecg_id: idx for idx, ecg_id in enumerate(cached_ecg_ids)}
    indices = [id_to_idx[ecg_id] for ecg_id in target_ecg_ids if ecg_id in id_to_idx]

    X_test = all_X[indices]
    Y_test = targets

    # Select 100 representative test cases (20 per class)
    sample_indices = []
    for c in range(5):
        c_pos = np.where(Y_test[:, c] == 1)[0]
        sample_indices.extend(c_pos[:20])
    sample_indices = np.array(sample_indices)
    print(f"Selected {len(sample_indices)} test cases across 5 superclasses for sanity check.")

    # 2. Build Trained Explainer
    print("\n[1/3] Initializing Trained Explainer...")
    trained_explainer = AnatomicalXAIExplainer(CHECKPOINT_PATH)

    # 3. Build Top-Layer Randomized Explainer
    print("\n[2/3] Initializing Top-Layer Randomized Explainer...")
    top_rand_explainer = AnatomicalXAIExplainer(CHECKPOINT_PATH)
    # Reinitialize final dense layer
    dense_layer = top_rand_explainer.model.get_layer('classification_head')
    w_shape = dense_layer.get_weights()[0].shape
    b_shape = dense_layer.get_weights()[1].shape
    dense_layer.set_weights([
        np.random.normal(0, 0.05, size=w_shape).astype(np.float32),
        np.zeros(shape=b_shape).astype(np.float32)
    ])

    # 4. Build Cascading Randomized Explainer (Head + Attention)
    print("\n[3/3] Initializing Cascading Randomized Explainer...")
    casc_rand_explainer = AnatomicalXAIExplainer(CHECKPOINT_PATH)
    for l_name in ['classification_head', 'territory_softmax_attention', 'territory_ex1']:
        l = casc_rand_explainer.model.get_layer(l_name)
        weights = l.get_weights()
        if len(weights) > 0:
            new_weights = [np.random.normal(0, 0.05, size=w.shape).astype(np.float32) for w in weights]
            l.set_weights(new_weights)

    # 5. Build Completely Randomized Explainer
    print("\n[4/4] Initializing Completely Randomized Model...")
    fresh_model = build_anatomical_se_ecg_classifier()
    branch_layers = ['inferior_c3b', 'septal_c3b', 'lateral_c3b', 'cavity_c3b']
    fresh_grad_model = tf.keras.Model(
        inputs=fresh_model.inputs,
        outputs=[fresh_model.get_layer(l).output for l in branch_layers] + [
            fresh_model.get_layer('territory_softmax_attention').output,
            fresh_model.output
        ]
    )
    full_rand_explainer = AnatomicalXAIExplainer.__new__(AnatomicalXAIExplainer)
    full_rand_explainer.model = fresh_model
    full_rand_explainer.grad_model = fresh_grad_model

    # 6. Run sanity checks across cases
    print("\nExecuting Sanity Check across cases...")
    ig_pearson_top, ig_spearman_top = [], []
    ig_pearson_casc, ig_spearman_casc = [], []
    ig_pearson_full, ig_spearman_full = [], []

    cam_pearson_top, cam_spearman_top = [], []
    cam_pearson_casc, cam_spearman_casc = [], []
    cam_pearson_full, cam_spearman_full = [], []

    for i, idx in enumerate(sample_indices):
        x = X_test[idx] # (1000, 12)
        probs, _ = trained_explainer.predict(x[np.newaxis, ...], batch_size=1)
        target_cls = int(np.argmax(probs[0]))

        # Trained attributions
        trained_branch_cam, _ = trained_explainer.explain_gradcam_1d(x, target_cls)
        trained_cam_flat = np.concatenate([trained_branch_cam[t] for t in ['Inferior', 'Antero-Septal', 'Lateral', 'Cavity']])
        trained_ig, _ = trained_explainer.explain_integrated_gradients(x, target_cls, m_steps=25)
        trained_ig_flat = trained_ig.flatten()

        # Top-layer randomized
        top_branch_cam, _ = top_rand_explainer.explain_gradcam_1d(x, target_cls)
        top_cam_flat = np.concatenate([top_branch_cam[t] for t in ['Inferior', 'Antero-Septal', 'Lateral', 'Cavity']])
        top_ig, _ = top_rand_explainer.explain_integrated_gradients(x, target_cls, m_steps=25)
        top_ig_flat = top_ig.flatten()

        cam_pearson_top.append(pearsonr(trained_cam_flat, top_cam_flat)[0])
        cam_spearman_top.append(spearmanr(trained_cam_flat, top_cam_flat)[0])
        ig_pearson_top.append(pearsonr(trained_ig_flat, top_ig_flat)[0])
        ig_spearman_top.append(spearmanr(trained_ig_flat, top_ig_flat)[0])

        # Cascading randomized
        casc_branch_cam, _ = casc_rand_explainer.explain_gradcam_1d(x, target_cls)
        casc_cam_flat = np.concatenate([casc_branch_cam[t] for t in ['Inferior', 'Antero-Septal', 'Lateral', 'Cavity']])
        casc_ig, _ = casc_rand_explainer.explain_integrated_gradients(x, target_cls, m_steps=25)
        casc_ig_flat = casc_ig.flatten()

        cam_pearson_casc.append(pearsonr(trained_cam_flat, casc_cam_flat)[0])
        cam_spearman_casc.append(spearmanr(trained_cam_flat, casc_cam_flat)[0])
        ig_pearson_casc.append(pearsonr(trained_ig_flat, casc_ig_flat)[0])
        ig_spearman_casc.append(spearmanr(trained_ig_flat, casc_ig_flat)[0])

        # Completely randomized
        full_branch_cam, _ = full_rand_explainer.explain_gradcam_1d(x, target_cls)
        full_cam_flat = np.concatenate([full_branch_cam[t] for t in ['Inferior', 'Antero-Septal', 'Lateral', 'Cavity']])
        full_ig, _ = full_rand_explainer.explain_integrated_gradients(x, target_cls, m_steps=25)
        full_ig_flat = full_ig.flatten()

        cam_pearson_full.append(pearsonr(trained_cam_flat, full_cam_flat)[0])
        cam_spearman_full.append(spearmanr(trained_cam_flat, full_cam_flat)[0])
        ig_pearson_full.append(pearsonr(trained_ig_flat, full_ig_flat)[0])
        ig_spearman_full.append(spearmanr(trained_ig_flat, full_ig_flat)[0])

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(sample_indices)} test cases...")

    def summarize(arr):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        return np.mean(arr), np.std(arr)

    print("\n" + "="*70)
    print("ADEBAYO MODEL RANDOMIZATION SANITY CHECK RESULTS")
    print("="*70)
    print(f"{'Condition':<35} | {'Pearson r':<16} | {'Spearman rho':<16}")
    print("-" * 70)

    m, s = summarize(cam_pearson_top); mr, sr = summarize(cam_spearman_top)
    print(f"{'Grad-CAM++ (Top Layer Random)':<35} | {m:+.4f} ± {s:.4f}    | {mr:+.4f} ± {sr:.4f}")
    m, s = summarize(cam_pearson_casc); mr, sr = summarize(cam_spearman_casc)
    print(f"{'Grad-CAM++ (Cascading Random)':<35} | {m:+.4f} ± {s:.4f}    | {mr:+.4f} ± {sr:.4f}")
    m, s = summarize(cam_pearson_full); mr, sr = summarize(cam_spearman_full)
    print(f"{'Grad-CAM++ (Full Network Random)':<35} | {m:+.4f} ± {s:.4f}    | {mr:+.4f} ± {sr:.4f}")
    print("-" * 70)
    m, s = summarize(ig_pearson_top); mr, sr = summarize(ig_spearman_top)
    print(f"{'Integrated Gradients (Top Layer)':<35} | {m:+.4f} ± {s:.4f}    | {mr:+.4f} ± {sr:.4f}")
    m, s = summarize(ig_pearson_casc); mr, sr = summarize(ig_spearman_casc)
    print(f"{'Integrated Gradients (Cascading)':<35} | {m:+.4f} ± {s:.4f}    | {mr:+.4f} ± {sr:.4f}")
    m, s = summarize(ig_pearson_full); mr, sr = summarize(ig_spearman_full)
    print(f"{'Integrated Gradients (Full Rand)':<35} | {m:+.4f} ± {s:.4f}    | {mr:+.4f} ± {sr:.4f}")
    print("=" * 70)

    # Save to CSV
    os.makedirs('experiments', exist_ok=True)
    df_out = pd.DataFrame({
        'Method': ['Grad-CAM++', 'Grad-CAM++', 'Grad-CAM++', 'Integrated Gradients', 'Integrated Gradients', 'Integrated Gradients'],
        'Randomization': ['Top-Layer', 'Cascading', 'Full-Network', 'Top-Layer', 'Cascading', 'Full-Network'],
        'Pearson_r_mean': [
            summarize(cam_pearson_top)[0], summarize(cam_pearson_casc)[0], summarize(cam_pearson_full)[0],
            summarize(ig_pearson_top)[0], summarize(ig_pearson_casc)[0], summarize(ig_pearson_full)[0]
        ],
        'Pearson_r_std': [
            summarize(cam_pearson_top)[1], summarize(cam_pearson_casc)[1], summarize(cam_pearson_full)[1],
            summarize(ig_pearson_top)[1], summarize(ig_pearson_casc)[1], summarize(ig_pearson_full)[1]
        ],
        'Spearman_rho_mean': [
            summarize(cam_spearman_top)[0], summarize(cam_spearman_casc)[0], summarize(cam_spearman_full)[0],
            summarize(ig_spearman_top)[0], summarize(ig_spearman_casc)[0], summarize(ig_spearman_full)[0]
        ],
        'Spearman_rho_std': [
            summarize(cam_spearman_top)[1], summarize(cam_spearman_casc)[1], summarize(cam_spearman_full)[1],
            summarize(ig_spearman_top)[1], summarize(ig_spearman_casc)[1], summarize(ig_spearman_full)[1]
        ]
    })
    df_out.to_csv('experiments/adebayo_sanity_check_results.csv', index=False)
    print("Results saved to experiments/adebayo_sanity_check_results.csv")

if __name__ == '__main__':
    main()
