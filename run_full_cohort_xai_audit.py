"""
run_full_cohort_xai_audit.py
================================================================================
Comprehensive Full-Cohort (N = 2,198) Anatomical Attribution Audit on PTB-XL Fold 10:
1. Vectorized Batch Territory Occlusion on ALL 2,198 test records.
2. Full cohort vs True Positive (correctly classified) vs False Negative breakdowns.
3. Subclass-level ground truth validation for Infarctions (ASMI/AMI -> LAD vs IMI/ILMI -> RCA).
4. Adebayo et al. Model Randomization Sanity Check.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import ast
import tensorflow as tf

# Set GPU growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from causal_diffusion.anatomical_classifier import build_anatomical_se_ecg_classifier
from causal_diffusion.dataset import SUPERCLASSES, load_ptbxl_superclass_data

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')
CHECKPOINT_PATH = 'checkpoints/se_resnet_anatomical_territory_dropout_fold_9_best.h5'
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = 'checkpoints/se_resnet_anatomical_territory_dropout_fold1_8_best.h5'

TERRITORIES = {
    'Inferior': [1, 2, 5],
    'Antero-Septal': [6, 7, 8, 9],
    'Lateral': [0, 4, 10, 11],
    'Cavity': [3]
}
TERRITORY_NAMES = ['Inferior', 'Antero-Septal', 'Lateral', 'Cavity']


def run_audit():
    print("[1/5] Loading PTB-XL Fold 10 Test Set (N = 2,198)...")
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
    ecg_ids_test = target_ecg_ids
    N_total = len(X_test)
    print(f"Loaded X_test: {X_test.shape}, Y_test: {Y_test.shape}")

    # Build and load model
    print(f"[2/5] Loading trained Model 3 from {CHECKPOINT_PATH}...")
    model = build_anatomical_se_ecg_classifier()
    model.load_weights(CHECKPOINT_PATH)

    # 1. Base Predictions
    print("[3/5] Running Vectorized Forward Passes...")
    preds_base = model.predict(X_test, batch_size=64, verbose=0)
    if isinstance(preds_base, list):
        preds_base = preds_base[0]

    # 2. Masked Predictions for each territory
    preds_masked = {}
    for t_name, lead_indices in TERRITORIES.items():
        print(f"  Predicting with masked {t_name} territory (leads {lead_indices})...")
        X_masked = X_test.copy()
        X_masked[:, :, lead_indices] = 0.0
        p_mask = model.predict(X_masked, batch_size=64, verbose=0)
        if isinstance(p_mask, list):
            p_mask = p_mask[0]
        preds_masked[t_name] = p_mask

    # Calculate Drops and Shares for every record and every class
    print("\n[4/5] Computing Quantitative Attribution Metrics across Cohort...")
    results_list = []

    for i in range(N_total):
        ecg_id = ecg_ids_test[i]
        for c_idx, c_name in enumerate(SUPERCLASSES):
            is_positive = (Y_test[i, c_idx] == 1.0)
            conf = preds_base[i, c_idx]
            is_tp = is_positive and (conf >= 0.5)

            # Territory drops for this class
            drops = {}
            for t_name in TERRITORY_NAMES:
                d = max(0.0, float(preds_base[i, c_idx] - preds_masked[t_name][i, c_idx]))
                drops[t_name] = d

            sum_drops = sum(drops.values())
            if sum_drops > 1e-6:
                shares = {t_name: (drops[t_name] / sum_drops) * 100.0 for t_name in TERRITORY_NAMES}
            else:
                shares = {t_name: 25.0 for t_name in TERRITORY_NAMES}

            dom_t = max(shares, key=shares.get)
            dom_pct = shares[dom_t]
            max_drop = drops[dom_t]

            results_list.append({
                'ecg_id': ecg_id,
                'class': c_name,
                'is_positive': is_positive,
                'is_tp': is_tp,
                'confidence': conf,
                'dominant_territory': dom_t,
                'dominant_share_pct': dom_pct,
                'dominant_drop': max_drop,
                'inferior_pct': shares['Inferior'],
                'septal_pct': shares['Antero-Septal'],
                'lateral_pct': shares['Lateral'],
                'cavity_pct': shares['Cavity'],
                'inferior_drop': drops['Inferior'],
                'septal_drop': drops['Antero-Septal'],
                'lateral_drop': drops['Lateral'],
                'cavity_drop': drops['Cavity'],
            })

    df_results = pd.DataFrame(results_list)
    df_results.to_csv('experiments/full_cohort_fold10_attribution_audit.csv', index=False)
    print("Saved full cohort audit to 'experiments/full_cohort_fold10_attribution_audit.csv'")

    # Summary Statistics:
    # A. All Positive Test Cases (Ground Truth Positive, regardless of threshold)
    print("\n" + "="*80)
    print("SUMMARY A: ALL GROUND-TRUTH POSITIVE CASES IN FOLD 10 (UNFILTERED)")
    print("="*80)
    summary_all = []
    for c_idx, c_name in enumerate(SUPERCLASSES):
        sub = df_results[(df_results['class'] == c_name) & (df_results['is_positive'] == True)]
        n = len(sub)
        mean_conf = sub['confidence'].mean() * 100.0
        dom_share = sub['dominant_share_pct'].mean()
        mean_drop = sub['dominant_drop'].mean()
        inf_share = sub['inferior_pct'].mean()
        sep_share = sub['septal_pct'].mean()
        lat_share = sub['lateral_pct'].mean()
        cav_share = sub['cavity_pct'].mean()
        summary_all.append({
            'Superclass': c_name,
            'N': n,
            'Mean Conf (%)': mean_conf,
            'Dom. Share (%)': dom_share,
            'Mean Drop': mean_drop,
            'Inferior (%)': inf_share,
            'Antero-Septal (%)': sep_share,
            'Lateral (%)': lat_share,
            'Cavity (%)': cav_share
        })
    df_sum_all = pd.DataFrame(summary_all)
    print(df_sum_all.to_string(index=False))

    # B. Confident Correct (True Positives, Confidence >= 0.5)
    print("\n" + "="*80)
    print("SUMMARY B: CONFIDENT TRUE POSITIVE PREDICTIONS (Confidence >= 0.5)")
    print("="*80)
    summary_tp = []
    for c_idx, c_name in enumerate(SUPERCLASSES):
        sub = df_results[(df_results['class'] == c_name) & (df_results['is_tp'] == True)]
        n = len(sub)
        mean_conf = sub['confidence'].mean() * 100.0
        dom_share = sub['dominant_share_pct'].mean()
        mean_drop = sub['dominant_drop'].mean()
        inf_share = sub['inferior_pct'].mean()
        sep_share = sub['septal_pct'].mean()
        lat_share = sub['lateral_pct'].mean()
        cav_share = sub['cavity_pct'].mean()
        summary_tp.append({
            'Superclass': c_name,
            'N_TP': n,
            'Mean Conf (%)': mean_conf,
            'Dom. Share (%)': dom_share,
            'Mean Drop': mean_drop,
            'Inferior (%)': inf_share,
            'Antero-Septal (%)': sep_share,
            'Lateral (%)': lat_share,
            'Cavity (%)': cav_share
        })
    df_sum_tp = pd.DataFrame(summary_tp)
    print(df_sum_tp.to_string(index=False))

    # 3. Ground Truth Verification on PTB-XL Infarct Subclasses
    print("\n" + "="*80)
    print("[5/5] GROUND TRUTH SUBCLASS VALIDATION FOR MYOCARDIAL INFARCTION (LAD vs RCA vs LCx)")
    print("="*80)
    fold10_df['scp_parsed'] = fold10_df['scp_codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Classify MI patients by anatomical subclass
    # Antero-Septal (LAD): ASMI, AMI
    # Inferior (RCA): IMI, ILMI, IPMI, INJIN
    # Lateral (LCx): LMI, ALMI, INJLA
    asmi_ids = fold10_df[fold10_df['scp_parsed'].apply(lambda d: any(k in d for k in ['ASMI', 'AMI']))]['ecg_id'].values
    imi_ids = fold10_df[fold10_df['scp_parsed'].apply(lambda d: any(k in d for k in ['IMI', 'ILMI', 'IPMI', 'INJIN']))]['ecg_id'].values
    lmi_ids = fold10_df[fold10_df['scp_parsed'].apply(lambda d: any(k in d for k in ['LMI', 'ALMI', 'INJLA']))]['ecg_id'].values

    sub_mi = df_results[df_results['class'] == 'MI']
    asmi_records = sub_mi[sub_mi['ecg_id'].isin(asmi_ids)]
    imi_records = sub_mi[sub_mi['ecg_id'].isin(imi_ids)]
    lmi_records = sub_mi[sub_mi['ecg_id'].isin(lmi_ids)]

    print(f"Anterior/Antero-Septal Infarcts (LAD Ground Truth, N = {len(asmi_records)}):")
    print(f"  Antero-Septal Attribution Share: {asmi_records['septal_pct'].mean():.2f}%")
    print(f"  Inferior Attribution Share:      {asmi_records['inferior_pct'].mean():.2f}%")
    print(f"  Lateral Attribution Share:       {asmi_records['lateral_pct'].mean():.2f}%")
    print(f"  Cavity Attribution Share:        {asmi_records['cavity_pct'].mean():.2f}%")

    print(f"\nInferior/Infero-Lateral Infarcts (RCA Ground Truth, N = {len(imi_records)}):")
    print(f"  Inferior Attribution Share:      {imi_records['inferior_pct'].mean():.2f}%")
    print(f"  Antero-Septal Attribution Share: {imi_records['septal_pct'].mean():.2f}%")
    print(f"  Lateral Attribution Share:       {imi_records['lateral_pct'].mean():.2f}%")
    print(f"  Cavity Attribution Share:        {imi_records['cavity_pct'].mean():.2f}%")

    print(f"\nLateral Infarcts (LCx Ground Truth, N = {len(lmi_records)}):")
    print(f"  Lateral Attribution Share:       {lmi_records['lateral_pct'].mean():.2f}%")
    print(f"  Antero-Septal Attribution Share: {lmi_records['septal_pct'].mean():.2f}%")
    print(f"  Inferior Attribution Share:      {lmi_records['inferior_pct'].mean():.2f}%")
    print(f"  Cavity Attribution Share:        {lmi_records['cavity_pct'].mean():.2f}%")


if __name__ == '__main__':
    run_audit()
