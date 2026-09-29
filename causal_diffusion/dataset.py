"""
causal_diffusion/dataset.py
Data loading and preprocessing for PTB-XL (v1.0.3) in TensorFlow/Keras.
Supports 100Hz and 500Hz sampling rates.
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf
from scipy.signal import butter, filtfilt



DIAGNOSTIC_CLASSES = ['NORM', 'LBBB', 'RBBB', 'MI', 'STTC']
CLASS_TO_IDX = {c: i for i, c in enumerate(DIAGNOSTIC_CLASSES)}


def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=100.0, order=3):
    """
    Applies zero-phase Butterworth bandpass filter to remove baseline wander and high-frequency noise.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


def load_ptbxl_metadata(data_dir):
    """
    Loads ptbxl_database.csv and scp_statements.csv and adds diagnostic targets.
    """
    db_path = os.path.join(data_dir, 'ptbxl_database.csv')
    df = pd.read_csv(db_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    def extract_label(scp):
        # Specific target check
        if 'NORM' in scp:
            return 'NORM'
        if 'CLBBB' in scp or 'ILBBB' in scp:
            return 'LBBB'
        if 'CRBBB' in scp or 'IRBBB' in scp:
            return 'RBBB'
        if any(k in scp for k in ['AMI', 'IMI', 'LMI', 'PMI', 'ISCA', 'ISCI']):
            return 'MI'
        if any(k in scp for k in ['STTC', 'NST_']):
            return 'STTC'
        return 'OTHER'

    df['primary_class'] = df.scp_codes.apply(extract_label)
    return df


def load_ecg_record(data_dir, filename_lr, apply_filter=True, fs=100.0):
    """
    Reads a single 12-lead record from WFDB format.
    Returns:
        sig: numpy array of shape (1000, 12)
    """
    rec_path = os.path.join(data_dir, filename_lr)
    sig, meta = wfdb.rdsamp(rec_path)
    sig = sig.astype(np.float32)

    if apply_filter:
        sig = butter_bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=fs)

    return sig


def estimate_qrs_and_p_masks(signal, fs=100.0):
    """
    Heuristic delineation of QRS and P-wave regions based on Lead II derivative energy.
    Produces binary mask W_anat where 1 indicates non-pathological segments (P-wave, baseline)
    and 0 indicates ventricular QRS complex.
    """
    T = signal.shape[0]
    lead2 = signal[:, 1]  # Lead II
    # High-frequency derivative
    diff = np.diff(lead2, prepend=lead2[0])
    energy = diff ** 2

    # Smooth energy
    kernel_size = int(0.08 * fs)  # 80ms window
    smoothed = np.convolve(energy, np.ones(kernel_size) / kernel_size, mode='same')

    # Threshold for QRS complex
    qrs_threshold = np.percentile(smoothed, 85)
    qrs_mask = smoothed > qrs_threshold

    # Dilate QRS mask by ~120ms (conduction window)
    qrs_dilate = int(0.12 * fs)
    qrs_expanded = np.convolve(qrs_mask.astype(float), np.ones(qrs_dilate), mode='same') > 0

    # Expand to 2D (T, 12)
    qrs_2d = np.tile(np.expand_dims(qrs_expanded, 1), (1, 12))
    anat_mask = (~qrs_expanded).astype(np.float32)
    anat_2d = np.tile(np.expand_dims(anat_mask, 1), (1, 12))
    return qrs_2d, anat_2d


SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def load_ptbxl_superclass_data(data_dir, folds, sampling_rate='lr'):
    """
    Loads PTB-XL records and multi-label superclass targets for specified folds.
    """
    db_path = os.path.join(data_dir, 'ptbxl_database.csv')
    scp_path = os.path.join(data_dir, 'scp_statements.csv')

    df = pd.read_csv(db_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    scp_df = pd.read_csv(scp_path, index_col=0)

    # Filter folds
    if isinstance(folds, int):
        folds = [folds]
    df = df[df['strat_fold'].isin(folds)].copy()

    # Map SCP codes to 5 superclasses
    def get_superclass_vector(scp_dict):
        vec = np.zeros(5, dtype=np.float32)
        for code in scp_dict.keys():
            if code in scp_df.index:
                diag_class = scp_df.loc[code, 'diagnostic_class']
                if pd.notna(diag_class) and diag_class in SUPERCLASSES:
                    idx = SUPERCLASSES.index(diag_class)
                    vec[idx] = 1.0
        return vec

    df['target_vec'] = df.scp_codes.apply(get_superclass_vector)

    filenames = df['filename_lr'].values if sampling_rate == 'lr' else df['filename_hr'].values
    targets = np.array(df['target_vec'].tolist(), dtype=np.float32)

    return filenames, targets


class ECGSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence generator for batching 12-lead ECG signals.
    """
    def __init__(self, data_dir, filenames, targets, batch_size=64, shuffle=True, apply_filter=True, fs=100.0):
        self.data_dir = data_dir
        self.filenames = filenames
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.apply_filter = apply_filter
        self.fs = fs
        self.indices = np.arange(len(self.filenames))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_files = self.filenames[batch_indices]
        batch_targets = self.targets[batch_indices]

        batch_x = []
        for fname in batch_files:
            sig = load_ecg_record(self.data_dir, fname, apply_filter=self.apply_filter, fs=self.fs)
            batch_x.append(sig)

        return np.array(batch_x, dtype=np.float32), np.array(batch_targets, dtype=np.float32)


