"""
cache_ptbxl_dataset.py
Fast parallel preprocessor to filter and binary-cache all 21,799 PTB-XL ECG signals into a single .npz file.
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

DATA_DIR = 'dataset-1.0.3'
CACHE_PATH = os.path.join(DATA_DIR, 'ptbxl_100hz_cached.npz')


def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=100.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


def process_record(args):
    ecg_id, rel_path = args
    rec_path = os.path.join(DATA_DIR, rel_path)
    sig, _ = wfdb.rdsamp(rec_path)
    sig = sig.astype(np.float32)
    filtered = butter_bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=100.0)
    return ecg_id, filtered


def main():
    if os.path.exists(CACHE_PATH):
        print(f"[Cache Check] Binary cache already exists at '{CACHE_PATH}'. Skipping generation.")
        return

    print("Reading ptbxl_database.csv...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'))
    records = list(zip(df['ecg_id'].values, df['filename_lr'].values))

    print(f"Filtering and caching {len(records)} ECG records across CPU processes...")
    ecg_ids = []
    signals = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_record, records), total=len(records)))

    for ecg_id, sig in results:
        ecg_ids.append(ecg_id)
        signals.append(sig)

    X = np.array(signals, dtype=np.float32)
    ecg_ids = np.array(ecg_ids, dtype=np.int32)

    print(f"Saving binary cache to '{CACHE_PATH}' (Shape: {X.shape}, Size: {X.nbytes / 1e6:.1f} MB)...")
    np.savez_compressed(CACHE_PATH, X=X, ecg_ids=ecg_ids)
    print("[Cache Complete] Binary cache successfully created!")


if __name__ == '__main__':
    main()
