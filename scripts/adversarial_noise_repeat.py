#!/usr/bin/env python3
"""
adversarial_noise_repeat.py

This script processes data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- features/all_features_combined.pkl
- data/train_test_splits.pkl
- dataset/real/*.wav
- dataset/synthetic/*/*.wav
Outputs:
- results/adversarial_noise_repeated.csv
Reproduces: Table 4.6 (repeated adversarial-noise robustness statistics).
"""

import pandas as pd
import numpy as np
import pickle
import librosa
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from feature_utils import extract_all_from_audio as extract_all

warnings.filterwarnings('ignore')

def add_gaussian_noise(y, snr_db):
    sp = np.mean(y ** 2)
    np_power = sp / (10 ** (snr_db / 10))
    return y + np.random.normal(0, np.sqrt(np_power), len(y))

def add_bg_noise(y, snr_db):
    from scipy.signal import lfilter
    white = np.random.randn(len(y))
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.000000000, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white)
    pink = pink / (np.std(pink) + 1e-10)
    sp = np.mean(y ** 2)
    np_power = sp / (10 ** (snr_db / 10))
    return y + pink * np.sqrt(np_power)


def main():
    N_REPEATS = 10
    print(f"Repeated noise trials: {N_REPEATS} reps each")

    df = pd.read_pickle('features/all_features_combined.pkl')
    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'real' else 1)

    with open('data/train_test_splits.pkl', 'rb') as f:
        splits = pickle.load(f)

    X_train = splits['X_train']
    y_train = splits['y_train']
    test_indices = splits['X_test'].index
    test_df = df.loc[test_indices].copy()

    def find_audio(filename):
        for d in ['dataset/real', 'dataset/synthetic/elevenlabs',
                   'dataset/synthetic/google', 'dataset/synthetic/polly']:
            p = Path(d) / filename
            if p.exists(): return str(p)
        return None

    # subset for speed
    np.random.seed(42)
    all_files = test_df['filename'].values
    all_labels = test_df['binary_label'].values
    idx = np.random.choice(len(all_files), min(50, len(all_files)), replace=False)
    files = all_files[idx]
    labels = all_labels[idx]

    feature_cols = list(X_train.columns)
    mfcc_cols = [c for c in feature_cols if c.startswith('mfcc')]

    configs = {'MFCC only': mfcc_cols, 'All features': feature_cols}
    trained = {}
    for name, cols in configs.items():
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[cols])
        model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        model.fit(X_tr, y_train)
        trained[name] = {'model': model, 'scaler': scaler, 'cols': cols}

    # only stochastic attacks
    noise_attacks = {
        'Noise SNR=30dB': lambda y, sr: add_gaussian_noise(y, 30),
        'Noise SNR=20dB': lambda y, sr: add_gaussian_noise(y, 20),
        'Noise SNR=10dB': lambda y, sr: add_gaussian_noise(y, 10),
        'BG Noise 20dB': lambda y, sr: add_bg_noise(y, 20),
        'BG Noise 10dB': lambda y, sr: add_bg_noise(y, 10),
    }

    results = []

    for attack_name, attack_fn in noise_attacks.items():
        print(f"\n--- {attack_name} ---")

        for rep in range(N_REPEATS):
            np.random.seed(rep * 100 + 42)

            rows = []
            for fname, label in zip(files, labels):
                path = find_audio(fname)
                if path is None: continue
                try:
                    y, sr = librosa.load(path, sr=16000)
                    y_att = attack_fn(y, sr)
                    feats = extract_all(y_att, sr)
                    feats['label'] = label
                    rows.append(feats)
                except: continue

            if not rows: continue
            att_df = pd.DataFrame(rows)
            att_df[feature_cols] = att_df[feature_cols].fillna(att_df[feature_cols].median())
            att_df[feature_cols] = att_df[feature_cols].fillna(0)
            y_true = att_df['label'].values

            for config_name, info in trained.items():
                X_att = att_df[info['cols']].values
                X_sc = info['scaler'].transform(X_att)
                y_pred = info['model'].predict(X_sc)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                results.append({
                    'attack': attack_name, 'rep': rep + 1,
                    'config': config_name, 'accuracy': acc, 'f1': f1
                })

            print(f"  rep {rep+1}/{N_REPEATS} done")

    res_df = pd.DataFrame(results)
    res_df.to_csv('results/adversarial_noise_repeated.csv', index=False)
    print(f"\nSaved results/adversarial_noise_repeated.csv")

    # summary with confidence intervals
    print("\nSUMMARY: Mean +/- Std across 10 repetitions")
    print(f"\n{'Attack':<18} {'Config':<18} {'Mean Acc':>9} {'Std':>7} {'Min':>7} {'Max':>7}")
    print("-" * 65)

    for attack in noise_attacks:
        for config in configs:
            subset = res_df[(res_df['attack'] == attack) & (res_df['config'] == config)]
            if len(subset) > 0:
                print(f"{attack:<18} {config:<18} {subset['accuracy'].mean():>8.4f} "
                      f"{subset['accuracy'].std():>7.4f} {subset['accuracy'].min():>7.4f} "
                      f"{subset['accuracy'].max():>7.4f}")

if __name__ == '__main__':
    main()
