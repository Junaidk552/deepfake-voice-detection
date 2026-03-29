#!/usr/bin/env python3
"""extract features from voice clone samples, test with trained models"""

import pandas as pd
import numpy as np
import pickle
import subprocess
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from feature_utils import extract_all_from_audio as extract_all

clone_real = list(Path('dataset/voice_clone/real').glob('real_junaid_*.wav'))
clone_synth = list(Path('dataset/voice_clone/synthetic').glob('synthetic_junaid_*.wav'))

print(f"Voice clone files: {len(clone_real)} real, {len(clone_synth)} synthetic")

temp_dir = Path('dataset_voice_clone_temp')
temp_real = temp_dir / 'real'
temp_synth = temp_dir / 'synthetic' / 'elevenlabs_clone'

temp_real.mkdir(parents=True, exist_ok=True)
temp_synth.mkdir(parents=True, exist_ok=True)

import shutil
for f in clone_real:
    shutil.copy2(f, temp_real / f.name)
for f in clone_synth:
    shutil.copy2(f, temp_synth / f.name)

print("Temp dataset created. Running original extraction scripts...")

import librosa

def get_all_files():
    files = []
    labels = []
    for f in sorted(temp_real.glob('*.wav')):
        files.append(f)
        labels.append('real')
    for f in sorted(temp_synth.glob('*.wav')):
        files.append(f)
        labels.append('synthetic_elevenlabs_clone')
    return files, labels

files, labels = get_all_files()
print(f"Total files: {len(files)}")

print("\nExtracting features...")
rows = []
for f, label in zip(files, labels):
    y, sr = librosa.load(str(f), sr=16000, duration=10)
    row = extract_all(y, sr)
    row['filename'] = f.name
    row['label'] = label
    rows.append(row)

combined = pd.DataFrame(rows)
feature_cols = [c for c in combined.columns if c not in ['filename', 'label']]
combined[feature_cols] = combined[feature_cols].fillna(combined[feature_cols].median())
combined['binary_label'] = combined['label'].apply(lambda x: 0 if x == 'real' else 1)

print(f"Combined: {combined.shape}")
print(f"Features: {len(feature_cols)}")

combined.to_pickle('features/voice_clone_features_v2.pkl')

print("\nVOICE CLONE DETECTION RESULTS (v2 - consistent extraction)")

with open('data/train_test_splits.pkl', 'rb') as f:
    splits = pickle.load(f)

X_train_full = splits['X_train']
y_train = splits['y_train']
y_true = combined['binary_label'].values

feature_groups = {
    'mfcc': [c for c in feature_cols if c.startswith('mfcc')],
    'cqcc': [c for c in feature_cols if c.startswith('cqcc')],
    'rqa': [c for c in feature_cols if c.startswith('rqa')],
    'entropy': [c for c in feature_cols if c.startswith('entropy')],
    'pauses': [c for c in feature_cols if c.startswith('pause')],
}

configs = {
    'MFCC only': feature_groups['mfcc'],
    'CQCC only': feature_groups['cqcc'],
    'RQA only': feature_groups['rqa'],
    'Pauses only': feature_groups['pauses'],
    'MFCC+CQCC': feature_groups['mfcc'] + feature_groups['cqcc'],
    'Novel only': (feature_groups['rqa'] + feature_groups['entropy'] +
                   feature_groups['pauses']),
    'All features': feature_cols,
}

print(f"\n{'Config':<18} {'Accuracy':>8} {'F1':>8} {'Real OK':>8} {'Fake OK':>8}")

results = []
for config_name, cols in configs.items():
    train_cols = [c for c in cols if c in X_train_full.columns]
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_full[train_cols])

    model = SVC(kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=42)
    model.fit(X_tr, y_train)

    X_clone = scaler.transform(combined[cols].values)
    y_pred = model.predict(X_clone)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    real_ok = np.sum((y_true == 0) & (y_pred == 0))
    fake_ok = np.sum((y_true == 1) & (y_pred == 1))

    print(f"{config_name:<18} {acc:>7.1%} {f1:>8.4f} {real_ok:>5}/25  {fake_ok:>5}/25")

    results.append({
        'config': config_name, 'accuracy': acc, 'f1': f1,
        'real_correct': int(real_ok), 'fake_correct': int(fake_ok)
    })

res_df = pd.DataFrame(results)
res_df.to_csv('results/voice_clone_results.csv', index=False)
print(f"\nSaved results/voice_clone_results.csv")

shutil.rmtree(temp_dir)
print("Temp files cleaned up.")
