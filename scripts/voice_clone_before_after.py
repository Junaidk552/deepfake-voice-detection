#!/usr/bin/env python3
"""
voice_clone_before_after.py

This script processes data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- features/voice_clone_features_v2.pkl
- features/before_common_voice/all_features_combined.pkl
- features/all_features_combined.pkl
Outputs:
- results/voice_clone_before_after.csv
Reproduces: Table 4.4 (before/after Common Voice comparison).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

clone = pd.read_pickle('features/voice_clone_features_v2.pkl')
feature_cols = [c for c in clone.columns
                if c not in ['filename', 'label', 'binary_label']]
y_true = clone['binary_label'].values

old_features = pd.read_pickle('features/before_common_voice/all_features_combined.pkl')
old_features['binary_label'] = old_features['label'].apply(
    lambda x: 0 if x == 'real' else 1)

new_features = pd.read_pickle('features/all_features_combined.pkl')
new_features['binary_label'] = new_features['label'].apply(
    lambda x: 0 if x == 'real' else 1)


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

print("VOICE CLONE: BEFORE vs AFTER Common Voice")
print("Same voice clone features, different training data\n")

print(f"{'Config':<18} {'BEFORE Acc':>10} {'AFTER Acc':>10} "
      f"{'BEFORE F1':>10} {'AFTER F1':>10} {'Change':>8}")
print("-" * 65)

results = []

for config_name, cols in configs.items():
    for version, train_df, version_name in [
        ('before', old_features, 'Before'),
        ('after', new_features, 'After')
    ]:
        train_cols = [c for c in cols if c in train_df.columns]
        X_train = train_df[train_cols].values
        y_train = train_df['binary_label'].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = SVC(kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=42)
        model.fit(X_train_scaled, y_train)

        X_clone = clone[cols].values
        X_clone_scaled = scaler.transform(X_clone)
        y_pred = model.predict(X_clone_scaled)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        real_ok = np.sum((y_true == 0) & (y_pred == 0))
        fake_ok = np.sum((y_true == 1) & (y_pred == 1))

        results.append({
            'config': config_name, 'version': version,
            'accuracy': acc, 'f1': f1,
            'real_correct': int(real_ok), 'fake_correct': int(fake_ok)
        })

    before = results[-2]
    after = results[-1]
    change = after['accuracy'] - before['accuracy']
    sign = '+' if change >= 0 else ''

    print(f"{config_name:<18} {before['accuracy']:>9.1%} {after['accuracy']:>9.1%} "
          f"{before['f1']:>10.4f} {after['f1']:>10.4f} "
          f"{sign}{change:>6.1%}")

print("\nDETAIL: Real/Fake breakdown")
print(f"\n{'Config':<18} {'BEFORE':>20} {'AFTER':>20}")
print(f"{'':18} {'Real OK  Fake OK':>20} {'Real OK  Fake OK':>20}")
print("-" * 60)

for config_name in configs:
    before = [r for r in results
              if r['config'] == config_name and r['version'] == 'before'][0]
    after = [r for r in results
             if r['config'] == config_name and r['version'] == 'after'][0]
    print(f"{config_name:<18} "
          f"{before['real_correct']:>5}/25  {before['fake_correct']:>5}/25    "
          f"{after['real_correct']:>5}/25  {after['fake_correct']:>5}/25")

res_df = pd.DataFrame(results)
res_df.to_csv('results/voice_clone_before_after.csv', index=False)
print(f"\nSaved results/voice_clone_before_after.csv")
