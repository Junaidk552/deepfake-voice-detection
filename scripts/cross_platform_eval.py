"""
cross_platform_eval.py

This script evaluates data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- features/all_features_combined.pkl
Outputs:
- results/cross_platform_results.csv
Reproduces: Table 4.2 (cross-platform generalisation benchmark).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve)


def calculate_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2


def main():
    print("Loading features...")
    df = pd.read_pickle('features/all_features_combined.pkl')

    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'real' else 1)
    # print(df['label'].value_counts())

    platforms = ['synthetic_elevenlabs', 'synthetic_google', 'synthetic_polly']
    real_df = df[df['label'] == 'real']

    feature_groups = {
        'mfcc': [c for c in df.columns if c.startswith('mfcc')],
        'cqcc': [c for c in df.columns if c.startswith('cqcc')],
        'rqa': [c for c in df.columns if c.startswith('rqa')],
        'entropy': [c for c in df.columns if c.startswith('entropy')],
        'pauses': [c for c in df.columns if c.startswith('pause')],
    }

    configs = {
        'MFCC only': feature_groups['mfcc'],
        'CQCC only': feature_groups['cqcc'],
        'Pauses only': feature_groups['pauses'],
        'RQA only': feature_groups['rqa'],
        'MFCC+CQCC': feature_groups['mfcc'] + feature_groups['cqcc'],
        'Novel only': (feature_groups['rqa'] + feature_groups['entropy'] +
                       feature_groups['pauses']),
        'All features': (feature_groups['mfcc'] + feature_groups['cqcc'] +
                         feature_groups['rqa'] + feature_groups['entropy'] +
                         feature_groups['pauses']),
    }

    # Reuse the best model family from main training so this test isolates domain shift, not model choice.
    clf = SVC(kernel='rbf', C=10, gamma='scale',
              probability=True, random_state=42)

    results = []

    for held_out in platforms:
        platform_name = held_out.replace('synthetic_', '')
        print(f"\nHELD OUT: {platform_name.upper()}")
        print(f"Training on the other 2 platforms + real speech")
        print(f"Testing on {platform_name} + real speech")

        # Leave-one-platform-out:
        # train on real + two synthetic sources, then test on real + the unseen source.
        # This is the closest check to "can it catch a spoofing system it never trained on?"
        train_platforms = [p for p in platforms if p != held_out]

        real_train = real_df.sample(frac=0.8, random_state=42)
        real_test = real_df.drop(real_train.index)

        synth_train = df[df['label'].isin(train_platforms)]
        synth_test = df[df['label'] == held_out]

        train_df = pd.concat([real_train, synth_train])
        test_df = pd.concat([real_test, synth_test])

        print(f"  Train: {len(train_df)} ({len(real_train)} real + "
              f"{len(synth_train)} synthetic)")
        print(f"  Test:  {len(test_df)} ({len(real_test)} real + "
              f"{len(synth_test)} {platform_name})")

        for config_name, feature_cols in configs.items():
            X_train = train_df[feature_cols].values
            y_train = train_df['binary_label'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['binary_label'].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(kernel='rbf', C=10, gamma='scale',
                        probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            eer = calculate_eer(y_test, y_proba)

            print(f"  {config_name:<18} acc={acc:.4f}  f1={f1:.4f}  "
                  f"auc={auc:.4f}  eer={eer:.4f}")

            results.append({
                'held_out': platform_name,
                'config': config_name,
                'accuracy': acc,
                'f1': f1,
                'auc': auc,
                'eer': eer,
                'train_size': len(train_df),
                'test_size': len(test_df)
            })

    res_df = pd.DataFrame(results)
    Path('results').mkdir(parents=True, exist_ok=True)
    res_df.to_csv('results/cross_platform_results.csv', index=False)
    print(f"\nSaved results/cross_platform_results.csv")

    print("\nAVERAGE ACROSS ALL HELD-OUT PLATFORMS")

    avg = res_df.groupby('config')[['accuracy', 'f1', 'auc', 'eer']].mean()
    avg = avg.sort_values('f1', ascending=False)

    print(f"\n{'Config':<18} {'Acc':>6} {'F1':>6} {'AUC':>6} {'EER':>6}")
    for config_name, row in avg.iterrows():
        print(f"{config_name:<18} {row['accuracy']:.4f} {row['f1']:.4f} "
              f"{row['auc']:.4f} {row['eer']:.4f}")

    print(f"\nDo novel features help with generalisation?")

    all_f1 = avg.loc['All features', 'f1']
    std_f1 = avg.loc['MFCC+CQCC', 'f1']
    mfcc_f1 = avg.loc['MFCC only', 'f1']
    novel_f1 = avg.loc['Novel only', 'f1']

    print(f"  MFCC only:      F1={mfcc_f1:.4f}")
    print(f"  MFCC+CQCC:      F1={std_f1:.4f}")
    print(f"  Novel only:     F1={novel_f1:.4f}")
    print(f"  All features:   F1={all_f1:.4f}")

    if all_f1 > std_f1:
        diff = (all_f1 - std_f1) / std_f1 * 100
        print(f"\n  Novel features improve cross-platform F1 by {diff:.1f}%")
    else:
        print(f"\n  Standard features still lead on cross-platform")


if __name__ == '__main__':
    main()
