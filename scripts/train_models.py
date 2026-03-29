#!/usr/bin/env python3
"""train models on different feature combos and compare"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

warnings.filterwarnings('ignore')


def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer


def main():
    print("Loading features...")
    df = pd.read_pickle('features/all_features_combined.pkl')

    # 0 = real, 1 = synthetic
    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'real' else 1)

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
        'RQA only': feature_groups['rqa'],
        'Entropy only': feature_groups['entropy'],
        'Pauses only': feature_groups['pauses'],
        'MFCC+CQCC': feature_groups['mfcc'] + feature_groups['cqcc'],
        'All features': (feature_groups['mfcc'] + feature_groups['cqcc'] +
                         feature_groups['rqa'] + feature_groups['entropy'] +
                         feature_groups['pauses']),
    }

    X_all = df[[c for c in df.columns
                if c not in ['filename', 'label', 'binary_label']]]
    y = df['binary_label']

    # 80/20 split, stratified
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train_full)}, Test: {len(X_test_full)}")
    print(f"Train balance: {y_train.value_counts().to_dict()}")
    print(f"Test balance: {y_test.value_counts().to_dict()}")

    Path('data').mkdir(parents=True, exist_ok=True)
    with open('data/train_test_splits.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train_full, 'X_test': X_test_full,
            'y_train': y_train, 'y_test': y_test
        }, f)

    classifiers = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {'C': [0.01, 0.1, 1, 10]}
        },
        'SVM': {
            'model': SVC(kernel='rbf', probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None]
            }
        }
    }

    results = []
    best_score = 0
    best_info = None
    best_name = ''

    total = len(configs) * len(classifiers)
    count = 0

    for config_name, feature_cols in configs.items():
        X_train = X_train_full[feature_cols]
        X_test = X_test_full[feature_cols]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for clf_name, clf_info in classifiers.items():
            count += 1
            print(f"\n[{count}/{total}] {config_name} + {clf_name} "
                  f"({len(feature_cols)} features)")

            grid = GridSearchCV(
                clf_info['model'], clf_info['params'],
                cv=5, scoring='f1', n_jobs=-1, refit=True
            )
            grid.fit(X_train_scaled, y_train)

            y_pred = grid.predict(X_test_scaled)
            y_proba = grid.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            eer = calculate_eer(y_test, y_proba)

            print(f"  params: {grid.best_params_}")
            print(f"  acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}  eer={eer:.4f}")

            results.append({
                'config': config_name,
                'classifier': clf_name,
                'n_features': len(feature_cols),
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'eer': eer,
                'best_params': str(grid.best_params_),
                'cv_best_score': grid.best_score_
            })

            if f1 > best_score:
                best_score = f1
                best_info = {
                    'model': grid.best_estimator_,
                    'scaler': scaler,
                    'features': feature_cols,
                    'config': config_name,
                    'classifier': clf_name
                }
                best_name = f"{config_name} + {clf_name}"

    Path('results').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv('results/model_comparison.csv', index=False)
    print("\nSaved results/model_comparison.csv")

    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_info, f)
    print(f"Saved models/best_model.pkl ({best_name})")

    print(f"\nBest overall: {best_name} (F1={best_score:.4f})")
    # print(res_df[['config', 'classifier', 'f1']].sort_values('f1', ascending=False).head())

    res_sorted = res_df.sort_values('f1', ascending=False)
    print(f"\n{'Config':<20} {'Classifier':<22} {'Acc':>6} {'F1':>6} "
          f"{'AUC':>6} {'EER':>6}")
    for _, row in res_sorted.iterrows():
        print(f"{row['config']:<20} {row['classifier']:<22} "
              f"{row['accuracy']:.4f} {row['f1']:.4f} "
              f"{row['auc']:.4f} {row['eer']:.4f}")

    all_best = res_df[res_df['config'] == 'All features'].sort_values(
        'f1', ascending=False).iloc[0]
    std_best = res_df[res_df['config'] == 'MFCC+CQCC'].sort_values(
        'f1', ascending=False).iloc[0]

    print(f"\nAll features best:  {all_best['classifier']} - "
          f"F1={all_best['f1']:.4f}, EER={all_best['eer']:.4f}")
    print(f"MFCC+CQCC best:     {std_best['classifier']} - "
          f"F1={std_best['f1']:.4f}, EER={std_best['eer']:.4f}")

    if all_best['f1'] > std_best['f1']:
        diff = (all_best['f1'] - std_best['f1']) / std_best['f1'] * 100
        print(f"\nNovel features improved F1 by {diff:.1f}%")
    else:
        print("\nNo F1 improvement, but novel features still add "
              "interpretability")


if __name__ == '__main__':
    main()
