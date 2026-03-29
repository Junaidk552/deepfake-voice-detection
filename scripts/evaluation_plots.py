"""generate evaluation plots - confusion matrix, ROC, per-platform, model comparison"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)

with open('data/train_test_splits.pkl', 'rb') as f:
    splits = pickle.load(f)

X_train = splits['X_train']
X_test = splits['X_test']
y_train = splits['y_train']
y_test = splits['y_test']

df = pd.read_pickle('features/all_features_combined.pkl')
df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'real' else 1)

feature_cols = [c for c in X_train.columns]

Path('results').mkdir(parents=True, exist_ok=True)

print("Plot 1: Confusion Matrix...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)

fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Real', 'Synthetic'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix: SVM (All Features)', fontsize=14)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved results/confusion_matrix.png")

print("Plot 2: ROC Curves...")

classifiers = {
    'Logistic Regression': LogisticRegression(C=1, max_iter=1000, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale',
                     probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                            random_state=42)
}

fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#2ecc71', '#e74c3c', '#3498db']

for (name, clf), color in zip(classifiers.items(), colors):
    clf.fit(X_train_scaled, y_train)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{name} (AUC={auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: All Features (77)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved results/roc_curves.png")

print("Plot 3: Per-Platform Performance...")

test_indices = X_test.index
test_labels = df.loc[test_indices, 'label']

svm_prob = SVC(kernel='rbf', C=10, gamma='scale',
               probability=True, random_state=42)
svm_prob.fit(X_train_scaled, y_train)
y_pred_all = svm_prob.predict(X_test_scaled)

platforms = ['real', 'synthetic_elevenlabs', 'synthetic_google', 'synthetic_polly']
platform_names = ['Real', 'ElevenLabs', 'Google TTS', 'Amazon Polly']
platform_acc = []

for platform in platforms:
    mask = test_labels == platform
    if mask.sum() > 0:
        correct = (y_pred_all[mask.values] == y_test[mask].values).sum()
        total = mask.sum()
        platform_acc.append(correct / total)
    else:
        platform_acc.append(0)

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
bars = ax.bar(platform_names, platform_acc, color=bar_colors, edgecolor='white')

for bar, acc in zip(bars, platform_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{acc:.1%}', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Detection Accuracy', fontsize=12)
ax.set_title('Detection Accuracy by TTS Platform', fontsize=14)
ax.set_ylim(0, 1.15)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('results/per_platform_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved results/per_platform_performance.png")

print("Plot 4: Model Comparison Chart...")

results = pd.read_csv('results/model_comparison.csv')
# print(results.head())

fig, ax = plt.subplots(figsize=(14, 7))

configs = results['config'].unique()
classifiers_list = results['classifier'].unique()
x = np.arange(len(configs))
width = 0.25
clf_colors = ['#2ecc71', '#e74c3c', '#3498db']

for i, clf_name in enumerate(classifiers_list):
    clf_data = results[results['classifier'] == clf_name]
    eers = []
    for config in configs:
        row = clf_data[clf_data['config'] == config]
        eers.append(row['eer'].values[0] if len(row) > 0 else 0)
    ax.bar(x + i * width, eers, width, label=clf_name,
           color=clf_colors[i], edgecolor='white')

ax.set_xlabel('Feature Configuration', fontsize=12)
ax.set_ylabel('Equal Error Rate (lower is better)', fontsize=12)
ax.set_title('EER Across All Model Configurations', fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/model_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved results/model_comparison_chart.png")

print("Plot 5: Cross-Platform Comparison...")

cross = pd.read_csv('results/cross_platform_results.csv')

fig, ax = plt.subplots(figsize=(14, 7))

configs_cross = cross['config'].unique()
platforms_held = cross['held_out'].unique()
x = np.arange(len(configs_cross))
width = 0.25
plat_colors = ['#e74c3c', '#3498db', '#f39c12']

for i, plat in enumerate(platforms_held):
    plat_data = cross[cross['held_out'] == plat]
    f1s = []
    for config in configs_cross:
        row = plat_data[plat_data['config'] == config]
        f1s.append(row['f1'].values[0] if len(row) > 0 else 0)
    ax.bar(x + i * width, f1s, width, label=f'Held out: {plat}',
           color=plat_colors[i], edgecolor='white')

ax.set_xlabel('Feature Configuration', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Cross-Platform Generalisation: F1 by Held-Out Platform',
             fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(configs_cross, rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.1)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/cross_platform_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved results/cross_platform_comparison.png")

print("\nAll plots saved!")
