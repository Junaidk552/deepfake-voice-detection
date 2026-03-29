#!/usr/bin/env python3
"""random forest feature importance"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

with open('data/train_test_splits.pkl', 'rb') as f:
    splits = pickle.load(f)

X_train = splits['X_train']
y_train = splits['y_train']

feature_cols = list(X_train.columns)

# need to retrain RF to get importances from this exact split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_scaled, y_train)

importances = rf.feature_importances_
# print(f'sum of importances: {importances.sum():.4f}')

imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
})

def get_category(name):
    if name.startswith('mfcc'): return 'MFCC'
    if name.startswith('cqcc'): return 'CQCC'
    if name.startswith('rqa'): return 'RQA'
    if name.startswith('entropy'): return 'Entropy'
    if name.startswith('pause'): return 'Pauses'
    return 'Other'

imp_df['category'] = imp_df['feature'].apply(get_category)
imp_df = imp_df.sort_values('importance', ascending=False)

print("FEATURE IMPORTANCE BY CATEGORY\n")

cat_summary = imp_df.groupby('category').agg(
    total_importance=('importance', 'sum'),
    mean_importance=('importance', 'mean'),
    n_features=('importance', 'count'),
    best_feature=('importance', 'idxmax')
).sort_values('total_importance', ascending=False)

for cat in cat_summary.index:
    cat_df = imp_df[imp_df['category'] == cat]
    best = cat_df.iloc[0]
    cat_summary.loc[cat, 'best_feature'] = best['feature']
    cat_summary.loc[cat, 'best_importance'] = best['importance']

print(f"\n{'Category':<12} {'Total':>8} {'Mean':>8} {'Count':>6} "
      f"{'Best Feature':<25} {'Best Imp':>8}")
for cat, row in cat_summary.iterrows():
    print(f"{cat:<12} {row['total_importance']:>8.4f} "
          f"{row['mean_importance']:>8.4f} {int(row['n_features']):>6} "
          f"{row['best_feature']:<25} {row['best_importance']:>8.4f}")

print(f"\nTOP 20 FEATURES\n")

top20 = imp_df.head(20)
for rank, (_, row) in enumerate(top20.iterrows(), 1):
    print(f"  {rank:>2}. {row['feature']:<28} {row['category']:<10} "
          f"{row['importance']:.4f}")

novel_in_top20 = top20[top20['category'].isin(['RQA', 'Entropy', 'Pauses'])]
print(f"\nNovel features in top 20: {len(novel_in_top20)}/20")

fig, ax = plt.subplots(figsize=(10, 8))

colors = {
    'MFCC': '#3498db',
    'CQCC': '#2ecc71',
    'RQA': '#e74c3c',
    'Entropy': '#f39c12',
    'Pauses': '#9b59b6'
}

bars = ax.barh(range(19, -1, -1), top20['importance'].values,
               color=[colors[c] for c in top20['category'].values],
               edgecolor='white', linewidth=0.5)

ax.set_yticks(range(19, -1, -1))
ax.set_yticklabels(top20['feature'].values, fontsize=9)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 20 Features by Random Forest Importance', fontsize=14)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[c], label=c)
                   for c in ['MFCC', 'CQCC', 'RQA', 'Entropy', 'Pauses']]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
Path('results').mkdir(parents=True, exist_ok=True)
plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
print(f"\nSaved results/feature_importance.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))

cats = cat_summary.index.tolist()
totals = cat_summary['total_importance'].values
cat_colors = [colors[c] for c in cats]

ax.bar(cats, totals, color=cat_colors, edgecolor='white')
ax.set_ylabel('Total Feature Importance', fontsize=12)
ax.set_title('Feature Importance by Category', fontsize=14)

for i, (cat, val) in enumerate(zip(cats, totals)):
    n = int(cat_summary.loc[cat, 'n_features'])
    ax.text(i, val + 0.005, f'{val:.3f}\n({n} features)',
            ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/feature_importance_by_category.png',
            dpi=150, bbox_inches='tight')
print(f"Saved results/feature_importance_by_category.png")
plt.close()

with open('results/feature_importance_analysis.txt', 'w') as f:
    f.write("Feature Importance Analysis\n")
    f.write("=" * 60 + "\n\n")
    f.write("Category Summary:\n")
    for cat, row in cat_summary.iterrows():
        f.write(f"  {cat}: total={row['total_importance']:.4f}, "
                f"mean={row['mean_importance']:.4f}, "
                f"n={int(row['n_features'])}, "
                f"best={row['best_feature']}\n")
    f.write(f"\nTop 20:\n")
    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        f.write(f"  {rank}. {row['feature']} ({row['category']}) "
                f"= {row['importance']:.4f}\n")
    f.write(f"\nNovel features in top 20: {len(novel_in_top20)}/20\n")

print(f"Saved results/feature_importance_analysis.txt")
