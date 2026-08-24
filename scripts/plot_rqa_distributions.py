"""
plot_rqa_distributions.py

This script visualises data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- features/rqa_features.pkl
Outputs:
- results/rqa_distributions.png
Reproduces: Figure 4.10.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

df = pd.read_pickle('features/rqa_features.pkl')

df['class'] = df['label'].apply(lambda x: 'Real' if x == 'real' else 'Synthetic')

rqa_cols = [c for c in df.columns if c.startswith('rqa')]
print(f"RQA features: {rqa_cols}\n")

display_names = {
    'rqa_recurrence_rate': 'Recurrence Rate',
    'rqa_determinism': 'Determinism',
    'rqa_avg_diagonal': 'Avg Diagonal Length',
    'rqa_max_diagonal': 'Max Diagonal Length',
    'rqa_entropy_diagonal': 'Diagonal Entropy',
    'rqa_laminarity': 'Laminarity',
    'rqa_trapping_time': 'Trapping Time'
}

print("Independent Samples T-Tests: Real vs Synthetic")

real = df[df['class'] == 'Real']
synth = df[df['class'] == 'Synthetic']

t_results = []
for col in rqa_cols:
    t_stat, p_val = stats.ttest_ind(real[col].dropna(), synth[col].dropna())
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    label = display_names.get(col, col)
    t_results.append((label, t_stat, p_val, sig))
    print(f"  {label:25s}  t={t_stat:8.3f}  p={p_val:.2e}  {sig}")

    # Cohen's d effect size
    pooled_std = np.sqrt((real[col].std()**2 + synth[col].std()**2) / 2)
    d = (real[col].mean() - synth[col].mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  {'':25s}  Cohen's d = {d:.3f}")

print("\n*** p<0.001  ** p<0.01  * p<0.05  ns = not significant")

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('RQA Feature Distributions: Real vs Synthetic Speech',
             fontsize=16, fontweight='bold', y=0.98)

axes = axes.flatten()

for i, col in enumerate(rqa_cols):
    ax = axes[i]
    label = display_names.get(col, col)

    real_vals = real[col].dropna()
    synth_vals = synth[col].dropna()

    bp = ax.boxplot([real_vals, synth_vals],
                    labels=['Real', 'Synthetic'],
                    patch_artist=True,
                    widths=0.6)

    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#e74c3c')
    bp['boxes'][1].set_alpha(0.7)

    _, p_val = stats.ttest_ind(real_vals, synth_vals)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ax.set_title(f"{label}\n(p={p_val:.2e}, {sig})", fontsize=10)
    ax.tick_params(labelsize=9)

# only 7 features, hide the 8th subplot
axes[7].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])

Path('results').mkdir(parents=True, exist_ok=True)
plt.savefig('results/rqa_distributions.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: results/rqa_distributions.png")
plt.close()

print("\nDone!")
