"""
plot_entropy_distributions.py

This script visualises data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- features/entropy_features.pkl
Outputs:
- results/entropy_distributions.png
Reproduces: Figure 4.11.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

df = pd.read_pickle('features/entropy_features.pkl')

df['class'] = df['label'].apply(lambda x: 'Real' if x == 'real' else 'Synthetic')

entropy_cols = [c for c in df.columns if c.startswith('entropy')]
print(f"Entropy features: {entropy_cols}\n")

real = df[df['class'] == 'Real']
synth = df[df['class'] == 'Synthetic']

# t-tests per scale
print("Independent Samples T-Tests: Real vs Synthetic")

for col in entropy_cols:
    scale = col.replace('entropy_scale_', 'Scale ')
    t_stat, p_val = stats.ttest_ind(real[col].dropna(), synth[col].dropna())
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    pooled_std = np.sqrt((real[col].std()**2 + synth[col].std()**2) / 2)
    d = (real[col].mean() - synth[col].mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  {scale:15s}  t={t_stat:8.3f}  p={p_val:.2e}  {sig}  Cohen's d={d:.3f}")

print("\n*** p<0.001  ** p<0.01  * p<0.05  ns = not significant")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Multiscale Sample Entropy: Real vs Synthetic Speech',
             fontsize=16, fontweight='bold', y=1.02)

ax1 = axes[0]
scales = range(1, len(entropy_cols) + 1)

real_means = [real[c].mean() for c in entropy_cols]
real_stds = [real[c].std() for c in entropy_cols]
synth_means = [synth[c].mean() for c in entropy_cols]
synth_stds = [synth[c].std() for c in entropy_cols]

ax1.errorbar(scales, real_means, yerr=real_stds, fmt='o-',
             color='#2ecc71', linewidth=2, markersize=6,
             capsize=4, label='Real', alpha=0.9)
ax1.errorbar(scales, synth_means, yerr=synth_stds, fmt='s-',
             color='#e74c3c', linewidth=2, markersize=6,
             capsize=4, label='Synthetic', alpha=0.9)

ax1.set_xlabel('Scale', fontsize=12)
ax1.set_ylabel('Sample Entropy', fontsize=12)
ax1.set_title('Mean Entropy +/- SD Across Scales', fontsize=12)
ax1.set_xticks(list(scales))
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
selected = [0, 2, 4, 9]  # scales 1, 3, 5, 10
box_data = []
box_labels = []
box_colors = []

for idx in selected:
    col = entropy_cols[idx]
    scale_num = idx + 1
    box_data.append(real[col].dropna().values)
    box_labels.append(f'S{scale_num}\nReal')
    box_colors.append('#2ecc71')
    box_data.append(synth[col].dropna().values)
    box_labels.append(f'S{scale_num}\nSynth')
    box_colors.append('#e74c3c')

bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_title('Entropy at Selected Scales (1, 3, 5, 10)', fontsize=12)
ax2.set_ylabel('Sample Entropy', fontsize=12)
ax2.tick_params(labelsize=9)

plt.tight_layout()
Path('results').mkdir(parents=True, exist_ok=True)
plt.savefig('results/entropy_distributions.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: results/entropy_distributions.png")
plt.close()

print("\nDone!")
