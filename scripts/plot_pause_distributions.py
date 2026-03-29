import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

df = pd.read_pickle('features/pause_features.pkl')

df['class'] = df['label'].apply(lambda x: 'Real' if x == 'real' else 'Synthetic')

pause_cols = [c for c in df.columns if c.startswith('pause')]
print(f"Pause features: {pause_cols}\n")

display_names = {
    'pause_mean_duration': 'Mean Duration',
    'pause_std_duration': 'Std Duration',
    'pause_rate': 'Pause Rate',
    'pause_median_duration': 'Median Duration',
    'pause_range': 'Pause Range',
    'pause_skewness': 'Skewness',
    'pause_kurtosis': 'Kurtosis',
    'pause_coeff_variation': 'Coeff of Variation'
}

real = df[df['class'] == 'Real']
synth = df[df['class'] == 'Synthetic']

# t-tests for each pause feature
print("Independent Samples T-Tests: Real vs Synthetic")

for col in pause_cols:
    label = display_names.get(col, col)
    r_vals = real[col].dropna()
    s_vals = synth[col].dropna()
    t_stat, p_val = stats.ttest_ind(r_vals, s_vals)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    pooled_std = np.sqrt((r_vals.std()**2 + s_vals.std()**2) / 2)
    d = (r_vals.mean() - s_vals.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  {label:22s}  t={t_stat:8.3f}  p={p_val:.2e}  {sig}  Cohen's d={d:.3f}")
    print(f"  {'':22s}  Real mean={r_vals.mean():.4f}  Synth mean={s_vals.mean():.4f}")

print("\n*** p<0.001  ** p<0.01  * p<0.05  ns = not significant")

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Biological Pause Pattern Distributions: Real vs Synthetic Speech',
             fontsize=16, fontweight='bold', y=0.98)

axes = axes.flatten()

for i, col in enumerate(pause_cols):
    ax = axes[i]
    label = display_names.get(col, col)

    r_vals = real[col].dropna()
    s_vals = synth[col].dropna()

    bp = ax.boxplot([r_vals, s_vals],
                    tick_labels=['Real', 'Synthetic'],
                    patch_artist=True, widths=0.6)

    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#e74c3c')
    bp['boxes'][1].set_alpha(0.7)

    _, p_val = stats.ttest_ind(r_vals, s_vals)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ax.set_title(f"{label}\n(p={p_val:.2e}, {sig})", fontsize=10)
    ax.tick_params(labelsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])

Path('results').mkdir(parents=True, exist_ok=True)
plt.savefig('results/pause_distributions.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: results/pause_distributions.png")
plt.close()

print("\nDone!")
