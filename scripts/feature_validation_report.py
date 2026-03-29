import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

df = pd.read_pickle('features/all_features_combined.pkl')
df['class'] = df['label'].apply(lambda x: 'Real' if x == 'real' else 'Synthetic')

feature_cols = [c for c in df.columns if c not in ['filename', 'label', 'class']]

real = df[df['class'] == 'Real']
synth = df[df['class'] == 'Synthetic']

results = []

for col in feature_cols:
    r_vals = real[col].dropna()
    s_vals = synth[col].dropna()
    t_stat, p_val = stats.ttest_ind(r_vals, s_vals)
    pooled_std = np.sqrt((r_vals.std()**2 + s_vals.std()**2) / 2)
    d = (r_vals.mean() - s_vals.mean()) / pooled_std if pooled_std > 0 else 0
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

    if col.startswith('mfcc'):
        cat = 'MFCC'
    elif col.startswith('cqcc'):
        cat = 'CQCC'
    elif col.startswith('rqa'):
        cat = 'RQA'
    elif col.startswith('entropy'):
        cat = 'Entropy'
    elif col.startswith('pause'):
        cat = 'Pauses'
    else:
        cat = 'Other'

    results.append({
        'feature': col,
        'category': cat,
        't_stat': t_stat,
        'p_value': p_val,
        'cohens_d': d,
        'significance': sig,
        'real_mean': r_vals.mean(),
        'synth_mean': s_vals.mean()
    })

res_df = pd.DataFrame(results)
# print(res_df.head())

Path('results').mkdir(parents=True, exist_ok=True)
report_path = 'results/feature_validation_report.txt'

with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("FEATURE VALIDATION REPORT — ALL 77 FEATURES\n")
    f.write("Independent Samples T-Tests: Real (n=500) vs Synthetic (n=500)\n")
    f.write("=" * 80 + "\n\n")

    f.write("-" * 80 + "\n")
    f.write("SUMMARY BY FEATURE CATEGORY\n")
    f.write("-" * 80 + "\n\n")

    for cat in ['MFCC', 'CQCC', 'RQA', 'Entropy', 'Pauses']:
        cat_df = res_df[res_df['category'] == cat]
        n_total = len(cat_df)
        n_sig = len(cat_df[cat_df['p_value'] < 0.05])
        n_high = len(cat_df[cat_df['p_value'] < 0.001])
        avg_d = cat_df['cohens_d'].abs().mean()
        max_d = cat_df.loc[cat_df['cohens_d'].abs().idxmax()]

        f.write(f"  {cat} ({n_total} features)\n")
        f.write(f"    Significant (p<0.05):  {n_sig}/{n_total}\n")
        f.write(f"    Highly sig (p<0.001):  {n_high}/{n_total}\n")
        f.write(f"    Mean |Cohen's d|:      {avg_d:.3f}\n")
        f.write(f"    Best discriminator:    {max_d['feature']} "
                f"(d={max_d['cohens_d']:.3f})\n\n")

    n_sig_total = len(res_df[res_df['p_value'] < 0.05])
    n_high_total = len(res_df[res_df['p_value'] < 0.001])
    f.write(f"  OVERALL: {n_sig_total}/77 significant (p<0.05), "
            f"{n_high_total}/77 highly significant (p<0.001)\n\n")

    f.write("-" * 80 + "\n")
    f.write("TOP 20 FEATURES BY EFFECT SIZE (|Cohen's d|)\n")
    f.write("-" * 80 + "\n\n")

    top20 = res_df.reindex(res_df['cohens_d'].abs().sort_values(ascending=False).index).head(20)
    f.write(f"  {'Rank':<5} {'Feature':<30} {'Category':<10} {'Cohen d':>9} "
            f"{'p-value':>12} {'Sig':>5}\n")
    f.write(f"  {'-'*5} {'-'*30} {'-'*10} {'-'*9} {'-'*12} {'-'*5}\n")

    for rank, (_, row) in enumerate(top20.iterrows(), 1):
        f.write(f"  {rank:<5} {row['feature']:<30} {row['category']:<10} "
                f"{row['cohens_d']:>9.3f} {row['p_value']:>12.2e} "
                f"{row['significance']:>5}\n")

    f.write("\n" + "-" * 80 + "\n")
    f.write("FULL RESULTS BY CATEGORY\n")
    f.write("-" * 80 + "\n")

    for cat in ['MFCC', 'CQCC', 'RQA', 'Entropy', 'Pauses']:
        cat_df = res_df[res_df['category'] == cat].copy()
        cat_df = cat_df.reindex(cat_df['cohens_d'].abs().sort_values(ascending=False).index)

        f.write(f"\n  {cat}\n")
        f.write(f"  {'Feature':<30} {'t-stat':>9} {'p-value':>12} {'Sig':>5} "
                f"{'Cohen d':>9} {'Real Mean':>11} {'Synth Mean':>11}\n")
        f.write(f"  {'-'*30} {'-'*9} {'-'*12} {'-'*5} {'-'*9} {'-'*11} {'-'*11}\n")

        for _, row in cat_df.iterrows():
            f.write(f"  {row['feature']:<30} {row['t_stat']:>9.3f} "
                    f"{row['p_value']:>12.2e} {row['significance']:>5} "
                    f"{row['cohens_d']:>9.3f} {row['real_mean']:>11.4f} "
                    f"{row['synth_mean']:>11.4f}\n")

    ns_df = res_df[res_df['p_value'] >= 0.05]
    f.write(f"\n" + "-" * 80 + "\n")
    f.write(f"NON-SIGNIFICANT FEATURES (p >= 0.05): {len(ns_df)}/77\n")
    f.write("-" * 80 + "\n\n")

    if len(ns_df) > 0:
        for _, row in ns_df.iterrows():
            f.write(f"  {row['feature']:<30} {row['category']:<10} "
                    f"p={row['p_value']:.3f}  d={row['cohens_d']:.3f}\n")
    else:
        f.write("  None — all features are significant!\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("*** p<0.001  ** p<0.01  * p<0.05  ns = not significant\n")
    f.write("Cohen's d: |0.2| small, |0.5| medium, |0.8| large\n")
    f.write("=" * 80 + "\n")

print(f"Report saved to: {report_path}")

print(f"\nQUICK SUMMARY")
print(f"Total features:           77")
print(f"Significant (p<0.05):     {n_sig_total}/77")
print(f"Highly significant:       {n_high_total}/77")
print(f"Non-significant:          {77 - n_sig_total}/77")
print(f"\nTop 5 by effect size:")
for rank, (_, row) in enumerate(top20.head(5).iterrows(), 1):
    print(f"  {rank}. {row['feature']} ({row['category']}) — d={row['cohens_d']:.3f}")
