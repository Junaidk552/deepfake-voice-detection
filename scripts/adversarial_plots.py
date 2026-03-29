"""plot adversarial robustness results"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

res = pd.read_csv('results/adversarial_results.csv')
Path('results').mkdir(parents=True, exist_ok=True)

attacks = res['attack'].unique()
x = np.arange(len(attacks))
width = 0.35

mfcc_acc = [res[(res['attack'] == a) & (res['config'] == 'MFCC only')]['accuracy'].values[0]
            for a in attacks]
all_acc = [res[(res['attack'] == a) & (res['config'] == 'All features')]['accuracy'].values[0]
           for a in attacks]

fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - width/2, mfcc_acc, width, label='MFCC only', color='#3498db')
ax.bar(x + width/2, all_acc, width, label='All features', color='#e74c3c')

ax.set_xlabel('Attack Type', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Detection Accuracy Under Adversarial Attacks', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(attacks, rotation=35, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/adversarial_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/adversarial_accuracy.png")

# degradation plot
fig, ax = plt.subplots(figsize=(12, 6))

mfcc_clean = res[(res['attack'] == 'Clean (baseline)') &
                 (res['config'] == 'MFCC only')]['accuracy'].values[0]
all_clean = res[(res['attack'] == 'Clean (baseline)') &
                (res['config'] == 'All features')]['accuracy'].values[0]

attack_names = [a for a in attacks if a != 'Clean (baseline)']
x = np.arange(len(attack_names))

mfcc_drop = [mfcc_clean - res[(res['attack'] == a) &
             (res['config'] == 'MFCC only')]['accuracy'].values[0]
             for a in attack_names]
all_drop = [all_clean - res[(res['attack'] == a) &
            (res['config'] == 'All features')]['accuracy'].values[0]
            for a in attack_names]

ax.bar(x - width/2, mfcc_drop, width, label='MFCC only', color='#3498db')
ax.bar(x + width/2, all_drop, width, label='All features', color='#e74c3c')

ax.set_xlabel('Attack Type', fontsize=12)
ax.set_ylabel('Accuracy Drop from Baseline', fontsize=12)
ax.set_title('Performance Degradation Under Adversarial Attacks', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(attack_names, rotation=35, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/adversarial_degradation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/adversarial_degradation.png")

# SNR curve
fig, ax = plt.subplots(figsize=(8, 6))

snr_attacks = ['Clean (baseline)', 'Noise SNR=30dB', 'Noise SNR=20dB',
               'Noise SNR=10dB']
snr_labels = ['Clean', '30dB', '20dB', '10dB']

snr_attacks_present = [a for a in snr_attacks if a in res['attack'].values]
snr_labels_present = [snr_labels[snr_attacks.index(a)]
                      for a in snr_attacks_present]

if len(snr_attacks_present) > 1:
    mfcc_snr = [res[(res['attack'] == a) &
                (res['config'] == 'MFCC only')]['accuracy'].values[0]
                for a in snr_attacks_present]
    all_snr = [res[(res['attack'] == a) &
               (res['config'] == 'All features')]['accuracy'].values[0]
               for a in snr_attacks_present]

    ax.plot(snr_labels_present, mfcc_snr, 'o-', color='#3498db',
            linewidth=2, markersize=8, label='MFCC only')
    ax.plot(snr_labels_present, all_snr, 's-', color='#e74c3c',
            linewidth=2, markersize=8, label='All features')

    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Detection Accuracy vs Noise Level', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/adversarial_snr_curve.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("Saved results/adversarial_snr_curve.png")
else:
    print("Not enough SNR data for curve plot")

# heatmap
fig, ax = plt.subplots(figsize=(12, 5))

attack_names_all = res['attack'].unique()
configs = res['config'].unique()

heatmap_data = np.zeros((len(configs), len(attack_names_all)))
for i, config in enumerate(configs):
    for j, attack in enumerate(attack_names_all):
        row = res[(res['config'] == config) & (res['attack'] == attack)]
        if len(row) > 0:
            heatmap_data[i, j] = row['accuracy'].values[0]

im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto',
               vmin=0.5, vmax=1.0)

ax.set_xticks(range(len(attack_names_all)))
ax.set_xticklabels(attack_names_all, rotation=35, ha='right', fontsize=9)
ax.set_yticks(range(len(configs)))
ax.set_yticklabels(configs, fontsize=10)

for i in range(len(configs)):
    for j in range(len(attack_names_all)):
        ax.text(j, i, f'{heatmap_data[i,j]:.2f}',
                ha='center', va='center', fontsize=9,
                color='black' if heatmap_data[i,j] > 0.7 else 'white')

ax.set_title('Detection Accuracy Heatmap: Config x Attack', fontsize=14)
plt.colorbar(im, ax=ax, label='Accuracy')
plt.tight_layout()
plt.savefig('results/adversarial_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved results/adversarial_heatmap.png")

print("\nAll adversarial plots done!")
