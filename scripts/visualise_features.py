import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def load_features():
    mfcc_df = pd.read_pickle('features/mfcc_features.pkl')
    cqcc_df = pd.read_pickle('features/cqcc_features.pkl')
    return mfcc_df, cqcc_df

def create_visualisations(mfcc_df, cqcc_df, output_dir='results'):
    Path(output_dir).mkdir(exist_ok=True)

    mfcc_df['is_real'] = mfcc_df['label'] == 'real'
    cqcc_df['is_real'] = cqcc_df['label'] == 'real'

    print("Creating visualisations...")
    print()

    print("1. MFCC distributions (first 6 coefficients)...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('MFCC Feature Distributions: Real vs Synthetic Speech',
                 fontsize=16, fontweight='bold')

    for idx, ax in enumerate(axes.flat):
        feature = f'mfcc_{idx}'

        ax.hist(mfcc_df[mfcc_df['is_real']][feature],
                bins=50, alpha=0.6, label='Real', color='green', density=True)
        ax.hist(mfcc_df[~mfcc_df['is_real']][feature],
                bins=50, alpha=0.6, label='Synthetic', color='red', density=True)

        ax.set_xlabel(f'{feature.upper()} Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{feature.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mfcc_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/mfcc_distributions.png")
    plt.close()

    print("2. CQCC distributions (first 6 coefficients)...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CQCC Feature Distributions: Real vs Synthetic Speech',
                 fontsize=16, fontweight='bold')

    for idx, ax in enumerate(axes.flat):
        feature = f'cqcc_{idx}'

        ax.hist(cqcc_df[cqcc_df['is_real']][feature],
                bins=50, alpha=0.6, label='Real', color='green', density=True)
        ax.hist(cqcc_df[~cqcc_df['is_real']][feature],
                bins=50, alpha=0.6, label='Synthetic', color='red', density=True)

        ax.set_xlabel(f'{feature.upper()} Value')
        ax.set_ylabel('Density')
        ax.set_title(f'{feature.upper()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cqcc_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/cqcc_distributions.png")
    plt.close()

    print("3. MFCC box plots...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    mfcc_features = [f'mfcc_{i}' for i in range(13)]
    real_data = mfcc_df[mfcc_df['is_real']][mfcc_features].values
    synthetic_data = mfcc_df[~mfcc_df['is_real']][mfcc_features].values

    positions = np.arange(13) * 3
    bp1 = ax.boxplot(real_data, positions=positions - 0.6, widths=0.5,
                     patch_artist=True, tick_labels=mfcc_features)
    bp2 = ax.boxplot(synthetic_data, positions=positions + 0.6, widths=0.5,
                     patch_artist=True, tick_labels=mfcc_features)

    for patch in bp1['boxes']:
        patch.set_facecolor('green')
        patch.set_alpha(0.6)
    for patch in bp2['boxes']:
        patch.set_facecolor('red')
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(mfcc_features, rotation=45)
    ax.set_xlabel('MFCC Coefficient')
    ax.set_ylabel('Feature Value')
    ax.set_title('MFCC Feature Comparison: Real (Green) vs Synthetic (Red)',
                 fontsize=14, fontweight='bold')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Real', 'Synthetic'], loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mfcc_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/mfcc_boxplots.png")
    plt.close()

    print("4. CQCC box plots...")

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    cqcc_features = [f'cqcc_{i}' for i in range(13)]
    real_data = cqcc_df[cqcc_df['is_real']][cqcc_features].values
    synthetic_data = cqcc_df[~cqcc_df['is_real']][cqcc_features].values

    positions = np.arange(13) * 3
    bp1 = ax.boxplot(real_data, positions=positions - 0.6, widths=0.5,
                     patch_artist=True, tick_labels=cqcc_features)
    bp2 = ax.boxplot(synthetic_data, positions=positions + 0.6, widths=0.5,
                     patch_artist=True, tick_labels=cqcc_features)

    for patch in bp1['boxes']:
        patch.set_facecolor('green')
        patch.set_alpha(0.6)
    for patch in bp2['boxes']:
        patch.set_facecolor('red')
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(cqcc_features, rotation=45)
    ax.set_xlabel('CQCC Coefficient')
    ax.set_ylabel('Feature Value')
    ax.set_title('CQCC Feature Comparison: Real (Green) vs Synthetic (Red)',
                 fontsize=14, fontweight='bold')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Real', 'Synthetic'], loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cqcc_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/cqcc_boxplots.png")
    plt.close()

    print("5. 2D scatter plot (MFCC_0 vs MFCC_1)...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.scatter(mfcc_df[mfcc_df['is_real']]['mfcc_0'],
               mfcc_df[mfcc_df['is_real']]['mfcc_1'],
               alpha=0.5, s=20, c='green', label='Real')
    ax.scatter(mfcc_df[~mfcc_df['is_real']]['mfcc_0'],
               mfcc_df[~mfcc_df['is_real']]['mfcc_1'],
               alpha=0.5, s=20, c='red', label='Synthetic')

    ax.set_xlabel('MFCC_0', fontsize=12)
    ax.set_ylabel('MFCC_1', fontsize=12)
    ax.set_title('MFCC Feature Space: Real vs Synthetic Speech',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mfcc_scatter.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_dir}/mfcc_scatter.png")
    plt.close()

    print()
    print("Done! Check the results/ folder.")

def print_statistics(mfcc_df, cqcc_df):
    from scipy import stats

    print("\nSTATISTICAL ANALYSIS\n")

    mfcc_df['is_real'] = mfcc_df['label'] == 'real'
    cqcc_df['is_real'] = cqcc_df['label'] == 'real'

    print("T-test results (MFCC features):")
    for i in range(3):
        feature = f'mfcc_{i}'
        real = mfcc_df[mfcc_df['is_real']][feature]
        synthetic = mfcc_df[~mfcc_df['is_real']][feature]

        t_stat, p_value = stats.ttest_ind(real, synthetic)

        significance = "SIGNIFICANT" if p_value < 0.05 else "Not significant"
        print(f"   {feature.upper()}: p-value = {p_value:.6f} -- {significance}")

    print("\nT-test results (CQCC features):")
    for i in range(3):
        feature = f'cqcc_{i}'
        real = cqcc_df[cqcc_df['is_real']][feature]
        synthetic = cqcc_df[~cqcc_df['is_real']][feature]

        t_stat, p_value = stats.ttest_ind(real, synthetic)
        # print(f't={t_stat:.3f}')

        significance = "SIGNIFICANT" if p_value < 0.05 else "Not significant"
        print(f"   {feature.upper()}: p-value = {p_value:.6f} -- {significance}")

    print("\nIf p-value < 0.05, features significantly differ between real and synthetic!")

if __name__ == "__main__":
    print("Feature Visualisation - Deepfake Voice Detection\n")

    mfcc_df, cqcc_df = load_features()

    print(f"Loaded MFCC features: {mfcc_df.shape}")
    print(f"Loaded CQCC features: {cqcc_df.shape}")
    print()

    create_visualisations(mfcc_df, cqcc_df)
    print_statistics(mfcc_df, cqcc_df)
