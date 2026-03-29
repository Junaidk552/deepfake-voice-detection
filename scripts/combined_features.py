import pandas as pd
import numpy as np
from pathlib import Path


def combine_features():
    print("Combining all feature sets...")

    mfcc_df = pd.read_pickle('features/mfcc_features.pkl')
    print(f"  MFCC:    {mfcc_df.shape[0]} samples, "
          f"{mfcc_df.shape[1] - 2} features")

    cqcc_df = pd.read_pickle('features/cqcc_features.pkl')
    print(f"  CQCC:    {cqcc_df.shape[0]} samples, "
          f"{cqcc_df.shape[1] - 2} features")

    rqa_df = pd.read_pickle('features/rqa_features.pkl')
    print(f"  RQA:     {rqa_df.shape[0]} samples, "
          f"{rqa_df.shape[1] - 2} features")

    entropy_df = pd.read_pickle('features/entropy_features.pkl')
    print(f"  Entropy: {entropy_df.shape[0]} samples, "
          f"{entropy_df.shape[1] - 2} features")

    pause_df = pd.read_pickle('features/pause_features.pkl')
    print(f"  Pauses:  {pause_df.shape[0]} samples, "
          f"{pause_df.shape[1] - 2} features")

    print("\nMerging...")
    df2 = mfcc_df.merge(cqcc_df, on=['filename', 'label'], how='inner')
    df2 = df2.merge(rqa_df, on=['filename', 'label'], how='inner')
    df2 = df2.merge(entropy_df, on=['filename', 'label'], how='inner')
    df2 = df2.merge(pause_df, on=['filename', 'label'], how='inner')

    feature_cols = [c for c in df2.columns
                    if c not in ['filename', 'label']]

    print(f"\nCombined dataset:")
    print(f"  Samples: {len(df2)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Columns: {df2.shape[1]} (features + filename + label)")

    nan_count = df2[feature_cols].isna().sum().sum()
    nan_feats = df2[feature_cols].isna().sum()
    nan_feats = nan_feats[nan_feats > 0]

    if nan_count > 0:
        print(f"\n  NaN values found: {nan_count}")
        for col, count in nan_feats.items():
            print(f"    {col}: {count} NaN values")
        print("  Filling NaN with column median...")
        df2[feature_cols] = df2[feature_cols].fillna(
            df2[feature_cols].median()
        )
    else:
        print(f"\n  No NaN values found.")

    print(f"\nFeature breakdown:")
    mfcc_feats = [c for c in feature_cols if c.startswith('mfcc')]
    cqcc_feats = [c for c in feature_cols if c.startswith('cqcc')]
    rqa_feats = [c for c in feature_cols if c.startswith('rqa')]
    entropy_feats = [c for c in feature_cols if c.startswith('entropy')]
    pause_feats = [c for c in feature_cols if c.startswith('pause')]

    print(f"  MFCC:    {len(mfcc_feats)}")
    print(f"  CQCC:    {len(cqcc_feats)}")
    print(f"  RQA:     {len(rqa_feats)}")
    print(f"  Entropy: {len(entropy_feats)}")
    print(f"  Pauses:  {len(pause_feats)}")
    print(f"  Total:   {len(feature_cols)}")

    print(f"\nLabel distribution:")
    print(df2['label'].value_counts().to_string())
    # print(df2.head())

    output_path = Path('features/all_features_combined.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df2.to_pickle(output_path)
    print(f"\nSaved to: {output_path}")

    return df2


if __name__ == "__main__":
    combined = combine_features()
