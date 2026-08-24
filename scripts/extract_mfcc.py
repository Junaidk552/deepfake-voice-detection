"""
extract_mfcc.py

This script extracts data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- *.wav
Outputs:
- features/mfcc_features.pkl
Reproduces: Reproduces intermediate outputs used by other scripts.
"""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_mfcc_features(audio_path, sr=16000, n_mfcc=13):
    """extract MFCCs + deltas + delta-deltas, returns 39-dim mean vector"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=10)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # first and second derivatives
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # 13 + 13 + 13 = 39
        features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        features_mean = np.mean(features, axis=1)

        return features_mean

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_dataset(dataset_dir, output_file):
    dataset_path = Path(dataset_dir)

    audio_files = []
    labels = []

    print("Scanning dataset...")

    real_dir = dataset_path / 'real'
    if real_dir.exists():
        real_files = list(real_dir.glob('*.wav'))
        audio_files.extend(real_files)
        labels.extend(['real'] * len(real_files))
        print(f"   Found {len(real_files)} real samples")

    synthetic_dir = dataset_path / 'synthetic'
    if synthetic_dir.exists():
        for platform_dir in synthetic_dir.iterdir():
            if platform_dir.is_dir():
                platform_files = list(platform_dir.glob('*.wav'))
                audio_files.extend(platform_files)
                labels.extend([f'synthetic_{platform_dir.name}'] * len(platform_files))
                print(f"   Found {len(platform_files)} {platform_dir.name} samples")

    total = len(audio_files)
    print(f"\nTotal samples to process: {total}")
    print(f"   Real: {labels.count('real')}")
    print(f"   Synthetic: {total - labels.count('real')}")
    print()

    feature_list = []
    filenames = []
    valid_labels = []

    print("Extracting MFCC features...")
    for audio_file, label in tqdm(zip(audio_files, labels), total=len(audio_files),
                                   desc="Processing", unit="file"):
        features = extract_mfcc_features(audio_file)

        if features is not None:
            feature_list.append(features)
            filenames.append(audio_file.name)
            valid_labels.append(label)

    print(f"\nSuccessfully processed {len(feature_list)}/{total} files")

    feature_array = np.array(feature_list)

    mfcc_cols = [f'mfcc_{i}' for i in range(13)]
    delta_cols = [f'mfcc_delta_{i}' for i in range(13)]
    delta2_cols = [f'mfcc_delta2_{i}' for i in range(13)]
    all_cols = mfcc_cols + delta_cols + delta2_cols

    df = pd.DataFrame(feature_array, columns=all_cols)
    df['filename'] = filenames
    df['label'] = valid_labels

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_pickle(output_path)

    print(f"\nSaved features to: {output_path}")
    print(f"   Shape: {feature_array.shape}")
    print(f"   Columns: {len(df.columns)}")

    print("\nLabel Distribution:")
    print(df['label'].value_counts())

    print("\nFeature Statistics (first 5 MFCCs):")
    print(df[['mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4']].describe())

    return df

if __name__ == "__main__":
    dataset_dir = "dataset"
    output_file = "features/mfcc_features.pkl"

    print("MFCC Feature Extraction - Deepfake Voice Detection\n")

    df = process_dataset(dataset_dir, output_file)

    print("\nMFCC extraction complete!")
