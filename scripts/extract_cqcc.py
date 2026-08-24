"""
extract_cqcc.py

This script extracts data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- *.wav
Outputs:
- features/cqcc_features.pkl
Reproduces: Reproduces intermediate outputs used by other scripts.
"""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from spafe.features.cqcc import cqcc
except ImportError:
    print("spafe not installed!")
    print("Run: pip install spafe --break-system-packages")
    exit(1)

def extract_cqcc_features(audio_path, sr=16000, num_ceps=13):
    """extract CQCC features, returns 13-dim mean vector"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=10)

        cqccs = cqcc(y, fs=sr, num_ceps=num_ceps)

        features_mean = np.mean(cqccs, axis=0)

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

    print("Extracting CQCC features...")
    for audio_file, label in tqdm(zip(audio_files, labels), total=len(audio_files),
                                   desc="Processing", unit="file"):
        features = extract_cqcc_features(audio_file)

        if features is not None:
            feature_list.append(features)
            filenames.append(audio_file.name)
            valid_labels.append(label)

    print(f"\nSuccessfully processed {len(feature_list)}/{total} files")

    feature_array = np.array(feature_list)

    cqcc_cols = [f'cqcc_{i}' for i in range(13)]

    df = pd.DataFrame(feature_array, columns=cqcc_cols)
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

    print("\nFeature Statistics (first 5 CQCCs):")
    print(df[['cqcc_0', 'cqcc_1', 'cqcc_2', 'cqcc_3', 'cqcc_4']].describe())

    return df

if __name__ == "__main__":
    dataset_dir = "dataset"
    output_file = "features/cqcc_features.pkl"

    print("CQCC Feature Extraction - Deepfake Voice Detection\n")

    df = process_dataset(dataset_dir, output_file)

    print("\nCQCC extraction complete!")
