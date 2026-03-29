import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False
    print("nolds not installed. Run: pip install nolds")
    exit(1)


def coarse_grain(signal, scale):
    """coarse-grain signal at given scale. scale=1 returns original."""
    if scale == 1:
        return signal

    n = len(signal)
    trimmed_length = n - (n % scale)
    trimmed = signal[:trimmed_length]
    reshaped = trimmed.reshape(-1, scale)
    return reshaped.mean(axis=1)


def sample_entropy(signal, m=2, r_factor=0.2):
    """sample entropy - standard params m=2, r=0.2*std"""
    try:
        r = r_factor * np.std(signal)
        if r == 0 or len(signal) < 20:
            return np.nan
        return nolds.sampen(signal, emb_dim=m, tolerance=r)
    except Exception:
        return np.nan


def extract_multiscale_entropy(audio_path, sr=16000, max_scale=10):
    """extract MSE at scales 1 to max_scale from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=10)
        y = librosa.effects.preemphasis(y, coef=0.97)

        # downsample to keep computation tractable
        if len(y) > 16000:
            indices = np.linspace(0, len(y) - 1, 16000, dtype=int)
            y = y[indices]

        entropy_values = {}
        for scale in range(1, max_scale + 1):
            coarsened = coarse_grain(y, scale)

            if len(coarsened) < 30:
                entropy_values[f'entropy_scale_{scale}'] = np.nan
            else:
                entropy_values[f'entropy_scale_{scale}'] = sample_entropy(coarsened)

        return entropy_values

    except Exception as e:
        print(f"  [ERROR] Failed to process {audio_path}: {e}")
        return None


def process_dataset(dataset_dir, output_file):
    """process all audio files and save entropy features"""
    dataset_path = Path(dataset_dir)

    audio_files = []
    labels = []

    real_dir = dataset_path / 'real'
    if real_dir.exists():
        for audio_file in sorted(real_dir.glob('*.wav')):
            audio_files.append(audio_file)
            labels.append('real')

    synthetic_dir = dataset_path / 'synthetic'
    if synthetic_dir.exists():
        for subdir in sorted(synthetic_dir.iterdir()):
            if subdir.is_dir():
                for audio_file in sorted(subdir.glob('*.wav')):
                    audio_files.append(audio_file)
                    labels.append(f'synthetic_{subdir.name}')

    print(f"Found {len(audio_files)} audio files")
    print(f"  Real: {labels.count('real')}")
    print(f"  Synthetic: {len(labels) - labels.count('real')}")
    print(f"\nParameters: m=2, r=0.2*std, scales 1-10")
    print(f"Estimated time: 5-15 minutes\n")

    feature_list = []
    filenames = []
    valid_labels = []

    for audio_file, label in tqdm(
        zip(audio_files, labels),
        total=len(audio_files),
        desc="Extracting entropy"
    ):
        features = extract_multiscale_entropy(audio_file)

        if features is not None:
            feature_list.append(features)
            filenames.append(audio_file.name)
            valid_labels.append(label)

    df = pd.DataFrame(feature_list)
    df['filename'] = filenames
    df['label'] = valid_labels

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)

    print(f"\nSuccessfully processed {len(df)}/{len(audio_files)} files")
    print(f"Saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFeature summary:")
    for col in df.columns:
        if col.startswith('entropy_'):
            valid = df[col].dropna()
            print(f"  {col}: mean={valid.mean():.6f}, std={valid.std():.6f}, "
                  f"nan_count={df[col].isna().sum()}")

    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())

    return df


if __name__ == "__main__":
    dataset_dir = "dataset"
    output_file = "features/entropy_features.pkl"

    df = process_dataset(dataset_dir, output_file)

    print("\nStatistical Validation: Real vs Synthetic")

    real_mask = df['label'] == 'real'
    synth_mask = df['label'] != 'real'

    print(f"\n{'Feature':<25} {'Real Mean':>10} {'Synth Mean':>12} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)

    for col in df.columns:
        if col.startswith('entropy_'):
            real_vals = df.loc[real_mask, col].dropna()
            synth_vals = df.loc[synth_mask, col].dropna()

            if len(real_vals) > 0 and len(synth_vals) > 0:
                t_stat, p_val = stats.ttest_ind(real_vals, synth_vals)
                sig = "YES" if p_val < 0.05 else "NO"
                print(f"  {col:<23} {real_vals.mean():>10.6f} {synth_vals.mean():>12.6f} "
                      f"{p_val:>10.6f} {sig:>5}")

    print("\nEntropy extraction complete. Ready for pause extraction next.")
