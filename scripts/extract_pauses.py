"""
extract_pauses.py

This script extracts data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- *.wav
Outputs:
- features/pauses_features.pkl
Reproduces: Reproduces intermediate outputs used by other scripts.
"""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')


def detect_pauses(y, sr, frame_length=0.02, threshold_percentile=20,
                  min_pause_duration=0.05):
    """detect pauses using RMS energy thresholding"""
    frame_samples = int(frame_length * sr)
    hop_samples = frame_samples // 2  # 50% overlap

    rms = librosa.feature.rms(y=y, frame_length=frame_samples,
                               hop_length=hop_samples)[0]

    threshold = np.percentile(rms, threshold_percentile)
    is_silent = rms < threshold

    pause_durations = []
    current_pause_frames = 0

    for silent in is_silent:
        if silent:
            current_pause_frames += 1
        else:
            if current_pause_frames > 0:
                duration = current_pause_frames * (hop_samples / sr)
                if duration >= min_pause_duration:
                    pause_durations.append(duration)
                current_pause_frames = 0

    if current_pause_frames > 0:
        duration = current_pause_frames * (hop_samples / sr)
        if duration >= min_pause_duration:
            pause_durations.append(duration)

    return pause_durations


def extract_pause_features(audio_path, sr=16000):
    """extract 8 pause stats from an audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=10)

        total_duration = len(y) / sr

        pause_durations = detect_pauses(y, sr)

        if len(pause_durations) < 2:
            return {
                'pause_mean_duration': pause_durations[0] if len(pause_durations) == 1 else 0.0,
                'pause_std_duration': 0.0,
                'pause_rate': len(pause_durations) / total_duration,
                'pause_median_duration': pause_durations[0] if len(pause_durations) == 1 else 0.0,
                'pause_range': 0.0,
                'pause_skewness': 0.0,
                'pause_kurtosis': 0.0,
                'pause_coeff_variation': 0.0
            }

        pauses = np.array(pause_durations)
        mean_dur = np.mean(pauses)

        return {
            'pause_mean_duration': mean_dur,
            'pause_std_duration': np.std(pauses),
            'pause_rate': len(pauses) / total_duration,
            'pause_median_duration': np.median(pauses),
            'pause_range': np.max(pauses) - np.min(pauses),
            'pause_skewness': scipy_stats.skew(pauses),
            'pause_kurtosis': scipy_stats.kurtosis(pauses),
            'pause_coeff_variation': np.std(pauses) / mean_dur if mean_dur > 0 else 0.0
        }

    except Exception as e:
        print(f"  [ERROR] Failed to process {audio_path}: {e}")
        return None


def process_dataset(dataset_dir, output_file):
    """process all audio files and extract pause features"""
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
    print(f"\nPause detection: RMS energy, 20ms frames, 20th percentile threshold")
    print(f"Minimum pause duration: 50ms\n")

    feature_list = []
    filenames = []
    valid_labels = []

    for audio_file, label in tqdm(
        zip(audio_files, labels),
        total=len(audio_files),
        desc="Extracting pauses"
    ):
        features = extract_pause_features(audio_file)

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

    zero_pause = (df['pause_mean_duration'] == 0).sum()
    print(f"\nSamples with no detected pauses: {zero_pause}/{len(df)}")

    print(f"\nFeature summary:")
    for col in df.columns:
        if col.startswith('pause_'):
            print(f"  {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}")

    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())

    return df


if __name__ == "__main__":
    dataset_dir = "dataset"
    output_file = "features/pause_features.pkl"

    df = process_dataset(dataset_dir, output_file)

    print("\nStatistical Validation: Real vs Synthetic")

    real_mask = df['label'] == 'real'
    synth_mask = df['label'] != 'real'

    print(f"\n{'Feature':<30} {'Real Mean':>10} {'Synth Mean':>12} {'p-value':>10} {'Sig':>5}")
    print("-" * 70)

    for col in df.columns:
        if col.startswith('pause_'):
            real_vals = df.loc[real_mask, col].dropna()
            synth_vals = df.loc[synth_mask, col].dropna()

            if len(real_vals) > 0 and len(synth_vals) > 0:
                t_stat, p_val = scipy_stats.ttest_ind(real_vals, synth_vals)
                sig = "YES" if p_val < 0.05 else "NO"
                print(f"  {col:<28} {real_vals.mean():>10.6f} {synth_vals.mean():>12.6f} "
                      f"{p_val:>10.6f} {sig:>5}")

    print("\nPause extraction complete. Ready for feature combination next.")
