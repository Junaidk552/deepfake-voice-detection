"""
extract_rqa.py

This script extracts data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- *.wav
Outputs:
- features/rqa_features.pkl
Reproduces: Reproduces intermediate outputs used by other scripts.
"""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Try to import pyrqa - fall back to manual implementation if unavailable
try:
    from pyrqa.time_series import TimeSeries
    from pyrqa.settings import Settings
    from pyrqa.analysis_type import Classic
    from pyrqa.neighbourhood import FixedRadius
    from pyrqa.metric import EuclideanMetric
    from pyrqa.computation import RQAComputation
    PYRQA_AVAILABLE = True
except ImportError:
    PYRQA_AVAILABLE = False


def downsample_signal(y, target_length=2000):
    """downsample to target_length. RQA is O(n^2) so this is necessary."""
    if len(y) <= target_length:
        return y
    indices = np.linspace(0, len(y) - 1, target_length, dtype=int)
    return y[indices]


def compute_rqa_manual(y, m=10, tau=1, epsilon_percent=10):
    """manual RQA using numpy - used when pyrqa not available"""
    y = downsample_signal(y, target_length=2000)

    N = len(y) - (m - 1) * tau
    if N < 10:
        return None

    embedded = np.array([y[i:i + m * tau:tau] for i in range(N)])

    dist_matrix = cdist(embedded, embedded, metric='euclidean')
    epsilon = epsilon_percent / 100.0 * np.std(y)
    recurrence_matrix = (dist_matrix <= epsilon).astype(int)

    n = recurrence_matrix.shape[0]
    total_points = n * n
    rec_points = np.sum(recurrence_matrix)

    recurrence_rate = rec_points / total_points

    diagonal_lines = []
    for k in range(-n + 1, n):
        diag = np.diag(recurrence_matrix, k)
        line_length = 0
        for val in diag:
            if val == 1:
                line_length += 1
            else:
                if line_length >= 2:
                    diagonal_lines.append(line_length)
                line_length = 0
        if line_length >= 2:
            diagonal_lines.append(line_length)

    if len(diagonal_lines) == 0:
        diagonal_lines = [0]

    det_points = sum(diagonal_lines)
    determinism = det_points / rec_points if rec_points > 0 else 0

    avg_diag = np.mean(diagonal_lines) if diagonal_lines else 0
    max_diag = max(diagonal_lines) if diagonal_lines else 0

    if len(diagonal_lines) > 0 and sum(diagonal_lines) > 0:
        hist, _ = np.histogram(diagonal_lines, bins=max(1, max(diagonal_lines)))
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        entropy_diag = -np.sum(probs * np.log(probs + 1e-10))
    else:
        entropy_diag = 0

    vertical_lines = []
    for col in range(n):
        line_length = 0
        for row in range(n):
            if recurrence_matrix[row, col] == 1:
                line_length += 1
            else:
                if line_length >= 2:
                    vertical_lines.append(line_length)
                line_length = 0
        if line_length >= 2:
            vertical_lines.append(line_length)

    if len(vertical_lines) == 0:
        vertical_lines = [0]

    lam_points = sum(vertical_lines)
    laminarity = lam_points / rec_points if rec_points > 0 else 0
    trapping_time = np.mean(vertical_lines) if vertical_lines else 0

    return {
        'rqa_recurrence_rate': recurrence_rate,
        'rqa_determinism': determinism,
        'rqa_avg_diagonal': avg_diag,
        'rqa_max_diagonal': max_diag,
        'rqa_entropy_diagonal': entropy_diag,
        'rqa_laminarity': laminarity,
        'rqa_trapping_time': trapping_time
    }


def compute_rqa_pyrqa(y, m=10, tau=1, epsilon_percent=10):
    """RQA via pyrqa (faster). Falls back to manual if pyrqa errors."""
    y = downsample_signal(y, target_length=2000)
    epsilon = epsilon_percent / 100.0 * np.std(y)

    try:
        time_series = TimeSeries(y.tolist(), embedding_dimension=m, time_delay=tau)
        settings = Settings(
            time_series,
            analysis_type=Classic,
            neighbourhood=FixedRadius(epsilon),
            similarity_measure=EuclideanMetric,
            theiler_corrector=1
        )
        computation = RQAComputation.create(settings)
        result = computation.run()

        return {
            'rqa_recurrence_rate': result.recurrence_rate,
            'rqa_determinism': result.determinism,
            'rqa_avg_diagonal': result.average_diagonal_line,
            'rqa_max_diagonal': result.longest_diagonal_line,
            'rqa_entropy_diagonal': result.entropy_diagonal_lines,
            'rqa_laminarity': result.laminarity,
            'rqa_trapping_time': result.trapping_time
        }
    except Exception:
        return compute_rqa_manual(y, m, tau, epsilon_percent)


def extract_rqa_features(audio_path, sr=16000):
    """load audio and extract RQA features"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=10)
        y = librosa.effects.preemphasis(y, coef=0.97)

        if PYRQA_AVAILABLE:
            return compute_rqa_pyrqa(y)
        else:
            return compute_rqa_manual(y)

    except Exception as e:
        print(f"  [ERROR] Failed to process {audio_path}: {e}")
        return None


def process_dataset(dataset_dir, output_file):
    """process all audio files and save RQA features"""
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

    if PYRQA_AVAILABLE:
        print("\nUsing pyrqa library")
    else:
        print("\npyrqa not available — using manual numpy implementation (slower)")

    print(f"\nThis will take a while — RQA is computationally expensive.")
    print(f"Estimated: 10-30 minutes for {len(audio_files)} samples.")
    print(f"Each sample is downsampled to 2000 points before computation.\n")

    feature_list = []
    filenames = []
    valid_labels = []

    for audio_file, label in tqdm(
        zip(audio_files, labels),
        total=len(audio_files),
        desc="Extracting RQA"
    ):
        features = extract_rqa_features(audio_file)

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
        if col.startswith('rqa_'):
            print(f"  {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}")

    print(f"\nLabel distribution:")
    print(df['label'].value_counts().to_string())

    return df


if __name__ == "__main__":
    dataset_dir = "dataset"
    output_file = "features/rqa_features.pkl"

    df = process_dataset(dataset_dir, output_file)

    # t-tests for each feature
    print("\nStatistical Validation: Real vs Synthetic")

    real_mask = df['label'] == 'real'
    synth_mask = df['label'] != 'real'

    print(f"\n{'Feature':<30} {'Real Mean':>10} {'Synth Mean':>12} {'p-value':>10} {'Sig':>5}")
    print("-" * 70)

    for col in df.columns:
        if col.startswith('rqa_'):
            real_vals = df.loc[real_mask, col].dropna()
            synth_vals = df.loc[synth_mask, col].dropna()

            if len(real_vals) > 0 and len(synth_vals) > 0:
                t_stat, p_val = stats.ttest_ind(real_vals, synth_vals)
                sig = "YES" if p_val < 0.05 else "NO"
                print(f"  {col:<28} {real_vals.mean():>10.6f} {synth_vals.mean():>12.6f} {p_val:>10.6f} {sig:>5}")

    print("\nRQA extraction complete. Ready for entropy extraction next.")
