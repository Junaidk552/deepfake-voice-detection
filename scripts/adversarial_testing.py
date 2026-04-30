import pandas as pd
import numpy as np
import pickle
import librosa
import soundfile as sf
import subprocess
import tempfile
import os
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from feature_utils import extract_all_from_audio as extract_all

warnings.filterwarnings('ignore')


def add_gaussian_noise(y, snr_db):
    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise


def mp3_compress(y, sr, bitrate='64k'):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        sf.write(tmp_wav.name, y, sr)
        tmp_mp3 = tmp_wav.name.replace('.wav', '.mp3')
        tmp_out = tmp_wav.name.replace('.wav', '_out.wav')

        subprocess.run(['ffmpeg', '-y', '-i', tmp_wav.name, '-b:a', bitrate,
                        tmp_mp3], capture_output=True)
        subprocess.run(['ffmpeg', '-y', '-i', tmp_mp3, '-ar', str(sr),
                        tmp_out], capture_output=True)

        y_out, _ = librosa.load(tmp_out, sr=sr)

        for f in [tmp_wav.name, tmp_mp3, tmp_out]:
            if os.path.exists(f):
                os.remove(f)
    return y_out


def pitch_shift(y, sr, semitones):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)


def add_background_noise(y, snr_db=20):
    # pink noise approximation - more realistic background noise than white
    white = np.random.randn(len(y))
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.000000000, -2.494956002, 2.017265875, -0.522189400]
    from scipy.signal import lfilter
    pink = lfilter(b, a, white)
    pink = pink / (np.std(pink) + 1e-10)

    signal_power = np.mean(y ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    pink = pink * np.sqrt(noise_power)
    return y + pink


def main():
    print("Adversarial robustness testing")

    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
    except FileNotFoundError:
        print("ffmpeg not found. Install it: brew install ffmpeg")
        print("Skipping MP3 compression tests.")
        has_ffmpeg = False
    else:
        has_ffmpeg = True

    df = pd.read_pickle('features/all_features_combined.pkl')
    df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'real' else 1)
    # print(df['label'].value_counts())

    with open('data/train_test_splits.pkl', 'rb') as f:
        splits = pickle.load(f)

    X_train = splits['X_train']
    y_train = splits['y_train']
    test_indices = splits['X_test'].index

    test_df = df.loc[test_indices].copy()
    test_files = test_df['filename'].values
    test_labels = test_df['binary_label'].values

    def find_audio(filename):
        dirs = ['dataset/real',
                'dataset/synthetic/elevenlabs',
                'dataset/synthetic/google',
                'dataset/synthetic/polly']
        for d in dirs:
            path = Path(d) / filename
            if path.exists():
                return str(path)
        return None

    attacks = {}
    attacks['Clean (baseline)'] = lambda y, sr: y
    attacks['Noise SNR=30dB'] = lambda y, sr: add_gaussian_noise(y, 30)
    attacks['Noise SNR=20dB'] = lambda y, sr: add_gaussian_noise(y, 20)
    attacks['Noise SNR=10dB'] = lambda y, sr: add_gaussian_noise(y, 10)
    if has_ffmpeg:
        attacks['MP3 128kbps'] = lambda y, sr: mp3_compress(y, sr, '128k')
        attacks['MP3 64kbps'] = lambda y, sr: mp3_compress(y, sr, '64k')
        attacks['MP3 32kbps'] = lambda y, sr: mp3_compress(y, sr, '32k')
    attacks['Pitch +2 semi'] = lambda y, sr: pitch_shift(y, sr, 2)
    attacks['Pitch -2 semi'] = lambda y, sr: pitch_shift(y, sr, -2)
    attacks['BG Noise 20dB'] = lambda y, sr: add_background_noise(y, 20)
    attacks['BG Noise 10dB'] = lambda y, sr: add_background_noise(y, 10)

    feature_cols = list(X_train.columns)

    mfcc_cols = [c for c in feature_cols if c.startswith('mfcc')]
    all_cols = feature_cols

    configs = {
        'MFCC only': mfcc_cols,
        'All features': all_cols
    }

    trained = {}
    for config_name, cols in configs.items():
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train[cols])
        model = SVC(kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=42)
        model.fit(X_tr, y_train)
        trained[config_name] = {'model': model, 'scaler': scaler, 'cols': cols}

    # Kept this to 50 files because RQA extraction is the bottleneck here,
    # especially once each file is attacked multiple ways.
    np.random.seed(42)
    if len(test_files) > 50:
        subset_idx = np.random.choice(len(test_files), 50, replace=False)
    else:
        subset_idx = np.arange(len(test_files))

    subset_files = test_files[subset_idx]
    subset_labels = test_labels[subset_idx]

    print(f"\nTesting on {len(subset_files)} samples from test set")
    print(f"Attacks: {len(attacks)}")

    results = []

    for attack_name, attack_fn in attacks.items():
        print(f"\n--- {attack_name} ---")

        rows = []
        errors = 0
        for i, (fname, label) in enumerate(zip(subset_files, subset_labels)):
            path = find_audio(fname)
            if path is None:
                errors += 1
                continue

            try:
                y, sr = librosa.load(path, sr=16000)
                y_attacked = attack_fn(y, sr)
                feats = extract_all(y_attacked, sr)
                feats['label'] = label
                rows.append(feats)
            except Exception as e:
                errors += 1
                continue

            if (i + 1) % 10 == 0:
                print(f"  processed {i+1}/{len(subset_files)}")

        if errors > 0:
            print(f"  ({errors} files skipped)")

        if len(rows) == 0:
            continue

        attack_df = pd.DataFrame(rows)
        attack_df[feature_cols] = attack_df[feature_cols].fillna(
            attack_df[feature_cols].median())
        attack_df[feature_cols] = attack_df[feature_cols].fillna(0)
        y_true = attack_df['label'].values

        for config_name, info in trained.items():
            X_attack = attack_df[info['cols']].values
            X_scaled = info['scaler'].transform(X_attack)
            y_pred = info['model'].predict(X_scaled)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            print(f"  {config_name:<18} acc={acc:.4f}  f1={f1:.4f}")

            results.append({
                'attack': attack_name,
                'config': config_name,
                'accuracy': acc,
                'f1': f1,
                'n_samples': len(rows)
            })

    res_df = pd.DataFrame(results)
    Path('results').mkdir(parents=True, exist_ok=True)
    res_df.to_csv('results/adversarial_results.csv', index=False)
    print(f"\nSaved results/adversarial_results.csv")

    print("\nSUMMARY")
    print(f"\n{'Attack':<20} {'MFCC Acc':>10} {'All Acc':>10} "
          f"{'MFCC F1':>10} {'All F1':>10}")

    for attack in attacks.keys():
        mfcc_row = res_df[(res_df['attack'] == attack) &
                          (res_df['config'] == 'MFCC only')]
        all_row = res_df[(res_df['attack'] == attack) &
                         (res_df['config'] == 'All features')]

        if len(mfcc_row) > 0 and len(all_row) > 0:
            print(f"{attack:<20} {mfcc_row['accuracy'].values[0]:>10.4f} "
                  f"{all_row['accuracy'].values[0]:>10.4f} "
                  f"{mfcc_row['f1'].values[0]:>10.4f} "
                  f"{all_row['f1'].values[0]:>10.4f}")

    print("\nDEGRADATION FROM CLEAN BASELINE")

    for config_name in ['MFCC only', 'All features']:
        clean = res_df[(res_df['attack'] == 'Clean (baseline)') &
                       (res_df['config'] == config_name)]
        if len(clean) == 0:
            continue
        clean_acc = clean['accuracy'].values[0]
        print(f"\n{config_name} (clean baseline: {clean_acc:.4f}):")

        config_results = res_df[res_df['config'] == config_name]
        for _, row in config_results.iterrows():
            if row['attack'] == 'Clean (baseline)':
                continue
            drop = clean_acc - row['accuracy']
            print(f"  {row['attack']:<20} acc={row['accuracy']:.4f}  "
                  f"drop={drop:+.4f}")


if __name__ == '__main__':
    main()
