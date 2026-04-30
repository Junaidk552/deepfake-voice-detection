import numpy as np
import librosa
from scipy.stats import skew, kurtosis


def extract_mfcc(y, sr=16000):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    deltas = librosa.feature.delta(mfccs)
    delta2s = librosa.feature.delta(mfccs, order=2)
    f = {}
    for i in range(13):
        f[f'mfcc_{i}'] = np.mean(mfccs[i])
    for i in range(13):
        f[f'mfcc_delta_{i}'] = np.mean(deltas[i])
    for i in range(13):
        f[f'mfcc_delta2_{i}'] = np.mean(delta2s[i])
    return f


def extract_cqcc(y, sr=16000):
    from spafe.features.cqcc import cqcc as spafe_cqcc
    try:
        cqcc_feat = spafe_cqcc(y, fs=sr, num_ceps=13)
        cqcc_mean = np.mean(cqcc_feat, axis=0)[:13]
        return {f'cqcc_{i}': cqcc_mean[i] for i in range(13)}
    # spafe occasionally fails on very short clips, return zeros rather than crash
    except:
        return {f'cqcc_{i}': 0.0 for i in range(13)}


def extract_rqa(y, sr=16000):
    # RQA blew up on full-length clips (O(n^2) distance matrix), so I cap to ~2000 points first.
    # I tested bigger targets but runtime jumped a lot with barely any lift in validation accuracy.
    target = 2000
    step = max(1, len(y) // target)
    signal = y[::step][:target].astype(float)

    m, tau = 10, 1
    eps = 0.1 * np.std(signal)
    N = len(signal) - (m - 1) * tau

    if N < 20:
        return {f'rqa_{k}': 0 for k in ['recurrence_rate', 'determinism',
                'avg_diagonal', 'max_diagonal', 'entropy_diagonal',
                'laminarity', 'trapping_time']}

    try:
        from pyrqa.time_series import TimeSeries
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Classic
        from pyrqa.neighbourhood import FixedRadius
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RQAComputation

        ts = TimeSeries(signal, embedding_dimension=m, time_delay=tau)
        settings = Settings(ts, analysis_type=Classic,
                          neighbourhood=FixedRadius(eps),
                          similarity_measure=EuclideanMetric)
        comp = RQAComputation.create(settings)
        result = comp.run()

        return {
            'rqa_recurrence_rate': result.recurrence_rate,
            'rqa_determinism': result.determinism,
            'rqa_avg_diagonal': result.average_diagonal_line,
            'rqa_max_diagonal': result.longest_diagonal_line,
            'rqa_entropy_diagonal': result.entropy_diagonal_lines,
            'rqa_laminarity': result.laminarity,
            'rqa_trapping_time': result.trapping_time
        }
    # fall through to numpy fallback below
    except:
        pass

    # pyrqa occasionally crashed on some machines, so I kept this numpy fallback for reproducibility.
    embedded = np.array([signal[i:i + m * tau:tau] for i in range(N)])
    chunk = 500
    recurrence = np.zeros((N, N), dtype=bool)
    for i in range(0, N, chunk):
        ei = min(i + chunk, N)
        for j in range(0, N, chunk):
            ej = min(j + chunk, N)
            dist = np.sqrt(np.sum(
                (embedded[i:ei, None, :] - embedded[None, j:ej, :]) ** 2,
                axis=2))
            recurrence[i:ei, j:ej] = dist < eps

    rec_points = np.sum(recurrence)
    rr = rec_points / (N * N)

    # diagonal lines capture repeating trajectories in the reconstructed phase space.
    dl = []
    for k in range(-N + 1, N):
        d = np.diagonal(recurrence, k)
        l = 0
        for v in d:
            if v:
                l += 1
            else:
                if l >= 2: dl.append(l)
                l = 0
        if l >= 2: dl.append(l)

    if dl:
        det = sum(dl) / rec_points if rec_points > 0 else 0
        ad = np.mean(dl)
        md = np.max(dl)
        counts = np.bincount(dl)
        probs = counts[counts > 0] / counts[counts > 0].sum()
        ed = -np.sum(probs * np.log(probs + 1e-10))
    else:
        det = ad = md = ed = 0

    # vertical lines track "stuck" regions, useful for laminarity/trapping-time.
    vl = []
    for col in range(N):
        l = 0
        for row in range(N):
            if recurrence[row, col]:
                l += 1
            else:
                if l >= 2: vl.append(l)
                l = 0
        if l >= 2: vl.append(l)

    lam = sum(vl) / rec_points if vl and rec_points > 0 else 0
    tt = np.mean(vl) if vl else 0

    return {
        'rqa_recurrence_rate': rr, 'rqa_determinism': det,
        'rqa_avg_diagonal': ad, 'rqa_max_diagonal': md,
        'rqa_entropy_diagonal': ed, 'rqa_laminarity': lam,
        'rqa_trapping_time': tt
    }


def extract_entropy(y, sr=16000):
    from nolds import sampen
    f = {}
    for scale in range(1, 11):
        if scale == 1:
            coarse = y
        else:
            # coarse graining for multiscale entropy: average non-overlapping windows.
            n = len(y) - (len(y) % scale)
            coarse = np.mean(y[:n].reshape(-1, scale), axis=1)
        try:
            # I trim to first 3000 points to keep sample entropy stable and avoid very slow runs.
            se = sampen(coarse[:3000], emb_dim=2,
                       tolerance=0.2 * np.std(coarse))
            f[f'entropy_scale_{scale}'] = se
        except:
            f[f'entropy_scale_{scale}'] = 0.0
    return f


def extract_pauses(y, sr=16000):
    frame_length = int(0.020 * sr)
    hop_length = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length,
                               hop_length=hop_length)[0]
    # 20th percentile worked best after trial runs; fixed thresholds broke across loud vs quiet clips.
    threshold = np.percentile(rms, 20)
    is_silent = rms < threshold

    pauses = []
    in_pause = False
    start = 0
    for i, s in enumerate(is_silent):
        if s and not in_pause:
            start = i
            in_pause = True
        elif not s and in_pause:
            dur = (i - start) * hop_length / sr
            # Ignore micro-pauses from plosives/breath noise; they were adding false pause spikes.
            if dur >= 0.02:
                pauses.append(dur)
            in_pause = False

    if len(pauses) >= 2:
        return {
            'pause_mean_duration': np.mean(pauses),
            'pause_std_duration': np.std(pauses),
            'pause_rate': len(pauses) / (len(y) / sr),
            'pause_median_duration': np.median(pauses),
            'pause_range': np.max(pauses) - np.min(pauses),
            'pause_skewness': skew(pauses),
            'pause_kurtosis': kurtosis(pauses),
            'pause_coeff_variation': np.std(pauses) / np.mean(pauses)
                                     if np.mean(pauses) > 0 else 0
        }
    else:
        return {
            'pause_mean_duration': np.mean(pauses) if pauses else 0,
            'pause_std_duration': 0,
            'pause_rate': len(pauses) / (len(y) / sr),
            'pause_median_duration': np.mean(pauses) if pauses else 0,
            'pause_range': 0,
            'pause_skewness': np.nan,
            'pause_kurtosis': np.nan,
            'pause_coeff_variation': 0
        }


def extract_all_features(filepath, sr=16000):
    # I cap clips at 10s so feature vectors are consistent and long files don't dominate runtime.
    # Most spoof artifacts I cared about showed up early anyway.
    y, _ = librosa.load(filepath, sr=sr, duration=10)
    f = {}
    f.update(extract_mfcc(y, sr))
    f.update(extract_cqcc(y, sr))
    f.update(extract_rqa(y, sr))
    f.update(extract_entropy(y, sr))
    f.update(extract_pauses(y, sr))
    return f


def extract_all_from_audio(y, sr=16000):
    # Same feature stack as file-based path, just for preloaded arrays (e.g., augmentation/testing).
    # TODO: add optional clipping here too so this path matches extract_all_features exactly.
    f = {}
    f.update(extract_mfcc(y, sr))
    f.update(extract_cqcc(y, sr))
    f.update(extract_rqa(y, sr))
    f.update(extract_entropy(y, sr))
    f.update(extract_pauses(y, sr))
    return f