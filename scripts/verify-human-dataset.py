import os
import librosa
from pathlib import Path

REAL_DIR = "dataset/real/"

total = 0
issues = []

if not os.path.exists(REAL_DIR):
    print(f"ERROR: {REAL_DIR} does not exist!")
    exit(1)

files = list(Path(REAL_DIR).glob("*.wav"))
print(f"LibriSpeech samples: {len(files)} files")
total = len(files)

import random
random.seed(42)
sample_files = random.sample(files, min(10, len(files)))

for f in sample_files:
    y, sr = librosa.load(f, sr=None)
    duration = len(y) / sr
    if sr < 16000:
        issues.append(f"{f.name}: sample rate {sr} < 16000")
    if duration < 3.0 or duration > 10.0:
        issues.append(f"{f.name}: duration {duration:.1f}s out of range")

print(f"\nTotal real samples: {total}")
print(f"Target: 500")
print(f"Status: {'PASS' if total >= 400 else 'FAIL - need at least 400'}")
print(f"Issues found: {len(issues)}")
for issue in issues:
    print(f"  - {issue}")
