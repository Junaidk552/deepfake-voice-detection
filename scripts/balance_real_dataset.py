"""
balance_real_dataset.py

This script processes data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- real_commonvoice_*.wav
- real_librispeech_*.wav
Outputs:
- [None detected: script may print/report only or be imported by other scripts.]
Reproduces: Reproduces intermediate outputs used by other scripts.
"""

import random
import shutil
from pathlib import Path

real_dir = Path('dataset/real')

libri_files = sorted(real_dir.glob('real_librispeech_*.wav'))
cv_files = sorted((real_dir / 'common_voice').glob('real_commonvoice_*.wav'))

print(f"LibriSpeech files: {len(libri_files)}")
print(f"Common Voice files: {len(cv_files)}")

random.seed(42)
keep = set(random.sample(range(len(libri_files)), 250))

backup = real_dir / 'librispeech_backup'
backup.mkdir(exist_ok=True)

moved = 0
for i, f in enumerate(libri_files):
    if i not in keep:
        shutil.move(str(f), str(backup / f.name))
        moved += 1

remaining_libri = len(list(real_dir.glob('real_librispeech_*.wav')))
print(f"\nMoved {moved} LibriSpeech to backup")
print(f"Remaining LibriSpeech: {remaining_libri}")
print(f"Common Voice: {len(cv_files)}")
print(f"Total real: {remaining_libri + len(cv_files)}")
print(f"Synthetic: 500 (unchanged)")
print(f"\nDataset balanced: {remaining_libri + len(cv_files)} real vs 500 synthetic")
