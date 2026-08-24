#!/usr/bin/env python3
"""
convert_recordings.py

This script processes data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- [None detected: script may rely on in-memory data or CLI/runtime context.]
Outputs:
- [None detected: script may print/report only or be imported by other scripts.]
Reproduces: Reproduces intermediate outputs used by other scripts.
"""

import librosa
import soundfile as sf
from pathlib import Path

INPUT_DIR = 'dataset/voice_clone/real'
count = 0

for i in range(70, 95):
    input_path = Path(INPUT_DIR) / f"New Recording {i}.m4a"
    if not input_path.exists():
        print(f"  Skipping: {input_path} (not found)")
        continue

    y, sr = librosa.load(str(input_path), sr=16000)
    output_path = Path(INPUT_DIR) / f"real_junaid_{count+1:03d}.wav"
    sf.write(str(output_path), y, 16000)
    count += 1
    print(f"  {input_path.name} -> {output_path.name} ({len(y)/16000:.1f}s)")

print(f"\nConverted {count} files to 16kHz WAV")
