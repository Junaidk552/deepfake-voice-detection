#!/usr/bin/env python3
"""convert iPhone m4a recordings to 16kHz WAV"""

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

# print(f'expected ~25 files, got {count}')
print(f"\nConverted {count} files to 16kHz WAV")
