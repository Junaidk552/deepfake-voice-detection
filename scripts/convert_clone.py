#!/usr/bin/env python3
"""rename and convert cloned voice mp3s to 16kHz WAV"""

import librosa
import soundfile as sf
from pathlib import Path

INPUT_DIR = Path('dataset/voice_clone/synthetic')

# sort by timestamp in filename so order matches recording order
mp3s = sorted(INPUT_DIR.glob('*.mp3'))
print(f"Found {len(mp3s)} files")
# print([f.name for f in mp3s[:3]])

for i, mp3 in enumerate(mp3s):
    y, sr = librosa.load(str(mp3), sr=16000)
    output = INPUT_DIR / f"synthetic_junaid_{i+1:03d}.wav"
    sf.write(str(output), y, 16000)
    print(f"  {i+1}. {output.name} ({len(y)/16000:.1f}s)")

print(f"\nDone! {len(mp3s)} WAV files created")
