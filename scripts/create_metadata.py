"""
create_metadata.py

This script processes data for the deepfake voice detection pipeline and dissertation experiments.
It is designed to run from the project root so file paths resolve consistently across dataset, features, and results directories.

Inputs:
- *.wav
Outputs:
- dataset/metadata.csv
Reproduces: Reproduces intermediate outputs used by other scripts.
"""
import os
import csv
import librosa
from pathlib import Path

metadata = []

print("Scanning dataset files...")

# real samples
for source in ["librispeech"]:
    dir_path = f"dataset/real/{source}"
    if not os.path.exists(dir_path):
        print(f"  {dir_path} not found, skipping...")
        continue

    files = sorted(Path(dir_path).glob("*.wav"))
    print(f"Processing {len(files)} files from {source}...")

    for wav in files:
        try:
            y, sr = librosa.load(wav, sr=None)
            duration = len(y) / sr
            metadata.append({
                "filename": str(wav),
                "label": "real",
                "source": source,
                "duration": round(duration, 2),
                "sample_rate": sr
            })
        except Exception as e:
            print(f"Error loading {wav}: {e}")

# synthetic samples
for source in ["elevenlabs", "google", "polly"]:
    dir_path = f"dataset/synthetic/{source}"
    if not os.path.exists(dir_path):
        print(f"  {dir_path} not found, skipping...")
        continue

    files = sorted(Path(dir_path).glob("*.wav"))
    print(f"Processing {len(files)} files from {source}...")

    for wav in files:
        try:
            y, sr = librosa.load(wav, sr=None)
            duration = len(y) / sr
            metadata.append({
                "filename": str(wav),
                "label": "synthetic",
                "source": source,
                "duration": round(duration, 2),
                "sample_rate": sr
            })
        except Exception as e:
            print(f"Error loading {wav}: {e}")

with open("dataset/metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "label", "source", "duration", "sample_rate"])
    writer.writeheader()
    writer.writerows(metadata)

real_count = sum(1 for m in metadata if m["label"] == "real")
synth_count = sum(1 for m in metadata if m["label"] == "synthetic")
avg_duration = sum(m["duration"] for m in metadata) / len(metadata)

print(f"\nDataset Summary:")
print(f"  Real samples:      {real_count}")
print(f"  Synthetic samples: {synth_count}")
print(f"  Total:             {len(metadata)}")
print(f"  Avg duration:      {avg_duration:.2f}s")
print(f"\nMetadata saved to dataset/metadata.csv")
