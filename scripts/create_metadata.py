"""scripts/create_metadata.py - Create master dataset tracking CSV"""
import os
import csv
import librosa
from pathlib import Path

metadata = []

print("Scanning dataset files...")

# Real samples
for source in ["librispeech"]:  # You only used LibriSpeech
    dir_path = f"dataset/real/{source}"
    if not os.path.exists(dir_path):
        print(f"⚠️  {dir_path} not found, skipping...")
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
            print(f"❌ Error loading {wav}: {e}")

# Synthetic samples
for source in ["elevenlabs", "google", "polly"]:
    dir_path = f"dataset/synthetic/{source}"
    if not os.path.exists(dir_path):
        print(f"⚠️  {dir_path} not found, skipping...")
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
            print(f"❌ Error loading {wav}: {e}")

# Write CSV
with open("dataset/metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "label", "source", "duration", "sample_rate"])
    writer.writeheader()
    writer.writerows(metadata)

# Summary stats
real_count = sum(1 for m in metadata if m["label"] == "real")
synth_count = sum(1 for m in metadata if m["label"] == "synthetic")
avg_duration = sum(m["duration"] for m in metadata) / len(metadata)

print(f"\n{'='*60}")
print(f"Dataset Summary:")
print(f"  Real samples:      {real_count}")
print(f"  Synthetic samples: {synth_count}")
print(f"  Total:             {len(metadata)}")
print(f"  Avg duration:      {avg_duration:.2f}s")
print(f"\n✓ Metadata saved to dataset/metadata.csv")
print(f"{'='*60}")