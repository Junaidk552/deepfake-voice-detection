import os
import librosa
import soundfile as sf
from pathlib import Path
import random

# === UPDATE THIS PATH to match your system ===
LIBRISPEECH_DIR = "/Users/junaidkhan/Downloads/LibriSpeech/dev-clean"  # macOS
# LIBRISPEECH_DIR = "C:/Users/YourName/Downloads/LibriSpeech/dev-clean"  # Windows

OUTPUT_DIR = "dataset/real/librispeech"
TARGET_COUNT = 500
TARGET_SR = 16000
MIN_DURATION = 3.0
MAX_DURATION = 10.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all .flac files
all_files = list(Path(LIBRISPEECH_DIR).rglob("*.flac"))
print(f"Found {len(all_files)} total files in LibriSpeech dev-clean")

# Shuffle for diversity across speakers
random.seed(42)
random.shuffle(all_files)

count = 0
skipped = 0

for audio_file in all_files:
    try:
        # Load audio (LibriSpeech is already 16kHz)
        y, sr = librosa.load(audio_file, sr=TARGET_SR)
        
        # Check duration
        duration = len(y) / sr
        if duration < MIN_DURATION or duration > MAX_DURATION:
            skipped += 1
            continue
        
        # Save with clear naming
        output_name = f"real_librispeech_{count+1:04d}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        sf.write(output_path, y, TARGET_SR)
        
        count += 1
        if count % 50 == 0:
            print(f"✓ Collected {count}/{TARGET_COUNT} samples...")
        
        if count >= TARGET_COUNT:
            break
            
    except Exception as e:
        print(f"Error processing {audio_file.name}: {e}")
        continue

print(f"\n{'='*60}")
print(f"✓ COLLECTION COMPLETE")
print(f"  Collected: {count} samples")
print(f"  Skipped: {skipped} (duration out of range)")
print(f"  Output: {OUTPUT_DIR}/")
print(f"{'='*60}")