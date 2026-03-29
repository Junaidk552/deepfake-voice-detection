"""Fix Polly PCM files by adding WAV headers"""
import os
import wave
import numpy as np

INPUT_DIR = "dataset/synthetic/polly"
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.wav')])

print(f"Converting {len(files)} Polly files to proper WAV format...")

for i, filename in enumerate(files):
    filepath = os.path.join(INPUT_DIR, filename)

    try:
        with open(filepath, 'rb') as f:
            pcm_data = f.read()

        # polly outputs raw PCM so we need to wrap it in a WAV container
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)

        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_array.tobytes())

        if (i + 1) % 20 == 0:
            print(f"Converted {i + 1}/{len(files)} files...")

    except Exception as e:
        print(f"Error converting {filename}: {e}")

print(f"\nAll files converted successfully!")
print(f"  Files: {INPUT_DIR}/")
