"""scripts/generate_elevenlabs.py - Generate ~70 TTS samples from ElevenLabs"""
import os
import json
import soundfile as sf
import numpy as np
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    raise ValueError("ELEVENLABS_API_KEY not found in .env file!")

OUTPUT_DIR = "dataset/synthetic/elevenlabs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = ElevenLabs(api_key=API_KEY)

with open("scripts/prompts.json") as f:
    prompts = json.load(f)

# first 150 prompts only
prompts = prompts[:150]

voices = [
    {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
    {"id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi"},
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella"},
    {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni"},
    {"id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh"},
    {"id": "VR6AewLTigWG4xSOukaG", "name": "Arnold"},
    {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
]

count = 0
char_count = 0
MAX_CHARS = 9000  # free tier limit with buffer

print(f"Starting ElevenLabs TTS generation...")
print(f"Target: {len(prompts)} samples")
print(f"Character limit: {MAX_CHARS}\n")

for i, prompt in enumerate(prompts):
    if char_count + len(prompt) > MAX_CHARS:
        print(f"\nApproaching character limit ({char_count} chars used). Stopping.")
        break

    voice = voices[i % len(voices)]

    try:
        audio = client.text_to_speech.convert(
            voice_id=voice["id"],
            text=prompt,
            model_id="eleven_turbo_v2_5",
            output_format="pcm_16000",
        )

        audio_bytes = b"".join(audio)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        output_name = f"elevenlabs_{count+1:04d}.wav"
        sf.write(os.path.join(OUTPUT_DIR, output_name), audio_array, 16000)

        char_count += len(prompt)
        count += 1

        if count % 10 == 0:
            print(f"Generated {count}/{len(prompts)} samples ({char_count} chars used)")

    except Exception as e:
        print(f"Error on sample {i+1}: {e}")
        continue

# print(f'voices used: {[voices[i % len(voices)]["name"] for i in range(count)]}')
print(f"\nElevenLabs generation complete!")
print(f"  Samples generated: {count}")
print(f"  Characters used: {char_count}/{MAX_CHARS}")
print(f"  Output: {OUTPUT_DIR}/")
