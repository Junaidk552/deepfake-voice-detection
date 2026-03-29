"""scripts/generate_google_tts.py - Generate 175 TTS samples from Google Cloud"""
import os
import json
from google.cloud import texttospeech
from dotenv import load_dotenv

load_dotenv()

creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not creds_path or not os.path.exists(creds_path):
    raise ValueError(f"Google credentials not found at: {creds_path}")

OUTPUT_DIR = "dataset/synthetic/google"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = texttospeech.TextToSpeechClient()

with open("scripts/prompts.json") as f:
    all_prompts = json.load(f)

# prompts 151-325 (google gets the middle batch)
prompts = all_prompts[150:325]

voices = [
    {"name": "en-GB-Neural2-A", "gender": texttospeech.SsmlVoiceGender.FEMALE, "lang": "en-GB"},
    {"name": "en-GB-Neural2-B", "gender": texttospeech.SsmlVoiceGender.MALE, "lang": "en-GB"},
    {"name": "en-GB-Neural2-C", "gender": texttospeech.SsmlVoiceGender.FEMALE, "lang": "en-GB"},
    {"name": "en-GB-Neural2-D", "gender": texttospeech.SsmlVoiceGender.MALE, "lang": "en-GB"},
    {"name": "en-US-Neural2-A", "gender": texttospeech.SsmlVoiceGender.MALE, "lang": "en-US"},
    {"name": "en-US-Neural2-C", "gender": texttospeech.SsmlVoiceGender.FEMALE, "lang": "en-US"},
    {"name": "en-US-Neural2-D", "gender": texttospeech.SsmlVoiceGender.MALE, "lang": "en-US"},
    {"name": "en-US-Neural2-E", "gender": texttospeech.SsmlVoiceGender.FEMALE, "lang": "en-US"},
    {"name": "en-AU-Neural2-A", "gender": texttospeech.SsmlVoiceGender.FEMALE, "lang": "en-AU"},
    {"name": "en-AU-Neural2-B", "gender": texttospeech.SsmlVoiceGender.MALE, "lang": "en-AU"},
    {"name": "en-AU-Neural2-C", "gender": texttospeech.SsmlVoiceGender.FEMALE, "lang": "en-AU"},
    {"name": "en-AU-Neural2-D", "gender": texttospeech.SsmlVoiceGender.MALE, "lang": "en-AU"},
]

print(f"Starting Google Cloud TTS generation...")
print(f"Target: {len(prompts)} samples (prompts 151-325)")
print(f"Voices: {len(voices)} different voices\n")

count = 0

for i, prompt in enumerate(prompts):
    voice_config = voices[i % len(voices)]

    try:
        synthesis_input = texttospeech.SynthesisInput(text=prompt)

        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_config["lang"],
            name=voice_config["name"],
            ssml_gender=voice_config["gender"]
        )

        # LINEAR16 = 16kHz WAV
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        output_name = f"google_{count+1:04d}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        with open(output_path, "wb") as out:
            out.write(response.audio_content)

        count += 1

        if count % 20 == 0:
            print(f"Generated {count}/{len(prompts)} samples...")

    except Exception as e:
        print(f"Error on sample {i+1}: {e}")
        continue

print(f"\nGoogle Cloud TTS generation complete!")
print(f"  Samples generated: {count}")
print(f"  Output: {OUTPUT_DIR}/")
