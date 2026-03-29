"""scripts/generate_polly_tts.py - Generate 175 TTS samples from Amazon Polly"""
import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")

if not aws_key or not aws_secret:
    raise ValueError("AWS credentials not found in .env file!")

OUTPUT_DIR = "dataset/synthetic/polly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

polly_client = boto3.client(
    'polly',
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret,
    region_name=aws_region
)

with open("scripts/prompts.json") as f:
    all_prompts = json.load(f)

# prompts 326-500 (polly gets the last batch)
prompts = all_prompts[325:500]

voices = [
    {"Id": "Amy", "Language": "en-GB"},
    {"Id": "Brian", "Language": "en-GB"},
    {"Id": "Emma", "Language": "en-GB"},
    {"Id": "Arthur", "Language": "en-GB"},
    {"Id": "Joanna", "Language": "en-US"},
    {"Id": "Matthew", "Language": "en-US"},
    {"Id": "Salli", "Language": "en-US"},
    {"Id": "Joey", "Language": "en-US"},
    {"Id": "Kendra", "Language": "en-US"},
    {"Id": "Justin", "Language": "en-US"},
    {"Id": "Olivia", "Language": "en-AU"},
    {"Id": "Ruth", "Language": "en-GB"},
]

print(f"Starting Amazon Polly TTS generation...")
print(f"Target: {len(prompts)} samples (prompts 326-500)")
print(f"Voices: {len(voices)} different voices")
print(f"Region: {aws_region}\n")

count = 0

for i, prompt in enumerate(prompts):
    voice = voices[i % len(voices)]

    try:
        response = polly_client.synthesize_speech(
            Text=prompt,
            OutputFormat='pcm',
            VoiceId=voice["Id"],
            Engine='neural',
            SampleRate='16000'
        )

        output_name = f"polly_{count+1:04d}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        with open(output_path, 'wb') as f:
            f.write(response['AudioStream'].read())

        count += 1

        if count % 20 == 0:
            print(f"Generated {count}/{len(prompts)} samples...")

    except Exception as e:
        print(f"Error on sample {i+1}: {e}")
        continue

print(f"\nAmazon Polly generation complete!")
print(f"  Samples generated: {count}")
print(f"  Output: {OUTPUT_DIR}/")
