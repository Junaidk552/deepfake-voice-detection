"""scripts/generate_polly_tts.py - Generate 175 TTS samples from Amazon Polly"""
import os
import json
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify credentials
aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "eu-west-2")

if not aws_key or not aws_secret:
    raise ValueError("AWS credentials not found in .env file!")

OUTPUT_DIR = "dataset/synthetic/polly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Polly client
polly_client = boto3.client(
    'polly',
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret,
    region_name=aws_region
)

# Load prompts (use 326-500 since ElevenLabs used 1-150, Google used 151-325)
with open("scripts/prompts.json") as f:
    all_prompts = json.load(f)

prompts = all_prompts[325:500]  # Get prompts 326-500 (175 total)

# Neural voices (higher quality, mix of GB/US/AU accents)
voices = [
    {"Id": "Amy", "Language": "en-GB"},      # British Female
    {"Id": "Brian", "Language": "en-GB"},    # British Male
    {"Id": "Emma", "Language": "en-GB"},     # British Female
    {"Id": "Arthur", "Language": "en-GB"},   # British Male
    {"Id": "Joanna", "Language": "en-US"},   # US Female
    {"Id": "Matthew", "Language": "en-US"},  # US Male
    {"Id": "Salli", "Language": "en-US"},    # US Female
    {"Id": "Joey", "Language": "en-US"},     # US Male
    {"Id": "Kendra", "Language": "en-US"},   # US Female
    {"Id": "Justin", "Language": "en-US"},   # US Male (child)
    {"Id": "Olivia", "Language": "en-AU"},   # Australian Female
    {"Id": "Ruth", "Language": "en-GB"},     # British Female
]

print(f"Starting Amazon Polly TTS generation...")
print(f"Target: {len(prompts)} samples (prompts 326-500)")
print(f"Voices: {len(voices)} different voices")
print(f"Region: {aws_region}\n")

count = 0

for i, prompt in enumerate(prompts):
    # Cycle through voices
    voice = voices[i % len(voices)]
    
    try:
        # Request speech synthesis
        response = polly_client.synthesize_speech(
            Text=prompt,
            OutputFormat='pcm',
            VoiceId=voice["Id"],
            Engine='neural',  # Use neural engine for better quality
            SampleRate='16000'
        )
        
        # Save the audio
        output_name = f"polly_{count+1:04d}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        # Write raw PCM data to file
        with open(output_path, 'wb') as f:
            f.write(response['AudioStream'].read())
        
        count += 1
        
        # Progress update every 20 samples
        if count % 20 == 0:
            print(f"✓ Generated {count}/{len(prompts)} samples...")
        
    except Exception as e:
        print(f"❌ Error on sample {i+1}: {e}")
        continue

print(f"\n{'='*60}")
print(f"✓ Amazon Polly generation complete!")
print(f"  Samples generated: {count}")
print(f"  Output: {OUTPUT_DIR}/")
print(f"{'='*60}")