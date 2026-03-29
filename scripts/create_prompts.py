"""scripts/create_prompts.py - Parse Harvard Sentences for TTS"""
import json
import re

with open("scripts/harvard_sentences.txt", "r") as f:
    lines = f.readlines()

prompts = []
for line in lines:
    sentence = line.strip()

    if not sentence:
        continue

    if sentence.startswith("H") and "Harvard Sentences" in sentence:
        continue

    sentence = re.sub(r'^\d+\.\s*', '', sentence)

    if len(sentence) > 20:
        prompts.append(sentence)

prompts = list(set(prompts))

print(f"Parsed {len(prompts)} unique sentences")

with open("prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"Saved to: scripts/prompts.json")

print(f"\nFirst 10 prompts:")
for i in range(min(10, len(prompts))):
    print(f"  {i+1}. {prompts[i]}")

print(f"\nStats:")
print(f"  Total prompts: {len(prompts)}")
print(f"  Shortest: {len(min(prompts, key=len))} chars")
print(f"  Longest: {len(max(prompts, key=len))} chars")
print(f"  Average: {sum(len(p) for p in prompts) // len(prompts)} chars")
