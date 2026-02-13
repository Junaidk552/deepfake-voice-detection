"""scripts/create_prompts.py - Parse Harvard Sentences for TTS"""
import json
import re

# Read Harvard Sentences
with open("scripts/harvard_sentences.txt", "r") as f:
    lines = f.readlines()

# Clean and filter
prompts = []
for line in lines:
    sentence = line.strip()
    
    # Skip empty lines
    if not sentence:
        continue
    
    # Skip headers like "H1 Harvard Sentences", "H2 Harvard Sentences"
    if sentence.startswith("H") and "Harvard Sentences" in sentence:
        continue
    
    # Remove leading numbers like "1. " or "10. "
    sentence = re.sub(r'^\d+\.\s*', '', sentence)
    
    # Only keep if it's a proper sentence (at least 20 chars)
    if len(sentence) > 20:
        prompts.append(sentence)

# Remove duplicates (just in case)
prompts = list(set(prompts))

print(f"✓ Parsed {len(prompts)} unique sentences")

# Save to JSON
with open("prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"✓ Saved to: scripts/prompts.json")

# Show some examples
print(f"\nFirst 10 prompts:")
for i in range(min(10, len(prompts))):
    print(f"  {i+1}. {prompts[i]}")

print(f"\nStats:")
print(f"  Total prompts: {len(prompts)}")
print(f"  Shortest: {len(min(prompts, key=len))} chars")
print(f"  Longest: {len(max(prompts, key=len))} chars")
print(f"  Average: {sum(len(p) for p in prompts) // len(prompts)} chars")
