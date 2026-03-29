"""select ~250 diverse clips from common voice and convert to 16kHz WAV"""

import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import shutil

CV_DIR = Path.home() / 'downloads' / 'archive'
OUTPUT_DIR = Path('dataset/real/common_voice')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_path = CV_DIR / 'cv-valid-train.csv'
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

print(f"Total validated train clips: {len(df)}")
print(f"\nGender distribution:\n{df['gender'].value_counts(dropna=False)}")
print(f"\nAccent distribution:\n{df['accent'].value_counts(dropna=False).head(10)}")

# only clips where we have both gender and accent info
has_info = df[df['gender'].notna() & df['accent'].notna()]
print(f"\nClips with gender + accent info: {len(has_info)}")

selected = []

accents_wanted = ['us', 'england', 'australia', 'canada', 'indian',
                  'ireland', 'scotland', 'wales', 'newzealand']

for accent in accents_wanted:
    accent_clips = has_info[has_info['accent'] == accent]
    if len(accent_clips) > 0:
        males = accent_clips[accent_clips['gender'] == 'male'].head(8)
        females = accent_clips[accent_clips['gender'] == 'female'].head(8)
        selected.append(males)
        selected.append(females)
        print(f"  {accent}: {len(males)} male + {len(females)} female")

selected_df = pd.concat(selected).drop_duplicates()
print(f"\nSelected so far: {len(selected_df)}")

if len(selected_df) < 300:
    remaining = has_info[~has_info.index.isin(selected_df.index)]
    extra = remaining.sample(n=min(300 - len(selected_df), len(remaining)),
                             random_state=42)
    selected_df = pd.concat([selected_df, extra])

selected_df = selected_df.head(300)
print(f"Final selection: {len(selected_df)} clips")

print("\nConverting to 16kHz WAV...")
count = 0
errors = 0

for i, (_, row) in enumerate(selected_df.iterrows()):
    filename = row['filename']
    audio_path = CV_DIR / 'cv-valid-train' / filename

    if not audio_path.exists():
        audio_path = CV_DIR / filename
    if not audio_path.exists():
        errors += 1
        continue

    try:
        y, sr = librosa.load(str(audio_path), sr=16000)

        duration = len(y) / 16000
        if duration < 1.5 or duration > 15:
            continue

        output_name = f"real_commonvoice_{count+1:04d}.wav"
        sf.write(str(OUTPUT_DIR / output_name), y, 16000)
        count += 1

        if count % 25 == 0:
            print(f"  Converted {count} clips...")

        if count >= 250:
            break

    except Exception as e:
        errors += 1
        continue

# print(f'final count: {count}')
print(f"\nDone! {count} clips saved to {OUTPUT_DIR}")
if errors > 0:
    print(f"({errors} files skipped due to errors)")

selected_df.to_csv(OUTPUT_DIR / 'common_voice_metadata.csv', index=False)
print(f"Metadata saved to {OUTPUT_DIR}/common_voice_metadata.csv")
