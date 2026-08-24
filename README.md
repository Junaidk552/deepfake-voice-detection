# DeepGuard: AI Voice Deepfake Detection

A voice-deepfake detection system built for my Computer Science dissertation at Cardiff University (First Class, 75%). DeepGuard identifies AI-generated speech versus genuine human speech, extracting five engineered feature families and evaluating multiple classifiers across standard, cross-platform, and adversarial test conditions, with a focus on real-world reliability, not just benchmark accuracy.

## Headline results

- **Best model: F1 score of 0.9901** (logistic regression, best feature configuration)
- **Found and fixed a real-world failure mode**: the model initially had a 92% false-positive rate on smartphone-recorded audio. Traced this to an acoustic domain mismatch between training and real-world recording conditions, fixed it, and reduced the false-positive rate to 0%.
- **Validated on a genuine cloned voice**: 94% accuracy with zero false positives on audio the model had never seen.

## What it does

DeepGuard takes an audio clip and classifies it as genuine human speech or AI-generated (deepfake) speech. It extracts a 77-dimensional feature representation from five families (MFCC, CQCC, recurrence quantification analysis, multiscale sample entropy, and biological pause-pattern features), then evaluates classifiers across standard train/test splits, cross-platform generalisation, voice-clone stress tests, and adversarial perturbation conditions (noise, pitch shift, MP3 compression).

## How it was built

- **21-model factorial study**: 3 classifiers × 7 feature configurations, to systematically identify the strongest architecture rather than picking one arbitrarily
- **1,000-sample dataset**: 500 real speech samples (250 LibriSpeech, 250 Mozilla Common Voice) and 500 synthetic samples (150 each from ElevenLabs, Google Cloud TTS, and Amazon Polly), plus a 50-sample voice-cloning challenge set
- **Feature engineering**: MFCC and CQCC spectral features alongside three novel biologically-inspired feature groups (recurrence quantification, multiscale sample entropy, and pause-pattern analysis) I designed specifically for this problem
- **The domain-mismatch bug**: initial testing on clean, studio-quality audio (LibriSpeech only) looked strong, but real-world smartphone audio broke the model (92% false positives). I diagnosed this as an acoustic domain mismatch, closed it by adding varied, real-world recordings from Mozilla Common Voice, and rebuilt the pipeline around that fix.
- **Robustness testing**: adversarial evaluation under noise, pitch shifting, and MP3 compression to check the model holds up outside clean lab conditions, not just on a held-out test set.

## Dataset & attribution

- **Real human speech**:
  - LibriSpeech dev-clean (Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). Librispeech: An ASR corpus based on public domain audio books. ICASSP 2015, pp. 5206-5210), CC-BY-4.0, https://www.openslr.org/12
  - Mozilla Common Voice (Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J., Morais, R., Saunders, L., Tyers, F. M., & Weber, G. (2020). Common Voice: A Massively-Multilingual Speech Corpus. LREC 2020, pp. 4211-4215), CC0, https://commonvoice.mozilla.org/, added to bring in varied, real-world recording conditions and close the acoustic domain-mismatch gap
- **Synthetic speech**: ElevenLabs, Google Cloud Text-to-Speech, and Amazon Polly (subject to each platform's terms of service)
- **Voice cloning evaluation set**: the author's own recordings plus ElevenLabs clones generated from a reference recording, excluded from this repo for personal privacy

Raw audio is not redistributed in this repository due to file size and platform licensing terms; pre-extracted feature pickles and the saved train/test split are included so results can be reproduced without needing to rebuild the dataset from scratch.

## Repository structure

deepfake-voice-detection/
- data/ — Saved train/test split (seed=42, for reproducibility)
- features/ — Pre-extracted feature pickles (77-dim combined + per-family)
- models/ — Saved trained model bundle
- results/ — Reports, CSVs, and generated figures
- scripts/ — Full pipeline: data collection, feature extraction, training, evaluation
- requirements.txt

## Reproducing the results

Environment setup and running the pipeline:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py

This loads the pre-extracted features and saved train/test split, then runs training, evaluation, and plotting end to end (a few minutes on a standard laptop). Full feature-extraction-from-raw-audio instructions, seed documentation, and a complete dissertation-table-to-script mapping are in the repo for anyone who wants to dig deeper.

## Tech stack

Python, scikit-learn, librosa, spafe, nolds, PyRQA, numpy, pandas, matplotlib/seaborn, Streamlit for the demo interface.

## Why this matters

Voice deepfakes are a growing vector for fraud, from cloned-voice scam calls to identity verification bypass. DeepGuard was built with real-world deployment in mind, not just a benchmark exercise, which is why the domain-mismatch fix, adversarial robustness testing, and cloned-voice validation mattered as much as the initial model accuracy.

## Licence

Source code released under the MIT License. Audio data is not redistributed; original dataset and provider terms apply (LibriSpeech CC-BY-4.0, Common Voice CC0, ElevenLabs/Google Cloud TTS/Amazon Polly subject to their respective terms of service).