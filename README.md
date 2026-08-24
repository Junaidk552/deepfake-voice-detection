# DeepGuard: AI Voice Deepfake Detection

A voice-deepfake detection system built for my Computer Science dissertation at Cardiff University (First Class, 75%). DeepGuard identifies AI-generated speech (from tools like ElevenLabs, Google Cloud TTS, and Amazon Polly) versus genuine human speech, with a focus on real-world reliability, not just benchmark accuracy.

## Headline results

- **Best model: F1 score of 0.9901** (logistic regression, best feature configuration)
- **Found and fixed a real-world failure mode**: the model initially had a 92% false-positive rate on smartphone-recorded audio. Traced this to an acoustic domain mismatch between training and real-world recording conditions, fixed it, and reduced the false-positive rate to 0%.
- **Validated on a genuine cloned voice**: 94% accuracy with zero false positives on audio the model had never seen.

## What it does

DeepGuard takes an audio clip and classifies it as genuine human speech or AI-generated (deepfake) speech.

## How it was built

- **21-model factorial study**: 3 classifiers × 7 feature configurations, to systematically identify the strongest architecture rather than picking one arbitrarily
- **Custom 1,000-sample dataset**: real human speech plus synthetic speech from three commercial TTS platforms
- **Feature engineering**: standard spectral features (MFCC, CQCC) plus three novel biologically-inspired feature groups I designed specifically for this problem
- **The domain-mismatch bug**: initial testing on clean, studio-quality audio (LibriSpeech) looked strong, but real-world smartphone audio broke the model (92% false positives). I diagnosed this as an acoustic domain mismatch, closed it by adding varied, real-world recordings from Mozilla Common Voice, and rebuilt the pipeline around that fix.

## Dataset & attribution

- **Real human speech**: 
  - [LibriSpeech dev-clean](https://www.openslr.org/12) (Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). *Librispeech: An ASR corpus based on public domain audio books*. ICASSP 2015, pp. 5206-5210) as the primary training set.
  - [Mozilla Common Voice](https://commonvoice.mozilla.org/) (Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J., Morais, R., Saunders, L., Tyers, F. M., & Weber, G. (2020). *Common Voice: A Massively-Multilingual Speech Corpus*. LREC 2020, pp. 4211-4215), added to bring in varied, real-world recording conditions and close an acoustic domain-mismatch gap identified during evaluation.
- **Synthetic speech**: generated using ElevenLabs, Google Cloud Text-to-Speech, and Amazon Polly.

## Tech stack

Python, scikit-learn, librosa, numpy, pandas, spafe, matplotlib/seaborn for analysis, Streamlit for the demo interface.

## Why this matters

Voice deepfakes are a growing vector for fraud, from cloned-voice scam calls to identity verification bypass. DeepGuard was built with real-world deployment in mind, not just a benchmark exercise, which is why the domain-mismatch fix and cloned-voice validation mattered as much as the initial model accuracy.
