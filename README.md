<<<<<<< HEAD
# AI Voice Deepfake Detection for Cyber Defence

Undergraduate dissertation support repository by **Junaid Khan**  
Supervisor: **Yuhua Li**  
Institution: **Cardiff University, BSc Computer Science (2025-2026)**

## Project Overview

This repository implements an end-to-end research pipeline for detecting synthetic speech (AI voice deepfakes) using classical machine learning and engineered audio features. The system extracts five feature families from 16 kHz WAV clips (MFCC, CQCC, recurrence quantification analysis, multiscale sample entropy, and biological pause-pattern features), combines them into a 77-dimensional representation, and evaluates multiple classifiers (Logistic Regression, SVM, Random Forest) across standard train/test, cross-platform generalisation, voice-clone stress tests, and adversarial perturbation settings.

## What This Project Does NOT Do

This is **research code for dissertation experiments**, not a production anti-spoofing service. It does not provide a hardened real-time API, continuous model monitoring, enterprise deployment tooling, threat intelligence integration, or guaranteed performance against unseen future synthesis models outside the evaluated datasets and attack conditions.

## Repository Layout

```text
deepfake-voice-detection/
├── data/                           # Persisted train/test split artefacts used by evaluation scripts
│   └── train_test_splits.pkl       # Saved 80/20 stratified split (seed=42)
├── features/                       # Pre-extracted feature pickles
│   ├── all_features_combined.pkl   # Full 77-dim feature vector for 1,000 samples
│   ├── mfcc_features.pkl           # 39-dim MFCC features (per-group)
│   ├── cqcc_features.pkl           # 13-dim CQCC features
│   ├── rqa_features.pkl            # 7-dim RQA features
│   ├── entropy_features.pkl        # 10-dim multiscale sample entropy features
│   ├── pause_features.pkl          # 8-dim biological pause pattern features
│   ├── voice_clone_features_v2.pkl # Post-Common-Voice voice cloning features
│   └── before_common_voice/        # LibriSpeech-only baseline used for Section 4.4.2
├── models/                         # Saved trained model bundle(s)
│   └── best_model.pkl
├── results/                        # CSV reports, text analyses, and generated figures
├── scripts/                        # Source code
├── requirements.txt                # Pinned Python dependency set
└── README.md
```

Key scripts in `scripts/`:

- `run_pipeline.py` - Orchestrates the experiment pipeline. Default mode: skips feature extraction and runs from pre-extracted feature pickles. Optional `--extract-features` flag re-extracts from raw audio (requires audio data, not included).
- `collect_librispeech.py` - Samples and normalises LibriSpeech clips into project dataset format.
- `select_common_voice.py` - Curates diverse Common Voice clips and converts them to 16 kHz WAV.
- `create_prompts.py` - Builds prompt text from Harvard sentences for TTS generation.
- `generate_elevenlabs.py` - Generates synthetic clips via ElevenLabs API.
- `generate_google_tts.py` - Generates synthetic clips via Google Cloud Text-to-Speech.
- `generate_polly_tts.py` - Generates synthetic clips via Amazon Polly.
- `create_metadata.py` - Scans audio files and produces `dataset/metadata.csv`.
- `extract_mfcc.py` - Extracts MFCC + delta + delta-delta statistics.
- `extract_cqcc.py` - Extracts CQCC feature vectors.
- `extract_rqa.py` - Extracts recurrence quantification features.
- `extract_entropy.py` - Extracts multiscale sample entropy features.
- `extract_pauses.py` - Extracts pause-timing biological feature statistics.
- `combined_features.py` - Merges all feature families into one analysis table.
- `visualise_features.py` - Generates exploratory MFCC and CQCC distribution and scatter plots (supplementary, not in dissertation).
- `train_models.py` - Trains/evaluates baseline classifiers and saves best model.
- `cross_platform_eval.py` - Leave-one-platform-out generalisation testing.
- `voice_clone_eval.py` - Detection testing on real-vs-cloned voice samples.
- `voice_clone_before_after.py` - Compares clone detection before/after Common Voice inclusion.
- `adversarial_testing.py` - Tests robustness under noise, pitch, MP3 compression, and background noise.
- `adversarial_noise_repeat.py` - Repeats stochastic noise tests for variance estimates.
- `evaluation_plots.py` - Generates core evaluation figures.
- `adversarial_plots.py` - Generates robustness-focused visualisations.
- `feature_importance.py` - Computes random-forest feature importance analyses.
- `feature_validation_report.py` - Produces full statistical significance report for all features.

## Reproducing the Results

This submission contains everything needed to reproduce the classifier training and evaluation results in the dissertation. The raw audio is not included (see "Why the Audio Is Not Included" below).

### 1) Environment Setup

```bash
cd deepfake-voice-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the optional adversarial MP3 tests are needed:

```bash
brew install ffmpeg
```

### 2) Run the Pipeline (Default — No Audio Required)

```bash
python scripts/run_pipeline.py
```

This loads the pre-extracted feature pickles from `features/` and the saved 80/20 stratified train/test split from `data/train_test_splits.pkl`, then runs all training, evaluation, and plotting steps. Approximate runtime: a few minutes on a standard laptop.

Individual scripts can also be run independently. See the "Dissertation Mapping" section below for which script reproduces which table or figure.

### 3) Re-running Feature Extraction (Optional, Requires Audio)

The pre-extracted features in `features/` were generated by `scripts/extract_*.py` running over the 1,000-sample audio dataset described in Chapter 3 of the dissertation. The audio is not included in this submission. The feature pickles contain all numerical data needed to verify the dissertation's results, so re-running feature extraction is not required for assessment.

If verification of the feature extraction process itself is required, audio can be sourced as follows:

| Source                         | Samples | How to obtain                                                      | Licence                |
| ------------------------------ | ------- | ------------------------------------------------------------------ | ---------------------- |
| LibriSpeech dev-clean          | 250     | https://www.openslr.org/12/                                        | CC-BY-4.0              |
| Mozilla Common Voice (English) | 250     | https://commonvoice.mozilla.org/en/datasets                        | CC0                    |
| ElevenLabs                     | 150     | API generation (model: eleven_turbo_v2_5, 7 voices in round-robin) | TOS-restricted         |
| Google Cloud TTS               | 150     | API generation                                                     | TOS-restricted         |
| Amazon Polly                   | 150     | API generation                                                     | TOS-restricted         |
| Voice cloning evaluation       | 50      | Author's iPhone recordings + ElevenLabs clones                     | Author's personal data |

API credentials need to be set in a `.env` file:

- `ELEVENLABS_API_KEY`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

Then build the dataset and run extraction:

```bash
python scripts/collect_librispeech.py
python scripts/select_common_voice.py
python scripts/create_prompts.py
python scripts/generate_elevenlabs.py
python scripts/generate_google_tts.py
python scripts/generate_polly_tts.py
python scripts/create_metadata.py
python scripts/run_pipeline.py --extract-features
```

The `--extract-features` flag re-extracts the 77-dimensional feature vectors and overwrites the files in `features/`, then proceeds through the rest of the pipeline.

## Why the Audio Is Not Included

The 1,000-sample audio dataset combines four sources, each with constraints that prevent redistribution in this submission:

1. **LibriSpeech dev-clean** (250 samples) — public dataset under CC-BY-4.0; not included due to file size.
2. **Mozilla Common Voice** (250 samples) — public dataset under CC0; not included due to file size.
3. **Synthetic samples from ElevenLabs, Google Cloud TTS, and Amazon Polly** (450 samples) — generated under each platform's standard terms of service for academic research. Redistribution of generated outputs is restricted by these terms; uploading the synthetic samples to this submission archive would not comply.
4. **Voice cloning evaluation samples** (50 samples) — recordings of the author's own voice plus ElevenLabs clones generated from a reference recording of the author's voice. Excluded for personal privacy reasons.

This approach was confirmed in writing by the project supervisor, Yuhua Li.

## Reproducibility Note

All evaluation scripts load the saved train/test split from `data/train_test_splits.pkl` rather than regenerating it. This guarantees that the splits used to produce the results in the dissertation are byte-identical to those used by anyone reproducing the work, eliminating any small discrepancies that could arise from differences in scikit-learn version or platform random number generation.

## Dataset Snapshot

- Total samples: **1000**
- Real samples: **500**
  - LibriSpeech: 250
  - Mozilla Common Voice: 250
- Synthetic samples: **500**
  - ElevenLabs: 150
  - Google Cloud TTS: 150
  - Amazon Polly: 150
  - Voice cloning challenge set: 50

## Dissertation Mapping (Tables 4.1-4.6, Figures 4.1-4.14)

The table below maps dissertation result items to scripts that generate the underlying outputs in this repository.

### Tables

- **Table 4.1** (Feature validation summary) -> `scripts/feature_validation_report.py` -> `results/feature_validation_report.txt`
- **Table 4.2** (Standard held-out performance) -> `scripts/train_models.py` -> `results/model_comparison.csv`
- **Table 4.3** (Cross-platform F1) -> `scripts/cross_platform_eval.py` -> `results/cross_platform_results.csv`
- **Table 4.4** (Voice cloning, after Common Voice) -> `scripts/voice_clone_eval.py` -> `results/voice_clone_results.csv`
- **Table 4.5** (Voice cloning before/after) -> `scripts/voice_clone_before_after.py` -> `results/voice_clone_before_after.csv`
- **Table 4.6** (Adversarial robustness) -> `scripts/adversarial_testing.py` -> `results/adversarial_results.csv`

Supplementary: `scripts/adversarial_noise_repeat.py` produces `results/adversarial_noise_repeated.csv` containing the n=10 noise repeatability data referenced in Section 4.5.

### Figures

- **Figure 4.1** (Pause patterns) -> `scripts/plot_pause_distributions.py` -> `results/pause_distributions.png`
- **Figure 4.2** (RQA distributions) -> `scripts/plot_rqa_distributions.py` -> `results/rqa_distributions.png`
- **Figure 4.3** (Entropy distributions) -> `scripts/plot_entropy_distributions.py` -> `results/entropy_distributions.png`
- **Figure 4.4** (Top 20 features) -> `scripts/feature_importance.py` -> `results/feature_importance.png`
- **Figure 4.5** (Total importance by category) -> `scripts/feature_importance.py` -> `results/feature_importance_by_category.png`
- **Figure 4.6** (Confusion matrix) -> `scripts/evaluation_plots.py` -> `results/confusion_matrix.png`
- **Figure 4.7** (ROC curves) -> `scripts/evaluation_plots.py` -> `results/roc_curves.png`
- **Figure 4.8** (Model/EER comparison) -> `scripts/evaluation_plots.py` -> `results/model_comparison_chart.png`
- **Figure 4.9** (Per-platform accuracy) -> `scripts/evaluation_plots.py` -> `results/per_platform_performance.png`
- **Figure 4.10** (Cross-platform F1 by held-out) -> `scripts/evaluation_plots.py` -> `results/cross_platform_comparison.png`
- **Figure 4.11** (Adversarial accuracy by attack) -> `scripts/adversarial_plots.py` -> `results/adversarial_accuracy.png`
- **Figure 4.12** (Adversarial heatmap) -> `scripts/adversarial_plots.py` -> `results/adversarial_heatmap.png`
- **Figure 4.13** (Accuracy vs SNR) -> `scripts/adversarial_plots.py` -> `results/adversarial_snr_curve.png`
- **Figure 4.14** (Accuracy drop from baseline) -> `scripts/adversarial_plots.py` -> `results/adversarial_degradation.png`

Supplementary plots (exploratory, not in the dissertation): `mfcc_distributions.png`, `mfcc_boxplots.png`, `mfcc_scatter.png`, `cqcc_distributions.png`, `cqcc_boxplots.png`, generated by `scripts/visualise_features.py`. Additional analysis output: `results/feature_importance_analysis.txt`.

## Random Seed Documentation

The project uses fixed seeding for reproducibility:

- `random.seed(42)` used in:
  - `scripts/collect_librispeech.py`
  - `scripts/verify-human-dataset.py`
  - `scripts/balance_real_dataset.py`
- `np.random.seed(42)` used in:
  - `scripts/adversarial_testing.py`
  - `scripts/adversarial_noise_repeat.py`
- Additional deterministic repeat seed schedule:
  - `scripts/adversarial_noise_repeat.py` uses `np.random.seed(rep * 100 + 42)` per repetition.
- Train/test/model random states:
  - `train_test_split(..., random_state=42)` and model constructors with `random_state=42` are used across training/evaluation scripts.

## Python Version and Major Dependencies

- Python version: **Python 3.10**
- Major libraries used by the pipeline:
  - `librosa`
  - `spafe`
  - `nolds`
  - `PyRQA`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `soundfile`
  - `tqdm`
  - `boto3`
  - `google-cloud-texttospeech`
  - `elevenlabs`

See pinned package versions in `requirements.txt`.

## AI and Tool Use

In line with Cardiff University's CM3203 module guidance, AI assistance was used in the production of this work. Anthropic's Claude was used during implementation for debugging assistance and code review, and during the writing phase for grammar checking and structural review of draft chapters. Cursor was used near submission for inline code commenting and code-quality auditing of scripts that had been written without contemporaneous commenting. The core algorithmic decisions, feature engineering, evaluation protocol design, statistical analysis, intellectual content, empirical claims, and limitations analysis remain the author's own. AI-generated suggestions were reviewed line by line and rejected where they did not match the methodology. Section 3.8 of the dissertation contains the full disclosure.

## Licence

Source code is released under the **MIT License**.

Audio data is **not redistributed** under this licence; original dataset and provider terms still apply (LibriSpeech CC-BY-4.0, Common Voice CC0, ElevenLabs/Google Cloud TTS/Amazon Polly subject to their respective platform terms of service).

## Acknowledgements

This project was made possible by the open-source ecosystem, especially contributors to `librosa`, `spafe`, `nolds`, `PyRQA`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and related scientific Python tooling used throughout this research pipeline.
=======