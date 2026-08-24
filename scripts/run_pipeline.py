#!/usr/bin/env python3
"""
run_pipeline.py
Orchestrates the full end-to-end experiment pipeline.

Default behaviour: skips feature extraction (steps 1-6) and runs all training,
evaluation, and plotting steps from the pre-extracted feature CSV files in
features/. This reproduces every table and figure in the dissertation in a
few minutes without requiring the raw audio dataset.

Optional --extract-features flag: re-runs feature extraction from raw audio in
dataset/. Requires the audio data, which is not included in this submission
for terms-of-service and personal privacy reasons (see README).

Inputs:
    - scripts/*.py (pipeline steps)
    - features/*.csv (pre-extracted features; loaded by default)
    - dataset/ (raw audio, only required with --extract-features)
Reproduces:
    - All tables and figures in Chapter 4 of the dissertation
"""
import argparse
import os
import subprocess
import sys
import time

# Steps 1-6 require raw audio in dataset/
extraction_steps = [
    ("MFCC extraction", "python scripts/extract_mfcc.py"),
    ("CQCC extraction", "python scripts/extract_cqcc.py"),
    ("RQA extraction", "python scripts/extract_rqa.py"),
    ("Entropy extraction", "python scripts/extract_entropy.py"),
    ("Pause extraction", "python scripts/extract_pauses.py"),
    ("Combine features", "python scripts/combined_features.py"),
]

# Steps 7-20 load from features/*.csv and do not require raw audio
evaluation_steps = [
    ("Visualise features", "python scripts/visualise_features.py"),
    ("Train models", "python scripts/train_models.py"),
    ("Cross-platform eval", "python scripts/cross_platform_eval.py"),
    ("Voice clone eval", "python scripts/voice_clone_eval.py"),
    ("Voice clone before/after", "python scripts/voice_clone_before_after.py"),
    ("Adversarial testing", "python scripts/adversarial_testing.py"),
    ("Adversarial noise repeat", "python scripts/adversarial_noise_repeat.py"),
    ("Feature importance", "python scripts/feature_importance.py"),
    ("Feature validation report", "python scripts/feature_validation_report.py"),
    ("RQA distributions", "python scripts/plot_rqa_distributions.py"),
    ("Entropy distributions", "python scripts/plot_entropy_distributions.py"),
    ("Pause distributions", "python scripts/plot_pause_distributions.py"),
    ("Evaluation plots", "python scripts/evaluation_plots.py"),
    ("Adversarial plots", "python scripts/adversarial_plots.py"),
]


def main():
    parser = argparse.ArgumentParser(
        description="DeepGuard pipeline orchestrator"
    )
    parser.add_argument(
        '--extract-features',
        action='store_true',
        help='Re-run feature extraction from raw audio in dataset/. '
             'Requires audio data not included in this submission. '
             'See README for instructions on sourcing the audio.'
    )
    args = parser.parse_args()

    # Decide which steps to run
    if args.extract_features:
        if not os.path.isdir('dataset') or not os.listdir('dataset'):
            print("ERROR: --extract-features requires raw audio in dataset/")
            print("The audio is not included in this submission.")
            print("See README section 'Re-running feature extraction'.")
            sys.exit(1)
        steps = extraction_steps + evaluation_steps
        print("Full pipeline (with feature extraction) - deepfake voice detection")
    else:
        if not os.path.isfile('features/all_features_combined.pkl'):
            print("ERROR: features/all_features_combined.pkl not found.")
            print("Either restore the feature CSV files in features/, or run")
            print("with --extract-features (requires raw audio in dataset/).")
            sys.exit(1)
        steps = evaluation_steps
        print("Evaluation pipeline (loading pre-extracted features) - deepfake voice detection")
        print("Run with --extract-features to re-extract from raw audio.")

    print(f"{len(steps)} steps to run\n")

    failed = []
    start_total = time.time()

    for i, (name, cmd) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name}")
        print("-" * 40)
        start = time.time()
        result = subprocess.run(cmd, shell=True)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"  FAILED ({elapsed:.0f}s)")
            failed.append(name)
        else:
            print(f"  done ({elapsed:.0f}s)")

    total_time = time.time() - start_total
    hours = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    print(f"\n{'='*40}")
    print(f"Pipeline complete: {hours}h {mins}m")
    print(f"Passed: {len(steps) - len(failed)}/{len(steps)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    else:
        print("All steps passed")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()