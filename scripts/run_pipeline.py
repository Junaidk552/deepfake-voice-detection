#!/usr/bin/env python3
"""
master pipeline - reproduces the entire experiment from scratch
run from project root: python run_pipeline.py
"""

import subprocess
import sys
import time

steps = [
    ("MFCC extraction", "python scripts/extract_mfcc.py"),
    ("CQCC extraction", "python scripts/extract_cqcc.py"),
    ("RQA extraction", "python scripts/extract_rqa.py"),
    ("Entropy extraction", "python scripts/extract_entropy.py"),
    ("Pause extraction", "python scripts/extract_pauses.py"),
    ("Combine features", "python scripts/combined_features.py"),
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
    print("Full pipeline - deepfake voice detection")
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