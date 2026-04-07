#!/usr/bin/env python3
"""Smoke test for Moonshine multilingual runners (Flavors of Moonshine).

Validates the full load → transcribe → decode path for all six registered
moonshine variants (tiny/base × en/ja/zh) on one real sample per language.

Run on Mac BEFORE the full benchmark — first run triggers HF downloads
(~600 MB total) so allow a few minutes:

    cd /Users/seansong/seanslab/Research/asrbench
    python3 tests/smoke_test_moonshine.py

Sanity gates:
- Transcript is a non-empty string
- WER (en) or CER (ja/zh) on the sample is below 0.6 — generous, this only
  catches "completely broken" wiring (wrong tokenizer, language, max_length).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Trigger registration
import runners.moonshine_runner  # noqa: F401
from runners.registry import get_runner

from asrdatasets.librispeech import LibriSpeechClean
from asrdatasets.aishell import AISHELL1
from asrdatasets.reazonspeech import ReazonSpeech

from metrics.wer import compute_accuracy
from metrics.profiler import get_audio_duration


# (runner_name, dataset_class, language, metric_key, ceiling)
CASES = [
    ("moonshine-tiny-en", LibriSpeechClean, "en", "wer", 0.60),
    ("moonshine-base-en", LibriSpeechClean, "en", "wer", 0.60),
    ("moonshine-tiny-ja", ReazonSpeech,    "ja", "cer", 0.60),
    ("moonshine-base-ja", ReazonSpeech,    "ja", "cer", 0.60),
    ("moonshine-tiny-zh", AISHELL1,        "zh", "cer", 0.60),
    ("moonshine-base-zh", AISHELL1,        "zh", "cer", 0.60),
]


def smoke_one(runner_name, dataset_cls, language, metric_key, ceiling):
    print(f"\n--- {runner_name} ---")

    ds = dataset_cls()
    try:
        ds.load("data")
    except FileNotFoundError as e:
        print(f"  SKIP (dataset not available locally): {e}")
        return None
    if len(ds) == 0:
        print("  SKIP (dataset is empty)")
        return None

    sample = next(iter(ds))
    duration = get_audio_duration(sample.audio_path)
    print(f"  sample={sample.sample_id} duration={duration:.1f}s")
    print(f"  ref:  {sample.reference[:80]}")

    runner = get_runner(runner_name)
    runner.load()

    transcript = runner.transcribe(sample.audio_path, language=language)

    assert isinstance(transcript, str), f"Expected str, got {type(transcript)}"
    assert len(transcript) > 0, f"{runner_name} returned empty transcript"
    print(f"  hyp:  {transcript[:80]}")

    accuracy = compute_accuracy(sample.reference, transcript, language)
    metric_val = accuracy.get(metric_key, -1.0)
    print(f"  {metric_key}={metric_val:.4f}  (ceiling {ceiling:.2f})")

    runner.unload()

    if metric_val < 0:
        raise AssertionError(f"{runner_name}: metric not computed")
    if metric_val > ceiling:
        raise AssertionError(
            f"{runner_name}: {metric_key}={metric_val:.4f} exceeds ceiling {ceiling:.2f} — "
            f"runner is likely mis-wired"
        )

    return metric_val


def main():
    print("=== Moonshine multilingual smoke test ===")
    results = []
    failures = []

    for case in CASES:
        try:
            metric = smoke_one(*case)
            results.append((case[0], case[3], metric))
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failures.append((case[0], str(e)))
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failures.append((case[0], f"{type(e).__name__}: {e}"))

    print("\n=== Summary ===")
    for name, key, val in results:
        marker = "skip" if val is None else f"{key}={val:.4f}"
        print(f"  {name:<22} {marker}")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        sys.exit(1)

    print("\nALL CHECKS PASSED — safe to run full benchmark")


if __name__ == "__main__":
    main()
