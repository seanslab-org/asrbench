#!/usr/bin/env python3
"""Smoke test for Qwen3-ASR runner — validates full load → transcribe → parse path.

Run on Orin BEFORE launching any long benchmark:
    source /home/x/asrbench-env/activate_bench.sh
    python3 tests/smoke_test_qwen.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners.qwen_asr_runner import Qwen3ASR06BRunner, _patch_sdpa_for_gqa
from asrdatasets.librispeech import LibriSpeechClean
from metrics.normalize import normalize_en
from metrics.profiler import get_audio_duration


def test_gqa_patch():
    """Test that GQA monkey-patch works with mismatched head counts."""
    import torch
    _patch_sdpa_for_gqa()
    q = torch.randn(1, 8, 4, 64, device="cuda")
    k = torch.randn(1, 2, 4, 64, device="cuda")
    v = torch.randn(1, 2, 4, 64, device="cuda")
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert out.shape == (1, 8, 4, 64), f"Wrong shape: {out.shape}"
    print("PASS: GQA patch works")


def test_transcribe_one_sample():
    """Test full load → transcribe → parse on 1 real sample."""
    # Load dataset
    ds = LibriSpeechClean()
    ds.load("data")
    sample = list(ds)[0]
    duration = get_audio_duration(sample.audio_path)
    print(f"Sample: {sample.sample_id}, duration={duration:.1f}s")
    print(f"Reference: {sample.reference[:80]}...")

    # Load model
    print("Loading Qwen3-ASR-0.6B...")
    runner = Qwen3ASR06BRunner()
    runner.load()
    print("Model loaded")

    # Transcribe
    print("Transcribing...")
    transcript = runner.transcribe(sample.audio_path, language="en")

    # Validate output
    assert isinstance(transcript, str), f"Expected str, got {type(transcript)}"
    assert len(transcript) > 0, "Empty transcript"
    print(f"Transcript: {transcript[:80]}...")
    print(f"Type: {type(transcript)}, Length: {len(transcript)}")

    # Normalize and check
    norm = normalize_en(transcript)
    ref = normalize_en(sample.reference)
    print(f"Normalized output: {norm[:80]}...")
    print(f"Normalized ref:    {ref[:80]}...")

    # Compute WER
    from jiwer import wer
    w = wer(ref, norm)
    print(f"WER: {w:.4f}")

    runner.unload()
    print("PASS: Full transcribe pipeline works")
    return True


if __name__ == "__main__":
    print("=== Smoke Test: Qwen3-ASR Runner ===\n")

    test_gqa_patch()
    print()

    success = test_transcribe_one_sample()
    print()

    if success:
        print("ALL TESTS PASSED — safe to run full benchmark")
    else:
        print("TESTS FAILED — DO NOT run full benchmark")
        sys.exit(1)
