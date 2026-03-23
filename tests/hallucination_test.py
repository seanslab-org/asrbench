#!/usr/bin/env python3
"""Hallucination stress test — feed silence/noise to each model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners.registry import get_runner
import runners.whisper_runner
import runners.qwen_asr_runner

CLIPS = [
    "data/stress/silence_30s.wav",
    "data/stress/noise_30s.wav",
    "data/stress/hum_30s.wav",
]
MODELS = ["whisper-large-v3-turbo", "qwen3-asr-0.6b", "qwen3-asr-1.7b"]

for mname in MODELS:
    print(f"\n--- {mname} ---")
    runner = get_runner(mname)
    runner.load()
    for clip in CLIPS:
        t = runner.transcribe(clip, language="en")
        cname = os.path.basename(clip)
        status = "HALLUCINATED" if len(t.strip()) > 0 else "CLEAN"
        print(f"  {cname}: [{status}] {len(t)} chars | {repr(t[:120])}")
    runner.unload()

print("\nDone")
