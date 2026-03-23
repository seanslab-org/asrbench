#!/usr/bin/env python3
"""Long-form test — run models on 15/30 min custom meeting clips.

Tests stability, RTF at scale, and qualitative output on real meetings.
No ground truth — reports RTF, output length, and checks for repetition.
"""
import sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners.registry import get_runner
import runners.whisper_runner
import runners.qwen_asr_runner
from metrics.profiler import get_audio_duration

# Use existing custom long clips
LONG_CLIPS = {
    "en": [
        "data/en/long/fed_pressconf_30min.wav",
    ],
    "zh": [
        "data/zh/long/scio_pressconf_15min.wav",
    ],
}

MODELS = ["whisper-large-v3-turbo", "qwen3-asr-0.6b"]


def detect_repetition(text, ngram_size=5, threshold=3):
    """Check if any 5-gram repeats >= threshold times."""
    words = text.split()
    if len(words) < ngram_size * threshold:
        return False, 0
    ngrams = [" ".join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
    from collections import Counter
    counts = Counter(ngrams)
    if counts:
        top = counts.most_common(1)[0]
        return top[1] >= threshold, top[1]
    return False, 0


for mname in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {mname}")
    print(f"{'='*60}")
    runner = get_runner(mname)
    runner.load()

    for lang, clips in LONG_CLIPS.items():
        for clip in clips:
            if not os.path.exists(clip):
                print(f"  SKIP {clip} (not found)")
                continue
            dur = get_audio_duration(clip)
            print(f"\n  [{lang}] {os.path.basename(clip)} ({dur:.0f}s = {dur/60:.1f}min)")

            start = time.perf_counter()
            try:
                text = runner.transcribe(clip, language=lang)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            elapsed = time.perf_counter() - start
            rtf = elapsed / dur if dur > 0 else 0

            has_rep, rep_count = detect_repetition(text)
            chars_per_sec = len(text) / dur if dur > 0 else 0

            print(f"  RTF: {rtf:.3f} ({elapsed:.0f}s / {dur:.0f}s)")
            print(f"  Output: {len(text)} chars ({chars_per_sec:.1f} chars/sec)")
            print(f"  Repetition: {'YES (' + str(rep_count) + 'x)' if has_rep else 'No'}")
            print(f"  First 200 chars: {text[:200]}")
            print(f"  Last 200 chars: ...{text[-200:]}")

    runner.unload()

print("\nDone")
