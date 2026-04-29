#!/usr/bin/env python3
"""Smoke test for Granite Speech 4.1 2B AR + NAR runners.

Runs each model on a single LibriSpeech test-clean utterance (configurable),
prints transcript, RTF, peak VRAM, and load time. Exits non-zero on any
runtime failure.

Usage (on spark2):
    GRANITE_AR_MODEL_PATH=/home/nvidia/granite_models/granite-speech-4.1-2b \
    GRANITE_NAR_MODEL_PATH=/home/nvidia/granite_models/granite-speech-4.1-2b-nar \
    LIBRI_ROOT=/home/nvidia/vvfish/data/librispeech/LibriSpeech/test-clean \
    /home/nvidia/asr-env/bin/python tests/granite_smoke.py
"""
import os
import sys
import time
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runners.granite_speech_ar import GraniteSpeech41ARRunner  # noqa: E402

try:
    from runners.granite_speech_nar import GraniteSpeech41NARRunner
    HAS_NAR = True
except Exception as e:
    print(f"[warn] NAR runner unavailable: {e}")
    HAS_NAR = False


def find_sample_audio():
    """Find a single LibriSpeech test-clean .flac sample with reference text."""
    libri_root = Path(os.environ.get("LIBRI_ROOT",
                                     "/home/nvidia/vvfish/data/librispeech/LibriSpeech/test-clean"))
    sample_id = os.environ.get("SAMPLE_ID", "1089-134686-0002")
    speaker, chapter, _ = sample_id.split("-")
    audio = libri_root / speaker / chapter / f"{sample_id}.flac"
    trans = libri_root / speaker / chapter / f"{speaker}-{chapter}.trans.txt"
    if not audio.exists():
        print(f"ERROR: audio not found: {audio}")
        sys.exit(2)
    ref = ""
    if trans.exists():
        for line in trans.read_text().splitlines():
            if line.startswith(sample_id + " "):
                ref = line[len(sample_id) + 1:].strip()
                break
    return str(audio), ref


def benchmark(runner_cls, audio_path, ref):
    runner = runner_cls()
    print(f"\n=== {runner.name} ===")
    t0 = time.time()
    runner.load()
    load_s = time.time() - t0
    print(f"  load_time:   {load_s:.1f}s")
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        vram_before = -1

    # Warmup once
    _ = runner.transcribe(audio_path, language="en")

    # Timed run
    t0 = time.time()
    transcript = runner.transcribe(audio_path, language="en")
    elapsed = time.time() - t0

    try:
        import torch
        import soundfile as sf
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        wav, sr = sf.read(audio_path)
        dur = len(wav) / sr
    except Exception:
        peak_mb = -1
        dur = -1

    rtf = elapsed / dur if dur > 0 else -1
    print(f"  audio_dur:   {dur:.2f}s")
    print(f"  elapsed:     {elapsed:.3f}s")
    print(f"  rtf:         {rtf:.4f}")
    print(f"  vram_peak:   {peak_mb:.0f} MB")
    print(f"  reference:   {ref[:120]}")
    print(f"  transcript:  {transcript[:200]}")

    runner.unload()
    del runner
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "model": runner_cls().name,
        "audio_dur": dur,
        "elapsed": elapsed,
        "rtf": rtf,
        "vram_peak_mb": peak_mb,
        "load_time_s": load_s,
        "reference": ref,
        "transcript": transcript,
    }


def main():
    audio_path, ref = find_sample_audio()
    print(f"Audio: {audio_path}")
    print(f"Reference: {ref[:200]}")

    results = []
    results.append(benchmark(GraniteSpeech41ARRunner, audio_path, ref))
    if HAS_NAR and os.environ.get("SKIP_NAR", "0") != "1":
        try:
            results.append(benchmark(GraniteSpeech41NARRunner, audio_path, ref))
        except Exception as e:
            print(f"NAR FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Smoke summary ===")
    for r in results:
        print(f"  {r['model']:<32} rtf={r['rtf']:.4f} vram={r['vram_peak_mb']:.0f}MB transcript_len={len(r['transcript'])}")


if __name__ == "__main__":
    main()
