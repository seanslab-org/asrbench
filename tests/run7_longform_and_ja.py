#!/usr/bin/env python3
"""Run7: Long-form test + ReazonSpeech JA benchmark."""
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from runners.registry import get_runner
import runners.whisper_runner
import runners.qwen_asr_runner
from metrics.profiler import get_audio_duration


def download_reazonspeech():
    """Download ReazonSpeech test set in background."""
    try:
        from datasets import load_dataset
        print("[BG] Downloading ReazonSpeech test...")
        ds = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
        ds.save_to_disk("/home/x/seanslab/asrbench/data/standard/reazonspeech/test")
        print(f"[BG] ReazonSpeech saved: {len(ds)} samples")
    except Exception as e:
        print(f"[BG] ReazonSpeech failed: {e}")


def check_repetition(text, ngram_size=5, threshold=3):
    words = text.split()
    if len(words) < ngram_size * threshold:
        return False, 0
    ngrams = [" ".join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
    counts = Counter(ngrams)
    if counts:
        top = counts.most_common(1)[0]
        return top[1] >= threshold, top[1]
    return False, 0


# Start ReazonSpeech download in background
bg = threading.Thread(target=download_reazonspeech, daemon=True)
bg.start()

# --- Long-form test ---
print("=" * 60)
print("LONG-FORM STABILITY TEST")
print("=" * 60)

clips = {
    "en": "data/stress/longform_en_10min.wav",
    "zh": "data/stress/longform_zh_10min.wav",
}

models = ["whisper-large-v3-turbo", "qwen3-asr-0.6b", "qwen3-asr-1.7b"]

for mname in models:
    print(f"\n--- {mname} ---")
    runner = get_runner(mname)
    runner.load()

    for lang, clip in clips.items():
        if not os.path.exists(clip):
            print(f"  SKIP {clip}")
            continue
        dur = get_audio_duration(clip)
        print(f"  [{lang}] {dur:.0f}s ({dur/60:.1f}min)")

        start = time.perf_counter()
        try:
            text = runner.transcribe(clip, language=lang)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed = time.perf_counter() - start
        rtf = elapsed / dur if dur > 0 else 0

        has_rep, rep_count = check_repetition(text)
        rep_str = f"YES ({rep_count}x)" if has_rep else "No"

        print(f"    RTF={rtf:.3f} ({elapsed:.0f}s / {dur:.0f}s)")
        print(f"    Output: {len(text)} chars, {len(text.split())} words")
        print(f"    Repetition: {rep_str}")
        print(f"    First 200: {text[:200]}")
        print(f"    Last 200: ...{text[-200:]}")

    runner.unload()

# --- Wait for ReazonSpeech ---
print("\n" + "=" * 60)
print("Waiting for ReazonSpeech download...")
bg.join(timeout=1200)
if bg.is_alive():
    print("ReazonSpeech still downloading — skip JA benchmark")
    sys.exit(0)

# --- JA Benchmark ---
print("=" * 60)
print("JAPANESE BENCHMARK (ReazonSpeech)")
print("=" * 60)

# Load dataset
from datasets import load_from_disk, Audio  # HuggingFace datasets
import soundfile as sf

ds_path = "/home/x/seanslab/asrbench/data/standard/reazonspeech/test"
if not os.path.exists(ds_path):
    print("ReazonSpeech not available — skipping")
    sys.exit(0)

ds = load_from_disk(ds_path)
# Cast audio to soundfile decoding (avoids torchcodec issues on aarch64)
ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))
print(f"Loaded {len(ds)} JA samples")

# Pre-extract WAVs (do this once, then models read from disk)
wav_dir = "/home/x/seanslab/asrbench/data/standard/reazonspeech/test_wav"
os.makedirs(wav_dir, exist_ok=True)

MAX_SAMPLES = min(500, len(ds))
print(f"Extracting {MAX_SAMPLES} WAVs...")
transcriptions = {}
for i in range(MAX_SAMPLES):
    wav_path = os.path.join(wav_dir, f"rs_{i:05d}.wav")
    if not os.path.exists(wav_path):
        item = ds[i]
        audio = item["audio"]
        sf.write(wav_path, audio["array"], audio["sampling_rate"])
    transcriptions[i] = ds[i]["transcription"]
print("WAVs ready")

from metrics.normalize import normalize_ja
from jiwer import cer

for mname in ["whisper-large-v3-turbo", "qwen3-asr-0.6b", "qwen3-asr-1.7b"]:
    print(f"\n--- {mname} ---")
    runner = get_runner(mname)
    runner.load()

    total_cer = 0
    total_rtf = 0
    count = 0

    for i in range(MAX_SAMPLES):
        wav_path = os.path.join(wav_dir, f"rs_{i:05d}.wav")
        dur = get_audio_duration(wav_path)
        if dur < 0.5:
            continue

        start = time.perf_counter()
        try:
            text = runner.transcribe(wav_path, language="ja")
        except Exception as e:
            continue
        elapsed = time.perf_counter() - start

        ref = normalize_ja(transcriptions[i])
        hyp = normalize_ja(text)
        if ref:
            c = cer(ref, hyp)
            total_cer += c
            count += 1
        total_rtf += elapsed / dur if dur > 0 else 0

        if (i + 1) % 100 == 0:
            avg_cer = total_cer / count if count > 0 else 0
            print(f"  [{i+1}/{MAX_SAMPLES}] running CER={avg_cer:.4f}")

    avg_cer = total_cer / count if count > 0 else 0
    avg_rtf = total_rtf / MAX_SAMPLES
    print(f"  RESULT: CER={avg_cer:.4f} ({avg_cer*100:.2f}%), RTF={avg_rtf:.3f}, samples={count}")

    runner.unload()

print("\n=== ALL DONE ===")
