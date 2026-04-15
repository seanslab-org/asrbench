#!/usr/bin/env python3
"""Benchmark Qwen3-ASR-1.7B on sugr-asr-bench custom test data.

Compares ASR output with provided human transcripts.
EN: WER with EnglishTextNormalizer
ZH: CER with CJK-only normalization (strips timestamps/speaker labels from refs)
"""
import sys, os, re, time, json, unicodedata
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from runners.qwen_asr_runner import Qwen3ASR17BRunner, _patch_sdpa_for_gqa
from metrics.profiler import get_audio_duration
from whisper.normalizers import EnglishTextNormalizer
from jiwer import wer, cer

DATA_DIR = Path("sugr-asr-bench")
en_normalizer = EnglishTextNormalizer()


def normalize_zh(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf]", "", text)
    return text


def strip_zh_metadata(text):
    """Strip timestamps and speaker labels from ZH reference.
    Format: '00:00:01 Speaker 1\\ntext\\n00:00:15 Speaker 2\\ntext...'
    """
    lines = text.strip().split("\n")
    content_lines = []
    for line in lines:
        line = line.strip()
        # Skip timestamp+speaker lines like "00:00:01 Speaker 1"
        if re.match(r"^\d{2}:\d{2}:\d{2}\s+Speaker\s+\d+", line):
            continue
        if line:
            content_lines.append(line)
    return "".join(content_lines)


def load_english_samples():
    samples = []
    d = DATA_DIR / "english"
    for i in range(1, 16):
        mp3 = d / f"{i}.mp3"
        txt = d / f"{i}.txt"
        if mp3.exists() and txt.exists():
            ref = txt.read_text(encoding="utf-8").strip()
            samples.append({"id": f"en_{i}", "audio": str(mp3), "ref": ref, "lang": "en"})
    return samples


def load_chinese_samples():
    samples = []
    d = DATA_DIR / "chinese"
    for f in sorted(d.glob("*.mp3")):
        txt = d / (f.stem + "-transcript.txt")
        if txt.exists():
            try:
                raw_ref = txt.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                raw_ref = txt.read_text(encoding="utf-8", errors="replace").strip()
            ref = strip_zh_metadata(raw_ref)
            samples.append({"id": f"zh_{f.stem}", "audio": str(f), "ref": ref, "lang": "zh"})
    return samples


def transcribe_chunked(runner, audio_path, language, chunk_seconds=30):
    """Transcribe long audio by chunking into segments."""
    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    chunk_samples = int(chunk_seconds * sr)
    total_samples = len(audio)
    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append((start, end))
        start = end

    # Write each chunk to temp file and transcribe
    import tempfile
    all_text = []
    for i, (s, e) in enumerate(chunks):
        chunk = audio[s:e].astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk, sr)
            text = runner.transcribe(f.name, language=language)
            os.unlink(f.name)
        all_text.append(text)

    return " ".join(all_text)


if __name__ == "__main__":
    _patch_sdpa_for_gqa()

    print("Loading Qwen3-ASR-1.7B...")
    runner = Qwen3ASR17BRunner()
    runner.load()
    print("Model loaded")

    results = []

    # --- English ---
    en_samples = load_english_samples()
    print(f"\n=== English: {len(en_samples)} samples ===")

    for s in en_samples:
        dur = get_audio_duration(s["audio"])
        print(f"\n  [{s['id']}] {dur:.0f}s")

        t1 = time.time()
        hyp = transcribe_chunked(runner, s["audio"], language="en")
        t2 = time.time()
        rtf = (t2 - t1) / dur if dur > 0 else 0

        ref_norm = en_normalizer(s["ref"])
        hyp_norm = en_normalizer(hyp)

        w = wer(ref_norm, hyp_norm) if ref_norm else 1.0
        print(f"    WER={w:.4f} RTF={rtf:.3f}")
        print(f"    Ref: {ref_norm[:100]}...")
        print(f"    Hyp: {hyp_norm[:100]}...")

        results.append({
            "id": s["id"], "lang": "en", "wer": w, "rtf": rtf,
            "ref_len": len(ref_norm.split()), "hyp_len": len(hyp_norm.split()),
            "duration_s": dur,
        })

    en_avg_wer = sum(r["wer"] for r in results if r["lang"] == "en") / len(en_samples)
    en_avg_rtf = sum(r["rtf"] for r in results if r["lang"] == "en") / len(en_samples)
    print(f"\n  EN AVERAGE: WER={en_avg_wer:.4f} ({en_avg_wer*100:.2f}%) RTF={en_avg_rtf:.3f}")

    # --- Chinese ---
    zh_samples = load_chinese_samples()
    print(f"\n=== Chinese: {len(zh_samples)} samples ===")

    for s in zh_samples:
        dur = get_audio_duration(s["audio"])
        print(f"\n  [{s['id']}] {dur:.0f}s")

        t1 = time.time()
        hyp = transcribe_chunked(runner, s["audio"], language="zh")
        t2 = time.time()
        rtf = (t2 - t1) / dur if dur > 0 else 0

        ref_norm = normalize_zh(s["ref"])
        hyp_norm = normalize_zh(hyp)

        c = cer(ref_norm, hyp_norm) if ref_norm else 1.0
        print(f"    CER={c:.4f} RTF={rtf:.3f}")
        print(f"    Ref: {ref_norm[:80]}...")
        print(f"    Hyp: {hyp_norm[:80]}...")

        results.append({
            "id": s["id"], "lang": "zh", "cer": c, "rtf": rtf,
            "ref_len": len(ref_norm), "hyp_len": len(hyp_norm),
            "duration_s": dur,
        })

    zh_results = [r for r in results if r["lang"] == "zh"]
    zh_avg_cer = sum(r["cer"] for r in zh_results) / len(zh_results) if zh_results else 0
    zh_avg_rtf = sum(r["rtf"] for r in zh_results) / len(zh_results) if zh_results else 0
    print(f"\n  ZH AVERAGE: CER={zh_avg_cer:.4f} ({zh_avg_cer*100:.2f}%) RTF={zh_avg_rtf:.3f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUGR-ASR-BENCH RESULTS — Qwen3-ASR-1.7B on Jetson Orin 32GB")
    print(f"{'='*60}")
    print(f"  English: WER={en_avg_wer*100:.2f}% (n={len(en_samples)}) RTF={en_avg_rtf:.3f}")
    print(f"  Chinese: CER={zh_avg_cer*100:.2f}% (n={len(zh_results)}) RTF={zh_avg_rtf:.3f}")

    runner.unload()

    # Save
    with open("results/sugr_bench_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to results/sugr_bench_results.json")
