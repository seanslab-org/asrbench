#!/usr/bin/env python3
"""Benchmark parakeet-tdt-1.1b on sugr_bench (new) English data.

Data: 10 EN WAV clips (NPR) in data/sugr_bench/wav/ with .txt ground truth.
Metric: WER with EnglishTextNormalizer.
Strategy: 30-second chunking (matches tests/sugr_bench_top3.py).
Incremental save: partial results written after each clip so a crash
doesn't lose work.
"""
import sys, os, time, json, gc, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from metrics.profiler import get_audio_duration

try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from jiwer import wer

DATA_DIR = Path("data/sugr_bench")
WAV_DIR = DATA_DIR / "wav"
RESULTS_DIR = Path("results/sugr_new_20260416")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / "parakeet.json"
MODEL_ID = "parakeet-tdt-1.1b"

en_normalizer = EnglishTextNormalizer()


def load_samples():
    samples = []
    if not WAV_DIR.exists():
        return samples
    for wav in sorted(WAV_DIR.glob("*.wav")):
        clip_id = wav.stem
        txt = DATA_DIR / f"{clip_id}.txt"
        if not txt.exists():
            print(f"  WARN: missing ref for {clip_id}, skipping")
            continue
        ref = txt.read_text(encoding="utf-8").strip()
        samples.append({"id": clip_id, "audio": str(wav), "ref": ref})
    return samples


def transcribe_chunked(runner, audio_path, language="en", chunk_seconds=30):
    """Transcribe long audio by chunking into 30s segments."""
    import soundfile as sf
    import numpy as np
    import tempfile

    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    chunk_samples = int(chunk_seconds * sr)
    chunks = []
    start = 0
    while start < len(audio):
        end = min(start + chunk_samples, len(audio))
        chunks.append((start, end))
        start = end

    all_text = []
    for s, e in chunks:
        chunk = audio[s:e].astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk, sr)
            try:
                text = runner.transcribe(f.name, language=language)
            finally:
                os.unlink(f.name)
        if text:
            all_text.append(text)
    return " ".join(all_text)


def save_partial(rows):
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main():
    samples = load_samples()
    print(f"Loaded {len(samples)} EN samples from {WAV_DIR}/")
    if not samples:
        print("ERROR: no samples found.")
        sys.exit(1)

    print(f"\n{'='*70}\nMODEL: {MODEL_ID}\n{'='*70}")
    print(f"  Loading {MODEL_ID}...")
    t_load_start = time.time()
    from runners.parakeet_runner import ParakeetTDT11BRunner
    runner = ParakeetTDT11BRunner()
    runner.load()
    load_time = time.time() - t_load_start
    print(f"  Loaded in {load_time:.1f}s")

    rows = []
    save_partial(rows)

    for s in samples:
        try:
            dur = get_audio_duration(s["audio"])
        except Exception as e:
            print(f"  [{s['id']}] duration probe failed: {e}")
            rows.append({"id": s["id"], "model": MODEL_ID, "error": f"duration: {e}"})
            save_partial(rows)
            continue

        print(f"\n  [{s['id']}] {dur:.0f}s")
        t1 = time.time()
        try:
            hyp = transcribe_chunked(runner, s["audio"], language="en", chunk_seconds=30)
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            rows.append({
                "id": s["id"], "model": MODEL_ID, "error": str(e),
                "duration_s": dur,
            })
            save_partial(rows)
            continue
        t2 = time.time()
        rtf = (t2 - t1) / dur if dur > 0 else 0

        ref_norm = en_normalizer(s["ref"])
        hyp_norm = en_normalizer(hyp)
        w = wer(ref_norm, hyp_norm) if ref_norm else 1.0

        print(f"    WER={w:.4f} RTF={rtf:.3f} "
              f"(ref={len(ref_norm.split())}w, hyp={len(hyp_norm.split())}w)")
        rows.append({
            "id": s["id"], "model": MODEL_ID,
            "wer": w, "rtf": rtf,
            "duration_s": dur,
            "ref_words": len(ref_norm.split()),
            "hyp_words": len(hyp_norm.split()),
            "hyp": hyp_norm,
        })
        save_partial(rows)

    try:
        runner.unload()
    except Exception:
        pass
    del runner
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    valid = [r for r in rows if "wer" in r]
    print(f"\n{'='*70}\nSUMMARY: {MODEL_ID} on sugr_new (n={len(valid)}/{len(samples)})\n{'='*70}")
    print(f"{'clip_id':<26}{'dur_s':>8}{'ref_w':>8}{'WER%':>10}{'RTF':>10}")
    for r in rows:
        if "wer" in r:
            print(f"{r['id']:<26}{r['duration_s']:>8.0f}{r['ref_words']:>8d}"
                  f"{r['wer']*100:>10.2f}{r['rtf']:>10.3f}")
        else:
            print(f"{r['id']:<26}{'ERR':>8}{'-':>8}{'-':>10}{'-':>10}")

    if valid:
        avg_wer = sum(r["wer"] for r in valid) / len(valid)
        avg_rtf = sum(r["rtf"] for r in valid) / len(valid)
        print(f"\n  AVERAGE: WER={avg_wer*100:.2f}% RTF={avg_rtf:.3f}")

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
