#!/usr/bin/env python3
"""Benchmark top-3 EN ASR models on sugr-asr-bench English data.

Models: cohere-transcribe-2b, parakeet-tdt-1.1b, qwen3-asr-1.7b
Data: 15 EN long-form MP3 clips with ground-truth .txt transcripts
Metric: WER with EnglishTextNormalizer
Strategy: 30-second chunking (matches existing sugr_bench.py for Qwen)
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

DATA_DIR = Path("sugr-asr-bench")
RESULTS_DIR = Path("results/sugr_top3_20260415")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

en_normalizer = EnglishTextNormalizer()


def load_english_samples():
    samples = []
    d = DATA_DIR / "english"
    for i in range(1, 16):
        mp3 = d / f"{i}.mp3"
        txt = d / f"{i}.txt"
        if mp3.exists() and txt.exists():
            ref = txt.read_text(encoding="utf-8").strip()
            samples.append({"id": f"en_{i}", "audio": str(mp3), "ref": ref})
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


def get_runner(model_id):
    """Lazy-import and instantiate a runner by name."""
    if model_id == "cohere-transcribe-2b":
        from runners.cohere_runner import CohereTranscribe2BRunner
        return CohereTranscribe2BRunner()
    if model_id == "parakeet-tdt-1.1b":
        from runners.parakeet_runner import ParakeetTDT11BRunner
        return ParakeetTDT11BRunner()
    if model_id == "qwen3-asr-1.7b":
        from runners.qwen_asr_runner import Qwen3ASR17BRunner, _patch_sdpa_for_gqa
        _patch_sdpa_for_gqa()
        return Qwen3ASR17BRunner()
    raise ValueError(f"Unknown model: {model_id}")


def bench_one_model(model_id, samples):
    print(f"\n{'='*70}\nMODEL: {model_id}\n{'='*70}")

    print(f"  Loading {model_id}...")
    t_load_start = time.time()
    runner = get_runner(model_id)
    runner.load()
    load_time = time.time() - t_load_start
    print(f"  Loaded in {load_time:.1f}s")

    rows = []
    for s in samples:
        try:
            dur = get_audio_duration(s["audio"])
        except Exception as e:
            print(f"  [{s['id']}] duration probe failed: {e}")
            continue

        print(f"\n  [{s['id']}] {dur:.0f}s")
        t1 = time.time()
        try:
            hyp = transcribe_chunked(runner, s["audio"], language="en", chunk_seconds=30)
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            rows.append({
                "id": s["id"], "model": model_id, "error": str(e),
                "duration_s": dur,
            })
            continue
        t2 = time.time()
        rtf = (t2 - t1) / dur if dur > 0 else 0

        ref_norm = en_normalizer(s["ref"])
        hyp_norm = en_normalizer(hyp)
        w = wer(ref_norm, hyp_norm) if ref_norm else 1.0

        print(f"    WER={w:.4f} RTF={rtf:.3f} (ref={len(ref_norm.split())}w, hyp={len(hyp_norm.split())}w)")
        rows.append({
            "id": s["id"], "model": model_id,
            "wer": w, "rtf": rtf,
            "duration_s": dur,
            "ref_words": len(ref_norm.split()),
            "hyp_words": len(hyp_norm.split()),
            "hyp": hyp_norm,  # for debugging
        })

    runner.unload()
    del runner
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    valid = [r for r in rows if "wer" in r]
    if valid:
        avg_wer = sum(r["wer"] for r in valid) / len(valid)
        avg_rtf = sum(r["rtf"] for r in valid) / len(valid)
        print(f"\n  {model_id} AVERAGE: WER={avg_wer*100:.2f}% RTF={avg_rtf:.3f} (n={len(valid)}/{len(samples)})")

    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append",
                    help="Model id(s) to run. Can be repeated. Default: all top-3.")
    ap.add_argument("--out", default=None,
                    help="Override output JSON filename inside results dir.")
    args = ap.parse_args()

    samples = load_english_samples()
    print(f"Loaded {len(samples)} EN samples from {DATA_DIR}/english/")
    if not samples:
        print("ERROR: no samples found. Make sure sugr-asr-bench/english/{i}.mp3 + {i}.txt exist.")
        sys.exit(1)

    models = args.model or ["cohere-transcribe-2b", "parakeet-tdt-1.1b", "qwen3-asr-1.7b"]
    out_path = RESULTS_DIR / (args.out or "results.json")
    all_rows = []
    for model_id in models:
        try:
            all_rows.extend(bench_one_model(model_id, samples))
        except Exception as e:
            print(f"FATAL on {model_id}: {e}")
            traceback.print_exc()
            all_rows.append({"model": model_id, "fatal_error": str(e)})

        # Save incrementally so a later crash doesn't lose earlier work
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)

    # Final summary
    print(f"\n{'='*70}\nSUGR-ASR-BENCH TOP-3 EN SUMMARY\n{'='*70}")
    print(f"{'Model':<26}{'N':>5}{'WER%':>10}{'RTF':>10}")
    for model_id in models:
        valid = [r for r in all_rows if r.get("model") == model_id and "wer" in r]
        if not valid:
            print(f"{model_id:<26}{'-':>5}{'FAIL':>10}{'-':>10}")
            continue
        avg_wer = sum(r["wer"] for r in valid) / len(valid)
        avg_rtf = sum(r["rtf"] for r in valid) / len(valid)
        print(f"{model_id:<26}{len(valid):>5}{avg_wer*100:>10.2f}{avg_rtf:>10.3f}")

    print(f"\nSaved to {out_path}")
