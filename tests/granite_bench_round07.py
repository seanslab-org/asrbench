#!/usr/bin/env python3
"""ASR Bench Round 07 — Granite Speech 4.1 2B AR + NAR.

Runs both Granite models against:
  1. LibriSpeech test-clean subset (50 samples by default)
  2. LibriSpeech test-other subset (50 samples by default, if available)
  3. Sugr-ASR-Bench NPR long-form (10 clips, 30-second chunking)

Captures WER + RTF + peak VRAM per model x dataset. Saves a results.json
in results/granite_round07_<UTC-date>/.

Env vars:
    GRANITE_AR_MODEL_PATH     local path to Granite-2b-ar model
    GRANITE_NAR_MODEL_PATH    local path to Granite-2b-nar model (optional)
    LIBRI_CLEAN_ROOT          /home/nvidia/vvfish/data/librispeech/LibriSpeech/test-clean
    LIBRI_OTHER_ROOT          /home/nvidia/vvfish/data/librispeech/LibriSpeech/test-other
    SUGR_ROOT                 /home/nvidia/Sugr-ASR-Bench/corpus
    LIMIT_LIBRI               default 50, samples per LS split
    LIMIT_SUGR                default 0 (all 10 clips)
    SKIP_NAR=1                skip NAR runner
    SKIP_AR=1                 skip AR runner
"""
import os
import sys
import time
import json
import gc
import traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402
import numpy as np  # noqa: E402
import tempfile  # noqa: E402

from runners.granite_speech_ar import GraniteSpeech41ARRunner  # noqa: E402

NAR_AVAILABLE = True
try:
    from runners.granite_speech_nar import GraniteSpeech41NARRunner
except Exception as e:
    print(f"[warn] NAR runner import failed: {e}")
    NAR_AVAILABLE = False
    GraniteSpeech41NARRunner = None  # type: ignore

# ---------------------------------------------------------------------------
# Normalizer + WER
# ---------------------------------------------------------------------------
try:
    from whisper.normalizers import EnglishTextNormalizer
except ImportError:
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from jiwer import wer

normalizer = EnglishTextNormalizer()


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalizer(reference).strip()
    hyp = normalizer(hypothesis).strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return wer(ref, hyp)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_librispeech(root: Path, limit: int = 50):
    if not root.exists():
        print(f"  [skip] {root} missing")
        return []
    samples = []
    for trans in sorted(root.rglob("*.trans.txt")):
        chap_dir = trans.parent
        for line in trans.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            uid, _, text = line.partition(" ")
            audio = chap_dir / f"{uid}.flac"
            if audio.exists():
                samples.append({"id": uid, "audio": str(audio), "ref": text})
                if 0 < limit <= len(samples):
                    return samples
    return samples


def load_sugr(root: Path, limit: int = 0):
    if not root.exists():
        print(f"  [skip] sugr root {root} missing")
        return []
    samples = []
    for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ref_file = clip_dir / "reference.txt"
        # Audio: prefer audio.opus (will need ffmpeg/decoder) or audio.wav
        audio = None
        for cand in ("audio.wav", "audio.opus", "audio.mp3", "audio.flac", "audio.m4a"):
            p = clip_dir / cand
            if p.exists():
                audio = p
                break
        if audio is None or not ref_file.exists():
            continue
        samples.append({
            "id": clip_dir.name,
            "audio": str(audio),
            "ref": ref_file.read_text(encoding="utf-8").strip(),
        })
        if 0 < limit <= len(samples):
            break
    return samples


# ---------------------------------------------------------------------------
# Long-form chunked transcription helper
# ---------------------------------------------------------------------------
def transcribe_chunked(runner, audio_path: str, chunk_seconds: int = 30,
                       language: str = "en") -> str:
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception:
            pass
    chunk_samples = int(chunk_seconds * sr)
    parts = []
    start = 0
    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    for i, s in enumerate(range(0, len(audio), chunk_samples)):
        e = min(s + chunk_samples, len(audio))
        chunk = audio[s:e].astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, chunk, sr)
            try:
                txt = runner.transcribe(f.name, language=language)
            except Exception as ex:
                print(f"    chunk {i+1}/{n_chunks} ERR: {ex}")
                txt = ""
            finally:
                os.unlink(f.name)
        if txt:
            parts.append(txt)
        if (i + 1) % 5 == 0 or (i + 1) == n_chunks:
            print(f"    chunk {i+1}/{n_chunks} ok (cumulative {sum(len(p) for p in parts)} chars)")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Per-model run on a dataset
# ---------------------------------------------------------------------------
def reset_peak():
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_peak_mb():
    try:
        import torch
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        return -1


def run_on_librispeech(runner, samples, dataset_name, results, save_fn):
    print(f"\n  -- {dataset_name}: {len(samples)} samples --")
    rtfs, wers, vrams = [], [], []
    for i, s in enumerate(samples):
        reset_peak()
        try:
            audio, sr = sf.read(s["audio"])
            dur = len(audio) / sr
        except Exception:
            dur = -1

        t0 = time.time()
        try:
            hyp = runner.transcribe(s["audio"], language="en")
        except Exception as ex:
            print(f"    [{i+1}/{len(samples)}] {s['id']} ERR: {ex}")
            results.append({
                "model": runner.name, "dataset": dataset_name,
                "sample_id": s["id"], "audio_dur": dur,
                "error": str(ex),
            })
            continue
        elapsed = time.time() - t0
        rtf = elapsed / dur if dur > 0 else -1
        peak = get_peak_mb()
        sample_wer = compute_wer(s["ref"], hyp)
        rtfs.append(rtf)
        wers.append(sample_wer)
        vrams.append(peak)
        print(f"    [{i+1}/{len(samples)}] {s['id']} dur={dur:.1f}s rtf={rtf:.3f} wer={sample_wer:.3%} vram={peak:.0f}MB")
        results.append({
            "model": runner.name, "dataset": dataset_name,
            "sample_id": s["id"], "audio_dur": dur,
            "elapsed": elapsed, "rtf": rtf, "vram_peak_mb": peak,
            "reference": s["ref"], "transcript": hyp, "wer": sample_wer,
        })
        if (i + 1) % 5 == 0:
            save_fn()
    if wers:
        avg_wer = sum(wers) / len(wers)
        avg_rtf = sum(rtfs) / len(rtfs)
        max_vram = max(vrams) if vrams else 0
        print(f"  -- {dataset_name} avg: WER={avg_wer:.3%} RTF={avg_rtf:.3f} VRAM_peak={max_vram:.0f}MB --")
    save_fn()


def run_on_sugr(runner, samples, dataset_name, results, save_fn,
                chunk_seconds: int = 30):
    print(f"\n  -- {dataset_name}: {len(samples)} clips --")
    for i, s in enumerate(samples):
        reset_peak()
        try:
            audio, sr = sf.read(s["audio"])
            dur = len(audio) / sr if audio.ndim < 2 else len(audio) / sr
        except Exception as ex:
            print(f"  could not read {s['audio']}: {ex}")
            dur = -1

        t0 = time.time()
        try:
            hyp = transcribe_chunked(runner, s["audio"],
                                     chunk_seconds=chunk_seconds, language="en")
        except Exception as ex:
            print(f"    [{i+1}/{len(samples)}] {s['id']} ERR: {ex}")
            results.append({
                "model": runner.name, "dataset": dataset_name,
                "sample_id": s["id"], "audio_dur": dur, "error": str(ex),
            })
            continue
        elapsed = time.time() - t0
        rtf = elapsed / dur if dur > 0 else -1
        peak = get_peak_mb()
        sample_wer = compute_wer(s["ref"], hyp)
        print(f"    [{i+1}/{len(samples)}] {s['id']} dur={dur:.0f}s "
              f"hyp_words={len(hyp.split())} ref_words={len(s['ref'].split())} "
              f"wer={sample_wer:.3%} rtf={rtf:.3f} vram={peak:.0f}MB")
        results.append({
            "model": runner.name, "dataset": dataset_name,
            "sample_id": s["id"], "audio_dur": dur,
            "elapsed": elapsed, "rtf": rtf, "vram_peak_mb": peak,
            "reference": s["ref"], "transcript": hyp, "wer": sample_wer,
        })
        save_fn()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    libri_clean_root = Path(os.environ.get(
        "LIBRI_CLEAN_ROOT",
        "/home/nvidia/vvfish/data/librispeech/LibriSpeech/test-clean"))
    libri_other_root = Path(os.environ.get(
        "LIBRI_OTHER_ROOT",
        "/home/nvidia/vvfish/data/librispeech/LibriSpeech/test-other"))
    sugr_root = Path(os.environ.get("SUGR_ROOT",
                                    "/home/nvidia/Sugr-ASR-Bench/corpus"))
    limit_libri = int(os.environ.get("LIMIT_LIBRI", "50"))
    limit_sugr = int(os.environ.get("LIMIT_SUGR", "0"))
    chunk_seconds = int(os.environ.get("SUGR_CHUNK_SECONDS", "30"))

    out_dir = Path(os.environ.get(
        "OUT_DIR",
        f"results/granite_round07_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    print(f"Saving to: {out_path}")

    samples_clean = load_librispeech(libri_clean_root, limit=limit_libri)
    samples_other = load_librispeech(libri_other_root, limit=limit_libri)
    samples_sugr = load_sugr(sugr_root, limit=limit_sugr)
    print(f"Loaded: LS-clean={len(samples_clean)}, LS-other={len(samples_other)}, "
          f"sugr={len(samples_sugr)}")

    all_results = []

    def save():
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    runners_to_run = []
    if os.environ.get("SKIP_AR", "0") != "1":
        runners_to_run.append(("granite-ar", GraniteSpeech41ARRunner))
    if NAR_AVAILABLE and os.environ.get("SKIP_NAR", "0") != "1":
        runners_to_run.append(("granite-nar", GraniteSpeech41NARRunner))

    for label, RunnerCls in runners_to_run:
        print(f"\n=========================================")
        print(f" Loading {label} ...")
        print(f"=========================================")
        runner = RunnerCls()
        t_load = time.time()
        try:
            runner.load()
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            traceback.print_exc()
            all_results.append({"model": runner.name, "load_error": str(e)})
            save()
            continue
        load_time = time.time() - t_load
        print(f"  loaded in {load_time:.1f}s")
        all_results.append({
            "model": runner.name,
            "event": "loaded", "load_time_s": load_time,
        })
        save()

        if samples_clean:
            run_on_librispeech(runner, samples_clean,
                               "librispeech-clean", all_results, save)
        if samples_other:
            run_on_librispeech(runner, samples_other,
                               "librispeech-other", all_results, save)
        if samples_sugr:
            run_on_sugr(runner, samples_sugr, "sugr-asr-bench-npr",
                        all_results, save, chunk_seconds=chunk_seconds)

        print(f"  unloading {label}")
        try:
            runner.unload()
        except Exception:
            pass
        del runner
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    save()
    print("\nDone.")


if __name__ == "__main__":
    main()
