#!/usr/bin/env python3
"""ASR Bench v2 — Benchmark ASR models on standard + custom datasets."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Import runners (triggers registration)
from runners.registry import RUNNERS, list_runners, get_runner
import runners.whisper_runner
import runners.faster_whisper_runner

# Conditionally import optional runners
for _mod in [
    "runners.kotoba_runner",
    "runners.qwen_asr_runner",
    "runners.sensevoice_runner",
    "runners.firered_runner",
    "runners.vibevoice_runner",
    "runners.parakeet_runner",
    "runners.voxtral_runner",
    "runners.cohere_runner",
    "runners.moonshine_runner",
    "runners.funasr_nano_onnx_runner",
    "runners.funasr_nano_runner",
    "runners.granite_speech_ar",
    "runners.granite_speech_nar",
]:
    try:
        __import__(_mod)
    except ImportError:
        pass

# Import dataset loaders
from asrdatasets.registry import list_datasets, get_dataset
import asrdatasets.librispeech
import asrdatasets.aishell
import asrdatasets.reazonspeech

from metrics.profiler import profile_inference, profile_load, get_audio_duration
from metrics.wer import compute_accuracy


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
LANGUAGES = ["en", "zh", "ja"]
WARMUP_SAMPLES = 2  # discard first N samples from timing


# ---------------------------------------------------------------------------
# Standard dataset benchmark
# ---------------------------------------------------------------------------


def run_standard_benchmark(
    models: list,
    dataset_names: list,
    data_dir: Path,
    results_dir: Path,
    limit: int = 0,
):
    """Run benchmark on standard datasets with ground truth."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        try:
            runner = get_runner(model_name)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Load model
        print(f"  Loading model...", end=" ", flush=True)
        try:
            load_time = profile_load(runner)
            print(f"OK ({load_time:.1f}s)")
        except Exception as e:
            print(f"FAIL: {e}")
            traceback.print_exc()
            continue

        for ds_name in dataset_names:
            print(f"\n  --- Dataset: {ds_name} ---")
            try:
                ds = get_dataset(ds_name)
                ds.load(str(data_dir))
            except Exception as e:
                print(f"  SKIP dataset {ds_name}: {e}")
                continue

            if not runner.supports_language(ds.language):
                print(f"  SKIP {ds_name} ({ds.language} not supported by {model_name})")
                continue

            sample_iter = list(ds)
            if limit and limit > 0:
                sample_iter = sample_iter[:limit]
            print(f"  {len(sample_iter)}/{len(ds)} samples, language={ds.language}")
            completed = 0
            errors = 0
            hallucinations = 0

            for i, sample in enumerate(sample_iter):
                # Warm-up: run but discard from results
                is_warmup = i < WARMUP_SAMPLES

                label = f"  [{i+1}/{len(sample_iter)}] {sample.sample_id}"
                if is_warmup:
                    label += " (warmup)"
                print(f"{label}...", end=" ", flush=True)

                with profile_inference(model_name, sample.audio_path, ds.language) as prof:
                    try:
                        prof.transcript = runner.transcribe(sample.audio_path, language=ds.language)
                    except Exception as e:
                        prof.error = str(e)
                        print(f"ERR: {e}")
                        errors += 1
                        continue

                # Hallucination detection
                is_hallucinated = detect_hallucination(
                    prof.transcript, prof.audio_duration_s
                )
                if is_hallucinated:
                    hallucinations += 1

                # Compute accuracy
                accuracy = compute_accuracy(sample.reference, prof.transcript, ds.language)

                result = prof.to_dict()
                result["dataset"] = ds_name
                result["sample_id"] = sample.sample_id
                result["reference"] = sample.reference
                result["load_time_s"] = load_time
                result["accuracy"] = accuracy
                result["is_warmup"] = is_warmup
                result["is_hallucinated"] = is_hallucinated

                # Print progress
                metric_key = "wer" if ds.language == "en" else "cer"
                metric_val = accuracy.get(metric_key, -1)
                metric_str = f"{metric_key}={metric_val:.3f}" if metric_val >= 0 else "N/A"
                hall_str = " [HALL]" if is_hallucinated else ""
                print(f"RTF={prof.rtf:.3f}, {metric_str}{hall_str}")

                if not is_warmup:
                    all_results.append(result)
                completed += 1

            print(f"  Done: {completed} samples, {errors} errors, {hallucinations} hallucinations")

        # Unload model
        print(f"  Unloading {model_name}...")
        runner.unload()

    # Save results
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print_standard_summary(all_results)
    return all_results


def detect_hallucination(transcript: str, audio_duration_s: float) -> bool:
    """Detect likely hallucinated output.

    Flags if:
    - Output is >3x longer than expected (based on ~15 chars/sec estimate)
    - Any 5-gram repeats ≥3 times
    """
    if not transcript or audio_duration_s < 1.0:
        return False

    # Length check: typical speech is ~10-15 chars/sec
    # Only flag if audio is >5s AND output is >5x expected (very conservative)
    expected_chars = audio_duration_s * 15
    if audio_duration_s > 5.0 and len(transcript) > expected_chars * 5 and len(transcript) > 500:
        return True

    # Repetition check: 5-gram repeats ≥3 times (only on longer outputs)
    words = transcript.split()
    if len(words) >= 30:
        ngrams = [" ".join(words[i:i+5]) for i in range(len(words) - 4)]
        from collections import Counter
        counts = Counter(ngrams)
        if counts and counts.most_common(1)[0][1] >= 3:
            return True

    return False


# ---------------------------------------------------------------------------
# Custom clips benchmark (legacy)
# ---------------------------------------------------------------------------


def discover_audio(data_dir: Path, clip_type: str = "short"):
    """Discover audio files organized by language."""
    clips = {}
    for lang in LANGUAGES:
        lang_dir = data_dir / lang / clip_type
        if lang_dir.exists():
            wavs = sorted(lang_dir.glob("*.wav"))
            if wavs:
                clips[lang] = wavs
    return clips


def load_ground_truth(data_dir: Path, lang: str):
    """Load ground truth transcriptions if available."""
    gt_path = data_dir / lang / "ground_truth.json"
    if gt_path.exists():
        with open(gt_path) as f:
            return json.load(f)
    return {}


def run_custom_benchmark(
    models: list,
    clip_types: list,
    data_dir: Path,
    results_dir: Path,
    num_runs: int = 1,
):
    """Run benchmark on custom clips (legacy mode)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        try:
            runner = get_runner(model_name)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Loading model...", end=" ", flush=True)
        try:
            load_time = profile_load(runner)
            print(f"OK ({load_time:.1f}s)")
        except Exception as e:
            print(f"FAIL: {e}")
            traceback.print_exc()
            continue

        for clip_type in clip_types:
            clips = discover_audio(data_dir, clip_type)
            if not clips:
                print(f"  No {clip_type} clips found")
                continue

            for lang, audio_files in clips.items():
                if not runner.supports_language(lang):
                    print(f"  SKIP {lang} (not supported)")
                    continue

                gt = load_ground_truth(data_dir, lang)

                for audio_path in audio_files:
                    duration = get_audio_duration(str(audio_path))
                    if duration < 0.5:
                        print(f"  SKIP {audio_path.name} (too short)")
                        continue

                    for run_idx in range(num_runs):
                        run_label = f"  [{clip_type}] {lang}/{audio_path.name}"
                        if num_runs > 1:
                            run_label += f" (run {run_idx+1}/{num_runs})"
                        print(f"{run_label} ({duration:.0f}s)...", end=" ", flush=True)

                        with profile_inference(model_name, str(audio_path), lang) as prof:
                            try:
                                prof.transcript = runner.transcribe(str(audio_path), language=lang)
                            except Exception as e:
                                prof.error = str(e)
                                print(f"ERR: {e}")
                                continue

                        result = prof.to_dict()
                        result["clip_type"] = clip_type
                        result["run_idx"] = run_idx
                        result["load_time_s"] = load_time

                        clip_key = f"{clip_type}/{audio_path.stem}"
                        if clip_key in gt:
                            accuracy = compute_accuracy(gt[clip_key], prof.transcript, lang)
                            result["accuracy"] = accuracy
                            acc_str = ", ".join(f"{k}={v:.3f}" for k, v in accuracy.items())
                            print(f"RTF={prof.rtf:.3f}, VRAM={prof.vram_peak_mb:.0f}MB, {acc_str}")
                        else:
                            print(f"RTF={prof.rtf:.3f}, VRAM={prof.vram_peak_mb:.0f}MB, {len(prof.transcript)} chars")

                        all_results.append(result)

        print(f"  Unloading {model_name}...")
        runner.unload()

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")
    print_summary(all_results)
    return all_results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def print_standard_summary(results: list):
    """Print summary for standard benchmark results."""
    if not results:
        return

    print(f"\n{'='*100}")
    print("STANDARD BENCHMARK SUMMARY")
    print(f"{'='*100}")

    # Group by model × dataset
    from collections import defaultdict
    table = defaultdict(lambda: defaultdict(list))

    for r in results:
        if r.get("error") or r.get("is_warmup") or r.get("is_hallucinated"):
            continue
        model = r["model_name"]
        ds = r["dataset"]
        acc = r.get("accuracy", {})
        table[model][ds].append({
            "rtf": r["rtf"],
            "vram": r["vram_peak_mb"],
            **acc,
        })

    # Print per-dataset results
    datasets_seen = sorted({r["dataset"] for r in results if not r.get("error")})

    for ds in datasets_seen:
        lang = results[0]["language"]  # approximate
        metric = "wer" if "librispeech" in ds else "cer"
        print(f"\n  Dataset: {ds} ({metric.upper()})")
        print(f"  {'Model':<30} {metric.upper():>8} {'RTF':>8} {'VRAM(MB)':>10}")
        print(f"  {'-'*56}")

        for model in sorted(table.keys()):
            if ds not in table[model]:
                continue
            entries = table[model][ds]
            avg_metric = sum(e.get(metric, 0) for e in entries) / len(entries)
            avg_rtf = sum(e["rtf"] for e in entries) / len(entries)
            max_vram = max(e["vram"] for e in entries)
            print(f"  {model:<30} {avg_metric:>8.4f} {avg_rtf:>8.3f} {max_vram:>10.0f}")


def print_summary(results: list):
    """Print summary for custom clip results (legacy)."""
    if not results:
        return

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    models = {}
    for r in results:
        if r.get("error"):
            continue
        name = r["model_name"]
        if name not in models:
            models[name] = {"rtfs": [], "vrams": [], "load": 0}
        models[name]["rtfs"].append(r["rtf"])
        models[name]["vrams"].append(r["vram_peak_mb"])
        models[name]["load"] = r.get("load_time_s", 0)

        lang = r["language"]
        acc_key = f"acc_{lang}"
        if acc_key not in models[name]:
            models[name][acc_key] = []
        acc = r.get("accuracy", {})
        if lang == "en" and "wer" in acc:
            models[name][acc_key].append(acc["wer"])
        elif "cer" in acc:
            models[name][acc_key].append(acc["cer"])

    header = f"{'Model':<30} {'Load(s)':>8} {'RTF':>8} {'VRAM(MB)':>10} {'EN WER':>8} {'ZH CER':>8} {'JA CER':>8}"
    print(header)
    print("-" * len(header))

    for name, data in sorted(models.items()):
        avg_rtf = sum(data["rtfs"]) / len(data["rtfs"]) if data["rtfs"] else 0
        max_vram = max(data["vrams"]) if data["vrams"] else 0

        en_acc = sum(data.get("acc_en", [])) / len(data.get("acc_en", [1])) if data.get("acc_en") else -1
        zh_acc = sum(data.get("acc_zh", [])) / len(data.get("acc_zh", [1])) if data.get("acc_zh") else -1
        ja_acc = sum(data.get("acc_ja", [])) / len(data.get("acc_ja", [1])) if data.get("acc_ja") else -1

        en_str = f"{en_acc:.3f}" if en_acc >= 0 else "N/A"
        zh_str = f"{zh_acc:.3f}" if zh_acc >= 0 else "N/A"
        ja_str = f"{ja_acc:.3f}" if ja_acc >= 0 else "N/A"

        print(f"{name:<30} {data['load']:>8.1f} {avg_rtf:>8.3f} {max_vram:>10.0f} {en_str:>8} {zh_str:>8} {ja_str:>8}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ASR Benchmark v2")
    sub = parser.add_subparsers(dest="command", help="Benchmark mode")

    # Standard benchmark mode
    std = sub.add_parser("standard", help="Run on standard datasets (LibriSpeech, AISHELL-1, etc.)")
    std.add_argument("--models", nargs="+", default=None)
    std.add_argument("--datasets", nargs="+", default=None,
                     help="Datasets to use (default: all available)")
    std.add_argument("--data-dir", type=Path, default=DATA_DIR)
    std.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    std.add_argument("--limit", type=int, default=0,
                     help="Cap samples per dataset (0 = all)")

    # Custom clips mode (legacy)
    cust = sub.add_parser("custom", help="Run on custom audio clips")
    cust.add_argument("--models", nargs="+", default=None)
    cust.add_argument("--clips", nargs="+", default=["short"], choices=["short", "long"])
    cust.add_argument("--data-dir", type=Path, default=DATA_DIR)
    cust.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    cust.add_argument("--runs", type=int, default=1)

    # List
    parser.add_argument("--list", action="store_true", help="List available models and datasets")

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name in list_runners():
            print(f"  - {name}")
        print("\nAvailable datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        return

    if args.command == "standard":
        models = args.models or list_runners()
        datasets = args.datasets or list_datasets()
        print(f"ASR Bench v2 — Standard Benchmark")
        print(f"  Models: {models}")
        print(f"  Datasets: {datasets}")
        run_standard_benchmark(models, datasets, args.data_dir, args.results_dir, limit=args.limit)

    elif args.command == "custom":
        models = args.models or list_runners()
        print(f"ASR Bench v2 — Custom Clips")
        print(f"  Models: {models}")
        print(f"  Clips: {args.clips}")
        run_custom_benchmark(models, args.clips, args.data_dir, args.results_dir, args.runs)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
