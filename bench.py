#!/usr/bin/env python3
"""ASR Bench — Benchmark ASR models on real-world meeting audio."""
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
import runners.moonshine_runner
import runners.funasr_runner

from metrics.profiler import profile_inference, profile_load, get_audio_duration
from metrics.wer import compute_accuracy


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
LANGUAGES = ["en", "zh", "ja"]


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


def run_benchmark(
    models: list,
    clip_types: list,
    data_dir: Path,
    results_dir: Path,
    num_runs: int = 1,
):
    """Run benchmark for specified models and clip types."""
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
                        print(f"  SKIP {audio_path.name} (too short: {duration:.1f}s)")
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

                        # Compute accuracy if ground truth available
                        clip_key = f"{clip_type}/{audio_path.stem}"
                        if clip_key in gt:
                            accuracy = compute_accuracy(gt[clip_key], prof.transcript, lang)
                            result["accuracy"] = accuracy
                            acc_str = ", ".join(f"{k}={v:.3f}" for k, v in accuracy.items())
                            print(f"RTF={prof.rtf:.3f}, VRAM={prof.vram_peak_mb:.0f}MB, {acc_str}")
                        else:
                            print(f"RTF={prof.rtf:.3f}, VRAM={prof.vram_peak_mb:.0f}MB, {len(prof.transcript)} chars")

                        all_results.append(result)

        # Unload model
        print(f"  Unloading {model_name}...")
        runner.unload()

    # Save results
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print_summary(all_results)

    return all_results


def print_summary(results: list):
    """Print a summary table of results."""
    if not results:
        return

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    # Group by model
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

        # Accuracy per language
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


def main():
    parser = argparse.ArgumentParser(description="ASR Benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to benchmark (default: all)")
    parser.add_argument("--clips", nargs="+", default=["short"],
                        choices=["short", "long"],
                        help="Clip types to test (default: short)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per clip (default: 1)")
    parser.add_argument("--list", action="store_true",
                        help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name in list_runners():
            print(f"  - {name}")
        return

    models = args.models or list_runners()
    print(f"ASR Bench — {len(models)} models, clips={args.clips}, runs={args.runs}")
    print(f"Available models: {list_runners()}")
    print(f"Selected: {models}")

    run_benchmark(
        models=models,
        clip_types=args.clips,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_runs=args.runs,
    )


if __name__ == "__main__":
    main()
