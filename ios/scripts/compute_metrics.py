#!/usr/bin/env python3
"""Compute WER from the JSON the iOS app writes to its Documents container.

Reuses asrbench's existing `metrics.normalize` so the iOS results are directly
comparable to the Jetson rows in RESULTS.md.

Usage:
    # 1. Pull the results JSON from the simulator container
    python3 ios/scripts/compute_metrics.py results-2026-04-07T10-30-00Z.json

    # 2. Or pass the manifest separately if it's not bundled with the JSON
    python3 ios/scripts/compute_metrics.py results-*.json --manifest ios/Resources/audio/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Reuse asrbench's normalization so iOS WER aligns with Jetson WER
ROOT = Path(__file__).resolve().parent.parent.parent  # asrbench/
sys.path.insert(0, str(ROOT))
from metrics.normalize import normalize  # noqa: E402

try:
    from jiwer import wer as jiwer_wer
except ImportError:
    sys.exit("pip install jiwer")


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """Return {sample_id: reference}."""
    data = json.loads(manifest_path.read_text())
    return {e["sampleId"]: e["reference"] for e in data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json", type=Path)
    ap.add_argument(
        "--manifest", type=Path,
        default=ROOT / "ios" / "Resources" / "audio" / "manifest.json",
        help="Reference manifest (default: ios/Resources/audio/manifest.json)",
    )
    args = ap.parse_args()

    if not args.results_json.exists():
        sys.exit(f"results not found: {args.results_json}")
    if not args.manifest.exists():
        sys.exit(f"manifest not found: {args.manifest}")

    refs = load_manifest(args.manifest)
    payload = json.loads(args.results_json.read_text())
    results = payload.get("results", payload)  # tolerate either schema

    # Group by runner
    by_runner: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_runner[r["runner"]].append(r)

    print(f"results from: {args.results_json.name}")
    print(f"device:       {payload.get('device', '?')}")
    print(f"build:        {payload.get('xcodeBuild', '?')}")
    print(f"runners:      {len(by_runner)}")
    print()
    print(f"  {'Runner':<25} {'N':>5} {'Errs':>5} {'Hall':>5} {'WER%':>8} {'RTF':>8} {'Latency(s)':>12}")
    print(f"  {'-'*72}")

    for runner in sorted(by_runner):
        rs = by_runner[runner]
        ok = [r for r in rs if not r.get("error")]
        errs = len(rs) - len(ok)

        # Compute corpus-level WER (concatenate refs and hyps)
        ref_texts = []
        hyp_texts = []
        halls = 0
        for r in ok:
            ref = refs.get(r["sampleId"])
            if ref is None:
                continue
            hyp = r["transcript"]
            # Hallucination heuristic — same as bench.py: hyp >5x expected length
            audio_dur = r.get("audioDurationS", 0)
            if audio_dur > 5 and len(hyp) > 75 * audio_dur and len(hyp) > 500:
                halls += 1
            ref_texts.append(normalize(ref, "en"))
            hyp_texts.append(normalize(hyp, "en"))

        if ref_texts:
            corpus_wer = jiwer_wer(ref_texts, hyp_texts)
        else:
            corpus_wer = float("nan")

        avg_rtf = sum(r["rtf"] for r in ok) / len(ok) if ok else 0
        avg_lat = sum(r["latencyS"] for r in ok) / len(ok) if ok else 0

        print(f"  {runner:<25} {len(rs):>5} {errs:>5} {halls:>5} "
              f"{corpus_wer*100:>7.2f}% {avg_rtf:>8.3f} {avg_lat:>11.3f}s")

    print()
    print("Note: simulator latency = Mac CPU latency, not iPhone Neural Engine.")
    print("      Phase 2 = real device run gives the iPhone hardware story.")


if __name__ == "__main__":
    main()
