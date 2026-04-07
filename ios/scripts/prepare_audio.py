#!/usr/bin/env python3
"""Convert N LibriSpeech FLAC samples → 16kHz mono 16-bit PCM WAV.

Bundled into the iOS app's `Resources/audio/` directory along with a manifest
JSON containing references for post-hoc WER computation.

Moonshine's Swift WAVLoader supports 16/24/32-bit PCM mono/stereo. We pick
16-bit mono 16 kHz which is the format the model was trained on — no resample
needed at runtime, smallest bundle size.

Usage:
    python3 ios/scripts/prepare_audio.py \\
        --src data/standard/librispeech/LibriSpeech/test-clean \\
        --dst ios/Resources/audio \\
        --count 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import numpy as np


def find_samples(src: Path, count: int):
    """Walk LibriSpeech, return [(flac_path, reference, sample_id), ...].

    LibriSpeech layout: speaker/chapter/<utt-id>.flac plus a *.trans.txt per chapter
    listing "<utt-id> TEXT" lines (uppercase, no punctuation).
    """
    out = []
    for trans in sorted(src.rglob("*.trans.txt")):
        chapter = trans.parent
        with trans.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, _, text = line.partition(" ")
                flac = chapter / f"{utt_id}.flac"
                if flac.exists():
                    out.append((flac, text, utt_id))
                    if len(out) >= count:
                        return out
    return out


def convert(flac_path: Path, dst_dir: Path) -> tuple[str, float]:
    audio, sr = sf.read(str(flac_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        # LibriSpeech is 16k already; if not, resample.
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except ImportError:
            sys.exit("librosa needed to resample; install or use 16k-native source")
        sr = 16000

    # 16-bit signed PCM WAV
    pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    out_name = f"{flac_path.stem}.wav"
    out_path = dst_dir / out_name
    sf.write(str(out_path), pcm16, sr, subtype="PCM_16")
    duration = len(audio) / sr
    return out_name, duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="LibriSpeech root, e.g. data/standard/librispeech/LibriSpeech/test-clean")
    ap.add_argument("--dst", type=Path, required=True,
                    help="Output dir, e.g. ios/Resources/audio")
    ap.add_argument("--count", type=int, default=50)
    args = ap.parse_args()

    if not args.src.exists():
        sys.exit(f"src not found: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    samples = find_samples(args.src, args.count)
    if not samples:
        sys.exit("no samples found in src")

    manifest = []
    total_duration = 0.0
    total_bytes = 0
    for flac, text, utt_id in samples:
        wav_name, dur = convert(flac, args.dst)
        wav_size = (args.dst / wav_name).stat().st_size
        total_duration += dur
        total_bytes += wav_size
        manifest.append({
            "sampleId": utt_id,
            "filename": wav_name,
            "reference": text,
            "language": "en",
        })

    manifest_path = args.dst / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    mb = total_bytes / 1024 / 1024
    print(f"Wrote {len(manifest)} samples to {args.dst}")
    print(f"  total audio: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"  total size:  {mb:.1f} MB")
    print(f"  manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
