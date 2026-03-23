"""AISHELL-1 test dataset loader.

Ground truth format: transcript/aishell_transcript_v0.8.txt
    Each line: "UTTERANCE-ID  W1 W2 W3 ..." (space-separated characters)
Audio format: wav/test/SPEAKER/UTTERANCE.wav

Download:
    wget https://www.openslr.org/resources/33/data_aishell.tgz
    tar xzf data_aishell.tgz -C data/standard/aishell1/
    # Then: cd data/standard/aishell1/data_aishell/wav && bash ../resource_aishell/untar_wav.sh
"""
from pathlib import Path
from typing import Iterator

from asrdatasets.base import ASRDataset, Sample
from asrdatasets.registry import register_dataset


@register_dataset("aishell1")
class AISHELL1(ASRDataset):
    name = "aishell1"
    language = "zh"

    def __init__(self):
        self.samples = []

    def load(self, data_dir: str) -> None:
        root = Path(data_dir) / "standard" / "aishell1" / "data_aishell"
        transcript_path = root / "transcript" / "aishell_transcript_v0.8.txt"
        wav_dir = root / "wav" / "test"

        if not transcript_path.exists():
            raise FileNotFoundError(
                f"AISHELL-1 transcript not found at {transcript_path}. "
                f"Download from https://www.openslr.org/33/"
            )

        # Load all transcripts
        transcripts = {}
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                utt_id = parts[0]
                # Characters are space-separated in the transcript; join them
                text = "".join(parts[1:])
                transcripts[utt_id] = text

        # Find test audio files
        self.samples = []
        if not wav_dir.exists():
            raise FileNotFoundError(f"AISHELL-1 test wav dir not found at {wav_dir}")

        for wav_file in sorted(wav_dir.rglob("*.wav")):
            utt_id = wav_file.stem
            if utt_id in transcripts:
                self.samples.append(Sample(
                    audio_path=str(wav_file),
                    reference=transcripts[utt_id],
                    language="zh",
                    sample_id=utt_id,
                ))

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
