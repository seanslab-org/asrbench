"""LibriSpeech test-clean and test-other dataset loaders.

Ground truth format: .trans.txt files with "UTTERANCE-ID TRANSCRIPTION" per line.
Audio format: .flac files organized as speaker/chapter/speaker-chapter-utterance.flac
Reference transcriptions are UPPERCASE, no punctuation.

Download:
    wget https://www.openslr.org/resources/12/test-clean.tar.gz
    wget https://www.openslr.org/resources/12/test-other.tar.gz
    tar xzf test-clean.tar.gz -C data/standard/librispeech/
    tar xzf test-other.tar.gz -C data/standard/librispeech/
"""
import os
from pathlib import Path
from typing import Iterator

from asrdatasets.base import ASRDataset, Sample
from asrdatasets.registry import register_dataset


class LibriSpeechBase(ASRDataset):
    """Base loader for LibriSpeech splits."""

    language = "en"
    split = "test-clean"

    def __init__(self):
        self.samples = []

    def load(self, data_dir: str) -> None:
        root = Path(data_dir) / "standard" / "librispeech" / "LibriSpeech" / self.split
        if not root.exists():
            raise FileNotFoundError(
                f"LibriSpeech {self.split} not found at {root}. "
                f"Download from https://www.openslr.org/12/"
            )
        self.samples = []
        # Walk speaker/chapter directories
        for trans_file in sorted(root.rglob("*.trans.txt")):
            chapter_dir = trans_file.parent
            with open(trans_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, text = parts
                    audio_path = chapter_dir / f"{utt_id}.flac"
                    if audio_path.exists():
                        self.samples.append(Sample(
                            audio_path=str(audio_path),
                            reference=text,  # UPPERCASE, raw from dataset
                            language="en",
                            sample_id=utt_id,
                        ))

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


@register_dataset("librispeech-clean")
class LibriSpeechClean(LibriSpeechBase):
    name = "librispeech-clean"
    split = "test-clean"


@register_dataset("librispeech-other")
class LibriSpeechOther(LibriSpeechBase):
    name = "librispeech-other"
    split = "test-other"
