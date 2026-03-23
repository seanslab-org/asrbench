"""Base class for ASR benchmark datasets."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Sample:
    """A single ASR evaluation sample."""
    audio_path: str
    reference: str       # ground truth transcription
    language: str        # "en", "zh", "ja"
    sample_id: str       # unique identifier
    duration_s: float = 0.0


class ASRDataset(ABC):
    """Base interface for benchmark datasets."""

    name: str = "base"
    language: str = "en"

    @abstractmethod
    def load(self, data_dir: str) -> None:
        """Load dataset from disk."""

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples."""
