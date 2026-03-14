"""Abstract base class for ASR model runners."""
from abc import ABC, abstractmethod
from typing import List, Optional


class ASRRunner(ABC):
    """Base interface for all ASR model runners."""

    name: str = "base"
    languages: List[str] = ["en", "zh", "ja"]
    requires_gpu: bool = True

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""

    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio file, return text."""

    def unload(self) -> None:
        """Free GPU/CPU memory."""
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def supports_language(self, lang: str) -> bool:
        return lang in self.languages
