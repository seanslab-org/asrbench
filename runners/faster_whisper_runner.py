"""Faster-Whisper runner (CTranslate2 optimized)."""
from typing import Optional
from .base import ASRRunner
from .registry import register


@register("faster-whisper-large-v3")
class FasterWhisperRunner(ASRRunner):
    name = "faster-whisper-large-v3"
    languages = ["en", "zh", "ja"]

    def load(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
        )

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
        )
        return " ".join(seg.text.strip() for seg in segments)

    def unload(self):
        del self.model
        super().unload()
