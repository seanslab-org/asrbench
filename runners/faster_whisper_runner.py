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
        import ctranslate2
        # Check if CUDA is supported by CTranslate2
        try:
            ctranslate2.get_supported_compute_types("cuda")
            device, compute = "cuda", "float16"
        except ValueError:
            device, compute = "cpu", "int8"
        self.model = WhisperModel(
            "large-v3",
            device=device,
            compute_type=compute,
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
