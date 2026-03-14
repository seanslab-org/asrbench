"""OpenAI Whisper runner."""
from typing import Optional
from .base import ASRRunner
from .registry import register


@register("whisper-large-v3")
class WhisperLargeV3Runner(ASRRunner):
    name = "whisper-large-v3"
    languages = ["en", "zh", "ja"]

    def load(self):
        import whisper
        self.model = whisper.load_model("large-v3")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        result = self.model.transcribe(
            audio_path,
            language=language,
            fp16=True,
        )
        return result["text"].strip()

    def unload(self):
        del self.model
        super().unload()


@register("whisper-large-v3-turbo")
class WhisperTurboRunner(ASRRunner):
    name = "whisper-large-v3-turbo"
    languages = ["en", "zh", "ja"]

    def load(self):
        import whisper
        self.model = whisper.load_model("turbo")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        result = self.model.transcribe(
            audio_path,
            language=language,
            fp16=True,
        )
        return result["text"].strip()

    def unload(self):
        del self.model
        super().unload()
