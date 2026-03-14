"""Moonshine ASR runner."""
from typing import Optional
from .base import ASRRunner
from .registry import register


@register("moonshine-base")
class MoonshineBaseRunner(ASRRunner):
    name = "moonshine-base"
    languages = ["en", "zh", "ja"]

    def load(self):
        from moonshine import transcribe as moon_transcribe
        import moonshine
        self._transcribe = moon_transcribe
        # Warm up / download model
        self.model_name = "moonshine/base"

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        result = self._transcribe(audio_path, self.model_name)
        if isinstance(result, list):
            return " ".join(result).strip()
        return str(result).strip()

    def unload(self):
        super().unload()


@register("moonshine-small")
class MoonshineSmallRunner(ASRRunner):
    name = "moonshine-small"
    languages = ["en", "zh", "ja"]

    def load(self):
        from moonshine import transcribe as moon_transcribe
        self._transcribe = moon_transcribe
        self.model_name = "moonshine/small"

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        result = self._transcribe(audio_path, self.model_name)
        if isinstance(result, list):
            return " ".join(result).strip()
        return str(result).strip()

    def unload(self):
        super().unload()
