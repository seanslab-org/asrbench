"""Kotoba-Whisper runners — Japanese-specialist distilled Whisper models.

kotoba-whisper-v2.2: JA only, 756M params, built-in diarization + punctuation
kotoba-whisper-bilingual-v1.0: JA + EN, 756M params

Uses the exact same HF pipeline as Whisper — drop-in replacement.
"""
import torch
from runners.base import ASRRunner
from runners.registry import register


@register("kotoba-whisper-v2.2")
class KotobaWhisperV22Runner(ASRRunner):
    name = "kotoba-whisper-v2.2"
    languages = ["ja"]
    requires_gpu = True

    def __init__(self):
        self.pipe = None

    def load(self):
        from transformers import pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="kotoba-tech/kotoba-whisper-v2.2",
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
        )

    def transcribe(self, audio_path: str, language: str = None) -> str:
        result = self.pipe(
            audio_path,
            generate_kwargs={"language": "ja", "task": "transcribe"},
            return_timestamps=True,
        )
        return result["text"]

    def unload(self):
        del self.pipe
        self.pipe = None
        super().unload()


@register("kotoba-whisper-bilingual")
class KotobaWhisperBilingualRunner(ASRRunner):
    name = "kotoba-whisper-bilingual"
    languages = ["en", "ja"]
    requires_gpu = True

    def __init__(self):
        self.pipe = None

    def load(self):
        from transformers import pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="kotoba-tech/kotoba-whisper-bilingual-v1.0",
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,
        )

    def transcribe(self, audio_path: str, language: str = None) -> str:
        lang = language if language in ("en", "ja") else "ja"
        result = self.pipe(
            audio_path,
            generate_kwargs={"language": lang, "task": "transcribe"},
            return_timestamps=True,
        )
        return result["text"]

    def unload(self):
        del self.pipe
        self.pipe = None
        super().unload()
