"""Voxtral Realtime 4B runner — Mistral's streaming ASR model.
Uses AutoProcessor + VoxtralRealtimeForConditionalGeneration (transformers >= 5.2).
Supports 13 languages including EN/ZH/JA.
Apache 2.0 license. ~4.4B params.
"""
import os
from typing import Optional
from .base import ASRRunner
from .registry import register


@register("voxtral-realtime-4b")
class VoxtralRealtime4BRunner(ASRRunner):
    name = "voxtral-realtime-4b"
    languages = ["en", "zh", "ja", "es", "fr", "de", "pt", "ru", "it", "nl", "hi", "ar", "ko"]
    requires_gpu = True

    def __init__(self):
        self.model = None
        self.processor = None
        self.target_sr = None
        self.model_id = os.environ.get(
            "VOXTRAL_MODEL_PATH",
            "mistralai/Voxtral-Mini-4B-Realtime-2602"
        )

    def load(self):
        import torch
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.target_sr = self.processor.feature_extractor.sampling_rate

        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        import torch
        import soundfile as sf

        audio, sr = sf.read(audio_path)

        # Voxtral expects specific sample rate (likely 24kHz) — resample if needed
        if sr != self.target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

        # Process audio through the feature extractor
        inputs = self.processor(audio, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        # Generate transcription
        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        # Decode output tokens
        text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        result = text[0] if text else ""
        return result.strip()

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        super().unload()
