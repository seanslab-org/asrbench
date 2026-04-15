"""Cohere Transcribe runner — 2B Fast-Conformer encoder + Transformer decoder.

Model: CohereLabs/cohere-transcribe-03-2026
Params: 2B (90%+ in encoder)
Languages: 14 (en, zh, ja, fr, de, it, es, pt, el, nl, pl, ko, vi, ar)
License: Apache 2.0
Requires: transformers with CohereAsrForConditionalGeneration support
"""
import os
from typing import Optional
from .base import ASRRunner
from .registry import register


@register("cohere-transcribe-2b")
class CohereTranscribe2BRunner(ASRRunner):
    name = "cohere-transcribe-2b"
    languages = [
        "en", "zh", "ja", "fr", "de", "it", "es",
        "pt", "el", "nl", "pl", "ko", "vi", "ar",
    ]
    requires_gpu = True

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_id = os.environ.get(
            "COHERE_MODEL_PATH",
            "CohereLabs/cohere-transcribe-03-2026",
        )

    def load(self):
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        import torch
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32")

        # Stereo → mono
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Resample to 16 kHz if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Cohere requires explicit language in processor call
        proc_kwargs = {"sampling_rate": 16000, "return_tensors": "pt"}
        if language:
            proc_kwargs["language"] = language

        inputs = self.processor(audio, **proc_kwargs)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        # Scale max_new_tokens by audio duration (generous 60 tok/s, floor 256, cap 16384)
        audio_duration_s = len(audio) / sr
        max_tokens = max(256, min(16384, int(audio_duration_s * 60)))

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

        # processor.decode on a 2D [batch, seq] tensor returns a list of strings;
        # str(list) produces '["text"]' which the whisper normalizer collapses
        # to empty. Use batch_decode or index to get a clean string.
        if hasattr(self.processor, "batch_decode"):
            texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
            text = texts[0] if texts else ""
        else:
            decoded = self.processor.decode(outputs, skip_special_tokens=True)
            text = decoded[0] if isinstance(decoded, list) else decoded
        return text.strip()

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        super().unload()
