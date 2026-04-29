"""IBM Granite Speech 4.1 2B NAR (non-autoregressive) runner.

Model: ibm-granite/granite-speech-4.1-2b-nar
Architecture: bidirectional editor over CTC text predictions; single LLM
forward pass per utterance ("Non-autoregressive Lattice Editor").
Languages: English, French, German, Spanish, Portuguese.
License: Apache 2.0.

Published reference numbers (model card):
  LibriSpeech test-clean WER 1.29%, test-other WER 2.75%
  RTFx ~1820 on H100 with batch_size=128 (BF16)

Requires:
    pip install transformers==4.57.6 (or 5.5.3) torchaudio
    pip install flash-attn==2.8.3 --no-build-isolation
The NAR decoder asserts attn_implementation == "flash_attention_2"; other
backends produce causal masks even with is_causal=False, garbling output.
"""
import os
from typing import Optional

from .base import ASRRunner
from .registry import register


@register("granite-speech-4.1-2b-nar")
class GraniteSpeech41NARRunner(ASRRunner):
    name = "granite-speech-4.1-2b-nar"
    languages = ["en"]
    requires_gpu = True

    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.model_id = os.environ.get(
            "GRANITE_NAR_MODEL_PATH",
            "ibm-granite/granite-speech-4.1-2b-nar",
        )
        self.attn_impl = os.environ.get(
            "GRANITE_NAR_ATTN", "flash_attention_2"
        )

    def load(self):
        import torch
        from transformers import AutoModel, AutoFeatureExtractor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        kwargs = dict(
            trust_remote_code=True,
            device_map=device,
            dtype=dtype,
        )
        if self.attn_impl:
            kwargs["attn_implementation"] = self.attn_impl

        self.model = AutoModel.from_pretrained(self.model_id, **kwargs).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def _load_audio(self, audio_path: str):
        import torch
        try:
            import torchaudio
            wav, sr = torchaudio.load(audio_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.squeeze(0)
        except Exception:
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            return torch.from_numpy(audio)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        import torch

        waveform = self._load_audio(audio_path)
        device = next(self.model.parameters()).device

        inputs = self.feature_extractor([waveform], device=str(device))
        inputs = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output = self.model.generate(**inputs)

        text = ""
        if hasattr(output, "text_preds"):
            preds = output.text_preds
            if preds:
                text = preds[0]
        return text.strip() if isinstance(text, str) else str(text).strip()

    def unload(self):
        del self.model
        del self.feature_extractor
        self.model = None
        self.feature_extractor = None
        super().unload()
