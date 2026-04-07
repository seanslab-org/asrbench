"""Moonshine ASR runners — Flavors of Moonshine multilingual edge models.

Models: UsefulSensors/moonshine-{tiny,base}-{en,ja,zh}
Paper:  arXiv 2509.02523 — "Flavors of Moonshine: Tiny Specialized ASR Models
        for Edge Devices" (Useful Sensors, Sep 2025)
Stack:  HuggingFace transformers (MoonshineForConditionalGeneration), CPU or GPU.
        Each language is a separate monolingual ~27M / ~61M parameter model.
License: Apache 2.0

Notes:
- Replaces the legacy `useful-moonshine` (ONNX) runner. The Flavors models are
  hosted as transformers checkpoints and cannot be loaded by the legacy package.
- For offline use on Jetson, populate the HF cache and run with
  `HF_HUB_OFFLINE=1`. No additional config needed.
"""
from typing import List, Optional

from .base import ASRRunner
from .registry import register


class _MoonshineRunner(ASRRunner):
    """Shared transformers-based loader for any UsefulSensors/moonshine-* model.

    Subclasses set ``name``, ``languages``, and ``model_id``.
    """

    model_id: str = ""  # HF repo ID, e.g. "UsefulSensors/moonshine-tiny-ja"
    requires_gpu = False  # CPU-friendly edge models; CUDA used if available

    def __init__(self):
        self.model = None
        self.processor = None
        self._device = "cpu"
        self._dtype = None

    def load(self):
        import torch
        from transformers import AutoProcessor, MoonshineForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        self.model = MoonshineForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self._dtype,
        ).to(self._device)
        self.model.eval()

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        import torch
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32")

        # Stereo → mono (take first channel)
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Resample to the model's expected sampling rate (16 kHz) if needed
        target_sr = self.processor.feature_extractor.sampling_rate
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        inputs = self.processor(
            audio,
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device, self._dtype)

        # Per Moonshine model card: limit max_length to avoid hallucination loops.
        # Token budget ≈ input frames × (13 tokens / sampling_rate seconds).
        # Floor of 8 protects very short clips from producing nothing.
        seq_lens = inputs.attention_mask.sum(dim=-1)
        max_length = max(8, int((seq_lens * (13.0 / target_sr)).max().item()))

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=max_length)

        text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return text.strip()

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        super().unload()


# ---------------------------------------------------------------------------
# English — original Moonshine (EN-only)
# ---------------------------------------------------------------------------

@register("moonshine-tiny-en")
class MoonshineTinyEnRunner(_MoonshineRunner):
    name = "moonshine-tiny-en"
    languages: List[str] = ["en"]
    model_id = "UsefulSensors/moonshine-tiny"


@register("moonshine-base-en")
class MoonshineBaseEnRunner(_MoonshineRunner):
    name = "moonshine-base-en"
    languages: List[str] = ["en"]
    model_id = "UsefulSensors/moonshine-base"


# ---------------------------------------------------------------------------
# Japanese — Flavors of Moonshine
# ---------------------------------------------------------------------------

@register("moonshine-tiny-ja")
class MoonshineTinyJaRunner(_MoonshineRunner):
    name = "moonshine-tiny-ja"
    languages: List[str] = ["ja"]
    model_id = "UsefulSensors/moonshine-tiny-ja"


@register("moonshine-base-ja")
class MoonshineBaseJaRunner(_MoonshineRunner):
    name = "moonshine-base-ja"
    languages: List[str] = ["ja"]
    model_id = "UsefulSensors/moonshine-base-ja"


# ---------------------------------------------------------------------------
# Chinese — Flavors of Moonshine
# ---------------------------------------------------------------------------

@register("moonshine-tiny-zh")
class MoonshineTinyZhRunner(_MoonshineRunner):
    name = "moonshine-tiny-zh"
    languages: List[str] = ["zh"]
    model_id = "UsefulSensors/moonshine-tiny-zh"


@register("moonshine-base-zh")
class MoonshineBaseZhRunner(_MoonshineRunner):
    name = "moonshine-base-zh"
    languages: List[str] = ["zh"]
    model_id = "UsefulSensors/moonshine-base-zh"
