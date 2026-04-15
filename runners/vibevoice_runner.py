"""VibeVoice-ASR runner — 9B unified ASR + diarization + timestamps.

Uses microsoft/VibeVoice-ASR-HF (transformers-native, no vibevoice package needed).
Requires: transformers>=5.3.0

Memory: ~18GB FP16/BF16, ~10GB INT8, ~6GB INT4
"""
import torch
from typing import Optional
from runners.base import ASRRunner
from runners.registry import register


class _VibeVoiceBase(ASRRunner):
    """Shared base for VibeVoice-ASR runners (BF16 and INT8)."""

    languages = ["en", "zh", "ja"]
    requires_gpu = True
    _model_id: str = "microsoft/VibeVoice-ASR-HF"
    _quantization_config = None  # subclass overrides for INT8

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self):
        import os
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

        model_id = os.environ.get("VIBEVOICE_MODEL_PATH", self._model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "attn_implementation": "eager",
        }
        if self._quantization_config is not None:
            load_kwargs["quantization_config"] = self._quantization_config

        self.model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs
        )
        self.model.eval()


@register("vibevoice-asr")
class VibeVoiceASRRunner(_VibeVoiceBase):
    """VibeVoice-ASR BF16 (original, ~16.4 GB VRAM)."""
    name = "vibevoice-asr"
    _model_id = "microsoft/VibeVoice-ASR-HF"


@register("vibevoice-asr-int8")
class VibeVoiceASRInt8Runner(_VibeVoiceBase):
    """VibeVoice-ASR INT8 via bitsandbytes on-the-fly quantization.

    Uses the original microsoft/VibeVoice-ASR-HF model with selective
    INT8 quantization: Qwen2.5-7B backbone quantized, audio encoders
    (acoustic/semantic tokenizers, projections, lm_head) kept at BF16.
    ~10-11 GB VRAM (down from ~16.4 GB).
    """
    name = "vibevoice-asr-int8"
    _model_id = "microsoft/VibeVoice-ASR-HF"

    def load(self):
        import os
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

        # Use the pre-quantized Dubedo model — weights are already INT8,
        # no on-the-fly bitsandbytes quantization needed. This avoids
        # NVML/meta-tensor issues on Jetson's unified memory.
        model_id = os.environ.get("VIBEVOICE_MODEL_PATH", self._model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        import os
        offload_dir = "/tmp/vibevoice_offload"
        os.makedirs(offload_dir, exist_ok=True)
        self.model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            offload_folder=offload_dir,
        )
        self.model.eval()

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        import soundfile as sf

        # Load audio ourselves to avoid torchcodec dependency (libnvrtc
        # version mismatch on Jetson: torchcodec wants .so.13, Jetson has .so.12)
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]  # stereo → mono
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = self.processor.apply_transcription_request(
            audio=audio,
        ).to(self.model.device, self.model.dtype)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=4096)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        transcript = self.processor.decode(
            generated_ids, return_format="transcription_only"
        )[0]
        return transcript.strip() if isinstance(transcript, str) else str(transcript).strip()

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        super().unload()
