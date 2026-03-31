"""NVIDIA Parakeet-TDT-1.1B runner — English ASR via NeMo.

1.1B params, ~3-4GB VRAM FP16, TDT (Token-and-Duration Transducer) decoder.
Outputs lowercase text without punctuation.

Requires: pip install nemo_toolkit[asr]
Model: nvidia/parakeet-tdt-1.1b (CC-BY-4.0)
"""
import torch
from typing import Optional
from .base import ASRRunner
from .registry import register


@register("parakeet-tdt-1.1b")
class ParakeetTDT11BRunner(ASRRunner):
    name = "parakeet-tdt-1.1b"
    languages = ["en"]
    requires_gpu = True

    def __init__(self):
        self.model = None

    def load(self):
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-tdt-1.1b"
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        output = self.model.transcribe([audio_path])
        # NeMo returns list of Hypothesis objects with .text attribute
        if hasattr(output[0], "text"):
            return output[0].text.strip()
        return str(output[0]).strip()

    def unload(self):
        del self.model
        self.model = None
        super().unload()
