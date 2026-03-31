"""NVIDIA Parakeet-TDT runners — NeMo-based ASR models.

Parakeet-TDT-1.1B: 1.1B params, English only, no punctuation, ~4.3GB VRAM.
Parakeet-TDT-0.6B-v3: 600M params, 25 European languages, punctuation+caps, ~2-3GB VRAM.

Note: v3 supports European languages only (en, de, fr, es, ru, etc.).
      No Japanese or Chinese support.

Requires: pip install nemo_toolkit[asr]
License: CC-BY-4.0

Local .nemo paths used when HuggingFace is unreachable (e.g. Jetson).
"""
import os
import torch
from typing import Optional
from .base import ASRRunner
from .registry import register


class ParakeetBase(ASRRunner):
    """Base runner for NeMo Parakeet-TDT models."""
    requires_gpu = True
    model_name = ""
    local_nemo_path = ""

    def __init__(self):
        self.model = None

    def load(self):
        import nemo.collections.asr as nemo_asr
        if self.local_nemo_path and os.path.isfile(self.local_nemo_path):
            self.model = nemo_asr.models.ASRModel.restore_from(self.local_nemo_path)
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
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


@register("parakeet-tdt-1.1b")
class ParakeetTDT11BRunner(ParakeetBase):
    name = "parakeet-tdt-1.1b"
    languages = ["en"]
    model_name = "nvidia/parakeet-tdt-1.1b"
    local_nemo_path = os.environ.get(
        "PARAKEET_11B_NEMO_PATH", "/home/x/models/parakeet-tdt-1.1b.nemo"
    )


@register("parakeet-tdt-0.6b-v3")
class ParakeetTDT06BV3Runner(ParakeetBase):
    name = "parakeet-tdt-0.6b-v3"
    languages = ["en"]  # 25 European langs but no JA/ZH
    model_name = "nvidia/parakeet-tdt-0.6b-v3"
    local_nemo_path = os.environ.get(
        "PARAKEET_06BV3_NEMO_PATH", "/home/x/models/parakeet-tdt-0.6b-v3.nemo"
    )
