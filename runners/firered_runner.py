"""FireRedASR-AED runner — SOTA Mandarin ASR (CER 3.18%).

1.1B params, ~3-5GB VRAM in FP16. Supports EN and ZH (no JA).
Audio must be ≤60s (hallucinates on longer audio).

Requires: torch, transformers, peft, kaldi_native_fbank, sentencepiece, cn2an
"""
import torch
from runners.base import ASRRunner
from runners.registry import register


@register("firered-asr-aed")
class FireRedASRAEDRunner(ASRRunner):
    name = "firered-asr-aed"
    languages = ["en", "zh"]
    requires_gpu = True

    def __init__(self):
        self.model = None
        self._model_dir = None

    def load(self):
        import sys
        from huggingface_hub import snapshot_download

        # Download model files
        self._model_dir = snapshot_download("FireRedTeam/FireRedASR-AED-L")

        # Add model dir to path for imports
        sys.path.insert(0, self._model_dir)

        # Load model using FireRedASR's own loading mechanism
        from model.fireredasr_model import FireRedASR
        self.model = FireRedASR.from_pretrained(
            "aed",
            self._model_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        )

    def transcribe(self, audio_path: str, language: str = None) -> str:
        results = self.model.transcribe(
            [audio_path],
            batch_size=1,
        )
        if results and len(results) > 0:
            return results[0].get("text", "")
        return ""

    def unload(self):
        del self.model
        self.model = None
        super().unload()
