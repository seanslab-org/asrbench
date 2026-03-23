"""Qwen3-ASR runners — LLM-based ASR with 52 language support.

Qwen3-ASR-0.6B: ~3GB VRAM, best quality/VRAM ratio
Qwen3-ASR-1.7B: ~7GB VRAM, state-of-the-art accuracy

Requires: pip install qwen-asr (Python 3.9+)

Note: Qwen3-ASR uses enable_gqa in scaled_dot_product_attention, which
requires PyTorch >= 2.6. We monkey-patch it for PyTorch 2.5 compatibility.
"""
import torch
from runners.base import ASRRunner
from runners.registry import register


def _patch_sdpa_for_gqa():
    """Monkey-patch scaled_dot_product_attention to handle enable_gqa on PyTorch < 2.6.

    GQA (Grouped Query Attention) uses fewer K/V heads than Q heads.
    PyTorch 2.6+ handles this natively with enable_gqa=True.
    For 2.5, we manually repeat K/V heads to match Q heads.
    """
    import torch.nn.functional as F
    _original_sdpa = F.scaled_dot_product_attention

    def _patched_sdpa(*args, **kwargs):
        enable_gqa = kwargs.pop("enable_gqa", False)
        if enable_gqa and len(args) >= 3:
            q, k, v = args[0], args[1], args[2]
            args = args[3:]
            # Expand K/V heads to match Q heads: (B, Hq, S, D) vs (B, Hkv, S, D)
            if q.dim() == 4 and k.dim() == 4 and q.shape[1] != k.shape[1]:
                n_rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            return _original_sdpa(q, k, v, *args, **kwargs)
        return _original_sdpa(*args, **kwargs)

    if not hasattr(F.scaled_dot_product_attention, "_patched"):
        F.scaled_dot_product_attention = _patched_sdpa
        F.scaled_dot_product_attention._patched = True


class QwenASRBase(ASRRunner):
    """Base runner for Qwen3-ASR models."""
    languages = ["en", "zh", "ja"]
    requires_gpu = True
    model_id = ""

    def __init__(self):
        self.model = None

    def load(self):
        _patch_sdpa_for_gqa()
        from qwen_asr import Qwen3ASRModel
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def transcribe(self, audio_path: str, language: str = None) -> str:
        result = self.model.transcribe(audio_path)
        # qwen-asr returns ASRTranscription objects, not dicts
        if isinstance(result, list):
            parts = []
            for r in result:
                if hasattr(r, "text"):
                    parts.append(r.text)
                elif isinstance(r, dict):
                    parts.append(r.get("text", ""))
                else:
                    parts.append(str(r))
            return " ".join(parts)
        elif hasattr(result, "text"):
            return result.text
        elif isinstance(result, dict):
            return result.get("text", "")
        return str(result)

    def unload(self):
        del self.model
        self.model = None
        super().unload()


@register("qwen3-asr-0.6b")
class Qwen3ASR06BRunner(QwenASRBase):
    name = "qwen3-asr-0.6b"
    model_id = "Qwen/Qwen3-ASR-0.6B"


@register("qwen3-asr-1.7b")
class Qwen3ASR17BRunner(QwenASRBase):
    name = "qwen3-asr-1.7b"
    model_id = "Qwen/Qwen3-ASR-1.7B"
