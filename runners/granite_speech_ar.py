"""IBM Granite Speech 4.1 2B (autoregressive) runner.

Model: ibm-granite/granite-speech-4.1-2b
Params: 2B (Granite LLM backbone + speech encoder)
Architecture: encoder + autoregressive LLM decoder via AutoModelForSpeechSeq2Seq
Languages: English (multilingual extension via prompt)
License: Apache 2.0

Published reference numbers (paper / model card):
  LibriSpeech test-clean WER 1.33%, test-other WER 2.50%, AMI 8.09%

Requires:
    pip install transformers>=4.52.1 torchaudio
"""
import os
from typing import Optional

from .base import ASRRunner
from .registry import register


@register("granite-speech-4.1-2b")
class GraniteSpeech41ARRunner(ASRRunner):
    name = "granite-speech-4.1-2b"
    languages = ["en"]
    requires_gpu = True

    DEFAULT_PROMPT = (
        "<|audio|>transcribe the speech with proper punctuation and capitalization."
    )

    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_id = os.environ.get(
            "GRANITE_AR_MODEL_PATH",
            "ibm-granite/granite-speech-4.1-2b",
        )
        self.max_new_tokens = int(os.environ.get("GRANITE_AR_MAX_NEW_TOKENS", "512"))
        # Optional multilingual hint, e.g. "fr"; default is empty (English).
        self.task_prompt = os.environ.get("GRANITE_AR_PROMPT", self.DEFAULT_PROMPT)

    def load(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = self.processor.tokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            device_map=device,
            torch_dtype=dtype,
        )
        self.model.eval()

    def _load_audio(self, audio_path: str):
        """Return mono 16 kHz waveform tensor of shape (1, num_samples)."""
        import torch
        try:
            import torchaudio
            wav, sr = torchaudio.load(audio_path, normalize=True)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            return wav  # (1, T)
        except Exception:
            import soundfile as sf
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            wav = torch.from_numpy(audio).unsqueeze(0)
            return wav

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        import torch

        wav = self._load_audio(audio_path)
        device = next(self.model.parameters()).device

        prompt_str = self.task_prompt
        chat = [{"role": "user", "content": prompt_str}]
        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.processor(
            prompt, wav, device=str(device), return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = outputs[0, num_input_tokens:].unsqueeze(0)
        decoded = self.tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        text = decoded[0] if decoded else ""
        return text.strip()

    def unload(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self.tokenizer = None
        super().unload()
