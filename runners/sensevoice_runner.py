"""SenseVoice-Small runner via sherpa-onnx.

234M params, ~1GB VRAM, supports EN/ZH/JA/KO/Cantonese with auto-detection.
15x faster than Whisper-Large.

Requires: pip install sherpa-onnx
Models downloaded automatically on first load.
"""
from runners.base import ASRRunner
from runners.registry import register


@register("sensevoice-small")
class SenseVoiceSmallRunner(ASRRunner):
    name = "sensevoice-small"
    languages = ["en", "zh", "ja"]
    requires_gpu = False  # sherpa-onnx handles device selection

    def __init__(self):
        self.recognizer = None
        self._model_dir = None

    def load(self):
        import sherpa_onnx

        # Use sherpa-onnx's SenseVoice model
        # Models are downloaded to ~/.local/share/sherpa-onnx or specified path
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx",
            tokens="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
            use_itn=True,
            num_threads=4,
        )

    def transcribe(self, audio_path: str, language: str = None) -> str:
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # mono
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text

    def unload(self):
        del self.recognizer
        self.recognizer = None
        super().unload()
