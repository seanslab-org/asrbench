"""FunASR runners: SenseVoice, Paraformer."""
from typing import Optional
from .base import ASRRunner
from .registry import register


@register("sensevoice-large")
class SenseVoiceLargeRunner(ASRRunner):
    name = "sensevoice-large"
    languages = ["en", "zh", "ja"]

    def load(self):
        from funasr import AutoModel
        self.model = AutoModel(
            model="iic/SenseVoiceSmall",  # SenseVoice-Large if available
            trust_remote_code=True,
            device="cuda",
        )

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        lang_map = {"en": "en", "zh": "zh", "ja": "ja"}
        lang = lang_map.get(language, "auto")
        result = self.model.generate(
            input=audio_path,
            language=lang,
            use_itn=True,
        )
        if result and len(result) > 0:
            text = result[0].get("text", "")
            return text.strip()
        return ""

    def unload(self):
        del self.model
        super().unload()


@register("paraformer-large")
class ParaformerLargeRunner(ASRRunner):
    name = "paraformer-large"
    languages = ["en", "zh", "ja"]

    def load(self):
        from funasr import AutoModel
        self.model = AutoModel(
            model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            trust_remote_code=True,
            device="cuda",
        )

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        result = self.model.generate(input=audio_path)
        if result and len(result) > 0:
            text = result[0].get("text", "")
            return text.strip()
        return ""

    def unload(self):
        del self.model
        super().unload()
