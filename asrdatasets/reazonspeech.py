"""ReazonSpeech test dataset loader (Japanese).

The dataset is loaded from HuggingFace and saved to disk.
Audio is in the 'audio' column, transcription in 'transcription'.

Download (on Mac, then rsync to Jetson):
    python -c "
    from datasets import load_dataset
    ds = load_dataset('reazon-research/reazonspeech', 'all', split='test')
    ds.save_to_disk('data/standard/reazonspeech/test')
    "
"""
import os
import soundfile as sf
import tempfile
from pathlib import Path
from typing import Iterator

from asrdatasets.base import ASRDataset, Sample
from asrdatasets.registry import register_dataset


@register_dataset("reazonspeech")
class ReazonSpeech(ASRDataset):
    name = "reazonspeech"
    language = "ja"

    def __init__(self):
        self.samples = []
        self._ds = None

    def load(self, data_dir: str) -> None:
        root = Path(data_dir) / "standard" / "reazonspeech" / "test"
        if not root.exists():
            raise FileNotFoundError(
                f"ReazonSpeech test not found at {root}. "
                f"Download via HuggingFace datasets and save_to_disk."
            )
        from datasets import load_from_disk
        # Disable audio decoding to avoid torchcodec dependency
        try:
            from datasets import Audio
            self._ds = load_from_disk(str(root))
            self._ds = self._ds.cast_column("audio", Audio(decode=False))
        except Exception:
            self._ds = load_from_disk(str(root))
        # Pre-extract audio to temp wav files for compatibility with runners
        self._wav_dir = Path(data_dir) / "standard" / "reazonspeech" / "test_wav"
        self._wav_dir.mkdir(exist_ok=True)
        self.samples = []
        for i, item in enumerate(self._ds):
            # Support both naming conventions (reazonspeech_XXXXX / rs_XXXXX)
            wav_path = self._wav_dir / f"reazonspeech_{i:05d}.wav"
            if not wav_path.exists():
                alt_path = self._wav_dir / f"rs_{i:05d}.wav"
                if alt_path.exists():
                    wav_path = alt_path
                else:
                    audio = item["audio"]
                    if isinstance(audio, dict) and "array" in audio:
                        sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
                    else:
                        # No more pre-extracted WAVs and can't decode audio
                        break
            self.samples.append(Sample(
                audio_path=str(wav_path),
                reference=item["transcription"],
                language="ja",
                sample_id=f"reazonspeech_{i:05d}",
            ))

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
