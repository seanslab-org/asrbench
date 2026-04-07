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

        self._wav_dir = Path(data_dir) / "standard" / "reazonspeech" / "test_wav"
        self._wav_dir.mkdir(exist_ok=True)
        self.samples = []

        # Fast path: read transcriptions via pyarrow directly from the arrow
        # shards. This avoids the HuggingFace `datasets` Audio decoder, which
        # in newer versions pulls in torchcodec and breaks on systems without
        # the matching libnvrtc.
        transcriptions = self._read_transcriptions_pyarrow(root)
        if transcriptions is not None:
            for i, text in enumerate(transcriptions):
                wav_path = self._resolve_wav(i)
                if wav_path is None:
                    break
                self.samples.append(Sample(
                    audio_path=str(wav_path),
                    reference=text,
                    language="ja",
                    sample_id=f"reazonspeech_{i:05d}",
                ))
            return

        # Fallback path: use the HuggingFace `datasets` library (slower, may
        # require torchcodec to be loadable even with decode=False).
        from datasets import load_from_disk
        try:
            from datasets import Audio
            self._ds = load_from_disk(str(root))
            self._ds = self._ds.cast_column("audio", Audio(decode=False))
        except Exception:
            self._ds = load_from_disk(str(root))
        for i, item in enumerate(self._ds):
            wav_path = self._resolve_wav(i)
            if wav_path is None:
                audio = item["audio"]
                if isinstance(audio, dict) and "array" in audio:
                    wav_path = self._wav_dir / f"reazonspeech_{i:05d}.wav"
                    sf.write(str(wav_path), audio["array"], audio["sampling_rate"])
                else:
                    break
            self.samples.append(Sample(
                audio_path=str(wav_path),
                reference=item["transcription"],
                language="ja",
                sample_id=f"reazonspeech_{i:05d}",
            ))

    def _resolve_wav(self, i: int):
        """Return the existing WAV path for sample i, or None if missing."""
        for stem in (f"reazonspeech_{i:05d}", f"rs_{i:05d}"):
            p = self._wav_dir / f"{stem}.wav"
            if p.exists():
                return p
        return None

    def _read_transcriptions_pyarrow(self, root: Path):
        """Read just the `transcription` column from the arrow shards.

        Returns a list of strings, or None if pyarrow reading fails (caller
        should fall back to the datasets-library path).
        """
        try:
            import pyarrow as pa  # noqa: F401
            import pyarrow.ipc as ipc
        except ImportError:
            return None
        shards = sorted(root.glob("data-*.arrow"))
        if not shards:
            return None
        out = []
        try:
            for shard in shards:
                with pa.memory_map(str(shard), "r") as source:
                    # HuggingFace save_to_disk writes file-format arrow shards
                    reader = ipc.RecordBatchFileReader(source)
                    for i in range(reader.num_record_batches):
                        batch = reader.get_batch(i)
                        col = batch.column("transcription")
                        out.extend(col.to_pylist())
        except Exception:
            return None
        return out

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
