"""Profiling utilities for VRAM, RTF, and latency measurement."""
import time
import wave
import contextlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProfileResult:
    model_name: str = ""
    audio_path: str = ""
    audio_duration_s: float = 0.0
    elapsed_s: float = 0.0
    rtf: float = 0.0  # real-time factor
    vram_before_mb: float = 0.0
    vram_peak_mb: float = 0.0
    vram_delta_mb: float = 0.0
    load_time_s: float = 0.0
    transcript: str = ""
    language: str = ""
    error: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


def get_audio_duration(path: str) -> float:
    """Get duration of an audio file in seconds (WAV, FLAC, etc.)."""
    # Try soundfile first (handles WAV, FLAC, OGG)
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        pass
    # Fallback to wave module (WAV only)
    try:
        with wave.open(path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        pass
    # Last resort: librosa
    try:
        import librosa
        return librosa.get_duration(path=path)
    except Exception:
        return 0.0


def get_vram_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def get_vram_peak_mb() -> float:
    """Get peak GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def reset_vram_peak():
    """Reset peak memory tracker."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


@contextlib.contextmanager
def profile_inference(model_name: str, audio_path: str, language: str = ""):
    """Context manager that profiles a single inference run."""
    result = ProfileResult(
        model_name=model_name,
        audio_path=audio_path,
        language=language,
        audio_duration_s=get_audio_duration(audio_path),
    )

    result.vram_before_mb = get_vram_mb()
    reset_vram_peak()

    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_s = time.perf_counter() - start
        result.vram_peak_mb = get_vram_peak_mb()
        result.vram_delta_mb = result.vram_peak_mb - result.vram_before_mb
        if result.audio_duration_s > 0:
            result.rtf = result.elapsed_s / result.audio_duration_s


def profile_load(runner) -> float:
    """Time model loading."""
    start = time.perf_counter()
    runner.load()
    return time.perf_counter() - start
