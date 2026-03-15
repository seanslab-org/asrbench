# ASR Bench — Lessons Learned

## Jetson Orin 16GB (JetPack 5.1.5) Constraints

1. **Python 3.8 lock-in**: JetPack 5 ships Python 3.8.10. Python 3.9 venv works but NVIDIA only provides cp38 PyTorch wheels — no CUDA PyTorch on Python 3.9+. This blocks Moonshine (Keras 3.6 needs 3.9+) and FunASR (setuptools incompatibility).

2. **CTranslate2 has no CUDA on aarch64**: The pip-installed CTranslate2 package for aarch64 lacks CUDA support. faster-whisper falls back to CPU int8, making it ~10x slower than GPU whisper-turbo. Building from source with CUDA would fix this but is non-trivial.

3. **HuggingFace unreachable from Jetson**: DNS resolves to wrong IPs (likely network policy). Workaround: cache models on Mac, rsync to Jetson, run with `HF_HUB_OFFLINE=1`.

4. **whisper-large-v3 OOMs on 16GB**: The 2.88GB model exceeds unified memory during inference. Only whisper-large-v3-turbo (distilled, smaller) fits.

5. **yt-dlp search is unreliable**: `ytsearch1:` queries return wrong videos — silence, music, or wrong language content. Use specific verified YouTube URLs instead.

## Benchmarking Best Practices

6. **Always auto-detect CUDA**: Don't hardcode `device="cuda"`. Check `ctranslate2.get_supported_compute_types("cuda")` and fall back to CPU gracefully. Prevents hangs on systems without GPU support.

7. **Verify audio clips before benchmarking**: Listen to clips, check duration matches expectations. Bad clips waste hours of benchmark time and produce meaningless results.

8. **Long-form CPU benchmarks are very slow**: faster-whisper CPU int8 on a 60-min clip takes ~2.6 hours. Plan accordingly or skip long clips for CPU-only models.
