# ASR Bench — System Design Document

## 1. Objective

Benchmark state-of-the-art open-source ASR models on an NVIDIA Jetson (16GB shared memory) across three languages: Japanese, Chinese (Mandarin), and English. Produce a ranked comparison to identify the best models for edge deployment in multilingual environments.

## 2. Hardware Target

| Spec | Value |
|------|-------|
| Platform | NVIDIA Jetson AGX Orin / Orin NX |
| Memory | 16 GB shared (CPU+GPU) |
| CUDA | 12.x |
| Storage | ≥50 GB free (for models) |
| SSH | `x@<jetson-ip>` via Tailscale |

## 3. Models

### Tier 1 — Multilingual (JA + ZH + EN)

| Model | Params | Est. VRAM | Architecture | Notes |
|-------|--------|-----------|-------------|-------|
| **Whisper large-v3** | 1.5B | ~3 GB | Encoder-Decoder Transformer | Gold standard multilingual |
| **Whisper large-v3-turbo** | 809M | ~1.5 GB | Distilled Whisper | 8x faster, near-same WER |
| **Faster-Whisper large-v3** | 1.5B | ~3 GB | CTranslate2 optimized | 4x faster than vanilla Whisper |
| **Moonshine v2** | 27M–330M | <1 GB | Streaming encoder | 107ms latency, edge-first |
| **SenseVoice-Large** | ~600M | ~2 GB | FunASR | Strong JA/ZH/EN |
| **Paraformer-Large** | ~600M | ~1 GB | Non-autoregressive (FunASR) | Very fast inference |
| **Fun-ASR-Nano** | 0.8B | ~2 GB | FunASR lightweight | 93% acc in noisy envs |
| **Qwen3-ASR-1.7B** | 1.7B | ~4 GB | LLM-based ASR | 52 languages, streaming |
| **Qwen3-ASR-0.6B** | 0.6B | ~1.5 GB | LLM-based ASR | Lighter variant |
| **Meta Omnilingual ASR** | 300M–1B | ~2 GB | 1600+ languages | Use ≤1B variant |
| **Samba-ASR** | ~600M | ~2 GB | SSM (Mamba) based | New architecture |

### Tier 2 — Strong in specific language(s)

| Model | Params | Est. VRAM | Best Language | Notes |
|-------|--------|-----------|--------------|-------|
| **FireRedASR-AED** | 1.1B | ~3 GB | ZH (Mandarin) | SOTA Mandarin CER 3.18% |
| **NVIDIA Canary-Qwen-2.5B** | 2.5B | ~6 GB | EN | #1 Open ASR Leaderboard |
| **NVIDIA Canary-1B-v2** | 1B | ~3 GB | EN + EU langs | Multilingual EU focus |
| **NVIDIA Parakeet-CTC-1.1B** | 1.1B | ~3 GB | EN | Very fast, EN only |
| **Kotoba-Whisper v2** | 1.5B | ~3 GB | JA | Best Japanese-specific |

### Tier 3 — Excluded (too large for 16GB)

| Model | Params | Reason |
|-------|--------|--------|
| FireRedASR-LLM | 8.3B | ~18 GB, exceeds 16GB |
| Meta Omnilingual 7B | 7B | ~15 GB, marginal fit |

## 4. Test Dataset

```
data/
├── en/
│   ├── clean_01.wav          # LibriSpeech clean
│   ├── clean_02.wav
│   ├── noisy_01.wav          # with background noise
│   └── ground_truth.json     # {"clean_01": "transcription...", ...}
├── zh/
│   ├── clean_01.wav          # AISHELL / CommonVoice
│   ├── noisy_01.wav
│   └── ground_truth.json
└── ja/
    ├── clean_01.wav          # CommonVoice / JSUT
    ├── noisy_01.wav
    └── ground_truth.json
```

- 16 kHz mono WAV, 10–30 seconds each
- 5–10 clips per language
- At least 1 noisy clip per language
- Ground truth manually verified

## 5. Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| **WER** | % | Word Error Rate (EN) |
| **CER** | % | Character Error Rate (ZH, JA) |
| **RTF** | ratio | Real-Time Factor (processing_time / audio_duration) |
| **VRAM Peak** | MB | Max GPU memory during inference |
| **Load Time** | sec | Time from init to first-inference-ready |
| **First Token** | ms | Latency to first output token (streaming models) |

## 6. Architecture

```
asrbench/
├── SDD.md                    # This document
├── bench.py                  # Main entry point
├── config.yaml               # Model list, paths, settings
├── runners/
│   ├── __init__.py
│   ├── base.py               # Abstract runner interface
│   ├── whisper_runner.py      # Whisper / Faster-Whisper
│   ├── moonshine_runner.py
│   ├── funasr_runner.py       # SenseVoice, Paraformer, FunASR-Nano
│   ├── qwen_asr_runner.py     # Qwen3-ASR
│   ├── firered_runner.py      # FireRedASR
│   ├── nvidia_runner.py       # Canary, Parakeet
│   ├── kotoba_runner.py       # Kotoba-Whisper
│   ├── samba_runner.py        # Samba-ASR
│   └── meta_runner.py         # Meta Omnilingual
├── metrics/
│   ├── __init__.py
│   ├── wer.py                 # WER/CER computation
│   └── profiler.py            # RTF, VRAM, latency measurement
├── data/                      # Test audio + ground truth
│   ├── en/
│   ├── zh/
│   └── ja/
├── results/                   # Output JSON + charts
├── tasks/
│   └── todo.md
└── requirements.txt
```

### Runner Interface

```python
class ASRRunner:
    name: str
    languages: list[str]       # ["en", "zh", "ja"]

    def load(self) -> None:
        """Load model into GPU memory."""

    def transcribe(self, audio_path: str, language: str) -> str:
        """Transcribe audio file, return text."""

    def unload(self) -> None:
        """Free GPU memory."""
```

### Profiler

```python
@contextmanager
def profile_inference(model_name, audio_path):
    """Measure VRAM, RTF, latency for a single inference."""
    # Record VRAM before
    # Start timer
    yield result
    # Record VRAM peak
    # Compute RTF = elapsed / audio_duration
```

## 7. Execution Plan

1. **Setup** — Create venv on Jetson, install base deps (torch, torchaudio, jiwer)
2. **Data** — Download/prepare test clips, write ground truth JSON
3. **Runners** — Implement one at a time, test each individually
4. **Bench** — `python bench.py` runs all models × all clips, writes `results/`
5. **Report** — Auto-generate markdown table + optional charts

## 8. Constraints & Risks

| Risk | Mitigation |
|------|-----------|
| Model exceeds 16GB VRAM | Skip and log; test with smaller variant |
| Model doesn't support a language | Skip that language, mark N/A |
| NeMo/ONNX dependency conflicts | Use separate venvs per model family if needed |
| Slow models timeout | Set 5-minute timeout per clip |
| Noisy ground truth | Use well-known benchmark clips with verified transcriptions |

## 9. Success Criteria

- ≥12 models benchmarked across 3 languages
- Reproducible results via `python bench.py`
- Clear winner(s) identified per language and overall
- Results pushed to GitHub
