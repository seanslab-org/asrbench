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
│   ├── short/                     # 30s clips — unit benchmarks
│   │   ├── earnings_call_01.wav   # e.g. Apple earnings call excerpt
│   │   ├── board_meeting_01.wav   # city council / board meeting
│   │   ├── conf_call_noisy_01.wav # conference call with background noise
│   │   └── ...
│   ├── long/                      # Full-length — endurance + real-world
│   │   ├── earnings_15min.wav     # ~15 min earnings call
│   │   ├── meeting_30min.wav      # ~30 min business meeting
│   │   └── hearing_60min.wav      # ~60 min congressional hearing / panel
│   └── ground_truth.json          # {"short/earnings_call_01": "...", "long/earnings_15min": "...", ...}
├── zh/
│   ├── short/
│   │   ├── press_conf_01.wav      # 国新办发布会 excerpt
│   │   ├── earnings_call_01.wav   # Alibaba/Tencent earnings
│   │   ├── conf_call_noisy_01.wav
│   │   └── ...
│   ├── long/
│   │   ├── earnings_15min.wav     # ~15 min 财报电话会
│   │   ├── press_conf_30min.wav   # ~30 min government press conference
│   │   └── meeting_60min.wav      # ~60 min business/shareholder meeting
│   └── ground_truth.json
└── ja/
    ├── short/
    │   ├── earnings_call_01.wav   # 決算説明会 excerpt
    │   ├── press_conf_01.wav      # 記者会見 excerpt
    │   ├── conf_call_noisy_01.wav
    │   └── ...
    ├── long/
    │   ├── earnings_15min.wav     # ~15 min 決算説明会
    │   ├── press_conf_30min.wav   # ~30 min 記者会見
    │   └── shareholder_60min.wav  # ~60 min 株主総会
    └── ground_truth.json
```

### Audio specs
- 16 kHz mono WAV
- **Short clips**: 10–30 seconds, 5–8 per language
- **Long clips**: 15 min / 30 min / 60 min, 1 each per language (3 per lang)
- At least 1 noisy clip per language (conference call / phone quality)
- Ground truth manually verified for short clips
- Ground truth from official transcripts / subtitles for long clips

### Why long audio matters
- Tests model stability over extended inference (memory leaks, drift)
- Tests chunking/streaming strategies
- Measures real-world meeting throughput (RTF at scale)
- Exposes OOM issues on 16GB Jetson
- Many ASR models degrade on long-form audio — this reveals it

## 5. Evaluation & Judges

### Phase 1: ASR Accuracy

| Metric | Unit | Tool | Description |
|--------|------|------|-------------|
| **WER** | % | `jiwer` | Word Error Rate — primary metric for EN |
| **CER** | % | `jiwer` | Character Error Rate — primary metric for ZH, JA |
| **SeMaScore** | 0–1 | `semascore` | Semantic similarity score — catches "correct meaning, different wording" |
| **RTF** | ratio | custom | Real-Time Factor (processing_time / audio_duration) |
| **VRAM Peak** | MB | `torch.cuda` | Max GPU memory during inference |
| **Load Time** | sec | custom | Time from init to first-inference-ready |
| **First Token** | ms | custom | Latency to first output token (streaming models) |
| **Long-form Drift** | % | custom | WER/CER delta between first 5min and last 5min of long clips |
| **Memory Stability** | MB | custom | VRAM growth over time during long-form inference |

**ASR Judging pipeline:**
1. **Normalization** — lowercase, strip punctuation, normalize unicode (NFKC for JA/ZH)
2. **WER/CER** via `jiwer` — standard lexical accuracy
3. **SeMaScore** — semantic similarity using sentence embeddings (catches paraphrase-level correctness that WER misses, e.g. "can't" vs "cannot")
4. **Composite score** = `0.6 × (1 - WER/CER) + 0.2 × SeMaScore + 0.1 × (1/RTF_norm) + 0.1 × (1/VRAM_norm)` — single ranking metric balancing accuracy, meaning, speed, efficiency

### Phase 2: Speaker Diarization

| Metric | Unit | Tool | Description |
|--------|------|------|-------------|
| **DER** | % | `pyannote.metrics` / `spyder` | Diarization Error Rate — gold standard |
| **JER** | % | `pyannote.metrics` | Jaccard Error Rate — per-speaker fairness |
| **Missed Speech** | % | `pyannote.metrics` | Reference speech not detected |
| **False Alarm** | % | `pyannote.metrics` | Speech detected where none exists |
| **Confusion** | % | `pyannote.metrics` | Speech attributed to wrong speaker |
| **cpWER** | % | custom | Concatenated minimum-permutation WER (ASR+diarization combined) |

**Diarization judging pipeline:**
1. Ground truth RTTM files with speaker labels + timestamps
2. **DER** via `pyannote.metrics` with 0.25s collar (standard forgiveness window)
3. **cpWER** — combines ASR accuracy with speaker assignment: concatenate each speaker's text in optimal permutation, compute WER. This is the ultimate "meeting transcription quality" metric.
4. **Composite diarization score** = `0.5 × (1 - DER) + 0.3 × (1 - cpWER) + 0.2 × (1/RTF_norm)`

## 5b. Speaker Diarization Models

### Standalone Diarization

| Model | Params | VRAM | Notes |
|-------|--------|------|-------|
| **pyannote/speaker-diarization-3.1** | ~5M | <1 GB | Best DER overall (11.2%), gold standard |
| **NVIDIA NeMo MSDD** | ~25M | ~1 GB | Multi-scale decoder, strong on meetings |
| **NVIDIA Sortformer** | ~18 layers | ~1 GB | End-to-end transformer, no pipeline stages |
| **SpeechBrain ECAPA-TDNN** | ~15M | <1 GB | Speaker embedding + clustering |
| **DiariZen** | ~10M | <1 GB | Open-source, 13.3% DER |

### Combined ASR + Diarization Pipelines

| Pipeline | ASR | Diarization | Notes |
|----------|-----|-------------|-------|
| **WhisperX** | Whisper large-v3 | pyannote 3.1 | Word-level speaker labels, 70x realtime |
| **NeMo Canary + MSDD** | Canary-1B | NeMo MSDD | NVIDIA native, optimized for Jetson |
| **FunASR SOND** | Paraformer | Built-in | End-to-end, speaker-overlap aware |
| **Qwen3-ASR + pyannote** | Qwen3-ASR | pyannote 3.1 | Custom pipeline |

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
├── diarization/
│   ├── __init__.py
│   ├── base.py                # Abstract diarization runner
│   ├── pyannote_runner.py     # pyannote 3.1
│   ├── nemo_runner.py         # NeMo MSDD / Sortformer
│   ├── speechbrain_runner.py  # ECAPA-TDNN
│   ├── diarizen_runner.py     # DiariZen
│   └── combined_runner.py     # WhisperX, NeMo Canary+MSDD, FunASR SOND
├── metrics/
│   ├── __init__.py
│   ├── wer.py                 # WER/CER computation
│   ├── semascore.py           # Semantic similarity scoring
│   ├── diarization.py         # DER/JER/cpWER computation
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

### Phase 1: ASR Benchmark
1. **Setup** — Create venv on Jetson, install base deps (torch, torchaudio, jiwer, semascore)
2. **Data** — Download/prepare test clips (short + long), write ground truth JSON
3. **Runners** — Implement one at a time, test each individually
4. **Bench** — `python bench.py --phase asr` runs all ASR models × all clips
5. **Report** — Generate ranked table with composite scores

### Phase 2: Speaker Diarization Benchmark
6. **Data prep** — Create RTTM ground truth files with speaker labels for all test clips
7. **Diarization runners** — Implement standalone + combined pipeline runners
8. **Bench** — `python bench.py --phase diarization` runs all diarization models
9. **Combined eval** — `python bench.py --phase combined` runs ASR+diarization pipelines, measures cpWER
10. **Report** — Final comparison: best ASR, best diarization, best combined pipeline

## 8. Constraints & Risks

| Risk | Mitigation |
|------|-----------|
| Model exceeds 16GB VRAM | Skip and log; test with smaller variant |
| Model doesn't support a language | Skip that language, mark N/A |
| NeMo/ONNX dependency conflicts | Use separate venvs per model family if needed |
| Slow models timeout | Set 5-minute timeout per clip |
| Noisy ground truth | Use well-known benchmark clips with verified transcriptions |

## 9. Success Criteria

- ≥12 ASR models benchmarked across 3 languages
- ≥4 diarization models/pipelines benchmarked
- ≥3 combined ASR+diarization pipelines with cpWER
- Long-form (15/30/60 min) stability tested
- Reproducible results via `python bench.py`
- Clear winner(s) identified: best ASR per language, best diarization, best combined pipeline
- Results pushed to GitHub
