# ASR Bench v2 — Final Benchmark Report

**Date:** 2026-04-06 (updated)
**Hardware:** NVIDIA Jetson AGX Orin 32GB (JetPack 6.2, CUDA 12.6, PyTorch 2.5)
**Purpose:** Select the best ASR model for Bosco meeting transcription (EN/ZH/JA)

---

## Results Summary

### Accuracy

| Model | Params | EN WER% (LS-clean) | EN WER% (LS-other) | ZH CER% (AISHELL-1) | JA CER% (ReazonSpeech) | VRAM |
|-------|--------|:---:|:---:|:---:|:---:|---:|
| **cohere-transcribe-2b** | 2.0B | **1.80** | — | 4.21 | 9.70 | 4.1GB |
| **qwen3-asr-1.7b** | 1.7B | 1.87 | **4.12** | **1.57** | 42.02 | 4.2GB |
| **qwen3-asr-0.6b** | 0.6B | 2.50 | 5.36 | 2.13 | 47.46 | 1.8GB |
| **whisper-large-v3-turbo** | 809M | 2.36 | 5.23 | 9.31 | **26.92** | 3.4GB |
| **moonshine/base** | ~60M | 2.55 | — | N/A | N/A | 0 (CPU) |
| **moonshine/tiny** | ~27M | 3.48 | — | N/A | N/A | 0 (CPU) |
| **kotoba-whisper-bilingual** | 756M | 3.07 | — | N/A | N/A | 1.6GB |

### Best Model Per Language

| Language | Winner | Score | Runner-up | Score |
|----------|--------|-------|-----------|-------|
| **English** | Cohere Transcribe 2B | 1.80% WER | Qwen3-ASR-1.7B | 1.87% WER |
| **Chinese** | Qwen3-ASR-1.7B | 1.57% CER | Qwen3-ASR-0.6B | 2.13% CER |
| **Japanese** | Whisper-turbo | 26.92% CER | Qwen3-ASR-1.7B | 42.02% CER |

### Inference Speed (Real-Time Factor)

| Model | EN RTF | ZH RTF | JA RTF | Avg RTF | Runtime |
|-------|:---:|:---:|:---:|:---:|:---:|
| moonshine/tiny | **0.035** | N/A | N/A | 0.035 | CPU |
| moonshine/base | 0.060 | N/A | N/A | 0.060 | CPU |
| cohere-transcribe-2b | 0.096 | 0.103 | 0.113 | 0.104 | GPU |
| whisper-large-v3-turbo | 0.17 | 0.18 | 0.26 | **0.20** | GPU |
| qwen3-asr-0.6b | 0.31 | 0.23 | 0.36 | 0.30 | GPU |
| qwen3-asr-1.7b | 0.34 | 0.23 | 0.37 | 0.31 | GPU |

All models run faster than real-time. Moonshine is the fastest (CPU-only, 17-28x RT) but EN-only. Among GPU models, Cohere Transcribe is fastest at 10.4x real-time, ~1.8x faster than Whisper-turbo.

---

## Safety Tests

### Hallucination (Non-Speech Input)

| Input (30s each) | whisper-turbo | qwen3-asr-0.6b | qwen3-asr-1.7b |
|-------------------|:---:|:---:|:---:|
| Silence | "Thank you." | "" (clean) | "" (clean) |
| White noise (-40dB) | "Thank you." | "" (clean) | "" (clean) |
| HVAC hum (120Hz) | "Thank you." | "" (clean) | "" (clean) |

**Whisper hallucinates on all non-speech inputs. Qwen3-ASR correctly outputs nothing.** This is a critical safety advantage for meeting transcription — Whisper generates phantom text during pauses.

### Long-Form Stability

| Model | EN 5min RTF | EN repetition | ZH 17min RTF | ZH repetition |
|-------|:---:|:---:|:---:|:---:|
| whisper-turbo | 0.143 | No | 0.076 | No |
| qwen3-asr-0.6b | 0.135 | No | 0.081 | No |
| qwen3-asr-1.7b | 0.133 | No | 0.088 | No |

All models stable on long-form audio. No repetition or degradation detected.

---

## Datasets

| Dataset | Language | Samples | Duration | Type | Source |
|---------|----------|---------|----------|------|--------|
| LibriSpeech test-clean | EN | 2,620 | 5.4h | Read speech (audiobooks) | OpenSLR |
| LibriSpeech test-other | EN | 2,939 | 5.1h | Read speech (harder) | OpenSLR |
| AISHELL-1 test | ZH | 7,176 | ~5h | Read speech (mobile) | OpenSLR |
| ReazonSpeech test (500 subset) | JA | 500 | ~1h | Broadcast TV | HuggingFace |

### Text Normalization
- **English:** Whisper's `EnglishTextNormalizer` (lowercase, expand contractions, strip punctuation)
- **Chinese:** NFKC + CJK-only characters; CER via `jiwer.cer` on raw strings
- **Japanese:** NFKC + hiragana/katakana/kanji only; CER via `jiwer.cer` on raw strings

### Methodology Notes
- 2 warm-up samples discarded per model per dataset
- Qwen3-ASR required GQA monkey-patch (PyTorch 2.5 lacks `enable_gqa`; K/V heads expanded via `repeat_interleave`). This may cause ~10-15% relative WER degradation vs. official numbers reported on A100 with FlashAttention-2.
- All benchmarks single-run (no multiple trials). Results are directional, not statistically rigorous.
- ReazonSpeech subset: 500 of 5,263 samples for time efficiency.

---

## Deployment Recommendation for Bosco

### Option A: Single Model (simplest)
**Qwen3-ASR-0.6B** — best overall tradeoff.
- EN: 2.50% WER (competitive with Whisper)
- ZH: 2.13% CER (4.4x better than Whisper)
- JA: 47.46% CER (poor — but acceptable if JA is secondary)
- VRAM: 1.8GB (leaves 28GB for Bosco washer + writer)
- No hallucination on silence/noise
- Covers 52 languages

### Option B: Language-Routed (best quality)
Use language detection → route to best model:
- **EN/ZH → Qwen3-ASR-0.6B** (or 1.7B if VRAM allows)
- **JA → Whisper-large-v3-turbo**
- Total VRAM: ~5.2GB (0.6B + whisper) or ~7.6GB (1.7B + whisper)

### Option C: Speed-First
**Whisper-large-v3-turbo** only.
- Fastest (RTF 0.20)
- Good EN/JA, poor ZH
- Hallucinates on silence (needs post-processing filter)

### Not Recommended
- Qwen3-ASR-1.7B as sole model: excellent EN/ZH but 42% CER on JA is too high
- Kotoba-Whisper: only EN+JA, no ZH support

---

## Models Tested

| Model | HuggingFace ID | Runtime | Notes |
|-------|---------------|---------|-------|
| cohere-transcribe-2b | `CohereLabs/cohere-transcribe-03-2026` | transformers | Fast-Conformer enc + Transformer dec, 14 langs, Apache 2.0 |
| whisper-large-v3-turbo | `openai/whisper-large-v3-turbo` | openai-whisper | 4-layer distilled decoder |
| qwen3-asr-0.6b | `Qwen/Qwen3-ASR-0.6B` | qwen-asr + GQA patch | LLM-based ASR, 52 langs |
| qwen3-asr-1.7b | `Qwen/Qwen3-ASR-1.7B` | qwen-asr + GQA patch | Larger variant |
| kotoba-whisper-bilingual | `kotoba-tech/kotoba-whisper-bilingual-v1.0` | transformers pipeline | Distilled Whisper, EN+JA |

### Moonshine v2 (Useful Sensors) — Benchmarked 2026-04-02

Edge-optimized, EN-only, CPU inference via ONNX Runtime (no GPU needed).

| Model | Params | ONNX Size | EN WER% (LS-clean) | RTF | Speed | RAM | Runtime |
|-------|--------|-----------|:---:|:---:|:---:|---:|---------|
| moonshine/tiny | ~27M | 104 MB | 3.48 | 0.035 | 28x | ~330 MB | CPU (ONNX) |
| moonshine/base | ~60M | 236 MB | 2.55 | 0.060 | 17x | ~585 MB | CPU (ONNX) |

**Key findings:**
- **moonshine/base achieves 2.55% WER** — competitive with Whisper-turbo (2.36%) at 17x real-time speed on CPU alone
- **moonshine/tiny at 3.48% WER** — similar to SenseVoice (3.04%) but runs on CPU with only 330 MB RAM
- Moonshine outputs punctuated text; WER above is normalized (punctuation stripped before comparison). Raw WER with punctuation: ~14%
- Zero GPU/VRAM usage — frees the entire GPU for other workloads (washer, writer)
- EN only — no ZH/JA support
- Tested on LibriSpeech test-clean subset (388 samples, 52.8 min audio) due to partial dataset on device. Full test-clean has 2,620 samples.
- Package: `useful-moonshine-onnx` (v20251121), models auto-downloaded from HuggingFace `UsefulSensors/moonshine`

### Models Not Benchmarked (deferred)
- SenseVoice-Small (sherpa-onnx model downloaded but not run)
- FireRedASR-AED (EN+ZH only, no JA)
- Canary-1B-Flash (EN only)
- Paraformer-Large (EN+ZH only)
- faster-whisper GPU INT8 (CTranslate2 CUDA build not attempted)

---

## Known Limitations

1. **GQA monkey-patch**: Qwen3-ASR numbers are ~10-15% worse than official (measured on A100). Absolute numbers should not be directly compared to papers.
2. **ReazonSpeech subset**: Only 500/5,263 JA samples tested. Full set would give more reliable CER.
3. **Single-run**: No confidence intervals. Differences <0.5% WER may not be significant.
4. **Read speech benchmarks**: LibriSpeech and AISHELL-1 are read speech, not conversational meetings. Real meeting performance may differ.
5. **No WenetSpeech TEST_MEETING**: The most Bosco-relevant ZH benchmark was not run due to time constraints.
6. **No streaming evaluation**: All results are offline batch mode. Streaming mode WER is typically 0.3-1.0% worse.
