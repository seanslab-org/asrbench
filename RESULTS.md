# ASR Bench v2 — Final Benchmark Report

**Date:** 2026-04-09 (updated — added iPad mini 6 + iOS Simulator results)
**Hardware:** NVIDIA Jetson AGX Orin 32GB, iPad mini 6 (A15 Bionic), Mac mini (Apple Silicon)
**Purpose:** Select the best ASR model for Bosco / lonote / Memio (EN/ZH/JA)

---

## Results Summary

### Accuracy

| Model | Params | EN WER% (LS-clean) | EN WER% (LS-other) | ZH CER% (AISHELL-1) | JA CER% (ReazonSpeech) | VRAM |
|-------|--------|:---:|:---:|:---:|:---:|---:|
| **cohere-transcribe-2b** | 2.0B | **1.80** | — | 4.21 | 9.70 | 4.1GB |
| **qwen3-asr-1.7b** | 1.7B | 1.87 | **4.12** | **1.57** | 42.02 | 4.2GB |
| **qwen3-asr-0.6b** | 0.6B | 2.50 | 5.36 | 2.13 | 47.46 | 1.8GB |
| **whisper-large-v3-turbo** | 809M | 2.36 | 5.23 | 9.31 | 26.92 | 3.4GB |
| **moonshine-base-{en,ja,zh}** † | 61M each | 4.54 | — | 6.88 ‡ | **7.11** | 153MB ea |
| **moonshine-tiny-{en,ja,zh}** † | 27M each | 5.90 | — | 8.99 ‡ | 13.59 | 78MB ea |
| **vibevoice-asr** §  | ~9B | 3.11 § | — | n/t | n/t | 16.4GB |
| **kotoba-whisper-bilingual** | 756M | 3.07 | — | N/A | N/A | 1.6GB |

† Moonshine "Flavors" — separate monolingual models per language (arXiv 2509.02523).
Listed VRAM is per loaded model on CUDA fp16; CPU fp32 is also viable at 0 VRAM.
‡ AISHELL-1 numbers are on a 1,000-sample subset (full set has 7,176).
§ VibeVoice-ASR benchmarked on a 386-sample LibriSpeech subset only (not full 2620).
ZH/JA not yet tested. Has built-in speaker diarization (unique among tested models).
n/t = not tested.

### Best Model Per Language

| Language | Winner | Score | Runner-up | Score |
|----------|--------|-------|-----------|-------|
| **English** | Cohere Transcribe 2B | 1.80% WER | Qwen3-ASR-1.7B | 1.87% WER |
| **Chinese** | Qwen3-ASR-1.7B | 1.57% CER | Qwen3-ASR-0.6B | 2.13% CER |
| **Japanese** | **Moonshine-base-ja** | **7.11% CER** | Cohere Transcribe 2B | 9.70% CER |

### Inference Speed (Real-Time Factor)

| Model | EN RTF | ZH RTF | JA RTF | Avg RTF | Runtime |
|-------|:---:|:---:|:---:|:---:|:---:|
| moonshine-tiny-{en,ja,zh} (transformers) | **0.088** | **0.128** | **0.131** | **0.116** | GPU fp16 |
| moonshine-base-{en,ja,zh} (transformers) | 0.112 | 0.187 | 0.164 | 0.154 | GPU fp16 |
| moonshine/tiny (legacy ONNX, EN only) | 0.035 | — | — | — | CPU |
| moonshine/base (legacy ONNX, EN only) | 0.060 | — | — | — | CPU |
| cohere-transcribe-2b | 0.096 | 0.103 | 0.113 | 0.104 | GPU |
| whisper-large-v3-turbo | 0.17 | 0.18 | 0.26 | 0.20 | GPU |
| qwen3-asr-0.6b | 0.31 | 0.23 | 0.36 | 0.30 | GPU |
| qwen3-asr-1.7b | 0.34 | 0.23 | 0.37 | 0.31 | GPU |
| vibevoice-asr | 0.84 | n/t | n/t | 0.84 | GPU (16.4GB) |

All models run faster than real-time. The Moonshine Flavors models on GPU fp16 sit in
the same speed class as Cohere (~6–11× real-time). The legacy ONNX runtime is roughly
3× faster on CPU for English-only deployments and uses zero VRAM, but cannot load the
new multilingual variants.

### iOS / iPad Results (Moonshine via moonshine-swift v0.0.51)

Benchmarked on iPad mini 6 (A15 Bionic, iPadOS 26.3) and iOS Simulator (Mac mini CPU).
50 LibriSpeech test-clean samples, single speaker subset.

| Model | Platform | WER% | RTF | Latency | Notes |
|-------|----------|:---:|:---:|:---:|---|
| moonshine-base-en | **iPad mini 6** | **2.45** | 0.053 | 0.396s | A15 Bionic, real hardware |
| moonshine-tiny-en | **iPad mini 6** | 4.69 | 0.036 | 0.267s | A15 Bionic, real hardware |
| apple-speech-en-US | **iPad mini 6** | 10.92 | 0.038 | 0.280s | SFSpeechRecognizer on-device |
| moonshine-base-en | Mac Simulator | 2.55 | 0.034 | 0.253s | Mac CPU, not iPhone hardware |
| moonshine-tiny-en | Mac Simulator | 4.59 | 0.024 | 0.177s | Mac CPU, not iPhone hardware |

**Key finding:** Moonshine-base beats Apple's native `SFSpeechRecognizer` by **4.5×
on accuracy** (2.45% vs 10.92% WER) at comparable speed on the same iPad hardware.
All runners at 19–28× real-time on A15 Bionic, 0 errors across 150 total records.

**Note:** Simulator RTF is Mac CPU latency, not iPhone hardware. The iPad numbers
are the authoritative real-device measurements. Apple's new iOS 26
`SpeechTranscriber` API crashes on iPad mini 6 (SIGTRAP — not in Apple's hardware
allowlist), so the legacy `SFSpeechRecognizer` was used instead.

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

## Deployment Recommendation for Bosco / lonote

### Option A: Single Model (recommended for "best EN")
**Cohere Transcribe 2B** — best overall across all three languages.
- EN: 1.80% WER (#1 — beats all other models)
- ZH: 4.21% CER (good, though Qwen3 is better at 1.57%)
- JA: 9.70% CER (now #2 — beaten by moonshine-base-ja at 7.11%)
- VRAM: 4.1GB (leaves ~28GB for Bosco washer + writer)
- Fastest GPU model: RTF 0.104 avg (~9.6x real-time)
- 14 languages, Apache 2.0 license

### Option B: Language-Routed (best quality per language)
Use language detection → route to best model. With Moonshine Flavors now in play:
- **EN → Cohere Transcribe 2B** (1.80% WER, beats Moonshine's 4.54%)
- **ZH → Qwen3-ASR-0.6B** (2.13% CER) or **1.7B** (1.57% CER)
- **JA → Moonshine-base-ja** (7.11% CER, **new leader**, 153 MB VRAM)
- Cohere + Qwen + Moonshine-ja total ≈ 6 GB VRAM. Leaves ~26 GB for the LLM.

### Option C: Minimum VRAM, full multilingual (Moonshine-only) — **new**
**Moonshine-base × {en, ja, zh}** — three small specialized models, all GPU fp16.
- EN: 4.54% WER (vs Cohere 1.80) — competitive but ~2.5× worse than the leader
- ZH: 6.88% CER on AISHELL-1 (1k subset) — beats Whisper-turbo (9.31%), worse than Cohere/Qwen
- JA: **7.11% CER — best in the bench**, beats Cohere (9.70%) and Whisper-turbo (26.92%) by a wide margin
- VRAM: ~460 MB total if all three loaded; ~150 MB if hot-swapped per call
- RTF 0.11–0.19 (5–9× real-time) on Orin GPU
- Trade-off: ~9× less VRAM than Cohere alone, with a JA win and an EN loss
- 0 errors, 0 hallucinations across 4,114 samples in this run

### Option D: Minimum VRAM with one model
**Qwen3-ASR-0.6B** — 1.8GB VRAM, good EN/ZH (2.50/2.13), poor JA (47.46% CER).
- Best choice when GPU memory is tight AND Japanese is not required
- No hallucination on silence/noise
- 52 languages

### Option E: Pure CPU, EN-only (legacy ONNX path)
**Moonshine/base via `useful-moonshine-onnx`** — 0 VRAM.
- 2.55% EN WER on a 388-sample LS-clean subset (April 2026 bench)
- 17× real-time on CPU
- Cannot load the new Flavors multilingual models — separate runtime

### Not Recommended
- **Whisper-large-v3-turbo** as sole model: hallucinates on silence, poor ZH (9.31% CER), poor JA (26.92% — beaten by tiny CPU models)
- **Qwen3-ASR-1.7B** as sole model: excellent EN/ZH but 42% CER on JA
- **Kotoba-Whisper**: only EN+JA, no ZH support

---

## Models Tested

| Model | HuggingFace ID | Runtime | Notes |
|-------|---------------|---------|-------|
| cohere-transcribe-2b | `CohereLabs/cohere-transcribe-03-2026` | transformers | Fast-Conformer enc + Transformer dec, 14 langs, Apache 2.0 |
| whisper-large-v3-turbo | `openai/whisper-large-v3-turbo` | openai-whisper | 4-layer distilled decoder |
| qwen3-asr-0.6b | `Qwen/Qwen3-ASR-0.6B` | qwen-asr + GQA patch | LLM-based ASR, 52 langs |
| qwen3-asr-1.7b | `Qwen/Qwen3-ASR-1.7B` | qwen-asr + GQA patch | Larger variant |
| kotoba-whisper-bilingual | `kotoba-tech/kotoba-whisper-bilingual-v1.0` | transformers pipeline | Distilled Whisper, EN+JA |
| moonshine-{tiny,base}-en | `UsefulSensors/moonshine-{tiny,base}` | transformers | EN-only edge ASR, ~27 M / 61 M params |
| moonshine-{tiny,base}-ja | `UsefulSensors/moonshine-{tiny,base}-ja` | transformers | JA-only Flavors, arXiv 2509.02523 |
| moonshine-{tiny,base}-zh | `UsefulSensors/moonshine-{tiny,base}-zh` | transformers | ZH-only Flavors, arXiv 2509.02523 |
| vibevoice-asr | `microsoft/VibeVoice-ASR-HF` | transformers | ~9B params, built-in diarization, 50+ langs |

### VibeVoice-ASR (Microsoft) — Benchmarked 2026-03-26

Unified speech-to-text model (~9B params, Qwen2.5 backbone) with built-in speaker
diarization. Handles 60-min long-form audio in a single pass, generating structured
transcriptions with Who (Speaker), When (Timestamps), and What (Content).

| Metric | Value |
|---|---|
| EN WER (LS-clean, 386 samples) | **3.11%** |
| WER (median) | 0.00% |
| Perfect transcriptions | 269/386 (70%) |
| RTF (avg) | 0.841 (barely real-time) |
| RTF (range) | 0.401 – 1.763 |
| VRAM (peak, BF16) | **16,434 MB** |
| Load time | 426.5s (~7 min from NTFS SSD) |
| Hallucinations | 0 |
| Errors | 0 |

**Key findings:**
- **Only model in the bench with built-in speaker diarization** — all others need
  pyannote bolted on separately
- **3.11% WER on a 386-sample subset** — competitive but not the leader (Cohere 1.80%,
  Qwen3-1.7B 1.87%). Official Open ASR Leaderboard reports 2.20% on full LS-clean
- **16.4 GB VRAM (BF16)** — eats half the Orin's 32 GB, leaving little room for LLM.
  Community INT8 quantization (~12 GB) and INT4 (~6.6 GB) exist but WER impact on ASR
  is unverified
- **RTF 0.84** — barely real-time. A 1-hour meeting takes ~50 min to transcribe.
  Slowest model in the bench by a wide margin
- Loaded via `VibeVoiceAsrForConditionalGeneration` from transformers with
  `attn_implementation="eager"` (SDPA not supported for acoustic tokenizer on Jetson)
- **ZH/JA not yet benchmarked** — listed as n/t in the summary table
- Model downloaded via parallel curl from HuggingFace CDN to Mac, transferred to Jetson
  via SCP (HF blocked on Jetson's network)

### Cohere Transcribe 2B (Cohere Labs) — Benchmarked 2026-04-05

Fast-Conformer encoder + Transformer decoder, 14 languages, Apache 2.0 license.

**Key findings:**
- **New EN WER leader at 1.80%** on LibriSpeech test-clean (2,620 samples) — beats Qwen3-ASR-1.7B (1.87%)
- **Dramatic JA improvement: 9.70% CER** on ReazonSpeech (500 samples) — 2.8x better than Whisper-turbo (26.92%)
- **Fastest GPU model** at RTF 0.104 avg (~9.6x real-time), ~1.9x faster than Whisper-turbo
- ZH: 4.21% CER on AISHELL-1 — solid but behind Qwen3-ASR (1.57%)
- VRAM: 4.1GB (bfloat16), fits comfortably on Orin 32GB alongside other Bosco services
- Model: `CohereLabs/cohere-transcribe-03-2026`, requires transformers ≥5.5.0
- Ran with `TRANSFORMERS_OFFLINE=1` on Jetson (model pre-cached via LAN rsync)

### Moonshine — Flavors of Moonshine multilingual — Benchmarked 2026-04-07

Useful Sensors' "Flavors of Moonshine" (arXiv 2509.02523, Sep 2025) — six monolingual
specialized ASR models, separate checkpoint per language. We benchmarked all six
EN/JA/ZH variants via HuggingFace transformers (`MoonshineForConditionalGeneration`)
on Jetson Orin 32GB CUDA fp16.

| Model | Params | EN WER % | ZH CER % | JA CER % | RTF | VRAM | Runtime |
|-------|--------|:---:|:---:|:---:|:---:|---:|---------|
| moonshine-tiny-en | 27M | **5.90** | — | — | 0.088 | 83 MB | GPU fp16 |
| moonshine-base-en | 61M | **4.54** | — | — | 0.112 | 161 MB | GPU fp16 |
| moonshine-tiny-zh | 27M | — | **8.99** ‡ | — | 0.128 | 69 MB | GPU fp16 |
| moonshine-base-zh | 61M | — | **6.88** ‡ | — | 0.187 | 139 MB | GPU fp16 |
| moonshine-tiny-ja | 27M | — | — | **13.59** | 0.131 | 78 MB | GPU fp16 |
| moonshine-base-ja | 61M | — | — | **7.11** | 0.164 | 153 MB | GPU fp16 |

‡ AISHELL-1 ZH numbers are on a 1,000-sample subset (full set has 7,176 samples).
EN/JA used full LS-clean (2,620) and ReazonSpeech (500) respectively.

**Key findings:**
- **moonshine-base-ja achieves 7.11% CER on ReazonSpeech** — beats every other model in the
  bench, including the previous JA leader Cohere Transcribe 2B (9.70%) and Whisper-turbo
  (26.92%). Does so with **27× less VRAM** than Cohere (153 MB vs 4.1 GB) and **33× fewer
  parameters** (61 M vs 2 B). This is the headline result.
- moonshine-base-zh at 6.88% CER beats Whisper-turbo (9.31%) but trails Cohere (4.21%) and
  Qwen3-ASR (1.57–2.13%). Still a strong CPU-friendly fallback.
- moonshine-base-en at 4.54% WER is roughly 2.5× the leader (Cohere 1.80%, Qwen3 1.87%).
  Competitive for EN-only deployments but not the top choice when GPU is plentiful.
- 0 errors, 0 hallucinations across 4,114 samples in the bench.
- Model card recipe: limit `max_length = ceil(input_frames × 13 / 16000)` to avoid
  hallucination loops. Confirmed not too aggressive — beam search and longer caps yield
  identical outputs on tested samples.
- Models load via `from transformers import MoonshineForConditionalGeneration, AutoProcessor`.
  Each repo is `UsefulSensors/moonshine-{tiny,base}-{en,ja,zh}` (~100–250 MB safetensors).
- Cached on Jetson via `HF_ENDPOINT=https://hf-mirror.com`. Direct huggingface.co was
  unreachable from both Mac and Jetson during this run; mirror was reliable.

### Moonshine v2 (legacy ONNX, EN-only) — Benchmarked 2026-04-02

Same underlying tiny/base weights as the EN Flavors models above, but loaded via the
`useful-moonshine-onnx` package and run on CPU (no GPU). Listed here for the speed-first
EN-only deployment option (Option E above). Cannot load the multilingual Flavors variants.

| Model | Params | ONNX Size | EN WER% (LS-clean, 388-sample subset) | RTF | RAM |
|-------|--------|-----------|:---:|:---:|---:|
| moonshine/tiny | ~27M | 104 MB | 3.48 | 0.035 | ~330 MB |
| moonshine/base | ~60M | 236 MB | 2.55 | 0.060 | ~585 MB |

The 2026-04-02 ONNX numbers were on a 388-sample subset; the full-set transformers
numbers (2620 samples, 5.90/4.54%) are roughly 1.5–1.8× higher. Subset selection and
runtime/decoding differences both contribute.

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
3. **AISHELL-1 subset for Moonshine**: Moonshine ZH numbers are on a 1,000-sample subset of AISHELL-1's 7,176 test set. Other models in the table used the full set. Full-set Moonshine ZH numbers should be re-run for direct comparability.
4. **Moonshine EN regression vs prior ONNX bench**: 4.54% (full LS-clean, 2,620 samples, transformers/CUDA) vs 2.55% (388-sample subset, ONNX/CPU). The gap is partly subset selection (full set is harder) and partly runtime/decoding differences. Not investigated further.
5. **Single-run**: No confidence intervals. Differences <0.5% WER may not be significant.
6. **Read speech benchmarks**: LibriSpeech and AISHELL-1 are read speech, not conversational meetings. Real meeting performance may differ.
7. **No WenetSpeech TEST_MEETING**: The most Bosco-relevant ZH benchmark was not run due to time constraints.
8. **No streaming evaluation**: All results are offline batch mode. Streaming mode WER is typically 0.3-1.0% worse.
9. **No safety/hallucination tests for Moonshine**: silence/non-speech behavior was not exercised. Worth confirming before production use.
