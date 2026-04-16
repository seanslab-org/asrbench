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
| **parakeet-tdt-1.1b** ◆ | 1.1B | 1.82 | **3.43** | N/A | N/A | 4.2GB |
| **qwen3-asr-1.7b** | 1.7B | 1.87 | 4.12 | **1.57** | 42.02 | 4.2GB |
| **qwen3-asr-0.6b** | 0.6B | 2.50 | 5.36 | 2.13 | 47.46 | 1.8GB |
| **parakeet-tdt-0.6b-v3** ◆ | 0.6B | 2.56 | 4.63 | N/A | N/A | 2.6GB |
| **whisper-large-v3-turbo** | 809M | 2.36 | 5.23 | 9.31 | 26.92 | 3.4GB |
| **moonshine-base-{en,ja,zh}** † | 61M each | 4.54 | — | 6.88 ‡ | **7.11** | 153MB ea |
| **moonshine-tiny-{en,ja,zh}** † | 27M each | 5.90 | — | 8.99 ‡ | 13.59 | 78MB ea |
| **vibevoice-asr (BF16)** §  | ~9B | 3.11 § | — | n/t | n/t | 16.4GB |
| **vibevoice-asr (INT8)** §¶ | ~9B | 5.51 §¶ | — | n/t | n/t | ~10GB |
| **kotoba-whisper-bilingual** | 756M | 3.07 | — | N/A | N/A | 1.6GB |

† Moonshine "Flavors" — separate monolingual models per language (arXiv 2509.02523).
Listed VRAM is per loaded model on CUDA fp16; CPU fp32 is also viable at 0 VRAM.
‡ AISHELL-1 numbers are on a 1,000-sample subset (full set has 7,176).
§ VibeVoice-ASR benchmarked on a 386-sample LibriSpeech subset only (not full 2620).
ZH/JA not yet tested. Has built-in speaker diarization (unique among tested models).
n/t = not tested.
¶ INT8 preliminary on 50 samples (DGX Spark1 GB10); 386-sample run in progress.
◆ Parakeet-TDT (NVIDIA NeMo) — English-only (1.1B) / 25 European languages (0.6B-v3); **no Chinese or Japanese**. Benchmarked 2026-04-15 on full LS-clean (2618) + LS-other (2937) on Jetson AGX Orin 32GB via NeMo toolkit (greedy decoding; CUDA graphs disabled due to Jetson OOM). 0 errors, 1 hallucination per model on LS-other.

### Best Model Per Language

| Language | Winner | Score | Runner-up | Score |
|----------|--------|-------|-----------|-------|
| **English (LS-clean)** | Cohere Transcribe 2B | 1.80% WER | Parakeet-TDT-1.1B | 1.82% WER |
| **English (LS-other)** | **Parakeet-TDT-1.1B** | **3.43% WER** | Qwen3-ASR-1.7B | 4.12% WER |
| **Chinese** | Qwen3-ASR-1.7B | 1.57% CER | Qwen3-ASR-0.6B | 2.13% CER |
| **Japanese** | **Moonshine-base-ja** | **7.11% CER** | Cohere Transcribe 2B | 9.70% CER |

Note: On LS-clean, Cohere (1.80%) and Parakeet-TDT-1.1B (1.82%) are statistically
indistinguishable (gap = 2 word-errors across 2618 utterances). Parakeet takes
LS-other decisively (0.69 pp lead). **Cohere remains the only top-tier EN model
that also handles ZH/JA** — Parakeet is English-only.

### Inference Speed (Real-Time Factor)

| Model | EN RTF | ZH RTF | JA RTF | Avg RTF | Runtime |
|-------|:---:|:---:|:---:|:---:|:---:|
| moonshine-tiny-{en,ja,zh} (transformers) | **0.088** | **0.128** | **0.131** | **0.116** | GPU fp16 |
| moonshine-base-{en,ja,zh} (transformers) | 0.112 | 0.187 | 0.164 | 0.154 | GPU fp16 |
| moonshine/tiny (legacy ONNX, EN only) | 0.035 | — | — | — | CPU |
| moonshine/base (legacy ONNX, EN only) | 0.060 | — | — | — | CPU |
| cohere-transcribe-2b | 0.096 | 0.103 | 0.113 | 0.104 | GPU |
| parakeet-tdt-0.6b-v3 | 0.039 | n/a | n/a | — | GPU fp32 (NeMo) |
| parakeet-tdt-1.1b | 0.054 | n/a | n/a | — | GPU fp32 (NeMo) |
| whisper-large-v3-turbo | 0.17 | 0.18 | 0.26 | 0.20 | GPU |
| qwen3-asr-0.6b | 0.31 | 0.23 | 0.36 | 0.30 | GPU |
| qwen3-asr-1.7b | 0.34 | 0.23 | 0.37 | 0.31 | GPU |
| vibevoice-asr | 0.84 | n/t | n/t | 0.84 | GPU (16.4GB) |

All models run faster than real-time. The Moonshine Flavors models on GPU fp16 sit in
the same speed class as Cohere (~6–11× real-time). The legacy ONNX runtime is roughly
3× faster on CPU for English-only deployments and uses zero VRAM, but cannot load the
new multilingual variants.

### Parakeet-TDT-1.1B on Sugr-ASR-Bench (10 clips, corpus-fixed) — 2026-04-16

Benchmark on the standalone Sugr-ASR-Bench corpus (NPR news/politics
content, 10-44 min clips, 16 kHz mono) after fixing 5 of 10 clips
that had mismatched audio/reference pairs upstream (see
"Sugr-ASR-Bench corpus fix" note below).

| Clip | Duration | Ref words | WER | RTF | Show |
|------|--------:|----------:|----:|----:|------|
| npr_rt8 | 20:00 | 3,402 | **4.76%** | 0.023 | Rough Translation |
| npr_ted7 | 20:00 | 3,458 | 5.49% | 0.026 | TED Radio Hour |
| npr_politics_ep16 | 22:05 | 4,164 | 6.24% | 0.023 | NPR Politics |
| npr_1a_ep20 | 33:01 | 6,144 | 6.87% | 0.024 | 1A Political Divisions |
| npr_1a_b3 | 44:55 | 8,711 | 7.16% | 0.036 | NPR Politics 10th Anniversary |
| npr_pol_b13 | 19:31 | 3,577 | 7.58% | 0.023 | NPR Politics |
| npr_politics_ep18 | 19:18 | 3,216 | 8.02% | 0.025 | NPR Politics |
| npr_pol_b17 | 17:43 | 3,375 | 8.47% | 0.026 | NPR Politics |
| npr_fa_b5 | 20:00 | 3,120 | 8.75% | 0.025 | Fresh Air Weekend |
| npr_politics_ep22 | 17:24 | 3,352 | 9.16% | 0.023 | NPR Politics |
| **AVG (10 clips)** | **234 min** | **42,519** | **7.25%** | **0.025** | |

All 10 clips WER in a tight 4.76-9.16% band. hyp_words match ref_words
within ±5% on every clip — no chunk truncation or EOS dropouts.
RTF 0.025 = ~40× real-time, consistent with Parakeet's earlier
sugr_top3 run on 15 different clips (5.28% on lecture/webinar content
vs 7.25% here on faster-paced NPR news/politics). The +2 pp gap
reflects harder content (3-5 speakers, rapid turns, music beds), not a
model regression.

**Sugr-ASR-Bench corpus fix (2026-04-16):**
Initial run produced bimodal WER (5 clips at 5-9%, 5 clips at 48-93%).
Investigation showed 5 clips had audio paired with transcripts from
different NPR episodes at ingestion. The audio was also artificially
trimmed to 600/1200 s while references covered the full 17-45 min
episodes. Fix script
(`Sugr-ASR-Bench/scripts/fix_mismatched_audio.py`) re-downloads
correct MP3 from `reference_url` (found via NPR's play.podtrac.com
embed link), preserves full episode length, and updates
`duration_sec` in meta.json. Committed to Sugr-ASR-Bench repo as
`4c61d89`. Original audio preserved as `audio.opus.bak` (gitignored).

Artifacts: `results/sugr_new_20260416/parakeet.json`,
driver `tests/sugr_new_bench.py`.

### Moonshine on iOS Simulator vs Parakeet-on-Orin — 2026-04-16

Same 10 Sugr-ASR-Bench clips, measured on iOS Simulator (Mac mini CPU,
M-series) for Moonshine vs Jetson Orin GPU for Parakeet. iPad hardware
numbers still pending — see caveats below.

| Model | Platform | Avg WER | Min | Max | Avg RTF | Params | VRAM/RAM |
|---|---|---:|---:|---:|---:|---:|---:|
| **parakeet-tdt-1.1b** | Jetson Orin GPU | **7.25%** | 4.76% | 9.16% | **0.025** | 1.1B | 4.2 GB |
| moonshine-base-en | iOS Sim (Mac CPU) | 13.04% | 8.58% | 16.53% | 0.064 | 61M | 153 MB |
| moonshine-tiny-en | iOS Sim (Mac CPU) | 14.68% | 12.46% | 16.62% | 0.052 | 27M | 78 MB |

**Per-clip WER:**

| Clip | Dur (s) | Ref words | Parakeet | Moonshine-base | Moonshine-tiny |
|------|--------:|----------:|---------:|---------------:|---------------:|
| npr_rt8 | 1200 | 3,402 | **4.76%** | 8.58% | 12.46% |
| npr_ted7 | 1200 | 3,458 | 5.49% | 9.83% | 13.01% |
| npr_politics_ep16 | 1325 | 4,164 | 6.24% | 13.33% | 13.57% |
| npr_1a_ep20 | 1981 | 6,144 | 6.87% | 13.48% | 14.47% |
| npr_1a_b3 | 2695 | 8,711 | 7.16% | 12.34% | 15.88% |
| npr_pol_b13 | 1171 | 3,577 | 7.58% | 14.76% | 16.05% |
| npr_politics_ep18 | 1158 | 3,216 | 8.02% | 15.14% | 15.21% |
| npr_pol_b17 | 1063 | 3,375 | 8.47% | 15.26% | 15.23% |
| npr_fa_b5 | 1200 | 3,120 | 8.75% | 11.15% | 14.36% |
| npr_politics_ep22 | 1044 | 3,352 | 9.16% | 16.53% | 16.62% |

**Key findings:**

1. **Parakeet wins decisively on long-form — 7.25% vs Moonshine-base 13.04%.**
   ~6 pp WER gap, consistent across all 10 clips (Parakeet wins every clip).
   Parakeet's 1.1B-param Fast-Conformer transducer handles NPR-style
   multi-speaker dialogue, music beds, and rapid turn-taking materially
   better than Moonshine's 61M-param model.
2. **Moonshine-base beats Moonshine-tiny by 1.6 pp WER** (13.04% vs 14.68%) at
   25% higher compute cost. Base is the right pick for on-device quality.
3. **RTF comparison is not apples-to-apples.** Moonshine numbers are Mac CPU
   latency via iOS Simulator (not A15 Bionic neural engine). Real iPad mini 6
   should run faster per earlier LS-clean bench (moonshine-base 2.45% WER @
   0.053 RTF on A15 vs 2.55% @ 0.034 on simulator). But both are ~50-70× less
   compute-budget than Parakeet's GPU — the RTF gap closes when Parakeet runs
   on CPU.
4. **On-device mobile transcription is viable but not equivalent.** Moonshine
   on iPad is usable for quick meeting drafts (~13% WER = most content
   captured, named entities and fast speech garbled). For archival-quality
   transcripts, Parakeet on Orin remains the right choice. Mobile + Orin
   fallback is the deployment pattern this data supports.
5. **Caveats:** (a) iPad real-device RTF not yet measured — headless
   xcodebuild from SSH hangs on provisioning cert auth; needs interactive
   Xcode GUI on Mac mini. WER is hardware-independent so the 13.04% / 14.68%
   figures are final. (b) Apple SFSpeechRecognizer excluded per direction
   (gated via `ASR_SKIP_APPLE=1`).

**Setup:** MoonshineRunner patched to decode any audio format (including
m4a) via AVAudioFile → Float32 16 kHz mono → Moonshine Transcriber. Avoids
Moonshine's built-in WAV loader which rejected WAVE_FORMAT_EXTENSIBLE.
`BenchRunner.swift` takes `ASR_AUDIO_DIR` env var to select corpus,
`ASR_SKIP_APPLE=1` to skip Apple Speech runner.

Artifacts: `results/sugr_new_20260416/moonshine.json` (raw), `moonshine_wer.json`
(normalized + scored). iOS project: `testmac:/Users/seanslab/seanslab/asrbench/ios/`.

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
Use language detection → route to best model. With Parakeet-TDT + Moonshine Flavors in play:
- **EN (noisy / LS-other-like) → Parakeet-TDT-1.1B** (3.43% WER — new leader on LS-other, 4.17 GB)
- **EN (clean) → Cohere 2B or Parakeet-TDT-1.1B** (1.80 vs 1.82% — tied)
- **ZH → Qwen3-ASR-0.6B** (2.13% CER) or **1.7B** (1.57% CER)
- **JA → Moonshine-base-ja** (7.11% CER, 153 MB VRAM)
- Parakeet-1.1B + Qwen + Moonshine-ja total ≈ 6.0 GB VRAM. Leaves ~26 GB for the LLM.
- If you prefer a single multilingual EN model (still covering 14 langs for hot-swap),
  swap Parakeet for Cohere at a 0.02 pp EN cost.

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
  INT8 quantization saves ~6 GB but degrades WER — see INT8 section below
- **RTF 0.84** — barely real-time. A 1-hour meeting takes ~50 min to transcribe.
  Slowest model in the bench by a wide margin
- Loaded via `VibeVoiceAsrForConditionalGeneration` from transformers with
  `attn_implementation="eager"` (SDPA not supported for acoustic tokenizer on Jetson)
- **ZH/JA not yet benchmarked** — listed as n/t in the summary table
- Model downloaded via parallel curl from HuggingFace CDN to Mac, transferred to Jetson
  via SCP (HF blocked on Jetson's network)

### VibeVoice-ASR INT8 (bitsandbytes) — Benchmarked 2026-04-11

Same model as above with selective INT8 quantization via bitsandbytes: Qwen2.5-7B
LLM backbone quantized to INT8, audio-critical components (acoustic/semantic
tokenizer encoders, projections, lm_head) kept at BF16. First-ever INT8 WER
benchmark for this model.

| Metric | BF16 (Jetson Orin) | INT8 (DGX Spark1) |
|---|---:|---:|
| EN WER (LS-clean) | **3.11%** (386 samples) | **5.51%** (50 samples, preliminary) |
| Perfect transcriptions | 269/386 (70%) | 33/50 (66%) |
| RTF | 0.841 | 0.553 |
| VRAM | 16.4 GB | ~10-11 GB |
| Load time | 426s | 1700s (incl. download) |
| Errors | 0 | 0 |

**Note:** 386-sample apples-to-apples INT8 run is in progress on Spark1
(ETA ~3 hours from 2026-04-11 21:00 UTC+8). The 50-sample preliminary
result above will be updated with the full 386-sample number when available.

**Key findings:**
- **INT8 degrades WER by +2.4% absolute** (3.11% → 5.51%). The selective
  quantization preserves audio encoder precision but the LLM decoder's
  reduced precision affects text generation accuracy
- **~6 GB VRAM saved** (~10-11 GB vs 16.4 GB) — makes VibeVoice viable
  alongside a smaller LLM on a 32 GB device
- **RTF not directly comparable** — different hardware (GB10 vs Orin).
  GB10 is faster, so the 0.553 RTF doesn't reflect INT8's speed impact
- **bitsandbytes INT8 fails on Jetson Orin** (unified memory architecture
  causes meta tensor errors). Only works on GB10 with 128 GB unified
  memory where no CPU offloading is needed
- **GPTQ/AWQ INT4 also fail on Jetson** (gptqmodel can't detect Jetson's
  custom PyTorch). Quantized VibeVoice requires ≥128 GB unified memory
  or a discrete GPU

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

### Parakeet-TDT (NVIDIA NeMo) — Benchmarked 2026-04-15

NVIDIA Parakeet TDT (Token-and-Duration-Transducer), Fast-Conformer encoder +
transducer decoder, CC-BY-4.0 license. NeMo toolkit runtime (`nemo_toolkit[asr]`).
Ran with `greedy` decoding — `greedy_batch` uses CUDA graphs that OOM on Jetson
Orin during NeMo warmup (fix in `runners/parakeet_runner.py:42–45`).

| Model | Params | LS-clean WER% | LS-other WER% | RTF (clean) | VRAM | Languages |
|-------|--------|:---:|:---:|:---:|---:|---|
| parakeet-tdt-0.6b-v3 | 0.6B | 2.56 | 4.63 | 0.038 | 2.60 GB | 25 European (no ZH/JA) |
| parakeet-tdt-1.1b | 1.1B | **1.82** | **3.43** | 0.052 | 4.17 GB | English only |

Full LS-clean (2618 samples) + LS-other (2937 samples). 0 errors, 1 hallucination
per model on LS-other.

**Key findings:**
- **Parakeet-TDT-1.1B ties Cohere on LS-clean** (1.82% vs 1.80%) — a 2-error
  difference across 2618 utterances, well within measurement noise.
- **Parakeet-TDT-1.1B is the new LS-other champion at 3.43%** — 0.69 pp ahead
  of Qwen3-ASR-1.7B (4.12%) and the best LS-other result in the bench.
- **Fastest GPU model in the bench** — RTF 0.052 (~19× real-time), nearly 2×
  faster than Cohere's 0.096. 0.6B-v3 is even faster at 0.038 but trades ~0.7 pp
  of WER on clean.
- **Critical limitation: English-only.** The 1.1B variant has no multilingual
  support; the 0.6B-v3 covers 25 European languages but **not Chinese or
  Japanese**. Not a viable single-model replacement for Bosco's EN/ZH/JA use case.
- VRAM: 4.17 GB (1.1B) is on par with Cohere's 4.1 GB; 0.6B-v3 at 2.60 GB is
  lighter than any other top-10 model except the Moonshine Flavors family.
- Model files: `nvidia/parakeet-tdt-1.1b`, `nvidia/parakeet-tdt-0.6b-v3`
  (pre-cached as local `.nemo` on Orin via LAN rsync — HF blocked).

### sugr-asr-bench — Top-3 EN on long-form meeting audio — Benchmarked 2026-04-15

15 English clips from `sugr-asr-bench/english/` (NPR-style meeting/lecture audio,
24 kHz MP3, **33-42 min each**, 572 min total, 95,302 reference words after
normalization). Each clip transcribed with 30-second chunking (`transcribe_chunked`
in `tests/sugr_bench_top3.py`). Metric: WER after `EnglishTextNormalizer`.

| Model | Avg WER | Min | Max | Avg RTF | Notes |
|-------|:---:|:---:|:---:|:---:|---|
| **qwen3-asr-1.7b** | **4.64%** | 2.69% | 18.23% | 0.325 | New long-form leader; 3× real-time on Orin |
| parakeet-tdt-1.1b | 5.28% | 3.01% | 13.56% | **0.026** | Fastest (~38× real-time); 0.64 pp behind Qwen3 |
| cohere-transcribe-2b | 25.62% | 10.98% | 63.10% | 0.060 | Truncates chunks — hyp/ref words 38–93% |

**Per-clip WER (all three models):**

| Clip | Dur (s) | Ref words | Qwen3 WER | Parakeet WER | Cohere WER | C hyp/ref |
|------|--------:|----------:|----------:|-------------:|-----------:|----------:|
| en_1  | 2523 | 6381 |  2.93% |  3.35% | 12.02% | 89.5% |
| en_2  | 2353 | 6904 |  4.03% |  6.03% | 25.77% | 77.9% |
| en_3  | 2504 | 5436 |  3.26% |  3.97% | 11.02% | 92.6% |
| en_4  | 2346 | 7275 |  3.85% |  6.20% | 20.78% | 82.1% |
| en_5  | 2225 | 6247 |  4.47% |  5.76% | **58.54%** | **42.7%** |
| en_6  | 2470 | 8116 |  3.12% |  4.97% | 10.98% | 91.9% |
| en_7  | 2478 | 8362 |  **2.69%** |  3.01% | 31.82% | 69.7% |
| en_8  | 2455 | 6339 |  4.18% |  4.64% | **46.10%** | **55.7%** |
| en_9  | 2210 | 6709 |  4.89% |  5.41% | 18.69% | 84.9% |
| en_10 | 2082 | 5279 |  4.68% |  5.53% | 12.62% | 91.1% |
| en_11 | 2083 | 5147 |  3.30% |  3.75% | 16.34% | 85.7% |
| en_12 | 2035 | 2677 | **18.23%** | **13.56%** | 24.36% | 82.9% |
| en_13 | 2140 | 6870 |  3.42% |  4.40% | 17.15% | 85.4% |
| en_14 | 2010 | 7353 |  2.86% |  3.22% | 15.10% | 87.2% |
| en_15 | 2401 | 6907 |  3.65% |  5.36% | **63.10%** | **38.0%** |

Qwen3 and Parakeet both produce hyp_words within ±2% of ref_words — no
chunk truncation. Cohere's hyp/ref % column (right) shows its systematic
under-generation on long-form content.

**Key findings:**
- **Qwen3-ASR-1.7B is the long-form accuracy leader at 4.64% avg WER**,
  beating Parakeet-TDT-1.1B (5.28%) by 0.64 pp and Cohere-Transcribe-2B
  (25.62%) by ~5.5×. Qwen3 wins 14 of 15 clips against Parakeet — only
  en_12 (the shortest, noisiest clip at 2,677 ref words) goes to Parakeet
  (13.56% vs Qwen3 18.23%).
- **Parakeet is ~12× faster.** RTF 0.026 vs 0.325 — Parakeet runs at
  ~38× real-time, Qwen3 at ~3× real-time on the same Jetson Orin. For
  most meeting-transcription workloads (not batch-archival), Parakeet's
  speed advantage outweighs the 0.64 pp WER gap. For overnight batch
  archival where accuracy dominates, Qwen3 wins.
- **Both Qwen3 and Parakeet produce faithful word counts** (hyp ±2% of
  ref on every clip). Cohere's hypothesis contains only 38–93% of the
  reference word count per clip, with worst offenders (en_5, en_8,
  en_15) missing 44–62% of content. The Cohere decoder emits early EOS
  on some 30-second chunks (silence / music / speaker-transition
  boundaries), losing whole segments.
- **This inverts the LS-clean ranking.** On LibriSpeech clean short
  utterances, Cohere (1.80%) > Qwen3 (1.87%) > Parakeet (1.82%) —
  statistically tied. On long-form chunked content: Qwen3 (4.64%) >
  Parakeet (5.28%) >> Cohere (25.62%). Short-utterance benchmarks do
  not predict long-form performance.
- **Cohere runner bug found and patched during this benchmark.** Initial
  run produced WER=100% on all 15 clips because `processor.decode()` on
  a 2D `[batch, seq]` tensor returns a Python list, and `str(list)` gives
  `'[" text..."]'` — which `whisper_normalizer` collapses to empty string.
  Fixed in `runners/cohere_runner.py` via `batch_decode()` with `[0]`
  indexing. Result above (25.62%) is post-fix. **Bug did not affect the
  earlier LibriSpeech Cohere 1.80% result** — different decode path.

### Model selection (updated)

- **Bosco English-only long-form (Orin, streaming):** **Parakeet-TDT-1.1B**
  — 5.28% WER at 0.026 RTF is the best speed/accuracy ratio, and real-time
  budget on Orin matters.
- **Bosco English-only long-form (Spark 2 / batch):** **Qwen3-ASR-1.7B** —
  0.64 pp accuracy gain is worth the 12× slowdown when runtime budget is
  overnight, not interactive.
- **Bosco multilingual (EN/ZH/JA):** **Cohere Transcribe 2B** remains the
  only viable top-tier option that covers all three languages. Its
  long-form truncation is a known failure mode to mitigate with larger
  chunks or a different inference strategy.

Artifacts: `results/sugr_top3_20260415/{cohere,parakeet,qwen3}.json`
(per-clip raw hypotheses + WER + RTF). Driver: `tests/sugr_bench_top3.py`.

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
10. **VibeVoice-ASR INT8/INT4 not achievable on Jetson**: bitsandbytes INT8 fails due to unified memory architecture (meta tensor errors, NVML assertions). GPTQ/AWQ fail because gptqmodel can't detect Jetson's custom PyTorch. Quantized VibeVoice requires a discrete GPU (A100/H100/RTX).

---

## Speaker Diarization Options

Speaker diarization identifies *who* spoke *when*. Most ASR models above produce
plain transcripts without speaker attribution. Only VibeVoice-ASR has built-in
diarization. For all other ASR models, a separate diarization pipeline is needed.

### iOS (iPad / iPhone)

| Library | Models | Speed | DER | RAM | License | Notes |
|---------|--------|------:|----:|----:|---------|-------|
| **[FluidAudio](https://github.com/FluidInference/FluidAudio)** | pyannote 3.0 segmentation + WeSpeaker embedding, CoreML | 60× RT (M1) | ~22% | ~50 MB | MIT/Apache 2.0 | Runs on Apple Neural Engine; Swift Package (SPM/CocoaPods) |
| **[SpeakerKit](https://www.argmaxinc.com/blog/speakerkit)** (WhisperKit) | pyannote-based, CoreML | ~1s for 4 min (iPhone) | — | — | — | Integrated with WhisperKit ASR; Interspeech 2025 paper |
| **[speech-swift](https://github.com/soniqo/speech-swift)** | pyannote via MLX | — | — | — | — | MLX-native, Apple Silicon only |

**Recommendation for Memio iPad:** FluidAudio — CoreML on ANE, ~50 MB, open source,
composable with any ASR (Moonshine). Combined stack: Moonshine (~585 MB) + FluidAudio
(~50 MB) ≈ 635 MB total, fits iPad mini 6's 4 GB.

### Android

| Library | Models | Speed | DER | License | Notes |
|---------|--------|------:|----:|---------|-------|
| **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** | pyannote 3.0 + 3D-Speaker or NeMo TitaNet-Small, ONNX | ~10-30× RT (est.) | ~20-25% | Apache 2.0 | Pre-built APKs for arm64/armeabi/x86; Java/Kotlin bindings; 12 languages |
| **[pyannote-onnx-extended](https://github.com/samson6460/pyannote-onnx-extended)** | pyannote 3.1, ONNX export | — | ~10% | MIT | Python-focused; needs integration work for Android |

**Recommendation for Memio Android:** sherpa-onnx — ONNX Runtime, same pyannote base
models as FluidAudio, pre-built APKs, Apache 2.0. Kotlin integration.

### Server / Jetson (Python)

| Library | Models | Speed | DER | Notes |
|---------|--------|------:|----:|-------|
| **[pyannote 3.1](https://github.com/pyannote/pyannote-audio)** | pyannote/segmentation-3.0 + WeSpeaker | 40× RT (V100) | ~10% | Gold standard; Python + PyTorch |
| **VibeVoice-ASR (built-in)** | Integrated in ASR model | 1.2× RT (Orin) | ~4.3% DER | No separate pipeline needed; 16.4 GB VRAM |

### Cross-platform comparison

| | iOS | Android | Server |
|---|---|---|---|
| **Best option** | FluidAudio (CoreML) | sherpa-onnx (ONNX) | pyannote 3.1 (PyTorch) |
| **Underlying models** | pyannote 3.0 + WeSpeaker | pyannote 3.0 + 3D-Speaker | pyannote 3.0 + WeSpeaker |
| **Runtime** | Apple Neural Engine | ONNX Runtime (CPU/NNAPI) | CUDA GPU |
| **Package** | Swift SPM | Gradle AAR/APK | pip |
| **DER** | ~22% | ~20-25% | ~10% |
| **Speed** | 60× RT | 10-30× RT | 40× RT |
| **License** | MIT/Apache 2.0 | Apache 2.0 | MIT (research license for models) |

All three platforms use the same pyannote-family models underneath. The accuracy
and speed differences come from the runtime (CoreML > ONNX > PyTorch on
consumer hardware) and model precision (CoreML fp32 vs ONNX quantized).

### Combined ASR + diarization stack for Memio

| Platform | ASR | Diarization | Total RAM | Stack |
|----------|-----|-------------|----------:|-------|
| **iPad** | Moonshine-base (moonshine-swift) | FluidAudio (CoreML) | ~635 MB | Swift, ANE |
| **Android** | Moonshine-base (ONNX) | sherpa-onnx (pyannote+3D-Speaker) | ~800 MB | Kotlin, ONNX |
| **Jetson** | Cohere/Moonshine (transformers) | pyannote 3.1 (PyTorch) | ~1-5 GB | Python, CUDA |
