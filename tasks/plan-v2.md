# ASR Bench v2 — Comprehensive Benchmark Plan

**Date:** 2026-03-19
**Author:** Sean + Claude Opus 4.6
**Status:** Revised after expert review
**Reviewer:** Senior ML Engineer (speech recognition evaluation specialist)

---

## 1. Current State Assessment

### What Exists
- Benchmark framework (`bench.py`) with runner registry, profiler (RTF/VRAM/load time), WER/CER metrics
- 2 working model runners: `whisper-large-v3-turbo` (GPU, RTF=0.19-0.42) and `faster-whisper-large-v3` (CPU only, RTF=1.9-2.1)
- 24 custom audio clips: 15 short (30s × 3 langs) + 9 long (15/30/60 min × 3 langs)
- Results from 1 benchmark run (2026-03-15)

### What's Broken
- **No ground truth transcriptions** → all WER/CER = N/A, making accuracy comparison impossible
- **~6 bad audio clips** (silence, truncated, wrong content)
- **Only 2 of ~15 planned models running** — most blocked by Python 3.8 (JetPack 5) or missing integration
- **Custom test data only** — results not comparable to any published benchmarks

### Hardware (CONFIRMED)

| Spec | Value |
|------|-------|
| **Device** | NVIDIA Jetson AGX Orin 32GB |
| **JetPack** | 6.2 (L4T R36.4.4) |
| **Ubuntu** | 22.04 |
| **Python** | 3.10.12 |
| **CUDA** | 12.6 |
| **Memory** | 32 GB shared (CPU+GPU), ~23GB usable after OS/CUDA overhead |
| **Tailscale IP** | `100.123.48.6` |
| **LAN IP** | `192.168.1.153` |
| **SSH** | `ssh x@100.123.48.6` (password: 12345678) |

**Why 32GB Orin:**
- Python 3.10 unblocks all modern ASR packages (Qwen3-ASR, Moonshine, FunASR)
- 32GB shared memory fits all models up to 1.7B comfortably (~23GB usable)
- JetPack 6 / CUDA 12.6 is the current-gen platform with long support life
- All VRAM estimates in this plan are validated against 23GB usable budget

---

## 2. Text Normalization (CRITICAL)

> **Expert review flagged this as the #1 methodological risk.** WER/CER is extremely sensitive to normalization. A mismatch between model output format and reference format can inflate error rates by 5-30%.

### The Problem

- Whisper-family models output cased, punctuated text ("Hello, how are you?")
- LibriSpeech references are UPPERCASE, no punctuation ("HELLO HOW ARE YOU")
- AISHELL-1 references have no spaces between characters
- CTC-based models (Parakeet, MMS) may output all-lowercase without punctuation
- Each model has different output conventions

### Normalization Pipeline

All model outputs AND reference texts pass through the **same** normalizer before WER/CER computation:

**English:**
```python
from whisper.normalizers import EnglishTextNormalizer
normalizer = EnglishTextNormalizer()
# Handles: lowercase, expand contractions, strip punctuation, normalize whitespace
```

**Chinese (Mandarin):**
```python
def normalize_zh(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)       # Unicode normalization
    text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf]', '', text)  # Keep CJK only
    return text
# CER computed character-by-character, no spaces
```

**Japanese:**
```python
def normalize_ja(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)       # Unicode normalization
    text = re.sub(r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf]', '', text)
    return text
# CER computed character-by-character (kanji + kana)
```

### Validation Step
Before running full benchmarks, validate normalization by running whisper-large-v3-turbo on 10 LibriSpeech utterances and comparing WER against published results (~3-4%). If off by >2%, debug normalization before proceeding.

---

## 3. Test Dataset Strategy

### Tier 1 — Required Standard Benchmarks

| Dataset | Lang | Test Duration | Utterances | License | Audio Type | Why |
|---------|------|--------------|------------|---------|------------|-----|
| **LibriSpeech test-clean** | EN | 5.4h | 2,620 | CC BY 4.0 | Read speech | Gold standard EN; every paper reports this |
| **LibriSpeech test-other** | EN | 5.1h | 2,939 | CC BY 4.0 | Read speech (harder) | Tests robustness on harder acoustics |
| **AISHELL-1 test** | ZH | ~5h | 7,176 | Apache 2.0 | Read mobile speech | Gold standard ZH; every paper reports this |
| **ReazonSpeech test** | JA | ~8-16h | 5,263 | Apache 2.0 | Broadcast TV | Primary JA benchmark (replaced FLEURS-ja) |

**Total: ~24-32h audio.** At average RTF 0.3, this is ~7-10h wall-clock per model.

> **Change from draft:** FLEURS-ja demoted from Tier 1 to Tier 2. Expert review: "400 utterances is too small for reliable benchmarking. A single bad utterance can swing results by 0.25%." ReazonSpeech promoted to primary JA benchmark.

### Tier 1.5 — Strongly Recommended (domain-relevant)

| Dataset | Lang | Test Duration | License | Audio Type | Why |
|---------|------|--------------|---------|------------|-----|
| **WenetSpeech TEST_MEETING** | ZH | 15h | CC BY 4.0 | Meeting (far-field) | Directly relevant to Bosco meeting transcription |

> **Promoted from Tier 2** per expert review: "Your target use case is meeting audio. AISHELL-1 is read speech."

### Tier 2 — Extended (optional, adds depth)

| Dataset | Lang | Test Duration | License | Audio Type | Why |
|---------|------|--------------|---------|------------|-----|
| **FLEURS test (ja)** | JA | ~1-2h | CC BY 4.0 | Read scripted | Small but covers different domain than ReazonSpeech |
| **CommonVoice 17 test (en/zh/ja)** | Multi | ~2-5h/lang | CC-0 | Crowd-sourced | Noisy, diverse speakers |

### Tier 3 — Custom Meeting Audio (qualitative only)

Keep existing 24 clips for qualitative evaluation (listen to outputs, judge naturalness). Fix the 6 bad clips. Create ground truth for short clips only via Whisper API + manual correction. These are **not** used for WER/CER scoring due to small sample size and unverified ground truth.

### Dataset Storage & Download

All datasets download to Mac first, then rsync to Jetson (HuggingFace is unreachable from Jetson per lessons.md).

```
data/
├── standard/
│   ├── librispeech/
│   │   ├── test-clean/        # 2,620 utterances, 5.4h (~346MB)
│   │   └── test-other/        # 2,939 utterances, 5.1h (~328MB)
│   ├── aishell1/
│   │   └── test/              # 7,176 utterances, ~5h (~1.2GB)
│   ├── reazonspeech/
│   │   └── test/              # 5,263 utterances, ~8-16h (~601MB)
│   └── wenet_meeting/
│       └── test/              # meeting audio, 15h (TBD size)
├── en/                        # existing custom audio
├── zh/
└── ja/
```

Estimated total download: ~3-5GB for Tier 1 + 1.5 datasets.

---

## 4. Model Integration Plan

### Final Roster: 11 models

> **Changes from draft per expert review:**
> - **Added:** faster-whisper-large-v3 GPU INT8 (expert: "arguably more important than half the Tier 2/3 models")
> - **Dropped:** Meta MMS-1b-all (expert: "CC-BY-NC, mediocre accuracy, main value proposition irrelevant to 3-lang bench")
> - **Dropped:** faster-whisper CPU from ranked scoring (keep for reference only, RTF > 1 = not viable)
> - **Moved:** Qwen3-ASR-1.7B to "32GB only" track (expert: "will almost certainly fail on 16GB after OS/CUDA overhead")

#### Already Running (2)

| # | Model | Params | VRAM | EN | ZH | JA | Notes |
|---|-------|--------|------|----|----|-----|-------|
| 1 | whisper-large-v3-turbo | 809M | 3.4GB | Y | Y | Y | Current champion |
| 2 | faster-whisper-large-v3 (CPU) | 1.5B | 8MB | Y | Y | Y | Reference only; RTF > 1, not viable for edge |

#### Tier 1 — Low Effort, High Value (4 models)

| # | Model | Params | Est. VRAM | EN | ZH | JA | Integration Approach | Effort |
|---|-------|--------|-----------|----|----|-----|---------------------|--------|
| 3 | **faster-whisper-large-v3 GPU INT8** | 1.5B | ~1.5GB | Y | Y | Y | Build CTranslate2 with CUDA on JetPack 6; `compute_type=int8` | 4h |
| 4 | **Kotoba-Whisper v2.2** | 756M | ~2GB | — | — | Y | Same HF pipeline as Whisper; change model ID only | 1h |
| 5 | **Kotoba-Whisper bilingual v1** | 756M | ~2GB | Y | — | Y | Same as above | 0.5h |
| 6 | **Moonshine v2 Medium** | 245M | <1GB | Y | Y* | Y* | `pip install moonshine-voice` or sherpa-onnx | 2h |

*Moonshine v2 has separate per-language models, not a single multilingual model.

#### Tier 2 — Medium Effort, Strong Models (3 models)

| # | Model | Params | Est. VRAM | EN | ZH | JA | Integration Approach | Effort |
|---|-------|--------|-----------|----|----|-----|---------------------|--------|
| 7 | **SenseVoice-Small** | 234M | ~1-2GB | Y | Y | Y | sherpa-onnx ONNX runtime (bypasses FunASR) | 3h |
| 8 | **Qwen3-ASR-0.6B** | 0.6B | ~3-4GB | Y | Y | Y | `pip install qwen-asr`; Python 3.10 on JetPack 6 | 3h |
| 9 | **FireRedASR-AED** | 1.1B | ~3-5GB | Y | Y | — | PyTorch + HuggingFace; SOTA Mandarin, no JA | 4h |

#### Tier 3 — Nice to Have (2 models)

| # | Model | Params | Est. VRAM | EN | ZH | JA | Integration Approach | Effort |
|---|-------|--------|-----------|----|----|-----|---------------------|--------|
| 10 | **Canary-1B-Flash** | 883M | ~2-4GB | Y | — | — | NeMo; EN only, NVIDIA-native Jetson support | 4h |
| 11 | **Paraformer-Large** | 600M | ~1-2GB | Y | Y | — | sherpa-onnx ONNX; no JA | 2h (reuse sherpa setup) |

#### Tier 2b — Larger Models (viable on 32GB Orin)

| # | Model | Params | Est. VRAM | EN | ZH | JA | Integration Approach | Effort |
|---|-------|--------|-----------|----|----|-----|---------------------|--------|
| 12 | **Qwen3-ASR-1.7B** | 1.7B | ~7-8GB | Y | Y | Y | Reuse 0.6B runner; fits comfortably on 32GB | 1h |

#### Excluded

| Model | Reason |
|-------|--------|
| Whisper large-v3 (FP16) | OOM on 16GB (confirmed) |
| Meta MMS-1b-all | CC-BY-NC license; mediocre on EN/ZH/JA vs. specialized models |
| SenseVoice-Large | Weights never publicly released |
| Samba-ASR | No weights, no code released |
| Fun-ASR-Nano | No ONNX export path |
| Canary-Qwen-2.5B | EN only, 2.5B too heavy |
| Voxtral Mini 4B | 4B borderline, needs quantization |

---

## 5. Benchmark Methodology

### Warm-up Protocol

> **Expert review:** "First inference on Jetson is always slower due to CUDA JIT compilation, memory allocation, and TensorRT engine building."

Each model run begins with 2 warm-up utterances (discarded from timing). This ensures:
- CUDA context is initialized
- JIT compilation is complete
- Memory allocators are warmed

### Hallucination Detection

> **Expert review:** "Whisper-family models are notorious for hallucinating on silence or low-SNR audio."

Flag utterances as hallucinated if:
- Output length > 3× expected (based on audio duration × chars-per-second estimate)
- Any 5-gram repeats ≥3 times in the output
- Output is identical across 3+ consecutive utterances

Hallucinated utterances are excluded from WER/CER calculation but reported separately as a "hallucination rate" metric.

### Long-form Audio Chunking

> **Expert review:** "Different chunking approaches produce materially different WER. This needs to be standardized."

For custom long-form clips (15/30/60 min), all models use the **same** chunking strategy:
- **VAD-based segmentation** using Silero VAD (open-source, lightweight)
- Minimum segment: 1s, maximum segment: 30s
- 0.5s padding on each side of detected speech segments
- Models that have built-in long-form handling (Whisper sequential decode) may use their native approach, but this is documented per model

Standard benchmark datasets (LibriSpeech, AISHELL-1, ReazonSpeech) are pre-segmented into utterances — no chunking needed.

### Power Mode & System State

> **Expert review:** "Power mode dramatically affects speed and thermal throttling. Lock it and document it."

All benchmarks run with:
- **Power mode:** MAXN (maximum performance)
- **Fan mode:** Maximum (prevent thermal throttling)
- **Background services:** Disabled (per devlog cleanup)
- **Verification:** Record `tegrastats` output during runs for VRAM/power/temp monitoring

```bash
# Lock power mode before benchmarking
sudo nvpmodel -m 0           # MAXN mode
sudo jetson_clocks            # Lock clocks at max
```

### Reproducibility Requirements

> **Expert review:** "Pin all package versions. Record exact model revision hashes."

- `requirements.txt` with pinned versions for all dependencies
- Model revision hash recorded per run (HuggingFace commit SHA or ONNX model checksum)
- Git commit of `asrbench` code recorded in results JSON
- `tegrastats` log saved alongside results
- All results include: JetPack version, CUDA version, Python version, torch version

---

## 6. Scoring

### Per-Language Results (primary reporting format)

Report raw metrics per model per dataset. No composite score needed for per-language analysis:

```
                      LibriSpeech    LibriSpeech    AISHELL-1    ReazonSpeech
Model                 test-clean↓    test-other↓    test↓        test↓         Avg RTF↓  VRAM↑
                      (EN WER%)      (EN WER%)      (ZH CER%)    (JA CER%)
─────────────────────────────────────────────────────────────────────────────────────────────
whisper-turbo         x.xx           x.xx           x.xx         x.xx          0.xxx     x.xGB
qwen3-asr-0.6b       x.xx           x.xx           x.xx         x.xx          0.xxx     x.xGB
sensevoice-small      x.xx           x.xx           x.xx         x.xx          0.xxx     x.xGB
...
```

### Composite Score (secondary, for quick ranking)

```
score = 0.60 × accuracy + 0.25 × speed + 0.15 × memory_efficiency
```

> **Changes from draft per expert review:**
> - **Coverage removed** from composite score. "Unfairly penalizes specialized models. Report as metadata."
> - **Accuracy weight increased** to 0.60 (from 0.50) to compensate for removed coverage
> - **Speed scoring** uses log-scale instead of linear clamp
> - **Memory scoring** uses step function instead of linear

Components:

**accuracy** = average of per-language scores, only over languages the model supports:
```
accuracy = mean(1 - WER_en, 1 - CER_zh, 1 - CER_ja)  # skip unsupported languages
```
> Note: WER and CER are not directly comparable scales. Per-language ranks are more meaningful than cross-language averages. The composite score is a rough guide, not a precise measure.

**speed** (log-scale):
```
speed = clamp(1 + log10(1/RTF) / 2, 0, 1)
# RTF=0.01 → 1.0, RTF=0.1 → 0.75, RTF=0.3 → 0.63, RTF=1.0 → 0.5, RTF=3.0 → 0.26
```

**memory_efficiency** (step function, based on 23GB usable on Orin 32GB):
```
if VRAM <= 6GB:   memory = 1.0    # comfortable, leaves room for Bosco washer/writer
elif VRAM <= 12GB: memory = 0.6   # usable but limits co-located services
elif VRAM <= 18GB: memory = 0.3   # marginal, ASR-only deployment
else:             memory = 0.0    # not viable on 32GB device
```

### Statistical Significance

> **Expert review:** "Differences of <0.3% WER on LibriSpeech test-clean are likely not significant."

Report 95% confidence intervals for WER/CER using bootstrap resampling (1000 iterations). When comparing two models, note if the difference exceeds the confidence interval. Do not claim Model A is "better" than Model B if the difference is within noise.

---

## 7. Execution Plan (14 working days)

> **Revised from 8 days per expert review:** "The timeline is roughly 50% too short." Key adjustments: dataset downloads take longer than expected, budget 1 day per new model, sequential compute is ~66h for all models × all datasets.

### Phase 0: Hardware & Environment Setup (Day 1-2)

- [ ] SSH into 32GB AGX Orin, verify: `python3 --version` (expect 3.10), `nvcc --version`, free disk/RAM
- [ ] Set power mode MAXN, lock clocks, disable unnecessary services
- [ ] Create Python venv, install base deps (torch, jiwer, librosa, soundfile)
- [ ] Verify existing whisper-turbo runner works on 32GB device
- [ ] Start dataset downloads in background (LibriSpeech, AISHELL-1, ReazonSpeech — may take 1-2 days)

### Phase 1: Normalization & Dataset Infrastructure (Day 2-4)

- [ ] Implement text normalization pipeline (EN: WhisperNormalizer, ZH: CJK-only, JA: kana+kanji)
- [ ] Validate normalization: run whisper-turbo on 10 LibriSpeech utterances, compare WER to published (~3-4%)
- [ ] Write dataset loaders for LibriSpeech, AISHELL-1, ReazonSpeech, WenetSpeech format
- [ ] Add `--dataset` flag to `bench.py`
- [ ] Add warm-up protocol (2 utterances discarded)
- [ ] Add hallucination detection to profiler
- [ ] Add `tegrastats` logging integration
- [ ] Run whisper-turbo on all Tier 1 datasets → baseline WER/CER numbers

### Phase 2: Tier 1 Model Integration (Day 4-6)

- [ ] **faster-whisper GPU INT8**: Build CTranslate2 with CUDA on JetPack 6 (~4h); if build fails, skip
- [ ] **Kotoba-Whisper v2.2** runner (change model ID in Whisper pipeline)
- [ ] **Kotoba-Whisper bilingual v1** runner
- [ ] **Moonshine v2 Medium** runner (via moonshine-voice)
- [ ] Run all Tier 1 models on LibriSpeech test-clean → first EN WER comparison
- [ ] Run JA models on ReazonSpeech → first JA CER comparison

### Phase 3: Tier 2 Model Integration (Day 6-9)

- [ ] **SenseVoice-Small** via sherpa-onnx (build sherpa-onnx with GPU on Jetson first)
- [ ] **Qwen3-ASR-0.6B** via `pip install qwen-asr`
- [ ] **FireRedASR-AED** via PyTorch
- [ ] Run Tier 2 models on all datasets
- [ ] **Qwen3-ASR-1.7B** on 32GB device (reuse 0.6B runner, verify memory fit)

### Phase 4: Tier 3 Models + Full Benchmark (Day 9-11)

- [ ] **Canary-1B-Flash** via NeMo (if NeMo installs cleanly on JetPack 6)
- [ ] **Paraformer-Large** via sherpa-onnx (reuse sherpa setup from SenseVoice)
- [ ] Full benchmark run: all models × all Tier 1 datasets
- [ ] Run WenetSpeech TEST_MEETING for ZH meeting domain comparison
- [ ] Fix bad custom audio clips; run custom clips for qualitative eval

### Phase 5: Analysis & Report (Day 11-14)

- [ ] Compute all WER/CER with confidence intervals (bootstrap)
- [ ] Generate ranked comparison tables (per-language + composite)
- [ ] Check for hallucination rates across models
- [ ] Write summary report with Bosco deployment recommendations
- [ ] Pin all versions, record model hashes, save tegrastats logs
- [ ] Push everything to GitHub

### Compute Budget

| Phase | Models | Datasets | Audio Hours | Est. Wall-Clock (RTF 0.3 avg) |
|-------|--------|----------|-------------|-------------------------------|
| Phase 1 | 1 (whisper-turbo) | 4 datasets | ~30h | ~9h |
| Phase 2 | 4 new | LibriSpeech-clean + ReazonSpeech | ~22h × 4 | ~26h |
| Phase 3 | 3-4 new | All Tier 1 | ~30h × 4 | ~36h |
| Phase 4 | 2-3 new + full re-run | All | ~30h × 11 | ~100h |

**Total compute: ~170h.** At 16h/day continuous running (overnight), this is ~11 days of Jetson time. Phases overlap with integration work — Jetson runs benchmarks overnight while writing new runners during the day.

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Text normalization mismatch inflates WER** | High | All results invalid | Validate against published numbers on Day 2-3 before proceeding |
| **CTranslate2 GPU build fails on JetPack 6** | Medium | Lose faster-whisper-INT8 | Budget 4h; if fails, skip — not a must-have |
| **Qwen3-ASR has undocumented issues on aarch64** | Medium | Lose best multilingual model | Test early (Day 6); fallback to antirez/qwen-asr C impl |
| **sherpa-onnx GPU build fails on Jetson** | Medium | SenseVoice/Paraformer CPU only | Pre-test build Day 6; CPU fallback acceptable for accuracy eval |
| **ReazonSpeech download is very large** | Low | JA benchmark delayed | Start download Day 1; can use FLEURS-ja as interim |
| **Model OOM on 32GB during long benchmarks** | Low | Crash mid-run | Run each model in subprocess; auto-skip on OOM |
| **NeMo incompatible with JetPack 6** | Medium | Lose Canary model | Canary is EN-only and Tier 3; acceptable loss |
| **Thermal throttling during long runs** | Medium | Inconsistent RTF numbers | Lock MAXN + max fan; monitor via tegrastats; re-run outliers |

---

## 9. Decisions Log (from expert review)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Use 32GB AGX Orin (JetPack 6) | Python 3.8 on JetPack 5 blocks most models |
| 2 | ReazonSpeech replaces FLEURS-ja as primary JA benchmark | FLEURS-ja too small (400 utterances) for reliable stats |
| 3 | WenetSpeech TEST_MEETING promoted to Tier 1.5 | Meeting audio directly relevant to Bosco use case |
| 4 | Added faster-whisper GPU INT8 | Most common production Whisper deployment; was missing |
| 5 | Dropped Meta MMS-1b-all | CC-BY-NC license; mediocre accuracy on target languages |
| 6 | Qwen3-ASR-1.7B promoted to Tier 2b | ~7GB VRAM fits on 32GB Orin (23GB usable) |
| 7 | Coverage removed from composite score | Unfairly penalizes specialized models (e.g., Kotoba JA-only) |
| 8 | Speed scoring → log-scale | Linear clamp gave no differentiation in RTF 0.1-0.5 range |
| 9 | Memory scoring → step function | Binary constraint (fits or doesn't); linear was misleading |
| 10 | SeMaScore dropped | Introduces second model's bias; WER/CER on standard benchmarks sufficient |
| 11 | Timeline extended to 14 days | Expert: "8 days is roughly 50% too short" |
| 12 | Diarization deferred to v3 | Out of scope for ASR accuracy benchmark; separate project |

---

## 10. Out of Scope (Deferred to v3)

- Speaker diarization benchmarking (DER, cpWER)
- Streaming/latency benchmarking (first-token latency, chunk-level RTF)
- Power consumption measurement (watts per utterance)
- Quantization experiments (INT4, FP8 variants)
- Long-form drift analysis (WER delta between first/last 5 minutes)
- Multi-GPU or TensorRT-optimized inference

---

## Appendix A: Model Reference

| Model | HuggingFace / Source | Runtime | Streaming? |
|-------|---------------------|---------|------------|
| whisper-large-v3-turbo | `openai/whisper-large-v3-turbo` | openai-whisper | No (sequential decode) |
| faster-whisper-large-v3 (CPU) | `Systran/faster-whisper-large-v3` | CTranslate2 CPU | No |
| faster-whisper-large-v3 (GPU INT8) | `Systran/faster-whisper-large-v3` | CTranslate2 GPU | No |
| kotoba-whisper-v2.2 | `kotoba-tech/kotoba-whisper-v2.2` | transformers pipeline | No |
| kotoba-whisper-bilingual | `kotoba-tech/kotoba-whisper-bilingual-v1.0` | transformers pipeline | No |
| moonshine-v2-medium | via `moonshine-voice` package | ONNX Runtime | Yes |
| sensevoice-small | `FunAudioLLM/SenseVoiceSmall` | sherpa-onnx (ONNX int8) | Yes |
| qwen3-asr-0.6b | `Qwen/Qwen3-ASR-0.6B` | qwen-asr / transformers | Yes (vLLM only) |
| qwen3-asr-1.7b | `Qwen/Qwen3-ASR-1.7B` | qwen-asr / transformers | Yes (vLLM only) |
| firered-asr-aed | `FireRedTeam/FireRedASR-AED-L` | PyTorch + peft | No |
| canary-1b-flash | `nvidia/canary-1b-flash` | NeMo | Yes |
| paraformer-large | via sherpa-onnx | sherpa-onnx (ONNX) | Yes |

## Appendix B: Dataset Download Commands

```bash
# LibriSpeech test sets (~674MB total)
wget https://www.openslr.org/resources/12/test-clean.tar.gz
wget https://www.openslr.org/resources/12/test-other.tar.gz

# AISHELL-1 full dataset (~15GB, test is ~1.2GB subset)
wget https://www.openslr.org/resources/33/data_aishell.tgz

# ReazonSpeech test (via HuggingFace, ~601MB)
python -c "
from datasets import load_dataset
ds = load_dataset('reazon-research/reazonspeech', 'all', split='test')
ds.save_to_disk('data/standard/reazonspeech/test')
"

# WenetSpeech TEST_MEETING (requires HuggingFace auth)
# See: https://huggingface.co/datasets/wenet-e2e/WenetSpeech
```

## Appendix C: Version Pinning Template

```
# requirements-bench.txt — pin for reproducibility
torch==2.5.0          # NVIDIA JetPack 6 wheel
torchaudio==2.5.0
jiwer==3.1.0
librosa==0.10.2
soundfile==0.12.1
openai-whisper==20241126
faster-whisper==1.1.0
moonshine-voice==0.0.51
qwen-asr==0.3.0
sherpa-onnx==1.10.38
transformers==4.48.0
```
