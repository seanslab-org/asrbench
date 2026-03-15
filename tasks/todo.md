# ASR Bench — Task List

## Project Goal
Benchmark top open-source ASR models on NVIDIA Jetson (16GB) for Japanese, Chinese, and English.

## Tasks

### Phase 1: Setup & Test Data
- [x] Initialize git repo and push to GitHub
- [x] Prepare SHORT test audio (5 clips × 30s per language)
  - [x] EN: earnings call excerpts, board meetings, conference calls
  - [x] ZH: 财报电话会 excerpts, 国新办发布会, business calls
  - [x] JA: 決算説明会 excerpts, 記者会見, shareholder meetings
  - [x] Include noisy samples (conference call / phone quality)
- [x] Prepare LONG test audio (15min / 30min / 60min per language)
  - [x] EN: full earnings call (15m), board meeting (30m), hearing (60m)
  - [x] ZH: full press conference (15m/30m), meeting (60m)
  - [x] JA: full 決算説明会, 記者会見, 株主総会
- [ ] Prepare ground truth transcription files — **see Ground Truth Plan below**
- [x] Set up Jetson environment (CUDA, Python 3.8, venv)
- [x] Clean Jetson 16GB system for dedicated ASR benchmarking (32GB→79GB free)

### Phase 2: Model Integration
- [x] Implement model runner interface (load, transcribe, measure)
- [x] Integrate each model (one runner per model):
  - [x] Whisper large-v3 — **OOM on 16GB, not viable**
  - [x] Whisper large-v3-turbo — **working, GPU, RTF=0.33**
  - [x] Faster-Whisper large-v3 — **working, CPU only (no aarch64 CUDA CTranslate2), RTF=1.99**
  - [ ] Moonshine Base / Small / v2 — **blocked: requires Python 3.9+ (Keras 3.6)**
  - [ ] SenseVoice-Large (FunASR) — **blocked: setuptools incompatibility on Python 3.8**
  - [ ] Paraformer-Large (FunASR) — **blocked: same as SenseVoice**
  - [ ] Qwen3-ASR-1.7B
  - [ ] Qwen3-ASR-0.6B
  - [ ] FireRedASR-AED (1.1B)
  - [ ] NVIDIA Canary-Qwen-2.5B
  - [ ] NVIDIA Canary-1B-v2
  - [ ] NVIDIA Parakeet-CTC-1.1B (EN only)
  - [ ] Kotoba-Whisper v2 (JA focused)
  - [ ] Samba-ASR
  - [ ] Meta Omnilingual ASR (1B or smaller variant)
- [x] Verify working models load and run on Jetson 16GB

### Phase 3: ASR Benchmarking
- [x] Run working models against short clips (15 clips × 2 models)
- [x] Run working models against long clips (9 clips × 2 models)
- [x] Collect metrics: RTF, VRAM peak, load time
- [ ] Compute WER/CER accuracy — **blocked on ground truth**
- [ ] Run additional models as they become available
- [ ] Generate composite score ranking

### Phase 4: Speaker Diarization Benchmarking
- [ ] Prepare RTTM ground truth files with speaker labels
- [ ] Integrate diarization models
- [ ] Run diarization benchmarks

### Phase 5: Analysis & Report
- [ ] Generate ranked comparison tables
- [ ] Write summary report with recommendations
- [ ] Push final results to GitHub

---

## Ground Truth Plan

### Problem
All WER/CER results are N/A because no ground truth transcriptions exist. Additionally, ~6 audio clips are bad (silence, wrong content, truncated).

### Bad Clips to Fix

| Clip | Issue | Action |
|------|-------|--------|
| `en/short/apple_earnings_01` | Silence/music → "Thank you." | Replace |
| `en/short/tesla_earnings_01` | Silence/music → "Thank you." | Replace |
| `en/long/apple_earnings_15min` | Repeating "Thank you." | Replace |
| `ja/long/toyota_earnings_60min` | Only 74s (labeled 60min) | Replace |
| `ja/long/softbank_earnings_15min` | Only 368s (labeled 15min) | Replace |
| `zh/short/huawei_launch_01` | Possibly song lyrics | Verify, replace if needed |
| `ja/short/cabinet_pressconf_01` | Garbled mixed-language output | Verify, replace if needed |

### Tier 1: Short Clips (15 × 30s = 7.5 min audio)

**Method:** OpenAI Whisper API draft → manual correction

1. Transfer 15 short WAVs from Jetson to Mac via `scp`
2. Run OpenAI Whisper API on all clips (~$0.05 total)
3. Cross-reference with existing whisper-turbo + faster-whisper transcripts
4. Listen to each clip, correct errors (~40 min work)
5. Write `ground_truth/{en,zh,ja}.json`

### Tier 2: Long Clips (9 clips, 15-60 min each)

**Method:** Segment sampling — full transcription of 4.5 hours is impractical

- Extract **3 × 2-min segments** per clip (early at ~1:00, middle, late before outro)
- 54 min total audio instead of 4.5 hours
- Tests accuracy at different points in recording, catches drift

**Ground truth format (supports both full and segmented):**
```json
{
  "short/fed_pressconf_01": "full transcription text...",
  "long/fed_pressconf_30min": [
    {"start": 60, "end": 180, "text": "segment 1 text..."},
    {"start": 870, "end": 990, "text": "segment 2 text..."},
    {"start": 1680, "end": 1800, "text": "segment 3 text..."}
  ]
}
```

### Code Changes Required
- `bench.py`: detect string vs list ground truth, handle segment extraction + evaluation
- `metrics/profiler.py`: add `extract_audio_segment()` utility
- Move ground truth to `ground_truth/` directory (outside gitignored `data/`)
- New helper: `data/scripts/extract_segments.py`

### Execution Order
1. **Phase 0:** Audit & fix bad clips (find verified YouTube URLs, re-download)
2. **Phase 1:** Short clip ground truth (Whisper API draft → manual correction)
3. **Phase 2:** Long clip ground truth (segment extraction → Whisper API → correction)
4. **Phase 3:** Code changes for segment-aware evaluation
5. **Phase 4:** Re-run benchmark with ground truth → WER/CER results

---

## Benchmark Results (2025-03-15, Jetson Orin 16GB)

| Model | Device | Load | Short RTF | Long RTF | VRAM |
|-------|--------|------|-----------|----------|------|
| whisper-large-v3-turbo | GPU | 16.5s | 0.422 | 0.185 | 3.4GB |
| faster-whisper-large-v3 | CPU int8 | 7.6s | 1.933 | 2.070 | 8MB |

**Conclusion:** whisper-large-v3-turbo (GPU) is the only viable model — 4-8x faster than real-time. faster-whisper is forced to CPU (no aarch64 CUDA CTranslate2) making it impractical (2x slower than real-time).

## Acceptance Criteria
- ≥12 ASR models benchmarked across 3 languages
- ≥4 diarization models benchmarked
- ≥3 combined ASR+diarization pipelines with cpWER
- Short clips (30s) AND long clips (15/30/60 min) tested
- WER/CER computed for all ASR models
- DER + cpWER computed for all diarization pipelines
- RTF and VRAM measured for all models
- Results reproducible via `python bench.py`
