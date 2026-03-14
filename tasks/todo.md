# ASR Bench — Task List

## Project Goal
Benchmark top open-source ASR models on NVIDIA Jetson (16GB) for Japanese, Chinese, and English.

## Tasks

### Phase 1: Setup & Test Data
- [ ] Initialize git repo and push to GitHub
- [ ] Prepare SHORT test audio (5–8 clips × 30s per language)
  - [ ] EN: earnings call excerpts, board meetings, conference calls
  - [ ] ZH: 财报电话会 excerpts, 国新办发布会, business calls
  - [ ] JA: 決算説明会 excerpts, 記者会見, shareholder meetings
  - [ ] Include noisy samples (conference call / phone quality)
- [ ] Prepare LONG test audio (15min / 30min / 60min per language)
  - [ ] EN: full earnings call (15m), board meeting (30m), hearing (60m)
  - [ ] ZH: full 财报电话会 (15m), press conference (30m), meeting (60m)
  - [ ] JA: full 決算説明会 (15m), 記者会見 (30m), 株主総会 (60m)
- [ ] Prepare ground truth transcription files (JSON format)
  - [ ] Short clips: manual transcription
  - [ ] Long clips: source from official transcripts / subtitles
- [ ] Set up Jetson environment (CUDA, Python 3.10+, venv)

### Phase 2: Model Integration
- [ ] Implement model runner interface (load, transcribe, measure)
- [ ] Integrate each model (one runner per model):
  - [ ] Whisper large-v3
  - [ ] Whisper large-v3-turbo
  - [ ] Faster-Whisper large-v3
  - [ ] Moonshine Base / Small / v2
  - [ ] SenseVoice-Large (FunASR)
  - [ ] Paraformer-Large (FunASR)
  - [ ] Fun-ASR-Nano (0.8B)
  - [ ] Qwen3-ASR-1.7B
  - [ ] Qwen3-ASR-0.6B
  - [ ] FireRedASR-AED (1.1B)
  - [ ] NVIDIA Canary-Qwen-2.5B
  - [ ] NVIDIA Canary-1B-v2
  - [ ] NVIDIA Parakeet-CTC-1.1B (EN only)
  - [ ] Kotoba-Whisper v2 (JA focused)
  - [ ] Samba-ASR
  - [ ] Meta Omnilingual ASR (1B or smaller variant)
- [ ] Verify each model loads and runs on Jetson 16GB

### Phase 3: ASR Benchmarking
- [ ] Run all models against short clips (3x per model for variance)
- [ ] Run all models against long clips (15/30/60 min)
- [ ] Collect metrics: WER/CER, SeMaScore, RTF, VRAM peak, load time, first-token latency
- [ ] Compute long-form drift and memory stability
- [ ] Generate composite score ranking

### Phase 4: Speaker Diarization Benchmarking
- [ ] Prepare RTTM ground truth files with speaker labels for all test clips
- [ ] Integrate diarization models:
  - [ ] pyannote/speaker-diarization-3.1
  - [ ] NVIDIA NeMo MSDD
  - [ ] NVIDIA Sortformer
  - [ ] SpeechBrain ECAPA-TDNN
  - [ ] DiariZen
- [ ] Integrate combined ASR+diarization pipelines:
  - [ ] WhisperX (Whisper + pyannote)
  - [ ] NeMo Canary + MSDD
  - [ ] FunASR SOND
  - [ ] Qwen3-ASR + pyannote
- [ ] Run diarization benchmarks: DER, JER, Missed/FA/Confusion
- [ ] Run combined pipeline benchmarks: cpWER
- [ ] Run on long-form audio (15/30/60 min)

### Phase 5: Analysis & Report
- [ ] Generate ranked comparison tables
  - [ ] Best ASR per language (EN, ZH, JA)
  - [ ] Best ASR all-rounder (JA+ZH+EN)
  - [ ] Best diarization model
  - [ ] Best combined ASR+diarization pipeline
  - [ ] Best for long-form meetings
- [ ] Write summary report with recommendations
- [ ] Push final results to GitHub

## Acceptance Criteria
- ≥12 ASR models benchmarked across 3 languages
- ≥4 diarization models benchmarked
- ≥3 combined ASR+diarization pipelines with cpWER
- Short clips (30s) AND long clips (15/30/60 min) tested
- WER/CER + SeMaScore computed for all ASR models
- DER + cpWER computed for all diarization pipelines
- RTF and VRAM measured for all models
- Results reproducible via `python bench.py`
