# ASR Bench — Task List

## Project Goal
Benchmark top open-source ASR models on NVIDIA Jetson (16GB) for Japanese, Chinese, and English.

## Tasks

### Phase 1: Setup & Test Data
- [ ] Initialize git repo and push to GitHub
- [ ] Prepare test audio dataset (5–10 clips per language, ~30s each)
  - [ ] English: LibriSpeech / CommonVoice samples with ground truth
  - [ ] Chinese (Mandarin): AISHELL / CommonVoice samples with ground truth
  - [ ] Japanese: CommonVoice / JSUT samples with ground truth
  - [ ] Include noisy samples (1–2 per language) for robustness testing
- [ ] Prepare ground truth transcription files (JSON format)
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

### Phase 3: Benchmarking
- [ ] Run all models against all test clips
- [ ] Collect metrics: WER/CER, RTF, VRAM peak, load time, first-token latency
- [ ] Run 3x per model for variance

### Phase 4: Analysis & Report
- [ ] Generate comparison table (model × language × metric)
- [ ] Identify top 3 per language
- [ ] Identify best "all-rounder" for JA+ZH+EN
- [ ] Write summary report with recommendations
- [ ] Push final results to GitHub

## Acceptance Criteria
- All models that fit in 16GB are benchmarked
- WER/CER computed against ground truth for all 3 languages
- RTF and VRAM measured for each model
- Results reproducible via single `python bench.py` command
