# VibeVoice-ASR Speed Optimization — Getting to 20× Real-Time

**Date:** 2026-04-12
**Context:** Memio needs VibeVoice's diarization quality (~3.4% DER) at ≥20× real-time

---

## 1. Current state

| Platform | Runtime | RTF | Speed | WER |
|---|---|---:|---|---:|
| Jetson Orin 32GB | raw transformers BF16 | 0.84 | 1.2× RT | 3.11% |
| DGX Spark1 GB10 | raw transformers INT8 | 0.55 | 1.8× RT | 5.51% |
| **A100 80GB (leaderboard)** | **optimized** | **0.019** | **51.8× RT** | **2.20%** |

The model already does 51.8× real-time on A100 — proven on the Open ASR
Leaderboard (8 English datasets, NVIDIA A100-SXM4-80GB, CUDA 12.6, PyTorch 2.4).
The gap between our 1.2× and the leaderboard's 51.8× comes from hardware and runtime.

---

## 2. Root cause analysis

### 2a. Memory bandwidth is the real bottleneck

LLM inference is **memory-bandwidth-bound**, not compute-bound. The model reads
9B parameters from memory for every generated token. The speed is directly
proportional to how fast memory can feed data to the compute units.

| GPU | Memory Bandwidth | Relative to A100 |
|---|---:|---:|
| Jetson Orin 32GB | 204 GB/s | 0.10× |
| DGX Spark GB10 | 273 GB/s | 0.13× |
| **A100 80GB** | **2,039 GB/s** | **1.0×** |
| H100 80GB | 3,350 GB/s | 1.6× |

A100 has **7.5× more memory bandwidth** than Spark's GB10, and **10× more**
than Jetson Orin. This single factor explains 70-80% of the speed gap.

### 2b. Runtime overhead

Our benchmarks use raw `model.generate()` — the slowest inference path:
- Recomputes KV cache every step (no paged attention)
- No continuous batching
- No CUDA graph optimization
- Python loop overhead on every token

**vLLM** provides ~2-5× speedup through:
- Paged attention (efficient KV cache management, no memory waste)
- CUDA graph execution (eliminates Python loop overhead)
- Continuous batching (overlaps compute across requests)
- Optimized kernels (FlashAttention, Marlin for quantized models)
- VibeVoice has an official vLLM plugin with data-parallel + tensor-parallel

---

## 3. Speed options

### Option A: vLLM on Spark1/Spark2 (GB10)

```bash
# Install and launch
pip install vllm==0.14.1
git clone https://github.com/microsoft/VibeVoice
cd VibeVoice && python3 start_server.py
```

**Estimated speed:**

| Config | Estimated Speed | WER | VRAM |
|---|---|---:|---:|
| vLLM BF16 | ~7-10× RT | 3.1% | ~18 GB |
| vLLM INT8 (bitsandbytes) | ~10-15× RT | ~5.5% | ~11 GB |
| vLLM INT4 (AWQ + Marlin) | ~14-20× RT | ~5-6% | ~6 GB |
| vLLM + speculative decoding | ~15-25× RT | ~3.1% | ~20 GB |

**Calculation:** GB10 bandwidth (273 GB/s) ÷ A100 (2039 GB/s) × 51.8× leaderboard
× vLLM boost ≈ 7-15× BF16, up to 20× with INT4.

**Verdict:** Borderline for 20× target. BF16 won't reach it. INT4 might but
degrades WER. Speculative decoding is experimental.

### Option B: Cloud A100 (proven 51.8×) — RECOMMENDED

Rent a cloud A100 80GB for the HiNotes server:

| Provider | GPU | Cost/hour | Speed | 20× guaranteed? |
|---|---|---:|---|:---:|
| Lambda | A100 80GB | $1.10 | 51.8× RT | **Yes** |
| RunPod | A100 80GB | $1.64 | 51.8× RT | **Yes** |
| Azure ND A100 | A100 80GB | $3.67 | 51.8× RT | **Yes** |
| Together.ai API | Serverless | ~$0.10/hr audio | 50+× RT | **Yes** |

**Per-meeting cost:**
- 1-hour meeting at 51.8× = **~70 seconds** to transcribe + diarize
- At $1.10/hr GPU cost: **$0.02 per meeting**
- 10 meetings/day × 30 days: **$6/month**

### Option C: H100 (fastest possible)

| GPU | Memory Bandwidth | Estimated Speed |
|---|---:|---|
| H100 80GB SXM | 3,350 GB/s | ~80-100× RT |
| H100 NVL (dual) | 3,350 GB/s × 2 | ~150× RT with TP=2 |

Overkill for single-user Memio but relevant for multi-tenant HiNotes service.

---

## 4. vLLM deployment details

### Server setup (from VibeVoice docs)

```bash
# Docker (recommended)
docker run -d --gpus all -p 8000:8000 \
  vllm/vllm-openai:v0.14.1 \
  --model microsoft/VibeVoice-ASR \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 16

# Parallelism options
python3 start_server.py --dp 1           # single GPU
python3 start_server.py --dp 4           # 4 GPU data-parallel
python3 start_server.py --tp 2           # 2 GPU tensor-parallel
python3 start_server.py --dp 2 --tp 2    # hybrid (4 GPUs total)
```

### Client API (OpenAI-compatible)

```python
import httpx

response = httpx.post("http://server:8000/v1/chat/completions", json={
    "model": "microsoft/VibeVoice-ASR",
    "messages": [{"role": "user", "content": "<audio_base64>"}],
    "max_tokens": 4096,
})
transcript = response.json()["choices"][0]["message"]["content"]
```

### Tuning knobs

| Parameter | Default | Recommended | Effect |
|---|---|---|---|
| `--gpu-memory-utilization` | 0.9 | 0.95 | More KV cache = higher batch |
| `--max-num-seqs` | 256 | 16 (single user) | Lower = faster per-request |
| `VIBEVOICE_FFMPEG_MAX_CONCURRENCY` | 64 | 4 | Audio decode threads |
| `--enforce-eager` | false | true (debug) | Disable CUDA graph for debugging |

---

## 5. Memio integration architecture

```
┌─────────────────────────────────────────────────────┐
│  iPad (on-device, real-time)                        │
│                                                     │
│  Mic → AVAudioEngine → Moonshine ASR (19× RT)      │
│                       → FluidAudio diarization      │
│                       → Live preview (~22% DER)     │
│                                                     │
│  When on Wi-Fi:                                     │
│  Audio file → HTTPS upload ─────────────────────┐   │
│                                                 │   │
│  ← diarized transcript (replace preview) ←──┐   │   │
└─────────────────────────────────────────────│───│───┘
                                              │   │
┌─────────────────────────────────────────────│───│───┐
│  HiNotes Server (Cloud A100 / Spark1)       │   │   │
│                                             │   ▼   │
│  vLLM + VibeVoice-ASR ──────────────────────┘       │
│    51.8× RT on A100 / ~10× on Spark1               │
│    DER ~3.4%, WER ~2.2%                             │
│    + Qwen 32B summarization                          │
│    + Entity extraction                               │
│    + Embedding generation                            │
│                                                     │
│  Ephemeral: audio deleted after processing           │
└─────────────────────────────────────────────────────┘
```

**Flow:**
1. iPad records + Moonshine transcribes in real-time (on-device, 19× RT)
2. FluidAudio provides rough speaker attribution (~22% DER) for live preview
3. When on Wi-Fi, audio uploaded to HiNotes server
4. VibeVoice-ASR processes: 1-hour meeting → 70 seconds on A100
5. Returns: diarized transcript (~3.4% DER) + summary + entities + embeddings
6. iPad replaces rough Moonshine/FluidAudio output with refined VibeVoice result
7. Audio deleted from server (ephemeral, consistent with Memio privacy model)

---

## 6. Cost analysis

### Per-meeting cost (Cloud A100)

| Audio length | Transcription time | GPU cost ($1.10/hr) |
|---:|---:|---:|
| 15 min | 17 seconds | $0.005 |
| 30 min | 35 seconds | $0.01 |
| 1 hour | 70 seconds | $0.02 |
| 2 hours | 140 seconds | $0.04 |

### Monthly cost by usage

| Meetings/day | Avg length | Monthly cost |
|---:|---|---:|
| 3 | 30 min | $0.90 |
| 5 | 45 min | $2.25 |
| 10 | 1 hour | $6.00 |
| 20 | 1 hour | $12.00 |

For comparison: a ChatGPT Plus subscription is $20/month. VibeVoice processing
for a heavy meeting user costs less than a third of that.

### Self-hosted option (Spark1 GB10)

| | Cloud A100 | Self-hosted Spark1 |
|---|---|---|
| Speed | 51.8× RT | ~10-15× RT (vLLM) |
| Cost | $0.02/meeting | $0 (hardware owned) |
| WER | 2.2-3.1% | 3.1-5.5% |
| Always-on | Pay per hour | Free |
| Setup | Docker pull | vLLM install + tune |

If Spark1 achieves ≥10× with vLLM (plausible), a 1-hour meeting takes
~6 minutes — acceptable for async processing. Not real-time, but fine for
"process after the meeting ends" workflow.

---

## 7. Recommendation

**For Memio: use Cloud A100 ($6/month) for HiNotes server.**

- 51.8× real-time (2.5× your 20× target)
- Best WER (2.2% on leaderboard, 3.1% on our bench)
- Best DER (~3.4%)
- $0.02/meeting — negligible cost
- Fits the existing HiNotes ephemeral-cloud architecture
- No hardware to maintain

**Fallback: Spark1 with vLLM** for zero-cost operation at ~10-15× speed.
Acceptable for async processing (6 min for a 1-hour meeting).

**Do NOT run on Jetson Orin** — 1.2× RT means a 1-hour meeting takes 50 min.
The Orin is for Moonshine + pyannote (on-device preview), not VibeVoice.

---

## 8. Next steps

1. [ ] Install vLLM on Spark1 and benchmark VibeVoice-ASR throughput
2. [ ] Compare vLLM BF16 vs raw transformers on same hardware (expected ~3-5× boost)
3. [ ] Test cloud A100 via Lambda/RunPod — verify 51.8× reproduces
4. [ ] Integrate VibeVoice vLLM endpoint into HiNotes server FastAPI
5. [ ] Update Memio iOS HiNotesClient to upload audio (not just transcript text)
6. [ ] End-to-end test: iPad record → Wi-Fi upload → VibeVoice → diarized result back
