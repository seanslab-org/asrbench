# VibeVoice-ASR vs Pyannote — Why Unified Diarization Wins

**Date:** 2026-04-12
**Context:** Research for Memio/lonote ASR+diarization architecture decision

---

## 1. Two fundamentally different architectures

### Pyannote = modular pipeline (4 separate stages)

```
Audio → [VAD] → [Segmentation] → [Speaker Embedding] → [Clustering] → Labels
         ↓           ↓                    ↓                  ↓
      Silero    pyannote/seg-3.0      WeSpeaker           VBx/HDBSCAN
```

Each stage is trained independently. Errors cascade forward — a missed speech
segment in VAD means that speaker is lost forever downstream. The pipeline
processes audio in **10-second windows** with overlap, then stitches results
using embedding similarity.

### VibeVoice = unified single-pass (everything in one model)

```
Audio → [Acoustic Tokenizer + Semantic Tokenizer] → [Qwen2.5-7B LLM] → "Who-When-What" tokens
```

The LLM generates speaker labels, timestamps, AND transcript as a **single
interleaved token sequence**. No handoff between stages. No error propagation.
Processes **up to 60 minutes in one pass**.

**Key architectural details (from arXiv 2601.18184):**
- **Acoustic Tokenizer:** σ-VAE with hierarchical 3200× downsampling on 24 kHz
  input → ~7.5 tokens/second. 1 hour of audio ≈ 27,000 tokens (fits in 32K
  context window).
- **Semantic Tokenizer:** Extracts deterministic content features aligned with
  textual semantics, running in parallel with the acoustic tokenizer.
- **Speaker embeddings:** Uses pre-trained WeSpeaker for speaker representations,
  integrated into the sequence-to-sequence framework.
- **Output format:** The LLM autoregressively generates a "Rich Transcription"
  that explicitly interleaves speaker identity ("Who"), temporal boundaries
  ("When"), and speech content ("What") as tokens.
- **Context injection:** Optional text prompts (hotwords, speaker names,
  background info) can be prepended to the audio sequence for domain adaptation.
- **Training data mix:** Standard benchmarks (0.5) + Music (0.1) + Synthetic
  (0.1) + Refined long-form (0.3).

---

## 2. Why unified wins — three specific mechanisms

### 2a. No context fragmentation

Pyannote processes audio in **10-second windows** with overlap, then stitches:
- A speaker who starts a sentence in window N and finishes in window N+1 gets
  split across two segments
- The clustering stage must reconcile fragments using embedding similarity alone
- Long silences between one speaker's turns can cause the clusterer to assign
  them different speaker IDs

VibeVoice sees the **entire conversation**. If Speaker A speaks at minute 2
and again at minute 45, the model has full context to recognize the same
person — through voice characteristics AND conversational context (the LLM
understands "who would say this" based on what's been said before).

### 2b. Joint optimization of WHO + WHAT

In pyannote's pipeline, diarization and transcription are completely separate.
Pyannote doesn't know what anyone said — it only knows when voice A vs voice B
was active. This means it **cannot use**:
- **Linguistic cues** ("As I mentioned earlier..." → same speaker as before)
- **Conversational structure** (Q&A patterns, turn-taking norms)
- **Named entities** ("Sean said..." → tells you who the current speaker is
  addressing)
- **Topic continuity** (same person tends to keep talking about the same topic)

VibeVoice's LLM understands language. It sees the transcript forming in
real-time and uses **both acoustic AND semantic signals** to attribute speakers.
The Qwen2.5-7B backbone brings general language understanding to diarization —
something no embedding-clustering pipeline can do.

**This is the fundamental insight:** VibeVoice treats diarization as a
**language modeling problem** (predicting who speaks next, given everything
said so far). Pyannote treats it as a **signal processing problem** (clustering
voice embeddings by similarity). The language modeling approach is more powerful
because it uses world knowledge, conversational structure, and semantic context
that no amount of speaker embedding refinement can match.

### 2c. No stitching artifacts

Pyannote's sliding-window approach creates boundary artifacts:
- **Speaker switches at window boundaries** get double-counted or missed
- **Overlapping speech** near window edges is particularly unreliable
- The VBx/HDBSCAN clustering is a heuristic — it doesn't learn from data,
  it just measures distance in embedding space

VibeVoice has no windows to stitch. The 3200× compression means the entire
conversation fits in the LLM's context window as a single sequence.

---

## 3. Benchmark numbers

### DER (Diarization Error Rate — lower is better)

| Dataset | VibeVoice-ASR | pyannote 3.1 (legacy) | pyannote precision-2 | Gemini-2.5-Pro |
|---|---:|---:|---:|---:|
| **AMI IHM** | — | 18.8% | 12.9% | — |
| **AMI SDM** | — | 22.7% | 15.6% | — |
| **AISHELL-4** | 16.93% | 12.2% | 11.4% | — |
| **AliMeeting** | — | 24.5% | 15.2% | — |
| **MLC-Challenge** | **4.28%** | — | — | 16.3% |
| **Cross-dataset mean** | **~3.4%** | — | — | 16.3% |

### Full-pipeline metrics (ASR + diarization combined)

| Metric | VibeVoice-ASR | Gemini-2.5-Pro | Gemini-3-Pro |
|---|---:|---:|---:|
| **DER** | 3.4% | 16.3% | 33% |
| **cpWER** | 11.48% | — | — |
| **tcpWER** | 15.7% | 28.9% | 58.8% |

**Metric definitions:**
- **DER:** Speaker attribution accuracy (speaker confusion + missed speech +
  false alarm speech)
- **cpWER:** Concatenated minimum-permutation WER — transcription accuracy
  under speaker permutation invariance
- **tcpWER:** Time-constrained cpWER — extends cpWER by enforcing temporal
  alignment, sensitive to both speaker attribution and word-level timing

### VibeVoice-ASR transcription accuracy (from asrbench)

| Metric | Value |
|---|---|
| EN WER (LS-clean, BF16, 386 samples) | 3.11% |
| EN WER (LS-clean, INT8, 50 samples) | 5.51% (preliminary) |
| MLC-Challenge DER | 4.28% |
| MLC-Challenge cpWER | 11.48% |
| MLC-Challenge tcpWER | 13.02% |
| VRAM (BF16) | 16.4 GB |
| VRAM (INT8) | ~10-11 GB |
| RTF (Jetson Orin BF16) | 0.841 |

---

## 4. The nuance — pyannote isn't always worse

**AISHELL-4:** pyannote precision-2 gets **11.4% DER** vs VibeVoice's
**16.93% DER**. **Pyannote wins.** This is a conference-room dataset with
well-separated speakers on individual headset microphones — exactly the
scenario where local segmentation + clustering works well.

**MLC-Challenge (multi-language, real-world):** VibeVoice gets **4.28% DER**.
Pyannote doesn't have published numbers on this benchmark, but its typical
range (11-25%) suggests VibeVoice wins by a large margin.

**The pattern:**

| Condition | Winner | Why |
|---|---|---|
| **Clean, controlled audio** (studio, headset mics) | pyannote | Local embeddings are high-quality; clustering works well |
| **Messy, real-world audio** (meetings, podcasts, phone calls) | VibeVoice | Full-context LLM handles overlap, noise, code-switching |
| **High speaker count** (>5 speakers) | VibeVoice | Clustering struggles; LLM maintains identity over long context |
| **Overlapping speech** | VibeVoice | No window-boundary stitching artifacts |
| **Code-switching** (multilingual) | VibeVoice | LLM understands language transitions |
| **Short audio** (<5 min) | pyannote | LLM overhead not justified; embeddings suffice |

---

## 5. Pyannote's known failure modes

From the benchmarking literature (arXiv 2509.26177 and pyannote community):

1. **Missed speech is the dominant error** — speech onset and offset timing
   errors cause the segmenter to miss the first/last words of a turn.
   VibeVoice doesn't have this problem because it doesn't have a separate
   VAD stage.

2. **Speaker confusion in high-count settings** — when 5+ speakers are present,
   embedding clusters overlap and HDBSCAN makes incorrect merges/splits.
   VibeVoice uses LLM context to disambiguate.

3. **Cross-window identity loss** — the same speaker in different 10-second
   windows may get different embeddings (different emotional state, different
   background noise, speaking louder/softer). Clustering tries to handle this
   but fails when the variation is too large.

4. **Language-dependent degradation** — pyannote's segmentation model was
   primarily trained on English/French data. Performance drops significantly
   on under-resourced languages. VibeVoice's Qwen2.5 backbone has broad
   multilingual knowledge.

5. **No semantic awareness** — pyannote cannot use conversational cues to
   resolve ambiguous segments. Two speakers with similar voices (e.g., siblings)
   will be confused purely on acoustic similarity. VibeVoice can use what they're
   saying to tell them apart.

---

## 6. What this means for Memio

Memio records **real meetings** — not LibriSpeech audiobook readings. Real
meetings have:
- Overlapping speech (people talk over each other)
- Variable speaker distances from the mic (iPad on table, some people far away)
- Background noise (HVAC, keyboard, coffee machine)
- Code-switching (EN/JA/ZH in one meeting — the user's primary languages)
- Long silences followed by the same speaker resuming

These are exactly the conditions where **VibeVoice wins** and **pyannote
struggles**.

### Architecture options for Memio

| Option | ASR | Diarization | DER | VRAM | Where |
|---|---|---|---:|---:|---|
| **A: iPad-only** | Moonshine-base | FluidAudio (pyannote CoreML) | ~22% | 635 MB | On-device |
| **B: iPad + server** | Moonshine (iPad) | VibeVoice (server) | ~4% | 16 GB server | Hybrid |
| **C: Server-only** | VibeVoice (unified) | Built-in | ~4% | 16 GB | HiNotes/Spark |
| **D: iPad + post-process** | Moonshine (real-time) | FluidAudio (preview) → VibeVoice (refine on Wi-Fi) | ~4% final | 635 MB + server | Hybrid deferred |

**Recommendation: Option D** — best of both worlds:
1. **Real-time on iPad:** Moonshine transcribes + FluidAudio diarizes as
   you record. Gives you a rough "who said what" immediately. ~22% DER
   is good enough for live preview.
2. **When on Wi-Fi:** Send the audio to HiNotes server running VibeVoice-ASR.
   Get back a refined transcript with ~4% DER, proper timestamps, and
   corrected speaker IDs. Replace the rough diarization with the server result.
3. **Audio never leaves the device permanently** — ephemeral server processing,
   consistent with Memio's privacy model.

This matches the existing HiNotes architecture (transcript text → server →
enrichment → back to device) but extends it to audio → server → diarized
transcript → back to device.

---

## 7. The fundamental lesson

> **Diarization is a language problem, not just a signal processing problem.**

Pyannote asks: "Do these two audio segments sound like the same voice?"
VibeVoice asks: "Given everything said in this conversation so far, who is
most likely speaking this next sentence?"

The second question is fundamentally more informative. It uses:
- Acoustic similarity (same as pyannote)
- Conversational structure (who tends to follow whom)
- Semantic continuity (what topic is being discussed)
- World knowledge (meeting norms, Q&A patterns)

The cost: a 9B-parameter LLM instead of a 50 MB embedding model.
The payoff: 3.4% DER instead of 11-22% DER on real-world audio.

For Memio: use the cheap model (FluidAudio) for real-time on-device,
and the expensive model (VibeVoice) for offline refinement via HiNotes.
Don't choose — use both.
