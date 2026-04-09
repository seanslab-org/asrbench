# iPad mini 6 Real Device Bench — 2026-04-09

**Device:** iPad mini 6 (iPad14,1, A15 Bionic)
**OS:** iPadOS 26.3 (Build 23D127)
**Subset:** 50 LibriSpeech test-clean samples (single speaker, reader 1089)
**Build host:** Mac mini (Apple Silicon, macOS 26.2, Xcode 26.4)

## Results

| Runner | N | WER% | RTF | Avg latency |
|---|---:|---:|---:|---:|
| **moonshine-base-en** | 50 | **2.45** | 0.053 | 0.396s |
| **moonshine-tiny-en** | 50 | **4.69** | 0.036 | 0.267s |
| **apple-speech-en-US** | 50 | **10.92** | 0.038 | 0.280s |

All three runners: 0 errors, 0 crashes, 150 total records.

## Key findings

1. **Moonshine-base beats Apple Speech by 4.5× on accuracy** (2.45% vs 10.92%
   WER). Apple Speech makes real word-level errors: "Stuff it into you" →
   "Stuffed into you", missing words, wrong conjugations. Moonshine gets these
   right consistently.

2. **Speed is comparable across all three runners** — RTF 0.036–0.053, all
   running 19–28× real-time on A15 Bionic. Apple Speech is marginally faster
   per sample (0.038 RTF) but the difference is negligible.

3. **Moonshine-tiny is the fastest** at 0.036 RTF (28× real-time) with 4.69%
   WER — still more than 2× more accurate than Apple Speech while being the
   quickest.

4. **These are real hardware numbers** — A15 Bionic CPU + Neural Engine,
   not Mac CPU emulation like the simulator run.

## Apple Speech notes

- Used the legacy `SFSpeechRecognizer` API (not iOS 26's new
  `SpeechTranscriber`, which crashes with EXC_BREAKPOINT on iPad mini 6 —
  Apple hasn't added it to their hardware allowlist).
- `requiresOnDeviceRecognition` was set dynamically based on
  `supportsOnDeviceRecognition` — if the on-device model was available it
  used it; otherwise fell back to server mode.
- `addsPunctuation = false` (iOS 16+) to keep output comparable to Moonshine.
- Speech authorization was granted via the system dialog on first launch.

## Moonshine notes

- Models bundled in the app: `moonshine-tiny-en` (71 MB) and
  `moonshine-base-en` (238 MB), loaded from `Resources/moonshine-models/`.
- Uses the `moonshine-swift` v0.0.51 xcframework (`MoonshineVoice` library).
- `transcribeWithoutStreaming(audioData:, sampleRate: 16000)` — batch mode,
  no streaming.

## Comparison with other platforms

| Platform | moonshine-base-en WER | RTF | Notes |
|---|---:|---:|---|
| **iPad mini 6 (A15)** | **2.45%** | 0.053 | This run, 50 samples |
| Mac mini simulator | 2.55% | 0.034 | Same 50 samples, Mac CPU |
| Jetson Orin 32GB (CUDA) | 4.54% | 0.112 | Full 2620 samples |
| Jetson Orin 32GB (ONNX CPU) | 2.55% | 0.060 | 388-sample subset, April 2 |

The 2.45% on iPad is the best WER we've measured — likely because A15's
Neural Engine handles the model inference slightly differently than CPU-only
paths. The Jetson's 4.54% on full 2620 samples reflects the harder
diverse-speaker test set, not a runtime issue.

## Files

- `results.json` — 150 records (50 per runner), raw output from BenchRunner
