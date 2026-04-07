# iOS Bench — Moonshine vs Apple Speech Framework

**Status:** scaffolded, waiting on Xcode install on Mac mini
**Started:** 2026-04-07
**Target:** asrbench/ios/ subdirectory, EN only first, simulator first then real iPhone

## Goal

Compare on-device speech recognition on iPhone between:
1. **Moonshine** (`moonshine-base-en` via [moonshine-swift](https://github.com/moonshine-ai/moonshine-swift) v0.0.51) — third-party, ONNX Runtime backend, ~150 MB
2. **Apple Speech framework** (`SFSpeechRecognizer` with `requiresOnDeviceRecognition = true`) — built into iOS, runs on Apple Neural Engine on real device

Apples-to-apples comparison on the same audio set (LibriSpeech test-clean subset), measuring:
- **WER** (word error rate, normalized) — accuracy
- **Wall-clock per-utterance latency** — speed
- **Confidence** — Apple Speech provides this, Moonshine does too via word timestamps

## Methodology constraints (read these before judging numbers)

- **Simulator runs the Mac CPU**, not iPhone hardware. Latency numbers from the simulator are *Mac CPU latency*, not Neural Engine. Accuracy is hardware-independent so the WER comparison is meaningful even on the simulator. **Phase 2 = real device** is what gives the iPhone speed story.
- **`SFSpeechRecognizer.supportsOnDeviceRecognition` is unreliable on simulator** in some Xcode versions — the framework may silently fall back to server mode (audio uploaded to Apple, datacenter compute). The runner will fail loudly if on-device support is not available rather than silently use server mode.
- **English only for now.** Moonshine multilingual Flavors are available via `python -m moonshine_voice.download --language ja|zh` (the Swift package supports them, contrary to my earlier guess), so adding JA/ZH is a follow-up — same Xcode project, just more bundled models and a different `--language` arg.

## Architecture

```
asrbench/ios/
├── README.md                          ← how to build/run on Mac mini
├── project.yml                        ← xcodegen spec (generates .xcodeproj)
├── Info.plist                         ← Speech permission strings
├── Sources/
│   ├── AsrBenchApp.swift              ← @main SwiftUI entry, runs benchmark on launch
│   ├── BenchRunner.swift              ← orchestration, loads manifest, iterates audio
│   ├── AppleSpeechRunner.swift        ← SFSpeechRecognizer with requiresOnDeviceRecognition
│   ├── MoonshineRunner.swift          ← MoonshineVoice.Transcriber wrapper
│   └── Result.swift                   ← Codable result struct + JSON writer
├── Resources/
│   ├── audio/                         ← bundled WAV files (16kHz mono 16-bit)
│   │   ├── 1089-134686-0000.wav
│   │   ├── ...
│   │   └── manifest.json              ← {audio_path, reference, sample_id, language}
│   └── moonshine-models/              ← .ort files, downloaded via moonshine_voice
│       ├── tiny-en/{encoder,decoder,tokenizer}
│       └── base-en/{encoder,decoder,tokenizer}
└── scripts/
    ├── prepare_audio.py               ← FLAC → WAV 16k mono converter, writes manifest
    ├── compute_metrics.py             ← reads result JSON, computes WER per runner
    └── setup_mini.sh                  ← one-shot install on Mac mini after Xcode is up
```

The app on launch:
1. Loads `Resources/audio/manifest.json`
2. For each runner (Apple Speech, Moonshine tiny-en, Moonshine base-en):
   - For each audio sample:
     - Load WAV → Float array
     - Time the transcription
     - Append `{runner, sample_id, transcript, duration_s, latency_s}` to results
3. Write results to `Documents/results-<timestamp>.json`
4. Display "Done — N samples × M runners" in the UI
5. Pulled back via `xcrun simctl get_app_container booted ai.moonshine.asrbench data` → file copy

## Acceptance criteria (definition of done)

### Phase 1: Simulator on Mac mini
- [ ] `xcodegen generate` succeeds in `asrbench/ios/`
- [ ] `xcodebuild -scheme AsrBenchIOS -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build` succeeds
- [ ] App installs to a booted iPhone 17 Pro simulator (`xcrun simctl install booted`)
- [ ] App launches and produces `results-*.json` in its Documents container
- [ ] `compute_metrics.py results-*.json` prints a per-runner WER table
- [ ] Apple Speech `supportsOnDeviceRecognition` returns true (or run fails loudly — no silent server fallback)
- [ ] Moonshine produces sensible transcripts on at least 5 sanity samples
- [ ] Results committed under `asrbench/results/ios_simulator_<timestamp>/`

### Phase 2: Real iPhone (deferred until you have a device + dev cert in Xcode)
- [ ] Same scheme builds for `generic/platform=iOS`
- [ ] Same app runs on physical iPhone, produces results JSON
- [ ] Latency numbers added to RESULTS.md as a new "iPhone hardware" row
- [ ] If results warrant it: a recommendation in RESULTS.md for iPhone-on-device transcription

## Steps (per CLAUDE.md "checkable items")

### Setup (Mac mini, after Xcode install completes)
- [ ] `xcodebuild -version` returns Xcode 26.x
- [ ] `xcrun simctl list devices available | grep iPhone` shows at least one simulator
- [ ] `brew install xcodegen` (project file generation, no Xcode runtime needed)
- [ ] `python3 -m pip install moonshine-voice` (model downloader)
- [ ] `python3 -m moonshine_voice.download --language en` (caches model files)

### Scaffold (this Mac, doesn't need Xcode)
- [x] Write this plan
- [ ] Create `asrbench/ios/` directory tree
- [ ] Write `project.yml` (xcodegen spec — moonshine-swift package + Speech framework)
- [ ] Write `Info.plist` (NSSpeechRecognitionUsageDescription, NSMicrophoneUsageDescription)
- [ ] Write `Sources/AsrBenchApp.swift`
- [ ] Write `Sources/BenchRunner.swift`
- [ ] Write `Sources/AppleSpeechRunner.swift`
- [ ] Write `Sources/MoonshineRunner.swift`
- [ ] Write `Sources/BenchResult.swift` (Codable)
- [ ] Write `scripts/prepare_audio.py` (FLAC → WAV 16k mono using soundfile)
- [ ] Run `prepare_audio.py` on 50 LibriSpeech samples → `Resources/audio/`
- [ ] Write `scripts/compute_metrics.py` (post-hoc WER, reuses asrbench/metrics)
- [ ] Write `scripts/setup_mini.sh` (the install commands above)
- [ ] Write `asrbench/ios/README.md` (build/run instructions)
- [ ] Commit + push as scaffold

### Build & run (Mac mini, after Xcode + scaffold both ready)
- [ ] Sync the asrbench/ios/ tree to Mac mini (rsync)
- [ ] Run `setup_mini.sh` on the mini
- [ ] Copy moonshine model `.ort` files into `Resources/moonshine-models/{tiny-en,base-en}/`
- [ ] `cd asrbench/ios && xcodegen generate`
- [ ] `xcodebuild -scheme AsrBenchIOS -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build`
- [ ] `xcrun simctl boot "iPhone 17 Pro"` (if not booted)
- [ ] `xcrun simctl install booted ./build/.../AsrBenchIOS.app`
- [ ] `xcrun simctl launch --console-pty booted ai.moonshine.asrbench`
- [ ] `xcrun simctl get_app_container booted ai.moonshine.asrbench data` → cp `Documents/results-*.json` back
- [ ] `python3 scripts/compute_metrics.py results-*.json`
- [ ] Commit results + WER summary

### Open questions / gotchas to confirm during build
- [ ] Does the moonshine-swift binaryTarget actually link for iOS Simulator architectures (arm64-sim, x86_64-sim)? The XCFramework should include both — verify with `lipo -info` or build error.
- [ ] Does `SFSpeechRecognizer.supportsOnDeviceRecognition(forLocale:)` return true on iPhone 17 simulator under Xcode 26? If not, the run fails with a clear "Apple Speech on-device unavailable on this simulator" error rather than silently uploading audio.
- [ ] WAV file bundling: SwiftPM resources vs. Xcode "Copy Bundle Resources" — xcodegen needs the resources declared in `project.yml`.
- [ ] Speech framework first-launch authorization: `SFSpeechRecognizer.requestAuthorization` is async and prompts the user. On a simulator with no UI focus, does it auto-approve? May need to pre-authorize via `defaults write` or `simctl privacy`.

## Why not just use the Mac path

Both Apple Speech framework and Moonshine ONNX exist for macOS too. We *could* benchmark them on the Mac directly without Xcode/Simulator. Two reasons not to:

1. **The artifact you ship is an iOS app**, not a Mac binary. Bundling, code signing, simulator differences from device — the only way to learn those is to build the actual iOS target. A Mac-only test would tell us nothing about whether the iOS path works.
2. **Apple Speech framework on macOS is the same API but a different runtime path** than iOS — the dictation models, the on-device vs server policy, and the language packs all differ. Mac results don't predict iPhone results.

Phase 1 (simulator) gives us "does it build, does it run, what's the accuracy". Phase 2 (real device) gives us "what's the iPhone latency and battery cost". Both phases use the same Xcode project.
