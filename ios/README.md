# asrbench/ios — Moonshine vs Apple Speech on iPhone

iOS subproject of asrbench. Compares **Moonshine** (`tiny-en` + `base-en` via
[moonshine-swift](https://github.com/moonshine-ai/moonshine-swift) v0.0.51) to
**Apple's Speech framework** (`SFSpeechRecognizer` with on-device recognition
forced) on a 50-sample LibriSpeech subset.

**Phase 1 = iOS Simulator** (validates the build, accuracy comparison).
**Phase 2 = real iPhone** (latency / Neural Engine measurements).

> ⚠️ Simulator runs on the host Mac CPU, not iPhone hardware. RTF numbers from
> the simulator are *Mac CPU* numbers, not iPhone Neural Engine. Accuracy (WER)
> is hardware-independent so that comparison is meaningful even on the simulator.

## Layout

```
ios/
├── README.md                    ← you are here
├── project.yml                  ← xcodegen spec, generates AsrBenchIOS.xcodeproj
├── Info.plist                   ← Speech framework permission strings
├── Sources/
│   ├── AsrBenchApp.swift        ← @main SwiftUI app, runs benchmark on launch
│   ├── BenchRunner.swift        ← orchestrates the runs, writes results JSON
│   ├── AppleSpeechRunner.swift  ← SFSpeechRecognizer (on-device, fail-loud)
│   ├── MoonshineRunner.swift    ← MoonshineVoice.Transcriber wrapper
│   └── BenchResult.swift        ← Codable result/manifest types
├── Resources/
│   ├── audio/                   ← bundled WAVs + manifest.json (50 samples)
│   └── moonshine-models/        ← .ort model files (populated by setup_mini.sh)
└── scripts/
    ├── prepare_audio.py         ← FLAC → WAV 16k mono converter (run on host)
    ├── compute_metrics.py       ← reads results JSON → WER table (run on host)
    └── setup_mini.sh            ← one-shot install for the Mac mini
```

## Build / run on Mac mini (CLI-only, no Xcode GUI needed)

Once Xcode is installed on the mini:

```bash
# 1. Sync the ios/ tree to the mini (first time only — or use git pull)
rsync -a /path/to/asrbench/ios/ phonemac:~/asrbench/ios/

# 2. SSH in and run the setup script
ssh phonemac
cd ~/asrbench/ios
./scripts/setup_mini.sh
```

`setup_mini.sh` will:
1. Verify `xcodebuild` and a simulator runtime
2. `brew install xcodegen`
3. `pip install moonshine-voice` and `python -m moonshine_voice.download --language en`
4. Stage `.ort` model files into `Resources/moonshine-models/{tiny-en,base-en}/`
5. `xcodegen generate` → `AsrBenchIOS.xcodeproj`
6. `xcodebuild build` for iPhone 17 Pro simulator

Then to actually run the benchmark:

```bash
# On the mini, after setup_mini.sh succeeds:
xcrun simctl boot "iPhone 17 Pro" || true   # ignore "already booted"
APP=$(find ~/Library/Developer/Xcode/DerivedData -name AsrBenchIOS.app -path '*Debug-iphonesimulator*' | head -1)
xcrun simctl install booted "$APP"
xcrun simctl launch --console-pty booted ai.moonshine.asrbench.AsrBenchIOS

# After it finishes, pull the results JSON
CONTAINER=$(xcrun simctl get_app_container booted ai.moonshine.asrbench.AsrBenchIOS data)
cp "$CONTAINER/Documents/results-*.json" /tmp/

# Compute WER on the host
python3 scripts/compute_metrics.py /tmp/results-*.json
```

Expected output:

```
results from: results-2026-04-07T....json
device:       Version 18.x (Build 22Axxx)
build:        1
runners:      3

  Runner                        N  Errs  Hall    WER%      RTF   Latency(s)
  ------------------------------------------------------------------------
  apple-speech-en-US           50     0     0    X.XX%   X.XXX     X.XXXs
  moonshine-base-en            50     0     0    X.XX%   X.XXX     X.XXXs
  moonshine-tiny-en            50     0     0    X.XX%   X.XXX     X.XXXs
```

## Acceptance gates (per `tasks/ios-bench-todo.md`)

The build is "done" when:
- `xcodebuild build` succeeds for the simulator
- App runs to completion in the simulator without errors
- `compute_metrics.py` produces a per-runner WER table
- `SFSpeechRecognizer.supportsOnDeviceRecognition` returned true (the runner
  fails loudly if not — we never silently use server mode)
- Results committed under `asrbench/results/ios_simulator_<timestamp>/`

## Open questions to resolve during build

1. Does the moonshine-swift binary `XCFramework` link cleanly for iOS Simulator
   (arm64-sim + x86_64-sim slices)?
2. Does Apple Speech `supportsOnDeviceRecognition(forLocale: en-US)` return
   `true` on iPhone 17 simulator under Xcode 26?
3. Does `SFSpeechRecognizer.requestAuthorization` auto-approve on a simulator
   without user interaction, or do we need `xcrun simctl privacy ... grant
   speech-recognition`?
4. Where exactly does `moonshine_voice.download --language en` write the `.ort`
   files? Cache layout may differ between versions; `setup_mini.sh` tries
   several known patterns and warns if none match.

## Phase 2: real iPhone (deferred)

Same Xcode project, just change the destination:

```bash
xcodebuild -scheme AsrBenchIOS -destination 'generic/platform=iOS' build
xcrun devicectl device install app --device <udid> AsrBenchIOS.app
```

Real-device runs need a paid Apple Developer cert in Xcode and a USB-attached
iPhone. Set `CODE_SIGNING_ALLOWED: YES` and `DEVELOPMENT_TEAM` in `project.yml`
when ready.
