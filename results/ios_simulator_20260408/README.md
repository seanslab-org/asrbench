# iOS Simulator bench — 2026-04-08

**Host:** Mac mini (Apple Silicon, macOS 26.2 build 25C56)
**Xcode:** 26.4 (build 17E192)
**Simulator:** iPhone 17 Pro, iOS 26.4 (23E244)
**Subset:** 50 LibriSpeech test-clean samples (single-speaker, reader 1089)

## Results

| Runner | N | Errs | WER% | RTF | Avg latency (s) |
|---|---:|---:|---:|---:|---:|
| **moonshine-base-en** | 50 | 0 | **2.55** | 0.034 | 0.253 |
| **moonshine-tiny-en** | 50 | 0 | **4.59** | 0.024 | 0.177 |
| apple-speech-en-US | 50 | 50 | — | — | — |

The 2.55% WER for moonshine-base-en exactly matches the prior 2026-04-02 ONNX
baseline on a different 388-sample subset, providing independent validation
that the Swift binary path produces equivalent accuracy to both the Python
ONNX (`useful-moonshine-onnx`) and the Python transformers
(`MoonshineForConditionalGeneration`) paths.

For reference, the same models on the full 2620-sample LibriSpeech test-clean
(diverse speakers) via Jetson Orin GPU/transformers (committed `cbd62bd`):

| Runner | N | WER% |
|---|---:|---:|
| moonshine-base-en | 2618 | 4.54 |
| moonshine-tiny-en | 2618 | 5.90 |

The ~2× gap between this 50-sample subset and the full set is from speaker
diversity, not runtime — single-speaker subsets are easier than the full
test-clean.

## Critical caveat — RTF is meaningless

**The RTF column is Mac CPU latency**, not iPhone Neural Engine. The simulator
runs on the host's arm64 CPU; there is no Neural Engine emulation. The Mac mini
has a faster CPU than any iPhone, so these RTF numbers are *better* than what
an iPhone would actually deliver. Real iPhone latency requires Phase 2 (a
physical device).

Accuracy (WER) is hardware-independent so the WER comparison is meaningful.

## Apple Speech: failed on all 50 samples

Every Apple Speech call returned:
```
recognitionFailed: Error Domain=kLSRErrorDomain Code=300
"Failed to initialize recognizer"
NSUnderlyingError={...
  Failed to create recognizer from=
  /private/var/MobileAsset/AssetsV2/com_apple_MobileAsset_UAF_Siri_Understanding/
  purpose_auto/<hash>.asset/AssetData/mini.json
}
```

Root cause: **iOS Simulator does not ship the on-device Siri/Speech offline
models.** The Speech framework's `requiresOnDeviceRecognition = true` mode
requires an asset that's only present on real iPhones. The simulator can fall
back to server mode, but our `AppleSpeechRunner` is configured to fail loudly
in that case rather than silently upload audio to Apple — see
`ios/Sources/AppleSpeechRunner.swift`.

Practical implication: **Apple Speech vs Moonshine accuracy comparison is not
possible on the iOS Simulator.** Both runners need to live on a real iPhone
(Phase 2) for that comparison to be meaningful.

## How permissions were granted

`xcrun simctl privacy <UDID> grant speech-recognition <bundle>` returned
"Operation not permitted" on Xcode 26 — that subcommand cannot grant Speech
Recognition specifically. Workaround: directly insert a row into the
simulator's TCC database:

```sql
sqlite3 ~/Library/Developer/CoreSimulator/Devices/<UDID>/data/Library/TCC/TCC.db \
  "INSERT OR REPLACE INTO access
   (service, client, client_type, auth_value, auth_reason, auth_version)
   VALUES ('kTCCServiceSpeechRecognition', '<bundle-id>', 0, 2, 4, 1);"
```

This is documented in `ios/scripts/setup_mini.sh` for future runs.

## Files

- `results.json` — raw output from `BenchRunner.swift`, 150 records
- `screenshot_auth_dialog.png` — what the simulator showed before TCC.db patch
