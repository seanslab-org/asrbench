# iPad Moonshine Sugr-ASR-Bench — Handoff

**Date:** 2026-04-16
**Status:** Data prep done, build+run handoff to Sean on Mac mini (Tailscale SSH from remote was too slow/flaky for the large file transfers + Xcode build automation).

## What's already done

### Parakeet baseline (DONE, committed as `115e84c`)

`parakeet-tdt-1.1b` on Jetson Orin 32GB, 10 clips of Sugr-ASR-Bench, 234 min total audio:

| Metric | Value |
|---|---|
| Avg WER | **7.25%** |
| WER range | 4.76% – 9.16% (tight band) |
| Avg RTF | 0.025 (40× real-time) |

Per-clip results: `/Users/seansong/seanslab/Research/asrbench/results/sugr_new_20260416/parakeet.json`

### Sugr-ASR-Bench corpus fixed (DONE, committed as `4c61d89` on `seanslab-org/Sugr-ASR-Bench`)

5 of 10 clips had mismatched audio/reference (audio trimmed to 600/1200s, and
paired with transcripts from wrong episodes). Re-downloaded correct full-length
MP3s from NPR's play.podtrac.com, updated `duration_sec` in `meta.json`.
Originals backed up as `audio.opus.bak` (gitignored).

Fix script: `Sugr-ASR-Bench/scripts/fix_mismatched_audio.py`.

### Data staged on Mac mini

`testmac:/Users/seanslab/seanslab/asrbench/ios/Resources/audio_sugr/`
- 10 × `npr_*.m4a` (10-44 min each, AAC 64 kbps, 16 kHz mono)
- 10 × `npr_*.txt` (reference transcripts)

Transfer over Tailscale was very slow (~50 KB/s observed). May not be 100%
complete when this doc was written — verify with `ls | wc -l` ≥ 20.

## What Sean needs to do on Mac mini (estimated 30-45 min)

### 1. Verify file transfer

```bash
ssh testmac "ls /Users/seanslab/seanslab/asrbench/ios/Resources/audio_sugr/ | wc -l"
# Expect 20 (10 m4a + 10 txt). If less, wait for the scp to finish or
# re-run from Mac: scp -C /Users/seansong/seanslab/Research/asrbench/data/sugr_bench/m4a/*.m4a \
#   /Users/seansong/seanslab/Research/asrbench/data/sugr_bench/*.txt \
#   testmac:/Users/seanslab/seanslab/asrbench/ios/Resources/audio_sugr/
```

### 2. Install xcodegen on Mac mini

```bash
ssh testmac
# on Mac mini:
cd /tmp && rm -rf XcodeGen
git clone --depth 1 https://github.com/yonaskolb/XcodeGen
cd XcodeGen && swift build -c release
sudo cp .build/release/xcodegen /usr/local/bin/
xcodegen --version
```

(Sean's password needed for `sudo`.)

### 3. Generate `manifest_sugr.json`

```bash
# still on Mac mini
python3 <<'PY'
import json
clips = ['npr_fa_b5','npr_pol_b13','npr_pol_b17','npr_rt8','npr_ted7',
        'npr_1a_b3','npr_1a_ep20','npr_politics_ep16','npr_politics_ep18','npr_politics_ep22']
entries = [{'sampleId': c, 'filename': c + '.m4a', 'language': 'en'} for c in clips]
with open('/Users/seanslab/seanslab/asrbench/ios/Resources/audio_sugr/manifest.json','w') as f:
    json.dump(entries, f, indent=2)
print(f'Wrote {len(entries)} entries')
PY
```

### 4. Patch `BenchRunner.swift` for env-var override

Edit `/Users/seanslab/seanslab/asrbench/ios/Sources/BenchRunner.swift`.

Find `loadManifest()`:
```swift
private func loadManifest() throws -> [ManifestEntry] {
    guard let url = Bundle.main.url(forResource: "audio/manifest", withExtension: "json") else {
```

Replace with:
```swift
private func loadManifest() throws -> [ManifestEntry] {
    let dir = ProcessInfo.processInfo.environment["ASR_AUDIO_DIR"] ?? "audio"
    guard let url = Bundle.main.url(forResource: "\(dir)/manifest", withExtension: "json") else {
```

Find `audioURL(for:)`:
```swift
private func audioURL(for filename: String) -> URL {
    return Bundle.main.bundleURL
        .appendingPathComponent("audio")
        .appendingPathComponent(filename)
}
```

Replace with:
```swift
private func audioURL(for filename: String) -> URL {
    let dir = ProcessInfo.processInfo.environment["ASR_AUDIO_DIR"] ?? "audio"
    return Bundle.main.bundleURL
        .appendingPathComponent(dir)
        .appendingPathComponent(filename)
}
```

### 5. Add `audio_sugr` folder to `project.yml`

Edit `/Users/seanslab/seanslab/asrbench/ios/project.yml`, in the `sources:` list under `AsrBenchIOS`:

```yaml
    sources:
      - path: Sources
      - path: Resources/audio
        type: folder
        buildPhase: resources
      - path: Resources/audio_sugr
        type: folder
        buildPhase: resources
      - path: Resources/moonshine-models
        type: folder
        buildPhase: resources
        optional: true
```

### 6. Regenerate + build

```bash
cd /Users/seanslab/seanslab/asrbench/ios
xcodegen
xcodebuild -project AsrBenchIOS.xcodeproj -scheme AsrBenchIOS \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  -derivedDataPath build clean build 2>&1 | tail -30
```

### 7. Run on iOS Simulator with env override

```bash
# boot simulator
xcrun simctl boot 'iPhone 16' 2>/dev/null || true

# install app
APP=$(find /Users/seanslab/seanslab/asrbench/ios/build/Build/Products -name 'AsrBenchIOS.app' | head -1)
xcrun simctl install booted "$APP"

# launch with env overrides
xcrun simctl launch --console --terminate-running-process \
  -e ASR_AUDIO_DIR audio_sugr \
  booted ai.moonshine.asrbench.AsrBenchIOS
```

Watch the log. For 10 clips × 3 runners (apple-speech, moonshine-tiny-en,
moonshine-base-en) on simulator CPU: roughly 20-40 min total.

### 8. Pull results back to Mac

```bash
DATA=$(xcrun simctl get_app_container booted ai.moonshine.asrbench.AsrBenchIOS data)
ls "$DATA/Documents/"  # look for results-<timestamp>.json
scp "testmac:$DATA/Documents/results-*.json" \
  /Users/seansong/seanslab/Research/asrbench/results/sugr_new_20260416/moonshine_simulator.json
```

### 9. Compute WER + LLM quality judge + HTML

From Mac:
```bash
cd /Users/seansong/seanslab/Research/asrbench
# (a) Compute per-clip WER with jiwer + EnglishTextNormalizer:
python3 tests/compute_moonshine_wer.py \
  --results results/sugr_new_20260416/moonshine_simulator.json \
  --refs data/sugr_bench/ \
  --out results/sugr_new_20260416/moonshine_wer.json

# (b) LLM quality judge (needs ANTHROPIC_API_KEY):
python3 tests/asr_quality_judge.py \
  --ref data/sugr_bench \
  --a results/sugr_new_20260416/parakeet.json --a-name "Parakeet-TDT-1.1B (Orin)" \
  --b results/sugr_new_20260416/moonshine_wer.json --b-name "Moonshine-base-en (iPad Sim)" \
  --out results/sugr_new_20260416/quality_judge.json

# (c) HTML comparison:
python3 /Users/seansong/seanslab/Research/SummaryBench/gen_compare.py \
  --mode asr \
  --a results/sugr_new_20260416/parakeet.json --a-name "Parakeet-TDT-1.1B" \
  --b results/sugr_new_20260416/moonshine_wer.json --b-name "Moonshine-base-en" \
  --judge results/sugr_new_20260416/quality_judge.json \
  -o results/sugr_new_20260416/comparison.html --open
```

Note: `compute_moonshine_wer.py`, `asr_quality_judge.py`, and `gen_compare.py
--mode asr` don't exist yet — they'll need to be written. Template them from
existing `tests/sugr_bench_top3.py` (WER logic) and `SummaryBench/round19.py`
(judge logic).

### 10. Real iPad numbers (optional, if Sean has iPad mini 6 + cable)

Same as step 7 but target the physical device:

```bash
xcrun devicectl device process launch \
  --terminate-existing \
  --device <IPAD_UDID> \
  --environment-variables ASR_AUDIO_DIR=audio_sugr \
  ai.moonshine.asrbench.AsrBenchIOS
```

Real-device RTF is the authoritative A15 Bionic number. Simulator RTF is Mac
CPU and not comparable to iPhone hardware.

## Why this hand-off

The Tailscale link from my remote Mac to testmac was transferring at
~50 KB/s — a 10-file m4a set (~90 MB) would take 30+ min each attempt, and
multiple retries consumed the session budget. Running these steps locally on
Mac mini (or from Sean's Mac plugged into the same LAN subnet) should be
minutes, not hours.

## Research expectations (so we know what to look for)

Based on the earlier iPad bench (April 9, LS-clean):
- `moonshine-base-en` on LS-clean: 2.45% WER — very short clean utterances.
- `moonshine-tiny-en` on LS-clean: 4.69% WER.
- `apple-speech-en-US` on LS-clean: 10.92% WER.

Expectation on Sugr-ASR-Bench (long-form NPR):
- Moonshine-base-en: likely **5-10% WER** (lower than Parakeet's 7.25% if Moonshine
  handles NPR better, higher if chunk stitching introduces errors).
- Apple Speech: likely **15-25% WER** (long-form is its weak point).

The LLM quality judge (step 9b) will catch the *qualitative* differences WER
misses — especially named-entity accuracy (NPR speaker names like Tamara Keith,
Ashley Lopez) and punctuation (Moonshine outputs lowercase, no punctuation; a
summarizer will struggle with that).
