#!/usr/bin/env bash
# One-shot setup on Mac mini after Xcode is installed.
# Run this from within the asrbench/ios/ directory on the mini.
#
# Idempotent — safe to re-run.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> verify Xcode"
if ! command -v xcodebuild >/dev/null; then
    echo "ERROR: xcodebuild not found. Install full Xcode first." >&2
    exit 1
fi
xcodebuild -version

echo "==> verify simulator runtime"
xcrun simctl list runtimes | head -10
if ! xcrun simctl list devices available | grep -q "iPhone"; then
    echo "ERROR: no iPhone simulators available. Open Xcode once or install runtime." >&2
    exit 1
fi

echo "==> install xcodegen (project file generator)"
if ! command -v xcodegen >/dev/null; then
    if ! command -v brew >/dev/null; then
        echo "ERROR: Homebrew not installed. Install from https://brew.sh first." >&2
        exit 1
    fi
    brew install xcodegen
fi
xcodegen --version

echo "==> install moonshine-voice (model downloader)"
python3 -m pip install --user --upgrade moonshine-voice

echo "==> download Moonshine en models"
# Downloads to ~/Library/Caches/moonshine_voice/ by default
python3 -m moonshine_voice.download --language en

CACHE_DIR="${HOME}/Library/Caches/moonshine_voice"
if [[ ! -d "${CACHE_DIR}" ]]; then
    echo "ERROR: moonshine_voice cache not found at ${CACHE_DIR}" >&2
    exit 1
fi

echo "==> stage model files into Resources/moonshine-models/"
mkdir -p Resources/moonshine-models/tiny-en Resources/moonshine-models/base-en
# Expected files per arch: encoder_model.ort, decoder_model_merged.ort, tokenizer.bin
# Exact subdirectory names inside the cache vary by version — adjust if download
# layout differs.
for arch in tiny base; do
    src_candidates=(
        "${CACHE_DIR}/en-${arch}"
        "${CACHE_DIR}/moonshine-${arch}-en"
        "${CACHE_DIR}/${arch}-en"
    )
    found=""
    for candidate in "${src_candidates[@]}"; do
        if [[ -d "$candidate" ]]; then
            found="$candidate"
            break
        fi
    done
    if [[ -z "$found" ]]; then
        echo "WARN: could not auto-locate ${arch}-en model dir under ${CACHE_DIR}" >&2
        echo "      contents are:" >&2
        ls -la "${CACHE_DIR}" >&2
        echo "      copy the files manually to Resources/moonshine-models/${arch}-en/" >&2
        continue
    fi
    cp -R "$found"/*.ort "Resources/moonshine-models/${arch}-en/"
    cp -R "$found"/*.bin "Resources/moonshine-models/${arch}-en/" 2>/dev/null || true
    echo "  staged ${arch}-en from $found"
    ls "Resources/moonshine-models/${arch}-en/"
done

echo "==> generate Xcode project"
xcodegen generate
echo "  generated AsrBenchIOS.xcodeproj"

echo "==> build for iPhone 17 Pro simulator"
xcodebuild \
    -project AsrBenchIOS.xcodeproj \
    -scheme AsrBenchIOS \
    -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
    -configuration Debug \
    build \
    | tail -30

echo
echo "==> setup complete"
echo
echo "Next steps:"
echo "  DEV=\$(xcrun simctl list devices available | grep 'iPhone 17 Pro (' | head -1 | sed -E 's/.*\\(([0-9A-F-]+)\\).*/\\1/')"
echo "  xcrun simctl boot \"\$DEV\" 2>/dev/null || true"
echo "  xcrun simctl bootstatus \"\$DEV\" -b"
echo "  xcrun simctl install \"\$DEV\" ./build/Build/Products/Debug-iphonesimulator/AsrBenchIOS.app"
echo
echo "  # Pre-grant Speech Recognition permission via TCC.db (simctl privacy"
echo "  # cannot grant 'speech-recognition' on Xcode 26 — returns Operation not"
echo "  # permitted). Without this the app blocks on the auth dialog forever."
echo "  TCC=~/Library/Developer/CoreSimulator/Devices/\$DEV/data/Library/TCC/TCC.db"
echo "  sqlite3 \"\$TCC\" \"INSERT OR REPLACE INTO access (service, client, client_type, auth_value, auth_reason, auth_version) VALUES ('kTCCServiceSpeechRecognition', 'ai.moonshine.asrbench.AsrBenchIOS', 0, 2, 4, 1);\""
echo
echo "  xcrun simctl launch \"\$DEV\" ai.moonshine.asrbench.AsrBenchIOS"
echo "  sleep 30   # bench takes ~25s for 50 samples × 3 runners on Mac CPU"
echo "  CONTAINER=\$(xcrun simctl get_app_container \"\$DEV\" ai.moonshine.asrbench.AsrBenchIOS data)"
echo "  cp \"\$CONTAINER/Documents/\"results-*.json /tmp/"
echo "  python3 scripts/compute_metrics.py /tmp/results-*.json"
echo
echo "Note: Apple Speech will fail on the simulator with kLSRErrorDomain Code=300"
echo "      because the simulator does not ship Siri offline models. Phase 2"
echo "      (real iPhone) is required for the Apple Speech comparison."
