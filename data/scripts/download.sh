#!/bin/bash
# ASR Bench — Download real-world business/meeting audio
# Uses yt-dlp search to find actual videos, plus verified sources
# Output: 16kHz mono WAV files in data/{en,zh,ja}/{short,long}/

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$DATA_DIR"/{en,zh,ja}/{short,long}

# Helper: search YouTube, download first result, convert to 16kHz mono WAV
# Usage: dl_search "SEARCH_QUERY" OUTPUT_PATH [START_TIME] [DURATION]
dl_search() {
    local query="$1"
    local out="$2"
    local start="${3:-}"
    local duration="${4:-}"

    if [ -f "$out" ]; then
        echo "  SKIP (exists): $(basename "$out")"
        return 0
    fi

    echo "  Searching: $query"
    local tmpfile=$(mktemp /tmp/asrbench_XXXXX)

    # Search YouTube and download first result audio
    if ! yt-dlp --cookies-from-browser chrome "ytsearch1:$query" -f 'ba' --no-playlist \
         -o "$tmpfile.%(ext)s" --no-warnings 2>/dev/null; then
        echo "  FAIL: could not download for query: $query"
        rm -f "$tmpfile"*
        return 1
    fi

    # Find the downloaded file
    local dlfile=$(ls "$tmpfile".* 2>/dev/null | head -1)
    if [ -z "$dlfile" ]; then
        echo "  FAIL: no file downloaded for: $query"
        return 1
    fi

    local ffargs=(-y -i "$dlfile")
    [ -n "$start" ] && ffargs+=(-ss "$start")
    [ -n "$duration" ] && ffargs+=(-t "$duration")
    ffargs+=(-ar 16000 -ac 1 -acodec pcm_s16le "$out")

    ffmpeg "${ffargs[@]}" 2>/dev/null
    rm -f "$tmpfile"*
    echo "  DONE: $(basename "$out") ($(du -h "$out" | cut -f1))"
}

# Helper: download from direct URL
# Usage: dl_url URL OUTPUT_PATH [START_TIME] [DURATION]
dl_url() {
    local url="$1"
    local out="$2"
    local start="${3:-}"
    local duration="${4:-}"

    if [ -f "$out" ]; then
        echo "  SKIP (exists): $(basename "$out")"
        return 0
    fi

    echo "  Downloading: $url"
    local tmpfile=$(mktemp /tmp/asrbench_XXXXX)

    if ! yt-dlp --cookies-from-browser chrome "$url" -f 'ba' --no-playlist \
         -o "$tmpfile.%(ext)s" --no-warnings 2>/dev/null; then
        echo "  FAIL: $url"
        rm -f "$tmpfile"*
        return 1
    fi

    local dlfile=$(ls "$tmpfile".* 2>/dev/null | head -1)
    if [ -z "$dlfile" ]; then
        echo "  FAIL: no file for $url"
        return 1
    fi

    local ffargs=(-y -i "$dlfile")
    [ -n "$start" ] && ffargs+=(-ss "$start")
    [ -n "$duration" ] && ffargs+=(-t "$duration")
    ffargs+=(-ar 16000 -ac 1 -acodec pcm_s16le "$out")

    ffmpeg "${ffargs[@]}" 2>/dev/null
    rm -f "$tmpfile"*
    echo "  DONE: $(basename "$out") ($(du -h "$out" | cut -f1))"
}


echo "============================================"
echo " ASR Bench — Downloading test audio"
echo "============================================"

# =====================================================================
# ENGLISH
# =====================================================================
echo ""
echo "=== ENGLISH ==="

echo "--- EN Short clips (30s) ---"

dl_search "Apple Q4 2025 earnings call full" \
    "$DATA_DIR/en/short/apple_earnings_01.wav" "00:02:00" "30"

dl_search "Tesla earnings call 2025 Elon Musk" \
    "$DATA_DIR/en/short/tesla_earnings_01.wav" "00:03:00" "30"

dl_search "Federal Reserve Jerome Powell press conference 2025 FOMC" \
    "$DATA_DIR/en/short/fed_pressconf_01.wav" "00:05:00" "30"

dl_search "US Senate hearing artificial intelligence 2025" \
    "$DATA_DIR/en/short/senate_hearing_01.wav" "00:10:00" "30"

dl_search "city council meeting 2025 full audio" \
    "$DATA_DIR/en/short/city_council_noisy_01.wav" "00:15:00" "30"

echo "--- EN Long clips ---"

dl_search "Apple quarterly earnings call 2025 full" \
    "$DATA_DIR/en/long/apple_earnings_15min.wav" "00:01:00" "900"

dl_search "Federal Reserve press conference 2025 full Jerome Powell FOMC" \
    "$DATA_DIR/en/long/fed_pressconf_30min.wav" "00:02:00" "1800"

dl_search "US Senate committee hearing 2025 full" \
    "$DATA_DIR/en/long/senate_hearing_60min.wav" "00:05:00" "3600"


# =====================================================================
# CHINESE (Mandarin)
# =====================================================================
echo ""
echo "=== CHINESE ==="

echo "--- ZH Short clips (30s) ---"

dl_search "外交部发言人记者会 2025 完整版" \
    "$DATA_DIR/zh/short/mofa_pressconf_01.wav" "00:02:00" "30"

dl_search "国新办新闻发布会 2025" \
    "$DATA_DIR/zh/short/scio_pressconf_01.wav" "00:05:00" "30"

dl_search "阿里巴巴 财报电话会 2025 业绩说明" \
    "$DATA_DIR/zh/short/alibaba_earnings_01.wav" "00:03:00" "30"

dl_search "华为发布会 2025 完整" \
    "$DATA_DIR/zh/short/huawei_launch_01.wav" "00:05:00" "30"

dl_search "中国企业 会议录音 讨论" \
    "$DATA_DIR/zh/short/meeting_noisy_01.wav" "00:10:00" "30"

echo "--- ZH Long clips ---"

dl_search "国新办新闻发布会 2025 完整版 全程" \
    "$DATA_DIR/zh/long/scio_pressconf_15min.wav" "00:02:00" "900"

dl_search "外交部发言人记者会 2025 完整 全程 回放" \
    "$DATA_DIR/zh/long/mofa_pressconf_30min.wav" "00:01:00" "1800"

dl_search "中国上市公司 业绩说明会 2025 完整" \
    "$DATA_DIR/zh/long/earnings_meeting_60min.wav" "00:01:00" "3600"


# =====================================================================
# JAPANESE
# =====================================================================
echo ""
echo "=== JAPANESE ==="

echo "--- JA Short clips (30s) ---"

dl_search "ソフトバンクグループ 決算説明会 2025 孫正義" \
    "$DATA_DIR/ja/short/softbank_earnings_01.wav" "00:03:00" "30"

dl_search "トヨタ自動車 決算説明会 2025" \
    "$DATA_DIR/ja/short/toyota_earnings_01.wav" "00:05:00" "30"

dl_search "官房長官 記者会見 2025 全編" \
    "$DATA_DIR/ja/short/cabinet_pressconf_01.wav" "00:01:00" "30"

dl_search "日銀総裁 記者会見 2025 植田" \
    "$DATA_DIR/ja/short/boj_pressconf_01.wav" "00:03:00" "30"

dl_search "日本企業 株主総会 2025" \
    "$DATA_DIR/ja/short/shareholder_meeting_01.wav" "00:10:00" "30"

echo "--- JA Long clips ---"

dl_search "ソフトバンクグループ 決算説明会 2025 全編 フル" \
    "$DATA_DIR/ja/long/softbank_earnings_15min.wav" "00:02:00" "900"

dl_search "官房長官 記者会見 2025 全編 フル" \
    "$DATA_DIR/ja/long/cabinet_pressconf_30min.wav" "00:01:00" "1800"

dl_search "トヨタ 決算説明会 2025 全編 フル" \
    "$DATA_DIR/ja/long/toyota_earnings_60min.wav" "00:01:00" "3600"


echo ""
echo "============================================"
echo " Download complete!"
echo "============================================"
echo ""
echo "Summary:"
for lang in en zh ja; do
    for len in short long; do
        count=$(ls "$DATA_DIR/$lang/$len/"*.wav 2>/dev/null | wc -l)
        size=$(du -sh "$DATA_DIR/$lang/$len/" 2>/dev/null | cut -f1)
        echo "  $lang/$len: $count files ($size)"
    done
done
echo ""
echo "Total:"
du -sh "$DATA_DIR" 2>/dev/null
