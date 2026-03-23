"""Text normalization for ASR evaluation.

CRITICAL: All model outputs AND reference texts must pass through the same
normalizer before WER/CER computation. This is the single biggest source of
bogus results in ASR benchmarking.

Reference: Whisper's normalizers are the de facto standard for EN.
For ZH/JA, we strip to CJK characters only for CER.
"""
import re
import unicodedata


def normalize_en(text: str) -> str:
    """Normalize English text for WER computation.

    Uses Whisper's EnglishTextNormalizer if available (handles contractions,
    number words, etc.), falls back to basic normalization.
    """
    try:
        from whisper.normalizers import EnglishTextNormalizer
        normalizer = EnglishTextNormalizer()
        return normalizer(text)
    except ImportError:
        pass
    # Fallback: basic normalization
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Remove punctuation but keep alphanumeric and spaces
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_zh(text: str) -> str:
    """Normalize Chinese text for CER computation.

    Keeps only CJK Unified Ideographs. Strips all punctuation, spaces,
    Latin characters, and numbers. CER is computed character-by-character.
    """
    text = unicodedata.normalize("NFKC", text)
    # Keep CJK Unified Ideographs (basic + extension A)
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf]", "", text)
    return text


def normalize_ja(text: str) -> str:
    """Normalize Japanese text for CER computation.

    Keeps hiragana, katakana, and CJK kanji. Strips punctuation, spaces,
    Latin characters, and numbers.
    """
    text = unicodedata.normalize("NFKC", text)
    # Keep: hiragana + katakana + CJK kanji
    text = re.sub(
        r"[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf]", "", text
    )
    return text


def normalize(text: str, language: str) -> str:
    """Normalize text based on language."""
    if language == "en":
        return normalize_en(text)
    elif language == "zh":
        return normalize_zh(text)
    elif language == "ja":
        return normalize_ja(text)
    else:
        # Generic fallback
        return normalize_en(text)
