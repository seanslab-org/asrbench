"""WER/CER computation with text normalization."""
import re
import unicodedata


def normalize_text(text: str, language: str = "en") -> str:
    """Normalize text for fair WER/CER comparison."""
    # Unicode NFKC normalization (important for CJK)
    text = unicodedata.normalize("NFKC", text)
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep CJK characters)
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_wer(reference: str, hypothesis: str, language: str = "en") -> dict:
    """Compute Word Error Rate."""
    from jiwer import wer, mer, wil
    ref = normalize_text(reference, language)
    hyp = normalize_text(hypothesis, language)
    if not ref:
        return {"wer": 1.0, "mer": 1.0, "wil": 1.0}
    return {
        "wer": wer(ref, hyp),
        "mer": mer(ref, hyp),
        "wil": wil(ref, hyp),
    }


def compute_cer(reference: str, hypothesis: str, language: str = "zh") -> dict:
    """Compute Character Error Rate (for CJK languages)."""
    from jiwer import cer
    ref = normalize_text(reference, language)
    hyp = normalize_text(hypothesis, language)
    # For CER, split into characters
    ref_chars = " ".join(list(ref.replace(" ", "")))
    hyp_chars = " ".join(list(hyp.replace(" ", "")))
    if not ref_chars:
        return {"cer": 1.0}
    return {"cer": cer(ref_chars, hyp_chars)}


def compute_accuracy(reference: str, hypothesis: str, language: str) -> dict:
    """Compute appropriate accuracy metric based on language."""
    if language == "en":
        return compute_wer(reference, hypothesis, language)
    else:
        result = compute_cer(reference, hypothesis, language)
        result.update(compute_wer(reference, hypothesis, language))
        return result
