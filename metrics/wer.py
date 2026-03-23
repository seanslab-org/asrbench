"""WER/CER computation with proper text normalization."""
from metrics.normalize import normalize


def compute_wer(reference: str, hypothesis: str, language: str = "en") -> dict:
    """Compute Word Error Rate (primary metric for English)."""
    from jiwer import wer
    ref = normalize(reference, language)
    hyp = normalize(hypothesis, language)
    if not ref:
        return {"wer": 1.0}
    return {"wer": wer(ref, hyp)}


def compute_cer(reference: str, hypothesis: str, language: str = "zh") -> dict:
    """Compute Character Error Rate (primary metric for ZH/JA).

    Uses jiwer.cer directly on normalized strings — it handles
    character-level edit distance internally. Do NOT space-separate
    and then call cer(), as spaces get counted as characters.
    """
    from jiwer import cer
    ref = normalize(reference, language)
    hyp = normalize(hypothesis, language)
    if not ref:
        return {"cer": 1.0}
    return {"cer": cer(ref, hyp)}


def compute_accuracy(reference: str, hypothesis: str, language: str) -> dict:
    """Compute appropriate accuracy metric based on language."""
    if language == "en":
        return compute_wer(reference, hypothesis, language)
    else:
        # For ZH/JA: CER is primary, also compute WER for reference
        result = compute_cer(reference, hypothesis, language)
        # Also compute WER on the original (lightly normalized) text
        from jiwer import wer
        from metrics.normalize import normalize_en
        ref_words = normalize_en(reference)
        hyp_words = normalize_en(hypothesis)
        if ref_words:
            result["wer"] = wer(ref_words, hyp_words)
        return result
