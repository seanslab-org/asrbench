#!/usr/bin/env python3
"""Unit test for CER computation — run before any ZH/JA benchmark."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.wer import compute_cer

def test_cer_single_sub():
    """1 substitution in 13 chars = 7.69% CER."""
    ref = "甚至出现交易几乎停滞的情况"
    hyp = "甚至出现交易几乎停止的情况"
    result = compute_cer(ref, hyp, "zh")
    expected = 1 / 13
    assert abs(result["cer"] - expected) < 0.01, f"CER {result['cer']:.4f} != {expected:.4f}"
    print(f"PASS: 1 sub/13 chars → CER={result['cer']:.4f} (expected {expected:.4f})")

def test_cer_perfect():
    """Identical strings = 0% CER."""
    ref = "一二线城市虽然也处于调整中"
    result = compute_cer(ref, ref, "zh")
    assert result["cer"] == 0.0, f"CER should be 0, got {result['cer']}"
    print(f"PASS: identical strings → CER=0.0")

def test_cer_all_wrong():
    """Completely different strings = 100% CER."""
    ref = "甲乙丙"
    hyp = "丁戊己"
    result = compute_cer(ref, hyp, "zh")
    assert result["cer"] == 1.0, f"CER should be 1.0, got {result['cer']}"
    print(f"PASS: all wrong → CER=1.0")

def test_cer_insertion():
    """1 insertion in 5 chars = 20% CER."""
    ref = "你好世界啊"
    hyp = "你好大世界啊"  # inserted 大
    result = compute_cer(ref, hyp, "zh")
    expected = 1 / 5  # 1 insertion / 5 ref chars
    assert abs(result["cer"] - expected) < 0.01, f"CER {result['cer']:.4f} != {expected:.4f}"
    print(f"PASS: 1 insertion/5 chars → CER={result['cer']:.4f} (expected {expected:.4f})")

def test_cer_with_punctuation():
    """Punctuation in hypothesis should be stripped by normalization."""
    ref = "甚至出现交易几乎停滞的情况"
    hyp = "甚至出现交易，几乎停滞的情况。"  # added Chinese punctuation
    result = compute_cer(ref, hyp, "zh")
    assert result["cer"] == 0.0, f"CER should be 0 after stripping punctuation, got {result['cer']}"
    print(f"PASS: punctuation stripped → CER=0.0")

if __name__ == "__main__":
    print("=== CER Unit Tests ===\n")
    test_cer_single_sub()
    test_cer_perfect()
    test_cer_all_wrong()
    test_cer_insertion()
    test_cer_with_punctuation()
    print("\nALL CER TESTS PASSED")
