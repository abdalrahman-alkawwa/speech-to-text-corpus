"""
tests/test_normalizer.py
Test suite for Stage 8: Deterministic Arabic Text Normalization
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.normalizer import normalize_arabic


def test_arabic_normalization():
    print("==============================================")
    print(" Running Stage 8: Arabic Normalization Tests  ")
    print("==============================================\n")

    # Test 1: Diacritics (tashkeel) removal
    t1 = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    res1 = normalize_arabic(t1)
    print(f"[TEST 1] Tashkeel removal:\n  IN:  {t1}\n  OUT: {res1}")
    assert "ِ" not in res1 and "ّ" not in res1 and "ْ" not in res1
    print("  -> PASSED\n")

    # Test 2: Alef variants unification
    t2 = "إبراهيم وأحمد وآية وٱمرأة"
    res2 = normalize_arabic(t2)
    print(f"[TEST 2] Alef unification:\n  IN:  {t2}\n  OUT: {res2}")
    assert "إ" not in res2 and "أ" not in res2 and "آ" not in res2 and "ٱ" not in res2
    print("  -> PASSED\n")

    # Test 3: Taa Marbuta and Alef Maqsura
    t3 = "مدرسة جميلة تذهب إلى ليلى"
    res3 = normalize_arabic(t3)
    print(f"[TEST 3] Taa Marbuta & Alef Maqsura:\n  IN:  {t3}\n  OUT: {res3}")
    assert "ة" not in res3 and "ى" not in res3
    print("  -> PASSED\n")

    # Test 4: Tatweel & Punctuation removal
    t4 = "هــــذا درس، هل فهمت؟ نعم!"
    res4 = normalize_arabic(t4)
    print(f"[TEST 4] Tatweel and Punctuation removal:\n  IN:  {t4}\n  OUT: {res4}")
    assert "ـ" not in res4 and "،" not in res4 and "؟" not in res4 and "!" not in res4
    print("  -> PASSED\n")

    # Test 5: Empty and None handling
    assert normalize_arabic("") == ""
    assert normalize_arabic(None) == ""
    print("[TEST 5] Edge cases (empty/None):\n  -> PASSED\n")

    print("==============================================")
    print(" ALL ARABIC NORMALIZATION TESTS PASSED!       ")
    print("==============================================")


if __name__ == "__main__":
    test_arabic_normalization()
