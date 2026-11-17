"""
src/normalizer.py
=================
Stage 8: Deterministic Arabic Text Normalization

Applies a fixed sequence of regex transformations to normalize Arabic text
for fair WER/CER comparison between ASR engines. All rules are deterministic
and reversible by design — no machine learning, no context sensitivity.

Normalization steps:
  1. Strip all Arabic diacritics (tashkeel).
  2. Remove tatweel / kashida (ـ).
  3. Unify Alef variants (إ أ آ ٱ → ا).
  4. Unify Taa Marbuta (ة → ه).
  5. Unify Alef Maqsura (ى → ي).
  6. Remove non-Arabic punctuation and symbols.
  7. Collapse whitespace.
"""

import re
from typing import Optional

# Arabic diacritics (tashkeel) Unicode range
_DIACRITICS_RE = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u0674]')

# Tatweel / kashida character
_TATWEEL_RE = re.compile(r'\u0640')

# Alef variants → plain Alef
_ALEF_VARIANTS = re.compile(r'[إأآٱ]')

# Taa Marbuta → Haa
_TAA_MARBUTA = re.compile(r'ة')

# Alef Maqsura → Yaa
_ALEF_MAQSURA = re.compile(r'ى')

# Non-Arabic / non-whitespace noise: keep only Arabic letters, digits, whitespace
_NON_ARABIC_RE = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\d]')

# Arabic punctuation (comma ، semicolon ؛ question mark ؟ etc.)
_ARABIC_PUNCT_RE = re.compile(r'[\u060C\u061B\u061F\u066B\u066C\u066D\u06D4]')

# Multiple whitespace
_MULTI_SPACE = re.compile(r'\s+')


def normalize_arabic(text: Optional[str]) -> str:
    """
    Applies the full normalization pipeline to an Arabic text string.
    Returns cleaned, normalized text suitable for WER/CER comparison.
    """
    if not text:
        return ""

    t = text

    # 1. Strip diacritics
    t = _DIACRITICS_RE.sub('', t)

    # 2. Remove tatweel
    t = _TATWEEL_RE.sub('', t)

    # 3. Unify Alef variants
    t = _ALEF_VARIANTS.sub('ا', t)

    # 4. Taa Marbuta → Haa
    t = _TAA_MARBUTA.sub('ه', t)

    # 5. Alef Maqsura → Yaa
    t = _ALEF_MAQSURA.sub('ي', t)

    # 6. Remove Arabic punctuation
    t = _ARABIC_PUNCT_RE.sub(' ', t)

    # 7. Remove non-Arabic characters
    t = _NON_ARABIC_RE.sub(' ', t)

    # 7. Collapse whitespace
    t = _MULTI_SPACE.sub(' ', t).strip()

    return t


if __name__ == "__main__":
    test_cases = [
        "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
        "أحمد ذهب إلى المدرسة",
        "هـــــذا نـــص طويـــل",
        "الحمد لله، والصلاة والسلام على رسول الله!",
        "ذَهَبَ أَحْمَدُ إِلَى ٱلْمَدْرَسَةِ",
    ]

    print("Arabic Text Normalization Test:")
    print("=" * 60)
    for case in test_cases:
        normalized = normalize_arabic(case)
        print(f"  IN:  {case}")
        print(f"  OUT: {normalized}")
        print()
