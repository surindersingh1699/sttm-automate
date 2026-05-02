"""Prototype: collapsed-phoneme encoder for Gurmukhi text.

Maps Gurmukhi (or Devanagari, after offset) to a small phoneme alphabet that
collapses the substitutions Whisper makes on sung kirtan:
  ਬ↔ਵ, ਸ↔ਸ਼, ਜ↔ਜ਼, ਫ↔ਫ਼, nasal markers, matras to 3 vowel buckets.

The output is a short ASCII string (lowercase a-z + space) — works as input
to char-n-gram inverted indexes or BK-trees.
"""
from __future__ import annotations

from src.transcription.transliterate import normalize_for_fullword_search


# Canonical phoneme buckets for each Gurmukhi consonant/vowel.
# Confusable pairs map to the same letter.
_PHONEME = {
    # vowels (independent + carriers)
    "ਅ": "a", "ਆ": "a",
    "ਇ": "i", "ਈ": "i", "ਏ": "i", "ਐ": "i", "ੲ": "i",
    "ਉ": "u", "ਊ": "u", "ਔ": "u", "ੳ": "u",
    "ੴ": "i",
    # consonants
    "ਕ": "k", "ਖ": "k", "ਖ਼": "k",
    "ਗ": "g", "ਘ": "g", "ਗ਼": "g",
    "ਙ": "n",
    "ਚ": "c",
    "ਛ": "c",
    "ਜ": "j", "ਝ": "j", "ਜ਼": "j",
    "ਞ": "n",
    "ਟ": "t", "ਠ": "t",
    "ਡ": "d", "ਢ": "d",
    "ਣ": "n",
    "ਤ": "t", "ਥ": "t",
    "ਦ": "d", "ਧ": "d",
    "ਨ": "n",
    "ਪ": "p",
    "ਫ": "f", "ਫ਼": "f",
    "ਬ": "v", "ਵ": "v",   # b/v collapse — confused in singing
    "ਭ": "v",
    "ਮ": "m",
    "ਯ": "y",
    "ਰ": "r", "ੜ": "r",
    "ਲ": "l", "ਲ਼": "l",
    "ਸ": "s", "ਸ਼": "s",
    "ਹ": "h",
}

# Matras (vowel marks). All collapse into one of {a, i, u} to mirror sung
# vowels which are routinely lengthened or shortened.
_MATRA = {
    "ਾ": "a",
    "ਿ": "i", "ੀ": "i", "ੇ": "i", "ੈ": "i",
    "ੁ": "u", "ੂ": "u", "ੋ": "u", "ੌ": "u",
}

# Diacritics we drop entirely (halant kills inherent vowel; nasals/visarga
# add no class info we can rely on from Whisper output).
_DROP = {"ੰ", "ਂ", "ਃ", "੍", "ੱ", "਼"}


def encode(text: str) -> str:
    """Encode Gurmukhi/Devanagari text → collapsed phoneme string.

    Whitespace is preserved between words.
    """
    if not text:
        return ""
    text = normalize_for_fullword_search(text)
    out: list[str] = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
            continue
        if ch in _DROP:
            continue
        if ch in _PHONEME:
            out.append(_PHONEME[ch])
            continue
        if ch in _MATRA:
            out.append(_MATRA[ch])
            continue
        # Anything else: skip (punctuation, ASCII, etc.)
    s = "".join(out)
    return " ".join(s.split())


def char_ngrams(s: str, n: int = 3) -> set[str]:
    """Char n-grams over the phoneme string. Word boundaries kept."""
    if not s:
        return set()
    grams: set[str] = set()
    for word in ("_" + s + "_").split():
        if len(word) < n:
            grams.add(word)
            continue
        for i in range(len(word) - n + 1):
            grams.add(word[i : i + n])
    return grams


def overlap_score(q_grams: set[str], cand_grams: set[str]) -> float:
    """Overlap coefficient: |q ∩ c| / min(|q|, |c|). Range [0, 1]."""
    if not q_grams or not cand_grams:
        return 0.0
    inter = len(q_grams & cand_grams)
    return inter / min(len(q_grams), len(cand_grams))
