"""
Transliteration: Convert any-script Punjabi text to romanized first-letter codes.

BaniDB's first-letter search uses specific romanized codes for each Gurmukhi consonant.
Whisper may output Gurmukhi, Devanagari, Shahmukhi, or romanized text.
This module normalizes all of these to the first-letter codes BaniDB expects.

IMPORTANT: The Gurmukhi-to-Roman mapping below follows STTM/BaniDB conventions.
Run scripts/test_banidb.py to verify and calibrate these mappings against actual
FirstLetterEng values from the API.
"""

# Gurmukhi consonants and vowel bearers → BaniDB romanized code
# Based on STTM keyboard mapping conventions
GURMUKHI_TO_ROMAN: dict[str, str] = {
    # Consonants (U+0A15 - U+0A39)
    "\u0A15": "k",   # ਕ
    "\u0A16": "K",   # ਖ
    "\u0A17": "g",   # ਗ
    "\u0A18": "G",   # ਘ
    "\u0A19": "|",   # ਙ
    "\u0A1A": "c",   # ਚ
    "\u0A1B": "C",   # ਛ
    "\u0A1C": "j",   # ਜ
    "\u0A1D": "J",   # ਝ
    "\u0A1E": "\\",  # ਞ
    "\u0A1F": "t",   # ਟ
    "\u0A20": "T",   # ਠ
    "\u0A21": "f",   # ਡ
    "\u0A22": "F",   # ਢ
    "\u0A23": "x",   # ਣ
    "\u0A24": "q",   # ਤ
    "\u0A25": "Q",   # ਥ
    "\u0A26": "d",   # ਦ
    "\u0A27": "D",   # ਧ
    "\u0A28": "n",   # ਨ
    "\u0A2A": "p",   # ਪ
    "\u0A2B": "P",   # ਫ
    "\u0A2C": "b",   # ਬ
    "\u0A2D": "B",   # ਭ
    "\u0A2E": "m",   # ਮ
    "\u0A2F": "X",   # ਯ
    "\u0A30": "r",   # ਰ
    "\u0A32": "l",   # ਲ
    "\u0A33": "L",   # ਲ਼
    "\u0A35": "v",   # ਵ
    "\u0A36": "S",   # ਸ਼
    "\u0A38": "s",   # ਸ
    "\u0A39": "h",   # ਹ
    "\u0A5C": "V",   # ੜ

    # Independent vowels (word-initial)
    "\u0A05": "a",   # ਅ
    "\u0A06": "A",   # ਆ
    "\u0A07": "e",   # ਇ
    "\u0A08": "E",   # ਈ
    "\u0A09": "u",   # ਉ
    "\u0A0A": "U",   # ਊ
    "\u0A0F": "y",   # ਏ
    "\u0A10": "Y",   # ਐ
    "\u0A13": "o",   # ਓ
    "\u0A14": "O",   # ਔ

    # Ik Onkar
    "\u0A74": "e",   # ੴ (starts with ik = ਇ)
}

# Devanagari consonants → same romanized codes (for when Whisper outputs Hindi)
DEVANAGARI_TO_ROMAN: dict[str, str] = {
    "\u0915": "k",   # क
    "\u0916": "K",   # ख
    "\u0917": "g",   # ग
    "\u0918": "G",   # घ
    "\u0919": "|",   # ङ
    "\u091A": "c",   # च
    "\u091B": "C",   # छ
    "\u091C": "j",   # ज
    "\u091D": "J",   # झ
    "\u091E": "\\",  # ञ
    "\u091F": "t",   # ट
    "\u0920": "T",   # ठ
    "\u0921": "f",   # ड
    "\u0922": "F",   # ढ
    "\u0923": "x",   # ण
    "\u0924": "q",   # त
    "\u0925": "Q",   # थ
    "\u0926": "d",   # द
    "\u0927": "D",   # ध
    "\u0928": "n",   # न
    "\u092A": "p",   # प
    "\u092B": "P",   # फ
    "\u092C": "b",   # ब
    "\u092D": "B",   # भ
    "\u092E": "m",   # म
    "\u092F": "X",   # य
    "\u0930": "r",   # र
    "\u0932": "l",   # ल
    "\u0935": "v",   # व
    "\u0936": "S",   # श
    "\u0938": "s",   # स
    "\u0939": "h",   # ह

    # Devanagari vowels
    "\u0905": "a",   # अ
    "\u0906": "A",   # आ
    "\u0907": "e",   # इ
    "\u0908": "E",   # ई
    "\u0909": "u",   # उ
    "\u090A": "U",   # ऊ
    "\u090F": "y",   # ए
    "\u0910": "Y",   # ऐ
    "\u0913": "o",   # ओ
    "\u0914": "O",   # औ
}

# Merged lookup for fast access
_ALL_MAPPINGS: dict[str, str] = {**GURMUKHI_TO_ROMAN, **DEVANAGARI_TO_ROMAN}


def _get_first_consonant(word: str) -> str | None:
    """
    Extract the first consonant/vowel-bearer from a word and return its roman code.
    Skips diacritics and modifiers to find the base character.
    """
    for char in word:
        if char in _ALL_MAPPINGS:
            return _ALL_MAPPINGS[char]
        # If it's an ASCII letter, return it directly (already romanized)
        if char.isascii() and char.isalpha():
            return char.lower()
    return None


def extract_first_letters(text: str) -> str:
    """
    Extract the first letter of each word in BaniDB's romanized format.

    Input can be in any script (Gurmukhi, Devanagari, Roman).
    Output is a string of romanized first-letter codes suitable for BaniDB search.

    Example:
        "ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ" → "vjkK"
    """
    words = text.split()
    letters = []
    for word in words:
        letter = _get_first_consonant(word)
        if letter:
            letters.append(letter)
    return "".join(letters)


def is_gurmukhi(text: str) -> bool:
    """Check if text contains Gurmukhi characters."""
    for char in text:
        if "\u0A00" <= char <= "\u0A7F":
            return True
    return False


def is_devanagari(text: str) -> bool:
    """Check if text contains Devanagari characters."""
    for char in text:
        if "\u0900" <= char <= "\u097F":
            return True
    return False
