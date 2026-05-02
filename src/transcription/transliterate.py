"""
Extract Gurmukhi first-letter codes from transcribed Punjabi text.

Output matches the ShabadOS first-letter indexing scheme used by
`lines.first_letters` in our local SQLite DB. (The format is the same one
BaniDB exposes, but we never call their API — all search is local.)
This module extracts the first consonant/vowel from each word, converting
Devanagari to Gurmukhi when needed.
"""

# Gurmukhi Unicode ranges for character classification
_GURMUKHI_CONSONANTS = range(0x0A15, 0x0A3A)   # ਕ-ਹ
_GURMUKHI_VOWELS = range(0x0A05, 0x0A15)        # ਅ-ਔ
_GURMUKHI_EXTRA = {0x0A5C, 0x0A74}              # ੜ, ੴ
_DEVANAGARI_OFFSET = 0x0100  # Devanagari → Gurmukhi offset

# ShabadOS first-letter indexing expects canonical vowel-carrier initials
# for several independent vowels (e.g. ਆ -> ਅ).
_FIRST_LETTER_NORMALIZE = {
    "ਆ": "ਅ",
    "ਇ": "ੲ",
    "ਈ": "ੲ",
    "ਏ": "ੲ",
    "ਐ": "ੲ",
    "ਉ": "ੳ",
    "ਊ": "ੳ",
    "ਔ": "ੳ",
}


def _is_gurmukhi_letter(cp: int) -> bool:
    """Check if a codepoint is a Gurmukhi consonant or independent vowel."""
    return cp in _GURMUKHI_CONSONANTS or cp in _GURMUKHI_VOWELS or cp in _GURMUKHI_EXTRA


def normalize_first_letter(letter: str) -> str:
    """Normalize initial letter to ShabadOS-compatible first-letter forms."""
    return _FIRST_LETTER_NORMALIZE.get(letter, letter)


def normalize_for_fullword_search(text: str) -> str:
    """
    Normalize mixed-script transcript into a Gurmukhi phrase for full-word search.

    - Devanagari is converted to Gurmukhi via Unicode offset.
    - Gurmukhi is kept as-is.
    - Non-Punjabi script chars become separators.
    """
    normalized_chars: list[str] = []
    for char in text:
        cp = ord(char)
        if 0x0A00 <= cp <= 0x0A7F:
            normalized_chars.append(char)
            continue
        if 0x0900 <= cp <= 0x097F:
            mapped = cp + _DEVANAGARI_OFFSET
            if 0x0A00 <= mapped <= 0x0A7F:
                normalized_chars.append(chr(mapped))
            else:
                normalized_chars.append(" ")
            continue
        if char.isspace():
            normalized_chars.append(" ")
            continue
        normalized_chars.append(" ")

    return " ".join("".join(normalized_chars).split())


def _devanagari_to_gurmukhi(char: str) -> str | None:
    """Convert a Devanagari character to its Gurmukhi equivalent via Unicode offset."""
    cp = ord(char)
    if 0x0900 <= cp <= 0x097F:
        gurmukhi_cp = cp + _DEVANAGARI_OFFSET
        if _is_gurmukhi_letter(gurmukhi_cp):
            return chr(gurmukhi_cp)
    return None


def _get_first_letter(word: str) -> str | None:
    """
    Extract the first consonant/vowel from a word as a Gurmukhi character.
    Skips diacritics, matras, and modifiers.
    """
    for char in word:
        cp = ord(char)
        # Direct Gurmukhi consonant or vowel
        if _is_gurmukhi_letter(cp):
            return normalize_first_letter(char)
        # Devanagari → convert to Gurmukhi
        if 0x0900 <= cp <= 0x097F:
            converted = _devanagari_to_gurmukhi(char)
            if converted:
                return normalize_first_letter(converted)
    return None


def extract_first_letters(text: str) -> str:
    """
    Extract the first Gurmukhi letter of each word.

    Input can be Gurmukhi or Devanagari (converted to Gurmukhi).
    Output is a string of Gurmukhi first-letter codes for ShabadOS first-letter search.

    Example:
        "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ" → "ਸਸਨਹਜਸਲਵ"
    """
    letters = []
    for word in text.split():
        letter = _get_first_letter(word)
        if letter:
            letters.append(letter)
    return "".join(letters)


# Gurmukhi Unicode → ShabadOS ASCII first-letter mapping
_GURMUKHI_TO_ASCII = {
    "ੳ": "a", "ਅ": "A", "ੲ": "e",  # vowel carriers
    "ਸ": "s", "ਹ": "h", "ਕ": "k", "ਖ": "K", "ਗ": "g", "ਘ": "G",
    "ਙ": "|", "ਚ": "c", "ਛ": "C", "ਜ": "j", "ਝ": "J", "ਞ": "\\",
    "ਟ": "t", "ਠ": "T", "ਡ": "f", "ਢ": "F", "ਣ": "x",
    "ਤ": "q", "ਥ": "Q", "ਦ": "d", "ਧ": "D", "ਨ": "n",
    "ਪ": "p", "ਫ": "P", "ਬ": "b", "ਭ": "B", "ਮ": "m",
    "ਯ": "X", "ਰ": "r", "ਲ": "l", "ਵ": "v", "ੜ": "V",
}


def gurmukhi_to_ascii(first_letters: str) -> str:
    """Convert Gurmukhi Unicode first-letter string to ShabadOS ASCII format.

    Example: "ਸਸਨਹਜਸਲਵ" → "ssnhjslv"
    """
    return "".join(_GURMUKHI_TO_ASCII.get(ch, ch) for ch in first_letters)


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


def devanagari_to_gurmukhi(text: str) -> str:
    """
    Convert Devanagari text to Gurmukhi script.

    Uses Unicode offset mapping (Devanagari U+0900-097F → Gurmukhi U+0A00-0A7F).
    Characters without a Gurmukhi equivalent are kept as-is (spaces, punctuation, digits).
    """
    result: list[str] = []
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            gurmukhi_cp = cp + _DEVANAGARI_OFFSET
            if 0x0A00 <= gurmukhi_cp <= 0x0A7F:
                result.append(chr(gurmukhi_cp))
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)
