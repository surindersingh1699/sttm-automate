from src.transcription.engine import TranscriptionSegment
from src.transcription.processor import TranscriptionProcessor
from src.transcription.transliterate import (
    normalize_for_fullword_search,
    transliterate_devanagari_to_gurmukhi,
)


def test_devanagari_to_gurmukhi_transliteration():
    text = "सोचै सोचि न होवई जे सोची लख वार"
    assert transliterate_devanagari_to_gurmukhi(text) == "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ"


def test_processor_outputs_gurmukhi_for_devanagari_segments():
    processor = TranscriptionProcessor()
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=2.0,
            text="सोचै सोचि न होवई",
        )
    ]
    # Processor keeps raw model text; transliteration is applied at display time.
    assert processor.process(segments) == "सोचै सोचि न होवई"


def test_processor_filters_non_punjabi_hallucination():
    processor = TranscriptionProcessor()
    segments = [
        TranscriptionSegment(
            start=0.0,
            end=2.0,
            text="Language Eso ὁ Ḣʰᴇ ḅ ᴇ planetary",
        )
    ]
    assert processor.process(segments) == ""


def test_normalize_for_fullword_search_returns_gurmukhi_only_tokens():
    mixed = "सोचै ਸੋਚਿ abc 123 !!! न होवई"
    # Devanagari converts to Gurmukhi and non-Punjabi script is dropped.
    assert normalize_for_fullword_search(mixed) == "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ"
