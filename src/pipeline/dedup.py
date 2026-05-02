"""Cross-window transcript dedup strategies.

Each strategy decides what part of a freshly-decoded transcript to keep when
the previous decode already committed some text. The choice matters most when
audio windows overlap (sliding-window decoding) — the same audio gets
transcribed twice and we don't want the duplicated tail to also reach the
matcher.

Strategies
----------
TextDedup
    Original behavior. Strip a prefix from the new transcript when it matches
    the tail of the previously committed text. Cheap and works well when the
    pipeline is doing windowed decoding.

    **Known limitation (REA-10):** rejects legitimate repetition — rahau
    (refrain) lines, repeated jaaps like "ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ ਵਾਹਿਗੁਰੂ", slow
    kirtan dwelling on a pankti, sangat echoing the ragi. The text-level
    matching can't distinguish window-overlap from real repetition.

AudioTimeDedup
    Anchor dedup to AUDIO time ranges instead of text. Two transcripts are
    treated as the same instance only if their audio time intervals overlap.
    Identical text from disjoint audio intervals is kept — solving the legit
    repetition bug.

NoOpDedup
    Pass everything through unchanged. The right choice when the streaming
    mode itself handles boundary commits (vad_segmented, local_agreement) —
    in those modes there is no window overlap to clean up.
"""

from __future__ import annotations

from typing import Protocol


class DedupStrategy(Protocol):
    """Pluggable cross-window dedup contract.

    `audio_start` / `audio_end` are absolute monotonic seconds describing the
    audio interval that produced `new_text`. They are required for
    `AudioTimeDedup` and ignored by the others.
    """

    def dedup(
        self,
        new_text: str,
        audio_start: float | None,
        audio_end: float | None,
    ) -> str: ...

    def reset(self) -> None: ...


def _strip_overlap_prefix(new_text: str, last_text: str) -> str:
    """Strip the longest prefix of `new_text` that matches a suffix of `last_text`.

    Mirrors the original heuristic in TranscriptionProcessor: if the new text
    starts with the last text's leading 20 chars (cheap pre-check), look for
    the longest suffix of last_text that is also a prefix of new_text and trim
    it. Search is bounded to half the new text so a fully-redundant window
    isn't wholly erased into nothing.
    """
    if not last_text or not new_text:
        return new_text
    if not new_text.startswith(last_text[:20]):
        return new_text
    overlap_len = min(len(last_text), len(new_text) // 2)
    for i in range(overlap_len, 0, -1):
        if new_text.startswith(last_text[-i:]):
            return new_text[i:].strip()
    return new_text


class TextDedup:
    """Strip overlapping text prefix relative to last commit (legacy behavior)."""

    def __init__(self) -> None:
        self._last_text = ""

    def dedup(
        self,
        new_text: str,
        audio_start: float | None,
        audio_end: float | None,
    ) -> str:
        result = _strip_overlap_prefix(new_text, self._last_text)
        if result:
            self._last_text = result
        return result

    def reset(self) -> None:
        self._last_text = ""


class AudioTimeDedup:
    """Only dedup when audio time ranges overlap.

    Lets the pipeline commit identical text twice when the singer genuinely
    repeated a line — the audio intervals don't overlap, so the new emission
    is preserved verbatim.
    """

    def __init__(self) -> None:
        self._last_text = ""
        self._last_audio_end: float | None = None

    def dedup(
        self,
        new_text: str,
        audio_start: float | None,
        audio_end: float | None,
    ) -> str:
        # Without timing info we can't reason about audio overlap; behave like
        # NoOp rather than silently falling back to text-overlap heuristics.
        if audio_start is None or audio_end is None:
            self._last_text = new_text or self._last_text
            self._last_audio_end = audio_end
            return new_text

        audio_overlap = (
            self._last_audio_end is not None
            and audio_start < self._last_audio_end
        )
        if audio_overlap:
            new_text = _strip_overlap_prefix(new_text, self._last_text)

        if new_text:
            self._last_text = new_text
        self._last_audio_end = audio_end
        return new_text

    def reset(self) -> None:
        self._last_text = ""
        self._last_audio_end = None


class NoOpDedup:
    """Pass-through. Use when the streaming mode already commits at boundaries."""

    def dedup(
        self,
        new_text: str,
        audio_start: float | None,
        audio_end: float | None,
    ) -> str:
        return new_text

    def reset(self) -> None:
        return None


VALID_STRATEGIES = ("text", "audio_time", "none")


def make_strategy(name: str) -> DedupStrategy:
    """Construct a fresh dedup strategy by name. Falls back to TextDedup."""
    if name == "audio_time":
        return AudioTimeDedup()
    if name == "none":
        return NoOpDedup()
    return TextDedup()
