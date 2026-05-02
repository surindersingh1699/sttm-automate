"""LocalAgreement-2 streaming buffer for whisper-streaming-style commits.

Implements the algorithm from Macháček et al. 2023 (whisper_streaming).
Each iteration the orchestrator:

    1. Appends fresh audio to ``LocalAgreementBuffer.append(samples)``.
    2. Periodically decodes ``working_audio()`` with Whisper.
    3. Hands the decoded segments to ``commit_decode(segments)`` which
       returns any text that **two consecutive decodes agree on** but that
       hasn't been emitted yet.
    4. The orchestrator forwards just-emitted text through the matcher
       pipeline (via ``_process_decoded_text``).
    5. When the audio buffer grows past ``max_buffer_samples``,
       ``maybe_anchor()`` drops the oldest portion and re-bases the agreement
       state. Without anchoring, the buffer would grow unbounded.

Why character-level prefixes
-----------------------------
faster-whisper's public ``Segment`` type carries text + start/end seconds but
not raw token IDs. Character-level longest-common-prefix is a reliable
substitute for Gurmukhi because whisper.cpp / faster-whisper tokenizers are
byte-stable across decodes — repeated decodes of the same audio prefix
produce identical leading characters in their output. Multi-codepoint
graphemes (Gurmukhi consonant clusters with matras) are not split because
the LCP works on Python's ``str`` which is codepoint-indexed, and Whisper
never emits a partial codepoint at the segment boundary.

Trade-offs vs. token-level
--------------------------
- Slightly less precise — a single char of disagreement can shrink the
  agreed prefix by one grapheme, which token-LCP would catch as just "this
  token differs".
- Catches the same gross-disagreement cases (whole-word substitutions,
  reorderings) that the token version does — those are the failure modes
  the algorithm targets anyway.
- Avoids an engine-level API change to expose token IDs.

If the accuracy gap proves to matter, we can extend ``TranscriptionSegment``
with optional ``token_ids`` later and switch the LCP to token comparison
without changing the orchestrator surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SAMPLE_RATE = 16000


@dataclass(frozen=True)
class CommitResult:
    """Result of one ``commit_decode`` call."""

    new_text: str          # newly-agreed text since the previous commit
    committed_total: str   # full agreed text since the last anchor


class LocalAgreementBuffer:
    """Audio + decode-history state for LocalAgreement-2 streaming.

    Single-threaded — owned by the orchestrator's ``_local_agreement_loop``.
    All numpy slicing is done on the orchestrator's event loop thread; only
    the actual Whisper inference inside ``working_audio()`` runs in a
    thread executor.
    """

    def __init__(
        self,
        agreement_n: int = 2,
        max_buffer_ms: int = 20_000,
    ) -> None:
        if agreement_n < 2:
            raise ValueError("agreement_n must be ≥ 2")
        self.agreement_n = int(agreement_n)
        self.max_buffer_samples = int(max_buffer_ms / 1000 * SAMPLE_RATE)
        self._audio: np.ndarray = np.empty(0, dtype=np.float32)
        # Last N decoded text strings — order is oldest → newest.
        self._history: list[str] = []
        # Total text already emitted to the orchestrator since the last anchor.
        self._committed_text: str = ""

    # ── Audio plumbing ────────────────────────────────────────────────

    def append(self, samples: np.ndarray) -> None:
        """Append new audio samples to the working buffer."""
        if samples.ndim > 1:
            samples = samples[:, 0]
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        if samples.size == 0:
            return
        self._audio = np.concatenate([self._audio, samples])

    def working_audio(self) -> np.ndarray:
        """Return a copy of the current audio buffer for decoding."""
        return self._audio.copy()

    def buffer_seconds(self) -> float:
        return float(self._audio.size) / SAMPLE_RATE

    def buffer_samples(self) -> int:
        return int(self._audio.size)

    # ── Commit logic ──────────────────────────────────────────────────

    def commit_decode(self, segments) -> CommitResult:
        """Record this decode and return any newly-agreed text.

        ``segments`` is the list returned by ``BaseTranscriptionEngine
        .transcribe`` — we use only ``.text`` here. Returns ``CommitResult``
        with the new text since the previous commit, or empty strings if
        agreement hasn't formed yet.
        """
        full_text = " ".join((s.text or "").strip() for s in segments).strip()
        self._history.append(full_text)
        if len(self._history) > self.agreement_n:
            self._history.pop(0)
        if len(self._history) < self.agreement_n:
            return CommitResult(new_text="", committed_total=self._committed_text)

        agreed = _longest_common_prefix(self._history)
        # Strip a trailing partial word — agreement that ends mid-word is
        # fragile and the next decode will just split it differently.
        agreed = _trim_to_word_boundary(agreed)
        if len(agreed) <= len(self._committed_text):
            return CommitResult(new_text="", committed_total=self._committed_text)

        new_text = agreed[len(self._committed_text):].lstrip()
        self._committed_text = agreed
        return CommitResult(new_text=new_text, committed_total=self._committed_text)

    # ── Anchoring (drop old audio when buffer fills) ──────────────────

    def maybe_anchor(self) -> bool:
        """If the buffer exceeds ``max_buffer_samples``, drop the oldest half
        and reset the agreement state. Returns True if it anchored.

        Anchoring loses the previous agreement context — the next ``agreement_n``
        decodes have to re-establish agreement before any new text is emitted.
        That's the cost of bounded buffer size.
        """
        if self._audio.size <= self.max_buffer_samples:
            return False
        keep = self.max_buffer_samples // 2
        self._audio = self._audio[-keep:]
        self._history.clear()
        self._committed_text = ""
        return True

    # ── Utility ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all state — used when the orchestrator flushes context."""
        self._audio = np.empty(0, dtype=np.float32)
        self._history.clear()
        self._committed_text = ""

    @property
    def committed_text(self) -> str:
        return self._committed_text


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _longest_common_prefix(strs: list[str]) -> str:
    """Longest common prefix across all input strings (codepoint-indexed)."""
    if not strs:
        return ""
    s = strs[0]
    for o in strs[1:]:
        i = 0
        upper = min(len(s), len(o))
        while i < upper and s[i] == o[i]:
            i += 1
        s = s[:i]
        if not s:
            return s
    return s


def _trim_to_word_boundary(text: str) -> str:
    """Trim a trailing partial word so we don't commit half a Gurmukhi word.

    Whisper's tokenizer can split mid-grapheme on the boundary of a streaming
    decode, producing a final character that flips between decodes. Stop the
    commit at the last whitespace.
    """
    if not text:
        return text
    # If the text ends with a space we're already at a boundary.
    if text[-1].isspace():
        return text.rstrip()
    last_space = text.rfind(" ")
    if last_space < 0:
        # No space at all — only one (possibly partial) word. Don't commit.
        return ""
    return text[: last_space + 1].rstrip()
