"""Unit tests for the gutka follow-along aligner."""

import pytest

from src.matcher.bani_loader import GutkaLine
from src.matcher.gutka_aligner import (
    AlignAction,
    GutkaAligner,
    _coverage_score,
    _lcs_len,
)


def _line(idx: int, fl: str) -> GutkaLine:
    """Build a minimal GutkaLine for tests — only fields the aligner reads."""
    return GutkaLine(
        bani_line_idx=idx,
        line_id=f"L{idx:03d}",
        shabad_id="S001",
        order_id=1000 + idx,
        first_letters_ascii=fl,
        gurmukhi_unicode="",
        english=None,
    )


# Synthetic 8-line "bani" — distinct first-letter strings so each line scores
# uniquely against itself and ~zero against the others.
SYNTHETIC_BANI = [
    _line(0, "abcdef"),     # line 0
    _line(1, "ghijkl"),     # line 1
    _line(2, "mnopqr"),     # line 2
    _line(3, "stuvwx"),     # line 3
    _line(4, "yzABCD"),     # line 4
    _line(5, "EFGHIJ"),     # line 5
    _line(6, "KLMNOP"),     # line 6
    _line(7, "QRSTUV"),     # line 7
]


# ─── primitive helpers ───────────────────────────────────────────────────


def test_lcs_basic():
    assert _lcs_len("abc", "abc") == 3
    assert _lcs_len("abc", "xbz") == 1
    assert _lcs_len("", "abc") == 0
    assert _lcs_len("abc", "") == 0
    assert _lcs_len("abcdef", "ace") == 3  # subsequence preserved


def test_coverage_score_perfect_and_zero():
    assert _coverage_score("abcdef", "abcdef") == 1.0
    assert _coverage_score("xyz", "abcdef") == 0.0
    assert _coverage_score("aXbXcXdXeXf", "abcdef") == 1.0  # interleaved is fine


def test_coverage_empty_inputs():
    assert _coverage_score("", "abcdef") == 0.0
    assert _coverage_score("abcdef", "") == 0.0


# ─── aligner behaviour ───────────────────────────────────────────────────


def test_clean_forward_read_advances_monotonically():
    aligner = GutkaAligner(SYNTHETIC_BANI)
    indices = []
    for line in SYNTHETIC_BANI:
        d = aligner.update(line.first_letters_ascii)
        indices.append(aligner.current_idx)
        # Line 0 has nowhere to advance to (idx already 0) so REPEAT is correct.
        # Lines 1+ should be ADVANCEs.
    assert indices == list(range(len(SYNTHETIC_BANI)))


def test_silence_holds_position():
    aligner = GutkaAligner(SYNTHETIC_BANI)
    aligner.reset(3)
    d = aligner.update("")
    assert d.action == AlignAction.HOLD
    assert aligner.current_idx == 3
    d = aligner.update("   ")
    assert d.action == AlignAction.HOLD
    assert aligner.current_idx == 3


def test_garbage_holds_position():
    aligner = GutkaAligner(SYNTHETIC_BANI)
    aligner.reset(3)
    d = aligner.update("zzzzzzz")  # matches nothing in the bani
    assert d.action == AlignAction.HOLD
    assert aligner.current_idx == 3


def test_repeat_fires_on_same_line():
    aligner = GutkaAligner(SYNTHETIC_BANI, rahau_repeat_enabled=True)
    aligner.reset(2)
    d = aligner.update("mnopqr")  # line 2's FL exactly
    assert d.action == AlignAction.REPEAT
    assert aligner.current_idx == 2


def test_repeat_can_be_disabled():
    aligner = GutkaAligner(SYNTHETIC_BANI, rahau_repeat_enabled=False)
    aligner.reset(2)
    d = aligner.update("mnopqr")
    assert d.action == AlignAction.HOLD
    assert aligner.current_idx == 2


def test_forward_only_refuses_backward_jump():
    # Reciter at line 5 suddenly fires line 2's FL — without bidirectional
    # the aligner must HOLD (it has no candidate matching that FL in its
    # forward window).
    aligner = GutkaAligner(SYNTHETIC_BANI, bidirectional=False)
    aligner.reset(5)
    d = aligner.update("mnopqr")  # line 2
    assert d.action == AlignAction.HOLD
    assert aligner.current_idx == 5


def test_bidirectional_jumps_back():
    aligner = GutkaAligner(
        SYNTHETIC_BANI, bidirectional=True, backward_window=4
    )
    aligner.reset(5)
    d = aligner.update("mnopqr")  # line 2 — within backward_window of 4
    assert d.action == AlignAction.BACK
    assert aligner.current_idx == 2


def test_set_bidirectional_toggles_live():
    aligner = GutkaAligner(SYNTHETIC_BANI, backward_window=4)
    aligner.reset(5)
    assert aligner.update("mnopqr").action == AlignAction.HOLD
    aligner.set_bidirectional(True)
    assert aligner.update("mnopqr").action == AlignAction.BACK


def test_skip_ahead_within_forward_window():
    # Reciter skipped 2 lines — aligner should catch up.
    aligner = GutkaAligner(SYNTHETIC_BANI, forward_window=6)
    aligner.reset(1)
    d = aligner.update("stuvwx")  # line 3
    assert d.action == AlignAction.ADVANCE
    assert aligner.current_idx == 3


def test_multi_line_chunk_advances_to_span_end():
    aligner = GutkaAligner(SYNTHETIC_BANI, forward_window=6, max_span_lines=3)
    aligner.reset(1)
    d = aligner.update("ghijklmnopqr")  # lines 1 + 2 in one fast-paath chunk
    assert d.action == AlignAction.ADVANCE
    assert aligner.current_idx == 2


def test_overlapping_fast_windows_keep_advancing():
    aligner = GutkaAligner(SYNTHETIC_BANI, forward_window=6, max_span_lines=3)
    assert aligner.update("abcdefghi").action == AlignAction.ADVANCE
    assert aligner.current_idx == 1
    assert aligner.update("ghijklmno").action == AlignAction.ADVANCE
    assert aligner.current_idx == 2
    assert aligner.update("mnopqrst").action == AlignAction.ADVANCE
    assert aligner.current_idx == 3


def test_skip_past_forward_window_holds():
    # Without a wider window, a far-ahead match isn't reachable.
    aligner = GutkaAligner(SYNTHETIC_BANI, forward_window=2)
    aligner.reset(1)
    d = aligner.update("KLMNOP")  # line 6 — beyond reach
    assert d.action == AlignAction.HOLD
    assert aligner.current_idx == 1


def test_min_margin_blocks_jitter_between_ambiguous_lines():
    # Two lines with identical FL — should HOLD, not flap.
    ambiguous = [
        _line(0, "aaaaaa"),
        _line(1, "aaaaaa"),  # exact duplicate
        _line(2, "aaaaaa"),  # exact duplicate
    ]
    aligner = GutkaAligner(ambiguous, min_margin=0.1)
    aligner.reset(0)
    d = aligner.update("aaaaaa")
    # All candidates tie → margin=0 < 0.1 → HOLD
    assert d.action == AlignAction.HOLD
    assert aligner.current_idx == 0


def test_reset_clamps_index_to_bani_bounds():
    aligner = GutkaAligner(SYNTHETIC_BANI)
    aligner.reset(-5)
    assert aligner.current_idx == 0
    aligner.reset(999)
    assert aligner.current_idx == len(SYNTHETIC_BANI) - 1


def test_empty_bani_raises():
    with pytest.raises(ValueError):
        GutkaAligner([])
