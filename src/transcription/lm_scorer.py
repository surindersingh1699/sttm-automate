"""KenLM char-LM scorer for the IndicConformer transcription path.

Loads a char-level n-gram KenLM (Gurbani canon, 6-gram by default — see
[Gurbani_ASR_v4/scripts/build_kenlm.py]) and scores transcription
hypotheses for canonical-Gurbani plausibility. Used by
``IndicConformerEngine`` as a post-decode hallucination gate, sitting
*after* pyctcdecode + the BPE-fusion LM.

Design:
- Single global instance, lazy-loaded on first ``score()`` call. The .bin
  is mmap'd, so load is cheap (~10 ms) and idempotent.
- Tokenization matches what ``build_kenlm.py`` wrote: chars separated by
  spaces with ``|`` marking word boundaries.
- ``score()`` returns logp, per-char logp, and per-char perplexity. The
  hallucination gate consumes ``per_char_ppl`` because it is length-
  independent.

Calibration (from held-out sanity check, see ``build_kenlm.py`` docs):

- Canon text (Mool Mantar, kirtan repetitions): per-char PPL ~3-4
- Modern Punjabi (legal Gurmukhi but not Gurbani): per-char PPL ~6-8
- Random Gurmukhi garbage: per-char PPL ~200+
- ``HALLUCINATION_PPL`` defaults to 25 — ~10× headroom over canon and
  ~8× below garbage, so it only fires on clear hallucinations and
  leaves real (even unusual) Gurbani alone.

This module is the sister of ``Gurbani_ASR_v4/apps/transcribe/lm_scorer.py``;
keep the two in sync if the LM recipe changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Project-root relative default. Override at runtime via STTM_LM_PATH env var,
# or pass an explicit path into ``configure()``. We don't read config.py here
# to keep this module free of import-time side effects.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LM_PATH = _PROJECT_ROOT / "models" / "lm" / "gurbani_canon_char_6gram.bin"

WORD_BOUNDARY = "|"


@dataclass
class LMScore:
    text: str
    logp: float                  # base-10 log probability, summed over chars
    per_char_logp: float
    per_char_ppl: float
    n_chars: int
    is_hallucination: bool       # per_char_ppl > hallucination_threshold
    is_low_confidence: bool      # per_char_ppl > low_confidence_threshold


def _char_tokenize(line: str) -> str:
    """Match ``build_kenlm.py``: ``'ੴ ਸਤਿ'`` -> ``'ੴ | ਸ ਤ ਿ'``."""
    out: list[str] = []
    for word in line.split():
        if out:
            out.append(WORD_BOUNDARY)
        out.extend(list(word))
    return " ".join(out)


class _CharLM:
    def __init__(self, path: Path):
        import kenlm  # type: ignore  # noqa: PLC0415
        self.path = path
        self.model = kenlm.Model(str(path))

    def score(
        self,
        text: str,
        hallucination_threshold: float,
        low_confidence_threshold: float,
    ) -> LMScore:
        tokens = _char_tokenize(text)
        n_chars = len(tokens.split())
        if n_chars == 0:
            return LMScore(
                text=text, logp=0.0, per_char_logp=0.0, per_char_ppl=1.0,
                n_chars=0, is_hallucination=False, is_low_confidence=False,
            )
        logp = self.model.score(tokens, bos=True, eos=True)
        # +1 for </s>, which counts as one token in KenLM's perplexity.
        denom = n_chars + 1
        per_char_logp = logp / denom
        per_char_ppl = 10 ** (-per_char_logp)
        return LMScore(
            text=text,
            logp=logp,
            per_char_logp=per_char_logp,
            per_char_ppl=per_char_ppl,
            n_chars=n_chars,
            is_hallucination=per_char_ppl > hallucination_threshold,
            is_low_confidence=per_char_ppl > low_confidence_threshold,
        )


# Lazy singleton — loaded once per (path) per process.
_GLOBAL: Optional[_CharLM] = None
_GLOBAL_PATH: Optional[Path] = None
_GLOBAL_ERR: Optional[str] = None


def _resolve_lm_path(override: Optional[Path] = None) -> Path:
    if override is not None:
        return override
    env = os.environ.get("STTM_LM_PATH")
    if env:
        return Path(env).expanduser()
    return _DEFAULT_LM_PATH


def configure(path: Optional[Path]) -> None:
    """Force the LM to (re)load from ``path`` on the next ``get_lm()``.

    Called by ``IndicConformerEngine`` when the user changes the LM file
    in config. Pass ``None`` to revert to the env/default resolution.
    """
    global _GLOBAL, _GLOBAL_PATH, _GLOBAL_ERR
    new_path = _resolve_lm_path(path)
    if _GLOBAL is not None and _GLOBAL_PATH == new_path:
        return
    _GLOBAL = None
    _GLOBAL_PATH = new_path
    _GLOBAL_ERR = None


def is_available() -> bool:
    """True iff the LM can be loaded right now. Cached after first call."""
    return get_lm() is not None


def get_lm() -> Optional[_CharLM]:
    global _GLOBAL, _GLOBAL_PATH, _GLOBAL_ERR
    if _GLOBAL is not None:
        return _GLOBAL
    if _GLOBAL_ERR is not None:
        return None
    path = _GLOBAL_PATH or _resolve_lm_path(None)
    _GLOBAL_PATH = path
    if not path.exists():
        _GLOBAL_ERR = f"LM not found at {path} — copy from Gurbani_ASR_v4 or download from HF"
        return None
    try:
        _GLOBAL = _CharLM(path)
        return _GLOBAL
    except ImportError:
        _GLOBAL_ERR = "kenlm Python bindings not installed (pip install kenlm)"
        return None
    except Exception as e:  # noqa: BLE001
        _GLOBAL_ERR = f"LM load failed: {e}"
        return None


def load_error() -> Optional[str]:
    """Returns last load-failure reason, or None if LM is loaded fine."""
    return _GLOBAL_ERR


def score(
    text: str,
    *,
    hallucination_threshold: float = 25.0,
    low_confidence_threshold: float = 12.0,
) -> Optional[LMScore]:
    """Score ``text``. Returns ``None`` if the LM isn't available."""
    lm = get_lm()
    if lm is None:
        return None
    return lm.score(text, hallucination_threshold, low_confidence_threshold)


def is_likely_hallucination(
    text: str,
    threshold: float = 25.0,
) -> bool:
    """Cheap boolean check — returns ``False`` if LM is unavailable (fail-open)."""
    s = score(text, hallucination_threshold=threshold)
    return s is not None and s.is_hallucination
