"""Optional learned lock-ranker for shabad candidates.

The rule-based matcher still owns retrieval. This module only turns an already
retrieved/scored candidate into a calibrated probability that the candidate is
the correct shabad for the current audio window.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

from src.config import config


RETRIEVAL_SOURCE_KEYS = (
    "type0",
    "type1",
    "type0_sub",
    "type1_sub",
    "type2",
    "multiline",
    "multiline3",
    "type3_words",
    "phonetic",
    "phonetic_sub",
    "ngram4",
    "rotation",
    "type_rotation",
)


FEATURE_NAMES = (
    "score",
    "score_gap",
    "word_overlap",
    "word_vote_hits",
    "word_vote_score",
    "dense_dominant",
    "speech_rate_lps",
    "window_seconds",
    "asr_confidence",
    "asr_avg_logprob",
    "asr_entropy",
    *[f"src_{name}" for name in RETRIEVAL_SOURCE_KEYS],
)


def candidate_features(
    candidate: dict,
    *,
    top_score: float,
    second_score: float,
    speech_rate_lps: float,
    window_seconds: float,
) -> list[float]:
    sources = set(candidate.get("retrieval_sources") or [])
    score = float(candidate.get("score") or 0.0)
    return [
        score,
        float(top_score - second_score),
        float(candidate.get("word_overlap") or 0),
        float(candidate.get("word_vote_hits") or 0),
        float(candidate.get("word_vote_score") or 0.0),
        1.0 if candidate.get("dense_dominant") else 0.0,
        float(speech_rate_lps or 0.0),
        float(window_seconds or 0.0),
        float(candidate.get("asr_confidence") or 0.0),
        float(candidate.get("asr_avg_logprob") or 0.0),
        float(candidate.get("asr_entropy") or 0.0),
        *[1.0 if key in sources else 0.0 for key in RETRIEVAL_SOURCE_KEYS],
    ]


@lru_cache(maxsize=1)
def _load_model():
    path = Path(config.matcher.learned_ranker_model_path)
    if not path.exists():
        return None
    try:
        import joblib  # type: ignore
    except ImportError:
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[LearnedRanker] Could not load {path}: {exc}")
        return None


def predict_probabilities(
    candidates: Iterable[dict],
    *,
    speech_rate_lps: float,
    window_seconds: float,
) -> list[float] | None:
    model = _load_model()
    candidates = list(candidates)
    if model is None or not candidates:
        return None
    scores = [float(c.get("score") or 0.0) for c in candidates]
    top_score = scores[0]
    second_score = scores[1] if len(scores) > 1 else 0.0
    x = [
        candidate_features(
            candidate,
            top_score=top_score,
            second_score=second_score,
            speech_rate_lps=speech_rate_lps,
            window_seconds=window_seconds,
        )
        for candidate in candidates
    ]
    if hasattr(model, "predict_proba"):
        return [float(p[1]) for p in model.predict_proba(x)]
    if hasattr(model, "decision_function"):
        import math
        return [1.0 / (1.0 + math.exp(-float(v))) for v in model.decision_function(x)]
    return None
