"""Sweep Silero VAD parameters against a cached eval session.

Usage:
    .venv/bin/python scripts/tune_vad.py
    .venv/bin/python scripts/tune_vad.py --audio path/to/file.opus
    .venv/bin/python scripts/tune_vad.py --max-seconds 120

What it does
------------
Decodes a kirtan audio file to 16 kHz mono float32, then runs the Silero VAD
batch segmenter across a grid of (threshold, min_silence_ms, min_speech_ms)
combinations. For each combo it reports:

  - n          : utterance count
  - voice%     : fraction of total time inside a detected utterance
  - p10/p50/p90 utterance duration (seconds)
  - micro      : count of suspect-short utterances (<300 ms)
  - giant      : count of suspect-long utterances (>15 s)
  - score      : composite quality heuristic (higher = better) —
                 prefers a kirtan-shaped distribution: median 2–5 s,
                 voice% 60–90 %, few micro/giant outliers.

The "winner" row is the highest-scoring combo. Use it (or a nearby cell with
similar score and a parameter you prefer for operational reasons) as the
default for ``config.streaming.vad_*``.

This is a diagnostic, not a substitute for live tuning — the canonical test
is to flip the settings on the dashboard and listen to a real program. But it
narrows the search space from "guess from priors" to "two or three good
candidates verified to produce kirtan-shaped utterance distributions".
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Ensure imports resolve when run from project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.vad import SAMPLE_RATE as SILERO_SAMPLE_RATE  # noqa: E402, alias kept
from src.audio.vad import segment_utterances  # noqa: E402


DEFAULT_AUDIO = ROOT / "tests" / "eval" / "cache" / "audio" / "-Dyi8-Qyx4I.opus"


def decode_audio_to_16k_mono(path: Path, max_seconds: float | None = None) -> np.ndarray:
    """Decode any file pyav can handle → 16 kHz mono float32 numpy array."""
    import av

    container = av.open(str(path))
    stream = next(s for s in container.streams if s.type == "audio")

    # Resampler: target 16 kHz mono float32. Silero requires this exact format.
    resampler = av.audio.resampler.AudioResampler(
        format="flt", layout="mono", rate=SILERO_SAMPLE_RATE
    )

    chunks: list[np.ndarray] = []
    samples_per_second = SILERO_SAMPLE_RATE
    target_samples = int(max_seconds * samples_per_second) if max_seconds else None
    total = 0

    for packet in container.demux(stream):
        for frame in packet.decode():
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                if arr.ndim > 1:
                    arr = arr[0]
                arr = arr.astype(np.float32, copy=False)
                chunks.append(arr)
                total += arr.shape[0]
                if target_samples is not None and total >= target_samples:
                    container.close()
                    full = np.concatenate(chunks)
                    return full[:target_samples]

    # Drain resampler (final partial packet).
    for resampled in resampler.resample(None):
        arr = resampled.to_ndarray()
        if arr.ndim > 1:
            arr = arr[0]
        chunks.append(arr.astype(np.float32, copy=False))

    container.close()
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def score_distribution(
    durations_s: list[float],
    voice_ratio: float,
    micro_count: int,
    giant_count: int,
    target_min_s: float = 2.0,
    target_max_s: float = 5.0,
    target_voice_min: float = 0.60,
    target_voice_max: float = 0.90,
) -> float:
    """Composite quality heuristic for a kirtan-shaped distribution.

    Rewards: median utterance length in [2 s, 5 s] (matches typical pankti).
    Rewards: voice ratio in [60 %, 90 %].
    Penalizes: micro utterances (<300 ms) — usually false starts on tabla hits.
    Penalizes: giant utterances (>15 s) — usually missed offset boundaries.
    Penalizes: empty result (0 utterances).
    """
    if not durations_s:
        return 0.0
    median = statistics.median(durations_s)
    # Median position score ∈ [0, 1] — peak at midpoint of target range.
    if median < target_min_s:
        median_score = max(0.0, median / target_min_s)
    elif median > target_max_s:
        median_score = max(0.0, 1.0 - (median - target_max_s) / target_max_s)
    else:
        median_score = 1.0

    if voice_ratio < target_voice_min:
        voice_score = max(0.0, voice_ratio / target_voice_min)
    elif voice_ratio > target_voice_max:
        # Voice ratio too high = likely catching instrumental as voice.
        voice_score = max(0.0, 1.0 - (voice_ratio - target_voice_max) / 0.10)
    else:
        voice_score = 1.0

    n = len(durations_s)
    micro_penalty = min(1.0, micro_count / max(n, 1))
    giant_penalty = min(1.0, giant_count / max(n, 1))

    return (
        0.45 * median_score
        + 0.35 * voice_score
        - 0.10 * micro_penalty
        - 0.10 * giant_penalty
    )


def sweep(
    audio: np.ndarray,
    thresholds: list[float],
    min_silence_list: list[int],
    min_speech_list: list[int],
    speech_pad_ms: int,
    max_utterance_ms: int,
    backend: str = "kirtan",
) -> list[dict]:
    """Run a Cartesian sweep, return list of result rows."""
    total_s = audio.shape[0] / SILERO_SAMPLE_RATE
    results = []
    combos = [
        (t, ms, msp)
        for t in thresholds
        for ms in min_silence_list
        for msp in min_speech_list
    ]
    print(f"\n  Sweeping {len(combos)} combinations on {total_s:.1f} s of audio (backend={backend})...\n")

    for i, (t, ms, msp) in enumerate(combos, 1):
        t0 = time.monotonic()
        utts = segment_utterances(
            audio,
            threshold=t,
            min_silence_ms=ms,
            min_speech_ms=msp,
            speech_pad_ms=speech_pad_ms,
            max_utterance_ms=max_utterance_ms,
            backend=backend,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        durations = [u.duration_s for u in utts]
        voice_time = sum(durations)
        voice_ratio = voice_time / total_s if total_s > 0 else 0.0
        micro = sum(1 for d in durations if d < 0.3)
        giant = sum(1 for d in durations if d > 15.0)
        if durations:
            sd = sorted(durations)
            p10 = sd[max(0, int(0.10 * len(sd)) - 1)]
            p50 = statistics.median(sd)
            p90 = sd[min(len(sd) - 1, int(0.90 * len(sd)))]
            mean = sum(sd) / len(sd)
        else:
            p10 = p50 = p90 = mean = 0.0
        score = score_distribution(durations, voice_ratio, micro, giant)
        results.append({
            "threshold": t,
            "min_silence_ms": ms,
            "min_speech_ms": msp,
            "n": len(utts),
            "voice_pct": round(voice_ratio * 100, 1),
            "p10": round(p10, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "mean": round(mean, 2),
            "micro": micro,
            "giant": giant,
            "elapsed_ms": int(elapsed_ms),
            "score": round(score, 3),
        })
        print(
            f"  [{i:2}/{len(combos):2}] t={t:.2f} sil={ms:>4}ms spk={msp:>3}ms"
            f" → n={len(utts):>3} voice={voice_ratio*100:>5.1f}%"
            f" p50={p50:>5.2f}s micro={micro:>2} giant={giant:>1}"
            f" score={score:.3f}  ({int(elapsed_ms)}ms)"
        )

    return results


def print_table(results: list[dict]) -> None:
    """Print full results sorted by score, then highlight winner."""
    print("\n  RESULTS (sorted by composite score, best first):\n")
    print(
        "  thr | sil | spk |  n  | voice%  p10  p50  p90  mean | micro giant | score | wall"
    )
    print(
        "  ----+-----+-----+-----+----------------------------+-------------+-------+------"
    )
    sorted_ = sorted(results, key=lambda r: r["score"], reverse=True)
    for r in sorted_:
        print(
            f"  {r['threshold']:.2f}|{r['min_silence_ms']:>4} |{r['min_speech_ms']:>4} |"
            f"{r['n']:>4} | "
            f"{r['voice_pct']:>5.1f}% {r['p10']:>4.2f} {r['p50']:>4.2f} {r['p90']:>4.2f} {r['mean']:>4.2f} |"
            f"  {r['micro']:>3}    {r['giant']:>2}  | {r['score']:.3f} |"
            f" {r['elapsed_ms']:>4}ms"
        )

    if not sorted_:
        return
    w = sorted_[0]
    print("\n  ─── Winner ───")
    print(
        f"  vad_threshold       = {w['threshold']:.2f}\n"
        f"  vad_min_silence_ms  = {w['min_silence_ms']}\n"
        f"  vad_min_speech_ms   = {w['min_speech_ms']}\n"
        f"  → {w['n']} utterances, voice {w['voice_pct']:.1f}%, "
        f"p50={w['p50']:.2f}s, micro={w['micro']}, giant={w['giant']}, score={w['score']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO,
        help=f"Audio file to tune against (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=180.0,
        help="Cap audio at this many seconds for faster sweeps (default: 180).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="kirtan",
        choices=["kirtan", "silero"],
        help="VAD backend. 'kirtan' = spectral voice-band detector (default, "
             "tuned for sung Gurbani). 'silero' = silero-vad (good for spoken "
             "katha but rejects sung kirtan; threshold semantics differ).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.45, 0.50, 0.55, 0.60, 0.65],
        help="VAD threshold sweep. KirtanVAD threshold = voice-band ratio "
             "(0.5–0.85 typical for sung kirtan); Silero threshold = speech-prob.",
    )
    parser.add_argument(
        "--min-silence-ms",
        type=int,
        nargs="+",
        default=[200, 400, 600, 800],
        help="Min silence durations to sweep.",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        nargs="+",
        default=[100, 200, 400],
        help="Min speech durations to sweep.",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=200,
        help="Speech pad (held constant across sweep).",
    )
    parser.add_argument(
        "--max-utterance-ms",
        type=int,
        default=30000,
        help="Max utterance length (held constant across sweep).",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"  ERROR: audio file not found: {args.audio}")
        return 2

    print(f"  Loading audio: {args.audio.name}")
    t0 = time.monotonic()
    audio = decode_audio_to_16k_mono(args.audio, max_seconds=args.max_seconds)
    decode_ms = (time.monotonic() - t0) * 1000.0
    if audio.size == 0:
        print("  ERROR: decoded zero samples.")
        return 2
    duration = audio.shape[0] / SILERO_SAMPLE_RATE
    print(f"  → {duration:.1f} s @ 16 kHz mono ({audio.size:,} samples, decoded in {decode_ms:.0f}ms)")

    results = sweep(
        audio,
        thresholds=args.thresholds,
        min_silence_list=args.min_silence_ms,
        min_speech_list=args.min_speech_ms,
        speech_pad_ms=args.speech_pad_ms,
        max_utterance_ms=args.max_utterance_ms,
        backend=args.backend,
    )
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
