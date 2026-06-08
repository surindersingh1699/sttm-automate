"""Measure first-letter accuracy of the current ASR engine vs. dataset GT.

For each vocal slide in `surindersinghssj/gurbani-kirtan-dataset-v2` whose
audio we have cached at `tests/eval/cache/audio/<video_id>.opus`:

  1. Slice the audio at [start_time, end_time].
  2. Transcribe via the engine pinned in `.runtime_settings.json`
     (IndicConformer by default).
  3. Extract first-letter codes from both the ASR output and the GT text.
  4. Score:
        - aligned char matches (Needleman-Wunsch, match=+1 / mismatch=-1 / gap=-1)
        - recall  = matches / len(GT)        ← "% of GT first-letters recovered"
        - precision = matches / len(ASR)
        - F1
        - positional accuracy on equal-length pairs (strict baseline)

Aggregates per-bucket (short ≤5 words, mid 6-12, long >12) so we can see
whether the model loses first-letters mostly on long lines (likely) or short
ones (more worrying for the matcher's lock heuristic).

The point of this script is to tell us which of the user's three regimes
the engine is in *right now*, before we build a first-letter aux head:
  ≥95%   → matcher already has enough to filter; fix the matcher, not the model
  70-90% → aux head will pay off (+5-10 pp = much higher matcher snap rate)
  <70%   → acoustic-level problem; aux head alone won't be enough

Usage:
    # Single video
    python scripts/measure_first_letter_accuracy.py -Dyi8-Qyx4I

    # Multiple videos, aggregated
    python scripts/measure_first_letter_accuracy.py -Dyi8-Qyx4I IZOsmkdmmcg

    # All cached videos
    python scripts/measure_first_letter_accuracy.py --all-cached

    # Cap slides per video (faster iteration)
    python scripts/measure_first_letter_accuracy.py --all-cached --limit 80
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config  # noqa: E402
from src.transcription.factory import pin_indic_best_settings  # noqa: E402
from src.transcription.transliterate import (  # noqa: E402
    extract_first_letters,
    gurmukhi_to_ascii,
)

AUDIO_CACHE = ROOT / "tests/eval/cache/audio"
RUN_DIR = ROOT / "tests/eval/runs"
SAMPLE_RATE = 16000


# ── runtime wiring (mirrors scripts/eval_indic_lm.py) ──────────────────────

def _wire_runtime() -> None:
    rt_path = ROOT / ".runtime_settings.json"
    rt = json.loads(rt_path.read_text(encoding="utf-8")) if rt_path.exists() else {}
    if (mid := rt.get("hf_model_id")) in config.whisper.available_models:
        config.whisper.apply_model_id(mid)
    config.whisper.engine = rt.get("engine", "indicconformer")
    if (prec := rt.get("onnx_precision")) in config.whisper.available_precisions:
        config.whisper.onnx_precision = prec
    config.whisper.lm_enabled = bool(rt.get("lm_enabled", False))
    pin_indic_best_settings()


# ── audio I/O ──────────────────────────────────────────────────────────────

def load_audio_pcm(path: Path) -> np.ndarray:
    """Decode an opus/mp3/wav to mono 16k float32 via ffmpeg."""
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(path),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def slice_pcm(
    pcm: np.ndarray,
    start_s: float,
    end_s: float,
    max_window_s: float | None = None,
) -> np.ndarray:
    """Slice [start_s, end_s) — optionally cap to a centered window of max_window_s.

    Long slides (a single line sustained for 20-50s while the kirtanis repeat
    the pankti) blow up ASR output length: the engine's internal chunker splits
    into N pieces and emits N copies of the line, which crushes precision and
    is unfair as a "per-word FL accuracy" measure. Centering on a single
    ≤max_window_s slice asks the more natural question: given ~one
    repetition of the line, how many first-letters did the model get right.
    """
    a = max(0, int(start_s * SAMPLE_RATE))
    b = min(pcm.size, int(end_s * SAMPLE_RATE))
    if max_window_s is not None:
        max_samples = int(max_window_s * SAMPLE_RATE)
        if b - a > max_samples:
            mid = (a + b) // 2
            a = max(0, mid - max_samples // 2)
            b = min(pcm.size, a + max_samples)
    return pcm[a:b].copy() if b > a else np.zeros(0, dtype=np.float32)


# ── scoring ────────────────────────────────────────────────────────────────

def fl_string(text: str) -> str:
    """Gurmukhi/Devanagari → ShabadOS ASCII first-letter codes (e.g. 'ssnhjslv')."""
    return gurmukhi_to_ascii(extract_first_letters(text or ""))


def lcs_length(a: str, b: str) -> int:
    """Longest common subsequence — order-preserving match count.

    This is the metric the matcher cares about: how many of GT's first-letters
    appear in ASR in the same order, allowing other ASR letters in between.
    LCS / len(GT) = "sequence-preserving recall".
    """
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[m]


def gt_contained_as_substring(gt: str, asr: str) -> bool:
    """Does ASR contain the full GT first-letter sequence as a contiguous substring?

    This is the cleanest matcher-success predicate: a 1-shot exact prefix
    lookup on `lines.first_letters LIKE '%<gt>%'` succeeds iff this is True.
    """
    return bool(gt) and gt in asr


def longest_gt_run_in_asr(gt: str, asr: str) -> int:
    """Length of the longest contiguous substring of GT that appears in ASR.

    The matcher's n-gram fallback (Strategy 9: char 4-grams) needs at least N
    consecutive matching first-letters somewhere in the ASR stream. This says
    'best contiguous run we managed to land'.
    """
    if not gt or not asr:
        return 0
    best = 0
    n = len(gt)
    # Try every starting position in GT, extend as long as the run appears in ASR.
    for i in range(n):
        # Binary-extend: find longest j such that gt[i:i+j] in asr.
        lo, hi = 1, n - i
        cur = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if gt[i : i + mid] in asr:
                cur = mid
                lo = mid + 1
            else:
                hi = mid - 1
        best = max(best, cur)
        if best == n:
            break
    return best


def needleman_wunsch_matches(a: str, b: str) -> int:
    """Number of character matches in the optimal global alignment.

    Match = +1, mismatch = -1, gap = -1. Standard NW. Counts only positions
    aligned to identical characters — insertions and deletions never count.
    """
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    # DP for score AND matches in parallel; backtrack to recover match count.
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        score[i][0] = -i
    for j in range(m + 1):
        score[0][j] = -j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (1 if a[i - 1] == b[j - 1] else -1)
            up = score[i - 1][j] - 1
            left = score[i][j - 1] - 1
            score[i][j] = max(diag, up, left)
    # Backtrack
    matches = 0
    i, j = n, m
    while i > 0 and j > 0:
        if (
            score[i][j] == score[i - 1][j - 1] + (1 if a[i - 1] == b[j - 1] else -1)
        ):
            if a[i - 1] == b[j - 1]:
                matches += 1
            i -= 1
            j -= 1
        elif score[i][j] == score[i - 1][j] - 1:
            i -= 1
        else:
            j -= 1
    return matches


@dataclass
class SlideResult:
    slide_index: int
    start_s: float
    end_s: float
    gt_text: str
    asr_text: str
    gt_fl: str
    asr_fl: str
    matches: int
    positional_matches: int | None  # only when len(gt_fl) == len(asr_fl)
    lcs: int                         # order-preserving matches
    full_substring_hit: bool         # GT FL contained verbatim in ASR FL
    longest_run: int                 # longest contiguous GT substring found in ASR

    @property
    def recall(self) -> float:
        return self.matches / len(self.gt_fl) if self.gt_fl else 0.0

    @property
    def precision(self) -> float:
        return self.matches / len(self.asr_fl) if self.asr_fl else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class Bucket:
    name: str
    n: int = 0
    gt_chars: int = 0
    asr_chars: int = 0
    matches: int = 0
    lcs_total: int = 0
    full_substring_hits: int = 0
    positional_eligible: int = 0
    positional_matches: int = 0
    positional_total: int = 0
    per_slide_recall: list[float] = field(default_factory=list)
    per_slide_lcs_recall: list[float] = field(default_factory=list)
    per_slide_longest_run_frac: list[float] = field(default_factory=list)

    def add(self, r: SlideResult) -> None:
        self.n += 1
        self.gt_chars += len(r.gt_fl)
        self.asr_chars += len(r.asr_fl)
        self.matches += r.matches
        self.lcs_total += r.lcs
        if r.full_substring_hit:
            self.full_substring_hits += 1
        self.per_slide_recall.append(r.recall)
        self.per_slide_lcs_recall.append(r.lcs / len(r.gt_fl) if r.gt_fl else 0.0)
        self.per_slide_longest_run_frac.append(
            r.longest_run / len(r.gt_fl) if r.gt_fl else 0.0
        )
        if r.positional_matches is not None:
            self.positional_eligible += 1
            self.positional_matches += r.positional_matches
            self.positional_total += len(r.gt_fl)

    def summary(self) -> dict:
        recall = self.matches / self.gt_chars if self.gt_chars else 0.0
        precision = self.matches / self.asr_chars if self.asr_chars else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
        lcs_recall = self.lcs_total / self.gt_chars if self.gt_chars else 0.0
        return {
            "bucket": self.name,
            "n_slides": self.n,
            "gt_first_letters_total": self.gt_chars,
            "asr_first_letters_total": self.asr_chars,
            "aligned_matches": self.matches,
            "recall_micro_pct": round(recall * 100, 1),
            "precision_micro_pct": round(precision * 100, 1),
            "f1_micro_pct": round(f1 * 100, 1),
            "lcs_recall_micro_pct": round(lcs_recall * 100, 1),
            "full_substring_hit_pct": round(
                self.full_substring_hits / self.n * 100, 1
            ) if self.n else 0.0,
            "recall_per_slide_median_pct": round(
                statistics.median(self.per_slide_recall) * 100, 1
            ) if self.per_slide_recall else 0.0,
            "lcs_recall_per_slide_median_pct": round(
                statistics.median(self.per_slide_lcs_recall) * 100, 1
            ) if self.per_slide_lcs_recall else 0.0,
            "longest_run_frac_median_pct": round(
                statistics.median(self.per_slide_longest_run_frac) * 100, 1
            ) if self.per_slide_longest_run_frac else 0.0,
            "positional_eligible_slides": self.positional_eligible,
            "positional_acc_pct": round(
                self.positional_matches / self.positional_total * 100, 1
            ) if self.positional_total else None,
        }


def _bucket_for(n_words: int) -> str:
    if n_words <= 5:
        return "short(<=5)"
    if n_words <= 12:
        return "mid(6-12)"
    return "long(>12)"


# ── per-video scoring ──────────────────────────────────────────────────────

def _score_one_video(
    video_id: str,
    ds_rows_by_video: dict[str, list],
    engine,
    limit: int | None,
    audio_window_s: float | None,
) -> tuple[list[SlideResult], Bucket, dict[str, Bucket], float]:
    audio_path = AUDIO_CACHE / f"{video_id}.opus"
    rows = ds_rows_by_video.get(video_id, [])
    if limit:
        rows = rows[:limit]
    if not rows:
        print(f"[FL-Acc] {video_id}: no GT rows — skipped")
        return [], Bucket(video_id), {}, 0.0

    print(f"\n[FL-Acc] === {video_id} === decoding {audio_path.name} …")
    pcm = load_audio_pcm(audio_path)
    print(f"[FL-Acc] {pcm.size / SAMPLE_RATE:.1f}s audio  |  {len(rows)} vocal slides")

    results: list[SlideResult] = []
    video_bucket = Bucket(video_id)
    sub_buckets = {
        "short(<=5)": Bucket("short(<=5)"),
        "mid(6-12)": Bucket("mid(6-12)"),
        "long(>12)": Bucket("long(>12)"),
    }

    t0 = time.monotonic()
    for i, row in enumerate(rows):
        start_s = float(row.get("start_time", 0) or 0)
        end_s = float(row.get("end_time", 0) or 0)
        if end_s <= start_s:
            continue
        gt_text = str(row.get("gurmukhi_text") or "").strip()
        gt_fl = fl_string(gt_text)
        if not gt_fl:
            continue
        audio = slice_pcm(pcm, start_s, end_s, max_window_s=audio_window_s)
        if audio.size == 0:
            continue
        try:
            segs = engine.transcribe(audio)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] slide {i}: {e}")
            continue
        asr_text = " ".join(s.text for s in segs).strip()
        asr_fl = fl_string(asr_text)
        matches = needleman_wunsch_matches(gt_fl, asr_fl)
        pos_matches = (
            sum(1 for a, b in zip(gt_fl, asr_fl) if a == b)
            if asr_fl and len(asr_fl) == len(gt_fl) else None
        )
        lcs = lcs_length(gt_fl, asr_fl)
        full_hit = gt_contained_as_substring(gt_fl, asr_fl)
        run = longest_gt_run_in_asr(gt_fl, asr_fl)
        r = SlideResult(
            slide_index=int(row.get("slide_index", i)),
            start_s=start_s, end_s=end_s,
            gt_text=gt_text, asr_text=asr_text,
            gt_fl=gt_fl, asr_fl=asr_fl,
            matches=matches, positional_matches=pos_matches,
            lcs=lcs, full_substring_hit=full_hit, longest_run=run,
        )
        results.append(r)
        video_bucket.add(r)
        sub_buckets[_bucket_for(len(gt_fl))].add(r)

        if i < 5 or (i + 1) % 50 == 0:
            print(
                f"  [{i+1:>3}/{len(rows)}] gt={gt_fl[:24]:<24} "
                f"asr={asr_fl[:24]:<24} "
                f"R={r.recall*100:5.1f}% P={r.precision*100:5.1f}%"
            )

    wall = time.monotonic() - t0
    print(f"[FL-Acc] {video_id}: {len(results)} slides scored in {wall:.1f}s")
    return results, video_bucket, sub_buckets, wall


def _print_bucket_table(buckets: dict[str, Bucket]) -> None:
    print(f"  {'bucket':<14}{'n':>5}{'GT_chars':>10}{'recall':>9}{'prec':>9}{'F1':>9}")
    for name in ("short(<=5)", "mid(6-12)", "long(>12)"):
        b = buckets.get(name)
        if not b or not b.n:
            continue
        s = b.summary()
        print(
            f"  {s['bucket']:<14}{s['n_slides']:>5}{s['gt_first_letters_total']:>10}"
            f"{s['recall_micro_pct']:>8.1f}%{s['precision_micro_pct']:>8.1f}%"
            f"{s['f1_micro_pct']:>8.1f}%"
        )


def _print_recall_distribution(per_slide_recall_pct: list[float]) -> None:
    if not per_slide_recall_pct:
        return
    s = sorted(per_slide_recall_pct)
    def q(p: float) -> float:
        return s[min(int(len(s) * p), len(s) - 1)]
    print("\n  Per-slide recall distribution (% of GT first-letters recovered):")
    print(f"    P10={q(0.10):.1f}%  P25={q(0.25):.1f}%  P50={q(0.50):.1f}%  "
          f"P75={q(0.75):.1f}%  P90={q(0.90):.1f}%")
    full = sum(1 for r in s if r >= 99.0) / len(s) * 100
    eighty = sum(1 for r in s if r >= 80.0) / len(s) * 100
    print(f"    Slides with ≥99% recall:  {full:.1f}%")
    print(f"    Slides with ≥80% recall:  {eighty:.1f}%")


# ── main ────────────────────────────────────────────────────────────────────

def main(
    video_ids: list[str],
    limit: int | None,
    save: bool,
    audio_window_s: float | None,
) -> None:
    missing = [v for v in video_ids if not (AUDIO_CACHE / f"{v}.opus").exists()]
    if missing:
        raise SystemExit(
            f"No cached audio for: {missing}. "
            f"Cached: {sorted(p.stem for p in AUDIO_CACHE.glob('*.opus'))}"
        )

    _wire_runtime()
    print(
        f"[FL-Acc] engine={config.whisper.engine} "
        f"model={config.whisper.hf_model_id} "
        f"precision={config.whisper.onnx_precision} "
        f"lm={config.whisper.lm_enabled}"
    )
    print(f"[FL-Acc] videos: {video_ids}")

    from src.transcription.factory import create_engine  # noqa: PLC0415
    engine = create_engine(config.whisper.engine)
    engine.load()

    from datasets import load_dataset  # noqa: PLC0415
    print(f"[FL-Acc] loading dataset for {len(video_ids)} video(s) …")
    ds = load_dataset("surindersinghssj/gurbani-kirtan-dataset-v2", split="train")
    audio_cols = [c for c in ds.column_names if "audio" in c.lower()]
    if audio_cols:
        ds = ds.remove_columns(audio_cols)
    wanted = set(video_ids)
    rows_by_video: dict[str, list] = {v: [] for v in video_ids}
    for r in ds:
        vid = str(r.get("video_id", ""))
        if vid not in wanted:
            continue
        if str(r.get("segment_type", "vocal")).lower() != "vocal":
            continue
        if float(r.get("match_score", 0) or 0) < 60.0:
            continue
        if not (r.get("gurmukhi_text") or "").strip():
            continue
        rows_by_video[vid].append(r)
    for vid in rows_by_video:
        rows_by_video[vid].sort(key=lambda r: int(r.get("slide_index", 0)))

    # ── per-video pass ──────────────────────────────────────────────────────
    per_video_results: dict[str, list[SlideResult]] = {}
    per_video_bucket: dict[str, Bucket] = {}
    per_video_subbuckets: dict[str, dict[str, Bucket]] = {}
    per_video_wall: dict[str, float] = {}

    overall = Bucket("ALL")
    overall_sub: dict[str, Bucket] = {
        "short(<=5)": Bucket("short(<=5)"),
        "mid(6-12)": Bucket("mid(6-12)"),
        "long(>12)": Bucket("long(>12)"),
    }

    for vid in video_ids:
        results, vb, sb, wall = _score_one_video(
            vid, rows_by_video, engine, limit, audio_window_s
        )
        per_video_results[vid] = results
        per_video_bucket[vid] = vb
        per_video_subbuckets[vid] = sb
        per_video_wall[vid] = wall
        for r in results:
            overall.add(r)
            overall_sub[_bucket_for(len(r.gt_fl))].add(r)

    # ── per-video report ────────────────────────────────────────────────────
    bar = "=" * 78
    print(f"\n{bar}")
    print("  FIRST-LETTER ACCURACY — per-video")
    print(bar)
    print(f"  {'video_id':<14}{'slides':>7}{'recall':>8}{'LCS_R':>8}"
          f"{'sub_hit':>9}{'prec':>8}{'F1':>7}")
    for vid in video_ids:
        vb = per_video_bucket[vid]
        if not vb.n:
            continue
        s = vb.summary()
        print(
            f"  {vid:<14}{s['n_slides']:>7}{s['recall_micro_pct']:>7.1f}%"
            f"{s['lcs_recall_micro_pct']:>7.1f}%{s['full_substring_hit_pct']:>8.1f}%"
            f"{s['precision_micro_pct']:>7.1f}%{s['f1_micro_pct']:>6.1f}%"
        )

    # ── aggregate report ────────────────────────────────────────────────────
    print(f"\n{bar}")
    print("  FIRST-LETTER ACCURACY — aggregate across all videos")
    print(bar)
    os_ = overall.summary()
    print(f"  Videos:                         {len([v for v in video_ids if per_video_bucket[v].n])}")
    print(f"  Slides scored:                  {os_['n_slides']}")
    print(f"  Total GT first-letters:         {os_['gt_first_letters_total']}")
    print(f"  Aligned matches:                {os_['aligned_matches']}")
    print(f"    Recall (micro, alignment-counted):  {os_['recall_micro_pct']}%")
    print(f"    Precision (micro):            {os_['precision_micro_pct']}%")
    print(f"    F1 (micro):                   {os_['f1_micro_pct']}%")
    print()
    print("  Matcher-realistic metrics (sequence must be preserved):")
    print(f"  ★ LCS-recall (micro, in-order GT first-letters in ASR):  "
          f"{os_['lcs_recall_micro_pct']}%")
    print(f"    LCS-recall median per slide:        {os_['lcs_recall_per_slide_median_pct']}%")
    print(f"  ★ Full-substring hit rate (GT FL ⊆ ASR FL as contiguous substring): "
          f"{os_['full_substring_hit_pct']}%")
    print(f"    Longest GT-run-in-ASR median (frac of GT):  "
          f"{os_['longest_run_frac_median_pct']}%")
    if os_["positional_acc_pct"] is not None:
        print(
            f"    Positional acc (n_eligible={os_['positional_eligible_slides']}):"
            f"  {os_['positional_acc_pct']}%"
        )

    print("\n  By GT length bucket (aggregate):")
    _print_bucket_table(overall_sub)

    all_recalls = [r.recall * 100 for results in per_video_results.values() for r in results]
    _print_recall_distribution(all_recalls)
    print(bar)

    # Quick interpretive verdict against the user's three regimes
    headline = os_["recall_micro_pct"]
    print("\n  Interpretation vs the three regimes:")
    if headline >= 95:
        print(f"  → {headline}% is in the ≥95% regime. Aux head won't help much;")
        print(f"    matcher should already be using first-letters as a hard filter.")
    elif headline >= 70:
        print(f"  → {headline}% is in the 70-90% regime. Aux head (approach #1) likely")
        print(f"    pays back — expect ~5-10pp lift and a step-change in matcher snap rate.")
    else:
        print(f"  → {headline}% is below 70%. Acoustic problem is fundamental — aux")
        print(f"    head alone won't be enough; investigate denoising / source separation.")

    if save:
        slug = video_ids[0] if len(video_ids) == 1 else f"multi_{len(video_ids)}v"
        out = RUN_DIR / f"fl_accuracy_{slug}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video_ids": video_ids,
            "engine": config.whisper.engine,
            "model": config.whisper.hf_model_id,
            "precision": config.whisper.onnx_precision,
            "lm_enabled": config.whisper.lm_enabled,
            "wall_s_per_video": {k: round(v, 1) for k, v in per_video_wall.items()},
            "overall": os_,
            "buckets_overall": {k: v.summary() for k, v in overall_sub.items()},
            "per_video": {
                vid: {
                    "overall": per_video_bucket[vid].summary(),
                    "buckets": {k: v.summary() for k, v in per_video_subbuckets[vid].items()},
                    "per_slide": [
                        {
                            "slide_index": r.slide_index,
                            "start_s": r.start_s, "end_s": r.end_s,
                            "gt_fl": r.gt_fl, "asr_fl": r.asr_fl,
                            "gt_text": r.gt_text, "asr_text": r.asr_text,
                            "matches": r.matches,
                            "recall_pct": round(r.recall * 100, 1),
                            "precision_pct": round(r.precision * 100, 1),
                        }
                        for r in per_video_results[vid]
                    ],
                }
                for vid in video_ids if per_video_bucket[vid].n
            },
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n[FL-Acc] saved → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video_ids", nargs="*",
                   help="YouTube video ids (must have cached opus). "
                        "Defaults to all cached videos when --all-cached is set "
                        "and no ids are given.")
    p.add_argument("--all-cached", action="store_true",
                   help="Score every cached video in tests/eval/cache/audio")
    p.add_argument("--limit", type=int, default=None,
                   help="Max slides per video (useful for fast iteration)")
    p.add_argument("--audio-window-s", type=float, default=12.0,
                   help="Cap each slide's audio to a centered window of this many "
                        "seconds (default 12). Set 0 to use the full slide span — "
                        "but precision will be deflated by ASR re-emitting "
                        "duplicate lines on long sustained slides.")
    p.add_argument("--no-save", action="store_true", help="Skip JSON report")
    args = p.parse_args()

    if args.all_cached:
        vids = sorted(p.stem for p in AUDIO_CACHE.glob("*.opus"))
    elif args.video_ids:
        vids = args.video_ids
    else:
        vids = ["-Dyi8-Qyx4I"]  # back-compat default
    window = None if args.audio_window_s and args.audio_window_s <= 0 else args.audio_window_s
    main(vids, args.limit, save=not args.no_save, audio_window_s=window)
