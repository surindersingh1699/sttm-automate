"""Load eval sessions from surindersinghssj/gurbani-kirtan-dataset-v2.

Each row in that dataset is one slide-aligned audio segment:
  gurmukhi_text  – canonical Gurbani line (ground truth)
  gurmukhi_ocr   – raw OCR text from the video frame
  start_time     – second offset inside the YouTube video (authoritative GT)
  end_time       – end second (authoritative GT)
  slide_index    – ordering key within a video
  video_id       – YouTube video ID
  segment_type   – "vocal" | "instrumental" | "silent"
  match_score    – 0-100 OCR→DB confidence
  channel, kirtan_style – metadata

We now build *session-level* SessionDescriptor objects — one per YouTube video
(or per continuous vocal stretch within a video). The per-slide audio array from
the dataset is intentionally discarded; the eval fetches whole-video audio from
YouTube instead (yt-dlp / Playwright) using the video_id.

Ground truth:
- start_time / end_time are preserved as GroundTruthEvent.start_s / end_s
  (normalised to 0-based virtual time from audio_t0).
- The first and last shabad of every session are excluded to avoid
  intro/outro irregularities.
- Instrumental/silent rows are kept in the GT timeline as None-shabad events
  so the pipeline experiences realistic gaps.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.transcription.transliterate import extract_first_letters, gurmukhi_to_ascii

_DB = Path(__file__).parent.parent.parent / "database.sqlite"
_SYNTH = 100_000_000
_DATASET = "surindersinghssj/gurbani-kirtan-dataset-v2"


# ── data models ────────────────────────────────────────────────────────────

@dataclass
class GroundTruthEvent:
    """One slide that was on screen during [start_s, end_s) in the eval session."""
    start_s: float           # seconds from audio_t0 (virtual eval time)
    end_s: float
    gurmukhi_text: str
    shabad_id: int | None    # resolved integer STTM shabad id; None for non-vocal
    line_order: int | None   # lines.order_id (global ordering key)
    line_idx_in_shabad: int  # 0-based index within shabad verses
    match_score: float       # 0-100 OCR confidence
    segment_type: str = "vocal"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def is_vocal(self) -> bool:
        return self.segment_type == "vocal" and self.shabad_id is not None


@dataclass
class SessionDescriptor:
    """Describes one eval session backed by a YouTube video.

    All time coordinates are in virtual seconds (0 = start of the session,
    which corresponds to audio_t0 seconds into the source YouTube video).
    """
    session_id: str
    video_id: str
    audio_t0: float          # YT-video absolute second where session audio starts
    audio_t_end: float       # YT-video absolute second where session audio ends
    gt_timeline: list[GroundTruthEvent]   # sorted by start_s (virtual time)
    channel: str = ""
    kirtan_style: str = ""

    @property
    def duration_s(self) -> float:
        return self.audio_t_end - self.audio_t0

    @property
    def vocal_gt(self) -> list[GroundTruthEvent]:
        return [e for e in self.gt_timeline if e.is_vocal]

    @property
    def shabad_transitions(self) -> list[tuple[float, int | None, int | None]]:
        """List of (virtual_t_s, old_shabad_id, new_shabad_id) at each GT shabad change."""
        transitions = []
        prev_sid: int | None = None
        for ev in self.gt_timeline:
            if ev.shabad_id != prev_sid:
                transitions.append((ev.start_s, prev_sid, ev.shabad_id))
                prev_sid = ev.shabad_id
        return transitions


# ── DB helpers ─────────────────────────────────────────────────────────────

def _open_db() -> sqlite3.Connection:
    return sqlite3.connect(_DB)


def _resolve_text_to_ids(text: str) -> tuple[int | None, int | None, int]:
    """Return (shabad_int_id, line_order, line_idx_in_shabad) for a canonical line.

    Searches by first-letter codes then disambiguates with Unicode text comparison.
    """
    if not text or not text.strip():
        return None, None, 0

    fl = gurmukhi_to_ascii(extract_first_letters(text.strip()))
    if len(fl) < 2:
        return None, None, 0

    try:
        conn = _open_db()
        cur = conn.cursor()

        # Exact first-letter match first
        rows = cur.execute(
            f"""SELECT l.order_id,
                       COALESCE(s.sttm_id, s.order_id + {_SYNTH}) AS shabad_int_id,
                       s.id AS shabad_str_id,
                       l.first_letters
                FROM lines l
                JOIN shabads s ON l.shabad_id = s.id
               WHERE l.first_letters = ?
               LIMIT 20""",
            (fl,),
        ).fetchall()

        if not rows:
            # Prefix fallback (first 4 chars)
            rows = cur.execute(
                f"""SELECT l.order_id,
                           COALESCE(s.sttm_id, s.order_id + {_SYNTH}) AS shabad_int_id,
                           s.id AS shabad_str_id,
                           l.first_letters
                    FROM lines l
                    JOIN shabads s ON l.shabad_id = s.id
                   WHERE l.first_letters LIKE ?
                   LIMIT 20""",
                (fl[:4] + "%",),
            ).fetchall()

        if not rows:
            conn.close()
            return None, None, 0

        best = next((r for r in rows if r[3] == fl), rows[0])
        line_order, shabad_int_id, shabad_str_id = best[0], best[1], best[2]

        idx_row = cur.execute(
            """SELECT COUNT(*) FROM lines l2
               JOIN shabads s2 ON l2.shabad_id = s2.id
              WHERE s2.id = ? AND l2.order_id <= ?""",
            (shabad_str_id, line_order),
        ).fetchone()
        line_idx = max(0, (idx_row[0] if idx_row else 1) - 1)

        conn.close()
        return shabad_int_id, line_order, line_idx

    except Exception:
        return None, None, 0


def _resolve_bulk(texts: list[str]) -> list[tuple[int | None, int | None, int]]:
    return [_resolve_text_to_ids(t) for t in texts]


# ── dataset loading ─────────────────────────────────────────────────────────

def load_eval_sessions(
    dataset_name: str = _DATASET,
    split: str = "train",
    limit_videos: int | None = None,
    min_match_score: float = 60.0,
    video_ids: list[str] | None = None,
) -> list[SessionDescriptor]:
    """Load SessionDescriptor objects from the HF dataset — one per YouTube video.

    Unlike the old VideoEvalJob loader, this does NOT:
    - cap session duration at 90 s
    - synthesize timestamps from concatenated audio lengths
    - include per-slide audio arrays

    Instead it:
    - Preserves dataset start_time/end_time as authoritative GT
    - Drops the first and last shabad of each session
    - Keeps instrumental/silent rows in the timeline (no vocal gap → shabad_id=None)
    """
    from datasets import load_dataset

    print(f"[Dataset] Loading {dataset_name} …")
    ds = load_dataset(dataset_name, split=split)
    audio_cols = [c for c in ds.column_names if "audio" in c.lower()]
    if audio_cols:
        ds = ds.remove_columns(audio_cols)
    print(f"[Dataset] {len(ds)} rows loaded.")

    by_video: dict[str, list[dict]] = {}
    for row in ds:
        vid = str(row.get("video_id", ""))
        if not vid:
            continue
        if video_ids and vid not in video_ids:
            continue
        by_video.setdefault(vid, []).append(row)

    for vid in by_video:
        by_video[vid].sort(key=lambda r: int(r.get("slide_index", 0)))

    all_vids = list(by_video.keys())
    if limit_videos:
        all_vids = all_vids[:limit_videos]

    sessions: list[SessionDescriptor] = []
    for vid in all_vids:
        rows = by_video[vid]
        session = _build_session(vid, rows, min_match_score=min_match_score)
        if session is not None:
            sessions.append(session)

    print(f"[Dataset] Built {len(sessions)} eval sessions from {len(all_vids)} videos.")
    return sessions


def _build_session(
    video_id: str,
    rows: list[dict],
    min_match_score: float,
) -> SessionDescriptor | None:
    """Build one SessionDescriptor from all rows of a video.

    Steps:
    1. Build GroundTruthEvent for every row (vocal + non-vocal), using the
       dataset's start_time/end_time as ground truth.
    2. Resolve canonical texts to (shabad_id, line_order, line_idx).
    3. Identify the first and last shabad ID that appear in vocal rows and
       exclude all events belonging to those shabads.
    4. Trim audio_t0 / audio_t_end to the remaining events.
    5. Normalise all event times to virtual (0-based from audio_t0).
    """
    if not rows:
        return None

    # --- resolve all vocal row texts up front (batch) ---
    vocal_texts = [
        str(r.get("gurmukhi_text", "") or "")
        if str(r.get("segment_type", "vocal")).lower() == "vocal"
        else ""
        for r in rows
    ]
    resolved = _resolve_bulk(vocal_texts)

    # --- build raw events using YT absolute timestamps ---
    raw: list[GroundTruthEvent] = []
    for row, (shabad_id, line_order, line_idx) in zip(rows, resolved):
        seg_type = str(row.get("segment_type", "vocal")).lower()
        score = float(row.get("match_score", 0) or 0)

        # Skip very low-confidence vocal rows from GT (keep non-vocal as context)
        if seg_type == "vocal" and score < min_match_score:
            continue

        start_t = float(row.get("start_time", 0) or 0)
        end_t = float(row.get("end_time", start_t) or start_t)
        if end_t <= start_t:
            end_t = start_t + float(row.get("duration", 1) or 1)

        gt_shabad_id = shabad_id if seg_type == "vocal" else None

        raw.append(GroundTruthEvent(
            start_s=start_t,
            end_s=end_t,
            gurmukhi_text=str(row.get("gurmukhi_text", "") or "").strip(),
            shabad_id=gt_shabad_id,
            line_order=line_order,
            line_idx_in_shabad=line_idx,
            match_score=score,
            segment_type=seg_type,
        ))

    if not raw:
        return None

    raw.sort(key=lambda e: e.start_s)

    # --- identify first and last shabad IDs (vocal only) ---
    vocal_ids = [e.shabad_id for e in raw if e.is_vocal]
    if not vocal_ids:
        return None

    # Collect distinct shabads in order of first appearance
    ordered_shabads: list[int] = []
    for sid in vocal_ids:
        if not ordered_shabads or sid != ordered_shabads[-1]:
            if sid not in ordered_shabads:
                ordered_shabads.append(sid)

    excluded_ids: set[int | None] = set()
    if len(ordered_shabads) >= 3:
        excluded_ids.add(ordered_shabads[0])   # first shabad (intro)
        excluded_ids.add(ordered_shabads[-1])  # last shabad (outro)
    elif len(ordered_shabads) == 2:
        excluded_ids.add(ordered_shabads[0])   # only exclude first (intro)

    # --- filter events, replacing excluded vocal rows with None-shabad markers ---
    events: list[GroundTruthEvent] = []
    for ev in raw:
        if ev.is_vocal and ev.shabad_id in excluded_ids:
            # Keep as non-vocal context (the audio still plays) but zero out GT
            events.append(GroundTruthEvent(
                start_s=ev.start_s,
                end_s=ev.end_s,
                gurmukhi_text=ev.gurmukhi_text,
                shabad_id=None,
                line_order=None,
                line_idx_in_shabad=0,
                match_score=ev.match_score,
                segment_type="excluded",
            ))
        else:
            events.append(ev)

    # Determine audio window: trim leading/trailing non-scorable events
    scorable = [e for e in events if e.is_vocal]
    if not scorable:
        return None

    audio_t0 = scorable[0].start_s
    audio_t_end = scorable[-1].end_s

    # --- normalise to virtual time (0 = audio_t0) ---
    for ev in events:
        ev.start_s = round(ev.start_s - audio_t0, 3)
        ev.end_s = round(ev.end_s - audio_t0, 3)

    first_row = rows[0]
    return SessionDescriptor(
        session_id=video_id,
        video_id=video_id,
        audio_t0=audio_t0,
        audio_t_end=audio_t_end,
        gt_timeline=events,
        channel=str(first_row.get("channel", "") or ""),
        kirtan_style=str(first_row.get("kirtan_style", "") or ""),
    )


# ── ground truth lookup ─────────────────────────────────────────────────────

def gt_at(timeline: list[GroundTruthEvent], t_s: float) -> GroundTruthEvent | None:
    """Return the ground truth event covering virtual time t_s, or None."""
    for ev in timeline:
        if ev.start_s <= t_s < ev.end_s:
            return ev
    return None


def available_videos(dataset_name: str = _DATASET) -> list[dict]:
    """Return a lightweight list of {video_id, channel, num_rows} for the UI."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")
    by_video: dict[str, dict] = {}
    for row in ds:
        vid = str(row.get("video_id", ""))
        if not vid:
            continue
        if vid not in by_video:
            by_video[vid] = {
                "video_id": vid,
                "channel": str(row.get("channel", "") or ""),
                "kirtan_style": str(row.get("kirtan_style", "") or ""),
                "num_rows": 0,
            }
        by_video[vid]["num_rows"] += 1
    return list(by_video.values())


# ── backwards-compat shim for anything still importing VideoEvalJob ────────

VideoEvalJob = SessionDescriptor  # alias so old imports don't crash during migration
