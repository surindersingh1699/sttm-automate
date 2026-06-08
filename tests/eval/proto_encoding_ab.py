"""End-to-end A/B of retrieval + line-matching encodings on REAL ASR output.

Two questions for the final matcher design:
  1. RETRIEVAL recall — does the candidate pool even contain the correct line?
     Compare FL-gram vs char-gram(matras-kept) vs word inverted indexes.
  2. SELECTION — once retrieved, does the encoding rank the correct line #1?
     Compare matra / exact / max(matra,exact), measured END-TO-END (no GT
     force-add — if retrieval misses it, it's a miss).

Encodings (similarity in [0,1] between ASR hypothesis and a DB line):
  FL        first-letters normalized-LCS coverage              (coarse baseline)
  exact     exact word-set overlap coefficient (matras kept)
  matra     normalized-LCS over full chars, matras KEPT        (N=100 selection winner)
  skeleton  normalized-LCS over chars, matras STRIPPED
  soft      skeleton + phonetic equality (ਬ↔ਵ ਨ↔ਣ ਤ↔ਟ ਦ↔ਡ)
  maxme     max(matra, exact)                                  (weightless ensemble)

Data: surindersinghssj/gurbani-kirtan-dataset-v2 (test+validation). Audio capped
to 14s and run through the real ASR (IndicConformer). Pairs are cached to disk so
ASR runs ONCE; later runs reuse the cache and are instant.

Run:  .venv/bin/python -u -m tests.eval.proto_encoding_ab
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import numpy as np

from src.matcher.offline_search import _to_unicode
from src.matcher.scorer import _GURMUKHI_MATRAS
from src.transcription.transliterate import (
    extract_first_letters,
    gurmukhi_to_ascii,
    normalize_for_fullword_search,
)

_DB = Path(__file__).parent.parent.parent / "database.sqlite"
_DATASET = "surindersinghssj/gurbani-kirtan-dataset-v2"
_SPLITS = "test+validation"
_N = 100
_SEED = 7
_POOL_CAP = 300    # pool size for recall measurement
_SCORE_CAP = 50    # candidates actually scored/ranked (what production would rank)
_MAX_DF = 800      # drop grams/words appearing in >this many lines (uninformative)
_QUERY_CAP = 80    # cap ASR query chars — repeated 14s-window text adds cost, not signal
_PAIRS_CACHE = Path(__file__).parent / "cache" / "encoding_ab_pairs.json"

_PHONETIC_PAIRS = [("ਬ", "ਵ"), ("ਨ", "ਣ"), ("ਤ", "ਟ"), ("ਦ", "ਡ")]
_PHONETIC_EQ: dict[str, str] = {}
for _a, _b in _PHONETIC_PAIRS:
    _PHONETIC_EQ.setdefault(_a, _a)
    _PHONETIC_EQ[_b] = _a


def _strip_matras(text: str) -> str:
    return "".join(ch for ch in text if ch not in _GURMUKHI_MATRAS)


def _phonetic_fold(text: str) -> str:
    return "".join(_PHONETIC_EQ.get(ch, ch) for ch in text)


def _grams(s: str, sizes=(3, 4)) -> set[str]:
    s = s or ""
    out: set[str] = set()
    for size in sizes:
        if len(s) >= size:
            out.update(s[i:i + size] for i in range(len(s) - size + 1))
        elif s:
            out.add(s)
    return out


def _lcs(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0] * (len(b) + 1)
        for j, cb in enumerate(b):
            curr[j + 1] = prev[j] + 1 if ca == cb else max(prev[j + 1], curr[j])
        prev = curr
    return prev[len(b)]


def _lcs_cov(query: str, target: str) -> float:
    if not target:
        return 0.0
    return _lcs(query, target) / len(target)


def _overlap_coeff(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ── encodings: (asr_repr, line) → score ──────────────────────────────────────

def enc_fl(fl, words, full, skel, cand):       return _lcs_cov(fl, cand["fl_ascii"])
def enc_exact(fl, words, full, skel, cand):    return _overlap_coeff(words, cand["words"])
def enc_matra(fl, words, full, skel, cand):    return _lcs_cov(full, cand["full"])
def enc_skeleton(fl, words, full, skel, cand): return _lcs_cov(skel, cand["skel"])
def enc_soft(fl, words, full, skel, cand):     return _lcs_cov(_phonetic_fold(skel), cand["skel_fold"])
def enc_maxme(fl, words, full, skel, cand):
    return max(_lcs_cov(full, cand["full"]), _overlap_coeff(words, cand["words"]))


ENCODINGS = {
    "FL": enc_fl, "exact": enc_exact, "matra": enc_matra,
    "skeleton": enc_skeleton, "soft": enc_soft, "maxme": enc_maxme,
}


def _build_db():
    """Load all DB lines; build FL/char/word inverted indexes + GT lookup."""
    conn = sqlite3.connect(str(_DB))
    rows = conn.execute("SELECT gurmukhi, first_letters FROM lines").fetchall()
    conn.close()

    lines = []
    fl_index: dict[str, list[int]] = {}
    char_index: dict[str, list[int]] = {}
    word_index: dict[str, list[int]] = {}
    gt_by_norm: dict[str, int] = {}
    for i, (ascii_g, fl_ascii) in enumerate(rows):
        fl_ascii = fl_ascii or ""
        unicode_text = _to_unicode(ascii_g or "")
        norm = normalize_for_fullword_search(unicode_text)
        words = set(w for w in norm.split() if len(w) >= 2)
        full = norm.replace(" ", "")
        skel = _strip_matras(full)
        lines.append({
            "fl_ascii": fl_ascii, "unicode": unicode_text, "words": words,
            "full": full, "skel": skel, "skel_fold": _phonetic_fold(skel),
        })
        if full and full not in gt_by_norm:
            gt_by_norm[full] = i
        for g in _grams(fl_ascii):
            fl_index.setdefault(g, []).append(i)
        for g in _grams(full):
            char_index.setdefault(g, []).append(i)
        for w in words:
            word_index.setdefault(w, []).append(i)
    # Drop high-document-frequency grams/words: a gram in >_MAX_DF lines is
    # uninformative for retrieval and dominates the posting-union cost.
    char_index = {g: p for g, p in char_index.items() if len(p) <= _MAX_DF}
    word_index = {w: p for w, p in word_index.items() if len(p) <= _MAX_DF}
    return lines, fl_index, char_index, word_index, gt_by_norm


def _pool(query_grams: set[str], index: dict[str, list[int]], cap: int) -> list[int]:
    hits: dict[int, int] = {}
    for g in query_grams:
        for idx in index.get(g, ()):
            hits[idx] = hits.get(idx, 0) + 1
    ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
    return [idx for idx, _ in ranked[:cap]]


def _find_gt(gt_text: str, lines, gt_by_norm) -> int | None:
    key = normalize_for_fullword_search(gt_text).replace(" ", "")
    if key in gt_by_norm:
        return gt_by_norm[key]
    best_i, best = None, 0.0
    for i, ln in enumerate(lines):
        c = _lcs_cov(key, ln["full"])
        if c > best:
            best, best_i = c, i
    return best_i if best >= 0.85 else None


def _asr_reprs(asr_text: str):
    fl_ascii = gurmukhi_to_ascii(extract_first_letters(asr_text))
    norm = normalize_for_fullword_search(asr_text)
    words = set(w for w in norm.split() if len(w) >= 2)
    full = norm.replace(" ", "")[:_QUERY_CAP]
    return fl_ascii, words, full, _strip_matras(full)


def _collect_pairs(lines, gt_by_norm) -> list[tuple[str, str]]:
    """Load cached (asr_text, gt_text) pairs, or run ASR once and cache them."""
    if _PAIRS_CACHE.exists():
        raw = json.loads(_PAIRS_CACHE.read_text())
        print(f"      Loaded {len(raw)} cached (ASR, GT) pairs from {_PAIRS_CACHE.name}.")
        return [(a, g) for a, g in raw]

    print("      No cache — loading ASR engine + dataset (slow, one time) …")
    from src.transcription.factory import create_engine
    engine = create_engine("indicconformer")
    engine.load()
    from datasets import load_dataset
    ds = load_dataset(_DATASET, split=_SPLITS)
    cand = [
        r for r in ds
        if r.get("segment_type") == "vocal"
        and 1.5 <= (r.get("duration") or 0) <= 25.0
        and r.get("gurmukhi_text")
        and len(normalize_for_fullword_search(r["gurmukhi_text"]).split()) >= 3
    ]
    random.seed(_SEED)
    random.shuffle(cand)

    pairs: list[tuple[str, str]] = []
    for r in cand:
        if len(pairs) >= _N:
            break
        audio = np.asarray(r["audio"]["array"], dtype=np.float32)[: 14 * 16000]
        try:
            segs = engine.transcribe(audio)
        except Exception:
            continue
        asr_text = " ".join(s.text for s in segs).strip()
        if not asr_text or _find_gt(r["gurmukhi_text"], lines, gt_by_norm) is None:
            continue
        pairs.append((asr_text, r["gurmukhi_text"]))
    _PAIRS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _PAIRS_CACHE.write_text(json.dumps(pairs, ensure_ascii=False, indent=1))
    print(f"      Collected + cached {len(pairs)} pairs → {_PAIRS_CACHE.name}.")
    return pairs


def main():
    print("[1/3] Loading DB + FL/char/word indices …")
    lines, fl_index, char_index, word_index, gt_by_norm = _build_db()
    print(f"      {len(lines)} DB lines indexed.")

    print("[2/3] Collecting (ASR, GT) pairs …")
    raw_pairs = _collect_pairs(lines, gt_by_norm)
    pairs = []
    for asr_text, gt_text in raw_pairs:
        gt = _find_gt(gt_text, lines, gt_by_norm)
        if gt is not None:
            pairs.append((asr_text, gt_text, gt))
    n = len(pairs)
    print(f"      {n} usable pairs.\n")

    # Pre-compute ASR reprs + per-retrieval pools (cap _POOL_CAP) once per pair.
    import time as _t
    prepped = []
    retrievals = {"fl": fl_index, "char": char_index, "word": word_index}
    print(f"      char_index grams={len(char_index)} word_index={len(word_index)} "
          f"(after DF<= {_MAX_DF} prune)", flush=True)
    _t0 = _t.monotonic()
    for k, (asr_text, gt_text, gt) in enumerate(pairs):
        fl, words, full, skel = _asr_reprs(asr_text)
        gq = {"fl": _grams(fl), "char": _grams(full), "word": words}
        pools = {name: _pool(gq[name], retrievals[name], _POOL_CAP) for name in retrievals}
        prepped.append((fl, words, full, skel, gt, pools))
        if (k + 1) % 10 == 0:
            print(f"      prepped {k + 1}/{len(pairs)} ({_t.monotonic() - _t0:.1f}s)", flush=True)

    # ── retrieval recall ─────────────────────────────────────────────────────
    print("[3/3] === RETRIEVAL recall (is GT in the pool?) ===")
    print(f"{'retrieval':10s}  recall@{_SCORE_CAP}   recall@{_POOL_CAP}")
    print("-" * 38)
    for name in retrievals:
        r50 = sum(1 for *_, gt, pools in prepped if gt in set(pools[name][:_SCORE_CAP]))
        rcap = sum(1 for *_, gt, pools in prepped if gt in set(pools[name]))
        print(f"{name:10s}    {r50:3d}/{n}     {rcap:3d}/{n}")

    # ── end-to-end: retrieval × selection, NO force-add ──────────────────────
    sel_names = ["matra", "exact", "maxme"]
    print("\n=== END-TO-END top1 (retrieve, then select; GT NOT force-added) ===")
    print(f"{'retrieval':10s}  " + "  ".join(f"{s:>8s}" for s in sel_names))
    print("-" * 40)
    for rname in retrievals:
        cells = []
        for sname in sel_names:
            fn = ENCODINGS[sname]
            hits = 0
            for fl, words, full, skel, gt, pools in prepped:
                pool = pools[rname][:_SCORE_CAP]
                if gt not in pool:
                    continue  # retrieval miss → end-to-end miss
                scored = sorted(((fn(fl, words, full, skel, lines[i]), i) for i in pool),
                                key=lambda x: x[0], reverse=True)
                if scored and scored[0][1] == gt:
                    hits += 1
            cells.append(f"{hits:3d}/{n}")
        print(f"{rname:10s}  " + "  ".join(f"{c:>8s}" for c in cells))

    # ── selection-only reference (best retrieval pool, GT force-added) ───────
    print(f"\n=== SELECTION-only (char pool, GT force-added) — isolates ranking power ===")
    print(f"{'encoding':10s}  top1     mean-rank")
    print("-" * 32)
    for name in ENCODINGS:
        fn = ENCODINGS[name]
        ranks = []
        for fl, words, full, skel, gt, pools in prepped:
            pool = set(pools["char"][:_SCORE_CAP]); pool.add(gt)
            scored = sorted(((fn(fl, words, full, skel, lines[i]), i) for i in pool),
                            key=lambda x: x[0], reverse=True)
            ranks.append(next(k for k, (_, i) in enumerate(scored) if i == gt))
        top1 = sum(1 for r in ranks if r == 0)
        mr = sum(ranks) / max(1, len(ranks))
        print(f"{name:10s}  {top1:3d}/{n}     {mr:5.1f}")


if __name__ == "__main__":
    main()
