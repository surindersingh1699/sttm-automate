"""Prototype: phoneme-ngram inverted index over the full ShabadOS DB.

Builds once, ~6 s. Queries: ~5 ms for top-50 lines.

Each (shabad_id, line_idx) is a "node" — the unit of state for the particle
filter downstream.
"""
from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from gurmukhiutils.unicode import unicode as _ascii_to_unicode_gurmukhi

from tests.eval.proto_phonetic import char_ngrams, encode

_DB = Path(__file__).parent.parent.parent / "database.sqlite"
_SYNTH = 100_000_000


@dataclass(frozen=True)
class LineNode:
    shabad_id: int
    line_idx: int          # 0-based within shabad
    line_order: int        # global ordering
    gurmukhi_unicode: str
    phoneme: str
    pgrams3: frozenset[str]
    pgrams4: frozenset[str]


@dataclass
class Corpus:
    nodes: list[LineNode]                              # all lines, ordered
    by_shabad: dict[int, list[int]]                     # shabad_id → [node_idx]
    inv_index_3: dict[str, list[int]]                  # 3gram → [node_idx]
    inv_index_4: dict[str, list[int]]                  # 4gram → [node_idx]

    def line_count(self) -> int:
        return len(self.nodes)

    def shabad_count(self) -> int:
        return len(self.by_shabad)


def build_corpus(db_path: Path = _DB, verbose: bool = True) -> Corpus:
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        f"""SELECT
              COALESCE(s.sttm_id, s.order_id + {_SYNTH}) AS sid,
              s.id AS sstr,
              l.gurmukhi AS asciiline,
              l.order_id AS oid
            FROM lines l
            JOIN shabads s ON l.shabad_id = s.id
            ORDER BY s.id, l.order_id"""
    ).fetchall()
    conn.close()

    nodes: list[LineNode] = []
    by_shabad: dict[int, list[int]] = defaultdict(list)
    inv3: dict[str, list[int]] = defaultdict(list)
    inv4: dict[str, list[int]] = defaultdict(list)

    current_shabad: int | None = None
    line_idx = 0
    for sid, _sstr, ascii_line, oid in rows:
        if sid != current_shabad:
            current_shabad = sid
            line_idx = 0
        try:
            uni = _ascii_to_unicode_gurmukhi(ascii_line) if ascii_line else ""
        except Exception:
            uni = ascii_line or ""
        phoneme = encode(uni)
        g3 = char_ngrams(phoneme, 3)
        g4 = char_ngrams(phoneme, 4)
        node = LineNode(
            shabad_id=int(sid),
            line_idx=line_idx,
            line_order=int(oid),
            gurmukhi_unicode=uni,
            phoneme=phoneme,
            pgrams3=frozenset(g3),
            pgrams4=frozenset(g4),
        )
        idx = len(nodes)
        nodes.append(node)
        by_shabad[node.shabad_id].append(idx)
        for g in g3:
            inv3[g].append(idx)
        for g in g4:
            inv4[g].append(idx)
        line_idx += 1

    if verbose:
        print(
            f"[Corpus] {len(nodes)} lines / {len(by_shabad)} shabads / "
            f"{len(inv3)} 3-grams / {len(inv4)} 4-grams in {time.time()-t0:.1f}s"
        )
    return Corpus(nodes=nodes, by_shabad=dict(by_shabad), inv_index_3=dict(inv3), inv_index_4=dict(inv4))


def search(
    corpus: Corpus,
    transcript: str,
    top_k: int = 50,
    min_overlap: float = 0.18,
) -> list[tuple[int, float]]:
    """Return [(node_idx, score)] for top-K lines by phoneme 3-gram overlap.

    Two-stage: candidate union from inverted index, then exact overlap-coef.
    """
    pho = encode(transcript)
    if not pho:
        return []
    q3 = char_ngrams(pho, 3)
    if not q3:
        return []

    # Tally hit counts via inverted index, then re-score top candidates.
    counts: dict[int, int] = defaultdict(int)
    for g in q3:
        for nidx in corpus.inv_index_3.get(g, ()):
            counts[nidx] += 1

    if not counts:
        return []

    # Sort by raw hit count then by node_idx to be deterministic across runs.
    # Widen the rescoring pool so a high-overlap line never gets truncated by
    # a hit-count tie.
    coarse = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: max(top_k * 12, 200)]

    qsize = len(q3)
    scored: list[tuple[int, float]] = []
    for nidx, _hits in coarse:
        cand = corpus.nodes[nidx].pgrams3
        if not cand:
            continue
        inter = len(q3 & cand)
        denom = min(qsize, len(cand))
        if denom == 0:
            continue
        score = inter / denom
        if score >= min_overlap:
            scored.append((nidx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
