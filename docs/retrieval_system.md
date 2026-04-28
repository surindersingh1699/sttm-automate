---
name: Full Retrieval & Matching System
description: Complete reference for the 9-strategy retrieval pipeline, candidate scoring, locked-state line scoring, and all config flags. Covers current code as of 2026-04-26.
type: project
originSessionId: 4357541d-c8db-44a8-85fe-e6afbd1a2476
---
# Full Retrieval & Matching System

## Overview

Audio → Whisper → Gurmukhi transcript → `extract_first_letters()` → **9-strategy retrieval** → candidate list → **`ConfidenceScorer`** → ranked top-N → tracker → LOCKED state → **line pointer scoring** → STTM navigate.

Database: `database.sqlite` (ShabadOS SQLite, ~60,555 lines, 5,549 shabads).  
Shabad ID: `COALESCE(s.sttm_id, s.order_id + 100_000_000)` — non-SGGS rows get synthetic IDs.

---

## Pre-search: Kirtan Filler Strip

Before any strategy runs, `_strip_filler_words(transcript_text)` removes known kirtan prefix/suffix words from the transcript edges and re-derives `first_letters` from the cleaned text.

**Filler set** (`_KIRTAN_FILLER`):
- ਵਾਹਿਗੁਰੂ, ਵਾਹਿਗੁਰ, ਵਾਹੁਗੁਰੂ, ਵਾਹਗੁਰੂ
- ਜੀਉ, ਜੀਓ, ਜੀ
- ਵਾਹੁ, ਵਾਹ
- ਰਾਮ
- ਸਤਿਨਾਮੁ, ਸਤਿਨਾਮ

**Why:** "ਵਾਹਿਗੁਰੂ ਅੰਮ੍ਰਿਤ ਨਾਮੁ..." adds ਵ to the FL string, changing score from 0.900 → 0.783 and shrinking the margin to #2 from 0.23 → 0.097.

---

## 9 Retrieval Strategies

All strategies fire inside `OfflineShabadSearcher.search(first_letters, transcript_text)`.  
Results are deduplicated by `sttm_id` via `_add_unique`. Each candidate carries a `retrieval_sources` set showing which strategies found it.

### Strategy 1 — `type0`: FL prefix match
- SQL: `first_letters LIKE 'query%'`
- Always fires when `len(first_letters) >= 3`.
- Best for: clean full-line transcripts where ASR captures from the start of the DB line.

### Strategy 2 — `type1`: FL substring/contains match
- SQL: `first_letters LIKE '%query%'`
- Fires when strategy 1 returns < 3 candidates.
- Best for: mid-shabad entry — ragi starts at a chorus line rather than line 1.

### Strategy 2b — `rotation`: cyclic rotations (≤4-char queries)
- Generates all cyclic rotations of the query (e.g. "tmm" → "mmt", "mtm") and runs substring search for each.
- Fires when < 2 candidates found AND `len(query) <= 4`.
- Best for: word-order mismatch between ASR transcript and DB line.

### Strategy 3 — `type0_sub` / `type1_sub`: shorter substring fallback
- Takes 4-char prefix, suffix, and mid-slices of the query; runs prefix + contains on each.
- Fires when < 2 candidates and `len(query) > 4`.
- Best for: Whisper prepends extra words before the tuk, so the full FL doesn't substring-match.

### Strategy 4 — `type2`: full-word phrase search
- `normalize_for_fullword_search(transcript_text)` → strip matras / normalize script → query `lines.gurmukhi` (ASCII) for exact word sequences.
- Fires whenever `transcript_text` is non-empty.
- Best for: clean transcripts with uncommon word combos.

### Strategy 5 — `multiline` / `multiline3`: multi-line split search
- Splits the FL query in half (2-way) or thirds (3-way), requires BOTH/ALL parts to hit consecutive DB lines within the same shabad.
- Fires when `len(first_letters) >= multi_line_min_query_length` (default 12; 3-way: 18).
- Best for: nitnem / dense recitation where one 3 s window spans 2–3 DB lines.

### Strategy 6 — `type_rotation`: rotation for 3–4 char queries (always-on)
- Unconditional rotation search for short queries (3–4 chars), separate from Strategy 2b.
- Best for: short transcripts where the single FL can still identify the shabad via rotation.

### Strategy 7 — `type3_words`: IDF-weighted word-vote retrieval
- Builds a word→shabad inverted index at first use (Unicode Gurmukhi, from ASCII via `_to_unicode`).
- IDF weight per word: `log(N_shabads / df)`. Stop-words (df/N > 0.25) get near-zero weight.
- A shabad passes the gate if: `(hits >= 2 AND score >= 1.5)` OR `(hits >= 1 AND score >= 3.5)`.
  - The single-hit path (`word_vote_single_hit_min_score = 3.5`) was added to catch kirtan repetition of one rare/distinctive word (e.g. "ਉਧਰੀਐ") that FL search misses.
- From passing shabads, picks the line whose FL best matches the query (subsequence coverage).
- Fires whenever `transcript_text` is non-empty and `word_vote_enabled = True`.

### Strategy 8 — `phonetic` / `phonetic_sub`: phonetic substitution
- Generates all b↔v variants of the ASCII FL (Whisper confuses ਬ/ਵ frequently in sung Punjabi).
- For each variant: prefix search + substring search. For long variants: also 4-char sub-slices.
- Fires when `len(first_letters) >= 3`, regardless of candidate count.
- Best for: ਬਿਸਾਰਹੁ heard as ਵਿਸਾਰਿ, etc.

### Strategy 9 — `ngram4`: char 4-gram Unicode retrieval *(added 2026-04-26)*
- Builds a char-4-gram inverted index over Unicode Gurmukhi text of all 60k lines at first use (~4–6 s cold).
- Query: `normalize_for_fullword_search(transcript_text)` → compute 4-gram set.
- Score per line: overlap coefficient `|q4 ∩ l4| / min(|q4|, |l4|)`. Floor: `ngram4_min_overlap = 0.30`.
- Groups by shabad, returns best line per shabad, top `ngram4_max_results = 8`.
- Fires when `transcript_text` is non-empty and `ngram4_search_enabled = True`.
- **Why:** Catches end-fragment kirtan patterns — ragi sings only the 2nd half of a DB line. The FL's first letter mismatches the DB line's start, making all FL strategies blind. 4-grams over the full Unicode text find it regardless of where in the line the match occurs.

---

## Candidate Scoring — `ConfidenceScorer.score_detailed()`

Applied to every returned candidate to produce a `score` in [0, 1].

### Signals computed

| Signal | Formula | Purpose |
|--------|---------|---------|
| `letter_ratio` | `SequenceMatcher(query_fl, candidate_fl).ratio()` | Overall FL similarity |
| `coverage` | `LCS(query, candidate_fl) / len(candidate_fl)` — subsequence coverage | Candidate letters as subsequence of noisy query |
| `dense_coverage` | `longest_common_substring(query_fl, shabad_concat_fl) / len(query_fl)` | Multi-line window spanning consecutive DB lines |
| `consec_ratio` | `longest_common_substring(query, candidate) / len(query)` | Consecutive letter run bonus |
| `letter_term` | `max(letter_ratio, coverage, dense_coverage)` | Best FL signal |
| `dense_dominant` | `dense_coverage > max(letter_ratio, coverage) + ε AND dense_coverage >= 0.65` | Flag: pure substring hit, needs corroboration |

### Final score formula

```
score = weight_letter_match * letter_term
      + weight_consecutive   * consec_ratio
      + weight_context       * context_score   # 1.0 if same shabad as current, else 0.5
      + weight_source        * source_score    # 1.0 if SGGS (source_id=G), else 0.5
```

Weights live in `MatcherConfig` (`weight_letter_match`, `weight_consecutive`, `weight_context`, `weight_source`).

### Word-vote bonus (applied in `_score_candidates` in orchestrator)

| word_vote_hits | bonus |
|----------------|-------|
| 2 | +0.05 |
| 3 | +0.10 |
| ≥ 4 | +0.15 |

Candidates that came ONLY from word-vote (`type3_words`) must also clear `word_vote_only_floor = 0.45` on the FL score before they can trigger an auto-lock.

### Classification thresholds

| classify() return | score range | action |
|-------------------|-------------|--------|
| `"auto"` | ≥ `auto_threshold` | Lock immediately |
| `"suggest"` | ≥ `suggest_threshold` | Show on dashboard, require confirmation |
| `"ignore"` | < `suggest_threshold` | Discard |

---

## Locked-State Line Scoring

When LOCKED, `_handle_locked` scores every verse in the shabad to find the best line match.

### Line 0 penalty *(added 2026-04-26)*

Line 0 is always the raag/mahala heading (e.g. "ਮਾਝ ਮਹਲਾ ੫ ॥"). It is never sung.  
When `penalize_heading_line = True`, line 0's score is clamped to **0.0** in both the fresh-window loop and the combined/pair/stitch fallback loop, preventing it from ever winning the line pointer race.

### Per-line scoring — fresh window (first pass)

Chooses the scoring path based on word count and config:

| Condition | Method used |
|-----------|-------------|
| `word_match_line_scoring = True` | `score_line_word_overlap(transcript, verse)` — set overlap |
| `word_count == 2` | `score_line_word_match(transcript, verse)` — sequential word subsequence |
| `ngram_line_scoring = True` AND `len(FL) <= 3` | `score_line_ngram(transcript, verse)` — char 4-gram overlap |
| otherwise | `score_line(FL, verse.first_letters)` + optionally `max(..., score_line_ngram(...))` |

### `score_line(query_fl, line_fl)`

```
letter_ratio   = SequenceMatcher(query, line).ratio()
consec         = longest_common_substring(query, line)
consec_ratio   = consec / max(len(query), 1)
coverage       = LCS(query, line) / len(line)       # subsequence
baseline       = 0.5 * letter_ratio + 0.5 * consec_ratio
coverage_path  = dense_coverage_weight * coverage + (1 - dense_coverage_weight) * consec_ratio
score          = max(baseline, coverage_path)
```

### `score_line_ngram(transcript, verse_unicode)`

Char-4-gram overlap coefficient between normalized transcript and verse Unicode text.  
`|q4 ∩ v4| / min(|q4|, |v4|)`. Verbatim → 1.0, partial → graceful decay.

### `score_line_word_overlap(transcript, verse_unicode)`

Matra-stripped word-set overlap: `|q_words ∩ v_words| / |q_words|`.  
Both sides: `normalize_for_fullword_search` → `_strip_matras` → `set.split()`.  
"ਨਾਮੁ" and "ਨਾਮਿ" are treated identically.

### `score_line_word_match(transcript, verse_unicode)`

Sequential word subsequence match (order-preserving).  
Query words must appear in the same order in the verse word list.  
Returns `matched / total`. Used only for 2-word queries within the locked shabad — never for challenger/shabad switching.

### Progression bias — `_apply_progression_bias(i, current_line, raw_score, time_pressure)`

Applies positional bonuses before the winner is selected:

| delta (`i - current_line`) | bonus |
|----------------------------|-------|
| 0 (current line) | base bias (keeps current line sticky) |
| +1 (next line) | grows with `time_pressure` from 0 → `predictive_time_bias_max` |
| other | no bonus |

High-confidence bypass: if `raw_score >= progression_high_confidence_bypass`, the raw score is returned unchanged for the current line.

### Fallback pass (combined + pair + triple scoring)

Only runs when best fresh-window score < `suggest_threshold`. Tries:
- Stitched windows: `prev_fl + current_fl`, `current_fl + prev_fl`
- Pair alignment: score query against `verse[i].fl + verse[i+1].fl`
- Triple alignment: score query against `verse[i].fl + verse[i+1].fl + verse[i+2].fl`

Line 0 is also clamped to 0.0 in this pass.

---

## Minimum-Word Gate

```python
_word_count = len(transcript_text.split())
_enough_words = _word_count >= min_words_for_line_advance  # default 2
should_update_line = not is_detour and _enough_words and (...)
```

- 1-word transcripts: line pointer never moves.
- 2-word transcripts: use sequential word match (`score_line_word_match`), skip challenger scan.
- 3+ words: full scoring pipeline.

---

## Key Config Flags (MatcherConfig)

| Flag | Default | Purpose |
|------|---------|---------|
| `word_vote_enabled` | `True` | Enable Strategy 7 (IDF word vote) |
| `word_vote_min_distinct_hits` | 2 | Minimum distinct words that must vote |
| `word_vote_min_score` | 1.5 | Minimum summed IDF weight |
| `word_vote_single_hit_min_score` | 3.5 | Allow single hit if its IDF alone clears this |
| `ngram4_search_enabled` | `True` | Enable Strategy 9 (char 4-gram Unicode) |
| `ngram4_min_overlap` | 0.30 | Minimum overlap coefficient for ngram4 |
| `ngram4_max_results` | 8 | Top-N from ngram4 search |
| `ngram_line_scoring` | `True` | Use ngram alongside FL in locked-state line scoring |
| `word_match_line_scoring` | `False` | Replace FL/ngram with word-set overlap for line scoring |
| `sw_line_scoring_enabled` | `True` | Add SW word-alignment score in locked-state line scoring (Change 6) |
| `min_words_for_line_advance` | 2 | Minimum transcript words before line pointer moves |
| `penalize_heading_line` | `True` | Clamp line 0 score to 0 in locked state |
| `multi_line_search` | `True` | Enable 2-way + 3-way multi-line split search |
| `dense_dominant_margin` | 0.65 | Threshold above which a dense-only win needs corroboration |
| `word_vote_only_floor` | 0.45 | word-vote-only candidates must clear this FL score for auto-lock |
| `high_confidence_lock_threshold` | 0.90 | Bypass gap check and lock instantly above this score (Change 4) |
| `gap_threshold` | 0.10 | Required lead of top-1 over top-2 before auto-lock fires (Change 4) |
| `suggest_confirmation_seconds` | 4.0 | Seconds top suggest-level candidate must hold to get promoted to lock (Change 1) |
| `strong_override_threshold` | 0.90 | Challenger score that triggers immediate shabad switch (Change 5) |
| `override_min_gap` | 0.05 | Minimum lead over current shabad for immediate switch (Change 5) |
| `challenger_confirmation_seconds` | 4.0 | Seconds challenger must hold lead before timed switch commits (Change 5) |
| `stale_memory_threshold_seconds` | 10.0 | Gap between windows that invalidates suggest/challenger timestamps |
| `phonetic_max_variants` | 32 | Cap on phonetic substitution variants per query (Change 2) |
| `alaap_detection_enabled` | `True` | Freeze line pointer on melismatic/vowel-only windows (Change 7) |
| `alaap_consecutive_windows` | 2 | Consecutive alaap windows before line freeze activates (Change 7) |
| `transition_mode_enabled` | `True` | Enter relaxed-threshold mode during probable shabad transitions (Change 8) |
| `transition_min_signals` | 2 | Number of transition signals needed to enter transition mode (Change 8) |
| `transition_max_duration_seconds` | 30.0 | Auto-exit transition mode after this many seconds (Change 8) |
| `transition_challenger_confirmation_s` | 1.5 | Relaxed challenger confirmation time in transition mode (Change 8) |
| `transition_override_threshold` | 0.80 | Relaxed override threshold in transition mode (Change 8) |

---

## Retrieval Improvements (2026-04-28)

### Change 1 — Tiered Auto-Lock

Suggest-level candidates (below `auto_threshold`) are promoted to lockable after `suggest_confirmation_seconds` (4 s) as the sustained top candidate. Resets when a different candidate takes the lead or when a lock fires.

### Change 2 — Extended Phonetic Map

`_phonetic_variants` now covers b↔v, s↔S, n↔N, t↔T, d↔D, plus h-drop, with two levels of combination, capped at `phonetic_max_variants = 32`.

### Change 3 — dense_dominant + ngram4-only Corroboration

When `dense_dominant=True` AND the only retrieval source is `ngram4`, the action is downgraded from `auto` to `suggest`. Change 1's timer handles eventual lock.

### Change 4 — Confidence Gap Check

If `top_score >= high_confidence_lock_threshold (0.90)` → lock bypasses gap check. Otherwise `(top_score − second_best) >= gap_threshold (0.10)` is required; insufficient gap kills the per-window auto path (but not the evidence/stability path).

### Change 5 — Time-Based Challenger Switch

Immediate switch when `challenger_score >= strong_override_threshold (0.90) AND gap >= override_min_gap (0.05)`. Otherwise a per-challenger timestamp tracks how long it has been outscoring; switch commits after `challenger_confirmation_seconds (4 s)`. Thresholds relax in transition mode (Change 8). All challenger timestamps reset when the current locked shabad wins a window.

### Change 6 — Word-Level Smith-Waterman Line Scoring

`score_line_sw(transcript, verse)` runs SW local alignment on matra-stripped word lists (match=2, mismatch=−1, gap=−1), normalized to [0,1]. In `_handle_locked` the result is `max`-blended with the existing FL/ngram score when `sw_line_scoring_enabled=True`. Never used during retrieval.

### Change 7 — Alaap Detection

`_is_alaap_output(text)` detects melismatic/non-lexical windows: empty text, >60% bare vowel words, ≤2 distinct tokens repeated ≥3 times, or all tokens ≤2 chars. When `alaap_consecutive_windows` consecutive windows fire while LOCKED, the line pointer is frozen and challenger logic is skipped for that window.

### Change 9 — Confident-Jump Bypass in Progression Bias

`_apply_progression_bias()` returns `raw_score` unchanged when `raw_score >= progression_confident_jump_threshold (0.85)` — for ALL lines including the current one. Also gates the 1-line `target_idx > old_line + 1` jump cap in `_handle_locked` behind the same threshold so a confident jump can land on a non-adjacent verse in a single window. Fixes the case where a clearly correct match elsewhere in the shabad lost to the current line's +0.05/+0.22 inertia bonus.

### Change 8 — Transition Mode

Entered when ≥2 of these signals are present while LOCKED:

1. Locked shabad line-score weak for ≥ `transition_weak_seconds` (6 s)
2. Current line is in the last 2 verses
3. Accumulated alaap/silence ≥ `transition_silence_seconds` (8 s)
4. Alaap window count ≥ `alaap_consecutive_windows`

In transition mode: `challenger_confirmation_s` → 1.5 s, override threshold → 0.80, override gap → 0.05. Exits on new lock or after `transition_max_duration_seconds` (30 s).

---

## Confirmed Bugs Fixed (2026-04-26)

| Bug | Root cause | Fix |
|-----|-----------|-----|
| word-vote misses single rare word | `min_distinct_hits=2` filtered high-IDF single word | Added `word_vote_single_hit_min_score=3.5` alternative gate |
| End-fragment kirtan pattern invisible | No char-level retrieval — FL always mismatched | Strategy 9: char 4-gram Unicode inverted index |
| Kirtan prefix words (ਵਾਹਿਗੁਰੂ) inflate FL | Extra first-letter shrinks score margin | `_KIRTAN_FILLER` strip before FL extraction in `search()` |
| Line pointer stuck on raag heading | Line 0 score never clamped — tiebreaker arithmetic wins | `penalize_heading_line`: clamp line 0 to 0.0 in both scoring loops |
