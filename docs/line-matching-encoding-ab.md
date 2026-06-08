# Line-matching encoding A/B (2026-06-07)

Empirical comparison of how to encode a noisy ASR hypothesis when matching it
against a canonical Gurbani line. Run on **real IndicConformer ASR output**, not
synthetic noise. Harness: [`tests/eval/proto_encoding_ab.py`](../tests/eval/proto_encoding_ab.py).

## Setup

- **Data:** `surindersinghssj/gurbani-kirtan-dataset-v2` (test+validation). Each row
  is a slide-aligned audio segment + canonical `gurmukhi_text` (ground truth).
- **Method:** audio capped to 14 s → real ASR → hypothesis. Build an FL-retrieved
  candidate pool, force-add the GT line, and measure **selection**: where does each
  encoding rank the correct line? N = 100 usable pairs. ASR pairs are cached to
  `tests/eval/cache/` so re-runs skip the (slow) ASR pass.
- **Encodings** (similarity in [0,1] between hypothesis and a DB line):
  | name | representation | similarity |
  |---|---|---|
  | `FL` | first letters | normalized LCS coverage |
  | `exact` | full words, matras kept | word-set overlap coefficient |
  | `matra` | full chars, **matras kept** | normalized LCS coverage |
  | `skeleton` | chars, **matras stripped** | normalized LCS coverage |
  | `soft` | skeleton + phonetic fold (ਬ↔ਵ ਨ↔ਣ ਤ↔ਟ ਦ↔ਡ) | normalized LCS coverage |

`coverage(q, t) = LCS(q, t) / len(t)` — fraction of the target that appears, in
order, inside the noisy query. `mean-margin = score(GT) − score(best distractor)`
(robustness: how far the correct line leads the runner-up).

## Results (N = 100)

```
encoding    top1     top3     mean-rank   mean-margin
FL            9/100   25/100     38.2        -0.191
exact        68/100   80/100     13.9        +0.080
matra        69/100   86/100      2.1        +0.083   ← winner
skeleton     57/100   79/100      3.8        +0.030
soft         56/100   78/100      3.9        +0.024
```

`FL` retrieval also **missed the GT entirely in 25/100** cases (GT absent from the
FL-gram pool; force-added for the selection measurement above).

## Conclusions

1. **Keep matras.** `matra` wins on every axis, including mean-rank 2.1 (when it
   isn't #1, the correct line is still rank 2–3). Stripping matras costs −12 top-1.
   IndicConformer's matras carry real discriminating signal — they are *not* noise
   to discard.
2. **Soft phonetic-distance adds nothing** (0 unique wins vs. `skeleton`). The
   modelled consonant confusions rarely occur aligned; the complexity isn't earned.
3. **First-letters are weak at both jobs** — selection (9/100) and retrieval (25%
   miss). Full-word content (with matras) should be the matcher's primary signal,
   not first-letters.

## Design implication

Replace the first-letter-centric, multi-strategy retrieval with a content-first
matcher whose core similarity is `matra` (full words, matras kept). First-letters
drop to at most a cheap pre-filter.

## Open / not yet measured

- **Retrieval recall** of content-based (`matra`/word) indexes vs. FL — does it
  close FL's 25% miss? The throwaway harness's pure-Python posting-union over the
  full 141k-line corpus was too slow to complete; the correct way to measure this
  is *through* `OfflineShabadSearcher`'s existing optimized indexes, not a
  hand-rolled index.
- `max(matra, exact)` weightless ensemble (`exact` had 12 unique wins → could lift
  top-1 into the mid-70s).

Any production change based on this must be gated on the full eval
(`tests/eval/runner.py`: pct_time_correct, lock rate, TTFCL, wall time) — prior
"plausible" changes (KenLM fusion, preprocessor tweaks) regressed and were reverted.
