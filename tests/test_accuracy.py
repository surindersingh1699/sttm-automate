"""
Accuracy test for STTM Automate matching pipeline.

Tests the core pipeline: Gurmukhi text → first-letter extraction → BaniDB search → scoring
using 4 well-known Gurbani shabads. Measures whether the correct shabad ranks #1.
"""

import sys
import time
import json

sys.path.insert(0, "/home/claude-bot/projects/projects/sttm-automate")

from src.transcription.transliterate import extract_first_letters
from src.matcher.search import ShabadSearcher
from src.matcher.scorer import ConfidenceScorer

# 4 well-known Gurbani shabads for testing
# Each has: name, first line in Gurmukhi, expected shabad ID, and ang (page)
TEST_SHABADS = [
    {
        "name": "Japji Sahib - Mool Mantar / Sochai Soch",
        "gurmukhi_line": "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ",
        "expected_shabad_id": 1,
        "ang": 1,
    },
    {
        "name": "Rehraas - So Dar",
        "gurmukhi_line": "ਸੋ ਦਰੁ ਕੇਹਾ ਸੋ ਘਰੁ ਕੇਹਾ ਜਿਤੁ ਬਹਿ ਸਰਬ ਸਮਾਲੇ",
        "expected_shabad_id": 27,
        "ang": 6,
    },
    {
        "name": "Anand Sahib - Anand Bhaiaa",
        "gurmukhi_line": "ਅਨੰਦੁ ਭਇਆ ਮੇਰੀ ਮਾਏ ਸਤਿਗੁਰੂ ਮੈ ਪਾਇਆ",
        "expected_shabad_id": 333375,
        "ang": 917,
    },
    {
        "name": "Tav Prasad Savaiye - Srawag Sudh",
        "gurmukhi_line": "ਸ੍ਰਾਵਗ ਸੁੱਧ ਸਮੂਹ ਸਿਧਾਨ ਕੇ ਦੇਖਿ ਫਿਰਿਓ ਘਰ ਜੋਗ ਜਤੀ ਕੇ",
        "expected_shabad_id": 7426,
        "ang": 12,
        "source": "Dasam Granth",
    },
]


def run_accuracy_test():
    """Run end-to-end accuracy test on 4 sample shabads."""
    searcher = ShabadSearcher()
    scorer = ConfidenceScorer()

    results = []
    total_tests = len(TEST_SHABADS)
    correct = 0

    print("=" * 80)
    print("STTM AUTOMATE — MATCHING PIPELINE ACCURACY TEST")
    print("=" * 80)
    print(f"\nTesting {total_tests} Gurbani shabads through the pipeline:")
    print("  Gurmukhi Text → First-Letter Extraction → BaniDB Search → Confidence Scoring\n")

    for i, shabad in enumerate(TEST_SHABADS, 1):
        print(f"\n{'─' * 70}")
        print(f"TEST {i}/{total_tests}: {shabad['name']}")
        print(f"{'─' * 70}")
        print(f"  Input:    {shabad['gurmukhi_line']}")

        # Step 1: Extract first letters
        start_time = time.time()
        first_letters = extract_first_letters(shabad["gurmukhi_line"])
        extract_time = time.time() - start_time
        print(f"  Letters:  {first_letters} ({len(first_letters)} letters)")

        # Step 2: Search BaniDB
        start_time = time.time()
        candidates = searcher.search(
            first_letters=first_letters,
            max_results=10,
            transcript_text=shabad["gurmukhi_line"],
        )
        search_time = time.time() - start_time
        print(f"  Search:   {len(candidates)} candidates found ({search_time:.2f}s)")

        # Step 3: Score all candidates
        start_time = time.time()
        scored = []
        for c in candidates:
            score = scorer.score(first_letters, c)
            action = scorer.classify(score)
            word_overlap = scorer.word_overlap_count(
                shabad["gurmukhi_line"], c.unicode
            )
            scored.append({
                "shabad_id": c.shabad_id,
                "score": score,
                "action": action,
                "unicode": c.unicode[:60],
                "word_overlap": word_overlap,
                "source_id": c.source_id,
                "page": c.page_no,
            })
        score_time = time.time() - start_time

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Step 4: Check accuracy
        top_match = scored[0] if scored else None
        expected_id = shabad.get("expected_shabad_id")
        is_correct = False
        rank_of_expected = None

        if expected_id is not None and scored:
            for rank, s in enumerate(scored, 1):
                if s["shabad_id"] == expected_id:
                    rank_of_expected = rank
                    if rank == 1:
                        is_correct = True
                    break

        # For Dasam Granth shabads, check if we got any reasonable result
        if expected_id is None and scored:
            # Consider it a pass if we got candidates from the right source
            source = shabad.get("source", "")
            if source == "Dasam Granth":
                # Check if any candidate matched
                is_correct = top_match["score"] >= 0.60 if top_match else False
                rank_of_expected = "N/A (Dasam Granth)"

        if is_correct:
            correct += 1

        # Print top 5 results
        print(f"\n  Top candidates (scored in {score_time*1000:.1f}ms):")
        for j, s in enumerate(scored[:5], 1):
            marker = " ✓" if s["shabad_id"] == expected_id else ""
            print(
                f"    #{j}: ID={s['shabad_id']:>5} | "
                f"Score={s['score']:.3f} ({s['action']:>7}) | "
                f"Words={s['word_overlap']} | "
                f"Src={s['source_id']} p.{s['page']}{marker}"
            )
            print(f"         {s['unicode']}")

        # Result summary
        status = "PASS ✓" if is_correct else "FAIL ✗"
        print(f"\n  Result: {status}")
        if rank_of_expected and rank_of_expected != "N/A (Dasam Granth)":
            print(f"  Expected shabad ID {expected_id} found at rank #{rank_of_expected}")
        elif expected_id is None:
            print(f"  Note: {shabad.get('source', 'Unknown source')} — expected ID unknown")
        else:
            print(f"  Expected shabad ID {expected_id} NOT FOUND in results")

        results.append({
            "name": shabad["name"],
            "input": shabad["gurmukhi_line"],
            "first_letters": first_letters,
            "num_candidates": len(candidates),
            "top_score": top_match["score"] if top_match else 0,
            "top_action": top_match["action"] if top_match else "none",
            "top_shabad_id": top_match["shabad_id"] if top_match else None,
            "expected_id": expected_id,
            "rank_of_expected": rank_of_expected,
            "correct": is_correct,
            "extract_time_ms": extract_time * 1000,
            "search_time_ms": search_time * 1000,
            "score_time_ms": score_time * 1000,
        })

    # Final report
    accuracy = correct / total_tests * 100
    print(f"\n\n{'=' * 80}")
    print(f"ACCURACY REPORT")
    print(f"{'=' * 80}")
    print(f"\n  Total Tests:     {total_tests}")
    print(f"  Correct (Rank 1): {correct}")
    print(f"  Accuracy:         {accuracy:.1f}%")

    print(f"\n  Per-test breakdown:")
    print(f"  {'Test':<45} {'Score':>7} {'Action':>8} {'Rank':>6} {'Result':>8}")
    print(f"  {'─' * 45} {'─' * 7} {'─' * 8} {'─' * 6} {'─' * 8}")
    for r in results:
        rank_str = f"#{r['rank_of_expected']}" if r["rank_of_expected"] and r["rank_of_expected"] != "N/A (Dasam Granth)" else "N/A"
        status = "PASS" if r["correct"] else "FAIL"
        print(
            f"  {r['name']:<45} {r['top_score']:>7.3f} {r['top_action']:>8} {rank_str:>6} {status:>8}"
        )

    avg_search = sum(r["search_time_ms"] for r in results) / len(results)
    avg_score = sum(r["score_time_ms"] for r in results) / len(results)
    print(f"\n  Avg search time:  {avg_search:.0f}ms")
    print(f"  Avg score time:   {avg_score:.1f}ms")

    print(f"\n  Pipeline: Gurmukhi → First Letters → BaniDB Search → Confidence Score")
    print(f"  Note: This tests the matching pipeline only (no audio/Whisper)")
    print(f"{'=' * 80}\n")

    return results, accuracy


if __name__ == "__main__":
    results, accuracy = run_accuracy_test()
