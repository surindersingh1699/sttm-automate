"""
Phase 1 Validation: Test BaniDB Python API.

Run: python scripts/test_banidb.py

Tests:
1. Basic connectivity to BaniDB API
2. First-letter search with known shabads
3. Different search types (0, 1, 4, 7)
4. Extract FirstLetterEng values to calibrate transliteration mapping
"""

import banidb


def test_search_types():
    """Test different BaniDB search types with known queries."""
    test_cases = [
        # (query, description)
        ("hjkn", "ਹਰਿ ਜੀ ਕ੍ਰਿਪਾ ਨਿਧਾਨ"),
        ("sDgj", "ਸਤਿਗੁਰ ਦਇਆਲ ਗੋਪਾਲ ਜੀ"),
        ("mhml", "ਮੇਰੇ ਹਰਿ ਮੋਹਨ ਲਾਲ"),
    ]

    search_types = {
        0: "First letter start (Gurmukhi)",
        1: "First letter anywhere",
        4: "Romanized first letters",
        7: "English first letters",
    }

    for query, description in test_cases:
        print(f"\n{'='*60}")
        print(f"Query: '{query}' (expected: {description})")
        print(f"{'='*60}")

        for st, st_name in search_types.items():
            try:
                results = banidb.search(query, searchtype=st)
                if results:
                    total = results.get("resultsInfo", {}).get("totalResults", 0)
                    print(f"\n  searchtype={st} ({st_name}): {total} results")
                    if total > 0:
                        verses = results.get("verses", [])
                        for i, verse in enumerate(verses[:3]):
                            v = verse.get("verse", {})
                            gurmukhi = v.get("gurmukhi", "N/A")
                            unicode_text = v.get("unicode", "N/A")
                            first_letter_eng = v.get("firstLetterEng", "N/A")
                            first_letter_str = v.get("firstLetterStr", "N/A")
                            shabad_id = verse.get("shabadId", "N/A")
                            print(f"    [{i+1}] Gurmukhi: {gurmukhi}")
                            print(f"        Unicode:  {unicode_text}")
                            print(f"        FirstLetterEng: {first_letter_eng}")
                            print(f"        FirstLetterStr: {first_letter_str}")
                            print(f"        ShabadID: {shabad_id}")
                else:
                    print(f"\n  searchtype={st} ({st_name}): No results (None returned)")
            except Exception as e:
                print(f"\n  searchtype={st} ({st_name}): ERROR - {e}")


def test_known_shabads():
    """Test with well-known shabads to verify API works."""
    known = [
        ("dhghmds", "ਧਨਾਸਰੀ ਮਹਲਾ 5 ਘਰ..."),
        ("nkjkwrr", "ਨਾਮ ਕੇ ਜੀਵ ਕੇ ਵੇਖ ਰੰਗ..."),
    ]

    print("\n\n" + "="*60)
    print("KNOWN SHABAD TESTS")
    print("="*60)

    for query, expected in known:
        print(f"\nQuery: '{query}' (expecting: {expected})")
        try:
            results = banidb.search(query, searchtype=7)
            if results:
                total = results.get("resultsInfo", {}).get("totalResults", 0)
                print(f"  Found {total} results")
                verses = results.get("verses", [])
                for i, verse in enumerate(verses[:2]):
                    v = verse.get("verse", {})
                    print(f"  [{i+1}] {v.get('unicode', 'N/A')}")
            else:
                print("  No results")
        except Exception as e:
            print(f"  ERROR: {e}")


def extract_mapping_samples():
    """Search for shabads starting with each Gurmukhi letter to build the mapping table."""
    print("\n\n" + "="*60)
    print("FIRST LETTER MAPPING EXTRACTION")
    print("="*60)
    print("Searching with single romanized letters to discover BaniDB's mapping...\n")

    # Try each letter used in STTM's first-letter search
    test_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for letter in test_letters:
        try:
            results = banidb.search(letter, searchtype=7)
            if results:
                total = results.get("resultsInfo", {}).get("totalResults", 0)
                if total > 0:
                    verse = results["verses"][0].get("verse", {})
                    unicode_text = verse.get("unicode", "")
                    first_char = unicode_text[0] if unicode_text else "?"
                    first_letter_eng = verse.get("firstLetterEng", "")
                    print(f"  '{letter}' → Gurmukhi: {first_char} | "
                          f"Unicode: {unicode_text[:30]}... | "
                          f"FirstLetterEng: {first_letter_eng[:20]}")
        except Exception:
            pass


def main():
    print("STTM Automate - BaniDB Validation")
    print("="*60)

    # Test basic API
    print("\n1. Testing random shabad fetch...")
    try:
        random_shabad = banidb.random()
        if random_shabad:
            print("   BaniDB API is working!")
            print(f"   Random: {random_shabad}")
        else:
            print("   WARNING: random() returned None")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test search types
    print("\n2. Testing search types...")
    test_search_types()

    # Test known shabads
    print("\n3. Testing known shabads...")
    test_known_shabads()

    # Extract mapping
    print("\n4. Extracting first-letter mapping...")
    extract_mapping_samples()

    print("\n\nDone! Review the output above to calibrate transliterate.py")


if __name__ == "__main__":
    main()
