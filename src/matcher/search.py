"""BaniDB search wrapper with multi-strategy matching.

Calls the BaniDB REST API directly (https://api.banidb.com/v2/) instead of
the banidb Python library, which fetches all result pages and is too slow.
"""

from dataclasses import dataclass

import httpx

_API_BASE = "https://api.banidb.com/v2"
_TIMEOUT = 8.0


def _extract_verse_first_letters(unicode_text: str) -> str:
    """Extract first Gurmukhi letter of each word from a verse's unicode field."""
    letters = []
    for word in unicode_text.split():
        if word and "\u0A00" <= word[0] <= "\u0A7F":
            letters.append(word[0])
    return "".join(letters)


@dataclass
class ShabadVerse:
    """A single verse/line within a shabad, with pre-extracted first letters."""
    verse_id: int
    unicode: str
    gurmukhi: str
    english: str
    first_letters: str  # pre-extracted Gurmukhi first letters for scoring


@dataclass
class ShabadCandidate:
    shabad_id: int
    gurmukhi: str
    unicode: str
    english: str
    source_id: str
    page_no: int


class ShabadSearcher:
    """Searches BaniDB using multiple strategies for best match coverage."""

    def __init__(self) -> None:
        self._client = httpx.Client(base_url=_API_BASE, timeout=_TIMEOUT)

    def search(self, first_letters: str, max_results: int = 10) -> list[ShabadCandidate]:
        """
        Search BaniDB with multiple strategies, merge and deduplicate results.

        Strategies tried in order:
        1. searchtype=0: First letter beginning (primary, romanized codes)
        2. searchtype=4: First letter anywhere (broader fallback)
        3. Substring search: Try shorter substrings if full query gets no results
        """
        if len(first_letters) < 3:
            return []

        candidates: list[ShabadCandidate] = []
        seen_ids: set[int] = set()

        # Strategy 1: First letter beginning (primary)
        results = self._search_api(first_letters, searchtype=0, limit=max_results)
        self._add_unique(results, candidates, seen_ids)

        # Strategy 2: First letter anywhere (broader fallback)
        if len(candidates) < 3:
            results = self._search_api(first_letters, searchtype=4, limit=max_results)
            self._add_unique(results, candidates, seen_ids)

        # Strategy 3: Try shorter substrings if we still have few results
        if len(candidates) < 2 and len(first_letters) > 4:
            for sub in [first_letters[:4], first_letters[-4:]]:
                results = self._search_api(sub, searchtype=0, limit=5)
                self._add_unique(results, candidates, seen_ids)

        return candidates

    def search_by_id(self, shabad_id: int) -> ShabadCandidate | None:
        """Fetch a specific shabad by its ID."""
        try:
            resp = self._client.get(f"/shabads/{shabad_id}")
            resp.raise_for_status()
            data = resp.json()
            verses = data.get("verses", [])
            if verses:
                v = verses[0]
                verse = v.get("verse", {})
                return ShabadCandidate(
                    shabad_id=shabad_id,
                    gurmukhi=verse.get("gurmukhi", ""),
                    unicode=verse.get("unicode", ""),
                    english=v.get("translation", {}).get("en", {}).get("bdb", ""),
                    source_id=v.get("source", {}).get("sourceId", "G"),
                    page_no=v.get("pageNo", 0),
                )
        except Exception as e:
            print(f"[Search] Error fetching shabad {shabad_id}: {e}")
        return None

    def fetch_all_verses(self, shabad_id: int) -> list[ShabadVerse]:
        """Fetch all verses of a shabad for line-level tracking."""
        try:
            resp = self._client.get(f"/shabads/{shabad_id}")
            resp.raise_for_status()
            data = resp.json()

            verses: list[ShabadVerse] = []
            for entry in data.get("verses", []):
                verse = entry.get("verse", {})
                unicode_text = verse.get("unicode", "")
                # Pre-extract first letters for efficient scoring
                first_letters = _extract_verse_first_letters(unicode_text)
                verses.append(ShabadVerse(
                    verse_id=entry.get("verseId", 0),
                    unicode=unicode_text,
                    gurmukhi=verse.get("gurmukhi", ""),
                    english=entry.get("translation", {}).get("en", {}).get("bdb", ""),
                    first_letters=first_letters,
                ))
            return verses

        except Exception as e:
            print(f"[Search] Error fetching verses for shabad {shabad_id}: {e}")
            return []

    def _search_api(self, query: str, searchtype: int, limit: int) -> list[ShabadCandidate]:
        """Call BaniDB REST API directly and parse results."""
        try:
            resp = self._client.get(
                f"/search/{query}",
                params={"searchtype": searchtype, "results": limit},
            )
            resp.raise_for_status()
            data = resp.json()

            total = data.get("resultsInfo", {}).get("totalResults", 0)
            if total == 0:
                return []

            candidates = []
            for entry in data.get("verses", [])[:limit]:
                verse = entry.get("verse", {})
                translation = entry.get("translation", {}).get("en", {}).get("bdb", "")
                candidates.append(ShabadCandidate(
                    shabad_id=entry.get("shabadId", 0),
                    gurmukhi=verse.get("gurmukhi", ""),
                    unicode=verse.get("unicode", ""),
                    english=translation,
                    source_id=entry.get("source", {}).get("sourceId", "G"),
                    page_no=entry.get("pageNo", 0),
                ))
            return candidates

        except Exception as e:
            print(f"[Search] API error (type={searchtype}): {e}")
            return []

    def _add_unique(
        self,
        new: list[ShabadCandidate],
        existing: list[ShabadCandidate],
        seen: set[int],
    ) -> None:
        """Add candidates that haven't been seen yet."""
        for c in new:
            if c.shabad_id not in seen:
                seen.add(c.shabad_id)
                existing.append(c)
