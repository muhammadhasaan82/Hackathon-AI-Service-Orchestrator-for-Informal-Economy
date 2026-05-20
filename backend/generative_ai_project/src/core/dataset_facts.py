"""
DatasetFacts — Dynamic dataset-derived facts service.

All city, area, category, and provider-count answers are derived from
the loaded DataFrame at startup.  Zero hardcoded business values.

Usage::

    from src.core.dataset_facts import DatasetFacts
    facts = DatasetFacts(providers_df)
    facts.get_available_cities()          # ["Faisalabad", "Islamabad", ...]
    facts.get_city_provider_count("Lahore")  # 8290
"""

import logging
import time
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

logger = logging.getLogger("core.dataset_facts")


class DatasetFacts:
    """
    Read-only fact store backed by the providers DataFrame.

    Every answer is dynamically derived — if the CSV changes tomorrow,
    answers change automatically.  All aggregations are pre-computed at
    construction time so query-time lookups are O(1) dict reads.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.snapshot_loaded_at = time.time()

        # ── Pre-computed lookup data ────────────────────────────
        self._cities: list[str] = sorted(df["city"].unique().tolist())
        self._categories: list[str] = sorted(df["category"].unique().tolist())
        self._areas_by_city: dict[str, list[str]] = {
            city: sorted(group["area"].unique().tolist())
            for city, group in df.groupby("city")
        }
        self._total_count: int = len(df)

        # ── Pre-aggregated counts ───────────────────────────────
        self._city_counts: dict[str, int] = df.groupby("city").size().to_dict()
        self._category_counts: dict[str, int] = df.groupby("category").size().to_dict()
        self._city_category_counts: dict[tuple, int] = (
            df.groupby(["city", "category"]).size().to_dict()
        )
        self._city_area_counts: dict[tuple, int] = (
            df.groupby(["city", "area"]).size().to_dict()
        )
        self._availability_counts: dict[str, int] = (
            df.groupby("availability").size().to_dict()
            if "availability" in df.columns else {}
        )
        self._availability_by_city: dict[tuple, int] = (
            df.groupby(["city", "availability"]).size().to_dict()
            if "availability" in df.columns else {}
        )
        self._availability_by_city_category: dict[tuple, int] = (
            df.groupby(["city", "category", "availability"]).size().to_dict()
            if "availability" in df.columns else {}
        )
        self._availability_by_city_category_area: dict[tuple, int] = (
            df.groupby(["city", "category", "area", "availability"]).size().to_dict()
            if "availability" in df.columns else {}
        )

        # ── Lowercase → canonical maps for fuzzy matching ───────
        self._city_lower: dict[str, str] = {c.lower(): c for c in self._cities}
        self._category_lower: dict[str, str] = {c.lower(): c for c in self._categories}
        self._area_lower_by_city: dict[str, dict[str, str]] = {
            city: {a.lower(): a for a in areas}
            for city, areas in self._areas_by_city.items()
        }

        logger.info(
            "DatasetFacts initialized | %d providers | %d cities | %d categories",
            self._total_count,
            len(self._cities),
            len(self._categories),
        )

    # ── Query methods ───────────────────────────────────────────

    def get_available_cities(self) -> list[str]:
        """Sorted list of all cities in the dataset."""
        return list(self._cities)

    def get_available_categories(self) -> list[str]:
        """Sorted list of all service categories in the dataset."""
        return list(self._categories)

    def get_areas_by_city(self, city: str) -> list[str]:
        """Sorted list of areas within *city*.  Returns [] if city unknown."""
        canonical = self._resolve_city(city)
        if canonical is None:
            return []
        return list(self._areas_by_city.get(canonical, []))

    def get_total_provider_count(self) -> int:
        return self._total_count

    def get_city_provider_count(self, city: str) -> int:
        canonical = self._resolve_city(city)
        if canonical is None:
            return 0
        return self._city_counts.get(canonical, 0)

    def get_category_provider_count(self, category: str) -> int:
        canonical = self._resolve_category(category)
        if canonical is None:
            return 0
        return self._category_counts.get(canonical, 0)

    def get_area_provider_count(self, city: str, area: str) -> int:
        canonical_city = self._resolve_city(city)
        canonical_area = self._resolve_area(city, area)
        if canonical_city is None or canonical_area is None:
            return 0
        return self._city_area_counts.get((canonical_city, canonical_area), 0)

    def get_category_city_coverage(self, category: str) -> dict[str, int]:
        """For a given category, return {city: provider_count}."""
        canonical = self._resolve_category(category)
        if canonical is None:
            return {}
        return {
            city: self._city_category_counts.get((city, canonical), 0)
            for city in self._cities
            if self._city_category_counts.get((city, canonical), 0) > 0
        }

    def get_service_counts_by_city(self) -> dict[str, int]:
        """Return {city: total_provider_count}."""
        return dict(self._city_counts)

    def get_service_counts_by_area(self, city: Optional[str] = None) -> dict[str, int]:
        """Return {area: count} for a city.  If city is None, all areas."""
        if city is not None:
            canonical = self._resolve_city(city)
            if canonical is None:
                return {}
            return {
                area: self._city_area_counts.get((canonical, area), 0)
                for area in self._areas_by_city.get(canonical, [])
            }
        # All areas across all cities
        result: dict[str, int] = {}
        for (c, a), count in self._city_area_counts.items():
            result[a] = result.get(a, 0) + count
        return result

    def get_availability_summary(
        self,
        city: Optional[str] = None,
        category: Optional[str] = None,
        area: Optional[str] = None,
    ) -> dict[str, int]:
        """Return provider availability counts for the requested scope."""
        canonical_city = self._resolve_city(city) if city else None
        canonical_category = self._resolve_category(category) if category else None
        canonical_area = self._resolve_area(canonical_city, area) if canonical_city and area else None

        if canonical_city and canonical_category and canonical_area:
            return {
                status: count
                for (c, cat, a, status), count in self._availability_by_city_category_area.items()
                if c == canonical_city and cat == canonical_category and a == canonical_area
            }
        if canonical_city and canonical_category:
            return {
                status: count
                for (c, cat, status), count in self._availability_by_city_category.items()
                if c == canonical_city and cat == canonical_category
            }
        if canonical_city:
            return {
                status: count
                for (c, status), count in self._availability_by_city.items()
                if c == canonical_city
            }
        return dict(self._availability_counts)

    def snapshot_metadata(self) -> dict:
        """Small observable summary of the loaded fact snapshot."""
        return {
            "loaded_at": self.snapshot_loaded_at,
            "providers": self._total_count,
            "cities": len(self._cities),
            "categories": len(self._categories),
        }

    # ── Existence / fuzzy-match helpers ─────────────────────────

    def has_city(self, name: str) -> tuple[bool, Optional[str]]:
        """Check if *name* matches a known city (case-insensitive).
        Returns (found, canonical_name)."""
        canonical = self._resolve_city(name)
        return (canonical is not None, canonical)

    def has_category(self, name: str) -> tuple[bool, Optional[str]]:
        """Check if *name* matches a known category (case-insensitive).
        Returns (found, canonical_name)."""
        canonical = self._resolve_category(name)
        return (canonical is not None, canonical)

    def has_area(self, city: str, area: str) -> tuple[bool, Optional[str]]:
        """Check if *area* exists within *city*.
        Returns (found, canonical_area_name)."""
        canonical = self._resolve_area(city, area)
        return (canonical is not None, canonical)

    def fuzzy_match_category(self, text: str, threshold: float = 0.72) -> Optional[str]:
        """Fuzzy-match *text* against known categories.
        Uses the same SequenceMatcher threshold as IntentAgent."""
        text_lower = text.strip().lower()
        # Exact check first
        if text_lower in self._category_lower:
            return self._category_lower[text_lower]
        # Fuzzy
        best_score = 0.0
        best_match: Optional[str] = None
        for canonical_lower, canonical in self._category_lower.items():
            score = SequenceMatcher(None, text_lower, canonical_lower).ratio()
            if score > best_score:
                best_score = score
                best_match = canonical
        if best_score >= threshold and best_match is not None:
            return best_match
        return None

    def match_category(
        self,
        text: str,
        threshold: float = 0.72,
        ambiguity_threshold: float = 0.58,
        close_match_margin: float = 0.12,
        max_candidates: int = 3,
    ) -> dict:
        """Return category match metadata without silently hiding ambiguity."""
        return self._match_catalog(
            text=text,
            values=self._categories,
            lower_map=self._category_lower,
            threshold=threshold,
            ambiguity_threshold=ambiguity_threshold,
            close_match_margin=close_match_margin,
            max_candidates=max_candidates,
        )

    def fuzzy_match_city(self, text: str, threshold: float = 0.78) -> Optional[str]:
        """Fuzzy-match *text* against known cities."""
        text_lower = text.strip().lower()
        if text_lower in self._city_lower:
            return self._city_lower[text_lower]
        best_score = 0.0
        best_match: Optional[str] = None
        for canonical_lower, canonical in self._city_lower.items():
            score = SequenceMatcher(None, text_lower, canonical_lower).ratio()
            if score > best_score:
                best_score = score
                best_match = canonical
        if best_score >= threshold and best_match is not None:
            return best_match
        return None

    def match_city(
        self,
        text: str,
        threshold: float = 0.78,
        ambiguity_threshold: float = 0.60,
        close_match_margin: float = 0.10,
        max_candidates: int = 3,
    ) -> dict:
        """Return city match metadata without silently hiding ambiguity."""
        return self._match_catalog(
            text=text,
            values=self._cities,
            lower_map=self._city_lower,
            threshold=threshold,
            ambiguity_threshold=ambiguity_threshold,
            close_match_margin=close_match_margin,
            max_candidates=max_candidates,
        )

    # ── Internal resolution helpers ─────────────────────────────

    def _resolve_city(self, name: str) -> Optional[str]:
        """Resolve a raw city name to its canonical form."""
        if not name:
            return None
        return self._city_lower.get(name.strip().lower())

    def _resolve_category(self, name: str) -> Optional[str]:
        """Resolve a raw category name to its canonical form."""
        if not name:
            return None
        return self._category_lower.get(name.strip().lower())

    def _resolve_area(self, city: str, area: str) -> Optional[str]:
        """Resolve a raw area name within a city to its canonical form."""
        canonical_city = self._resolve_city(city)
        if canonical_city is None or not area:
            return None
        city_areas = self._area_lower_by_city.get(canonical_city, {})
        return city_areas.get(area.strip().lower())

    def _match_catalog(
        self,
        text: str,
        values: list[str],
        lower_map: dict[str, str],
        threshold: float,
        ambiguity_threshold: float,
        close_match_margin: float,
        max_candidates: int,
    ) -> dict:
        raw_text = str(text or "").strip()
        text_lower = raw_text.lower()
        if not text_lower:
            return self._empty_match()

        if text_lower in lower_map:
            value = lower_map[text_lower]
            return {
                "matched": True,
                "value": value,
                "score": 1.0,
                "source": "exact",
                "candidates": [{"value": value, "score": 1.0}],
                "ambiguous": False,
            }

        candidates = []
        normalized_text = self._normalize_text(text_lower)
        for value in values:
            normalized_value = self._normalize_text(value)
            score = self._best_similarity(normalized_text, normalized_value)
            candidates.append({"value": value, "score": round(float(score), 4)})

        candidates.sort(key=lambda item: item["score"], reverse=True)
        top_candidates = candidates[: max(1, max_candidates)]
        best = top_candidates[0] if top_candidates else None
        if not best:
            return self._empty_match()

        close = [
            item for item in top_candidates
            if item["score"] >= ambiguity_threshold
            and best["score"] - item["score"] <= close_match_margin
        ]
        ambiguous = len(close) > 1
        matched = best["score"] >= threshold and not ambiguous

        return {
            "matched": bool(matched),
            "value": best["value"] if matched else None,
            "score": best["score"],
            "source": "fuzzy" if best["score"] >= ambiguity_threshold else "none",
            "candidates": close if ambiguous else top_candidates,
            "ambiguous": ambiguous,
        }

    def _best_similarity(self, normalized_text: str, normalized_value: str) -> float:
        if not normalized_text or not normalized_value:
            return 0.0
        if f" {normalized_value} " in f" {normalized_text} ":
            return 1.0
        tokens = normalized_text.split()
        value_tokens = normalized_value.split()
        best = SequenceMatcher(None, normalized_text, normalized_value).ratio()
        window_sizes = {len(value_tokens) - 1, len(value_tokens), len(value_tokens) + 1}
        for size in window_sizes:
            if size <= 0:
                continue
            for index in range(0, max(0, len(tokens) - size) + 1):
                window = " ".join(tokens[index:index + size])
                best = max(best, SequenceMatcher(None, window, normalized_value).ratio())
        return float(best)

    def _normalize_text(self, value: str) -> str:
        normalized_chars = []
        for char in str(value or "").casefold():
            normalized_chars.append(char if char.isalnum() else " ")
        return " ".join("".join(normalized_chars).split())

    def _empty_match(self) -> dict:
        return {
            "matched": False,
            "value": None,
            "score": 0.0,
            "source": "none",
            "candidates": [],
            "ambiguous": False,
        }
