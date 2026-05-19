"""
Tests for concurrent execution correctness.

Verifies that:
1. batch_score_providers produces identical results to sequential scoring.
2. batch_calculate_distances produces identical results to sequential calc.
3. parallel_map preserves input order (deterministic ranking).
4. Thread pool handles errors gracefully.
5. MAX_WORKERS env var is respected.

Run with:  python -m pytest tests/test_concurrency.py -v
"""

import math
import os
import sys
import pytest
from unittest.mock import patch
from pathlib import Path

# ── Ensure project root is on sys.path ─────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.concurrency import parallel_map, get_executor, shutdown_executor, _get_max_workers
from src.agents.tools import (
    calculate_distance,
    score_provider,
    batch_score_providers,
    batch_calculate_distances,
)


# ═══════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════

SCORING_CONFIG = {
    "weights": {
        "distance": 0.30,
        "rating": 0.25,
        "availability": 0.20,
        "experience": 0.15,
        "response_time": 0.10,
    },
    "availability_scores": {
        "Available": 1.0,
        "Available Soon": 0.7,
        "Busy": 0.3,
        "Offline": 0.0,
    },
    "distance": {
        "decay_constant_km": 5.0,
        "max_distance_km": 50.0,
        "default_distance_km": 10.0,
    },
    "rating": {"min_value": 1.0, "max_value": 5.0},
    "experience": {"max_years_cap": 20},
    "response_time": {"ideal_minutes": 5, "max_minutes": 120},
    "verification_bonus": 0.05,
    "price_modifiers": {
        "Low": {"budget_conscious": 0.15, "premium_seeker": -0.05},
        "Medium": {"budget_conscious": 0.05, "premium_seeker": 0.05},
        "High": {"budget_conscious": -0.10, "premium_seeker": 0.15},
    },
    "thresholds": {"min_composite_score": 0.25},
}


def _make_provider(pid: int, lat: float = 24.8607, lon: float = 67.0011, **kwargs) -> dict:
    """Create a mock provider dict for testing."""
    defaults = {
        "provider_name": f"Provider_{pid}",
        "category": "Plumber",
        "city": "Karachi",
        "area": "Clifton",
        "latitude": lat,
        "longitude": lon,
        "rating": 4.0 + (pid % 10) * 0.1,
        "availability": "Available",
        "experience_years": 5 + pid % 15,
        "completed_jobs": 100 + pid * 10,
        "response_time_minutes": 10 + pid % 30,
        "price_range": ["Low", "Medium", "High"][pid % 3],
        "verified_provider": pid % 2 == 0,
        "languages_supported": "English, Urdu",
        "phone_number": f"+92300{pid:07d}",
        "email": f"provider{pid}@test.com",
        "provider_id": pid,
    }
    defaults.update(kwargs)
    return {"id": str(pid), "metadata": defaults}


def _make_providers(n: int) -> list[dict]:
    """Generate n mock providers with varying coordinates."""
    providers = []
    for i in range(n):
        lat = 24.8 + (i * 0.01)
        lon = 67.0 + (i * 0.01)
        providers.append(_make_provider(i, lat=lat, lon=lon))
    return providers


# ═══════════════════════════════════════════════════════════════
# Test: parallel_map preserves order
# ═══════════════════════════════════════════════════════════════

class TestParallelMap:
    """Verify that parallel_map is order-preserving and correct."""

    def test_preserves_order(self):
        """Results must match input order for deterministic ranking."""
        items = list(range(100))
        results = parallel_map(lambda x: x * 2, items, preserve_order=True)
        assert results == [x * 2 for x in items]

    def test_empty_input(self):
        """Empty input should return empty list without errors."""
        results = parallel_map(lambda x: x, [], preserve_order=True)
        assert results == []

    def test_single_item(self):
        """Single item should work without pool overhead issues."""
        results = parallel_map(lambda x: x + 1, [42], preserve_order=True)
        assert results == [43]

    def test_error_propagation(self):
        """Exceptions in worker threads must propagate to the caller."""
        def _fail(x):
            if x == 5:
                raise ValueError("intentional failure")
            return x

        with pytest.raises(ValueError, match="intentional failure"):
            parallel_map(_fail, range(10), preserve_order=True)

    def test_unordered_still_correct(self):
        """preserve_order=False should still produce all correct values."""
        items = list(range(50))
        results = parallel_map(lambda x: x ** 2, items, preserve_order=False)
        # Results are in index-order even with as_completed (our impl fills by index)
        assert results == [x ** 2 for x in items]


# ═══════════════════════════════════════════════════════════════
# Test: batch_score_providers matches sequential
# ═══════════════════════════════════════════════════════════════

class TestBatchScoring:
    """Verify concurrent scoring produces identical results to serial."""

    def test_batch_matches_sequential(self):
        """Batch scoring must be bit-identical to sequential scoring."""
        providers = _make_providers(50)
        user_lat, user_lon = 24.8607, 67.0011

        # Sequential scoring (ground truth)
        sequential_scores = []
        for p in providers:
            result = score_provider(
                provider=p,
                scoring_config=SCORING_CONFIG,
                user_lat=user_lat,
                user_lon=user_lon,
                price_preference="budget",
            )
            sequential_scores.append(result)

        # Concurrent batch scoring
        batch_scores = batch_score_providers(
            candidates=providers,
            scoring_config=SCORING_CONFIG,
            user_lat=user_lat,
            user_lon=user_lon,
            price_preference="budget",
        )

        # Must be identical
        assert len(batch_scores) == len(sequential_scores)
        for i, (seq, batch) in enumerate(zip(sequential_scores, batch_scores)):
            assert seq["composite_score"] == batch["composite_score"], (
                f"Provider {i}: sequential={seq['composite_score']} != batch={batch['composite_score']}"
            )
            assert seq["breakdown"] == batch["breakdown"], f"Provider {i}: breakdown mismatch"

    def test_batch_no_coordinates(self):
        """Batch scoring without user coordinates should use neutral distance."""
        providers = _make_providers(10)

        batch_scores = batch_score_providers(
            candidates=providers,
            scoring_config=SCORING_CONFIG,
            user_lat=None,
            user_lon=None,
        )

        for score in batch_scores:
            assert score["breakdown"]["distance"]["score"] == 0.5  # neutral
            assert score["breakdown"]["distance"]["distance_km"] is None

    def test_deterministic_ranking_order(self):
        """Sorting batch results must produce the same order every run."""
        providers = _make_providers(30)

        # Run multiple times to check for non-determinism
        orders = []
        for _ in range(5):
            scores = batch_score_providers(
                candidates=providers,
                scoring_config=SCORING_CONFIG,
                user_lat=24.8607,
                user_lon=67.0011,
            )
            ranked = sorted(
                zip(range(len(providers)), scores),
                key=lambda x: x[1]["composite_score"],
                reverse=True,
            )
            order = [idx for idx, _ in ranked]
            orders.append(order)

        # All runs must produce the same ranking
        for run_order in orders[1:]:
            assert run_order == orders[0], "Ranking order is not deterministic!"


# ═══════════════════════════════════════════════════════════════
# Test: batch_calculate_distances matches sequential
# ═══════════════════════════════════════════════════════════════

class TestBatchDistances:
    """Verify concurrent distance calculation matches serial."""

    def test_batch_matches_sequential(self):
        """Batch distances must be bit-identical to sequential."""
        providers = _make_providers(50)
        user_lat, user_lon = 24.8607, 67.0011

        # Sequential
        sequential_dists = []
        for p in providers:
            meta = p.get("metadata", p)
            d = calculate_distance(
                user_lat, user_lon,
                meta.get("latitude", 0), meta.get("longitude", 0),
            )
            sequential_dists.append(d)

        # Concurrent
        batch_dists = batch_calculate_distances(providers, user_lat, user_lon)

        assert len(batch_dists) == len(sequential_dists)
        for i, (seq, batch) in enumerate(zip(sequential_dists, batch_dists)):
            assert seq == batch, f"Provider {i}: sequential={seq} != batch={batch}"

    def test_empty_providers(self):
        """Empty list should return empty distances."""
        result = batch_calculate_distances([], 24.0, 67.0)
        assert result == []


# ═══════════════════════════════════════════════════════════════
# Test: MAX_WORKERS env var
# ═══════════════════════════════════════════════════════════════

class TestMaxWorkers:
    """Verify MAX_WORKERS environment variable is respected."""

    def test_default_value(self):
        """Default should be 8 when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove MAX_WORKERS if present
            os.environ.pop("MAX_WORKERS", None)
            assert _get_max_workers() == 8

    def test_custom_value(self):
        """Custom value should be respected."""
        with patch.dict(os.environ, {"MAX_WORKERS": "4"}):
            assert _get_max_workers() == 4

    def test_invalid_value_defaults(self):
        """Invalid value should fallback to 8."""
        with patch.dict(os.environ, {"MAX_WORKERS": "not_a_number"}):
            assert _get_max_workers() == 8

    def test_zero_defaults(self):
        """Zero workers should fallback to 8."""
        with patch.dict(os.environ, {"MAX_WORKERS": "0"}):
            assert _get_max_workers() == 8

    def test_negative_defaults(self):
        """Negative workers should fallback to 8."""
        with patch.dict(os.environ, {"MAX_WORKERS": "-1"}):
            assert _get_max_workers() == 8


# ═══════════════════════════════════════════════════════════════
# Test: Thread safety of score_provider (pure function)
# ═══════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Verify that concurrent scoring doesn't corrupt shared state."""

    def test_no_cross_contamination(self):
        """
        Each provider's score must be independent — no data leaking
        between threads. We verify by checking that each provider's
        rating score matches its individual rating.
        """
        providers = _make_providers(20)

        scores = batch_score_providers(
            candidates=providers,
            scoring_config=SCORING_CONFIG,
            user_lat=24.8607,
            user_lon=67.0011,
        )

        for i, (provider, score) in enumerate(zip(providers, scores)):
            meta = provider["metadata"]
            expected_rating = meta["rating"]
            actual_rating = score["breakdown"]["rating"]["value"]
            assert actual_rating == expected_rating, (
                f"Provider {i}: expected rating={expected_rating}, got={actual_rating}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
