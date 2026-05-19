"""
Concurrency Manager — Centralized thread pool for CPU-bound parallel work.

WHY THREADING:
    Provider scoring, distance calculation, and availability checking are
    CPU-bound, independent-per-provider operations over a 50K dataset.
    A shared ThreadPoolExecutor lets us parallelize these across N workers
    while keeping a single pool (no per-request pool overhead).

    We use threads (not processes) because:
    1. Each task is lightweight math (haversine, weighted sums) — the GIL
       releases during math.sin/cos/exp in CPython's C layer.
    2. Thread creation overhead is near-zero vs. process fork.
    3. Shared memory avoids serialization cost for provider dicts.

SAFETY:
    - The executor is module-level and lazily initialized (thread-safe via Lock).
    - All parallel tasks are pure functions (no shared mutable state).
    - Results are gathered via futures; the final sort is single-threaded
      to guarantee deterministic ordering.

CONFIGURATION:
    MAX_WORKERS env var controls the pool size (default: 8).
    Set lower on resource-constrained deployments.
"""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Callable, Iterable, TypeVar

logger = logging.getLogger("core.concurrency")

T = TypeVar("T")

# ── Module-level singleton pool ─────────────────────────────────
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_max_workers() -> int:
    """Read MAX_WORKERS from environment with validation."""
    raw = os.getenv("MAX_WORKERS", "8")
    try:
        workers = int(raw)
        if workers < 1:
            raise ValueError
        return workers
    except (ValueError, TypeError):
        logger.warning(f"Invalid MAX_WORKERS='{raw}', defaulting to 8")
        return 8


def get_executor() -> ThreadPoolExecutor:
    """
    Return the shared ThreadPoolExecutor (lazy, thread-safe init).

    Using a singleton avoids creating/destroying pools per request,
    which would negate the performance benefit of pooling.
    """
    global _executor
    if _executor is None:
        with _executor_lock:
            # Double-checked locking pattern
            if _executor is None:
                max_workers = _get_max_workers()
                _executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="orchestrator-worker",
                )
                logger.info(f"ThreadPoolExecutor initialized: max_workers={max_workers}")
    return _executor


def parallel_map(
    fn: Callable[..., T],
    items: Iterable,
    *,
    preserve_order: bool = True,
) -> list[T]:
    """
    Apply `fn` to each item in `items` using the shared thread pool.

    Args:
        fn: A pure function (no side effects on shared state).
        items: Iterable of arguments to map over.
        preserve_order: If True, results match input order (required for
                        deterministic ranking). If False, results arrive
                        in completion order (faster for fire-and-forget).

    Returns:
        List of results. If any task raises, the exception propagates.

    WHY THIS WRAPPER:
        Encapsulates the future-gathering pattern so callers just pass
        a function + items. Error handling, logging, and pool access
        are centralized here.
    """
    items_list = list(items)  # materialize for indexing
    if not items_list:
        return []

    executor = get_executor()
    results: list[T] = [None] * len(items_list)  # type: ignore[list-item]

    if preserve_order:
        # Submit all, then gather in index order → deterministic output
        futures: list[Future] = []
        for item in items_list:
            futures.append(executor.submit(fn, item))

        for idx, future in enumerate(futures):
            try:
                results[idx] = future.result()
            except Exception:
                logger.error(
                    f"parallel_map task {idx} failed for fn={fn.__name__}",
                    exc_info=True,
                )
                raise
    else:
        # as_completed ordering — used when order doesn't matter
        future_to_idx = {
            executor.submit(fn, item): idx
            for idx, item in enumerate(items_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                logger.error(
                    f"parallel_map task {idx} failed for fn={fn.__name__}",
                    exc_info=True,
                )
                raise

    return results


def shutdown_executor() -> None:
    """Gracefully shut down the thread pool (call during app shutdown)."""
    global _executor
    if _executor is not None:
        logger.info("Shutting down ThreadPoolExecutor...")
        _executor.shutdown(wait=True)
        _executor = None
