"""Tests for the cross-process rate limiter."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from hgrag.llm.rate_limit import SharedRateLimiter

# A short window keeps the tests fast: the limiter only has to *space* the
# calls out, the exact length of the window is irrelevant for the behaviour.
WINDOW = 0.5


def test_limiter_allows_up_to_rpm_immediately(tmp_path) -> None:
    path = tmp_path / "rl.json"
    limiter = SharedRateLimiter(path, rpm=5, window=WINDOW)
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    assert time.monotonic() - start < 0.4


def test_limiter_blocks_the_extra_call(tmp_path) -> None:
    path = tmp_path / "rl.json"
    limiter = SharedRateLimiter(path, rpm=2, window=WINDOW)
    limiter.acquire()
    limiter.acquire()
    start = time.monotonic()
    # The third call has to wait for the first one to leave the window.
    limiter.acquire()
    assert time.monotonic() - start > 0.1


def test_limiter_is_shared_between_instances(tmp_path) -> None:
    """Two limiters on the same file share the budget."""
    path = tmp_path / "rl.json"
    a = SharedRateLimiter(path, rpm=3, window=WINDOW)
    b = SharedRateLimiter(path, rpm=3, window=WINDOW)
    a.acquire()
    a.acquire()
    b.acquire()
    # ``b`` saw the two calls of ``a`` -> only one slot was left.
    start = time.monotonic()
    b.acquire()
    assert time.monotonic() - start > 0.1


def test_limiter_survives_a_corrupt_file(tmp_path) -> None:
    path = tmp_path / "rl.json"
    path.write_text("not json at all", encoding="utf-8")
    limiter = SharedRateLimiter(path, rpm=2, window=WINDOW)
    limiter.acquire()
    limiter.acquire()


def test_limiter_under_concurrency(tmp_path) -> None:
    path = tmp_path / "rl.json"
    limiter = SharedRateLimiter(path, rpm=8, window=WINDOW)

    def work(_: int) -> None:
        limiter.acquire()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(8)))
