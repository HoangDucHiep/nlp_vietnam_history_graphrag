"""Tests for the token-based shared limiter.

The free tier of the Gemma models is capped by *input tokens per minute*
(``GenerateContentInputTokensPerModelPerMinute``), so the limiter has to pace
by tokens, not by request count.  See ``docs/10_quota_gemini_free_tier.md``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from hgrag.llm.rate_limit import SharedTokenLimiter

WINDOW = 0.5


def test_token_limiter_allows_budget_immediately(tmp_path) -> None:
    limiter = SharedTokenLimiter(tmp_path / "tl.json", tpm=1000, window=WINDOW)
    start = time.monotonic()
    limiter.acquire(400)
    limiter.acquire(400)
    assert time.monotonic() - start < 0.4


def test_token_limiter_blocks_when_budget_is_exhausted(tmp_path) -> None:
    limiter = SharedTokenLimiter(tmp_path / "tl.json", tpm=100, window=WINDOW)
    limiter.acquire(100)
    start = time.monotonic()
    # The budget is full: the next call has to wait for the window to slide.
    limiter.acquire(50)
    assert time.monotonic() - start > 0.1


def test_token_limiter_is_shared_between_instances(tmp_path) -> None:
    """Two limiters on the same file share one global token budget."""
    path = tmp_path / "tl.json"
    a = SharedTokenLimiter(path, tpm=100, window=WINDOW)
    b = SharedTokenLimiter(path, tpm=100, window=WINDOW)
    a.acquire(60)
    b.acquire(30)
    # Only 10 tokens are left in the shared budget.
    start = time.monotonic()
    b.acquire(50)
    assert time.monotonic() - start > 0.1


def test_token_limiter_survives_a_corrupt_file(tmp_path) -> None:
    path = tmp_path / "tl.json"
    path.write_text("{not json", encoding="utf-8")
    limiter = SharedTokenLimiter(path, tpm=100, window=WINDOW)
    limiter.acquire(50)
    limiter.acquire(50)


def test_token_limiter_under_concurrency(tmp_path) -> None:
    """Concurrent workers never exceed the shared budget."""
    path = tmp_path / "tl.json"
    limiter = SharedTokenLimiter(path, tpm=200, window=5.0)

    def work(_: int) -> None:
        limiter.acquire(20)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(8)))

    pairs = limiter._state.read()
    used = sum(amount for _, amount in pairs)
    assert used <= 200
