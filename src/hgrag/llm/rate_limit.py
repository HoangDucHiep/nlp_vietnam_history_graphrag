"""Rate limiters shared between several *processes*.

The free tier of Google AI Studio applies the quota *globally* -- per project
and per model -- not per process.  The evaluation harness runs several worker
processes in parallel, and each of them has its own in-memory limiter, so the
sum of the workers would blow through the quota and every worker would spend
its time eating 429s.

Two limits are needed:

``SharedRateLimiter``
    Limits the number of **calls** per minute.  Used for the models whose
    free tier is capped by requests/day (the Gemini Flash family).

``SharedTokenLimiter``
    Limits the number of **input tokens** per minute.  The Gemma family is
    capped by ``GenerateContentInputTokensPerModelPerMinute`` (16 000
    tokens/min), so pacing by request count alone is not enough: a single
    long prompt can exhaust the whole minute.  See
    ``docs/10_quota_gemini_free_tier.md``.

Both limiters coordinate the workers through a small file that every process
locks (``fcntl.flock``) before it inspects and updates the state of the last
minute.  The critical section is a few hundred microseconds, so the
contention is negligible compared to the multi-second latency of a call.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

__all__ = ["SharedRateLimiter", "SharedTokenLimiter"]


class _FileLockState:
    """The shared state of the last window, stored as JSON in a locked file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Make sure the file exists so that the flock is always taken on a
        # regular file.
        if not self.path.exists():
            try:
                self.path.touch()
            except OSError:  # pragma: no cover - best effort
                pass

    def read(self) -> list[list[float]]:
        """Read the recorded ``(timestamp, amount)`` pairs."""
        try:
            with open(self.path, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.seek(0)
                    raw = handle.read()
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - best effort
            return []
        if not raw.strip():
            return []
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(payload, list):
            return []
        pairs: list[list[float]] = []
        for item in payload:
            try:
                pairs.append([float(item[0]), float(item[1])])
            except (TypeError, ValueError, IndexError):
                continue
        return pairs

    def write(self, pairs: list[list[float]]) -> None:
        """Replace the recorded pairs (the caller holds the lock)."""
        try:
            with open(self.path, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.seek(0)
                    handle.truncate()
                    handle.write(json.dumps([[round(t, 3), round(a, 3)] for t, a in pairs]))
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - best effort
            pass


class SharedRateLimiter:
    """Limit the number of calls per minute across every process on the host.

    Parameters
    ----------
    path:
        The file used to coordinate the workers.  It should live on a local
        filesystem (a shared lock does not work over NFS).
    rpm:
        The maximum number of calls per minute.
    """

    def __init__(self, path: str | Path, rpm: int, window: float = 60.0) -> None:
        self.path = Path(path)
        self.rpm = max(1, int(rpm))
        self.window = float(window)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Make sure the file exists so that the flock is always taken on a
        # regular file.
        if not self.path.exists():
            try:
                self.path.touch()
            except OSError:  # pragma: no cover - best effort
                pass

    def acquire(self) -> None:
        """Block until a slot is free, then consume it."""
        while True:
            wait_for = self._try_consume(self.window)
            if wait_for <= 0.0:
                return
            time.sleep(wait_for)

    def _try_consume(self, window: float) -> float:
        """Consume a slot; return the seconds to wait when none is free."""
        now = time.time()
        with open(self.path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                raw = handle.read()
                try:
                    stamps = [float(x) for x in json.loads(raw)] if raw.strip() else []
                except (json.JSONDecodeError, ValueError):
                    stamps = []
                stamps = [t for t in stamps if now - t < window]
                if len(stamps) >= self.rpm:
                    wait_for = window - (now - stamps[0]) + 0.1
                else:
                    stamps.append(now)
                    wait_for = 0.0
                    handle.seek(0)
                    handle.truncate()
                    handle.write(json.dumps([round(t, 3) for t in stamps]))
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return max(0.0, wait_for)


class SharedTokenLimiter:
    """Limit the number of **input tokens** per minute across processes.

    The free tier of the Gemma models is capped by
    ``GenerateContentInputTokensPerModelPerMinute`` (16 000 tokens/min).  A
    request-count limiter is not sufficient: a request with a long prompt can
    use a large part of the budget on its own.

    The limiter keeps the ``(timestamp, tokens)`` pairs of the last window in
    a file that every process locks, so the workers of a parallel evaluation
    share one global budget.

    Parameters
    ----------
    path:
        The file used to coordinate the workers.
    tpm:
        The maximum number of input tokens per minute.
    window:
        The width of the sliding window in seconds (the API uses 60 s).
    """

    def __init__(self, path: str | Path, tpm: int, window: float = 60.0) -> None:
        self.path = Path(path)
        self.tpm = max(1, int(tpm))
        self.window = float(window)
        self._state = _FileLockState(self.path)

    def acquire(self, tokens: int) -> None:
        """Block until ``tokens`` fit into the current window, then book them."""
        amount = max(1, int(tokens))
        while True:
            wait_for = self._try_consume(amount)
            if wait_for <= 0.0:
                return
            time.sleep(wait_for)

    def _try_consume(self, amount: int) -> float:
        """Book ``amount`` tokens; return the seconds to wait when it does not fit."""
        now = time.time()
        with open(self.path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                raw = handle.read()
                try:
                    pairs = [
                        [float(p[0]), float(p[1])]
                        for p in json.loads(raw)
                        if isinstance(p, (list, tuple)) and len(p) == 2
                    ] if raw.strip() else []
                except (json.JSONDecodeError, ValueError, TypeError):
                    pairs = []
                pairs = [p for p in pairs if now - p[0] < self.window]
                used = sum(p[1] for p in pairs)
                if used + amount > self.tpm:
                    if used > 0:
                        # The budget frees up progressively: wait until the
                        # oldest entry leaves the window, then re-check.
                        wait_for = self.window - (now - pairs[0][0]) + 0.1
                    else:
                        # A single request bigger than the whole budget
                        # cannot be split -- let it through once the window is
                        # empty, otherwise the limiter would loop forever.
                        pairs.append([now, amount])
                        wait_for = 0.0
                else:
                    pairs.append([now, amount])
                    wait_for = 0.0
                if wait_for <= 0.0:
                    handle.seek(0)
                    handle.truncate()
                    handle.write(json.dumps([[round(t, 3), round(a, 3)] for t, a in pairs]))
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return max(0.0, wait_for)
