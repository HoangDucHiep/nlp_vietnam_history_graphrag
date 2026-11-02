"""A rate limiter shared between several *processes*.

The free tier of Google AI Studio applies the requests-per-minute quota *per
project and per model*, i.e. **globally** -- not per process.  The evaluation
harness runs several worker processes in parallel, and each of them has its
own in-memory limiter, so the sum of the workers would blow through the quota
and every worker would spend its time eating 429s.

This module implements the limit with a small file that every process locks
(``fcntl.flock``) before it inspects and updates the list of the calls of the
last minute.  The critical section is a few hundred microseconds, so the
contention is negligible compared to the multi-second latency of a call.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path


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
