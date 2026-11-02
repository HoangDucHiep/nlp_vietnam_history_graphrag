"""Check which chat models still have quota *and* belong to their own bucket.

The free tier gives every model its own daily/minute quota, so two models
that both work can double the throughput -- but only if they do not share a
bucket.  This script probes the survivors in parallel and reports the RPM
each of them sustains.
"""

from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")

from google import genai  # noqa: E402

from hgrag.config import get_settings  # noqa: E402

CANDIDATES = [
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemini-3-flash-preview",
    "gemini-robotics-er-2-preview",
    "gemini-3.5-transcribe",
]

settings = get_settings()
client = genai.Client(api_key=settings.gemini_api_key)


def probe(model: str) -> tuple[str, float, str]:
    t0 = time.monotonic()
    try:
        client.models.generate_content(
            model=model, contents="Trả lời bằng một từ: ok"
        )
        return model, time.monotonic() - t0, "OK"
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        m = re.search(r"limit: (\d+), model: ([\w.-]+)", text)
        r = re.search(r"retry in ([\d.]+)(s|m|h)", text)
        reset = ""
        if r:
            secs = float(r.group(1)) * {"s": 1, "m": 60, "h": 3600}[r.group(2)]
            reset = f"{secs/60:.0f}min"
        return (
            model,
            time.monotonic() - t0,
            f"quota={m.group(1) if m else '?'}/{m.group(2) if m else '?'} {reset}",
        )


with ThreadPoolExecutor(max_workers=len(CANDIDATES)) as pool:
    for model, dt, info in pool.map(probe, CANDIDATES):
        print(f"{model:32s} {dt:6.2f}s  {info}")
