#!/usr/bin/env python
"""Đo tác động của bản vá `_is_self_consistent` trên cache verdict thật.

So sánh bản cũ (tìm chuỗi trên cả `reason`) với bản mới (tách mệnh đề) trên
toàn bộ verdict đã cache, để biết:
  * bao nhiêu verdict trước đây bị loại oan;
  * bao nhiêu verdict thật sự mâu thuẫn vẫn bị loại;
  * các verdict bị loại oan có thật sự là đáp án đúng hay không.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hgrag.eval.judge import (  # noqa: E402
    _PRAISE_MARKERS,
    _is_self_consistent,
    _split_clauses,
)

_SENTENCE_RE = __import__("re").compile(r"[^.!?]+[.!?]?")


def old_consistent(payload: dict) -> bool:
    """Bản cũ: tìm chuỗi khen trên toàn bộ ``reason``."""
    reason = str(payload.get("reason") or "")
    if not reason.strip():
        return False
    try:
        score = float(payload.get("score", 0) or 0)
    except (TypeError, ValueError):
        return False
    lowered = reason.lower()
    praise = any(marker in lowered for marker in _PRAISE_MARKERS)
    if praise and score < 1.0:
        return False
    sentences = [s.strip().lower() for s in _SENTENCE_RE.findall(reason) if len(s.strip()) > 12]
    if sentences:
        counts: Counter[str] = Counter()
        for sentence in sentences:
            counts[sentence] += 1
        if max(counts.values()) >= 2:
            return False
    return True


def main() -> int:
    stats = Counter()
    rescued: list[dict] = []
    still_rejected: list[dict] = []
    for path in (REPO_ROOT / "artifacts/llm_cache").rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        text = data.get("text", "") if isinstance(data, dict) else ""
        if '"supported_evidence_ids"' not in text:
            continue
        try:
            payload = json.loads(text)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        stats["total"] += 1
        old_ok = old_consistent(payload)
        new_ok = _is_self_consistent(payload)
        if old_ok and new_ok:
            stats["both_ok"] += 1
        elif old_ok and not new_ok:
            stats["new_rejects"] += 1
            still_rejected.append(payload)
        elif not old_ok and new_ok:
            stats["old_rejects_new_accepts"] += 1
            rescued.append(payload)
        else:
            stats["both_reject"] += 1
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print()
    print(f"--- {len(rescued)} verdict bị loại oan (nay được giữ lại) ---")
    for payload in rescued[:15]:
        print(f"  score={payload.get('score')} | {str(payload.get('reason'))[:110]}")
    print()
    print(f"--- {len(still_rejected)} verdict bị loại oan (nay vẫn bị loại) ---")
    for payload in still_rejected[:10]:
        print(f"  score={payload.get('score')} | {str(payload.get('reason'))[:110]}")
    print()
    print("--- ngữ cảnh mẫu của mệnh đề bị tách ---")
    for payload in rescued[:5]:
        for clause in _split_clauses(str(payload.get("reason") or "")):
            print("   *", clause[:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
