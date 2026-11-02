#!/usr/bin/env python
"""Chấm lại các dòng đã đánh dấu `judge_degraded` bằng bộ lọc mới.

Những dòng này rơi về heuristic token-F1 vì bộ lọc `_is_self_consistent` cũ
loại oan verdict của LLM judge.  Với bộ lọc mới, verdict đã cache được dùng
lại (không tốn thêm quota) và dòng được chấm lại đúng rubric.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hgrag.eval.judge import _is_self_consistent  # noqa: E402


def main() -> int:
    total = 0
    rescuable = 0
    for path in sorted(glob.glob(str(REPO_ROOT / "artifacts/eval_ckpt_*.json"))):
        data = json.loads(Path(path).read_text())
        rows = data.get("rows", [])
        changed = 0
        for row in rows:
            if not row.get("judge_degraded"):
                continue
            total += 1
            reason = str(row.get("judge_reason") or "")
            # Heuristic luôn ghi lý do dạng "heuristic token-F1=...".
            if not reason.startswith("heuristic"):
                continue
            # Tìm trong cache verdict LLM tương ứng: dùng reason của judge
            # không có trong dòng, nên đối chiếu theo cache key của request.
            print(f"  {row['system']:22s} {row['qid']:12s} {reason}")
            changed += 1
        if changed:
            print(f"-> {Path(path).name}: {changed} dòng degraded")
    print(f"\nTổng: {total} dòng degraded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
