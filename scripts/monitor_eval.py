#!/usr/bin/env python
"""Theo dõi chất lượng của lượt đánh giá đang chạy.

Đọc các file checkpoint, gộp lại và báo cáo:
* số dòng theo hệ thống;
* số câu trả lời **rỗng** (lỗi "thinking ăn hết ngân sách" chưa được xử lý);
* số dòng mà judge rơi về heuristic (``judge_degraded``);
* phân bố điểm và độ trễ.

Dùng để kiểm tra bản sửa *trong lúc* lượt chạy còn tiếp diễn.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANNED = "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Theo dõi lượt đánh giá")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    root = REPO_ROOT / args.artifacts
    rows: list[dict] = []
    for path in sorted(root.glob("eval_ckpt_*.json")):
        try:
            rows.extend(json.loads(path.read_text(encoding="utf-8")).get("rows", []))
        except (json.JSONDecodeError, OSError):
            continue

    by_system: dict[str, list[dict]] = {}
    for row in rows:
        by_system.setdefault(row["system"], []).append(row)

    total_expected = 150 * 11
    print(f"Tổng số dòng: {len(rows)}/{total_expected} ({100 * len(rows) / total_expected:.1f}%)")
    print(f"{'hệ thống':<26}{'n':>5}{'rỗng':>7}{'heur':>7}{'canned':>8}{'p50(s)':>9}{'đúng%':>8}")
    n_empty = n_degraded = n_canned = 0
    for system in sorted(by_system):
        subset = by_system[system]
        empty = sum(1 for r in subset if not (r.get("answer") or "").strip())
        degraded = sum(1 for r in subset if r.get("judge_degraded"))
        canned = sum(1 for r in subset if (r.get("answer") or "").strip() == CANNED)
        latencies = [float(r.get("elapsed") or 0.0) for r in subset]
        p50 = statistics.median(latencies) if latencies else 0.0
        acc = 100 * sum(1 for r in subset if r.get("judge_correct")) / len(subset)
        n_empty += empty
        n_degraded += degraded
        n_canned += canned
        print(
            f"{system:<26}{len(subset):>5}{empty:>7}{degraded:>7}{canned:>8}{p50:>9.1f}{acc:>8.1f}"
        )
    print()
    print(f"Tổng: rỗng={n_empty}  judge_degraded={n_degraded}  canned_refusal={n_canned}")

    if args.verbose:
        scores = Counter(r.get("judge_score") for r in rows)
        print("Phân bố điểm:", dict(sorted(scores.items(), key=lambda kv: str(kv[0]))))
        types = Counter(r.get("question_type") for r in rows)
        print("Theo loại câu hỏi:", dict(types))
    return 0


if __name__ == "__main__":
    sys.exit(main())
