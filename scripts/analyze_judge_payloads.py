#!/usr/bin/env python
"""Phân tích tối thiểu về payload judge mâu thuẫn.

Payload quan sát được:

    {"score":0,"correct":false,"grounded":true,
     "supported_evidence_ids":["E1"],
     "reason":"Đáp án đúng về nghĩa và đầy đủ so với đáp án tham chiếu và
     gold facts. Đáp án ứng viên là một câu hoàn chỉnh. Đáp án ứng viên là
     một câu câu hoàn chỉnh. Đáp án ứng viên là một câu hoàn chỉnh."}

Hai dấu hiệu: (1) điểm 0 nhưng lý do khen đáp án; (2) lý do **lặp vòng**
(degenerate repetition).  Script này quét toàn bộ cache để đo:

* có bao nhiêu payload bị lặp;
* có bao nhiêu payload mâu thuẫn (lý do khen / điểm thấp);
* tần suất theo model và theo ngân sách token.

Kết quả cho biết đây là lỗi *phổ biến* (cần sửa ở tầng judge) hay chỉ là
nhiễu hiếm (chấp nhận được, chỉ cần báo cáo).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_SENT = re.compile(r"[^.!?]+[.!?]?")

PRAISE = (
    "đúng về nghĩa",
    "đúng hoàn toàn",
    "chính xác và đầy đủ",
    "đúng và đầy đủ",
    "đúng và chính xác",
    "trả lời đúng",
)


def sentences(reason: str) -> list[str]:
    return [s.strip().lower() for s in _SENT.findall(reason or "") if len(s.strip()) > 12]


def is_repetitive(reason: str) -> bool:
    sents = sentences(reason)
    if not sents:
        return False
    return Counter(sents).most_common(1)[0][1] >= 2


def is_contradictory(payload: dict) -> bool:
    reason = str(payload.get("reason") or "").lower()
    if not any(p in reason for p in PRAISE):
        return False
    try:
        score = float(payload.get("score", 0) or 0)
    except (TypeError, ValueError):
        return False
    return score < 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quét cache judge tìm payload mâu thuẫn")
    parser.add_argument("--cache", default="artifacts/llm_cache")
    args = parser.parse_args(argv)

    root = Path(args.cache)
    total = 0
    parsed = 0
    repetitive = 0
    contradictory = 0
    by_model: Counter[str] = Counter()
    rep_by_model: Counter[str] = Counter()
    contra_by_model: Counter[str] = Counter()
    examples: list[dict] = []

    for path in sorted(root.glob("*/*.json")):
        total += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        if not isinstance(obj, dict) or "score" not in obj:
            continue
        parsed += 1
        model = str(payload.get("model") or "?")
        by_model[model] += 1
        rep = is_repetitive(str(obj.get("reason") or ""))
        contra = is_contradictory(obj)
        if rep:
            repetitive += 1
            rep_by_model[model] += 1
        if contra:
            contradictory += 1
            contra_by_model[model] += 1
            if len(examples) < 10:
                examples.append(
                    {"path": str(path), "model": model, "score": obj.get("score"), "reason": obj.get("reason")}
                )

    report = {
        "cache_entries": total,
        "judge_payloads": parsed,
        "n_repetitive": repetitive,
        "n_contradictory": contradictory,
        "pct_repetitive": round(100 * repetitive / parsed, 2) if parsed else 0.0,
        "pct_contradictory": round(100 * contradictory / parsed, 2) if parsed else 0.0,
        "by_model": dict(by_model),
        "repetitive_by_model": dict(rep_by_model),
        "contradictory_by_model": dict(contra_by_model),
        "examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
