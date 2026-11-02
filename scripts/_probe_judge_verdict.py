#!/usr/bin/env python
"""Tái hiện phán quyết của LLM judge cho một dòng bị chấm sai.

Dùng để kiểm chứng xem payload thô của judge có thật sự tự mâu thuẫn hay
chỉ bị bộ lọc `_is_self_consistent` loại oan (rơi về heuristic token-F1).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hgrag.eval.judge import (  # noqa: E402
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    JUDGE_TEMPLATE,
    _is_self_consistent,
)

QID = os.environ.get("PROBE_QID", "vhe_00001")
SYSTEM = os.environ.get("PROBE_SYSTEM", "ours")


def main() -> int:
    from google import genai

    ckpt = Path("artifacts/eval_ckpt_b.json")
    if not ckpt.exists():
        print("không có checkpoint")
        return 1
    rows = json.loads(ckpt.read_text())["rows"]
    row = next(
        (r for r in rows if r["qid"] == QID and r["system"] == SYSTEM),
        None,
    )
    if row is None:
        print("không tìm thấy dòng", QID, SYSTEM)
        return 1

    prompt = JUDGE_TEMPLATE.format(
        question=row["question"],
        reference=row["reference"] or "(không có đáp án tham chiếu)",
        gold_facts="(không có)",
        context="(không có context)",
        answer=row["answer"] or "(không có đáp án)",
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    req = {
        "contents": prompt,
        "config": {
            "temperature": 0.0,
            "max_output_tokens": 512,
            "response_mime_type": "application/json",
            "response_schema": JUDGE_SCHEMA,
            "system_instruction": JUDGE_SYSTEM,
        },
    }
    for attempt in range(3):
        resp = client.models.generate_content(model="gemma-4-26b-a4b-it", **req)
        raw = resp.text or ""
        try:
            payload = json.loads(raw)
        except ValueError:
            print(f"[{attempt}] JSON hỏng: {raw[:200]!r}")
            continue
        print(f"[{attempt}] payload: {json.dumps(payload, ensure_ascii=False)}")
        print(f"      self_consistent={_is_self_consistent(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
