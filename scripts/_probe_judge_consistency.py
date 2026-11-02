#!/usr/bin/env python
"""Tái lập lỗi judge trả `score=0` nhưng lý do nói "đáp án đúng".

Hiện tượng: payload của judge cho `vhe_00044` là

    {"score":0, "correct":false, "grounded":true,
     "supported_evidence_ids":["E1"],
     "reason":"Đáp án đúng về nghĩa và đầy đủ so với đáp án tham chiếu và
               gold facts. Đáp án ứng viên là một câu hoàn chỉnh. Đáp án ứng
               viên là một câu câu hoàn chỉnh. Đáp án ứng viên là một câu hoàn
               chỉnh."}

Lý do lặp vòng (degenerate repetition) là dấu hiệu điển hình của Gemma khi
bị ép *structured output*.  Script này gọi lại đúng prompt đó nhiều lần để
xem (a) lỗi có tái lập không, (b) ảnh hưởng của ngân sách token, và (c) ảnh
hưởng của việc bỏ ép schema.
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hgrag.eval.judge import JUDGE_SCHEMA, JUDGE_SYSTEM, JUDGE_TEMPLATE  # noqa: E402

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")

QUESTION = "Sự kiện sinh ra Đế Nghi và sự kiện sinh ra Kinh Dương Vương cùng do nhân vật nào thực hiện?"
REFERENCE = "Đế Minh"
GOLD_FACTS = "- sinh ra Đế Nghi -INVOLVES-> Đế Minh\n- sinh ra Kinh Dương Vương -INVOLVES-> Đế Minh"
CONTEXT = (
    "[E1] Xưa cháu ba đời của Viêm Đế họ Thần Nông là Đế Minh sinh ra Đế Nghi, "
    "sau Đế Minh nhân đi tuần phương Nam, đến Ngũ Lĩnh lấy con gái Vụ Tiên, "
    "sinh ra vua [Kinh Dương Vương]."
)
ANSWER = "Cả hai sự kiện sinh ra Đế Nghi và sinh ra Kinh Dương Vương đều do Đế Minh thực hiện [E1]."


def run(client, prompt: str, budget: int, schema: bool) -> dict:
    config: dict = {"max_output_tokens": budget, "temperature": 0.0}
    if schema:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = JUDGE_SCHEMA
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={**config, "system_instruction": JUDGE_SYSTEM},
    )
    text = getattr(resp, "text", "") or ""
    usage = getattr(resp, "usage_metadata", None)
    try:
        payload = json.loads(text)
    except ValueError:
        payload = None
    return {
        "budget": budget,
        "schema": schema,
        "score": (payload or {}).get("score"),
        "correct": (payload or {}).get("correct"),
        "reason": (payload or {}).get("reason"),
        "raw_len": len(text),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = JUDGE_TEMPLATE.format(
        question=QUESTION,
        reference=REFERENCE,
        gold_facts=GOLD_FACTS,
        context=CONTEXT,
        answer=ANSWER,
    )
    for label, budget, schema in [
        ("schema-512", 512, True),
        ("schema-512", 512, True),
        ("schema-512", 512, True),
        ("schema-1024", 1024, True),
        ("schema-2048", 2048, True),
        ("noschema-1024", 1024, False),
        ("noschema-1024", 1024, False),
    ]:
        row = run(client, prompt, budget, schema)
        row["label"] = label
        print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
