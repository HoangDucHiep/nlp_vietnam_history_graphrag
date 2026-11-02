#!/usr/bin/env python
"""Hai câu hỏi còn lại về ngân sách output của Gemma 4.

**Q1 -- `response_schema` có tắt thinking không?**
Judge dùng `max_output_tokens=256` mà vẫn chấm được, còn phần sinh câu trả
lời (budget 2048) thì hay rỗng.  Giả thuyết: khi ép *structured output*
(``response_mime_type=application/json`` + ``response_schema``) thì mô hình
không suy luận, nên toàn bộ ngân sách dành cho câu trả lời.

**Q2 -- Câu hỏi khó cần ngân sách bao nhiêu?**
Đo với context thật (12 chunk) và câu hỏi template VHE ở 1024 / 2048 / 4096 /
8192, để biết ngân sách tối thiểu để câu trả lời kết thúc.
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hgrag.eval.judge import JUDGE_SCHEMA, JUDGE_SYSTEM, JUDGE_TEMPLATE  # noqa: E402
from hgrag.qa.answer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")
CORPUS = os.environ.get("PROBE_CORPUS", "data/processed/vhe_corpus.jsonl")

QUESTIONS = [
    "Trong sự kiện do Đế Minh thực hiện, đối tượng được nhắc đến là gì?",
    "Sự kiện sinh ra Đế Nghi và sự kiện sinh ra Kinh Dương Vương cùng do nhân vật nào thực hiện?",
    "Sự kiện Lý Ông Trọng đảm nhiệm chức vụ nhằm mục đích gì, và diễn ra vào khoảng thời gian nào?",
]


def load_chunks(limit: int) -> list[str]:
    chunks: list[str] = []
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if text:
                chunks.append(text)
            if len(chunks) >= limit:
                break
    return chunks


def probe(client, name: str, contents: str, budget: int, system: str, schema=None) -> dict:
    config: dict = {"max_output_tokens": budget, "temperature": 0.0}
    if system:
        config["system_instruction"] = system
    if schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = schema
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
    except Exception as exc:  # noqa: BLE001
        return {"case": name, "budget": budget, "error": str(exc)[:200]}
    usage = getattr(resp, "usage_metadata", None)
    text = getattr(resp, "text", None)
    reasons = [
        str(getattr(cand, "finish_reason", "")) for cand in (getattr(resp, "candidates", None) or [])
    ]
    return {
        "case": name,
        "budget": budget,
        "schema": schema is not None,
        "elapsed": round(time.perf_counter() - started, 1),
        "text_len": len(text or ""),
        "text_head": (text or "")[:150],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "finish_reasons": reasons,
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    chunks = load_chunks(12)
    context = "\n\n".join(f"[E{i + 1}] {c}" for i, c in enumerate(chunks))

    print("### Q1: response_schema co tat thinking khong?", flush=True)
    judge_prompt = JUDGE_TEMPLATE.format(
        question="Ai là người sinh ra Đế Nghi?",
        reference="�ế Minh",
        gold_facts="- Đế Minh -PARTICIPATED_IN-> Đế Minh sinh ra Đế Nghi",
        context="[E1] " + chunks[0][:600],
        answer="Đế Minh là người sinh ra Đế Nghi [E1].",
    )
    for budget in (256, 1024):
        row = probe(client, "judge_schema", judge_prompt, budget, JUDGE_SYSTEM, JUDGE_SCHEMA)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1.0)
    row = probe(client, "judge_noschema", judge_prompt, budget, JUDGE_SYSTEM, None)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    time.sleep(1.0)

    print("### Q2: ngan sach toi thieu cho cau hoi kho", flush=True)
    for question in QUESTIONS:
        prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question, plan="")
        for budget in (1024, 2048, 4096, 8192):
            row = probe(client, question[:38], prompt, budget, SYSTEM_PROMPT)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
