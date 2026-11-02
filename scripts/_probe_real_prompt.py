#!/usr/bin/env python
"""Tái hiện đúng prompt của pipeline để xác định nguyên nhân câu trả lời rỗng.

Các đo đạc trước dùng prompt *đơn giản hóa*.  Script này dùng **đúng**
`SYSTEM_PROMPT` / `USER_PROMPT_TEMPLATE` của `hgrag.qa.answer` và context
thật do pipeline dựng, rồi báo cáo `finish_reason` + số token suy luận.

Mục đích: phân biệt hai khả năng cho câu trả lời rỗng --
(a) mô hình thực sự *từ chối* (có câu trả lời), hay
(b) mô hình hết ngân sách token trong lúc suy luận (`text` rỗng).
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hgrag.qa.answer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")
CORPUS = os.environ.get("PROBE_CORPUS", "data/processed/vhe_corpus.jsonl")
BUDGETS = (256, 1024, 2048)

QUESTIONS = [
    "Ai là người sinh ra Đế Nghi?",
    "Trong sự kiện do Đế Minh thực hiện, đối tượng được nhắc đến là gì?",
    "Sự kiện sinh ra Đế Nghi và sự kiện sinh ra Kinh Dương Vương cùng do nhân vật nào thực hiện?",
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


def probe(client, name: str, prompt: str, budget: int) -> dict:
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "max_output_tokens": budget,
                "temperature": 0.0,
                "system_instruction": SYSTEM_PROMPT,
            },
        )
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
        "elapsed": round(time.perf_counter() - started, 1),
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "text_len": len(text or ""),
        "text_head": (text or "")[:220],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "finish_reasons": reasons,
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    chunks = load_chunks(12)
    context = "\n\n".join(f"[E{i + 1}] {c}" for i, c in enumerate(chunks))
    for question in QUESTIONS:
        prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question, plan="")
        for budget in BUDGETS:
            row = probe(client, question[:40], prompt, budget)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
