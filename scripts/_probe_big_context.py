#!/usr/bin/env python
"""Mô phỏng đúng điều kiện của hệ thống `ours` (context lớn + câu hỏi khó).

Đo đạc trước:
* context **nhỏ** (1–8 chunk) + câu hỏi khó -> kết thúc (thinking 250–800);
* **không** context + câu hỏi khó -> `MAX_TOKENS` ở mọi mức ngân sách.

Nhưng trong lượt chạy thật, `ours` dùng `context_max_tokens=6000` và vẫn trả
về rỗng cho 32/50 câu.  Script này dựng context đúng kích thước đó (nhiều
chunk thật, ghép lại) và đo ngân sách cần thiết để câu trả lời kết thúc.
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")
CORPUS = os.environ.get("PROBE_CORPUS", "data/processed/vhe_corpus.jsonl")

SYSTEM = (
    "Bạn là trợ lý trả lời câu hỏi dựa trên CONTEXT.\n"
    "Quy tắc:\n"
    "1. Chỉ dùng thông tin trong CONTEXT.\n"
    "2. Trích dẫn [E#] cho mỗi mệnh đề.\n"
    "3. Kết thúc bằng mục \"## Suy luận\" gồm các bước đánh số ngắn gọn.\n"
    "4. Nếu CONTEXT không đủ, nói rõ là không đủ thông tin."
)

HARD = "Trong sự kiện do Lâm Lư hầu Chu Táo thực hiện, đối tượng được nhắc đến là gì?"
EASY = "Ai là người sinh ra Đế Nghi?"


def load_chunks() -> list[str]:
    chunks: list[str] = []
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if text:
                chunks.append(text)
    return chunks


def build_context(chunks: list[str], n: int) -> str:
    return "\n\n".join(f"[E{i + 1}] {c}" for i, c in enumerate(chunks[:n]))


def probe(client, name: str, context: str, question: str, budget: int) -> dict:
    prompt = f"CONTEXT:\n{context}\n\nCÂU HỎI: {question}\n\nHãy trả lời theo quy tắc ở trên."
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config={
                "max_output_tokens": budget,
                "temperature": 0.0,
                "system_instruction": SYSTEM,
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
        "text_head": (text or "")[:200],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "finish_reasons": reasons,
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    chunks = load_chunks()
    print(f"# corpus chunks: {len(chunks)}", flush=True)

    cases: list[tuple[str, str, str, int]] = []
    for n in (12, 40):
        ctx = build_context(chunks, n)
        cases.append((f"hard_ctx{n}", ctx, HARD, 2048))
        cases.append((f"easy_ctx{n}", ctx, EASY, 2048))
    # ngân sách lớn cho context đầy đủ
    ctx40 = build_context(chunks, 40)
    cases.append(("hard_ctx40_big", ctx40, HARD, 8192))
    cases.append(("hard_ctx40_4k", ctx40, HARD, 4096))

    for name, ctx, question, budget in cases:
        row = probe(client, name, ctx, question, budget)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
