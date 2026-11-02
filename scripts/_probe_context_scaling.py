#!/usr/bin/env python
"""Kiểm tra: độ dài context có làm Gemma 4 suy luận không kết thúc không?

Đo đạc trước cho thấy:
* prompt ``grounded`` với context **nhỏ** kết thúc ở ~733 token suy luận;
* mọi biến thể prompt ``closed_book`` (không context) không kết thúc, kể cả
  ở 8192 token.

Trong lượt chạy thật, hệ thống ``ours`` cũng trả về rỗng (32/50 câu), nơi
context lên tới 6000 token.  Giả thuyết cần kiểm tra: **Gemma 4 suy luận
vô hạn khi không có câu trả lời chắc chắn** (closed-book với câu hỏi ngoài
kiến thức, hoặc context không đủ để trả lời).

Script đo:
1. context nhỏ / vừa / lớn (0 / 1 / 4 / 8 chunk thật từ corpus);
2. câu hỏi biết đáp án (có context) vs. câu hỏi không có trong context;
3. ngân sách 2048 và 8192.
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
    "Trả lời chỉ dựa trên CONTEXT, trích dẫn [E#] cho mỗi mệnh đề."
)


def load_chunks(limit: int) -> list[str]:
    chunks: list[str] = []
    if not os.path.exists(CORPUS):
        return chunks
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if text:
                chunks.append(text)
            if len(chunks) >= limit:
                break
    return chunks


def build_context(chunks: list[str]) -> str:
    return "\n\n".join(f"[E{i + 1}] {c}" for i, c in enumerate(chunks))


def probe(client, name: str, prompt: str, budget: int) -> dict:
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
        "text_head": (text or "")[:150],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "finish_reasons": reasons,
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    chunks = load_chunks(8)
    print(f"# corpus chunks loaded: {len(chunks)}", flush=True)

    ask_known = "Ai là người sinh ra Đế Nghi?"
    ask_absent = "Người nào đã tham gia Hội nghị TW Đại hội XIII của Đảng?"

    cases: list[tuple[str, str, int]] = []
    for n_chunks in (0, 1, 4, 8):
        context = build_context(chunks[:n_chunks])
        body = f"CONTEXT:\n{context}\n\n" if context else ""
        cases.append((f"known_ctx{n_chunks}", f"{body}CÂU HỎI: {ask_known}", 2048))
        cases.append((f"absent_ctx{n_chunks}", f"{body}CÂU HỎI: {ask_absent}", 2048))
    # ngân sách lớn cho trường hợp đã biết là hỏng
    cases.append(("known_ctx4_big", f"CONTEXT:\n{build_context(chunks[:4])}\n\nCÂU HỎI: {ask_known}", 8192))
    cases.append(("absent_ctx4_big", f"CONTEXT:\n{build_context(chunks[:4])}\n\nCÂU HỎI: {ask_absent}", 8192))

    for name, prompt, budget in cases:
        row = probe(client, name, prompt, budget)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
