#!/usr/bin/env python
"""Cô lập nguyên nhân khiến Gemma 4 không kết thúc ở prompt closed-book.

Đo đạc đã xác nhận:
* prompt có CONTEXT (grounded) -> kết thúc, thinking 250–800 token;
* prompt judge -> kết thúc, thinking ~650 token;
* **mọi** biến thể closed-book -> `MAX_TOKENS` ở 2048 *và* 8192 token.

Script này thay từng yếu tố một để tìm yếu tố kích hoạt vòng lặp suy luận:
1. cụm "closed-book / KIẾN THỨC CỦA CHÍNH BẠN";
2. yêu cầu "## Suy luận";
3. câu hỏi khó (thực thể VHE hiếm) so với câu hỏi dễ;
4. câu hỏi ngoài phạm vi kiến thức.
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")
BUDGET = 2048

EASY = "Ai là người sinh ra Đế Nghi?"
HARD = "Trong sự kiện do Lâm Lư hầu Chu Táo thực hiện, đối tượng được nhắc đến là gì?"
SIMPLE = "Thủ tướng đầu tiên của Việt Nam Dân chủ Cộng hòa là ai?"

EXPERT = "Bạn là chuyên gia lịch sử Việt Nam."

CASES: list[tuple[str, str, str]] = [
    # (tên, system, câu hỏi)
    ("expert_easy", EXPERT, SIMPLE),
    ("expert_hard", EXPERT, EASY),
    ("expert_harder", EXPERT, HARD),
    (
        "closedbook_easy",
        EXPERT
        + " Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), "
        "không có tài liệu tham chiếu nào được cung cấp.",
        SIMPLE,
    ),
    (
        "closedbook_hard",
        EXPERT
        + " Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), "
        "không có tài liệu tham chiếu nào được cung cấp.",
        EASY,
    ),
    (
        "closedbook_harder",
        EXPERT
        + " Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), "
        "không có tài liệu tham chiếu nào được cung cấp.",
        HARD,
    ),
    (
        "closedbook_hard_no_phrase",
        EXPERT + " Bạn trả lời bằng kiến thức của bạn.",
        EASY,
    ),
    (
        "closedbook_hard_plus_suy_luan",
        EXPERT
        + " Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), "
        "không có tài liệu tham chiếu nào được cung cấp. Kết thúc bằng mục "
        "\"## Suy luận\" gồm các bước đánh số (1., 2.) ngắn gọn.",
        EASY,
    ),
    (
        "closedbook_hard_forced_answer",
        EXPERT
        + " Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), "
        "không có tài liệu tham chiếu nào được cung cấp. KHÔNG hỏi lại người dùng. "
        "Đưa ra câu trả lời tốt nhất có thể dù kiến thức của bạn không chắc chắn.",
        EASY,
    ),
]


def probe(client, name: str, system: str, question: str, budget: int) -> dict:
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=f"CÂU HỎI: {question}\n\nHãy trả lời theo quy tắc ở trên.",
            config={
                "max_output_tokens": budget,
                "temperature": 0.0,
                "system_instruction": system,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return {"case": name, "error": str(exc)[:200]}
    usage = getattr(resp, "usage_metadata", None)
    text = getattr(resp, "text", None)
    reasons = [
        str(getattr(cand, "finish_reason", "")) for cand in (getattr(resp, "candidates", None) or [])
    ]
    return {
        "case": name,
        "budget": budget,
        "elapsed": round(time.perf_counter() - started, 1),
        "text_len": len(text or ""),
        "text_head": (text or "")[:200],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "finish_reasons": reasons,
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for name, system, question in CASES:
        row = probe(client, name, system, question, BUDGET)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
