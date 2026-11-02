#!/usr/bin/env python
"""Tìm prompt thoát cho baseline closed-book (B0).

Đo đạc đã xác nhận: Gemma 4 **không kết thúc** suy luận khi câu hỏi nằm ngoài
kiến thức tham số *và* không có CONTEXT -- mọi mức ngân sách (256 → 8192) đều
dẫn tới `MAX_TOKENS` với `text` rỗng.  Đây chính là nguyên nhân 65/110 câu
closed-book của lượt chạy trước bị đổi thành câu "từ chối có sẵn".

Script này thử các cách cho mô hình một **lối thoát** rõ ràng (trả lời
"không biết" là hợp lệ), đồng thời giữ nguyên ngữ nghĩa của baseline
closed-book: không CONTEXT, không trích dẫn.
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")
BUDGET = 2048

HARD = "Ai là người sinh ra Đế Nghi?"
HARDER = "Trong sự kiện do Lâm Lư hầu Chu Táo thực hiện, đối tượng được nhắc đến là gì?"
HARDEST = "Sự kiện Lý Ông Trọng đảm nhiệm chức vụ nhằm mục đích gì, và diễn ra vào khoảng thời gian nào?"

BASE = (
    "Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC "
    "CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp."
)

VARIANTS: dict[str, str] = {
    # Cho phép "không biết" là một câu trả lời hợp lệ
    "allow_unknown": (
        BASE
        + "\nQuy tắc:\n"
        "1. Trả lời trực tiếp, súc tích, bằng tiếng Việt.\n"
        "2. KHÔNG gắn trích dẫn [E#].\n"
        "3. Nếu bạn KHÔNG CHẮC CHẮN hoặc không biết, hãy trả lời đúng câu: "
        "\"Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.\" "
        "Đây là một câu trả lời HỢP LỆ, không phải lỗi.\n"
        "4. KHÔNG suy luận dài, KHÔNG tự hỏi lại. Chỉ viết câu trả lời cuối cùng."
    ),
    # Cấm rõ ràng phần "suy luận"
    "no_reasoning": (
        BASE
        + "\nQuy tắc:\n"
        "1. Trả lời trong TỐI ĐA 2 câu, bằng tiếng Việt.\n"
        "2. KHÔNG gắn trích dẫn [E#].\n"
        "3. KHÔNG viết phần suy luận, KHÔNG giải thích, KHÔNG hỏi lại người dùng.\n"
        "4. Nếu không biết, viết: \"Tôi không biết.\""
    ),
    # Bản ngắn nhất: chỉ yêu cầu một câu trả lời
    "one_line": (
        "Trả lời câu hỏi lịch sử bằng kiến thức của bạn. "
        "Chỉ viết câu trả lời cuối cùng, không giải thích."
    ),
    # Có ngữ cảnh "bạn được phép không biết" + yêu cầu trả lời ngắn
    "short_with_unknown": (
        BASE
        + "\nTrả lời ngắn gọn (tối đa 2 câu). Nếu không biết chắc, viết: \"Tôi không biết\"."
    ),
}


def probe(client, name: str, system: str, question: str, budget: int) -> dict:
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=f"CÂU HỎI: {question}",
            config={
                "max_output_tokens": budget,
                "temperature": 0.0,
                "system_instruction": system,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return {"variant": name, "error": str(exc)[:200]}
    usage = getattr(resp, "usage_metadata", None)
    text = getattr(resp, "text", None)
    reasons = [
        str(getattr(cand, "finish_reason", "")) for cand in (getattr(resp, "candidates", None) or [])
    ]
    return {
        "variant": name,
        "budget": budget,
        "elapsed": round(time.perf_counter() - started, 1),
        "text_len": len(text or ""),
        "text": (text or "")[:220],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "finish_reasons": reasons,
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for name, system in VARIANTS.items():
        for question in (HARD, HARDER, HARDEST):
            row = probe(client, name, system, question, BUDGET)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
