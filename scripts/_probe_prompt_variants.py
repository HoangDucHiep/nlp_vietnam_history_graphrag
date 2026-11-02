#!/usr/bin/env python
"""Tìm biến thể prompt closed-book mà Gemma 4 kết thúc được.

Đo đạc (`scripts/_probe_thinking.py`) cho thấy prompt closed-book hiện tại khiến
Gemma suy luận vô hạn: `finish_reason=MAX_TOKENS` ở mọi mức ngân sách, kể cả
8192 token.  Script này thử các biến thể (bỏ mục "## Suy luận", thêm giới hạn
độ dài, rút gọn system prompt) để tìm dạng prompt vừa giữ được ngữ nghĩa
"closed-book + chuỗi suy luận" vừa kết thúc trong ngân sách.
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")
QUESTION = "Ai là người sinh ra Đế Nghi?"
BUDGET = 2048

VARIANTS: dict[str, tuple[str, str]] = {
    # (system, user) -- giữ nguyên ngữ nghĩa closed-book, bỏ yêu cầu "## Suy luận"
    "no_suy_luan": (
        "Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC "
        "CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp.\n"
        "Trả lời trực tiếp, súc tích, bằng tiếng Việt, đúng mốc thời gian và địa danh. "
        "KHÔNG gắn trích dẫn [E#]. Nếu không chắc chắn, hãy nói rõ.",
        f"CÂU HỎI: {QUESTION}\n\nTrả lời ngắn gọn (tối đa 3 câu).",
    ),
    # Vẫn giữ chuỗi suy luận nhưng yêu cầu rõ ràng số bước và độ dài
    "suy_luan_ngắn": (
        "Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC "
        "CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp.\n"
        "Quy tắc:\n"
        "1. Trả lời trực tiếp, súc tích, bằng tiếng Việt.\n"
        "2. KHÔNG gắn trích dẫn [E#].\n"
        "3. Nếu không chắc chắn, hãy nói rõ rõ ràng.\n"
        "4. Cuối câu trả lời thêm đúng 2 bước suy luận đánh số (1., 2.), mỗi bước "
        "không quá 15 từ.",
        f"CÂU HỎI: {QUESTION}\n\nHãy trả lời theo quy tắc ở trên.",
    ),
    # Bản tối giản: ưu tiên kết thúc nhanh
    "minimal": (
        "Bạn là chuyên gia lịch sử Việt Nam, trả lời closed-book bằng kiến thức "
        "của bạn. Trả lời ngắn gọn, không suy luận dài.",
        f"CÂU HỎI: {QUESTION}",
    ),
    # Giống no_suy_luan nhưng ép trả lời dứt câu (chống lặp)
    "no_suy_luan_direct": (
        "Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC "
        "CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp.\n"
        "Trả lời trực tiếp, súc tích, bằng tiếng Việt. KHÔNG gắn trích dẫn [E#]. "
        "Nếu không chắc chắn, hãy nói rõ. Đừng hỏi lại người dùng, hãy đưa ra câu "
        "trả lời tốt nhất có thể.",
        f"CÂU HỎI: {QUESTION}\n\nTrả lời ngắn gọn (tối đa 3 câu).",
    ),
}


def probe(client, name: str, system: str, user: str, budget: int) -> dict:
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=user,
            config={
                "max_output_tokens": budget,
                "temperature": 0.0,
                "system_instruction": system,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return {"variant": name, "budget": budget, "error": str(exc)[:200]}
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
        "text": (text or "")[:400],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "finish_reasons": reasons,
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for name, (system, user) in VARIANTS.items():
        row = probe(client, name, system, user, BUDGET)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
