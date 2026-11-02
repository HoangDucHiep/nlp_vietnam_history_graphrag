#!/usr/bin/env python
"""Đo đạc: JSON mode của Gemma 4 -- chất lượng `answer` và `reasoning_steps`.

Câu hỏi: có thể dùng JSON mode (response_schema) cho câu trả lời closed-book
mà không làm hỏng chất lượng không?  Vì JSON mode tắt thinking, nên Gemma 4
luôn kết thúc được.
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")

QUESTIONS = [
    "Ai là người sinh ra Đế Nghi?",          # ngoài kiến thức tham số -> phải từ chối
    "Ai là người thành lập Văn Lang?",       # trong kiến thức tham số
    "Năm 1945 Việt Nam giành được độc lập sau sự kiện nào?",
]

SYSTEM_V1 = """Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp.

Quy tắc bắt buộc:
1. Trả lời trực tiếp, súc tích, bằng tiếng Việt, đúng mốc thời gian và địa danh.
2. KHÔNG gắn trích dẫn [E#] vì không có nguồn nào được cung cấp.
3. Nếu bạn KHÔNG CHẮC CHẮN hoặc không biết, hãy trả lời đúng câu: "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này." Đây là một câu trả lời HỢP LỆ, không phải lỗi.
4. KHÔNG suy luận dài, KHÔNG tự hỏi lại người dùng. Chỉ viết câu trả lời cuối cùng.
5. Kết thúc bằng một mục "## Suy luận" gồm các bước suy luận đánh số (1., 2., 3.) ngắn gọn.
"""

SYSTEM_V2 = """Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp.

Quy tắc bắt buộc:
1. `answer`: trả lời trực tiếp, súc tích, bằng tiếng Việt, đúng mốc thời gian và địa danh. Tối đa 4 câu.
2. KHÔNG gắn trích dẫn [E#] vì không có nguồn nào được cung cấp.
3. Nếu bạn KHÔNG CHẮC CHẮN hoặc không biết, `answer` phải đúng câu: "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này." Đây là một câu trả lời HỢP LỆ, không phải lỗi.
4. `reasoning_steps`: đúng 3 bước, mỗi bước là MỘT CÂU TIẾNG VIỆT ngắn (tối đa 20 từ), không sao chép văn bản, không viết tiếng Anh, không giải thích quy trình nội bộ.
5. KHÔNG tự hỏi lại người dùng.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "reasoning_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "reasoning_steps"],
}


def probe(client, system: str, question: str) -> dict:
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=f"CÂU HỎI: {question}\n\nHãy trả lời theo quy tắc ở trên.",
            config={
                "temperature": 0.0,
                "max_output_tokens": 2048,
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_schema": SCHEMA,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:300]}
    usage = getattr(resp, "usage_metadata", None)
    text = getattr(resp, "text", None) or ""
    return {
        "elapsed": round(time.perf_counter() - started, 1),
        "text_len": len(text),
        "text": text[:800],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for question in QUESTIONS:
        for label, system in (("v1", SYSTEM_V1), ("v2", SYSTEM_V2)):
            row = probe(client, system, question)
            print(json.dumps({"system": label, "q": question, **row}, ensure_ascii=False), flush=True)
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
