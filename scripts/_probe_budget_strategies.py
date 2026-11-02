#!/usr/bin/env python
"""Đo đạc: prompt closed-book THẬT + các chiến lược ngân sách thinking.

Câu hỏi: Gemma 4 có kết thúc được không khi câu hỏi nằm ngoài kiến thức
tham số của nó, và nếu không thì có cách nào ép nó dừng lại?

Các biến thể:
  * ``as_is``          -- prompt thật, ngân sách 4096 (cấu hình hiện tại);
  * ``budget_8192``    -- prompt thật, ngân sách 8192;
  * ``thinking_0``     -- tắt thinking (``thinking_config.thinking_budget=0``);
  * ``thinking_512``   -- giới hạn thinking còn 512 token;
  * ``schema_json``    -- JSON mode (response_schema) -> API tắt thinking.
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")
QUESTION = os.environ.get("PROBE_QUESTION", "Ai là người sinh ra Đế Nghi?")

SYSTEM = """Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp.

Quy tắc bắt buộc:
1. Trả lời trực tiếp, súc tích, bằng tiếng Việt, đúng mốc thời gian và địa danh.
2. KHÔNG gắn trích dẫn [E#] vì không có nguồn nào được cung cấp.
3. Nếu bạn KHÔNG CHẮC CHẮN hoặc không biết, hãy trả lời đúng câu: "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này." Đây là một câu trả lời HỢP LỆ, không phải lỗi.
4. KHÔNG suy luận dài, KHÔNG tự hỏi lại người dùng. Chỉ viết câu trả lời cuối cùng.
5. Kết thúc bằng một mục "## Suy luận" gồm các bước suy luận đánh số (1., 2., 3.) ngắn gọn.
"""

USER = f"CÂU HỎI: {QUESTION}\n\nHãy trả lời theo quy tắc ở trên."


def probe(client, name: str, config: dict, contents: str = USER) -> dict:
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config={**config, "system_instruction": SYSTEM},
        )
    except Exception as exc:  # noqa: BLE001
        return {"variant": name, "error": str(exc)[:300]}
    usage = getattr(resp, "usage_metadata", None)
    text = getattr(resp, "text", None)
    reasons = [
        str(getattr(cand, "finish_reason", ""))
        for cand in (getattr(resp, "candidates", None) or [])
    ]
    return {
        "variant": name,
        "elapsed": round(time.perf_counter() - started, 1),
        "text_len": len(text or ""),
        "text": (text or "")[:500],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "finish_reasons": reasons,
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    base = {"temperature": 0.0}
    runs = [
        ("as_is_4096", {**base, "max_output_tokens": 4096}),
        ("thinking_0_2048", {**base, "max_output_tokens": 2048, "thinking_config": {"thinking_budget": 0}}),
        ("thinking_512_2048", {**base, "max_output_tokens": 2048, "thinking_config": {"thinking_budget": 512}}),
        ("include_thoughts_false_2048", {**base, "max_output_tokens": 2048, "include_thoughts": False}),
    ]
    for name, config in runs:
        row = probe(client, name, config)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
