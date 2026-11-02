#!/usr/bin/env python
"""Đo cách Gemma 4 phân bổ ngân sách output (thinking vs. câu trả lời).

Gemma 4 tính **token suy luận (thinking)** vào `max_output_tokens`.  Nếu ngân
sách quá nhỏ, toàn bộ ngân sách bị thinking ăn hết và phần trả lời trả về
rỗng (`finish_reason=MAX_TOKENS`, `text=None`).  Đây chính là nguyên nhân
khiến ~59 % câu trả lời closed-book của lượt chạy trước rơi về câu "từ chối
có sẵn".

Script đo với các prompt thật của hệ thống (closed-book, grounded, judge) ở
nhiều mức ngân sách và báo cáo:

* ``candidates_token_count`` -- token câu trả lời thật
* ``thoughts_token_count``    -- token suy luận
* ``finish_reason``           -- ``STOP`` / ``MAX_TOKENS``
* độ dài câu trả lời
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")

CONTEXT = """[E1] Xưa cháu ba đời của Viêm Đế họ Thần Nông là Đế Minh sinh ra Đế Nghi, sau Đế Minh nhân đi tuần phương Nam, đến Ngũ Lĩnh lấy con gái Vụ Tiên, sinh ra vua Kinh Dương Vương.
[E2] Kinh Dương Vương có vợ là Hoàng Cung, sinh ra Đế Nghi."""

PROMPTS: dict[str, str] = {
    "closed_book": (
        "Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC "
        "CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp.\n\n"
        "Quy tắc bắt buộc:\n"
        "1. Trả lời trực tiếp, súc tích, bằng tiếng Việt.\n"
        "2. KHÔNG gắn trích dẫn [E#].\n"
        "3. Nếu không chắc chắn, hãy nói rõ.\n"
        "4. Kết thúc bằng mục \"## Suy luận\" gồm các bước đánh số ngắn gọn.\n\n"
        "CÂU HỎI: Ai là người sinh ra Đế Nghi?\n\n"
        "Hãy trả lời theo quy tắc ở trên."
    ),
    "grounded": (
        "Bạn là trợ lý trả lời câu hỏi dựa trên CONTEXT.\n\n"
        "CONTEXT:\n" + CONTEXT + "\n\n"
        "CÂU HỎI: Ai là người sinh ra Đế Nghi?\n\n"
        "Trả lời, trích dẫn [E#] và kết thúc bằng mục \"## Suy luận\" ngắn gọn."
    ),
    "judge": (
        "Bạn là giám khảo khách quan. Chấm điểm 0-2 và trả về DUY NHẤT một JSON hợp lệ.\n\n"
        "CÂU HỎI: Ai là người sinh ra Đế Nghi?\n"
        "ĐÁP ÁN THAM CHIẾU: Đế Minh\n"
        "GOLD FACTS:\n- Đế Minh -PARTICIPATED_IN-> Đế Minh sinh ra Đế Nghi\n"
        "CONTEXT:\n" + CONTEXT + "\n"
        "ĐÁP ÁN ỨNG VIÊN: Đế Minh là người sinh ra Đế Nghi [E1].\n\n"
        'JSON: {"score": 2, "correct": true, "grounded": true, '
        '"supported_evidence_ids": ["E1"], "reason": "đúng"}'
    ),
}

BUDGETS = (256, 1024, 2048, 4096, 8192)


def probe(client, prompt_key: str, budget: int) -> dict:
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=PROMPTS[prompt_key],
            config={"max_output_tokens": budget, "temperature": 0.0},
        )
    except Exception as exc:  # noqa: BLE001
        return {"prompt": prompt_key, "budget": budget, "error": str(exc)[:200]}
    usage = getattr(resp, "usage_metadata", None)
    text = getattr(resp, "text", None)
    reasons = [
        str(getattr(cand, "finish_reason", "")) for cand in (getattr(resp, "candidates", None) or [])
    ]
    return {
        "prompt": prompt_key,
        "budget": budget,
        "elapsed": round(time.perf_counter() - started, 1),
        "text_len": len(text or ""),
        "text_head": (text or "")[:90],
        "candidates_tokens": getattr(usage, "candidates_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        "finish_reasons": reasons,
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for prompt_key in ("closed_book", "grounded", "judge"):
        for budget in BUDGETS:
            row = probe(client, prompt_key, budget)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
