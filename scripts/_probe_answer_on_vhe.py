"""Feed the real VHE context to the answer generator and show the raw output."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.llm.client import build_llm_client
from hgrag.qa.answer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

CONTEXT = """[E1] (Nguồn: Wikipedia - "Hồ Chí Minh")
Ngày 23 tháng 9 năm 1945, quân Pháp tấn công Sài Gòn. Quân dân địa phương chống cự quyết liệt.

[E2] (Nguồn: Wikipedia)
Nam Hải -ABOUT-> Nhà Tần thành lập Nam Hải -ABOUT-> Tượng Quận -ABOUT-> gặp gỡ

[E3] (Nguồn: Wikipedia - "VHE dvsktt-s121")
Nhà Tần phát những người trốn tránh, người ở rể người đi buôn, ở các đạo ra làm binh, sai hiệu uý Đồ Thư đem quân lâu thuyền, sai Sử Lộc đào ngòi vận lương, đi sâu vào đất Lĩnh Nam, đánh lấy miền đất Lục Dương, đặt các quận Quế Lâm, Nam Hải (nay là tỉnh Quảng Đông) và Tượng Quận (tức là An Nam).

[E4] (Nguồn: Wikipedia)
Nam Hải -ABOUT-> Nhà Tần thành lập Nam Hải -ABOUT-> Tượng Quận -ABOUT-> thành lập"""

QUESTION = "Ai là người thành lập Nam Hải?"


def main() -> None:
    settings = get_settings()
    client = build_llm_client(settings)
    prompt = USER_PROMPT_TEMPLATE.format(
        context=CONTEXT, question=QUESTION, plan="\n(Gợi ý phân tích: loại câu hỏi = single_hop, số bước suy luận dự kiến = 1.)\n"
    )
    from hgrag.llm.schemas import LLMMessage, LLMRequest

    request = LLMRequest(
        messages=[
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ],
        task="answer_grounded",
        temperature=0.0,
    )
    response = client.generate(request)
    print("FINISH:", getattr(response, "finish_reason", None))
    print("THOUGHTS:", getattr(response, "thoughts_token_count", None))
    print("CANDIDATES:", getattr(response, "candidates_token_count", None))
    print("--- RAW ---")
    print(response.text)


if __name__ == "__main__":
    main()
