#!/usr/bin/env python
"""Tại sao judge trả `score=0` khi lý do lại nói "đáp án đúng"?

Khác biệt giữa prompt tổng hợp (judge trả score=2) và prompt thật
(judge trả score=0) là **độ dài CONTEXT**: prompt thật có 11 evidence, trong
đó chỉ [E1] liên quan.

Script này dựng lại đúng prompt thật (từ cache của lượt chạy trước) và thử:

1. schema vs. không schema;
2. CONTEXT đầy đủ (11 evidence) vs. chỉ [E1] vs. [E1]+[E2];
3. lặp lại để đo tần suất.

Nếu `response_schema` là thủ phạm thì bỏ ép schema sẽ cho điểm đúng.
"""

from __future__ import annotations

import json
import os
import sys
import time

from google import genai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hgrag.eval.judge import JUDGE_SCHEMA, JUDGE_SYSTEM, JUDGE_TEMPLATE  # noqa: E402

MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")

QUESTION = "Sự kiện sinh ra Đế Nghi và sự kiện sinh ra Kinh Dương Vương cùng do nhân vật nào thực hiện?"
REFERENCE = "Đế Minh"
GOLD_FACTS = (
    "- sinh ra Đế Nghi -INVOLVES-> Đế Minh\n"
    "- sinh ra Kinh Dương Vương -INVOLVES-> Đế Minh"
)
ANSWER = "Cả hai sự kiện sinh ra Đế Nghi và sinh ra Kinh Dương Vương đều do Đế Minh thực hiện [E1]."

E1 = (
    "[E1] Xưa cháu ba đời của Viêm Đế họ Thần Nông là Đế Minh sinh ra Đế Nghi, "
    "sau Đế Minh nhân đi tuần phương Nam, đến Ngũ Lĩnh lấy con gái Vụ Tiên, "
    "sinh ra vua [Kinh Dương Vương]."
)
E2 = "[E2] Kinh Dương Vương -PARTICIPATED_IN-> xưng (2879)"
E3 = (
    "[E3] nhỏ|250px|phải|Rặng duối cổ ở thôn Cam Lâm thuộc Đường Lâm. Tương truyền "
    "Ngô Quyền đã buộc voi chiến và ngựa chiến ở đây. * Thiên Sách Vương Ngô Xương Ngập: "
    "Là con trai trưởng của Ngô Quyền, được phỏng đoán sinh ra vào khoảng thập niên thứ hai "
    "của thế kỉ 10. Tiền Ngô Vương truyền ngôi cho Ngô Xương Ngập nhưng bị Dương Tam Kha cướp "
    "ngôi, Xương Ngập phải bỏ trốn. Năm 950, Dương Tam Kha bị lật đổ, ông được em là Nam Tấn Vương "
    "Ngô Xương Văn đón về, hai anh em cùng làm vua. Năm 954 ông mất. * Nam Tấn Vương Ngô Xương Văn: "
    "Là con trai thứ hai của Ngô Quyền, mẹ là Dương hậu. Ông đã làm đảo chính, phế"
)
E4 = (
    "[E4] [Lý] Thái Tổ nhận mà không từ, đó là muốn khoe khoang để cho đời sau không ai hơn được. "
    "Thế là sai. Sau, Tôn Thái Tông cũng đến gần 50 chữ là vì bắt chước ở đây.\" Lý Công Uẩn lên "
    "ngôi hoàng đế ngày 21 tháng 11 năm 1009, chính thức lập ra nhà Lý. Sử còn gọi triều đại của ông "
    "là Hậu Lý để phân biệt với nhà Tiền Lý do Lý Nam Đế sáng lập. Vua đặt niên hiệu là Thuận Thiên, "
    "nghĩa là \"theo ý trời\". Ông phong chú là Vũ Đạo vương, anh ruột là Vũ Uy vương, em ruột là "
    "Dực Thánh vương. Ông lập 6 hoàng hậu, con trưởng Lý Phật Mã được lập làm Thái"
)
E5 = (
    "[E5] Trước khi Long Việt bị ám sát và Long Đĩnh lên ngôi, ở chùa Ứng Thiên Tâm, châu Cổ Pháp "
    "có con chó sinh con lông trắng với đốm lông đen thành hình hai chữ \"Thiên tử\". Kẻ thức giả "
    "khi ấy cho đó là điềm có người sinh năm Tuất làm thiên tử. Lý Công Uẩn cũng sinh vào năm Tuất "
    "974. Trong 5 năm trị vì (1005 - 1009), Lê Long Đĩnh thực hiện nhiều cuộc đánh diệt loạn đảo "
    "trong nước, đồng thời đối mặt với những áp lực ngoại giao và hiểm họa xâm lăng từ nhà Tống. "
    "Sử không chép gì về hoạt động của Lý Công Uẩn vào thời điểm ấy. Các cuộc đánh diệt loạn đảo của "
    "Long Đĩnh đều được sử chép rõi"
)
E6 = (
    "[E6] Điều đó cũng cho thấy lịch sử không chỉ là những con số cố định, mà còn là hành trình khám "
    "phát và tái hiện lại quá khứ qua nhiều lớp tư liệu khác nhau. Dương Tam Kha cướp ngôi cháu, tự "
    "lập mình làm vua, xưng là Dương Bình Vương, lấy Ngô Xương Văn, con trai thứ hai của Ngô Quyền "
    "làm con nuôi. Ngô Xương Văn sau dẫn quân quay lại lật đổ Dương Tam Kha. Năm 950, Ngô Xương Văn "
    "tự xưng làm Thiên Sách Vương, đóng đô ở Cổ Loa. Ngô Xương Văn cho người đón anh trai Ngô Tất "
    "Thế đang trốn ở Nam Sách trở về. Ngô Tất Thế lên ngôi làm vua, tự xưng là Thiên Sách Vương "
    "(951–954). Từ năm 951 đến 956, lần lượt các thế"
)
E7 = (
    "[E7] May sao thời mở vận trung hưng, sinh ra tiên tổ họ Trịnh là Thế Tổ Thái Vương, Chúa "
    "Tông Triết Vương dấy lên từ người áo vải, xướng xuất nghĩa quân việc diệt tàn bạo cho thiên "
    "hạ, giữ yên xã tắc, tôn lập Hoàng tổ ta là Trang Tông Dụ Hoàng Đế, Trung Tông Vũ Hoàng Đế."
)
E8 = (
    "[E8] Mùa xuân, tháng 3, vua băng ở điện Trường Xuân, gọi là Đại Hành Hoàng Đế, sau nhân đó "
    "dùng làm miếu hiệu mà không đổi, chôn ở sơn lăng châu Trường Yên (sách Địa Chí bản cũ chép "
    "vua băng ở năm Bính Ngọ [1006], đó là lấy khi Lê Ngoạ Triều xin miếu (nhà Tống) mà nói, không "
    "phải vậy. Nay theo Lê Văn Hưu là đúng)."
)
E9 = (
    "[E9] Vua ra trường đấu xem bọn đại thần Lê Sát và các quan văn võ trong ấu có tế cáo trời, đất, "
    "thần kỳ danh sơn, giết ngựa trắng lấy máu cùng thề."
)
E10 = (
    "[E10] Sau Tồn Phúc giết Tồn Lộc và Đương Đạo, chiếm luôn cả đất, tiếm xưng là Chiêu Thánh Hoàng "
    "Đế, lập A Nùng làm Minh Đức Hoàng Hậu, phong cho con là Trí Thông làm Nam Nha Vương, đổi châu "
    "ấy gọi là nước Trường Sinh, sửa soạn binh giáp, đắp thành kính cổ để tự giữ, không nộp cống "
    "xưng thần nữa."
)
E11 = (
    "[E11] Bấy giờ vua mới 7 tuổi, tôn mẹ đẻ là Lan nguyên phi làm Hoàng thái phi, tôn mẹ đích là "
    "Thượng Dương thái hậu họ Dương làm Hoàng thái hậu, buông rem cùng thề."
)

FULL_CONTEXT = "\n\n".join([E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11])

CASES = [
    ("full-schema", FULL_CONTEXT, True, 512),
    ("full-schema", FULL_CONTEXT, True, 512),
    ("full-noschema", FULL_CONTEXT, False, 1024),
    ("full-noschema", FULL_CONTEXT, False, 1024),
    ("e1-only-schema", E1, True, 512),
    ("e1+e2-schema", E1 + "\n\n" + E2, True, 512),
    ("full-schema-2048", FULL_CONTEXT, True, 2048),
]


def run(client, context: str, schema: bool, budget: int) -> dict:
    prompt = JUDGE_TEMPLATE.format(
        question=QUESTION,
        reference=REFERENCE,
        gold_facts=GOLD_FACTS,
        context=context,
        answer=ANSWER,
    )
    config: dict = {"max_output_tokens": budget, "temperature": 0.0}
    if schema:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = JUDGE_SCHEMA
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={**config, "system_instruction": JUDGE_SYSTEM},
    )
    text = getattr(resp, "text", "") or ""
    usage = getattr(resp, "usage_metadata", None)
    try:
        payload = json.loads(text)
    except ValueError:
        payload = {}
    return {
        "budget": budget,
        "schema": schema,
        "n_evidence_blocks": context.count("[E"),
        "score": payload.get("score"),
        "correct": payload.get("correct"),
        "reason": payload.get("reason"),
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
    }


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for label, context, schema, budget in CASES:
        row = run(client, context, schema, budget)
        row["label"] = label
        print(json.dumps(row, ensure_ascii=False), flush=True)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
