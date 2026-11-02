#!/usr/bin/env python
"""Tìm cách để LLM judge trả về verdict *tự nhất quán*.

Hiện trạng: với CONTEXT thật, Gemma trả `score=0` kèm lý do khen
("Đáp án đúng hoàn toàn so với đáp án tham chiếu...").  Vì `temperature=0`
nên các lần chấm lại cho kết quả y hệt, và dòng rơi về heuristic token-F1 --
một heuristic chặt hơn rubric, khiến đáp án đúng bị báo cáo là sai.

Script này thử nhiều cách hỏi khác nhau trên *cùng* một câu hỏi thật (có
CONTEXT thật) và báo cáo verdict của từng cách.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hgrag.eval.judge import (  # noqa: E402
    JUDGE_SCHEMA,
    JUDGE_SYSTEM,
    _is_self_consistent,
)

QID = os.environ.get("PROBE_QID", "vhe_00001")
SYSTEM = os.environ.get("PROBE_SYSTEM", "ours")
MODEL = os.environ.get("PROBE_MODEL", "gemma-4-26b-a4b-it")

#: Prompt gốc (như trong judge.py).
BASE_SYSTEM = JUDGE_SYSTEM

#: Biến thể: nhấn mạnh rằng `score` là điểm số, và yêu cầu tự kiểm tra.
SCORE_FIRST_SYSTEM = """Bạn là giám khảo khách quan cho bài đánh giá hệ thống hỏi đáp lịch sử Việt Nam.

Bạn được cho: CÂU HỎI, ĐÁP ÁN THAM CHIẾU, GOLD FACTS, CÁC ĐÁP ÁN ỨNG VIÊN (có kèm trích dẫn [E#]) và CONTEXT.

Quy tắc chấm điểm (score):
- 2 = đáp án đúng VỀ NGHĨA và đầy đủ so với đáp án tham chiếu / gold facts.
- 1 = đáp án đúng một phần (thiếu chi tiết quan trọng, hoặc sai một phần nhỏ).
- 0 = đáp án sai, hoặc không liên quan, hoặc bịa đặt (hallucinate) thông tin không có trong CONTEXT.

Quy tắc grounded:
- grounded = true nếu mọi mệnh đề khẳng định trong đáp án đều được CONTEXT hỗ trợ.
- Nếu đáp án có thông tin không có trong CONTEXT thì grounded = false.

Quy tắc supported_evidence_ids:
- Liệt kê các mã evidence (ví dụ "E1") thực sự chứng minh mệnh đề trong đáp án.

Quy tắc về độ dài:
- Trường "reason" chỉ ghi TỐI ĐA 2 câu ngắn (khoảng 40 từ). Không lặp lại câu hỏi, không trích lại CONTEXT.

QUAN TRỌNG -- kiểm tra trước khi trả lời:
1. Chấm `score` TRƯỚC, rồi viết `reason` khớp với điểm đã chấm:
   - score=2 -> reason phải nói đáp án đúng và đầy đủ;
   - score=1 -> reason phải nói đáp án đúng một phần;
   - score=0 -> reason phải nói đáp án sai / không đáp ứng / thiếu căn cứ.
2. TUYỆT ĐỐI không viết một lý do khen đáp án khi điểm là 0, và không viết
   một lý do chê đáp án khi điểm là 2.
3. `correct` phải đúng bằng (score >= 2).

Trả về DUY NHẤT một JSON object hợp lệ, không markdown fence.
"""


def load_case() -> tuple[dict | None, str]:
    """Dòng trong checkpoint + CONTEXT thật (dựng lại qua pipeline)."""
    row = None
    for path in (REPO_ROOT / "artifacts").glob("eval_ckpt_*.json"):
        data = json.loads(path.read_text())
        for candidate in data.get("rows", []):
            if candidate["qid"] == QID and candidate["system"] == SYSTEM:
                row = candidate
                break
        if row is not None:
            break
    if row is None:
        return None, ""

    # Dựng lại CONTEXT từ evidence thật (checkpoint không lưu evidence).
    import logging

    logging.basicConfig(level=logging.ERROR)
    from hgrag.config import get_settings
    from hgrag.eval.dataset import EvalDataset
    from hgrag.qa.pipeline import build_pipeline

    settings = get_settings()
    pipeline = build_pipeline(settings, mock=False)
    dataset = EvalDataset.load_jsonl(REPO_ROOT / "data/processed/eval_questions.jsonl")
    question = next(q for q in dataset if q.qid == QID)
    result = pipeline.ask(question.question, system=SYSTEM)

    from hgrag.eval.judge import _render_evidence

    evidence = list(result.evidence or [])
    context = _render_evidence(evidence)
    row = dict(row)
    row["_context"] = context
    row["_gold_facts"] = list(question.gold_relations)
    return row, context


def main() -> int:
    from google import genai

    row, context = load_case()
    if row is None:
        print("không tìm thấy dòng")
        return 1

    from hgrag.eval.judge import _render_evidence

    gold = row.get("_gold_facts") or []
    if isinstance(gold, str):
        gold = [gold]

    template = """CÂU HỎI:
{question}

ĐÁP ÁN THAM CHIẾU:
{reference}

GOLD FACTS:
{gold_facts}

CONTEXT (evidence được trích dẫn):
{context}

ĐÁP ÁN ỨNG VIÊN:
{answer}

Hãy chấm điểm theo rubric trên.
"""

    prompt = template.format(
        question=row["question"],
        reference=row["reference"] or "(không có đáp án tham chiếu)",
        gold_facts="\n".join(f"- {f}" for f in gold) or "(không có)",
        context=context,
        answer=row["answer"] or "(không có đáp án)",
    )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    variants = [
        ("base", BASE_SYSTEM),
        ("score_first", SCORE_FIRST_SYSTEM),
    ]
    for name, system in variants:
        started = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                    "response_mime_type": "application/json",
                    "response_schema": JUDGE_SCHEMA,
                    "system_instruction": system,
                },
            )
            raw = resp.text or ""
            payload = json.loads(raw)
            ok = _is_self_consistent(payload)
            print(
                json.dumps(
                    {
                        "variant": name,
                        "elapsed": round(time.perf_counter() - started, 1),
                        "payload": payload,
                        "self_consistent": ok,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"variant": name, "error": str(exc)[:200]}), flush=True)
        time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
