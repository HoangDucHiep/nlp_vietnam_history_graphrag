#!/usr/bin/env python
"""Kiểm chứng bản sửa "thinking ăn hết ngân sách output" trên prompt thật.

Chạy pipeline thật (Neo4j + Qdrant + Gemini) trên vài câu của bộ đánh giá
và báo cáo:
* có câu trả lời hay không (trước đây: câu "từ chối có sẵn");
* verdict của judge là LLM hay heuristic (``judge_degraded``);
* ngân sách token thực tế đã dùng.

Dùng ``--systems`` để chọn hệ thống (mặc định ``B0,ours``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.config import get_settings  # noqa: E402
from hgrag.eval.judge import build_judge  # noqa: E402
from hgrag.eval.run_eval import _answer_system  # noqa: E402
from hgrag.llm.client import build_llm_client  # noqa: E402
from hgrag.qa.pipeline import build_pipeline  # noqa: E402

CANNED = "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm chứng bản sửa ngân sách output")
    parser.add_argument("--questions", default="data/processed/eval_questions.jsonl")
    parser.add_argument("--systems", nargs="+", default=["B0", "ours"])
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--out", default="artifacts/validate_thinking_fix.json")
    args = parser.parse_args(argv)

    settings = get_settings()
    llm = build_llm_client(settings)
    pipeline = build_pipeline(settings)
    judge = build_judge(llm)

    questions: list[dict] = []
    with open(REPO_ROOT / args.questions, encoding="utf-8") as fh:
        for line in fh:
            questions.append(json.loads(line))

    # Chọn câu hỏi đại diện cho từng nhóm: câu hỏi VHE (thực thể hiếm, suy
    # luận khó) và câu hỏi ngoài phạm vi.
    picked: list[dict] = []
    seen_types: set[str] = set()
    for row in questions:
        qtype = row.get("question_type", "")
        if qtype in seen_types:
            continue
        seen_types.add(qtype)
        picked.append(row)
        if len(picked) >= args.limit:
            break

    rows: list[dict] = []
    for system in args.systems:
        for question in picked:
            answer = _answer_system(pipeline, question["question"], system, {})
            text = (answer.answer or "").strip()
            verdict = judge.grade(
                question=question["question"],
                answer=text,
                reference=question.get("answer", ""),
                gold_facts=question.get("gold_relations") or [],
                evidence=getattr(answer, "evidence", []) or [],
                out_of_kb=question.get("question_type") == "out_of_kb",
                closed_book=(system == "B0"),
            )
            rows.append(
                {
                    "system": system,
                    "qid": question.get("qid"),
                    "question_type": question.get("question_type"),
                    "question": question["question"],
                    "answer_len": len(text),
                    "is_canned_refusal": text == CANNED,
                    "answer_head": text[:220],
                    "judge_score": verdict.score,
                    "judge_source": verdict.source,
                    "judge_degraded": verdict.degraded,
                    "judge_reason": verdict.reason[:200],
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    n_empty = sum(1 for r in rows if r["answer_len"] == 0)
    n_canned = sum(1 for r in rows if r["is_canned_refusal"])
    n_degraded = sum(1 for r in rows if r["judge_degraded"])
    summary = {
        "n_rows": len(rows),
        "n_empty_answers": n_empty,
        "n_canned_refusals": n_canned,
        "n_judge_degraded": n_degraded,
        "rows": rows,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    return 0 if n_empty == 0 and n_degraded == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
