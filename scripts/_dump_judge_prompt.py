#!/usr/bin/env python
"""In ra đúng prompt mà judge nhận, để so sánh với prompt tổng hợp.

Lỗi `score=0` + lý do "đáp án đúng" tái hiện 5/5 lần với evidence thật của
pipeline, nhưng *không* tái hiện với prompt tổng hợp.  Script này dựng lại
prompt đó (cùng hàm `_ask` của judge) để tìm ra khác biệt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hgrag.config import get_settings  # noqa: E402
from hgrag.eval.judge import (  # noqa: E402
    JUDGE_TEMPLATE,
    _render_evidence,
)
from hgrag.eval.run_eval import _answer_system  # noqa: E402
from hgrag.qa.pipeline import build_pipeline  # noqa: E402

QID = "vhe_00044"


def main() -> int:
    settings = get_settings()
    pipeline = build_pipeline(settings)
    questions = {}
    with open(REPO_ROOT / "data/processed/eval_questions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            questions[row["qid"]] = row

    question = questions[QID]
    answer = _answer_system(pipeline, question["question"], "ours", {})
    evidence = list(getattr(answer, "evidence", []) or [])

    prompt = JUDGE_TEMPLATE.format(
        question=question["question"],
        reference=question.get("answer", ""),
        gold_facts="\n".join(f"- {r}" for r in (question.get("gold_relations") or [])),
        context=_render_evidence(evidence),
        answer=(answer.answer or "").strip(),
    )
    print("=" * 78)
    print(prompt)
    print("=" * 78)
    print(f"n_evidence = {len(evidence)}")
    print(f"answer = {(answer.answer or '').strip()!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
