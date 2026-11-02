#!/usr/bin/env python
"""Kiểm chứng: lần chấm lại của judge có **thật sự khác** lần đầu.

Với `temperature=0`, cùng một prompt luôn cho cùng một kết quả.  Nếu lần
chấm lại chỉ đổi cache key mà không đổi prompt, ba lần thử sẽ trả về ba
payload giống hệt nhau -- đúng những gì đã đo được trên `vhe_00001`.

Script này gọi judge với CONTEXT thật (dựng lại qua pipeline) và in ra payload
 của từng lần thử.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.ERROR)

QID = os.environ.get("PROBE_QID", "vhe_00001")
SYSTEM = os.environ.get("PROBE_SYSTEM", "ours")

from hgrag.config import get_settings  # noqa: E402
from hgrag.eval import judge as judge_mod  # noqa: E402
from hgrag.eval.dataset import EvalDataset  # noqa: E402
from hgrag.qa.pipeline import build_pipeline  # noqa: E402


def main() -> int:
    settings = get_settings()
    pipeline = build_pipeline(settings, mock=False)
    dataset = EvalDataset.load_jsonl(REPO_ROOT / "data/processed/eval_questions.jsonl")
    question = next(q for q in dataset if q.qid == QID)

    result = pipeline.ask(question.question, system=SYSTEM)
    evidence = list(result.evidence or [])

    print("CÂU HỎI:", question.question)
    print("ĐÁP ÁN:", result.answer)
    print("REFERENCE:", question.answer)
    print()

    # Gọi thẳng ``_ask`` để thấy từng lần thử.
    j = judge_mod.LLMJudge(pipeline.llm_client, max_retries=3, retry_base_delay=0.5)
    payload = j._ask(
        judge_mod.JUDGE_TEMPLATE.format(
            question=question.question,
            reference=question.answer or "(không có)",
            gold_facts="\n".join(f"- {f}" for f in question.gold_relations) or "(không có)",
            context=judge_mod._render_evidence(evidence) or "(không có context)",
            answer=result.answer or "(không có đáp án)",
        )
    )
    print("PAYLOAD CUỐI:", json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
