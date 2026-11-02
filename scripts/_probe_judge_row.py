#!/usr/bin/env python
"""Ghi lại payload thô của LLM judge cho một câu hỏi, kèm CONTEXT thật.

Mục đích: kiểm chứng xem một dòng bị đánh dấu `judge_degraded` là do
(a) judge thật sự trả về payload hỏng, hay (b) bộ lọc
`_is_self_consistent` loại oan một payload hợp lệ, khiến dòng rơi về
heuristic token-F1 và bị chấm sai.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.WARNING)

from hgrag.config import get_settings  # noqa: E402
from hgrag.eval import judge as judge_mod  # noqa: E402

QID = os.environ.get("PROBE_QID", "vhe_00001")
SYSTEM = os.environ.get("PROBE_SYSTEM", "ours")

_orig_ask = judge_mod.LLMJudge._ask


def _spy_ask(self, prompt: str):  # type: ignore[no-untyped-def]
    payload = _orig_ask(self, prompt)
    print("=" * 70)
    print("PROMPT:\n" + prompt[:1500])
    print("-" * 70)
    print("RAW PAYLOAD:", json.dumps(payload, ensure_ascii=False))
    if isinstance(payload, dict):
        print("self_consistent:", judge_mod._is_self_consistent(payload))
    return payload


judge_mod.LLMJudge._ask = _spy_ask  # type: ignore[method-assign]

# Cũng chặn ở tầng thấp hơn để thấy payload thô *trước* khi bị bộ lọc loại.
_orig_consistent = judge_mod._is_self_consistent


def _spy_consistent(payload):  # type: ignore[no-untyped-def]
    ok = _orig_consistent(payload)
    if not ok:
        print("  >> REJECTED payload:", json.dumps(payload, ensure_ascii=False))
    return ok


judge_mod._is_self_consistent = _spy_consistent  # type: ignore[assignment]

from hgrag.eval.dataset import EvalDataset  # noqa: E402
from hgrag.qa.pipeline import build_pipeline  # noqa: E402


def main() -> int:
    settings = get_settings()
    pipeline = build_pipeline(settings, mock=False)
    dataset = EvalDataset.load_jsonl(REPO_ROOT / "data/processed/eval_questions.jsonl")
    question = next(q for q in dataset if q.qid == QID)

    j = judge_mod.build_judge(pipeline.llm_client, mock=False)
    if SYSTEM == "B0":
        answer = pipeline.answer_generator.generate_closed_book(question.question)
        answer_text = answer.answer
        evidence = []
    else:
        result = pipeline.ask(question.question, system=SYSTEM)
        answer_text = result.answer
        evidence = list(result.evidence or [])

    verdict = j.grade(
        question=question.question,
        answer=answer_text,
        reference=question.answer,
        gold_facts=list(question.gold_relations),
        evidence=evidence,
        out_of_kb=judge_mod.is_out_of_kb(question),
        closed_book=(SYSTEM == "B0"),
    )
    print("=" * 70)
    print("VERDICT:", json.dumps(verdict.to_dict(), ensure_ascii=False))
    print("degraded:", verdict.degraded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
