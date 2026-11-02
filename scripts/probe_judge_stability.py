#!/usr/bin/env python
"""Đo độ ổn định của judge trên dữ liệu thật của pipeline.

Một dòng trong lượt chạy trước có `score=0` nhưng lý do lại viết "Đáp án đúng
về nghĩa...".  Script này gọi `LLMJudge.grade` lặp lại trên *cùng* một câu
hỏi với *đúng* evidence mà pipeline trả về, để đo:
* phân bố điểm;
* tần suất payload tự mâu thuẫn (lý do nói "đúng" nhưng điểm = 0);
* tần suất lý do bị lặp (degenerate repetition).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hgrag.config import get_settings  # noqa: E402
from hgrag.eval.judge import build_judge  # noqa: E402
from hgrag.eval.run_eval import _answer_system  # noqa: E402
from hgrag.llm.client import build_llm_client  # noqa: E402
from hgrag.qa.pipeline import build_pipeline  # noqa: E402

_SENT = re.compile(r"[^.!?]+[.!?]?")


def is_repetitive(reason: str) -> bool:
    """Whether the reason repeats the same sentence (degenerate output)."""
    sentences = [s.strip().lower() for s in _SENT.findall(reason or "") if len(s.strip()) > 12]
    if not sentences:
        return False
    counts = Counter(sentences)
    return counts.most_common(1)[0][1] >= 2


def contradicts(reason: str, score: float) -> bool:
    """Whether the reason praises the answer while the score is 0."""
    lowered = (reason or "").lower()
    praise = any(p in lowered for p in ("đúng về nghĩa", "đúng hoàn toàn", "chính xác và đầy đủ", "đúng và đầy đủ"))
    return praise and score < 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đo độ ổn định của judge")
    parser.add_argument("--qid", default="vhe_00044")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--system", default="ours")
    parser.add_argument("--out", default="artifacts/judge_consistency.json")
    args = parser.parse_args(argv)

    settings = get_settings()
    llm = build_llm_client(settings)
    pipeline = build_pipeline(settings)
    judge = build_judge(llm)

    questions = {}
    with open(REPO_ROOT / "data/processed/eval_questions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            questions[row["qid"]] = row

    question = questions[args.qid]
    answer = _answer_system(pipeline, question["question"], args.system, {})
    text = (answer.answer or "").strip()

    rows = []
    for i in range(args.repeats):
        verdict = judge.grade(
            question=question["question"],
            answer=text,
            reference=question.get("answer", ""),
            gold_facts=question.get("gold_relations") or [],
            evidence=getattr(answer, "evidence", []) or [],
            out_of_kb=question.get("question_type") == "out_of_kb",
            closed_book=(args.system == "B0"),
        )
        row = {
            "run": i,
            "score": verdict.score,
            "correct": verdict.correct,
            "grounded": verdict.grounded,
            "reason": verdict.reason,
            "repetitive": is_repetitive(verdict.reason),
            "contradictory": contradicts(verdict.reason, verdict.score),
            "source": verdict.source,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "qid": args.qid,
        "system": args.system,
        "scores": [r["score"] for r in rows],
        "n_repetitive": sum(1 for r in rows if r["repetitive"]),
        "n_contradictory": sum(1 for r in rows if r["contradictory"]),
        "rows": rows,
    }
    out = REPO_ROOT / args.out
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
