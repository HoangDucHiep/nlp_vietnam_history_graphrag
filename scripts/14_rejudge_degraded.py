#!/usr/bin/env python
"""Chấm lại các dòng bị *degraded* bằng judge đã sửa.

Khi bộ lọc `_is_self_consistent` loại oan verdict của LLM judge, dòng đó rơi
về heuristic token-F1 -- một heuristic *chặt hơn* rubric, nên đáp án đúng bị
báo cáo là sai.  Script này:

1. đọc các checkpoint `artifacts/eval_ckpt_*.json`;
2. với mỗi dòng `judge_degraded`, dựng lại **cùng câu trả lời** (đã cache nên
   không tốn quota sinh đáp án) và chấm lại bằng judge hiện hành;
3. ghi đè các dòng đã sửa **ngay trong checkpoint** và lưu bản sao lưu.

Dùng để vá lỗi chấm điểm mà **không** phải chạy lại toàn bộ đánh giá.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.config import get_settings  # noqa: E402
from hgrag.eval.dataset import EvalDataset  # noqa: E402
from hgrag.eval.judge import build_judge, is_out_of_kb  # noqa: E402
from hgrag.eval.run_eval import ABLATIONS, _answer_system  # noqa: E402
from hgrag.qa.pipeline import build_pipeline  # noqa: E402

logger = logging.getLogger("rejudge")


def _ablation_for(system: str) -> dict[str, bool]:
    """Cấu hình ablation của một hệ thống (khớp với ``run_eval``)."""
    name = ABLATIONS.get(system)
    if not name:
        return {}
    return {name: True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chấm lại các dòng judge_degraded")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="ghi đè các dòng đã sửa ngay trong checkpoint (mặc định: tạo file mới)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    pipeline = build_pipeline(settings, mock=False)
    judge = build_judge(pipeline.llm_client, mock=False)

    dataset = EvalDataset.load_jsonl(REPO_ROOT / "data/processed/eval_questions.jsonl")
    by_qid = {q.qid: q for q in dataset}

    checkpoints = sorted((REPO_ROOT / "artifacts").glob("eval_ckpt_*.json"))
    # Bản sao lưu trước khi chấm lại có tên ``*.pre_rejudge.json`` -- chúng
    # **không** phải checkpoint của lượt chạy, nên phải loại khỏi danh sách
    # (nếu không thì mỗi dòng hỏng sẽ được chấm lại hai lần).
    checkpoints = [
        p
        for p in checkpoints
        if "rejudged" not in p.name and not p.name.endswith(".pre_rejudge.json")
    ]

    total_fixed = 0
    for path in checkpoints:
        data = json.loads(path.read_text())
        rows = data.get("rows", [])
        degraded = [r for r in rows if r.get("judge_degraded")]
        if not degraded:
            continue
        logger.info("%s: %d dòng degraded", path.name, len(degraded))
        if args.dry_run:
            for row in degraded:
                print(f"  {row['system']:22s} {row['qid']:12s} {row.get('judge_reason', '')[:40]}")
            continue

        if args.in_place:
            shutil.copy(path, path.with_suffix(".pre_rejudge.json"))

        for row in degraded:
            question = by_qid.get(row["qid"])
            if question is None:
                logger.warning("Không tìm thấy câu hỏi %s -- bỏ qua", row["qid"])
                continue
            result = _answer_system(
                pipeline, question.question, row["system"], _ablation_for(row["system"])
            )
            evidence = list(getattr(result, "evidence", []) or [])
            verdict = judge.grade(
                question=question.question,
                answer=str(getattr(result, "answer", "") or ""),
                reference=question.answer,
                gold_facts=list(question.gold_relations),
                evidence=evidence,
                out_of_kb=is_out_of_kb(question),
                closed_book=(row["system"] == "B0"),
            )
            row["judge_score"] = verdict.score
            row["judge_correct"] = bool(verdict.correct)
            row["grounded"] = bool(verdict.grounded)
            row["judge_reason"] = verdict.reason
            row["judge_degraded"] = bool(getattr(verdict, "degraded", False))
            total_fixed += 1
            logger.info(
                "  %s %s: score=%.1f correct=%s degraded=%s",
                row["system"],
                row["qid"],
                verdict.score,
                verdict.correct,
                row["judge_degraded"],
            )

        if args.in_place:
            path.write_text(json.dumps(data, ensure_ascii=False))
            logger.info("Đã cập nhật %s", path.name)

    logger.info("Tổng: %d dòng đã chấm lại", total_fixed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
