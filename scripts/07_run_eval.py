#!/usr/bin/env python
"""Step 7 of the pipeline: run the evaluation and write the report.

Usage
-----
    python scripts/07_run_eval.py --limit 20
    python scripts/07_run_eval.py --systems ours B1 --no-ablations
    python scripts/07_run_eval.py --mock --limit 10        # offline, fast
    python scripts/07_run_eval.py --questions data/processed/eval_questions.jsonl

Outputs
-------
    reports/results.csv        bảng số liệu theo hệ thống × nhóm câu hỏi
    reports/results.md         bảng chính + bảng ablation + thống kê
    reports/figures/*.png      biểu đồ
    artifacts/eval_run.json    raw rows (để phân tích lỗi)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.config import get_settings  # noqa: E402
from hgrag.eval.dataset import EvalDataset  # noqa: E402
from hgrag.eval.judge import build_judge  # noqa: E402
from hgrag.eval.report import build_report  # noqa: E402
from hgrag.eval.run_eval import SYSTEMS, compare_systems, run_evaluation  # noqa: E402
from hgrag.qa.pipeline import QAPipeline  # noqa: E402

logger = logging.getLogger("07_run_eval")

DEFAULT_QUESTIONS = "data/processed/eval_questions.jsonl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chạy đánh giá và sinh báo cáo")
    p.add_argument("--questions", type=str, default=DEFAULT_QUESTIONS, help="file JSONL câu hỏi")
    p.add_argument("--limit", type=int, default=None, help="chỉ đánh giá N câu đầu")
    p.add_argument(
        "--systems",
        type=str,
        nargs="+",
        default=list(SYSTEMS),
        help="các hệ thống cần chạy (B0 B1 B2 ours)",
    )
    p.add_argument("--no-ablations", action="store_true", help="bỏ 7 ablation")
    p.add_argument("--mock", action="store_true", help="chạy offline (mock LLM + embedder)")
    p.add_argument("--out", type=str, default=None, help="ghi raw rows ra file JSON")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="file checkpoint (JSON) để resume khi run bị gián đoạn",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=5, help="ghi checkpoint sau mỗi N dòng"
    )
    p.add_argument("--no-resume", action="store_true", help="bỏ qua checkpoint cũ")
    p.add_argument("--reports", type=str, default="reports", help="thư mục output của report")
    p.add_argument("--bootstrap", type=int, default=10_000, help="số lần lặp bootstrap")
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="số câu hỏi xử lý song song (hạn mức req/phút do client quản lý)",
    )
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--verbose", action="store_true", help="bật log DEBUG")
    return p


def build_pipeline(mock: bool, settings: Any) -> QAPipeline:
    from hgrag.qa.pipeline import build_pipeline as _factory

    return _factory(settings, mock=mock)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    settings.ensure_dirs()

    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = REPO_ROOT / questions_path
    dataset = EvalDataset.load_jsonl(questions_path)
    if not len(dataset):
        print(
            json.dumps(
                {"error": f"Không đọc được câu hỏi từ {questions_path}"}, ensure_ascii=False
            )
        )
        return 2

    try:
        pipeline = build_pipeline(args.mock, settings)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"error": f"Không khởi tạo được pipeline: {exc}"}, ensure_ascii=False
            )
        )
        return 3

    judge = build_judge(pipeline.llm_client, mock=args.mock)
    checkpoint = args.checkpoint
    if checkpoint and not Path(checkpoint).is_absolute():
        checkpoint = str(REPO_ROOT / checkpoint)
    run = run_evaluation(
        pipeline,
        dataset,
        systems=list(args.systems),
        ablations=not args.no_ablations,
        judge=judge,
        limit=args.limit,
        skip_existing=not args.no_resume,
        checkpoint_path=checkpoint,
        checkpoint_every=args.checkpoint_every,
        concurrency=args.concurrency,
    )
    if not run.rows:
        print(json.dumps({"error": "Không có dòng kết quả nào"}, ensure_ascii=False))
        return 4

    comparisons = [
        compare_systems(
            run,
            baseline=baseline,
            target="ours",
            n_samples=args.bootstrap,
            seed=args.seed,
        )
        for baseline in ("B0", "B1", "B2")
        if baseline in args.systems
    ]

    reports_dir = Path(args.reports)
    if not reports_dir.is_absolute():
        reports_dir = REPO_ROOT / reports_dir
    artefacts = build_report(run, reports_dir, comparisons)

    out_path = Path(args.out) if args.out else (settings.artifacts / "eval_run.json")
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    run.save(out_path)

    summary = {
        "n_questions": run.meta.get("n_questions"),
        "n_rows": len(run.rows),
        "systems": sorted({r.system for r in run.rows}),
        "artifacts": artefacts,
        "raw_rows": str(out_path),
        "comparisons": comparisons,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
