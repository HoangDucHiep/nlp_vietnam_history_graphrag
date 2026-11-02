#!/usr/bin/env python
"""Kiểm chứng hai lỗi đã sửa trên pipeline thật.

1. **Baseline B0 (closed-book)**: trước đây 98/150 câu trả về câu từ chối
   có sẵn vì Gemma 4 dùng hết ngân sách cho kênh suy luận.  Sau khi chuyển
   sang JSON mode (tắt thinking) và thêm bộ phát hiện lặp, baseline phải
   trả lời được phần lớn câu hỏi mà nó biết.
2. **Truy xuất đồ thị**: câu hỏi VHE neo vào *theme* ("Ai là người thành
   lập Văn Lang?") trước đây không có đường đi nào; sau khi thêm traversal
   ngược qua event hub thì phải tìm được đường đi chứa đáp án.

Usage:
    python scripts/validate_fixes.py [--n 8]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hgrag.config import get_settings  # noqa: E402
from hgrag.eval.dataset import EvalDataset  # noqa: E402
from hgrag.eval.metrics import path_hit_rate  # noqa: E402
from hgrag.qa.answer import AnswerGenerator  # noqa: E402
from hgrag.qa.pipeline import build_pipeline  # noqa: E402

REFUSAL = "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này."


def check_closed_book(dataset: EvalDataset, n: int) -> dict:
    """The B0 baseline must now answer (and refuse) instead of going blank."""
    settings = get_settings()
    pipeline = build_pipeline(settings)
    generator = AnswerGenerator(pipeline.llm_client, settings)
    questions = [q for q in dataset.questions if q.question_type != "out_of_kb"][:n]
    rows: list[dict] = []
    for question in questions:
        started = time.perf_counter()
        result = generator.generate_closed_book(question.question)
        elapsed = time.perf_counter() - started
        answer = result.answer.strip()
        rows.append(
            {
                "qid": question.qid,
                "question": question.question,
                "reference": question.answer,
                "answer": answer[:200],
                "refused": result.refused,
                "is_canned": answer == REFUSAL,
                "elapsed": round(elapsed, 2),
            }
        )
        print(
            f"[B0] {question.qid} {elapsed:6.1f}s  refused={result.refused!s:5}  "
            f"canned={answer == REFUSAL!s:5}  {answer[:90]!r}"
        )
    return {
        "n": len(rows),
        "n_canned": sum(1 for r in rows if r["is_canned"]),
        "n_refused": sum(1 for r in rows if r["refused"]),
        "avg_elapsed": round(statistics.mean(r["elapsed"] for r in rows), 2)
        if rows
        else 0.0,
        "rows": rows,
    }


def check_graph_paths(dataset: EvalDataset, n: int) -> dict:
    """The VHE template questions must now have a gold path among the paths."""
    settings = get_settings()
    pipeline = build_pipeline(settings)
    questions = [
        q
        for q in dataset.questions
        if q.source == "vhe_template" and q.gold_paths
    ][:n]
    rows: list[dict] = []
    for question in questions:
        understanding = pipeline._understand(question.question)
        linked = pipeline._link(understanding, {})
        result = pipeline._graph_search(linked, understanding, pipeline.max_hops, {})
        paths = list(getattr(result, "paths", []) or [])
        retrieved = [[str(node.get("name") or "") for node in p.nodes] for p in paths]
        hit = path_hit_rate(question.gold_paths, retrieved)
        rows.append(
            {
                "qid": question.qid,
                "question": question.question,
                "gold_paths": question.gold_paths,
                "n_paths": len(paths),
                "path_hit": hit,
                "linked": [e.name for e in linked],
            }
        )
        print(
            f"[KG] {question.qid} paths={len(paths):3d} hit={hit:.0f}  "
            f"{question.question[:70]}"
        )
        for path in paths[:2]:
            print(f"       {path.to_text()[:110]}")
    return {
        "n": len(rows),
        "n_hit": sum(1 for r in rows if r["path_hit"] > 0),
        "avg_paths": round(statistics.mean(r["n_paths"] for r in rows), 2) if rows else 0,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm chứng các bản sửa")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--questions",
        default="data/processed/eval_questions.jsonl",
    )
    args = parser.parse_args(argv)

    dataset = EvalDataset.load_jsonl(REPO_ROOT / args.questions)
    print(f"== Baseline B0 (closed-book, JSON mode) -- {args.n} câu")
    closed_book = check_closed_book(dataset, args.n)
    print(
        f"   -> canned={closed_book['n_canned']}/{closed_book['n']}  "
        f"refused={closed_book['n_refused']}/{closed_book['n']}  "
        f"avg={closed_book['avg_elapsed']}s\n"
    )

    print(f"== Truy xuất đồ thị (event hub ngược) -- {args.n} câu")
    graph = check_graph_paths(dataset, args.n)
    print(
        f"   -> path_hit={graph['n_hit']}/{graph['n']}  "
        f"avg_paths={graph['avg_paths']}\n"
    )

    out = {
        "closed_book": closed_book,
        "graph_paths": graph,
    }
    path = REPO_ROOT / "artifacts" / "validate_fixes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
