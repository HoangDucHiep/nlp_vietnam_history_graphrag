#!/usr/bin/env python
"""In các đường đi đồ thị tìm được (kèm văn bản tự nhiên) cho một câu hỏi.

Dùng để kiểm tra: đường đi chứa đáp án vàng có thực sự đi vào CONTEXT hay
bị diversity cap / ngân sách evidence loại mất.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
logging.basicConfig(level=logging.ERROR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="In đường đi đồ thị của một câu hỏi")
    parser.add_argument("qids", nargs="+")
    parser.add_argument("--gold", default="", help="chuỗi cần tìm trong đường đi")
    args = parser.parse_args(argv)

    from hgrag.config import get_settings
    from hgrag.eval.dataset import EvalDataset
    from hgrag.qa.pipeline import build_pipeline

    settings = get_settings()
    dataset = EvalDataset.load_jsonl(REPO_ROOT / "data/processed/eval_questions.jsonl")
    by_qid = {q.qid: q for q in dataset.questions}

    pipeline = build_pipeline(settings, mock=False)

    for qid in args.qids:
        q = by_qid[qid]
        print("=" * 90)
        print(f"[{qid}] {q.question}")
        print(f"GOLD: {q.answer}")
        understanding = pipeline._understand(q.question)
        linked = pipeline._link(understanding, {})
        graph_result = pipeline._graph_search(linked, understanding, pipeline.max_hops, {})
        paths = list(getattr(graph_result, "paths", []) or [])
        print(f"linked: {[getattr(e, 'name', '') for e in (linked or [])]}")
        print(f"{len(paths)} đường đi")
        for index, path in enumerate(paths):
            natural = str(path.to_natural_text() or "")
            flag = ""
            if args.gold and args.gold.lower() in natural.lower():
                flag = "   <<< CHỨA ĐÁP ÁN VÀNG"
            print(f"--- path #{index} score={path.score:.3f} hops={path.hops}{flag}")
            print(natural[:600])
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
