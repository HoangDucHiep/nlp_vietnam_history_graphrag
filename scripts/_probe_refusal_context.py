#!/usr/bin/env python
"""In ra CONTEXT thực tế của một câu hỏi để xem vì sao hệ thống từ chối.

Không gọi LLM sinh câu trả lời (đã cache): chỉ dựng lại *context* để đối chiếu
với đáp án vàng.
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
    parser = argparse.ArgumentParser(description="In CONTEXT của một câu hỏi")
    parser.add_argument("qids", nargs="+")
    parser.add_argument("--chars", type=int, default=4000)
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
        vector_hits = pipeline._vector_search(
            q.question, pipeline.top_k, understanding, {}
        )
        vector_hits = pipeline._rerank(q.question, vector_hits, {})
        evidence = pipeline._fuse(vector_hits, graph_result, understanding, {})
        context_pack = pipeline._build_context(evidence, {})
        text = getattr(context_pack, "text", "") or ""
        print(f"--- CONTEXT ({len(text)} ký tự, {len(evidence)} evidence) ---")
        print(text[: args.chars])
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
