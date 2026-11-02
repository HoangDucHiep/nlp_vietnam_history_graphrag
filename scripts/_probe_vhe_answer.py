#!/usr/bin/env python
"""So sánh *kho kiến thức* với *câu trả lời* cho một câu hỏi VHE.

Mục tiêu: biết câu trả lời sai là do (a) kho không có thông tin, (b) entity
linking hỏng, hay (c) bước sinh câu trả lời chọn sai entity trong kho.
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

QID = os.environ.get("PROBE_QID", "vhe_00013")

from hgrag.config import get_settings  # noqa: E402
from hgrag.eval.dataset import EvalDataset  # noqa: E402
from hgrag.qa.pipeline import build_pipeline  # noqa: E402


def main() -> int:
    settings = get_settings()
    pipeline = build_pipeline(settings, mock=False)
    dataset = EvalDataset.load_jsonl(REPO_ROOT / "data/processed/eval_questions.jsonl")
    q = next(x for x in dataset if x.qid == QID)

    print("QID:", q.qid)
    print("CÂU HỎI:", q.question)
    print("LOẠI:", q.question_type)
    print("REFERENCE:", q.answer)
    print("GOLD RELATIONS:", json.dumps(q.gold_relations, ensure_ascii=False))
    print("GOLD ENTITY:", json.dumps(q.gold_entities, ensure_ascii=False))
    print()

    understanding = pipeline._understand(q.question)
    print("=== UNDERSTANDING ===")
    print(json.dumps(understanding.__dict__, ensure_ascii=False, default=str)[:900])
    print()

    linked = pipeline._link(understanding, {})
    print("=== LINKED ENTITIES ===")
    print(json.dumps(
        [getattr(e, "__dict__", e) for e in (linked or [])],
        ensure_ascii=False, default=str)[:900])
    print()

    graph_result = pipeline._graph_search(linked, understanding, 2, {})
    paths = getattr(graph_result, "paths", []) or []
    print(f"=== GRAPH PATHS: {len(paths)} ===")
    for p in paths[:5]:
        print(" -", str(getattr(p, "to_natural_text", lambda: "")())[:220])
    print()

    result = pipeline.ask(q.question, system="ours")
    print("=== ANSWER ===")
    print(result.answer)
    print()
    print("refused:", result.refused)
    print()
    print("=== CONTEXT (evidence) ===")
    for item in (result.evidence or [])[:6]:
        print(" -", str(item.get("text"))[:220])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
