#!/usr/bin/env python
"""Kiểm chứng *tầng fusion* mà không gọi LLM sinh câu trả lời.

Chỉ cần biết context cuối cùng có gì: có đủ đường đi đồ thị không, có câu
trùng lặp không, và câu trả lời vàng có nằm trong đó không.
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
    print("REFERENCE:", q.answer)
    print("GOLD RELATIONS:", json.dumps(q.gold_relations, ensure_ascii=False))
    print()

    understanding = pipeline._understand(q.question)
    linked = pipeline._link(understanding, {})
    graph_result = pipeline._graph_search(linked, understanding, 2, {})
    paths = list(getattr(graph_result, "paths", []) or [])
    vector_hits = pipeline._vector_search(q.question, pipeline.top_k, understanding, {})
    vector_hits = pipeline._rerank(q.question, vector_hits, {})
    evidence = pipeline._fuse(vector_hits, graph_result, understanding, {})
    pack = pipeline._build_context(evidence, {})

    print(f"đường đi đồ thị: {len(paths)}")
    for p in paths[:6]:
        print("  -", str(getattr(p, "to_natural_text", lambda: "")())[:150].replace("\n", " | "))
    print()
    print(f"evidence sau fusion: {len(evidence)}")
    seen: dict[str, int] = {}
    for item in evidence:
        text = str(getattr(item, "text", "") or "")
        eid = str(getattr(item, "evidence_id", "") or "")
        kind = str(getattr(item, "kind", "") or "")
        key = " ".join(text.lower().split())
        seen[key] = seen.get(key, 0) + 1
        print(f"  [{eid}|{kind}] {text[:130]}".replace("\n", " | "))
    dupes = {k: v for k, v in seen.items() if v > 1}
    print()
    print("câu trùng lặp:", len(dupes))
    print("số evidence đồ thị:", sum(1 for i in evidence if getattr(i, "kind", "") == "graph_path"))

    # Câu trả lời vàng có nằm trong context không?
    gold = (q.answer or "").strip().lower()
    if gold:
        joined = " ".join(str(getattr(i, "text", "") or "").lower() for i in evidence)
        print(f"đáp án vàng '{gold}' có trong context: {gold in joined}")
    print()
    print("=== CONTEXT TEXT ===")
    print((pack.text if pack else "")[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
