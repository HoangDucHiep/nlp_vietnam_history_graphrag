#!/usr/bin/env python
"""Đo *độ phủ context* offline: đáp án vàng có nằm trong context không?

Không gọi LLM sinh câu trả lời (chỉ gọi `_understand`, đã cache), nên phép đo
này miễn phí và chạy được cho cả tập 150 câu.

Chỉ số:
* ``in_context``      -- chuỗi đáp án vàng xuất hiện trong context;
* ``in_graph``        -- xuất hiện trong phần evidence *đồ thị*;
* ``n_graph``         -- số evidence đồ thị lọt vào context;
* ``dupes``           -- số câu trùng lặp trong context.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
logging.basicConfig(level=logging.ERROR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đo độ phủ context offline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="artifacts/context_coverage.json")
    parser.add_argument("--system", default="ours")
    args = parser.parse_args(argv)

    from hgrag.config import get_settings
    from hgrag.eval.dataset import EvalDataset
    from hgrag.qa.pipeline import build_pipeline

    settings = get_settings()
    pipeline = build_pipeline(settings, mock=False)
    dataset = EvalDataset.load_jsonl(REPO_ROOT / "data/processed/eval_questions.jsonl")
    questions = list(dataset.questions)
    if args.limit:
        questions = questions[: args.limit]

    rows = []
    for q in questions:
        try:
            understanding = pipeline._understand(q.question)
            linked = pipeline._link(understanding, {})
            graph_result = pipeline._graph_search(linked, understanding, pipeline.max_hops, {})
            vector_hits = pipeline._vector_search(
                q.question, pipeline.top_k, understanding, {}
            )
            vector_hits = pipeline._rerank(q.question, vector_hits, {})
            evidence = pipeline._fuse(vector_hits, graph_result, understanding, {})
        except Exception as exc:  # noqa: BLE001
            rows.append({"qid": q.qid, "error": str(exc)[:120]})
            continue

        texts = [str(getattr(e, "text", "") or "") for e in evidence]
        kinds = [str(getattr(e, "kind", "") or "") for e in evidence]
        joined = " ".join(texts).lower()
        graph_text = " ".join(t for t, k in zip(texts, kinds, strict=False) if k == "graph_path").lower()
        gold = (q.answer or "").strip().lower()
        seen: dict[str, int] = {}
        for t in texts:
            key = " ".join(t.lower().split())
            seen[key] = seen.get(key, 0) + 1
        rows.append(
            {
                "qid": q.qid,
                "question_type": q.question_type,
                "gold": q.answer,
                "in_context": bool(gold) and gold in joined,
                "in_graph": bool(gold) and gold in graph_text,
                "n_evidence": len(evidence),
                "n_graph": sum(1 for k in kinds if k == "graph_path"),
                "dupes": sum(v - 1 for v in seen.values() if v > 1),
            }
        )

    total = len(rows)
    ok = [r for r in rows if "error" not in r]
    summary = {
        "n": total,
        "n_ok": len(ok),
        "in_context": sum(1 for r in ok if r["in_context"]),
        "in_graph": sum(1 for r in ok if r["in_graph"]),
        "mean_n_graph": round(sum(r["n_graph"] for r in ok) / max(1, len(ok)), 2),
        "mean_dupes": round(sum(r["dupes"] for r in ok) / max(1, len(ok)), 2),
        "errors": [r for r in rows if "error" in r],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (REPO_ROOT / args.out).write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2)
    )
    print(f"Đã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
