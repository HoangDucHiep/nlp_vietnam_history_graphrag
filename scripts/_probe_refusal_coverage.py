#!/usr/bin/env python
"""Đo: với các câu bị từ chối, đáp án vàng có nằm trong CONTEXT không?

Phân biệt hai nguyên nhân rất khác nhau:

* **gold trong CONTEXT** -> lỗi *sinh* (prompt / mô hình từ chối oan);
* **gold KHÔNG trong CONTEXT** -> lỗi *truy xuất* (đồ thị / vector chưa tìm
  được đáp án).

Không gọi LLM sinh câu trả lời: chỉ dựng lại context.
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


def _hits(text: str, gold: str) -> bool:
    import re

    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()  # noqa: E731
    gold_n = norm(gold)
    if not gold_n:
        return False
    text_n = norm(text)
    if gold_n in text_n:
        return True
    parts = [p for p in re.split(r"[;,\s]+", gold_n) if len(p) > 2]
    return bool(parts) and all(p in text_n for p in parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đo độ phủ đáp án vàng trong CONTEXT")
    parser.add_argument("--run", default="artifacts/eval_run.json")
    parser.add_argument("--system", default="ours")
    parser.add_argument("--out", default="artifacts/refusal_context_coverage.json")
    args = parser.parse_args(argv)

    from hgrag.config import get_settings
    from hgrag.eval.dataset import EvalDataset
    from hgrag.qa.pipeline import build_pipeline

    payload = json.loads((REPO_ROOT / args.run).read_text(encoding="utf-8"))
    rows = [
        r
        for r in payload.get("rows") or []
        if r["system"] == args.system
        and r.get("refused")
        and r.get("question_type") != "out_of_kb"
    ]
    dataset = EvalDataset.load_jsonl(REPO_ROOT / "data/processed/eval_questions.jsonl")
    by_qid = {q.qid: q for q in dataset.questions}

    settings = get_settings()
    pipeline = build_pipeline(settings, mock=False)

    out_rows = []
    for row in rows:
        qid = row["qid"]
        q = by_qid[qid]
        try:
            understanding = pipeline._understand(q.question)
            linked = pipeline._link(understanding, {})
            graph_result = pipeline._graph_search(linked, understanding, pipeline.max_hops, {})
            vector_hits = pipeline._vector_search(
                q.question, pipeline.top_k, understanding, {}
            )
            vector_hits = pipeline._rerank(q.question, vector_hits, {})
            evidence = pipeline._fuse(vector_hits, graph_result, understanding, {})
            context = pipeline._build_context(evidence, {})
            text = getattr(context, "text", "") or ""
            hit = _hits(text, q.answer)
            n_evidence = len(evidence)
        except Exception as exc:  # noqa: BLE001
            hit = False
            text = ""
            n_evidence = 0
            print(f"  lỗi {qid}: {exc}")
        out_rows.append(
            {
                "qid": qid,
                "question": q.question,
                "gold": q.answer,
                "gold_in_context": hit,
                "n_evidence": n_evidence,
                "n_ctx_chars": len(text),
            }
        )
        flag = "CÓ" if hit else "KHÔNG"
        print(f"[{qid}] gold trong CONTEXT: {flag}  ({len(text)} ký tự) — {q.question[:60]}")

    n_hit = sum(1 for r in out_rows if r["gold_in_context"])
    summary = {
        "system": args.system,
        "n_refusals": len(out_rows),
        "gold_in_context": n_hit,
        "gold_missing": len(out_rows) - n_hit,
        "ratio_in_context": (n_hit / len(out_rows)) if out_rows else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (REPO_ROOT / args.out).write_text(
        json.dumps({"summary": summary, "rows": out_rows}, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
