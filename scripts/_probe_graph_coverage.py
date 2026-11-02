#!/usr/bin/env python
"""Kiểm tra *offline* đường đi đồ thị có chạm tới đáp án vàng không.

Không gọi LLM sinh câu trả lời: chỉ `_understand` (đã cache), entity linking và
graph search.  Nhờ vậy có thể quét cả tập câu hỏi để tìm ra câu nào mà đồ thị
**không** dựng được đường đi tới đáp án vàng — tức là lỗi *truy xuất*, không
phải lỗi *sinh câu trả lời*.
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

    normalise = lambda s: re.sub(r"\s+", " ", s.lower()).strip()  # noqa: E731
    gold_n = normalise(gold)
    if not gold_n:
        return False
    text_n = normalise(text)
    if gold_n in text_n:
        return True
    # VHE gold đôi khi là một cụm ("Lâm Lư hầu Chu Táo") còn đường đi chỉ
    # nêu từng thành phần.
    parts = [p for p in re.split(r"[;,\s]+", gold_n) if len(p) > 2]
    return bool(parts) and all(p in text_n for p in parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đo độ phủ gold của đồ thị (offline)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="artifacts/graph_coverage.json")
    args = parser.parse_args(argv)

    from hgrag.config import get_settings
    from hgrag.eval.dataset import EvalDataset
    from hgrag.qa.pipeline import build_pipeline

    settings = get_settings()
    pipeline = build_pipeline(settings, mock=False)
    dataset = EvalDataset.load_jsonl(REPO_ROOT / "data/processed/eval_questions.jsonl")
    questions = [q for q in dataset.questions if (q.answer or "").strip()]
    if args.limit:
        questions = questions[: args.limit]

    rows = []
    for q in questions:
        try:
            understanding = pipeline._understand(q.question)
            linked = pipeline._link(understanding, {})
            graph_result = pipeline._graph_search(
                linked, understanding, pipeline.max_hops, {}
            )
            paths = list(getattr(graph_result, "paths", []) or [])
        except Exception as exc:  # noqa: BLE001
            rows.append({"qid": q.qid, "error": str(exc)[:120], "hit": False})
            continue
        blob = " ".join(
            str(getattr(p, "to_natural_text", lambda: "")()) for p in paths
        )
        blob += " " + " ".join(str(getattr(e, "name", "")) for e in (linked or []))
        rows.append(
            {
                "qid": q.qid,
                "question_type": q.question_type,
                "question": q.question,
                "gold": q.answer,
                "n_linked": len(linked or []),
                "n_paths": len(paths),
                "hit": _hits(blob, q.answer),
            }
        )

    ok = [r for r in rows if "error" not in r]
    misses = [r for r in ok if not r["hit"]]
    summary = {
        "n": len(ok),
        "hit": sum(1 for r in ok if r["hit"]),
        "miss": len(misses),
        "errors": [r for r in rows if "error" in r],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n--- câu không dựng được đường đi tới đáp án vàng ---")
    for r in misses[:30]:
        print(f"[{r['qid']}] {r['question'][:70]}  -> {r['gold']}  "
              f"(linked={r['n_linked']}, paths={r['n_paths']})")
    (REPO_ROOT / args.out).write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2)
    )
    print(f"\nĐã ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
