#!/usr/bin/env python
"""Kiểm tra tính nhất quán giữa bộ câu hỏi đánh giá và đồ thị KG.

Script này *không* chạy LLM: nó đối chiếu các node trong ``gold_paths``
với các node thực sự có trong Neo4j để đo mức độ khớp.  Kết quả dùng để
quyết định xem metric ``path_hit_rate`` có đo được điều gì hay không
(docs/04 §5.4 -- "đường đi sai" là một nhóm lỗi cần lượng hoá).

Usage
-----
    python scripts/10_validate_eval_paths.py
    python scripts/10_validate_eval_paths.py --questions data/processed/eval_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.eval.dataset import EvalDataset  # noqa: E402

DEFAULT_QUESTIONS = "data/processed/eval_questions.jsonl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Đối chiếu gold_paths với đồ thị Neo4j")
    p.add_argument("--questions", type=str, default=DEFAULT_QUESTIONS)
    p.add_argument("--store", type=str, default="neo4j", help="neo4j | inmemory")
    p.add_argument("--out", type=str, default=None, help="ghi kết quả JSON ra file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = REPO_ROOT / questions_path
    dataset = EvalDataset.load_jsonl(questions_path)
    if not len(dataset):
        print(json.dumps({"error": f"Không đọc được {questions_path}"}, ensure_ascii=False))
        return 2

    from hgrag.config import get_settings

    settings = get_settings()
    if args.store == "neo4j":
        from hgrag.kg.store import Neo4jGraphStore

        ncfg = settings.neo4j
        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=str(ncfg.get("user", "neo4j")),
            password=settings.neo4j_password,
            database=str(ncfg.get("database", "neo4j")),
        )
    else:
        from hgrag.kg.store import InMemoryGraphStore

        store = InMemoryGraphStore()

    records = store._run("MATCH (e:Entity) RETURN e.name AS name, e.entity_type AS t")
    names = {str(r["name"]) for r in records}
    lowered = {n.lower() for n in names}

    per_source: Counter[str] = Counter()
    per_source_missing: Counter[str] = Counter()
    missing_examples: dict[str, list[str]] = {}
    total_nodes = 0
    missing_nodes = 0
    exact = 0
    case_only = 0
    substring = 0
    n_questions_with_paths = 0

    for question in dataset.questions:
        paths = [list(p) for p in (question.gold_paths or []) if p]
        if not paths:
            continue
        n_questions_with_paths += 1
        for path in paths:
            for node in path:
                total_nodes += 1
                per_source[question.source] += 1
                if node in names:
                    exact += 1
                    continue
                if node.lower() in lowered:
                    case_only += 1
                    missing_nodes += 1
                    per_source_missing[question.source] += 1
                    missing_examples.setdefault(question.source, []).append(node)
                    continue
                # Substring: the node is contained in a longer KG node (e.g.
                # the gold "sinh ra Đế Nghi" inside "Đế Minh sinh ra Đế Nghi").
                if any(node in candidate for candidate in names):
                    substring += 1
                    continue
                missing_nodes += 1
                per_source_missing[question.source] += 1
                missing_examples.setdefault(question.source, []).append(node)

    report = {
        "n_questions": len(dataset.questions),
        "n_questions_with_paths": n_questions_with_paths,
        "n_gold_nodes": total_nodes,
        "exact": exact,
        "case_only": case_only,
        "substring": substring,
        "missing": missing_nodes,
        "match_rate_exact": round(exact / total_nodes, 4) if total_nodes else 0.0,
        "match_rate_with_substring": (
            round((exact + case_only + substring) / total_nodes, 4) if total_nodes else 0.0
        ),
        "per_source_nodes": dict(per_source),
        "per_source_unmatched": dict(per_source_missing),
        "missing_examples": {
            k: sorted(set(v))[:20] for k, v in missing_examples.items()
        },
    }
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["out"] = str(out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
