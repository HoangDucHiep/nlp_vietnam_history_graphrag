"""Measure whether the graph traversal reaches the gold event of each VHE question."""
from __future__ import annotations

import json
import sys
from collections import Counter

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.eval.dataset import EvalDataset
from hgrag.qa.pipeline import build_pipeline


def main() -> None:
    settings = get_settings()
    pipeline = build_pipeline(settings)
    dataset = EvalDataset.load_jsonl("data/processed/eval_questions.jsonl")

    stats = Counter()
    examples: list[str] = []
    for question in dataset.questions:
        if question.source != "vhe_template":
            continue
        stats["n"] += 1
        gold_paths = [list(p) for p in (question.gold_paths or []) if p]
        if not gold_paths:
            stats["no_gold_path"] += 1
            continue
        gold_nodes = {n for path in gold_paths for n in path}

        understanding = pipeline._understand(question.question)
        linked = pipeline._link(understanding, {})
        stats["linked"] += 1 if linked else 0
        if not linked:
            stats["no_anchor"] += 1
            if len(examples) < 15:
                examples.append(f"[no-anchor] {question.qid} {question.question}")
            continue

        uids = [e.uid for e in linked]
        if any(u.startswith("vhe") for u in uids):
            stats["anchor_vhe"] += 1
        else:
            stats["anchor_non_vhe"] += 1

        graph = pipeline._graph_search(linked, understanding, pipeline.max_hops, {})
        reached = {
            str(node.get("name") or "") for path in graph.paths for node in path.nodes
        }
        hit = bool(gold_nodes & reached)
        stats["gold_reached"] += 1 if hit else 0
        if not hit:
            stats["gold_missed"] += 1
            if len(examples) < 15:
                examples.append(
                    f"[miss] {question.qid} {question.question}\n"
                    f"        anchors={[(e.name, e.uid) for e in linked]}\n"
                    f"        gold_nodes={sorted(gold_nodes)}\n"
                    f"        reached={sorted(x for x in reached if x)[:10]}"
                )

    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print()
    for line in examples:
        print(line)


if __name__ == "__main__":
    main()
