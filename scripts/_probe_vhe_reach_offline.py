"""Offline check: does the traversal reach the gold event of each VHE question?

The query-understanding step calls the LLM, which makes the check slow.  This
variant reuses the *question* text directly as the anchor mention, so the
measurement isolates the graph layer (linker + traversal) from the LLM.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.eval.dataset import EvalDataset
from hgrag.kg.store import Neo4jGraphStore
from hgrag.retrieval.entity_linker import EntityLinker
from hgrag.retrieval.graph_search import GraphSearcher


def main() -> None:
    settings = get_settings()
    ncfg = getattr(settings, "neo4j", {}) or {}
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str(ncfg.get("user", "neo4j")),
        password=settings.neo4j_password,
        database=str(ncfg.get("database", "neo4j")),
    )
    from hgrag.kg.alias import AliasResolver

    resolver = AliasResolver(store=store, settings=settings)
    resolver.build_alias_index(store.list_entities())
    linker = EntityLinker(resolver=resolver, settings=settings)
    searcher = GraphSearcher(store=store, settings=settings)

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

        mentions = list(question.gold_entities or [])
        linked = linker.link(mentions)
        if not linked:
            stats["no_anchor"] += 1
            if len(examples) < 10:
                examples.append(f"[no-anchor] {question.qid} {question.question} {mentions}")
            continue
        stats["linked"] += 1
        if any(e.uid.startswith("vhe") for e in linked):
            stats["anchor_vhe"] += 1
        else:
            stats["anchor_non_vhe"] += 1

        result = searcher.search(
            [e.uid for e in linked],
            relation_plan=[],
            max_hops=3,
            max_paths=20,
            query_text=question.question,
        )
        reached = {str(node.get("name") or "") for path in result.paths for node in path.nodes}
        if gold_nodes & reached:
            stats["gold_reached"] += 1
        else:
            stats["gold_missed"] += 1
            if len(examples) < 10:
                examples.append(
                    f"[miss] {question.qid} {question.question}\n"
                    f"        anchors={[(e.name, e.uid) for e in linked]}\n"
                    f"        gold={sorted(gold_nodes)}\n"
                    f"        reached={sorted(x for x in reached if x)[:8]}"
                )

    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    for line in examples:
        print(line)


if __name__ == "__main__":
    main()
