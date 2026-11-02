"""Diagnose the VHE refusals: dump the FULL context handed to the answerer."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.qa.pipeline import build_pipeline

QUESTIONS = [
    "Ai là người cưới Sơn Tinh?",
    "Ai là người tấn công Nam Chiếu?",
    "Ai là người thành lập Nam Hải?",
    "Sự kiện qua đời Tần Thuỷ Hoàng diễn ra ở đâu?",
]


def main() -> None:
    settings = get_settings()
    pipeline = build_pipeline(settings)
    for question in QUESTIONS:
        understanding = pipeline._understand(question)
        linked = pipeline._link(understanding, {})
        graph = pipeline._graph_search(linked, understanding, pipeline.max_hops, {})
        vector_hits = pipeline._vector_search(
            question, pipeline.top_k, understanding, {}
        )
        vector_hits = pipeline._rerank(question, vector_hits, {})
        evidence = pipeline._fuse(vector_hits, graph, understanding, {})
        pack = pipeline._build_context(evidence, {})
        print("=" * 78)
        print("Q:", question)
        print("linked:", [(e.name, e.uid) for e in linked])
        print(
            "n_vector:",
            len(vector_hits),
            "n_paths:",
            len(graph.paths),
            "n_evidence:",
            len(pack.evidence_ids),
            "tokens:",
            pack.n_tokens_estimate,
            "truncated:",
            pack.truncated,
        )
        print("---- CONTEXT ----")
        print(pack.text)
        print()


if __name__ == "__main__":
    main()
