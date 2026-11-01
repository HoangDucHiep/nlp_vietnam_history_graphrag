#!/usr/bin/env python
"""Step 6 of the pipeline: ask a question and print a grounded answer.

Usage
-----
    python scripts/06_ask.py "Trần Hưng Đạo đánh bại quân Mông Cổ vào năm nào?"
    python scripts/06_ask.py "..." --json
    python scripts/06_ask.py "..." --system B0|B1|B2|ours
    python scripts/06_ask.py "..." --no-graph --no-rerank      # ablation
    python scripts/06_ask.py "..." --explain                   # show subgraph
    python scripts/06_ask.py --mock                           # offline mode

The script prints a human-readable answer by default and a JSON payload with
``--json``.  It never crashes because of a single failing stage: every error
is logged and the pipeline degrades gracefully.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.config import get_settings  # noqa: E402
from hgrag.kg.alias import AliasResolver  # noqa: E402
from hgrag.kg.store import InMemoryGraphStore, Neo4jGraphStore  # noqa: E402
from hgrag.llm.client import MockLLMClient, build_llm_client  # noqa: E402
from hgrag.qa.pipeline import ABLATION_FLAGS, QAPipeline  # noqa: E402
from hgrag.retrieval.embeddings import MockEmbedder, build_embedder  # noqa: E402
from hgrag.retrieval.entity_linker import EntityLinker  # noqa: E402
from hgrag.retrieval.graph_search import GraphSearcher  # noqa: E402
from hgrag.retrieval.rerank import build_reranker  # noqa: E402
from hgrag.retrieval.vector_store import NumpyVectorStore, QdrantVectorStore  # noqa: E402

logger = logging.getLogger("06_ask")

SYSTEMS = ("ours", "B0", "B1", "B2")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hỏi đáp trên KG lịch sử Việt Nam")
    p.add_argument("question", nargs="+", help="câu hỏi tiếng Việt")
    p.add_argument("--json", action="store_true", help="in kết quả dạng JSON")
    p.add_argument(
        "--system",
        type=str,
        default="ours",
        choices=SYSTEMS,
        help="hệ thống cần chạy (default: ours)",
    )
    p.add_argument("--top-k", type=int, default=None, help="số chunk lấy từ vector store")
    p.add_argument("--max-hops", type=int, default=None, help="số hop tối đa của graph search")
    p.add_argument("--mock", action="store_true", help="chạy offline với mock LLM/embedder")
    p.add_argument("--explain", action="store_true", help="in thêm subgraph và chuỗi suy luận")
    p.add_argument("--show-evidence", action="store_true", help="in danh sách evidence")
    p.add_argument(
        "--ablation",
        type=str,
        default="",
        help="danh sách cờ ablation phân tách bởi dấu phẩy (vd: no_graph,no_rerank)",
    )
    p.add_argument("--verbose", action="store_true", help="bật log DEBUG")
    return p


def parse_ablation(raw: str) -> dict[str, bool]:
    flags: dict[str, bool] = {name: False for name in ABLATION_FLAGS}
    for part in (raw or "").split(","):
        key = part.strip()
        if not key:
            continue
        if key not in flags:
            raise SystemExit(
                f"Cờ ablation không hợp lệ: {key!r}. Chọn trong: {', '.join(ABLATION_FLAGS)}"
            )
        flags[key] = True
    return flags


def build_store(kind: str, settings: Any) -> Any:
    if kind == "inmemory":
        return InMemoryGraphStore()
    cfg = settings.neo4j
    return Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str(cfg.get("user", "neo4j")),
        password=settings.neo4j_password,
        database=str(cfg.get("database", "neo4j")),
    )


def build_vector_store(settings: Any, mock: bool) -> Any:
    if mock:
        return None
    cfg = settings.qdrant
    try:
        return QdrantVectorStore(
            url=str(cfg.get("url", "http://localhost:6333")),
            vector_size=int(cfg.get("vector_size", 768)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không kết nối được Qdrant (%s) -> dùng NumpyVectorStore", exc)
        return NumpyVectorStore()


def build_pipeline(mock: bool, settings: Any) -> QAPipeline:
    if mock:
        llm_client = MockLLMClient(embed_dim=settings.embed_dim, seed=settings.seed)
        embedder = MockEmbedder(dim=settings.embed_dim)
    else:
        llm_client = build_llm_client(settings)
        embedder = build_embedder(settings, llm_client)

    store = build_store("neo4j", settings)
    vector_store = build_vector_store(settings, mock)
    if vector_store is None:
        vector_store = NumpyVectorStore()

    linker = EntityLinker(
        resolver=AliasResolver(
            store=store,
            llm_client=llm_client,
            embedder=embedder,
            settings=settings,
        ),
        settings=settings,
    )
    searcher = GraphSearcher(store=store, settings=settings, embedder=embedder)
    reranker = build_reranker(settings, embedder)

    return QAPipeline(
        llm_client=llm_client,
        embedder=embedder,
        graph_store=store,
        vector_store=vector_store,
        entity_linker=linker,
        graph_searcher=searcher,
        reranker=reranker,
        settings=settings,
    )


def render_text(result: Any, show_evidence: bool, explain: bool) -> str:
    lines: list[str] = []
    lines.append(f"❓ {result.question}")
    lines.append("")
    lines.append(result.answer)
    if result.citations:
        lines.append("")
        lines.append("Trích dẫn: " + ", ".join(result.citations))
    if result.reasoning_chain:
        lines.append("")
        lines.append("## Suy luận")
        for step in result.reasoning_chain:
            lines.append(f"  - {step}")
    if show_evidence and result.evidence:
        lines.append("")
        lines.append("## Evidence")
        for ev in result.evidence:
            title = (ev.get("meta") or {}).get("title", "")
            lines.append(
                f"  [{ev.get('evidence_id')}] ({ev.get('kind')}, score="
                f"{float(ev.get('score', 0.0)):.4f}) {title}"
            )
            snippet = str(ev.get("text") or "")[:220].replace("\n", " ")
            lines.append(f"      {snippet}")
    if explain and result.explanation:
        lines.append("")
        lines.append("## Subgraph")
        subgraph = result.explanation.get("subgraph") or {}
        nodes = subgraph.get("nodes") or []
        edges = subgraph.get("edges") or []
        lines.append(f"  nodes: {len(nodes)} | edges: {len(edges)}")
        for edge in edges[:20]:
            src = (edge.get("src") or {}).get("label", edge.get("src"))
            dst = (edge.get("dst") or {}).get("label", edge.get("dst"))
            lines.append(f"    ({src}) -[{edge.get('label')}]-> ({dst})")
        trace = result.explanation.get("retrieval_trace") or {}
        if trace:
            lines.append("")
            lines.append("## Trace")
            for key, value in trace.items():
                lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append(
        f"[system={result.system} | refused={result.refused} | "
        f"elapsed={result.elapsed:.2f}s]"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    question = " ".join(args.question).strip()
    if not question:
        print("Cần ít nhất một câu hỏi.", file=sys.stderr)
        return 2

    try:
        ablation = parse_ablation(args.ablation)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    settings = get_settings()
    settings.ensure_dirs()

    try:
        pipeline = build_pipeline(args.mock, settings)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"Không khởi tạo được pipeline: {exc}"}, ensure_ascii=False))
        return 3

    try:
        if args.system == "B0":
            result = pipeline.answer_llm_only(question, system="B0")
        elif args.system == "B1":
            result = pipeline.answer_vanilla_rag(question, system="B1")
        elif args.system == "B2":
            result = pipeline.answer_rag_plus_graph(question, system="B2")
        else:
            result = pipeline.ask(
                question,
                top_k=args.top_k,
                max_hops=args.max_hops,
                ablation=ablation,
                system="ours",
            )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"Hỏi đáp thất bại: {exc}"}, ensure_ascii=False))
        return 4

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_text(result, args.show_evidence, args.explain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
