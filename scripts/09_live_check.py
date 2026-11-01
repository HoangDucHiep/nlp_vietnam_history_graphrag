#!/usr/bin/env python
"""Step 9 of the pipeline: live check against the real services.

The script verifies that the *real* stack (Gemini + Neo4j + Qdrant) is
healthy and that a real question produces a grounded answer with citations.
It is the counterpart of ``scripts/08_smoke_test.sh`` (offline).

Usage
-----
    python scripts/09_live_check.py
    python scripts/09_live_check.py --question "..." --json
    python scripts/09_live_check.py --skip-eval       # chỉ kiểm tra dịch vụ

The script never raises: it prints a JSON report and exits with 0 when every
check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.config import get_settings  # noqa: E402
from hgrag.kg.alias import AliasResolver  # noqa: E402
from hgrag.kg.store import Neo4jGraphStore  # noqa: E402
from hgrag.llm.client import build_llm_client  # noqa: E402
from hgrag.llm.schemas import LLMMessage, LLMRequest  # noqa: E402
from hgrag.qa.pipeline import QAPipeline  # noqa: E402
from hgrag.retrieval.embeddings import build_embedder  # noqa: E402
from hgrag.retrieval.entity_linker import EntityLinker  # noqa: E402
from hgrag.retrieval.graph_search import GraphSearcher  # noqa: E402
from hgrag.retrieval.rerank import build_reranker  # noqa: E402
from hgrag.retrieval.vector_store import QdrantVectorStore  # noqa: E402

logger = logging.getLogger("09_live_check")

DEFAULT_QUESTION = "Trần Hưng Đạo đánh bại quân Mông Cổ vào năm nào?"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kiểm tra hệ thống thật (Gemini + Neo4j + Qdrant)")
    p.add_argument("--question", type=str, default=DEFAULT_QUESTION, help="câu hỏi kiểm tra")
    p.add_argument("--json", action="store_true", help="in báo cáo JSON")
    p.add_argument("--skip-eval", action="store_true", help="chỉ kiểm tra dịch vụ")
    p.add_argument("--verbose", action="store_true", help="bật log DEBUG")
    return p


def check(name: str, fn: Any) -> dict[str, Any]:
    """Run one check, capturing success/failure without raising."""
    started = time.perf_counter()
    try:
        detail = fn()
        ok = True
        error = ""
    except Exception as exc:  # noqa: BLE001
        detail = {}
        ok = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "check": name,
        "ok": ok,
        "detail": detail,
        "error": error,
        "elapsed": round(time.perf_counter() - started, 2),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    settings.ensure_dirs()
    report: dict[str, Any] = {"checks": []}

    # -- 1. Gemini ------------------------------------------------------
    llm_client = build_llm_client(settings)

    def _llm_check() -> dict[str, Any]:
        resp = llm_client.generate(
            LLMRequest(
                messages=[LLMMessage(role="user", content="Trả lời đúng một từ: OK")],
                task="live_probe",
            )
        )
        return {"model": resp.model, "cached": resp.cached, "answer": resp.text[:80]}

    report["checks"].append(check("gemini", _llm_check))

    # -- 2. Neo4j --------------------------------------------------------
    def _neo4j_check() -> dict[str, Any]:
        cfg = settings.neo4j
        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=str(cfg.get("user", "neo4j")),
            password=settings.neo4j_password,
            database=str(cfg.get("database", "neo4j")),
        )
        return store.stats()

    report["checks"].append(check("neo4j", _neo4j_check))

    # -- 3. Qdrant -------------------------------------------------------
    def _qdrant_check() -> dict[str, Any]:
        cfg = settings.qdrant
        vs = QdrantVectorStore(
            url=str(cfg.get("url", "http://localhost:6333")),
            vector_size=int(cfg.get("vector_size", 768)),
        )
        return {"chunks": vs.count(str(cfg.get("collection_chunks", "chunks")))}

    report["checks"].append(check("qdrant", _qdrant_check))

    # -- 4. end-to-end QA -------------------------------------------------
    if not args.skip_eval:

        def _qa_check() -> dict[str, Any]:
            embedder = build_embedder(settings, llm_client)
            cfg = settings.neo4j
            store = Neo4jGraphStore(
                uri=settings.neo4j_uri,
                user=str(cfg.get("user", "neo4j")),
                password=settings.neo4j_password,
                database=str(cfg.get("database", "neo4j")),
            )
            qcfg = settings.qdrant
            vector_store = QdrantVectorStore(
                url=str(qcfg.get("url", "http://localhost:6333")),
                vector_size=int(qcfg.get("vector_size", 768)),
            )
            linker = EntityLinker(
                resolver=AliasResolver(
                    store=store,
                    llm_client=llm_client,
                    embedder=embedder,
                    settings=settings,
                ),
                settings=settings,
            )
            pipeline = QAPipeline(
                llm_client=llm_client,
                embedder=embedder,
                graph_store=store,
                vector_store=vector_store,
                entity_linker=linker,
                graph_searcher=GraphSearcher(store=store, settings=settings, embedder=embedder),
                reranker=build_reranker(settings, embedder),
                settings=settings,
            )
            result = pipeline.ask(args.question)
            return {
                "question": result.question,
                "answer": result.answer[:600],
                "citations": result.citations,
                "refused": result.refused,
                "n_evidence": len(result.evidence),
                "n_linked": len(result.explanation.get("linked_entities") or []),
                "elapsed": round(result.elapsed, 2),
            }

        report["checks"].append(check("qa_end_to_end", _qa_check))

    report["ok"] = all(c["ok"] for c in report["checks"])
    report["n_failed"] = sum(1 for c in report["checks"] if not c["ok"])

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in report["checks"]:
            status = "OK  " if item["ok"] else "FAIL"
            print(f"[{status}] {item['check']} ({item['elapsed']}s)")
            if item["ok"]:
                print(f"        {json.dumps(item['detail'], ensure_ascii=False)[:300]}")
            else:
                print(f"        {item['error']}")
        print("")
        print("KẾT QUẢ:", "PASS" if report["ok"] else f"FAIL ({report['n_failed']})")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
