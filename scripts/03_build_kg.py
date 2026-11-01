#!/usr/bin/env python
"""Step 3 of the pipeline: build the knowledge graph from the corpus.

Usage
-----
    python scripts/03_build_kg.py --limit 30 --mock --reset
    python scripts/03_build_kg.py --corpus data/processed/corpus.jsonl --store inmemory
    python scripts/03_build_kg.py --no-extract --out artifacts/kg_stats.json

The script prints a JSON summary on stdout and never crashes because of a
single failing chunk.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.config import get_settings  # noqa: E402
from hgrag.data.records import Chunk  # noqa: E402
from hgrag.kg.alias import AliasResolver  # noqa: E402
from hgrag.kg.builder import KGBuilder  # noqa: E402
from hgrag.kg.extractor import EventExtractor  # noqa: E402
from hgrag.kg.ontology import ONTOLOGY  # noqa: E402
from hgrag.kg.store import InMemoryGraphStore, Neo4jGraphStore  # noqa: E402
from hgrag.llm.client import MockLLMClient, build_llm_client  # noqa: E402
from hgrag.retrieval.embeddings import build_embedder  # noqa: E402

logger = logging.getLogger("03_build_kg")

DEFAULT_CORPUS = "data/processed/corpus.jsonl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the KG (Neo4j hoặc in-memory)")
    p.add_argument("--corpus", type=str, default=DEFAULT_CORPUS, help="đường dẫn corpus.jsonl")
    p.add_argument("--limit", type=int, default=None, help="giới hạn số chunk xử lý")
    p.add_argument(
        "--store",
        type=str,
        default="neo4j",
        choices=("neo4j", "inmemory"),
        help="backend lưu trữ (default: neo4j)",
    )
    p.add_argument(
        "--extract",
        dest="extract",
        action="store_true",
        default=True,
        help="chạy trích xuất event bằng LLM (mặc định)",
    )
    p.add_argument(
        "--no-extract",
        dest="extract",
        action="store_false",
        help="không gọi LLM (dùng cache trích xuất nếu có)",
    )
    p.add_argument("--mock", action="store_true", help="ép dùng MockLLMClient (offline)")
    p.add_argument("--reset", action="store_true", help="xoá dữ liệu cũ trong store trước khi build")
    p.add_argument("--out", type=str, default=None, help="ghi stats ra file JSON")
    p.add_argument("--no-resolve", action="store_true", help="bỏ qua bước định danh thực thể")
    p.add_argument(
        "--offline",
        action="store_true",
        help="không gọi mạng (Wikipedia/Wikidata) khi định danh thực thể",
    )
    p.add_argument(
        "--online",
        dest="online",
        action="store_true",
        help="cho phép gọi Wikipedia/Wikidata khi định danh thực thể (chậm hơn)",
    )
    p.add_argument(
        "--live",
        dest="live",
        action="store_true",
        help="bật tra cứu trực tiếp Wikipedia/Wikidata cho mỗi mention (rất chậm)",
    )
    p.add_argument("--verbose", action="store_true", help="bật log DEBUG")
    return p


def read_corpus(path: Path) -> Iterator[Chunk]:
    """Stream the chunks of a JSON-Lines corpus file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            try:
                yield Chunk.from_dict(payload)
            except (TypeError, ValueError):
                continue


def make_store(kind: str, settings: Any) -> Any:
    if kind == "inmemory":
        return InMemoryGraphStore()
    cfg = settings.neo4j
    return Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str(cfg.get("user", "neo4j")),
        password=settings.neo4j_password,
        database=str(cfg.get("database", "neo4j")),
        batch_size=int(cfg.get("batch_size", 500)),
    )


def reset_store(store: Any) -> None:
    """Delete every node and relationship of the store."""
    query = "MATCH (n) DETACH DELETE n"
    runner = getattr(store, "_run", None)
    if callable(runner):
        runner(query)
        logger.info("Đã xoá sạch dữ liệu cũ trong Neo4j")
        return
    if isinstance(store, InMemoryGraphStore):
        store.entities.clear()
        store.relations.clear()
        logger.info("Đã xoá sạch dữ liệu cũ trong InMemoryGraphStore")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    settings.ensure_dirs()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = REPO_ROOT / corpus_path
    if not corpus_path.exists():
        print(json.dumps({"error": f"Không tìm thấy corpus: {corpus_path}"}, ensure_ascii=False))
        return 2

    # -- LLM ------------------------------------------------------------
    if args.mock:
        llm_client = MockLLMClient(embed_dim=settings.embed_dim, seed=settings.seed)
    else:
        llm_client = build_llm_client(settings)

    # -- store ----------------------------------------------------------
    try:
        store = make_store(args.store, settings)
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"error": f"Không kết nối được Neo4j ({exc}); thử --store inmemory"},
                ensure_ascii=False,
            )
        )
        return 3

    if args.reset:
        try:
            reset_store(store)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reset store thất bại: %s", exc)

    # -- ontology / extractor / resolver --------------------------------
    ontology = ONTOLOGY
    extractor = EventExtractor(
        llm_client=llm_client,
        ontology=ontology,
        cache_dir=settings.artifacts / "extractions",
        max_events=int(settings.extraction.get("max_events_per_chunk", 4)),
    )

    resolver = None
    if not args.no_resolve:
        embedder = None
        try:
            embedder = build_embedder(settings, llm_client)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không khởi tạo được embedder: %s", exc)
        wikidata = None
        wikipedia = None
        # The Wikidata client is always created: even in "offline" mode the
        # anchor seeding needs the P31/P27/alias properties of the 10
        # documents' QIDs (a single batched request, cached on disk).
        try:
            from hgrag.data.wikidata import WikidataClient

            wikidata = WikidataClient(cache_dir=settings.cache_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không khởi tạo được client Wikidata: %s", exc)
        # The *live* lookups (Wikipedia redirect + Wikidata search) need one
        # HTTP request per mention.  During the bulk build there are
        # thousands of mentions and the Wikimedia API answers 429 long before
        # the build finishes, so they are opt-in (``--live``).  The alias
        # index built from the anchors already carries the Wikipedia redirect
        # titles, so the offline tiers cover the same ground.
        live = bool(args.live) and not args.offline
        if live:
            try:
                from hgrag.data.wikipedia import WikipediaClient

                wikipedia = WikipediaClient(cache_dir=settings.data_raw / "wiki")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Không khởi tạo được client Wikipedia: %s", exc)
        resolver = AliasResolver(
            store=store,
            llm_client=llm_client,
            embedder=embedder,
            wikidata=wikidata,
            wikipedia=wikipedia,
            settings=settings,
            offline=not live,
            ontology=ontology,
        )

    builder = KGBuilder(
        store=store,
        ontology=ontology,
        extractor=extractor if args.extract else None,
        resolver=resolver,
        settings=settings,
        llm_client=llm_client,
        wikidata=wikidata,
    )

    # -- build ----------------------------------------------------------
    chunks = list(read_corpus(corpus_path))
    stats = builder.build_from_chunks(chunks, max_chunks=args.limit, progress=True)

    payload: dict[str, Any] = {
        "corpus": str(corpus_path),
        "n_chunks_total": len(chunks),
        "limit": args.limit,
        "store": args.store,
        "mock": bool(args.mock),
        "extract": bool(args.extract),
        "online_resolve": bool(args.online) and not args.offline,
        "live_resolve": bool(args.live) and not args.offline,
        "ontology": {
            "source": ontology.source,
            "n_event_types": len(ontology.event_types),
            "n_roles": len(ontology.roles),
        },
        "stats": stats.export(),
        "store_stats": builder.export_stats().get("store", {}),
        "llm_calls": getattr(extractor, "n_llm_calls", 0),
        "cache_hits": getattr(extractor, "n_cache_hits", 0),
    }

    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        payload["out"] = str(out_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        store.close()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
