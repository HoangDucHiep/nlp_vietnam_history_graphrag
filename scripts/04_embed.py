#!/usr/bin/env python
"""Step 4 of the pipeline: embed the corpus chunks into the vector store.

Usage
-----
    python scripts/04_embed.py --corpus data/processed/corpus.jsonl --collection chunks
    python scripts/04_embed.py --limit 30 --mock
    python scripts/04_embed.py --store numpy --out artifacts/embed_stats.json

The script prints a JSON summary on stdout.  It is idempotent: the chunk ids
are stable, so re-running it only refreshes the vectors that changed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.config import get_settings  # noqa: E402
from hgrag.data.records import Chunk  # noqa: E402
from hgrag.retrieval.embeddings import MockEmbedder, build_embedder  # noqa: E402
from hgrag.retrieval.vector_store import (  # noqa: E402
    NumpyVectorStore,
    QdrantVectorStore,
)

logger = logging.getLogger("04_embed")

DEFAULT_CORPUS = "data/processed/corpus.jsonl"
DEFAULT_COLLECTION = "chunks"

#: Namespace used to derive a deterministic UUID from a chunk id.  Qdrant
#: only accepts unsigned integers or UUIDs as point ids, while the chunk ids
#: of the corpus are strings such as ``ck_24731941d15b2235``.  The mapping is
#: stable (uuid5) so that re-running the script refreshes the same points.
POINT_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id of a chunk id."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, str(chunk_id)))



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed corpus chunks into the vector store")
    p.add_argument("--corpus", type=str, default=DEFAULT_CORPUS, help="đường dẫn corpus.jsonl")
    p.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="tên collection")
    p.add_argument("--dim", type=int, default=768, help="số chiều embedding")
    p.add_argument("--batch-size", type=int, default=64, help="số chunk/lần embed")
    p.add_argument(
        "--store",
        type=str,
        default="qdrant",
        choices=("qdrant", "numpy"),
        help="backend vector store (default: qdrant)",
    )
    p.add_argument(
        "--embedder",
        type=str,
        default="auto",
        choices=("auto", "gemini", "local", "mock"),
        help="nguồn embedding (auto: Gemini, fallback local khi hết quota)",
    )
    p.add_argument("--limit", type=int, default=None, help="giới hạn số chunk xử lý")
    p.add_argument("--out", type=str, default=None, help="ghi stats ra file JSON")
    p.add_argument("--mock", action="store_true", help="dùng MockEmbedder (offline)")
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


def build_text(chunk: Chunk) -> str:
    """The text that is embedded: title + section + body."""
    parts = [chunk.title or "", chunk.section or "", chunk.text or ""]
    return "\n".join(p for p in parts if p).strip()


def build_payload(chunk: Chunk) -> dict[str, Any]:
    """The Qdrant payload of a chunk (full provenance)."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section": chunk.section,
        "source": chunk.source,
        "url": chunk.url,
        "year_min": chunk.year_min,
        "year_max": chunk.year_max,
        "wikidata_id": chunk.wikidata_id,
        "text": chunk.text,
    }


def make_store(kind: str, settings: Any, dim: int) -> Any:
    if kind == "numpy":
        return NumpyVectorStore(vector_size=dim)
    try:
        return QdrantVectorStore(url=settings.qdrant_url, vector_size=dim)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không kết nối được Qdrant (%s) -> dùng NumPy fallback", exc)
        return NumpyVectorStore(vector_size=dim)


def _store_id(store: Any, chunk_id: str) -> str:
    """Point id of a chunk for the given store.

    Qdrant only accepts unsigned integers or UUIDs, so the string chunk ids
    are mapped to a deterministic UUID (uuid5).  The NumPy fallback keeps
    the readable chunk id.
    """
    if isinstance(store, NumpyVectorStore):
        return str(chunk_id)
    return point_id(chunk_id)


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

    dim = int(args.dim or settings.embed_dim)
    if args.mock:
        embedder: Any = MockEmbedder(dim=dim, seed=settings.seed)
    elif args.embedder == "mock":
        embedder = MockEmbedder(dim=dim, seed=settings.seed)
    elif args.embedder == "local":
        from hgrag.retrieval.embeddings import LocalEmbedder

        embedder = LocalEmbedder(str(settings.embedding.get("local_model", "")))
    elif args.embedder == "gemini":
        from hgrag.llm.client import build_llm_client
        from hgrag.retrieval.embeddings import LLMEmbedder

        embedder = LLMEmbedder(build_llm_client(settings))
    else:  # auto
        embedder = build_embedder(settings)
    dim = int(getattr(embedder, "dim", dim) or dim)

    # Mỗi embedder có một collection riêng: vector Gemini 768 chiều không
    # thể nằm chung collection với vector cục bộ 384 chiều.
    if args.collection == DEFAULT_COLLECTION and str(getattr(embedder, "name", "")).startswith(
        "local:"
    ):
        args.collection = str(settings.qdrant.get("collection_chunks_local", "chunks_local"))

    store = make_store(args.store, settings, dim)
    try:
        store.ensure_collection(args.collection, dim)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"Không tạo được collection: {exc}"}, ensure_ascii=False))
        return 3

    chunks = list(read_corpus(corpus_path))
    if args.limit is not None:
        chunks = chunks[: max(0, int(args.limit))]

    started = time.time()
    batch_size = max(1, int(args.batch_size))
    n_embedded = 0
    n_batches = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [build_text(c) for c in batch]
        vectors = embedder.embed(texts)
        store.upsert(
            args.collection,
            [_store_id(store, c.chunk_id) for c in batch],
            vectors,
            [build_payload(c) for c in batch],
        )
        n_embedded += len(batch)
        n_batches += 1
        logger.info("Đã embed %d/%d chunk", n_embedded, len(chunks))
    elapsed = time.time() - started

    try:
        n_points = int(store.count(args.collection))
    except Exception:  # noqa: BLE001
        n_points = n_embedded

    payload: dict[str, Any] = {
        "corpus": str(corpus_path),
        "collection": args.collection,
        "store": args.store,
        "dim": dim,
        "embedder": getattr(embedder, "name", "unknown"),
        "mock": bool(args.mock),
        "n_chunks": n_embedded,
        "n_batches": n_batches,
        "batch_size": batch_size,
        "n_points_in_collection": n_points,
        "elapsed": round(elapsed, 3),
    }

    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["out"] = str(out_path)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        store.close()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
