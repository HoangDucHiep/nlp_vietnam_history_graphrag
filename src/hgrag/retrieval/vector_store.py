"""Vector store protocol + Qdrant implementation (with NumPy fallback)."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Minimal vector-store interface used by the retrieval layer."""

    def ensure_collection(self, name: str, dim: int) -> None: ...

    def upsert(
        self, name: str, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]
    ) -> None: ...

    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def count(self, name: str) -> int: ...

    def close(self) -> None: ...


def _qdrant_filter(filters: dict[str, Any] | None) -> Any:
    """Translate a simple flat filter dict into a Qdrant filter."""
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    must = []
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            must.append(FieldCondition(key=key, match=MatchAny(any=list(value))))
        else:
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=must)


class QdrantVectorStore:
    """Qdrant-backed :class:`VectorStore`."""

    def __init__(self, url: str = "http://localhost:6333", vector_size: int = 768):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url, timeout=30)
        self.vector_size = vector_size

    def ensure_collection(self, name: str, dim: int | None = None) -> None:
        from qdrant_client.models import Distance, VectorParams

        dim = dim or self.vector_size
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Đã tạo Qdrant collection '%s' (dim=%d)", name, dim)

    def upsert(
        self, name: str, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]
    ) -> None:
        from qdrant_client.models import PointStruct

        if not ids:
            return
        points = [
            PointStruct(id=pid, vector=vec, payload=pl)
            for pid, vec, pl in zip(ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=name, points=points)

    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import PointStruct  # noqa: F401  (ensure import works)

        resp = self.client.query_points(
            collection_name=name,
            query=query,
            limit=top_k,
            query_filter=_qdrant_filter(filters),
            with_payload=True,
        )
        return [
            {"id": str(p.id), "score": float(p.score), "payload": p.payload or {}}
            for p in resp.points
        ]

    def count(self, name: str) -> int:
        try:
            return int(self.client.count(collection_name=name, exact=True).count)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            return 0

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass


class NumpyVectorStore:
    """Offline fallback: brute-force cosine similarity over an in-memory index."""

    def __init__(self, vector_size: int = 768):
        self.vector_size = vector_size
        self.collections: dict[str, dict[str, Any]] = {}

    def _col(self, name: str) -> dict[str, Any]:
        return self.collections.setdefault(name, {"ids": [], "vectors": [], "payloads": []})

    def ensure_collection(self, name: str, dim: int | None = None) -> None:
        self._col(name)

    def upsert(
        self, name: str, ids: list[str], vectors: list[list[float]], payloads: list[dict[str, Any]]
    ) -> None:
        import numpy as np

        col = self._col(name)
        index = {pid: i for i, pid in enumerate(col["ids"])}
        for pid, vec, pl in zip(ids, vectors, payloads, strict=True):
            if pid in index:
                i = index[pid]
                col["vectors"][i] = np.asarray(vec, dtype="float32")
                col["payloads"][i] = pl
            else:
                col["ids"].append(pid)
                col["vectors"].append(np.asarray(vec, dtype="float32"))
                col["payloads"].append(pl)

    def search(
        self,
        name: str,
        query: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        import numpy as np

        col = self._col(name)
        if not col["ids"]:
            return []
        mat = np.vstack(col["vectors"])
        q = np.asarray(query, dtype="float32")
        qn = np.linalg.norm(q) or 1.0
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1e-9
        sims = (mat @ q) / (norms * qn)
        order = np.argsort(-sims)[: top_k * 5]
        out: list[dict[str, Any]] = []
        for i in order:
            pl = col["payloads"][i]
            if filters and not all(pl.get(k) == v for k, v in filters.items()):
                continue
            out.append({"id": col["ids"][i], "score": float(sims[i]), "payload": pl})
            if len(out) >= top_k:
                break
        return out

    def count(self, name: str) -> int:
        return len(self._col(name)["ids"])

    def close(self) -> None:  # pragma: no cover - trivial
        return None


def build_vector_store(settings: Any | None = None, prefer_remote: bool = True) -> VectorStore:
    """Build a Qdrant store if reachable, otherwise fall back to NumPy."""
    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    qc = settings.qdrant
    if prefer_remote:
        try:
            store = QdrantVectorStore(url=settings.qdrant_url, vector_size=qc.get("vector_size", 768))
            store.client.get_collections()
            return store
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không kết nối được Qdrant (%s) -> dùng NumPy fallback", exc)
    return NumpyVectorStore(vector_size=qc.get("vector_size", 768))


def build_graph_store(settings: Any | None = None, prefer_remote: bool = True) -> Any:
    """Build a Neo4j store if reachable, otherwise fall back to in-memory."""
    if settings is None:
        from ..config import get_settings

        settings = get_settings()
    if prefer_remote:
        try:
            from ..kg.store import Neo4jGraphStore

            nc = settings.neo4j
            return Neo4jGraphStore(
                uri=settings.neo4j_uri,
                user=nc.get("user", "neo4j"),
                password=settings.neo4j_password,
                database=nc.get("database", "neo4j"),
                batch_size=nc.get("batch_size", 500),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không kết nối được Neo4j (%s) -> dùng InMemory fallback", exc)
    from ..kg.store import InMemoryGraphStore

    return InMemoryGraphStore()
