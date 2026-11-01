"""Shared pytest fixtures for the Vietnamese History GraphRAG test-suite.

Everything here is **offline by default**: an autouse fixture blocks socket
creation so a stray network call fails loudly instead of hitting the internet.
Tests that really need the network are marked ``@pytest.mark.network`` and
opt back in.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from hgrag.config import Settings
from hgrag.data.records import Chunk, Entity, Relation
from hgrag.kg.store import InMemoryGraphStore
from hgrag.llm.client import MockLLMClient
from hgrag.retrieval.embeddings import MockEmbedder
from hgrag.retrieval.vector_store import NumpyVectorStore

# --------------------------------------------------------------------------
# network guard
# --------------------------------------------------------------------------


class _BlockedSocket(socket.socket):
    """A socket that refuses to connect (network blocked in tests)."""

    def connect(self, *args: Any, **kwargs: Any):  # noqa: D102
        raise RuntimeError(
            "Network access is blocked in tests. Mark the test with @pytest.mark.network "
            "to allow it."
        )

    def connect_ex(self, *args: Any, **kwargs: Any):  # noqa: D102
        raise RuntimeError("Network access is blocked in tests.")


@pytest.fixture(autouse=True)
def block_network(request: pytest.FixtureRequest):
    """Chặn network trong test mặc định; bỏ qua với marker ``network``."""
    if request.node.get_closest_marker("network") is not None:
        yield
        return

    original_socket = socket.socket
    original_create = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo

    def _blocked(*args: Any, **kwargs: Any):
        raise RuntimeError("Network access is blocked in tests.")

    socket.socket = _BlockedSocket  # type: ignore[misc,assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[misc,assignment]
        socket.create_connection = original_create  # type: ignore[assignment]
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

DEFAULT_RAW: dict[str, Any] = {
    "paths": {
        "data_raw": "data/raw",
        "data_processed": "data/processed",
        "artifacts": "artifacts",
        "reports": "reports",
        "cache": "artifacts/llm_cache",
    },
    "llm": {
        "provider": "mock",
        "model": "gemini-3.5-flash",
        "embed_model": "gemini-embedding-001",
        "temperature": 0.0,
        "max_output_tokens": 2048,
        "request_timeout": 30,
        "max_retries": 2,
        "retry_base_delay": 0.01,
        "cache_enabled": True,
    },
    "embedding": {"provider": "mock", "dim": 64, "batch_size": 16, "local_model": ""},
    "neo4j": {
        "uri": "bolt://localhost:7687",
        "http_uri": "http://localhost:7474",
        "user": "neo4j",
        "default_password": "history2025",
        "database": "neo4j",
        "batch_size": 100,
    },
    "qdrant": {
        "url": "http://localhost:6333",
        "collection_chunks": "chunks",
        "collection_nodes": "nodes",
        "vector_size": 64,
        "distance": "Cosine",
    },
    "ingest": {
        "request_delay": 0.0,
        "wikipedia_max_articles": 5,
        "wikipedia_expand_rounds": 0,
        "wikipedia_min_chars": 100,
        "history_chat_limit": 5,
        "user_agent": "test/0.1",
    },
    "chunking": {"max_chars": 400, "overlap_chars": 50, "min_chars": 60},
    "extraction": {"max_chunks": 10, "max_events_per_chunk": 4},
    "retrieval": {
        "vector_top_k": 10,
        "rerank_top_k": 5,
        "graph_max_hops": 2,
        "graph_max_paths": 10,
        "rrf_k": 60,
        "context_max_tokens": 2000,
        "max_evidence": 6,
        "max_evidence_per_doc": 2,
        "temporal_boost": 0.15,
        "entity_link_min_confidence": 0.7,
    },
    "evaluation": {"eval_size_target": 10, "bootstrap_samples": 10, "bootstrap_ci": 0.95, "seed": 42},
}


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings rooted in ``tmp_path`` (không pollute ``artifacts/`` thật)."""
    raw = json.loads(json.dumps(DEFAULT_RAW))
    cfg = Settings(raw=raw, root=tmp_path)
    cfg.ensure_dirs()
    return cfg


# --------------------------------------------------------------------------
# LLM / embeddings
# --------------------------------------------------------------------------


@pytest.fixture()
def mock_llm() -> MockLLMClient:
    return MockLLMClient(embed_dim=64, seed=42)


@pytest.fixture()
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dim=64, seed=42)


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    Chunk(
        chunk_id="ck_tran_hung_dao_1",
        doc_id="wiki:Trần_Hưng_Đạo",
        title="Trần Hưng Đạo",
        text=(
            "Trần Hưng Đạo (tên thật Trần Quốc Tuấn) là chính trị gia, nhà chiến lược "
            "quân sự Đại Việt thời nhà Trần. Năm 1285, ông được giao chỉ huy quân chống "
            "quân Nguyên Mông xâm lược."
        ),
        url="https://vi.wikipedia.org/wiki/Trần_Hưng_Đạo",
        source="wikipedia",
        wikidata_id="Q511375",
        year_min=1284,
        year_max=1288,
    ),
    Chunk(
        chunk_id="ck_ly_thuong_kiet_1",
        doc_id="wiki:Lý_Thường_Kiệt",
        title="Lý Thường Kiệt",
        text=(
            "Lý Thường Kiệt (1023–1072) là quan, nhà chính trị và nhà chiến lược quân sự "
            "thời nhà Lý. Ông được ban chức Quốc công tiết chế, tổ chức quân độ và cải tổ "
            "chính sách."
        ),
        url="https://vi.wikipedia.org/wiki/Lý_Thường_Kiệt",
        source="wikipedia",
        wikidata_id="Q708388",
        year_min=1044,
        year_max=1072,
    ),
    Chunk(
        chunk_id="ck_ngo_quyen_1",
        doc_id="wiki:Ngô_Quyền",
        title="Ngô Quyền",
        text=(
            "Ngô Quyền (895–941) là một nhà chính trị và nhà chỉ huy quân sự, người lập ra "
            "nhà Ngô sau khi đánh tan chế độ nhà Kiều Công Tiễn."
        ),
        url="https://vi.wikipedia.org/wiki/Ngô_Quyền",
        source="wikipedia",
        wikidata_id="Q312321",
        year_min=938,
        year_max=941,
    ),
]


@pytest.fixture()
def tmp_corpus(tmp_path: Path) -> Path:
    """Ghi vài Chunk JSONL vào ``tmp_path`` và trả về path của file."""
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for chunk in SAMPLE_CHUNKS:
            fh.write(chunk.to_json())
            fh.write("\n")
    return path


# --------------------------------------------------------------------------
# graph store
# --------------------------------------------------------------------------


@pytest.fixture()
def in_memory_graph() -> InMemoryGraphStore:
    """KG mẫu về lịch sử Việt Nam (Trần Hưng Đạo, Lý Thường Kiệt, ...)."""
    store = InMemoryGraphStore()
    entities = [
        Entity(
            uid="pe:Q511375",
            name="Trần Hưng Đạo",
            entity_type="Person",
            wikidata_id="Q511375",
            aliases=["Trần Quốc Tuấn", "Hưng Đạo Đại Vương"],
            name_key="tran hung dao",
            description="Chính trị gia, nhà chiến lược quân sự thời nhà Trần",
            real_name="Trần Quốc Tuấn",
            title="Hưng Đạo Đại Vương",
            period="Nhà Trần",
            year_min=1228,
            year_max=1300,
            source="wikipedia",
            confidence=0.95,
            source_chunk_ids=["ck_tran_hung_dao_1"],
        ),
        Entity(
            uid="pe:Q708388",
            name="Lý Thường Kiệt",
            entity_type="Person",
            wikidata_id="Q708388",
            aliases=["Lý Thường Kiệt"],
            name_key="ly thuong kiet",
            description="Quan, nhà chính trị thời nhà Lý",
            period="Nhà Lý",
            year_min=1023,
            year_max=1072,
            source="wikipedia",
            confidence=0.95,
            source_chunk_ids=["ck_ly_thuong_kiet_1"],
        ),
        Entity(
            uid="pe:Q312321",
            name="Ngô Quyền",
            entity_type="Person",
            wikidata_id="Q312321",
            aliases=["Ngô Quyen"],
            name_key="ngo quyen",
            description="Nhà chính trị, lập ra nhà Ngô",
            period="Nhà Ngô",
            year_min=895,
            year_max=941,
            source="wikipedia",
            confidence=0.9,
            source_chunk_ids=["ck_ngo_quyen_1"],
        ),
        Entity(
            uid="pl:Q1017",
            name="Thăng Long",
            entity_type="Place",
            wikidata_id="Q1017",
            aliases=["Hà Nội", "Thăng Long"],
            name_key="thang long",
            description="Kinh đô của nhà Lý và nhà Trần",
            source="wikipedia",
            confidence=0.9,
        ),
        Entity(
            uid="ev:wiki:Tran_Hung_Dao:chunk#1",
            name="Trận Bạch Đằng (1285)",
            entity_type="Event",
            description="Trận đánh bại quân Nguyên Mông trên sông Bạch Đằng",
            year_min=1285,
            year_max=1285,
            source="wikipedia",
            confidence=0.9,
            source_chunk_ids=["ck_tran_hung_dao_1"],
        ),
        Entity(
            uid="per:Nhà Trần",
            name="Nhà Trần",
            entity_type="Period",
            name_key="nha tran",
            description="Triều đại nhà Trần (1225–1400)",
            year_min=1225,
            year_max=1400,
            source="wikipedia",
            confidence=0.9,
        ),
    ]
    relations = [
        Relation(
            uid="r1",
            src="pe:Q511375",
            dst="ev:wiki:Tran_Hung_Dao:chunk#1",
            rel_type="PARTICIPATED_IN",
            text="Trần Hưng Đạo chỉ huy quân chống Nguyên Mông tại trận Bạch Đằng năm 1285",
            year=1285,
            source="wikipedia",
            confidence=0.95,
            source_chunk_ids=["ck_tran_hung_dao_1"],
        ),
        Relation(
            uid="r2",
            src="pe:Q511375",
            dst="pl:Q1017",
            rel_type="OCCURRED_AT",
            text="Trận Bạch Đằng diễn ra gần Thăng Long",
            year=1285,
            source="wikipedia",
            confidence=0.9,
            source_chunk_ids=["ck_tran_hung_dao_1"],
        ),
        Relation(
            uid="r3",
            src="pe:Q511375",
            dst="per:Nhà Trần",
            rel_type="BELONGED_TO_DYNASTY",
            text="Trần Hưng Đạo là công thần nhà Trần",
            year=1284,
            source="wikipedia",
            confidence=0.95,
        ),
        Relation(
            uid="r4",
            src="ev:wiki:Tran_Hung_Dao:chunk#1",
            dst="per:Nhà Trần",
            rel_type="OCCURRED_IN",
            text="Trận Bạch Đằng xảy ra trong thời nhà Trần",
            year=1285,
            source="wikipedia",
            confidence=0.9,
        ),
        Relation(
            uid="r5",
            src="pe:Q708388",
            dst="pl:Q1017",
            rel_type="OCCURRED_AT",
            text="Lý Thường Kiệt tổ chức quân độ tại Thăng Long",
            year=1069,
            source="wikipedia",
            confidence=0.9,
            source_chunk_ids=["ck_ly_thuong_kiet_1"],
        ),
        Relation(
            uid="r6",
            src="pe:Q312321",
            dst="pl:Q1017",
            rel_type="OCCURRED_AT",
            text="Ngô Quyền đóng đô ở Thăng Long",
            year=939,
            source="wikipedia",
            confidence=0.9,
            source_chunk_ids=["ck_ngo_quyen_1"],
        ),
    ]
    store.upsert_entities(entities)
    store.upsert_relations(relations)
    return store


# --------------------------------------------------------------------------
# vector store
# --------------------------------------------------------------------------


@pytest.fixture()
def mock_vector_store(mock_embedder: MockEmbedder) -> NumpyVectorStore:
    """Vector store numpy đã nạp sẵn embedding của corpus mẫu."""
    store = NumpyVectorStore(vector_size=mock_embedder.dim)
    store.ensure_collection("chunks", mock_embedder.dim)
    vectors = mock_embedder.embed([c.text for c in SAMPLE_CHUNKS])
    store.upsert(
        "chunks",
        [c.chunk_id for c in SAMPLE_CHUNKS],
        vectors,
        [
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "text": c.text,
                "year_min": c.year_min,
                "source": c.source,
            }
            for c in SAMPLE_CHUNKS
        ],
    )
    return store
