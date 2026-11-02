"""Offline tests for the retrieval layer (Wave 3).

Everything here runs without network access: the entity resolver is a fake,
the graph is an :class:`~hgrag.kg.store.InMemoryGraphStore` and the
embedder is the deterministic :class:`~hgrag.retrieval.embeddings.MockEmbedder`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from hgrag.data.records import Chunk, Entity, Evidence, Relation
from hgrag.kg.store import InMemoryGraphStore
from hgrag.retrieval.context import ContextBuilder, estimate_tokens, render_for_prompt
from hgrag.retrieval.embeddings import MockEmbedder
from hgrag.retrieval.entity_linker import (
    EntityLinker,
    extract_vietnamese_mentions,
)
from hgrag.retrieval.fusion import (
    apply_diversity_cap,
    apply_temporal_boost,
    fuse,
    reciprocal_rank_fusion,
)
from hgrag.retrieval.graph_search import GraphSearcher, extract_subgraph
from hgrag.retrieval.rerank import HeuristicReranker

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


class FakeCandidate:
    """Mimics :class:`hgrag.kg.alias.EntityCandidate`."""

    def __init__(
        self,
        uid: str,
        name: str,
        score: float,
        tier: int,
        wikidata_id: str = "",
        entity_type: str = "Person",
        matched_alias: str = "",
        reason: str = "",
    ) -> None:
        self.uid = uid
        self.name = name
        self.score = score
        self.tier = tier
        self.wikidata_id = wikidata_id
        self.entity_type = entity_type
        self.matched_alias = matched_alias
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "wikidata_id": self.wikidata_id,
            "entity_type": self.entity_type,
            "score": self.score,
            "tier": self.tier,
            "matched_alias": self.matched_alias,
            "reason": self.reason,
        }


class FakeResolver:
    """Offline stand-in for :class:`hgrag.kg.alias.AliasResolver`.

    It encodes the two hard cases of ``docs/07``:

    * ``Trần Quốc Tuấn`` -> ``Trần Hưng Đạo`` (same person, alias hit);
    * ``Thụy Ba Công chúa`` -> **not** Trần Hưng Đạo (kinship guard).
    """

    def __init__(self, store: InMemoryGraphStore | None = None) -> None:
        self.store = store
        self.records: dict[str, Any] = {}
        self.calls: list[str] = []

    def resolve(self, mention: str, hints: dict[str, Any] | None = None) -> list[FakeCandidate]:
        self.calls.append(mention)
        key = mention.strip().casefold()
        if key in ("trần quốc tuấn", "trần hưng đạo", "hưng đạo đại vương"):
            return [
                FakeCandidate(
                    "pe:Q511375",
                    "Trần Hưng Đạo",
                    0.97,
                    1,
                    wikidata_id="Q511375",
                    matched_alias=mention,
                    reason="T1: alias Wikidata",
                ),
                FakeCandidate(
                    "pe:Q999999",
                    "Trần Hưng Đạo (khác)",
                    0.62,
                    5,
                    wikidata_id="Q999999",
                    reason="T5: embedding top-k",
                ),
            ]
        if key == "ngô quyền":
            return [
                FakeCandidate(
                    "pe:Q312321",
                    "Ngô Quyền",
                    0.95,
                    1,
                    wikidata_id="Q312321",
                    matched_alias="Ngô Quyền",
                    reason="T1: alias Wikidata",
                )
            ]
        if key == "trận bạch đằng":
            return [
                FakeCandidate(
                    "ev:bach_dang",
                    "Trận Bạch Đằng",
                    0.93,
                    2,
                    entity_type="Event",
                    matched_alias="Trận Bạch Đằng",
                    reason="T2: Wikipedia redirect",
                )
            ]
        # "Thụy Ba Công chúa" và các mention không rõ -> không khớp
        return []


@pytest.fixture()
def fake_resolver() -> FakeResolver:
    return FakeResolver()


@pytest.fixture()
def small_graph() -> InMemoryGraphStore:
    """Đồ thị nhỏ: Trần Hưng Đạo -> Trận Bạch Đằng -> Văn Yên."""
    store = InMemoryGraphStore()
    store.upsert_entities(
        [
            Entity(
                uid="pe:Q511375",
                name="Trần Hưng Đạo",
                entity_type="Person",
                wikidata_id="Q511375",
                aliases=["Trần Quốc Tuấn"],
                name_key="tran hung dao",
                year_min=1228,
                year_max=1300,
            ),
            Entity(
                uid="ev:bach_dang",
                name="Trận Bạch Đằng",
                entity_type="Event",
                name_key="tran bach dang",
                year_min=1285,
                year_max=1285,
            ),
            Entity(
                uid="pl:van_yen",
                name="Văn Yên",
                entity_type="Place",
                name_key="van yen",
                year_min=1284,
                year_max=1285,
            ),
            Entity(
                uid="ev:ky_hiep",
                name="Trận Tốt Động - Chúc Động",
                entity_type="Event",
                year_min=1426,
                year_max=1427,
            ),
        ]
    )
    store.upsert_relations(
        [
            Relation(
                uid="r1",
                src="pe:Q511375",
                dst="ev:bach_dang",
                rel_type="PARTICIPATED_IN",
                text="Trần Hưng Đạo chỉ huy quân tại trận Bạch Đằng năm 1285",
                year=1285,
                confidence=0.95,
                source_chunk_ids=["ck_1"],
            ),
            Relation(
                uid="r2",
                src="ev:bach_dang",
                dst="pl:van_yen",
                rel_type="OCCURRED_AT",
                text="Trận Bạch Đằng diễn ra ở Văn Yên",
                year=1285,
                confidence=0.9,
                source_chunk_ids=["ck_2"],
            ),
            Relation(
                uid="r3",
                src="pe:Q511375",
                dst="ev:ky_hiep",
                rel_type="PARTICIPATED_IN",
                text="Tham gia trận Tốt Động - Chúc Động",
                year=1426,
                confidence=0.5,
                source_chunk_ids=["ck_3"],
            ),
        ]
    )
    return store


@pytest.fixture()
def event_hub_graph() -> InMemoryGraphStore:
    """Đồ thị theo mô hình *event hub* của VHE.

    ``agent -PARTICIPATED_IN-> event -ABOUT-> theme``

    Câu hỏi neo vào *theme* ("Ai là người thành lập Văn Lang?") trước đây
    không có đường đi nào, vì cạnh ``ABOUT`` trỏ theo hướng ngược lại.
    """
    store = InMemoryGraphStore()
    store.upsert_entities(
        [
            Entity(
                uid="pe:hung_vuong",
                name="Hùng Vương",
                entity_type="Person",
                name_key="hung vuong",
            ),
            Entity(
                uid="ev:thanh_lap_van_lang",
                name="Hùng Vương thành lập Văn Lang",
                entity_type="Event",
                name_key="hung vuong thanh lap van lang",
            ),
            Entity(
                uid="th:van_lang",
                name="Văn Lang",
                entity_type="Entity",
                name_key="van lang",
            ),
        ]
    )
    store.upsert_relations(
        [
            Relation(
                uid="r_part",
                src="pe:hung_vuong",
                dst="ev:thanh_lap_van_lang",
                rel_type="PARTICIPATED_IN",
                text="Hùng Vương thành lập Văn Lang",
                year=0,
                confidence=0.9,
                source_chunk_ids=["ck_vhe_1"],
            ),
            Relation(
                uid="r_about",
                src="ev:thanh_lap_van_lang",
                dst="th:van_lang",
                rel_type="ABOUT",
                text="đặt quốc hiệu là Văn Lang",
                year=0,
                confidence=0.9,
                source_chunk_ids=["ck_vhe_1"],
            ),
        ]
    )
    return store


# --------------------------------------------------------------------------
# entity linker
# --------------------------------------------------------------------------


def test_entity_linker_resolves_alias(fake_resolver: FakeResolver) -> None:
    linker = EntityLinker(fake_resolver)
    linked = linker.link(["Trần Quốc Tuấn"])
    assert len(linked) == 1
    assert linked[0].uid == "pe:Q511375"
    assert linked[0].wikidata_id == "Q511375"
    assert linked[0].confidence >= 0.7
    assert linked[0].tier == 1


def test_entity_linker_rejects_kinship_mismatch(fake_resolver: FakeResolver) -> None:
    linker = EntityLinker(fake_resolver)
    linked = linker.link(["Thụy Ba Công chúa"])
    assert linked == []


def test_entity_linker_dedupes_uids(fake_resolver: FakeResolver) -> None:
    linker = EntityLinker(fake_resolver)
    linked = linker.link(["Trần Hưng Đạo", "Trần Quốc Tuấn"])
    assert len(linked) == 1


def test_entity_linker_keeps_other_candidates(fake_resolver: FakeResolver) -> None:
    linker = EntityLinker(fake_resolver)
    linked = linker.link(["Trần Quốc Tuấn"])
    assert linked[0].candidates
    assert linked[0].candidates[0]["uid"] == "pe:Q999999"


def test_entity_linker_no_match_returns_empty(fake_resolver: FakeResolver) -> None:
    linker = EntityLinker(fake_resolver)
    assert linker.link(["Một thực thể hoàn toàn không tồn tại"]) == []


def test_entity_linker_ablation_only_tier0(fake_resolver: FakeResolver) -> None:
    linker = EntityLinker(fake_resolver, use_resolution=False)
    # T1 bị bỏ -> không match được
    assert linker.link(["Trần Quốc Tuấn"]) == []
    # T0 (QID) vẫn dùng được
    linked = linker.link(["Q511375"])
    assert len(linked) == 1
    assert linked[0].uid == "pe:Q511375"
    assert linked[0].tier == 0


def test_entity_linker_link_from_query_with_entities(fake_resolver: FakeResolver) -> None:
    linker = EntityLinker(fake_resolver)
    linked = linker.link_from_query("Ai chỉ huy trận Bạch Đằng?", entities=["Trần Quốc Tuấn"])
    assert [item.uid for item in linked] == ["pe:Q511375"]


def test_entity_linker_link_from_query_without_entities(fake_resolver: FakeResolver) -> None:
    linker = EntityLinker(fake_resolver)
    linked = linker.link_from_query("Vì sao Trần Hưng Đạo thắng trận Bạch Đằng?")
    uids = {item.uid for item in linked}
    assert "pe:Q511375" in uids


def test_extract_vietnamese_mentions() -> None:
    mentions = extract_vietnamese_mentions("Vì sao nhà Trần gả Trần Khánh Dư cho vua Đinh?")
    assert "Trần Khánh Dư" in mentions
    assert "Đinh" not in mentions


# --------------------------------------------------------------------------
# graph search
# --------------------------------------------------------------------------


def test_graph_searcher_finds_expected_path(small_graph: InMemoryGraphStore) -> None:
    searcher = GraphSearcher(small_graph, embedder=MockEmbedder(dim=64))
    result = searcher.search(["pe:Q511375"], max_hops=2, max_paths=10, query_text="Trần Hưng Đạo")
    assert result.paths
    texts = [p.to_text() for p in result.paths]
    assert any("Trận Bạch Đằng" in t for t in texts)
    assert any("Văn Yên" in t for t in texts)


def test_graph_searcher_respects_max_hops(small_graph: InMemoryGraphStore) -> None:
    searcher = GraphSearcher(small_graph)
    result = searcher.search(["pe:Q511375"], max_hops=1, max_paths=50)
    assert result.paths
    assert all(p.hops <= 1 for p in result.paths)


def test_graph_searcher_path_text_format(small_graph: InMemoryGraphStore) -> None:
    searcher = GraphSearcher(small_graph)
    result = searcher.search(["pe:Q511375"], max_hops=2, max_paths=5)
    path = result.paths[0]
    text = path.to_text()
    assert " -" in text and "-> " in text
    assert path.nodes[0]["name"] == "Trần Hưng Đạo"
    assert path.edges[0]["rel_type"]


def test_graph_searcher_max_paths(small_graph: InMemoryGraphStore) -> None:
    searcher = GraphSearcher(small_graph)
    result = searcher.search(["pe:Q511375"], max_hops=3, max_paths=2)
    assert len(result.paths) <= 2


def test_graph_searcher_follows_edge_direction(small_graph: InMemoryGraphStore) -> None:
    """Path phải theo chiều của cạnh, không đảo ngược."""
    searcher = GraphSearcher(small_graph)
    result = searcher.search(["pe:Q511375"], max_hops=3, max_paths=50)
    for path in result.paths:
        for index, edge in enumerate(path.edges):
            assert path.nodes[index]["uid"] == edge["src"]
            assert path.nodes[index + 1]["uid"] == edge["dst"]
    # "Văn Yên" chỉ được đi tới từ "Trận Bạch Đằng", không phải từ "Trần Hưng Đạo"
    first_steps = {p.edges[0]["dst"] for p in result.paths if p.edges}
    assert "pl:van_yen" not in first_steps


def test_graph_searcher_empty_anchor(small_graph: InMemoryGraphStore) -> None:
    searcher = GraphSearcher(small_graph)
    result = searcher.search([])
    assert result.paths == []
    assert result.anchor_uids == []


def test_graph_searcher_scores_are_sorted(small_graph: InMemoryGraphStore) -> None:
    searcher = GraphSearcher(small_graph, embedder=MockEmbedder(dim=64))
    result = searcher.search(
        ["pe:Q511375"],
        relation_plan=["PARTICIPATED_IN"],
        max_hops=3,
        query_text="Trần Hưng Đạo tham gia trận Bạch Đằng năm 1285",
    )
    scores = [p.score for p in result.paths]
    assert scores == sorted(scores, reverse=True)


def test_extract_subgraph(small_graph: InMemoryGraphStore) -> None:
    searcher = GraphSearcher(small_graph)
    result = searcher.search(["pe:Q511375"], max_hops=2, max_paths=5)
    graph = extract_subgraph(result.paths)
    assert graph["n_nodes"] >= 2
    assert graph["n_edges"] >= 1
    assert all("uid" in n for n in graph["nodes"])


# --------------------------------------------------------------------------
# graph search -- event hub ngược
# --------------------------------------------------------------------------


def test_graph_searcher_reaches_agent_from_theme(event_hub_graph: InMemoryGraphStore) -> None:
    """Neo vào *theme* vẫn phải tìm được *agent* qua event hub."""
    searcher = GraphSearcher(event_hub_graph)
    result = searcher.search(
        ["th:van_lang"], max_hops=3, max_paths=10, query_text="Ai là người thành lập Văn Lang?"
    )
    assert result.paths, "không tìm được đường đi nào từ theme"
    texts = [p.to_text() for p in result.paths]
    assert any("Hùng Vương" in t for t in texts)


def test_graph_searcher_theme_path_keeps_provenance(event_hub_graph: InMemoryGraphStore) -> None:
    """Đường đi đi ngược vẫn phải giữ được chunk nguồn của sự kiện."""
    searcher = GraphSearcher(event_hub_graph)
    result = searcher.search(["th:van_lang"], max_hops=3, max_paths=10)
    assert any("ck_vhe_1" in path.source_chunk_ids for path in result.paths)


def test_graph_searcher_theme_path_direction(event_hub_graph: InMemoryGraphStore) -> None:
    """Cạnh đi ngược được đảo chiều để path vẫn là chuỗi có hướng."""
    searcher = GraphSearcher(event_hub_graph)
    result = searcher.search(["th:van_lang"], max_hops=3, max_paths=10)
    for path in result.paths:
        for index, edge in enumerate(path.edges):
            assert path.nodes[index]["uid"] == edge["src"]
            assert path.nodes[index + 1]["uid"] == edge["dst"]


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------


def test_reciprocal_rank_fusion_orders_correctly() -> None:
    rankings = {
        "vector": [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.8}, {"id": "c", "score": 0.7}],
        "graph": [{"id": "c", "score": 0.95}, {"id": "b", "score": 0.5}, {"id": "d", "score": 0.1}],
    }
    fused = reciprocal_rank_fusion(rankings, k=60)
    ids = [item["id"] for item in fused]
    scores = {item["id"]: item["rrf_score"] for item in fused}
    # "c" đứng đầu danh sách vector và đầu danh sách graph -> điểm cao nhất
    assert ids[0] == "c"
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    # "b" xuất hiện ở cả hai danh sách -> phải vượt "a" (chỉ ở vector)
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["d"]
    assert set(ids) == {"a", "b", "c", "d"}
    assert fused[0]["vector_rank"] == 3
    assert fused[0]["graph_rank"] == 1


def test_reciprocal_rank_fusion_ranks_are_one_based() -> None:
    fused = reciprocal_rank_fusion({"vector": [{"id": "a", "score": 1.0}]}, k=60)
    assert fused[0]["vector_rank"] == 1
    assert fused[0]["rrf_score"] == pytest.approx(1 / 61)


def test_apply_temporal_boost() -> None:
    items = [
        {"id": "a", "rrf_score": 0.03, "text": "Năm 1285 diễn ra trận Bạch Đằng"},
        {"id": "b", "rrf_score": 0.031, "text": "Không có năm nào cả"},
    ]
    boosted = apply_temporal_boost(items, [1285], boost=0.15)
    assert boosted[0]["id"] == "a"
    assert boosted[0]["rrf_score"] > 0.03
    assert boosted[1]["temporal_match"] is False


def test_apply_diversity_cap() -> None:
    items = [
        {"id": f"c{i}", "rrf_score": 0.1 - i * 0.001, "payload": {"doc_id": "docA"}}
        for i in range(5)
    ] + [
        {"id": "other", "rrf_score": 0.09, "payload": {"doc_id": "docB"}},
    ]
    capped = apply_diversity_cap(items, max_per_doc=3)
    doc_a = [i for i in capped if i["payload"]["doc_id"] == "docA"]
    assert len(doc_a) == 3
    assert any(i["id"] == "other" for i in capped)


def test_fuse_combines_vector_and_graph(small_graph: InMemoryGraphStore) -> None:
    searcher = GraphSearcher(small_graph)
    result = searcher.search(["pe:Q511375"], max_hops=2, max_paths=3)
    vector_hits = [
        {"id": "ck_1", "score": 0.9, "payload": {"doc_id": "wiki:Trần_Hưng_Đạo", "text": "năm 1285"}},
        {"id": "ck_2", "score": 0.8, "payload": {"doc_id": "wiki:Trận_Bạch_Đằng", "text": "Văn Yên"}},
    ]
    fused = fuse(vector_hits, result.paths, k=60, year_refs=[1285], max_per_doc=3)
    assert len(fused.items) >= 2
    assert all("rrf_score" in item for item in fused.items)
    assert any(item["vector_rank"] for item in fused.items)
    assert any(item["graph_rank"] for item in fused.items)


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------


def _evidence(n: int, length: int = 400, doc_id: str = "doc1") -> Evidence:
    return Evidence(
        evidence_id=f"E{n}",
        kind="chunk",
        text="x" * length,
        score=1.0 / n,
        meta={"title": f"Doc {doc_id}", "section": "Thân thế", "source": "Wikipedia"},
    )


def test_context_builder_respects_budget(settings) -> None:
    builder = ContextBuilder(settings)
    evidence = [_evidence(i) for i in range(1, 11)]
    pack = builder.build_context(evidence, max_tokens=100)
    assert pack.truncated
    assert pack.n_tokens_estimate <= 200
    assert len(pack.evidence_ids) < len(evidence)


def test_context_builder_citations_are_contiguous(settings) -> None:
    builder = ContextBuilder(settings)
    evidence = [_evidence(i) for i in range(1, 6)]
    pack = builder.build_context(evidence, max_tokens=100000)
    assert pack.evidence_ids == ["E1", "E2", "E3", "E4", "E5"]
    assert set(pack.citations_map) == set(pack.evidence_ids)
    assert pack.text.startswith("[E1]")
    assert "[E5]" in pack.text


def test_context_builder_keeps_at_least_one(settings) -> None:
    builder = ContextBuilder(settings)
    pack = builder.build_context([_evidence(1, length=5000)], max_tokens=10)
    assert len(pack.evidence_ids) == 1
    assert pack.truncated


def test_context_builder_format_includes_source(settings) -> None:
    builder = ContextBuilder(settings)
    pack = builder.build_context([_evidence(1)], max_tokens=10000)
    assert "Nguồn: Wikipedia" in pack.text
    assert 'mục "Thân thế"' in pack.text


def test_render_for_prompt(settings) -> None:
    builder = ContextBuilder(settings)
    pack = builder.build_context([_evidence(1)], max_tokens=10000)
    rendered = render_for_prompt(pack)
    assert "=== CONTEXT ===" in rendered
    assert "[E1]" in rendered


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 30) == 10


# --------------------------------------------------------------------------
# rerank
# --------------------------------------------------------------------------


def test_heuristic_reranker_sorts_descending() -> None:
    reranker = HeuristicReranker()
    candidates = [
        {"id": "a", "text": "Trận Bạch Đằng diễn ra năm 1285 tại Văn Yên", "payload": {}},
        {"id": "b", "text": "Một đoạn văn hoàn toàn không liên quan", "payload": {}},
        {"id": "c", "text": "Trần Hưng Đạo chỉ huy quân đánh quân Nguyên", "payload": {}},
    ]
    ranked = reranker.rerank("Trận Bạch Đằng năm 1285", candidates, top_k=3)
    assert [c["id"] for c in ranked][0] == "a"
    scores = [c["rerank_score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_heuristic_reranker_respects_top_k() -> None:
    reranker = HeuristicReranker()
    candidates = [{"id": str(i), "text": f"đoạn {i}", "payload": {}} for i in range(10)]
    ranked = reranker.rerank("đoạn", candidates, top_k=3)
    assert len(ranked) == 3


def test_heuristic_reranker_empty_input() -> None:
    assert HeuristicReranker().rerank("q", []) == []


# --------------------------------------------------------------------------
# script 04_embed
# --------------------------------------------------------------------------


def test_script_04_embed_mock(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    with corpus.open("w", encoding="utf-8") as fh:
        for i in range(5):
            fh.write(
                json.dumps(
                    {
                        "chunk_id": f"ck_{i}",
                        "doc_id": f"wiki:Doc{i}",
                        "title": f"Doc {i}",
                        "section": "Thân thế",
                        "text": f"Nội dung chunk số {i} về lịch sử Việt Nam.",
                        "source": "wikipedia",
                        "year_min": 1285,
                        "year_max": 1288,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    out = tmp_path / "embed_stats.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "04_embed.py"),
            "--corpus",
            str(corpus),
            "--store",
            "numpy",
            "--collection",
            "chunks_test",
            "--dim",
            "768",
            "--limit",
            "20",
            "--mock",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    stats = json.loads(proc.stdout)
    assert stats["n_chunks"] == 5
    assert stats["dim"] == 768
    assert stats["collection"] == "chunks_test"
    assert stats["store"] == "numpy"
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["n_chunks"] == 5


def test_script_04_embed_builds_expected_text(tmp_path: Path) -> None:
    """Text nhúng phải chứa cả title, section và body."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "embed_script", REPO_ROOT / "scripts" / "04_embed.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(REPO_ROOT / "scripts"))

    chunk = Chunk(
        chunk_id="ck_1",
        doc_id="wiki:X",
        title="Trần Hưng Đạo",
        section="Thân thế",
        text="Nội dung.",
    )
    text = module.build_text(chunk)
    assert text == "Trần Hưng Đạo\nThân thế\nNội dung."
    payload = module.build_payload(chunk)
    assert payload["chunk_id"] == "ck_1"
    assert payload["title"] == "Trần Hưng Đạo"
    assert payload["section"] == "Thân thế"
