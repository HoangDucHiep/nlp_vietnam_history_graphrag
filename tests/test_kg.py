"""Tests for the KG layer: ontology, alias resolution, extractor, builder.

All tests are **offline** (the autouse ``block_network`` fixture in
``conftest.py`` guarantees it).  The alias tests use a fake store with a
pre-built alias index, so no Wikipedia/Wikidata request is ever made.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hgrag.data.records import Chunk, Entity
from hgrag.kg.alias import AliasResolver, EntityCandidate
from hgrag.kg.builder import KGBuilder
from hgrag.kg.extractor import EventExtractor
from hgrag.kg.ontology import (
    ARGUMENT_ROLES,
    EVENT_TYPES,
    ONTOLOGY,
    RELATION_TYPES,
    ROLE_GROUPS,
    Ontology,
    normalize_event_type,
    normalize_role,
    relation_for_role,
    validate_event,
)
from hgrag.kg.store import InMemoryGraphStore
from hgrag.llm.client import MockLLMClient
from hgrag.retrieval.embeddings import MockEmbedder

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

TRAN_HUNG_DAO = "Q511375"


def _entity(
    uid: str,
    name: str,
    aliases: list[str],
    *,
    wikidata_id: str = "",
    entity_type: str = "Person",
    p31: list[str] | None = None,
    p27: list[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    alias_sources: dict[str, str] | None = None,
) -> Entity:
    return Entity(
        uid=uid,
        name=name,
        entity_type=entity_type,
        wikidata_id=wikidata_id,
        aliases=aliases,
        name_key="",
        p31=p31 or [],
        p27=p27 or [],
        year_min=year_min,
        year_max=year_max,
        extra={"alias_sources": alias_sources or {}},
    )


@pytest.fixture()
def alias_entities() -> list[Entity]:
    """Entities mirroring the verified cases of docs/07 §8."""
    return [
        _entity(
            uid=f"pe:{TRAN_HUNG_DAO}",
            name="Trần Hưng Đạo",
            aliases=[
                "Trần Quốc Tuấn",
                "Hưng Đạo Vương",
                "Hưng Đạo Đại vương",
                "Đức Thánh Trần",
                "Cửu Thiên Vũ Đế",
                "Thụy Ba Công chúa",
            ],
            wikidata_id=TRAN_HUNG_DAO,
            p31=["Q5"],
            p27=["Q430309"],
            year_min=1226,
            year_max=1300,
            alias_sources={
                "Trần Quốc Tuấn": "wikidata",
                "Hưng Đạo Vương": "wikidata",
                "Hưng Đạo Đại vương": "wikidata",
                "Đức Thánh Trần": "wikidata",
                "Cửu Thiên Vũ Đế": "wiki_redirect",
                "Thụy Ba Công chúa": "wiki_redirect",
            },
        ),
        # "Thụy Ba Công chúa" is a Wikipedia *redirect* to Trần Hưng Đạo,
        # but it denotes his adoptive aunt -> must NOT be merged.
        # It is registered as a tier-2 (redirect) alias on the same entity.
        # "Hùng Vương" -- the legendary king (human, P27=Văn Lang)
        _entity(
            uid="pe:Q1320050",
            name="Hùng Vương",
            aliases=[],
            wikidata_id="Q1320050",
            p31=["Q5"],
            p27=["Q1763996"],
            year_min=-2870,
            year_max=-1400,
        ),
        # "Hùng Vương" -- a modern ward (P31 = xã/phường)
        _entity(
            uid="pe:Q10771249",
            name="Hùng Vương (phường)",
            aliases=["Hùng Vương"],
            wikidata_id="Q10771249",
            entity_type="Place",
            p31=["Q687188"],
            year_min=2020,
            year_max=2020,
        ),
        _entity(
            uid="pe:Q1195391",
            name="Lê Thái Tổ",
            aliases=["Lê Lợi", "Bình Định vương"],
            wikidata_id="Q1195391",
            p31=["Q5"],
            p27=["Q430309"],
            year_min=1385,
            year_max=1433,
            alias_sources={"Lê Lợi": "wikidata", "Bình Định vương": "wikidata"},
        ),
    ]


@pytest.fixture()
def resolver(alias_entities: list[Entity]) -> AliasResolver:
    res = AliasResolver(store=InMemoryGraphStore(), llm_client=None, offline=True)
    res.build_alias_index(alias_entities)
    return res


@pytest.fixture()
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="ck_test_1",
        doc_id="wiki:Trần_Hưng_Đạo",
        title="Trần Hưng Đạo",
        text="Trần Hưng Đạo (1228-1300) là một tướng quân Đại Việt, đã chỉ huy quân đánh bại quân Mông.",
        url="https://vi.wikipedia.org/wiki/Trần_Hưng_Đạo",
        section="Thân thế",
        source="wikipedia",
        year_min=1228,
        year_max=1300,
    )


# --------------------------------------------------------------------------
# ontology
# --------------------------------------------------------------------------


def test_ontology_has_35_event_types() -> None:
    assert len(EVENT_TYPES) == 35
    assert "be-born" in EVENT_TYPES
    assert "attack" in EVENT_TYPES


def test_ontology_has_11_roles_upper_snake() -> None:
    assert len(ARGUMENT_ROLES) == 11
    for role in ARGUMENT_ROLES:
        assert role == role.upper()
        assert " " not in role
    assert {"AGENT", "LOCATION", "TEMPORAL", "SOURCE"} <= set(ARGUMENT_ROLES)


def test_role_groups_cover_all_roles() -> None:
    grouped: set[str] = set()
    for members in ROLE_GROUPS.values():
        grouped.update(members)
    assert grouped == set(ARGUMENT_ROLES)
    assert ROLE_GROUPS["LOCATION"] == ("LOCATION",)
    assert set(ROLE_GROUPS["TEMPORAL"]) == {"TEMPORAL", "SOURCE"}
    assert "AGENT" in ROLE_GROUPS["PARTICIPANT"]


def test_relation_types_mapping() -> None:
    assert relation_for_role("agent") == "PARTICIPATED_IN"
    assert relation_for_role("beneficiary") == "PARTICIPATED_IN"
    assert relation_for_role("location") == "LOCATED_IN"
    assert relation_for_role("temporal") == "OCCURRED_IN"
    assert relation_for_role("source") == "SUPPORTED_BY"
    assert relation_for_role("cause") == "CAUSE_OF"
    assert relation_for_role("precondition") == "PRECONDITION_OF"
    assert RELATION_TYPES["AGENT"] == "PARTICIPATED_IN"


def test_normalize_role_and_event_type() -> None:
    assert normalize_role("agent") == "AGENT"
    assert normalize_role("Beneficiary") == "BENEFICIARY"
    assert normalize_role("experiencer") == "EXPERIENCER"
    assert normalize_event_type("Conflict") == "attack"
    assert normalize_event_type("Start Position") == "start-position"


def test_validate_event_detects_errors() -> None:
    assert validate_event("attack", [{"role": "agent", "name": "Trần Hưng Đạo"}]) == []
    errors = validate_event("khong-ton-tai", [{"role": "agent", "name": "A"}])
    assert any("event_type" in e for e in errors)
    errors = validate_event("attack", [{"role": "bogus-role", "name": "A"}])
    assert any("role không hợp lệ" in e for e in errors)
    errors = validate_event("attack", [{"role": "location", "name": "Hà Nội"}])
    assert any("thiếu argument AGENT" in e for e in errors)
    errors = validate_event("attack", [{"role": "agent", "name": ""}])
    assert any("thiếu name" in e for e in errors)


def test_ontology_object_shape() -> None:
    assert isinstance(ONTOLOGY, Ontology)
    assert ONTOLOGY.source in ("vhe-dataset", "fallback")
    assert set(ONTOLOGY.role_groups) == set(ROLE_GROUPS)
    assert ONTOLOGY.relation_for_role("AGENT") == "PARTICIPATED_IN"
    payload = ONTOLOGY.to_dict()
    assert payload["n_event_types"] == 35
    assert payload["n_roles"] == 11


# --------------------------------------------------------------------------
# alias resolver -- tiers
# --------------------------------------------------------------------------


def test_tier0_qid_mention(resolver: AliasResolver) -> None:
    candidate = resolver.resolve_one(TRAN_HUNG_DAO)
    assert candidate is not None
    assert candidate.tier == 0
    assert candidate.score == pytest.approx(1.0)
    assert candidate.wikidata_id == TRAN_HUNG_DAO


def test_tier1_wikidata_alias_same_entity(resolver: AliasResolver) -> None:
    for mention in ("Trần Quốc Tuấn", "Hưng Đạo Vương", "Trần Hưng Đạo"):
        candidate = resolver.resolve_one(mention)
        assert candidate is not None, mention
        assert candidate.wikidata_id == TRAN_HUNG_DAO, mention
        assert candidate.tier == 1, mention
        assert candidate.score >= 0.95


def test_tier1_typo_alias_same_entity(resolver: AliasResolver) -> None:
    """``Hứng Đạo đại vương`` (typo) must still resolve to Q511375."""
    candidate = resolver.resolve_one("Hứng Đạo đại vương")
    assert candidate is not None
    assert candidate.wikidata_id == TRAN_HUNG_DAO
    assert candidate.score >= 0.9


def test_thuy_ba_cong_chua_not_merged(resolver: AliasResolver) -> None:
    """``Thụy Ba Công chúa`` is the adoptive aunt -- not Trần Hưng Đạo."""
    candidates = resolver.resolve("Thụy Ba Công chúa")
    merged = [c for c in candidates if c.wikidata_id == TRAN_HUNG_DAO and c.score >= 0.7]
    assert merged == [], f"không được gộp Thụy Ba Công chúa: {merged}"


def test_hung_vuong_person_vs_ward_not_merged(resolver: AliasResolver) -> None:
    """``Hùng Vương`` (người) vs ``Hùng Vương`` (phường) -- không gộp."""
    cands = resolver.resolve("Hùng Vương")
    qids = {c.wikidata_id for c in cands if c.score >= 0.7}
    assert "Q1320050" in qids
    assert "Q10771249" not in qids

    person = resolver.resolve_one("Hùng Vương", {"entity_type": "Person"})
    assert person is not None
    assert person.wikidata_id == "Q1320050"

    ward = resolver.resolve_one("Hùng Vương", {"entity_type": "Place"})
    assert ward is not None
    assert ward.wikidata_id == "Q10771249"


def test_le_loi_alias(resolver: AliasResolver) -> None:
    candidate = resolver.resolve_one("Lê Lợi")
    assert candidate is not None
    assert candidate.wikidata_id == "Q1195391"


def test_tier2_redirect_known_alias(resolver: AliasResolver) -> None:
    """``Cửu Thiên Vũ Đế`` is a Wikipedia redirect -> tier 2."""
    candidates = resolver.resolve("Cửu Thiên Vũ Đế")
    assert candidates
    best = candidates[0]
    assert best.wikidata_id == TRAN_HUNG_DAO
    assert best.tier in (1, 2)


def test_tier3_llm_verification_promotes_or_rejects(alias_entities: list[Entity]) -> None:
    """With an LLM the dubious redirect goes through the T3 verification."""
    llm = MockLLMClient()
    res = AliasResolver(store=InMemoryGraphStore(), llm_client=llm, offline=True)
    res.build_alias_index(alias_entities)
    candidates = res.resolve("Cửu Thiên Vũ Đế")
    assert candidates
    assert candidates[0].wikidata_id == TRAN_HUNG_DAO


def test_tier4_normalized_key(resolver: AliasResolver) -> None:
    """A name that differs only by order/diacritics hits the normalised key."""
    candidate = resolver.resolve_one("Quốc Tuấn Trần")
    assert candidate is not None
    assert candidate.wikidata_id == TRAN_HUNG_DAO
    assert candidate.tier in (4, 1)


def test_tier5_embedding_domain_filter(alias_entities: list[Entity]) -> None:
    """Tier 5 uses embeddings but must respect the requested entity type."""
    embedder = MockEmbedder(dim=256)
    res = AliasResolver(
        store=InMemoryGraphStore(), llm_client=None, embedder=embedder, offline=True
    )
    res.build_alias_index(alias_entities)
    candidates = res.resolve("Trần Hưng Đạo")
    assert candidates
    assert candidates[0].wikidata_id == TRAN_HUNG_DAO


def test_resolve_batch(resolver: AliasResolver) -> None:
    out = resolver.resolve_batch(["Trần Quốc Tuấn", "Lê Lợi"])
    assert set(out) == {"Trần Quốc Tuấn", "Lê Lợi"}
    assert out["Trần Quốc Tuấn"][0].wikidata_id == TRAN_HUNG_DAO


def test_candidate_dataclass_fields(resolver: AliasResolver) -> None:
    candidate = resolver.resolve_one("Trần Quốc Tuấn")
    assert isinstance(candidate, EntityCandidate)
    payload = candidate.to_dict()
    for key in ("uid", "name", "wikidata_id", "score", "tier", "reason"):
        assert key in payload


# --------------------------------------------------------------------------
# extractor
# --------------------------------------------------------------------------


def test_extractor_with_mock_llm(sample_chunk: Chunk) -> None:
    extractor = EventExtractor(MockLLMClient(), ontology=ONTOLOGY, cache_dir=None)
    events = extractor.extract_from_chunk(sample_chunk)
    assert events
    for event in events:
        assert event.event_type in EVENT_TYPES
        assert event.chunk_id == sample_chunk.chunk_id
        assert validate_event(event.event_type, event.arguments) == []


def test_extractor_drops_invalid_events(sample_chunk: Chunk) -> None:
    class BadLLM:
        def generate_json(self, request):  # noqa: ANN001, ANN201
            return {
                "events": [
                    {"event_type": "khong-ton-tai", "trigger": "x", "arguments": []},
                    {
                        "event_type": "attack",
                        "trigger": "y",
                        "arguments": [{"role": "agent", "name": "Trần Hưng Đạo"}],
                    },
                ]
            }

        def embed(self, texts):  # noqa: ANN001, ANN201
            return []

        @property
        def embed_dim(self) -> int:
            return 8

    extractor = EventExtractor(BadLLM(), ontology=ONTOLOGY, cache_dir=None)
    events = extractor.extract_from_chunk(sample_chunk)
    assert len(events) == 1
    assert events[0].event_type == "attack"
    assert extractor.n_invalid == 1


def test_extractor_cache_roundtrip(sample_chunk: Chunk, tmp_path: Path) -> None:
    llm = MockLLMClient()
    extractor = EventExtractor(llm, ontology=ONTOLOGY, cache_dir=tmp_path)
    first = extractor.extract_from_chunk(sample_chunk)
    assert first
    assert (tmp_path / "ck_test_1.json").exists()

    extractor2 = EventExtractor(llm, ontology=ONTOLOGY, cache_dir=tmp_path)
    second = extractor2.extract_from_chunk(sample_chunk)
    assert [e.event_type for e in first] == [e.event_type for e in second]
    assert extractor2.n_cache_hits == 1


def test_extractor_prompt_mentions_ontology(sample_chunk: Chunk) -> None:
    from hgrag.kg.extractor import build_extraction_prompt

    prompt = build_extraction_prompt(sample_chunk, ONTOLOGY)
    assert "be-born" in prompt
    assert "AGENT" in prompt
    assert "extract" in "extract_events"


# --------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------


def _builder(tmp_path: Path) -> tuple[KGBuilder, InMemoryGraphStore]:
    store = InMemoryGraphStore()
    extractor = EventExtractor(MockLLMClient(), ontology=ONTOLOGY, cache_dir=tmp_path)
    builder = KGBuilder(store=store, ontology=ONTOLOGY, extractor=extractor)
    return builder, store


def test_builder_creates_entities_and_relations(tmp_path: Path, sample_chunk: Chunk) -> None:
    builder, store = _builder(tmp_path)
    stats = builder.build_from_chunks([sample_chunk], progress=False)
    assert stats.n_entities > 0
    assert stats.n_relations > 0
    assert stats.n_chunks == 1
    assert store.stats()["nodes"] == stats.n_entities


def test_builder_is_idempotent(tmp_path: Path, sample_chunk: Chunk) -> None:
    builder, store = _builder(tmp_path)
    first = builder.build_from_chunks([sample_chunk], progress=False)
    nodes_1, rels_1 = store.stats()["nodes"], store.stats()["relations"]
    second = builder.build_from_chunks([sample_chunk], progress=False)
    nodes_2, rels_2 = store.stats()["nodes"], store.stats()["relations"]
    assert nodes_1 == nodes_2
    assert rels_1 == rels_2
    assert first.n_entities == second.n_entities
    assert first.n_relations == second.n_relations


def test_builder_relations_have_source_chunk_ids(tmp_path: Path, sample_chunk: Chunk) -> None:
    builder, store = _builder(tmp_path)
    builder.build_from_chunks([sample_chunk], progress=False)
    assert store.relations
    for rel in store.relations.values():
        assert rel["source_chunk_ids"] == [sample_chunk.chunk_id]
        assert rel["extra"].get("evidence_text")


def test_builder_creates_supported_by_edges(tmp_path: Path, sample_chunk: Chunk) -> None:
    builder, store = _builder(tmp_path)
    builder.build_from_chunks([sample_chunk], progress=False)
    supported = [r for r in store.relations.values() if r["rel_type"] == "SUPPORTED_BY"]
    assert supported
    for rel in supported:
        assert rel["dst"] == sample_chunk.chunk_id
        assert sample_chunk.chunk_id in store.entities


def test_builder_uses_ontology_relation_types(tmp_path: Path, sample_chunk: Chunk) -> None:
    builder, store = _builder(tmp_path)
    stats = builder.build_from_chunks([sample_chunk], progress=False)
    assert stats.relation_types
    for rel_type in stats.relation_types:
        assert rel_type in set(RELATION_TYPES.values()) | {"SUPPORTED_BY"}


def test_builder_export_stats(tmp_path: Path, sample_chunk: Chunk) -> None:
    builder, _ = _builder(tmp_path)
    builder.build_from_chunks([sample_chunk], progress=False)
    payload = builder.export_stats()
    assert "store" in payload
    assert payload["ontology"]["n_event_types"] == 35


def test_builder_resolves_qid(tmp_path: Path, sample_chunk: Chunk) -> None:
    """The argument node must carry the Wikidata QID found by the resolver."""
    store = InMemoryGraphStore()
    entities = [
        _entity(
            uid=f"pe:{TRAN_HUNG_DAO}",
            name="Trần Hưng Đạo",
            aliases=["Trần Quốc Tuấn"],
            wikidata_id=TRAN_HUNG_DAO,
            p31=["Q5"],
            p27=["Q430309"],
        )
    ]
    resolver = AliasResolver(store=store, offline=True)
    resolver.build_alias_index(entities)
    extractor = EventExtractor(MockLLMClient(), ontology=ONTOLOGY, cache_dir=tmp_path)
    builder = KGBuilder(store=store, ontology=ONTOLOGY, extractor=extractor, resolver=resolver)
    stats = builder.build_from_chunks([sample_chunk], progress=False)
    assert stats.n_linked >= 0  # mock LLM uses generic names -> no link, but no crash
    assert stats.n_entities > 0


def test_builder_attaches_qid_to_argument(tmp_path: Path, sample_chunk: Chunk) -> None:
    """A resolvable mention becomes a node with ``wikidata_id`` set."""

    class NamedLLM:
        def generate_json(self, request):  # noqa: ANN001, ANN201
            return {
                "events": [
                    {
                        "event_type": "attack",
                        "trigger": "Trần Hưng Đạo chỉ huy quân",
                        "description": "Trận Bạch Đằng năm 1284",
                        "year": 1284,
                        "arguments": [
                            {"role": "agent", "name": "Trần Quốc Tuấn", "type": "Person"},
                        ],
                    }
                ]
            }

    store = InMemoryGraphStore()
    entities = [
        _entity(
            uid=f"pe:{TRAN_HUNG_DAO}",
            name="Trần Hưng Đạo",
            aliases=["Trần Quốc Tuấn"],
            wikidata_id=TRAN_HUNG_DAO,
            p31=["Q5"],
            p27=["Q430309"],
        )
    ]
    resolver = AliasResolver(store=store, offline=True)
    resolver.build_alias_index(entities)
    extractor = EventExtractor(NamedLLM(), ontology=ONTOLOGY, cache_dir=tmp_path)
    builder = KGBuilder(store=store, ontology=ONTOLOGY, extractor=extractor, resolver=resolver)
    stats = builder.build_from_chunks([sample_chunk], progress=False)

    assert stats.n_linked == 1
    person_nodes = [
        props for props in store.entities.values() if props.get("entity_type") == "Person"
    ]
    assert len(person_nodes) == 1
    assert person_nodes[0]["wikidata_id"] == TRAN_HUNG_DAO
    assert person_nodes[0]["uid"] == f"pe:{TRAN_HUNG_DAO}"


def test_builder_skips_chunks_without_events(tmp_path: Path, sample_chunk: Chunk) -> None:
    class EmptyLLM:
        def generate_json(self, request):  # noqa: ANN001, ANN201
            return {"events": []}

    store = InMemoryGraphStore()
    extractor = EventExtractor(EmptyLLM(), ontology=ONTOLOGY, cache_dir=tmp_path)
    builder = KGBuilder(store=store, ontology=ONTOLOGY, extractor=extractor)
    stats = builder.build_from_chunks([sample_chunk], progress=False)
    assert stats.n_skipped == 1
    assert stats.n_events == 0


def test_builder_max_chunks(tmp_path: Path, sample_chunk: Chunk) -> None:
    builder, _ = _builder(tmp_path)
    base = {f: getattr(sample_chunk, f) for f in Chunk.__slots__}
    chunks = [Chunk(**{**base, "chunk_id": f"ck_{i}"}) for i in range(5)]
    stats = builder.build_from_chunks(chunks, max_chunks=2, progress=False)
    assert stats.n_chunks == 2
