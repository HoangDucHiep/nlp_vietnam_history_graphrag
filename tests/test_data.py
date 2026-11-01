"""Tests for the data layer (normalisation, chunking, loaders, ingest).

Most tests run **offline** using local fixtures / stub clients.  Tests that
really hit the network are marked ``@pytest.mark.network`` and are skipped
unless ``HGRAG_RUN_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hgrag.data import chunking, history_chat, normalize, vhe, wikidata, wikipedia
from hgrag.data.ingest import ingest_all, write_corpus
from hgrag.data.records import Chunk

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

SAMPLE_TEXT = """Trần Hưng Đạo (chữ Hán: 陳興道; tên thật: Trần Quốc Tuấn) là chính trị gia,
nhà chiến lược quân sự Đại Việt thời nhà Trần. Ông nổi tiếng với hai lần chống
Nguyên Mông xâm lược.

== Thân thế và thiếu thời ==
Trần Quốc Tuấn là con trai của tông thất Trần Liễu, anh trai ruột của
quốc vương Đại Việt Trần Cảnh. Ông sinh ra ở quê hương dấu nghiệp nhà Trần.

=== Mối thù nội tộc ===
Sau khi quốc vương Trần Thái Tông qua đời, Trần Cảnh lên ngôi.
Trần Quốc Tuấn từ chối chức vị.

== Sự nghiệp ==
Năm 1285, quân Nguyôn Mông tràn vào Đại Việt. Trần Hưng Đạo được giao
chỉ huy quân chống giặc ngoại xâm.

== Di sản và đánh giá ==
Trần Hưng Đạo được nhân dân Việt Nam tôn sùng là anh hùng dân tộc.
"""


@pytest.fixture()
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture()
def offline_settings(tmp_path: Path):
    """A :class:`Settings` object rooted in a temporary directory."""
    from hgrag.config import Settings

    raw = {
        "paths": {
            "data_raw": "data/raw",
            "data_processed": "data/processed",
            "artifacts": "artifacts",
            "reports": "reports",
            "cache": "artifacts/llm_cache",
        },
        "ingest": {
            "request_delay": 0.0,
            "wikipedia_min_chars": 100,
            "wikipedia_max_articles": 5,
            "wikipedia_expand_rounds": 0,
            "history_chat_limit": 10,
            "history_chat_min_year": 905,
            "history_chat_max_year": 1975,
            "user_agent": "test/0.1",
        },
        "chunking": {"max_chars": 400, "overlap_chars": 50, "min_chars": 60},
    }
    settings = Settings(raw=raw, root=tmp_path)
    settings.ensure_dirs()
    return settings


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------


class TestNormalize:
    def test_normalize_keeps_diacritics(self):
        out = normalize.normalize_name("Trần Quốc Tuấn")
        assert out == "trần quốc tuấn"
        assert "ầ" in out  # dấu vẫn còn

    def test_normalize_nfc_and_lowercase(self):
        # "Hưng Ðạo" with the old Đ glyph must become "Hưng Đạo".
        assert normalize.normalize_name("Hưng Ðạo") == "hưng đạo"
        assert normalize.normalize_name("Hưng Đạo") == "hưng đạo"

    def test_normalize_strips_punctuation(self):
        assert normalize.normalize_name("Trần Quốc Tuấn, (1228-1300)!") == (
            "trần quốc tuấn 1228 1300"
        )

    def test_strip_honorifics(self):
        assert normalize.strip_honorifics("Đại vương Trần Hưng Đạo") == "Trần Hưng Đạo"
        assert normalize.strip_honorifics("vua Lê Lợi") == "Lê Lợi"
        assert normalize.strip_honorifics("Tiên Đế Lý Thái Tổ") == "Lý Thái Tổ"
        assert normalize.strip_honorifics("Đại tướng quân Võ Nguyên Giáp") == "Võ Nguyên Giáp"

    def test_strip_honorifics_keeps_diacritics(self):
        out = normalize.strip_honorifics("Hoàng đế Thái Tổ Lê Lợi")
        assert "Lê Lợi" in out
        assert "Hoàng đế" not in out

    def test_name_key_ignores_diacritics_and_order(self):
        assert normalize.name_key("Trần Quốc Tuấn") == normalize.name_key(
            "Quốc Tuấn Trần"
        )
        assert normalize.name_key("Trần Quốc Tuấn") == normalize.name_key("tran quoc tuan")

    def test_name_key_strips_honorifics(self):
        assert normalize.name_key("Đại vương Trần Hưng Đạo") == normalize.name_key(
            "Trần Hưng Đạo"
        )

    def test_name_key_removes_parenthetical(self):
        assert normalize.name_key("Lê Lợi (kỹ thuật)") == normalize.name_key("Lê Lợi")

    def test_name_key_empty(self):
        assert normalize.name_key("") == ""
        assert normalize.normalize_name("") == ""
        assert normalize.strip_honorifics("") == ""

    def test_split_name_parts(self):
        family, given = normalize.split_name_parts("Trần Hưng Đạo")
        assert family == "Trần"
        assert given == "Hưng Đạo"


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


class TestChunking:
    def test_split_sections(self, sample_text: str):
        sections = chunking.split_sections(sample_text)
        headings = [h for h, _ in sections]
        assert "" in headings  # preamble
        assert "Thân thế và thiếu thời" in headings
        assert "Sự nghiệp" in headings

    def test_chunk_document_respects_max_chars(self, sample_text: str):
        chunks = chunking.chunk_document(
            "wiki:Tran_Hung_Dao",
            "Trần Hưng Đạo",
            sample_text,
            max_chars=300,
            overlap_chars=40,
            min_chars=50,
        )
        assert chunks
        assert all(isinstance(c, Chunk) for c in chunks)
        for c in chunks:
            assert len(c.text) <= 500  # allow the long-sentence hard split
            assert c.doc_id == "wiki:Tran_Hung_Dao"
            assert c.source == "wikipedia"

    def test_chunk_ids_are_stable(self, sample_text: str):
        a = chunking.chunk_document("d", "t", sample_text, max_chars=300)
        b = chunking.chunk_document("d", "t", sample_text, max_chars=300)
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
        assert all(c.chunk_id.startswith("ck_") for c in a)

    def test_chunk_ids_differ_between_docs(self, sample_text: str):
        a = chunking.chunk_document("d1", "t", sample_text)
        b = chunking.chunk_document("d2", "t", sample_text)
        assert a[0].chunk_id != b[0].chunk_id

    def test_no_sentence_is_cut_in_the_middle(self):
        text = "Câu một. Câu hai. Câu ba. " * 20
        chunks = chunking.chunk_document("d", "t", text, max_chars=120, overlap_chars=0)
        assert chunks
        for c in chunks:
            assert c.text.endswith(".") or len(c.text) < 60

    def test_chunk_document_with_sections(self):
        sections = [
            {"heading": "A", "text": "Nội dung A."},
            {"heading": "B", "text": "Nội dung B."},
        ]
        chunks = chunking.chunk_document(
            "d", "t", "", sections=sections, max_chars=400, min_chars=10
        )
        assert [c.section for c in chunks] == ["A", "B"]

    def test_empty_text(self):
        assert chunking.chunk_document("d", "t", "") == []
        assert chunking.chunk_text("") == []
        assert chunking.split_sections("") == []


# --------------------------------------------------------------------------
# wikipedia (offline: stubbed client)
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _FakeSession:
    """Minimal stub for ``requests.Session``."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.headers: dict[str, str] = {}

    def get(self, url, params=None, timeout=None):  # noqa: ARG002
        return _FakeResponse(self._payload)


def _fake_page_payload(title: str = "Trần Hưng Đạo") -> dict:
    return {
        "query": {
            "pages": {
                "123": {
                    "pageid": 123,
                    "title": title,
                    "extract": (
                        "Trần Hưng Đạo là chính trị gia Đại Việt thời nhà Trần.\n"
                        "== Sự nghiệp ==\n"
                        "Năm 1285, ông chống quân Nguyên Mông xâm lược Đại Việt."
                    ),
                    "pageprops": {"wikibase_item": "Q511375"},
                    "categories": [{"title": "Thể loại:Nhà văn nhà Trần"}],
                    "redirects": [{"title": "Trần Quốc Tuấn"}],
                }
            }
        }
    }


def _fake_batch_payload() -> dict:
    """A two-page response using ``prop=revisions`` (wikitext)."""
    return {
        "query": {
            "normalized": [{"from": "tran hung dao", "to": "Trần Hưng Đạo"}],
            "redirects": [{"from": "Hưng Đạo vương", "to": "Trần Hưng Đạo"}],
            "pages": {
                "123": {
                    "pageid": 123,
                    "title": "Trần Hưng Đạo",
                    "revisions": [
                        {
                            "slots": {
                                "main": {
                                    "*": (
                                        "{{Infobox person\n| tên = Trần Hưng Đạo\n}}\n"
                                        "Trần Hưng Đạo là chính trị gia Đại Việt.\n"
                                        "== Sự nghiệp ==\n"
                                        "Năm 1285, ông chống quân Nguyên Mông xâm lược."
                                    )
                                }
                            }
                        }
                    ],
                    "pageprops": {"wikibase_item": "Q511375"},
                    "categories": [{"title": "Thể loại:Nhà văn nhà Trần"}],
                    "redirects": [{"title": "Trần Quốc Tuấn"}],
                },
                "456": {
                    "pageid": 456,
                    "title": "Lê Lợi",
                    "revisions": [
                        {
                            "slots": {
                                "main": {
                                    "*": (
                                        "Lê Lợi là một nhà chính trị, nhà lãnh tụ.\n"
                                        "== Lam Sơn ==\n"
                                        "Năm 1418, Lê Lợi lên ngôi."
                                    )
                                }
                            }
                        }
                    ],
                    "pageprops": {"wikibase_item": "Q1195391"},
                    "categories": [{"title": "Thể loại:Nhà vua nhà Lê"}],
                    "redirects": [],
                },
            },
        }
    }


class TestWikipediaOffline:
    def test_fetch_page_parses_payload(self, tmp_path: Path):
        client = wikipedia.WikipediaClient(
            delay=0, cache_dir=tmp_path, session=_FakeSession(_fake_page_payload())
        )
        page = client.fetch_page("Trần Hưng Đạo")
        assert page["exists"] is True
        assert page["wikidata_id"] == "Q511375"
        assert page["redirects"] == ["Trần Quốc Tuấn"]
        assert page["categories"] == ["Thể loại:Nhà văn nhà Trần"]
        assert "Sự nghiệp" in [s["heading"] for s in page["sections"]]
        assert "1285" in page["extract"]

    def test_fetch_page_missing_article(self, tmp_path: Path):
        payload = {"query": {"pages": {"-1": {"title": "x", "missing": ""}}}}
        client = wikipedia.WikipediaClient(
            delay=0, cache_dir=tmp_path, session=_FakeSession(payload)
        )
        page = client.fetch_page("Không Tồn Tại")
        assert page["exists"] is False
        assert page["extract"] == ""

    def test_cache_is_used(self, tmp_path: Path):
        client = wikipedia.WikipediaClient(
            delay=0, cache_dir=tmp_path, session=_FakeSession(_fake_page_payload())
        )
        first = client.fetch_page("Trần Hưng Đạo")
        assert (tmp_path / "Trần_Hưng_Đạo.json").exists()
        # A broken session now -> the cached copy must still be returned.
        client.session = _FakeSession({"query": {"pages": {}}})
        second = client.fetch_page("Trần Hưng Đạo")
        assert second["wikidata_id"] == first["wikidata_id"]

    def test_seed_titles_count(self):
        assert len(wikipedia.DEFAULT_SEED_TITLES) >= 40
        # de-duplicated
        assert len(set(wikipedia.DEFAULT_SEED_TITLES)) == len(wikipedia.DEFAULT_SEED_TITLES)

    def test_seed_titles_cover_key_entities(self):
        joined = " ".join(wikipedia.DEFAULT_SEED_TITLES)
        for needed in ("Trần Hưng Đạo", "Lê Lợi", "Hồ Chí Minh", "Điện Biên Phủ"):
            assert needed in joined

    def test_history_scope_filter(self):
        page = {"categories": ["Thể loại:Nhà vua nhà Trần"], "title": "X", "extract": ""}
        assert wikipedia.is_in_history_scope(page) is True
        other = {
            "categories": ["Thể loại:Trang hướng"],
            "title": "Y",
            "extract": "một bài về thời tiết",
        }
        assert wikipedia.is_in_history_scope(other) is False

    def test_title_slug(self):
        assert wikipedia.title_slug("Trần Hưng Đạo") == "Trần_Hưng_Đạo"
        assert "/" not in wikipedia.title_slug("A/B")

    def test_fetch_pages_batches_wikitext(self, tmp_path: Path):
        client = wikipedia.WikipediaClient(
            delay=0, cache_dir=tmp_path, session=_FakeSession(_fake_batch_payload())
        )
        pages = client.fetch_pages(["Trần Hưng Đạo", "Lê Lợi"])
        assert len(pages) == 2
        by_title = {p["title"]: p for p in pages}
        assert set(by_title) == {"Trần Hưng Đạo", "Lê Lợi"}
        thd = by_title["Trần Hưng Đạo"]
        assert thd["wikidata_id"] == "Q511375"
        assert "1285" in thd["extract"]
        # The infobox template must be stripped from the plain text.
        assert "Infobox person" not in thd["extract"]
        assert "Sự nghiệp" in [s["heading"] for s in thd["sections"]]

    def test_fetch_pages_resolves_redirects(self, tmp_path: Path):
        client = wikipedia.WikipediaClient(
            delay=0, cache_dir=tmp_path, session=_FakeSession(_fake_batch_payload())
        )
        pages = client.fetch_pages(["Hưng Đạo vương"])
        assert pages[0]["title"] == "Trần Hưng Đạo"
        assert pages[0]["exists"] is True

    def test_fetch_pages_uses_cache(self, tmp_path: Path):
        client = wikipedia.WikipediaClient(
            delay=0, cache_dir=tmp_path, session=_FakeSession(_fake_batch_payload())
        )
        client.fetch_pages(["Lê Lợi"])
        assert (tmp_path / "Lê_Lợi.json").exists()
        # A broken session now -> the cached copy must still be returned.
        client.session = _FakeSession({"query": {"pages": {}}})
        pages = client.fetch_pages(["Lê Lợi"])
        assert pages[0]["wikidata_id"] == "Q1195391"


# --------------------------------------------------------------------------
# wikidata (offline: stubbed client)
# --------------------------------------------------------------------------


class TestWikidataOffline:
    def test_get_entities_batches(self, tmp_path: Path):
        payload = {
            "entities": {
                "Q511375": {
                    "labels": {"vi": {"value": "Trần Hưng Đạo"}},
                    "aliases": {"vi": [{"value": "Trần Quốc Tuấn"}]},
                    "descriptions": {"vi": {"value": "chính trị gia nhà Trần"}},
                    "claims": {
                        "P31": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {"value": {"id": "Q5"}},
                                }
                            }
                        ],
                        "P27": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {"value": {"id": "Q430309"}},
                                }
                            }
                        ],
                        "P569": [
                            {
                                "mainsnak": {
                                    "snaktype": "value",
                                    "datavalue": {"value": {"time": "+1226-00-00T00:00:00Z"}},
                                }
                            }
                        ],
                    },
                }
            }
        }
        client = wikidata.WikidataClient(
            delay=0, cache_dir=tmp_path, session=_FakeSession(payload)
        )
        entities = client.get_entities(["Q511375"])
        assert "Q511375" in entities
        assert client.entity_aliases("Q511375") == ["Trần Hưng Đạo", "Trần Quốc Tuấn"]
        props = client.entity_props("Q511375")
        assert props["p31"] == ["Q5"]
        assert props["p27"] == ["Q430309"]
        assert props["p569"] == ["+1226-00-00T00:00:00Z"]
        assert props["description"] == "chính trị gia nhà Trần"

    def test_missing_entity(self, tmp_path: Path):
        client = wikidata.WikidataClient(
            delay=0, cache_dir=tmp_path, session=_FakeSession({"entities": {}})
        )
        assert client.get_entities(["Q1"]) == {}
        assert client.entity_aliases("Q1") == []
        assert client.entity_props("Q1")["p31"] == []

    def test_empty_inputs(self, tmp_path: Path):
        client = wikidata.WikidataClient(delay=0, cache_dir=tmp_path)
        assert client.get_entities([]) == {}
        assert client.entity_aliases("") == []


# --------------------------------------------------------------------------
# VHE
# --------------------------------------------------------------------------


class TestVHE:
    def test_builtin_ontology_has_35_types_and_11_roles(self):
        assert len(vhe.DEFAULT_EVENT_TYPES) == 35
        assert len(vhe.DEFAULT_ROLES) == 11

    def test_load_ontology_offline_falls_back(self, tmp_path: Path):
        onto = vhe.load_ontology(cache_dir=tmp_path, offline=True)
        assert onto["n_event_types"] == 35
        assert onto["n_roles"] == 11
        assert onto["source"] == "builtin-fallback"
        assert "be-born" in onto["event_types"]
        assert "agent" in onto["roles"]

    def test_load_ontology_from_local_files(self, tmp_path: Path):
        base = tmp_path / "event-extraction"
        base.mkdir(parents=True)
        (base / "label2id.json").write_text(
            json.dumps({"NA": 0, "be-born": 1, "die": 2}), encoding="utf-8"
        )
        (base / "role2id.json").write_text(
            json.dumps({"NA": 0, "temporal": 1, "agent": 2}), encoding="utf-8"
        )
        onto = vhe.load_ontology(cache_dir=tmp_path, offline=True)
        assert onto["source"] == "vhe-dataset"
        assert onto["event_types"] == ["be-born", "die"]
        assert onto["roles"] == ["temporal", "agent"]

    def test_load_instances_offline_empty(self, tmp_path: Path):
        assert vhe.load_instances(cache_dir=tmp_path, offline=True) == []

    def test_load_instances_from_local_file(self, tmp_path: Path):
        base = tmp_path / "event-extraction"
        base.mkdir(parents=True)
        (base / "event.json").write_text(
            json.dumps(
                [
                    {
                        "id": "a",
                        "text": "Xưa cháu ba đời ...",
                        "events": [
                            {
                                "type": "be-born",
                                "trigger_word": "sinh",
                                "arguments": [
                                    {"mention": "Đế Minh", "offset": [44, 51], "role": "agent"}
                                ],
                            }
                        ],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        inst = vhe.load_instances(cache_dir=tmp_path, offline=True)
        assert len(inst) == 1
        assert inst[0]["events"][0]["type"] == "be-born"
        mentions = vhe.extract_mentions(inst)
        assert "Đế Minh" in mentions

    def test_load_raw_unknown_key(self, tmp_path: Path):
        with pytest.raises(KeyError):
            vhe.load_raw("nope", cache_dir=tmp_path)


# --------------------------------------------------------------------------
# history chat
# --------------------------------------------------------------------------


class TestHistoryChat:
    def test_row_to_record(self):
        row = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Vì sao xảy ra sự kiện Đặt ách bảo hộ (1884)?"},
                {"role": "assistant", "content": "analysis", "channel": "analysis"},
                {"role": "assistant", "content": "final answer", "channel": "final"},
            ]
        }
        rec = history_chat.row_to_record(row)
        assert rec["question"].startswith("Vì sao")
        assert rec["answer"] == "final answer"
        assert rec["year"] == 1884

    def test_year_filter(self, tmp_path: Path):
        cache = tmp_path / "history_chat.json"
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "Sự kiện năm 2020?"},
                    {"role": "assistant", "content": "abc", "channel": "final"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Trận Điện Biên Phủ 1954?"},
                    {"role": "assistant", "content": "abc", "channel": "final"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Câu hỏi không có năm?"},
                    {"role": "assistant", "content": "abc", "channel": "final"},
                ]
            },
        ]
        cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        recs = history_chat.load_history_chat(limit=10, cache_dir=tmp_path, offline=True)
        years = [r["year"] for r in recs]
        assert 2020 not in years  # out of scope (>1975)
        assert 1954 in years
        assert None in years  # yearless rows are kept

    def test_offline_empty(self, tmp_path: Path):
        assert history_chat.load_history_chat(limit=5, cache_dir=tmp_path, offline=True) == []


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


class TestIngest:
    def test_write_corpus(self, tmp_path: Path):
        chunks = [
            Chunk(chunk_id="ck_1", doc_id="wiki:A", title="A", text="x", source="wikipedia"),
            Chunk(chunk_id="ck_2", doc_id="wiki:B", title="B", text="y", source="wikipedia"),
        ]
        out = tmp_path / "corpus.jsonl"
        n = write_corpus(chunks, out)
        assert n == 2
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["chunk_id"] == "ck_1"

    def test_ingest_all_offline_vhe_only(self, offline_settings):
        stats = ingest_all(
            settings=offline_settings,
            limit=3,
            sources=["vhe"],
            out_dir=offline_settings.data_processed,
            offline=True,
        )
        assert stats["n_chunks"] == 0  # VHE is ontology+gold, not corpus (ADR-01)
        assert stats["vhe"]["n_event_types"] == 35
        assert stats["vhe"]["n_roles"] == 11
        meta = offline_settings.data_processed / "corpus_meta.json"
        assert meta.exists()
        assert json.loads(meta.read_text(encoding="utf-8"))["n_chunks"] == 0

    def test_ingest_all_never_raises(self, offline_settings):
        stats = ingest_all(
            settings=offline_settings,
            limit=2,
            sources=["wikipedia", "vhe", "history_chat"],
            out_dir=offline_settings.data_processed,
            offline=True,
        )
        assert "n_chunks" in stats
        assert isinstance(stats["warnings"], list)


# --------------------------------------------------------------------------
# network tests (opt-in)
# --------------------------------------------------------------------------

network_only = pytest.mark.skipif(
    os.getenv("HGRAG_RUN_NETWORK_TESTS") != "1",
    reason="set HGRAG_RUN_NETWORK_TESTS=1 to run network tests",
)


@pytest.mark.network
@network_only
def test_network_wikipedia_fetch():
    client = wikipedia.WikipediaClient(delay=0.2)
    page = client.fetch_page("Trần Hưng Đạo")
    assert page["exists"] is True
    assert page["wikidata_id"] == "Q511375"
    assert len(page["extract"]) > 500


@pytest.mark.network
@network_only
def test_network_wikidata_props():
    client = wikidata.WikidataClient(delay=0.2)
    props = client.entity_props("Q511375")
    assert props["label"] == "Trần Hưng Đạo"
    assert props["p31"]
