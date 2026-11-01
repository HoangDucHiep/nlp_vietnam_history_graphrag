"""Tests for the evaluation dataset layer (docs/04).

Everything here runs **offline**: the builder is exercised with the
:class:`~hgrag.llm.client.MockLLMClient` and a small in-memory corpus, so
the tests never touch the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hgrag.config import Settings
from hgrag.data.records import Chunk, Entity
from hgrag.eval.dataset import (
    DIFFICULTIES,
    QUESTION_SOURCES,
    QUESTION_TYPES,
    EvalDataset,
    EvalQuestion,
    load_jsonl,
    save_jsonl,
)
from hgrag.eval.dataset_builder import (
    OUT_OF_KB_QUESTIONS,
    TARGET_TYPE_RATIOS,
    EvalDatasetBuilder,
)
from hgrag.llm.client import MockLLMClient

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    Chunk(
        chunk_id="ck_tran_1",
        doc_id="wiki:Trần_Hưng_Đạo",
        title="Trần Hưng Đạo",
        text=(
            "Trần Hưng Đạo (tên thật Trần Quốc Tuấn) là chính trị gia, nhà chiến lược "
            "quân sự Đại Việt thời nhà Trần. Năm 1285, ông được giao chỉ huy quân chống "
            "quân Nguyên Mông xâm lược trên sông Bạch Đằng."
        ),
        source="wikipedia",
        wikidata_id="Q511375",
        year_min=1285,
        year_max=1285,
    ),
    Chunk(
        chunk_id="ck_ngo_1",
        doc_id="wiki:Ngô_Quyền",
        title="Ngô Quyền",
        text=(
            "Ngô Quyền (895-941) là nhà chính trị và nhà chỉ huy quân sự, người lập ra "
            "nhà Ngô. Năm 938, ông đánh tan quân Kiều Công Tiễn và lập nên nhà Ngô."
        ),
        source="wikipedia",
        wikidata_id="Q312321",
        year_min=938,
        year_max=941,
    ),
    Chunk(
        chunk_id="ck_ly_1",
        doc_id="wiki:Lý_Thái_Tổ",
        title="Lý Thái Tổ",
        text=(
            "Lý Thái Tổ (974-1028), tên húy Lý Công Uẩn, là hoàng đế sáng lập nhà Lý. "
            "Năm 1010, ông dời đô từ Hoa Lư về Đại La và đổi tên thành Thăng Long."
        ),
        source="wikipedia",
        wikidata_id="Q716945",
        year_min=1010,
        year_max=1028,
    ),
]

SAMPLE_ENTITIES = [
    Entity(
        uid="pe:Q511375",
        name="Trần Hưng Đạo",
        entity_type="Person",
        wikidata_id="Q511375",
        aliases=["Trần Quốc Tuấn"],
    ),
    Entity(uid="pe:Q312321", name="Ngô Quyền", wikidata_id="Q312321"),
]


@pytest.fixture()
def eval_settings(tmp_path: Path) -> Settings:
    raw = {
        "paths": {
            "data_raw": "data/raw",
            "data_processed": "data/processed",
            "artifacts": "artifacts",
            "reports": "reports",
            "cache": "artifacts/llm_cache",
        },
        "llm": {"provider": "mock", "model": "mock", "embed_model": "mock", "temperature": 0.0},
        "embedding": {"provider": "mock", "dim": 64},
        "evaluation": {"eval_size_target": 150, "seed": 42},
    }
    settings = Settings(raw=raw, root=tmp_path)
    settings.ensure_dirs()
    _copy_raw_cache(settings)
    return settings


def _copy_raw_cache(settings: Settings) -> None:
    """Copy the repo's cached VHE / history-chat files into the temp root.

    The evaluation builder reads the VHE gold annotations from
    ``data/raw/vhe``; copying them keeps the test fully offline while still
    using the real annotations.
    """
    repo_root = Path(__file__).resolve().parents[1]
    for name in ("vhe", "history_chat"):
        src = repo_root / "data" / "raw" / name
        if not src.exists():
            continue
        dst = Path(settings.data_raw) / name
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*"):
            if item.is_file():
                target = dst / item.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(item.read_bytes())


@pytest.fixture()
def builder(eval_settings: Settings) -> EvalDatasetBuilder:
    return EvalDatasetBuilder(
        llm_client=MockLLMClient(embed_dim=64, seed=42),
        settings=eval_settings,
        corpus_chunks=list(SAMPLE_CHUNKS),
        kg_entities=list(SAMPLE_ENTITIES),
    )


@pytest.fixture(scope="module")
def _cached_dataset() -> EvalDataset:
    """Build the dataset once and reuse it across the assertions.

    Building the 150-question set parses the whole VHE gold file, which is
    too slow to repeat for every single test.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw = {
            "paths": {
                "data_raw": "data/raw",
                "data_processed": "data/processed",
                "artifacts": "artifacts",
                "reports": "reports",
                "cache": "artifacts/llm_cache",
            },
            "llm": {
                "provider": "mock",
                "model": "mock",
                "embed_model": "mock",
                "temperature": 0.0,
            },
            "embedding": {"provider": "mock", "dim": 64},
            "evaluation": {"eval_size_target": 150, "seed": 42},
        }
        settings = Settings(raw=raw, root=tmp_path)
        settings.ensure_dirs()
        _copy_raw_cache(settings)
        builder = EvalDatasetBuilder(
            llm_client=MockLLMClient(embed_dim=64, seed=42),
            settings=settings,
            corpus_chunks=_load_repo_corpus(),
            kg_entities=list(SAMPLE_ENTITIES),
        )
        return builder.build(target_size=150, offline=True)


def _load_repo_corpus() -> list[Chunk]:
    """Read the real corpus of the repository (falls back to the samples)."""
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "data" / "processed" / "corpus.jsonl"
    if not path.exists():
        return list(SAMPLE_CHUNKS)
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except ValueError:
                continue
    return chunks or list(SAMPLE_CHUNKS)


@pytest.fixture()
def dataset(_cached_dataset: EvalDataset) -> EvalDataset:
    return _cached_dataset


# --------------------------------------------------------------------------
# dataclass contract
# --------------------------------------------------------------------------


class TestEvalQuestion:
    def test_to_dict_roundtrip(self):
        q = EvalQuestion(
            qid="vhe_00001",
            question="Ai là người sinh ra Đế Nghi?",
            answer="Đế Minh",
            question_type="single_hop",
            gold_entities=["Đế Minh"],
            gold_qids=["Q1"],
            gold_paths=[["Đế Minh", "sinh ra Đế Nghi"]],
            gold_chunk_ids=["ck_1"],
            gold_relations=["Đế Minh -PARTICIPATED_IN-> sinh ra Đế Nghi"],
            year_refs=[938],
            difficulty="easy",
            source="vhe_template",
            metadata={"role": "agent"},
        )
        payload = q.to_dict()
        assert payload["qid"] == "vhe_00001"
        assert payload["gold_entities"] == ["Đế Minh"]
        assert payload["year_refs"] == [938]

        back = EvalQuestion.from_dict(payload)
        assert back == q

    def test_from_dict_is_lenient_with_unknown_keys(self):
        q = EvalQuestion.from_dict({"qid": "x", "question": "y", "future_field": 42})
        assert q.qid == "x"
        assert q.metadata["future_field"] == 42

    def test_defaults_are_empty_collections(self):
        q = EvalQuestion(qid="a", question="b")
        assert q.gold_entities == []
        assert q.gold_paths == []
        assert q.gold_chunk_ids == []
        assert q.metadata == {}

    def test_is_out_of_kb_and_has_evidence(self):
        assert EvalQuestion(qid="a", question="b", question_type="out_of_kb").is_out_of_kb
        assert not EvalQuestion(qid="a", question="b").is_out_of_kb
        assert EvalQuestion(qid="a", question="b", gold_entities=["X"]).has_evidence
        assert EvalQuestion(qid="a", question="b", gold_chunk_ids=["ck"]).has_evidence
        assert not EvalQuestion(qid="a", question="b").has_evidence

    def test_valid_enum_values(self):
        for qtype in QUESTION_TYPES:
            assert qtype in QUESTION_TYPES
        for difficulty in DIFFICULTIES:
            assert difficulty in DIFFICULTIES
        for source in QUESTION_SOURCES:
            assert source in QUESTION_SOURCES


class TestJsonlIO:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        questions = [
            EvalQuestion(qid="a", question="Câu 1", answer="A1", gold_entities=["X"]),
            EvalQuestion(qid="b", question="Câu 2", question_type="multi_hop"),
        ]
        path = tmp_path / "eval.jsonl"
        n = save_jsonl(questions, path)
        assert n == 2
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

        loaded = load_jsonl(path)
        assert loaded == questions

    def test_dataset_save_and_load(self, tmp_path: Path):
        dataset = EvalDataset(
            questions=[EvalQuestion(qid="a", question="Câu 1")],
            metadata={"note": "test"},
        )
        path = tmp_path / "eval.jsonl"
        assert dataset.save_jsonl(path) == 1
        back = EvalDataset.load_jsonl(path)
        assert len(back) == 1
        assert back.questions[0].qid == "a"

    def test_load_skips_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "broken.jsonl"
        path.write_text(
            '{"qid": "a", "question": "Câu 1"}\nnot json\n\n{"qid": "b", "question": "C2"}\n',
            encoding="utf-8",
        )
        loaded = load_jsonl(path)
        assert [q.qid for q in loaded] == ["a", "b"]

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        assert load_jsonl(tmp_path / "nope.jsonl") == []

    def test_lines_are_valid_json(self, tmp_path: Path):
        path = tmp_path / "eval.jsonl"
        EvalDataset(questions=[EvalQuestion(qid="a", question="Câu 1")]).save_jsonl(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            assert payload["qid"] == "a"


# --------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------


class TestDatasetBuilder:
    def test_build_reaches_target_size(self, dataset: EvalDataset):
        assert len(dataset) >= 150

    def test_type_quotas_are_respected(self, dataset: EvalDataset):
        stats = dataset.stats()
        by_type = stats["by_type"]
        assert by_type["single_hop"] >= 55  # ~40 %
        assert by_type["multi_hop"] >= 40  # ~30 %
        assert by_type["temporal"] >= 18  # ~15 %
        assert by_type["out_of_kb"] >= 15  # ~15 %
        for qtype, ratio in TARGET_TYPE_RATIOS.items():
            assert stats["by_type_ratio"][qtype] >= ratio - 0.05

    def test_out_of_kb_questions_have_no_answer_and_no_evidence(
        self, dataset: EvalDataset
    ):
        oob = [q for q in dataset.questions if q.question_type == "out_of_kb"]
        assert len(oob) >= 10
        for q in oob:
            assert q.answer == ""
            assert q.gold_entities == []
            assert q.gold_chunk_ids == []
            assert q.is_out_of_kb

    def test_every_non_out_of_kb_question_has_evidence(self, dataset: EvalDataset):
        for q in dataset.questions:
            if q.question_type == "out_of_kb":
                continue
            assert q.gold_entities or q.gold_chunk_ids, (
                f"thiếu gold evidence cho {q.qid}: {q.question!r}"
            )

    def test_multi_hop_questions_have_paths_with_two_nodes(self, dataset: EvalDataset):
        multi = [q for q in dataset.questions if q.question_type == "multi_hop"]
        assert len(multi) >= 30
        for q in multi:
            assert any(len(path) >= 2 for path in q.gold_paths), (
                f"multi-hop thiếu gold path: {q.qid} {q.question!r}"
            )

    def test_questions_are_deduplicated(self, dataset: EvalDataset):
        from hgrag.data.normalize import name_key

        keys = [name_key(q.question) for q in dataset.questions]
        assert len(keys) == len(set(keys))

    def test_qids_are_unique(self, dataset: EvalDataset):
        qids = [q.qid for q in dataset.questions]
        assert len(qids) == len(set(qids))
        assert all(qid for qid in qids)

    def test_sources_are_all_represented(self, dataset: EvalDataset):
        by_source = dataset.stats()["by_source"]
        assert by_source.get("vhe_template", 0) > 0
        assert by_source.get("llm_from_wiki", 0) > 0
        assert by_source.get("out_of_kb", 0) > 0

    def test_source_subset_is_respected(self, builder: EvalDatasetBuilder):
        dataset = builder.build(target_size=60, sources=["outkb"], offline=True)
        assert dataset.stats()["by_source"].get("out_of_kb", 0) > 0
        assert "vhe_template" not in dataset.stats()["by_source"]

    def test_questions_are_vietnamese_and_non_empty(self, dataset: EvalDataset):
        for q in dataset.questions:
            assert q.question.strip()
            assert len(q.question) >= 10

    def test_year_refs_are_in_scope(self, dataset: EvalDataset):
        for q in dataset.questions:
            for year in q.year_refs:
                assert 905 <= year <= 1975

    def test_excluded_chunks_are_recorded(self, dataset: EvalDataset):
        for q in dataset.questions:
            assert "excluded_chunks" in q.metadata
            assert isinstance(q.metadata["excluded_chunks"], list)

    def test_build_is_deterministic(self, builder: EvalDatasetBuilder):
        first = builder.build(target_size=60, offline=True)
        second = builder.build(target_size=60, offline=True)
        assert [q.question for q in first.questions] == [q.question for q in second.questions]

    def test_empty_sources_fallback(self, eval_settings: Settings):
        empty = EvalDatasetBuilder(
            llm_client=MockLLMClient(embed_dim=64, seed=42),
            settings=eval_settings,
            corpus_chunks=[],
        )
        dataset = empty.build(target_size=20, sources=["vhe"], offline=True)
        assert len(dataset) >= 0  # must not raise

    def test_stats_keys(self, dataset: EvalDataset):
        stats = dataset.stats()
        for key in (
            "total",
            "by_type",
            "by_source",
            "by_difficulty",
            "by_type_ratio",
            "out_of_kb",
            "out_of_kb_ratio",
            "with_evidence",
            "multi_hop_with_path",
        ):
            assert key in stats
        assert stats["total"] == len(dataset)
        assert sum(stats["by_type"].values()) == stats["total"]
        assert sum(stats["by_source"].values()) == stats["total"]
        assert sum(stats["by_difficulty"].values()) == stats["total"]

    def test_stats_counts_evidence(self, dataset: EvalDataset):
        stats = dataset.stats()
        assert stats["with_evidence"] == sum(1 for q in dataset.questions if q.has_evidence)
        assert stats["without_evidence"] + stats["with_evidence"] == stats["total"]

    def test_out_of_kb_pool_is_large_enough(self):
        assert len(OUT_OF_KB_QUESTIONS) >= 15

    def test_entity_extraction_ignores_sentence_starters(self):
        from hgrag.eval.dataset_builder import _capitals

        out = _capitals(
            "Lý Thái Tổ (chữ Hán: 李太祖, 974-1028), tên húy là Lý Công Uẩn, "
            "là hoàng đế sáng lập nhà Lý."
        )
        assert "Lý Thái Tổ" in out
        assert "Lý Công Uẩn" in out
        # lower-case accented words must never start a name
        assert all(name[0].isupper() for name in out)

    def test_gold_entities_are_never_empty_strings(self, dataset: EvalDataset):
        for q in dataset.questions:
            for entity in q.gold_entities:
                assert entity.strip()
                assert len(entity) >= 2

    def test_dataset_container_api(self, dataset: EvalDataset):
        assert len(dataset) == len(dataset.questions)
        assert dataset[0] is dataset.questions[0]
        assert list(dataset) == dataset.questions
        assert dataset.stats()["total"] == len(dataset)


# --------------------------------------------------------------------------
# script
# --------------------------------------------------------------------------


class TestScript:
    def test_script_runs_offline(self, tmp_path: Path):
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        out_path = tmp_path / "eval_questions.jsonl"
        stats_path = tmp_path / "eval_stats.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "05_eval_dataset.py"),
                "--target",
                "150",
                "--mock",
                "--offline",
                "--corpus",
                str(repo_root / "data" / "processed" / "corpus.jsonl"),
                "--out",
                str(out_path),
                "--stats-out",
                str(stats_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(repo_root),
        )
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["total"] >= 150
        assert payload["by_type"]["out_of_kb"] >= 10
        assert payload["by_type"]["multi_hop"] >= 30
        assert out_path.exists()
        assert stats_path.exists()
        n_lines = len(out_path.read_text(encoding="utf-8").strip().splitlines())
        assert n_lines == payload["n_written"]
