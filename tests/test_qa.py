"""Tests for the QA layer (Wave 4): query understanding, grounded answers,
explanations and the end-to-end pipeline (including baselines + ablations).

Everything runs offline: the LLM is mocked, the KG lives in memory and the
vector store is a numpy fake.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from hgrag.data.records import Evidence
from hgrag.kg.alias import AliasResolver
from hgrag.llm.client import MockLLMClient
from hgrag.qa.answer import (
    AnswerGenerator,
    is_degenerate,
    is_refusal,
    parse_reasoning_chain,
    validate_citations,
)
from hgrag.qa.explain import Explainer, format_paths
from hgrag.qa.pipeline import ABLATION_FLAGS, REFUSAL_ANSWER, QAPipeline
from hgrag.qa.query_understanding import (
    QueryUnderstanding,
    QueryUnderstandingModule,
    detect_intent,
    extract_years,
)
from hgrag.retrieval.entity_linker import EntityLinker
from hgrag.retrieval.graph_search import GraphSearcher
from hgrag.retrieval.rerank import HeuristicReranker

QUESTION = "Trần Hưng Đạo đánh bại quân Mông Cổ vào năm nào?"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def understanding_module(mock_llm: MockLLMClient) -> QueryUnderstandingModule:
    return QueryUnderstandingModule(mock_llm)


@pytest.fixture()
def pipeline(
    settings: Any,
    mock_llm: MockLLMClient,
    mock_embedder: Any,
    in_memory_graph: Any,
    mock_vector_store: Any,
) -> QAPipeline:
    """A fully offline pipeline wired to the in-memory KG."""
    linker = EntityLinker(
        resolver=AliasResolver(store=in_memory_graph, llm_client=mock_llm, settings=settings),
        settings=settings,
    )
    searcher = GraphSearcher(store=in_memory_graph, settings=settings, embedder=mock_embedder)
    return QAPipeline(
        llm_client=mock_llm,
        embedder=mock_embedder,
        graph_store=in_memory_graph,
        vector_store=mock_vector_store,
        entity_linker=linker,
        graph_searcher=searcher,
        reranker=HeuristicReranker(),
        settings=settings,
    )


# --------------------------------------------------------------------------
# query understanding
# --------------------------------------------------------------------------


def test_extract_years_only_history_range() -> None:
    assert extract_years("năm 1285 và 1975") == [1285, 1975]
    assert extract_years("năm 2024 và 800") == []
    assert extract_years("không có năm") == []


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Trần Hưng Đạo đánh bại quân Mông Cổ vào năm nào?", "temporal"),
        ("Vì sao nhà Trần thua trận?", "causal"),
        ("So sánh Lý Thường Kiệt và Trần Hưng Đạo", "comparison"),
        ("Ai là người lập ra nhà Ngô?", "factual"),
    ],
)
def test_detect_intent(question: str, intent: str) -> None:
    assert detect_intent(question) == intent


def test_heuristic_understanding_fallback() -> None:
    module = QueryUnderstandingModule(None)
    understanding = module.understand(QUESTION)
    assert isinstance(understanding, QueryUnderstanding)
    assert understanding.source == "heuristic"
    assert understanding.time_refs == []
    assert understanding.relation_plan  # non-empty for a known intent


def test_understanding_uses_llm_when_available(understanding_module) -> None:
    understanding = understanding_module.understand(QUESTION)
    assert understanding.source == "llm"
    assert understanding.intent
    assert isinstance(understanding.expected_hops, int)


def test_understanding_rejects_empty_entity_payload() -> None:
    class _BadLLM:
        def generate_json(self, request):  # noqa: D102
            return {"intent": "factual", "entities": []}

    understanding = QueryUnderstandingModule(_BadLLM()).understand(QUESTION)
    assert understanding.source == "heuristic"


# --------------------------------------------------------------------------
# citations / refusal / reasoning
# --------------------------------------------------------------------------


def test_validate_citations_splits_valid_and_invalid() -> None:
    valid, invalid = validate_citations("A [E1] và B [E9] và C [E2]", ["E1", "E2"])
    assert valid == ["E1", "E2"]
    assert invalid == ["E9"]


def test_is_refusal_detects_vietnamese_refusal() -> None:
    assert is_refusal("Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.")
    assert not is_refusal("Trận Bạch Đằng diễn ra năm 1285.")


def test_parse_reasoning_chain_from_heading() -> None:
    response = "Câu trả lời [E1].\n\n## Suy luận\n1. Bước một [E1]\n2. Bước hai [E2]"
    steps = parse_reasoning_chain(response)
    assert steps == ["Bước một [E1]", "Bước hai [E2]"]


def test_answer_generator_refuses_without_context(mock_llm, settings) -> None:
    generator = AnswerGenerator(mock_llm, settings)
    result = generator.generate(QUESTION, "", evidence=[])
    assert result.refused
    assert result.answer == REFUSAL_ANSWER
    assert result.citations == []


def test_answer_generator_cites_evidence(mock_llm, settings) -> None:
    generator = AnswerGenerator(mock_llm, settings)
    evidence = [
        Evidence(evidence_id="E1", kind="chunk", text="Trận Bạch Đằng năm 1285."),
        Evidence(evidence_id="E2", kind="chunk", text="Trần Hưng Đạo chỉ huy quân."),
    ]
    result = generator.generate(
        QUESTION, "[E1] ... [E2] ...", evidence=evidence
    )
    assert result.citations
    assert set(result.citations) <= {"E1", "E2"}
    assert not result.invalid_citations


def test_answer_generator_strips_reasoning_from_answer_body(mock_llm, settings) -> None:
    generator = AnswerGenerator(mock_llm, settings)
    result = generator.generate(
        QUESTION,
        "[E1] nội dung",
        evidence=[Evidence(evidence_id="E1", kind="chunk", text="x")],
    )
    assert "## Suy luận" not in result.answer


# --------------------------------------------------------------------------
# baseline B0 (closed-book) -- JSON mode + bộ phát hiện lặp
# --------------------------------------------------------------------------


class _ScriptedLLM:
    """A fake client that returns a canned payload per ``task``."""

    name = "scripted"

    def __init__(self, payloads: dict[str, str]) -> None:
        self.payloads = payloads
        self.calls: list[Any] = []

    def generate(self, request: Any) -> Any:
        self.calls.append(request)
        from hgrag.llm.schemas import LLMResponse

        return LLMResponse(text=self.payloads.get(request.task, ""), model="scripted")


def test_closed_book_uses_json_schema() -> None:
    """B0 phải dùng JSON mode: có schema thì API tắt kênh suy luận."""
    llm = _ScriptedLLM({"answer_closed_book": '{"answer": "Hùng Vương."}'})
    generator = AnswerGenerator(llm, None)
    result = generator.generate_closed_book("Ai là người thành lập Văn Lang?")
    assert result.answer == "Hùng Vương."
    assert llm.calls[0].response_schema is not None
    assert llm.calls[0].response_schema["properties"]["answer"]["type"] == "string"


def test_closed_book_parses_reasoning_steps() -> None:
    llm = _ScriptedLLM(
        {
            "answer_closed_book": json.dumps(
                {
                    "answer": "Hùng Vương.",
                    "reasoning_steps": ["Bước một", "Bước hai"],
                },
                ensure_ascii=False,
            )
        }
    )
    generator = AnswerGenerator(llm, None)
    result = generator.generate_closed_book("Ai là người thành lập Văn Lang?")
    assert result.reasoning_chain == ["Bước một", "Bước hai"]
    assert "Hùng Vương." in result.answer


def test_closed_book_rejects_degenerate_answer() -> None:
    """Câu trả lời lặp (Gemma rơi vào vòng lặp) không được dùng làm đáp án."""
    llm = _ScriptedLLM(
        {
            "answer_closed_book": json.dumps(
                {"answer": "phất phất phất phất phất phất phất phất"}, ensure_ascii=False
            ),
            "answer_closed_book_text": "phất phất phất phất phất phất phất phất",
        }
    )
    generator = AnswerGenerator(llm, None)
    result = generator.generate_closed_book("Ai là người sinh ra Đế Nghi?")
    assert result.refused
    assert result.answer == REFUSAL_ANSWER


def test_closed_book_falls_back_to_plain_text() -> None:
    """Khi JSON hỏng thì thử lại bằng prompt văn bản thuần."""
    llm = _ScriptedLLM(
        {
            "answer_closed_book": "không phải JSON",
            "answer_closed_book_text": "Đế Minh là người sinh ra Đế Nghi.",
        }
    )
    generator = AnswerGenerator(llm, None)
    result = generator.generate_closed_book("Ai là người sinh ra Đế Nghi?")
    assert "Đế Minh" in result.answer
    assert not result.refused


@pytest.mark.parametrize(
    ("text", "degenerate"),
    [
        ("phất phất phất phất phất phất", True),
        (
            "theo truyền thuyết, theo truyền thuyết, theo truyền thuyết, "
            "theo truyền thuyết, theo truyền thuyết, x",
            True,
        ),
        ("Trần Hưng Đạo chỉ huy quân tại trận Bạch Đằng năm 1285.", False),
        ("Hùng Vương là người thành lập Văn Lang.", False),
        ("Đế Minh sinh ra Đế Nghi. Đế Minh sinh ra Kinh Dương Vương.", False),
    ],
)
def test_is_degenerate(text: str, degenerate: bool) -> None:
    assert is_degenerate(text) is degenerate


# --------------------------------------------------------------------------
# explanation
# --------------------------------------------------------------------------


def test_explainer_returns_nodes_and_edges(pipeline, in_memory_graph) -> None:
    explainer = Explainer(pipeline.graph_searcher)
    understanding = pipeline._understand(QUESTION)
    linked = pipeline._link(understanding, {})
    graph_result = pipeline._graph_search(linked, understanding, 2, {})
    explanation = explainer.explain(QUESTION, linked, graph_result, {"n_vector_hits": 3})
    payload = explanation.to_dict()
    assert "nodes" in payload["subgraph"]
    assert "edges" in payload["subgraph"]
    assert payload["retrieval_trace"]["n_vector_hits"] == 3


def test_format_paths_handles_empty_input() -> None:
    assert format_paths([]) == []


# --------------------------------------------------------------------------
# end-to-end pipeline
# --------------------------------------------------------------------------


def test_pipeline_ask_returns_grounded_answer(pipeline) -> None:
    result = pipeline.ask(QUESTION)
    assert result.question == QUESTION
    assert result.answer
    assert result.system == "ours"
    assert result.evidence
    assert result.elapsed >= 0


def test_pipeline_ask_includes_citations_and_chain(pipeline) -> None:
    result = pipeline.ask(QUESTION)
    assert result.citations
    assert all(c.startswith("E") for c in result.citations)


def test_pipeline_refuses_when_no_context(pipeline) -> None:
    class _EmptyStore:
        def search(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return []

    pipeline.vector_store = _EmptyStore()
    pipeline.entity_linker = None
    result = pipeline.ask("Một câu hỏi hoàn toàn ngoài kho kiến thức lịch sử?")
    assert result.refused
    assert result.answer == REFUSAL_ANSWER


@pytest.mark.parametrize("flag", ABLATION_FLAGS)
def test_each_ablation_flag_is_accepted(pipeline, flag: str) -> None:
    result = pipeline.ask(QUESTION, ablation={flag: True})
    assert result.answer
    assert result.explanation.get("retrieval_trace", {}).get("ablation", {}).get(flag) is True


def test_no_graph_ablation_drops_paths(pipeline) -> None:
    with_graph = pipeline.ask(QUESTION)
    without = pipeline.ask(QUESTION, ablation={"no_graph": True})
    assert with_graph.explanation.get("retrieval_trace", {}).get("n_graph_paths", 0) >= 0
    assert without.explanation.get("retrieval_trace", {}).get("n_graph_paths", 0) == 0


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------


def test_baseline_b0_llm_only(pipeline) -> None:
    result = pipeline.answer_llm_only(QUESTION)
    assert result.system == "B0"
    assert result.answer
    assert result.evidence == []


def test_baseline_b1_vanilla_rag(pipeline) -> None:
    result = pipeline.answer_vanilla_rag(QUESTION)
    assert result.system == "B1"
    assert result.evidence


def test_baseline_b2_rag_plus_graph(pipeline) -> None:
    result = pipeline.answer_rag_plus_graph(QUESTION)
    assert result.system == "B2"
    assert result.evidence
    assert "nodes" in result.explanation["subgraph"]


def test_qa_result_to_dict_is_json_serialisable(pipeline) -> None:
    import json

    result = pipeline.ask(QUESTION)
    payload = result.to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert payload["system"] == "ours"
