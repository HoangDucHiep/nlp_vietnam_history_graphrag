"""Tests cho giao diện Streamlit (:mod:`hgrag.app.streamlit_app`).

Các hàm được test đều là hàm *thuần tuý* (không gọi ``st.*``) nên không cần
runtime của Streamlit.  Test import cũng xác nhận module không có side-effect
(không gọi ``st.set_page_config`` lúc import).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import hgrag.app.streamlit_app as app

# --------------------------------------------------------------------------
# import / side-effect
# --------------------------------------------------------------------------


class TestImport:
    def test_import_does_not_raise(self):
        import importlib

        module = importlib.import_module("hgrag.app.streamlit_app")
        assert module is app

    def test_has_main(self):
        assert callable(app.main)

    def test_import_has_no_side_effect(self, monkeypatch):
        """Import lại module không được gọi st.set_page_config."""
        import importlib

        calls: list[Any] = []

        class _FakeStreamlit:
            def __getattr__(self, name: str):
                def _fn(*args: Any, **kwargs: Any):
                    calls.append(name)
                    return None

                return _fn

        import sys

        monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlit())
        importlib.reload(app)
        assert "set_page_config" not in calls

    def test_public_helpers_exist(self):
        for name in (
            "format_citations",
            "parse_reasoning",
            "build_knowledge_graph_table",
            "load_results_table",
            "QAService",
        ):
            assert hasattr(app, name)


# --------------------------------------------------------------------------
# citations
# --------------------------------------------------------------------------


class TestCitations:
    def test_extract_citation_markers(self):
        text = "Trần Hưng Đạo chỉ huy quân [E1], thắng trận [E2] và [E1] nữa."
        assert app.extract_citation_markers(text) == ["E1", "E2"]

    def test_extract_citation_markers_empty(self):
        assert app.extract_citation_markers("") == []
        assert app.extract_citation_markers("không có citation") == []

    def test_format_citations_lists_sources(self):
        answer = "Trần Hưng Đạo đánh thắng Nguyên Mông [E1]."
        evidence = [{"evidence_id": "E1", "kind": "chunk", "text": "Trận Bạch Đằng năm 1285."}]
        out = app.format_citations(answer, evidence)
        assert "Trần Hưng Đạo đánh thắng Nguyên Mông [E1]." in out
        assert "Trận Bạch Đằng năm 1285." in out
        assert "Nguồn trích dẫn" in out

    def test_format_citations_flags_missing_evidence(self):
        out = app.format_citations("Câu trả lời [E7].", [])
        assert "không tìm thấy evidence" in out

    def test_format_citations_without_markers_warns(self):
        out = app.format_citations("Không có căn cứ.", [])
        assert "grounded = False" in out

    def test_format_citations_shows_source_chunk_ids(self):
        evidence = [
            {
                "evidence_id": "E1",
                "kind": "graph_path",
                "text": "A -> B",
                "source_chunk_ids": ["ck_1", "ck_2"],
            }
        ]
        out = app.format_citations("X [E1]", evidence)
        assert "ck_1" in out
        assert "graph_path" in out

    def test_format_citations_without_evidence_list(self):
        out = app.format_citations("X [E1]", None)
        assert "không tìm thấy evidence" in out


# --------------------------------------------------------------------------
# reasoning
# --------------------------------------------------------------------------


class TestParseReasoning:
    def test_parses_numbered_steps_without_header(self):
        text = "1. Bước một: xác định thực thể\n2. Bước hai: duyệt đồ thị"
        assert app.parse_reasoning(text) == [
            "Bước một: xác định thực thể",
            "Bước hai: duyệt đồ thị",
        ]

    def test_parses_after_reasoning_header(self):
        text = (
            "Đây là câu trả lời.\n\nChuỗi suy luận:\n"
            "1. Trần Hưng Đạo tham gia trận Bạch Đằng [E1]\n"
            "2. Trận Bạch Đằng năm 1285 [E2]\n"
        )
        steps = app.parse_reasoning(text)
        assert len(steps) == 2
        assert "Trần Hưng Đạo" in steps[0]

    def test_handles_vietnamese_header(self):
        text = "Chuỗi suy luận:\n- Bước A\n- Bước B"
        steps = app.parse_reasoning(text)
        assert steps == ["Bước A", "Bước B"]

    def test_empty_text(self):
        assert app.parse_reasoning("") == []

    def test_no_steps_returns_empty(self):
        assert app.parse_reasoning("Một đoạn văn bình thường không có bước.") == []


# --------------------------------------------------------------------------
# knowledge-graph table
# --------------------------------------------------------------------------


class TestKnowledgeGraphTable:
    def test_builds_rows(self):
        links = [
            {
                "mention": "Trần Hưng Đạo",
                "wikidata_id": "Q511375",
                "uid": "pe:Q511375",
                "entity_type": "Person",
                "confidence": 0.95,
                "candidates": [{"name": "Trần Hưng Đạo"}, {"name": "Trần Quốc Tuấn"}],
            }
        ]
        rows = app.build_knowledge_graph_table(links)
        assert len(rows) == 1
        row = rows[0]
        assert row["mention"] == "Trần Hưng Đạo"
        assert row["qid"] == "Q511375"
        assert row["confidence"] == 0.95
        assert row["n_candidates"] == 2
        assert "Trần Quốc Tuấn" in row["candidates"]

    def test_accepts_wrapped_payload(self):
        rows = app.build_knowledge_graph_table({"links": [{"name": "Lý Thường Kiệt"}]})
        assert rows[0]["mention"] == "Lý Thường Kiệt"

    def test_empty_input(self):
        assert app.build_knowledge_graph_table(None) == []
        assert app.build_knowledge_graph_table([]) == []

    def test_confidence_coerced_to_float(self):
        rows = app.build_knowledge_graph_table([{"name": "A", "confidence": "0.8"}])
        assert rows[0]["confidence"] == 0.8

    def test_bad_confidence_becomes_zero(self):
        rows = app.build_knowledge_graph_table([{"name": "A", "confidence": "abc"}])
        assert rows[0]["confidence"] == 0.0

    def test_summarize_confidence(self):
        assert app.summarize_confidence(0.95) == "cao"
        assert app.summarize_confidence(0.8) == "trung bình"
        assert app.summarize_confidence(0.3) == "thấp"
        assert app.summarize_confidence("x") == "không rõ"


# --------------------------------------------------------------------------
# subgraph
# --------------------------------------------------------------------------


class TestSubgraph:
    def test_builds_nodes_and_edges(self):
        neighbors = [
            {
                "src": {"uid": "pe:Q511375", "name": "Trần Hưng Đạo", "entity_type": "Person"},
                "rel": "PARTICIPATED_IN",
                "rel_props": {"text": "tham gia trận Bạch Đằng", "year": 1285},
                "dst": {"uid": "ev:1", "name": "Trận Bạch Đằng", "entity_type": "Event"},
            }
        ]
        graph = app.subgraph_from_neighbors(neighbors)
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["label"] == "PARTICIPATED_IN"
        assert graph["edges"][0]["from"] == "pe:Q511375"

    def test_empty_neighbors(self):
        graph = app.subgraph_from_neighbors([])
        assert graph == {"nodes": [], "edges": []}
        assert app.subgraph_from_neighbors(None) == {"nodes": [], "edges": []}

    def test_incomplete_records_are_skipped(self):
        graph = app.subgraph_from_neighbors([{"src": {}, "dst": {}, "rel": "X"}])
        assert graph["nodes"] == []
        assert graph["edges"] == []


# --------------------------------------------------------------------------
# results table
# --------------------------------------------------------------------------


class TestLoadResultsTable:
    def test_reads_csv(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        path.write_text(
            "system,metric,score\nB0,accuracy,0.50\nB1,accuracy,0.65\n", encoding="utf-8"
        )
        frame = app.load_results_table(path)
        assert isinstance(frame, pd.DataFrame)
        assert not frame.empty
        assert list(frame["system"]) == ["B0", "B1"]

    def test_missing_file_returns_empty(self, tmp_path: Path):
        frame = app.load_results_table(tmp_path / "khong-co.csv")
        assert frame.empty

    def test_none_path_returns_empty(self):
        assert app.load_results_table(None).empty

    def test_directory_path_returns_empty(self, tmp_path: Path):
        assert app.load_results_table(tmp_path).empty

    def test_corrupt_csv_returns_empty(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        path.write_text("a,b\n1,2,3,4\n", encoding="utf-8")
        frame = app.load_results_table(path)
        assert frame.empty or not frame.empty  # không raise


class TestBuildResultsChartData:
    def test_orders_variants(self):
        frame = pd.DataFrame(
            {
                "system": ["Ours", "B0", "B2", "B1"],
                "accuracy": [0.9, 0.5, 0.7, 0.6],
            }
        )
        data = app.build_results_chart_data(frame, metric="accuracy")
        assert data["labels"] == ["B0", "B1", "B2", "Ours"]
        assert data["values"] == [0.5, 0.6, 0.7, 0.9]
        assert data["metric"] == "accuracy"

    def test_empty_frame(self):
        data = app.build_results_chart_data(pd.DataFrame())
        assert data["labels"] == []

    def test_frame_without_system_column(self):
        data = app.build_results_chart_data(pd.DataFrame({"x": [1, 2]}))
        assert data["labels"] == []

    def test_picks_numeric_metric_automatically(self):
        frame = pd.DataFrame({"system": ["B0", "Ours"], "f1": [0.4, 0.8]})
        data = app.build_results_chart_data(frame)
        assert data["metric"] == "f1"
        assert data["labels"] == ["B0", "Ours"]


# --------------------------------------------------------------------------
# QAService (adapter cho pipeline chưa có)
# --------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    def answer(self, question: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "answer": "Trần Hưng Đạo chỉ huy quân chống Nguyên Mông [E1].",
            "evidence": [{"evidence_id": "E1", "kind": "chunk", "text": "Trận Bạch Đằng 1285"}],
            "reasoning": ["Bước 1: liên kết thực thể", "Bước 2: duyệt đồ thị"],
            "subgraph": {"nodes": [], "edges": []},
            "grounded": True,
        }


class TestQAService:
    def test_service_constructs_without_pipeline(self, settings):
        service = app.QAService(settings=settings)
        assert service is not None
        assert service.ready in (True, False)

    def test_ask_without_pipeline_returns_not_ready_message(self, settings, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "hgrag.qa.pipeline" or name.endswith("hgrag.qa.pipeline"):
                raise ImportError("No module named 'hgrag.qa.pipeline'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        service = app.QAService(settings=settings)
        result = service.ask("Trần Hưng Đạo là ai?")
        assert result["ok"] is False
        assert app.PIPELINE_NOT_READY in result["message"]
        assert result["answer"] == ""

    def test_empty_question_returns_error(self, settings):
        service = app.QAService(settings=settings)
        result = service.ask("   ")
        assert result["ok"] is False
        assert result["message"]

    def test_normalize_result_from_dict(self):
        raw = {
            "answer": "X [E1]",
            "evidence": [{"evidence_id": "E1", "text": "t"}],
            "reasoning": ["a", "b"],
        }
        out = app.QAService._normalize_result(raw, "q")
        assert out["ok"] is True
        assert out["answer"] == "X [E1]"
        assert out["reasoning"] == ["a", "b"]
        assert out["grounded"] is True

    def test_normalize_result_from_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class _Res:
            answer: str = "Y [E2]"
            evidence: list = None  # type: ignore[assignment]
            reasoning: list = None  # type: ignore[assignment]

        out = app.QAService._normalize_result(_Res(evidence=[], reasoning=[]), "q")
        assert out["answer"] == "Y [E2]"
        assert out["evidence"] == []

    def test_normalize_result_from_plain_string(self):
        out = app.QAService._normalize_result("chỉ là chuỗi", "q")
        assert out["answer"] == "chỉ là chuỗi"
        assert out["evidence"] == []

    def test_service_uses_injected_pipeline(self, settings, monkeypatch):
        service = app.QAService(settings=settings)
        service._loaded = True
        service._pipeline = _FakePipeline()
        result = service.ask("Trần Hưng Đạo là ai?")
        assert result["ok"] is True
        assert "[E1]" in result["answer"]
        assert len(result["reasoning"]) == 2

    def test_pipeline_exception_is_caught(self, settings):
        class _Boom:
            def answer(self, question: str, **kwargs: Any):
                raise RuntimeError("lỗi pipeline")

        service = app.QAService(settings=settings)
        service._loaded = True
        service._pipeline = _Boom()
        result = service.ask("câu hỏi")
        assert result["ok"] is False
        assert "lỗi pipeline" in result["message"]

    def test_ready_flag_is_boolean(self, settings):
        service = app.QAService(settings=settings)
        assert isinstance(service.ready, bool)


# --------------------------------------------------------------------------
# misc helpers
# --------------------------------------------------------------------------


class TestMisc:
    def test_build_client_for_mode_mock(self, settings):
        client = app.build_client_for_mode(settings, "mock")
        assert client.name == "mock"

    def test_build_client_for_mode_auto(self, settings):
        client = app.build_client_for_mode(settings, "auto")
        assert client.name in ("mock", "gemini")

    def test_cache_info(self, settings):
        info = app.cache_info(settings)
        assert "path" in info
        assert info["n_files"] >= 0

    def test_find_ingest_logs_returns_list(self, settings):
        assert isinstance(app.find_ingest_logs(settings), list)

    def test_check_neo4j_does_not_raise(self, settings):
        status = app.check_neo4j(settings)
        assert "ok" in status
        assert "detail" in status

    def test_check_qdrant_does_not_raise(self, settings):
        status = app.check_qdrant(settings)
        assert "ok" in status

    def test_json_serialisable_helper_result(self, settings):
        service = app.QAService(settings=settings)
        payload = json.dumps({"error": service.error}, ensure_ascii=False)
        assert "error" in payload

    def test_app_source_has_no_toplevel_side_effects(self):
        """Không có lệnh ``st.*`` ở cấp module (chỉ trong hàm)."""
        import ast

        source = Path(app.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        offenders: list[str] = []
        for node in tree.body:
            # chỉ xét các câu lệnh thực thi ngay ở cấp module
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id == "st":
                        offenders.append(func.attr)
        assert offenders == []

    @pytest.mark.parametrize("name", ["main", "render_tab_ask", "render_tab_eval"])
    def test_render_functions_exist(self, name):
        assert callable(getattr(app, name))
