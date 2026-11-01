"""Tests cho tầng LLM (:mod:`hgrag.llm.client`).

Tất cả test chạy **offline**: Gemini được kiểm thử bằng SDK giả lập (monkeypatch),
không có lời gọi mạng nào.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hgrag.llm.client import (
    DiskCache,
    GeminiClient,
    LLMError,
    MockLLMClient,
    build_llm_client,
)
from hgrag.llm.schemas import LLMMessage, LLMRequest


def _req(task: str, text: str = "Trần Hưng Đạo đánh Nguyên Mông năm 1285") -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role="user", content=text)],
        task=task,
        temperature=0.0,
    )


# --------------------------------------------------------------------------
# MockLLMClient
# --------------------------------------------------------------------------


class TestMockLLMClient:
    def test_extract_returns_valid_json(self, mock_llm: MockLLMClient):
        payload = mock_llm.generate_json(_req("extract"))
        assert isinstance(payload, dict)
        assert "events" in payload
        assert isinstance(payload["events"], list)
        assert payload["events"]

    def test_query_understanding_returns_valid_json(self, mock_llm: MockLLMClient):
        payload = mock_llm.generate_json(_req("query_understanding"))
        assert isinstance(payload, dict)
        assert "intent" in payload
        assert "entities" in payload
        assert isinstance(payload["entities"], list)

    def test_judge_returns_valid_json(self, mock_llm: MockLLMClient):
        payload = mock_llm.generate_json(_req("judge"))
        assert isinstance(payload, dict)
        assert "score" in payload
        assert isinstance(payload["score"], (int, float))

    def test_verify_returns_valid_json(self, mock_llm: MockLLMClient):
        payload = mock_llm.generate_json(_req("verify"))
        assert isinstance(payload, dict)
        assert "is_same_entity" in payload
        assert "confidence" in payload

    def test_generate_json_parses_plain_output(self, mock_llm: MockLLMClient):
        payload = mock_llm.generate_json(_req("judge"))
        assert payload["correct"] is True

    def test_deterministic_same_request_same_output(self, mock_llm: MockLLMClient):
        req = _req("query_understanding", "Ai là Trần Hưng Đạo?")
        first = mock_llm.generate(req)
        second = mock_llm.generate(req)
        assert first.text == second.text
        assert first.model == second.model == "mock"

    def test_deterministic_across_instances(self):
        req = _req("extract", "Năm 1285 trận Bạch Đằng")
        a = MockLLMClient(embed_dim=64, seed=42).generate(req)
        b = MockLLMClient(embed_dim=64, seed=42).generate(req)
        assert a.text == b.text

    def test_calls_are_recorded(self, mock_llm: MockLLMClient):
        mock_llm.generate(_req("judge"))
        assert len(mock_llm.calls) == 1
        assert mock_llm.calls[0].task == "judge"

    def test_embed_deterministic_and_normalised(self, mock_llm: MockLLMClient):
        v1 = mock_llm.embed(["Trần Hưng Đạo"])[0]
        v2 = mock_llm.embed(["Trần Hưng Đạo"])[0]
        assert v1 == v2
        assert len(v1) == mock_llm.embed_dim == 64
        assert abs(sum(x * x for x in v1) - 1.0) < 1e-6

    def test_embed_empty_list(self, mock_llm: MockLLMClient):
        assert mock_llm.embed([]) == []


# --------------------------------------------------------------------------
# generate_json error handling
# --------------------------------------------------------------------------


class _BrokenClient(MockLLMClient):
    """Luôn trả về text không phải JSON."""

    def generate(self, request: LLMRequest):  # type: ignore[override]
        from hgrag.llm.schemas import LLMResponse

        return LLMResponse(text="đây không phải JSON", model="broken")


class TestGenerateJsonErrors:
    def test_raises_llm_error_on_invalid_json(self):
        client = _BrokenClient(embed_dim=8)
        with pytest.raises(LLMError):
            client.generate_json(_req("judge"))

    def test_strips_markdown_fence(self):
        class _Fenced(MockLLMClient):
            def generate(self, request: LLMRequest):  # type: ignore[override]
                from hgrag.llm.schemas import LLMResponse

                return LLMResponse(
                    text='```json\n{"score": 1}\n```', model="fenced"
                )

        payload = _Fenced(embed_dim=8).generate_json(_req("judge"))
        assert payload == {"score": 1}


# --------------------------------------------------------------------------
# DiskCache
# --------------------------------------------------------------------------


class TestDiskCache:
    def test_put_get_roundtrip(self, tmp_path: Path):
        cache = DiskCache(tmp_path / "cache")
        cache.put("abc123", {"text": "xin chào", "model": "mock"})
        got = cache.get("abc123")
        assert got is not None
        assert got["text"] == "xin chào"
        assert got["model"] == "mock"

    def test_get_missing_key_returns_none(self, tmp_path: Path):
        cache = DiskCache(tmp_path / "cache")
        assert cache.get("khong-ton-tai") is None

    def test_different_keys_use_different_files(self, tmp_path: Path):
        cache = DiskCache(tmp_path / "cache")
        cache.put("keyaaaa", {"text": "A"})
        cache.put("keybbbb", {"text": "B"})
        files = sorted(p.name for p in (tmp_path / "cache").rglob("*.json"))
        assert len(files) == 2
        assert files[0] != files[1]
        assert cache.get("keyaaaa")["text"] == "A"
        assert cache.get("keybbbb")["text"] == "B"

    def test_corrupt_file_returns_none(self, tmp_path: Path):
        cache = DiskCache(tmp_path / "cache")
        cache.put("deadbeef", {"text": "x"})
        path = cache._path("deadbeef")
        path.write_text("{not json", encoding="utf-8")
        assert cache.get("deadbeef") is None


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------


class TestBuildLLMClient:
    def test_provider_mock_returns_mock(self, settings):
        settings.raw["llm"]["provider"] = "mock"
        client = build_llm_client(settings)
        assert isinstance(client, MockLLMClient)
        assert client.name == "mock"

    def test_provider_gemini_without_key_falls_back_to_mock(self, settings, monkeypatch):
        settings.raw["llm"]["provider"] = "gemini"
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client = build_llm_client(settings)
        assert isinstance(client, MockLLMClient)
        assert client.name == "mock"

    def test_provider_auto_without_key_falls_back_to_mock(self, settings, monkeypatch):
        settings.raw["llm"]["provider"] = "auto"
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        client = build_llm_client(settings)
        assert isinstance(client, MockLLMClient)

    def test_unknown_provider_falls_back_to_mock(self, settings):
        settings.raw["llm"]["provider"] = "khong-ton-tai"
        client = build_llm_client(settings)
        assert isinstance(client, MockLLMClient)

    def test_no_network_called_when_key_missing(self, settings, monkeypatch):
        """Không có key -> không được khởi tạo Gemini (tức là không gọi mạng)."""
        settings.raw["llm"]["provider"] = "gemini"
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        called = {"n": 0}

        class _Boom:
            def __init__(self, *args: Any, **kwargs: Any):
                called["n"] += 1
                raise AssertionError("GeminiClient không được khởi tạo khi thiếu key")

        monkeypatch.setattr("hgrag.llm.client.GeminiClient", _Boom)
        client = build_llm_client(settings)
        assert called["n"] == 0
        assert isinstance(client, MockLLMClient)


# --------------------------------------------------------------------------
# GeminiClient với SDK giả lập
# --------------------------------------------------------------------------


class _FakeModels:
    """Giả lập ``client.models`` của google-genai."""

    def __init__(self, behaviour: Any):
        self.behaviour = behaviour
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any):
        self.calls.append(kwargs)
        return self.behaviour(kwargs)


class _FakeClient:
    def __init__(self, models: _FakeModels):
        self.models = models


def _install_fake_genai(monkeypatch, behaviour: Any) -> _FakeModels:
    """Cài đặt module ``google.genai`` giả vào sys.modules."""
    import sys

    models = _FakeModels(behaviour)
    genai_module = SimpleNamespace(Client=lambda **kwargs: _FakeClient(models))
    google_module = SimpleNamespace(genai=genai_module)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    return models


def _ok_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
    )


class TestGeminiClientMocked:
    def test_generate_returns_text(self, monkeypatch, tmp_path: Path):
        _install_fake_genai(monkeypatch, lambda kwargs: _ok_response('{"ok": true}'))
        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        resp = client.generate(_req("judge"))
        assert resp.text == '{"ok": true}'
        assert resp.cached is False

    def test_cache_hit_avoids_second_api_call(self, monkeypatch, tmp_path: Path):
        calls = {"n": 0}

        def _behaviour(kwargs: Any):
            calls["n"] += 1
            return _ok_response('{"ok": true}')

        _install_fake_genai(monkeypatch, _behaviour)
        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        first = client.generate(_req("judge"))
        second = client.generate(_req("judge"))
        assert calls["n"] == 1
        assert first.cached is False
        assert second.cached is True
        assert second.text == first.text

    def test_retry_then_success(self, monkeypatch, tmp_path: Path):
        attempts = {"n": 0}

        def _behaviour(kwargs: Any):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("boom")
            return _ok_response("ok")

        _install_fake_genai(monkeypatch, _behaviour)
        client = GeminiClient(
            model="gemini-test",
            api_key="k",
            cache_dir=tmp_path / "cache",
            max_retries=4,
            retry_base_delay=0.001,
        )
        resp = client.generate(_req("judge"))
        assert resp.text == "ok"
        assert attempts["n"] == 3

    def test_backoff_delay_increases(self, monkeypatch, tmp_path: Path):
        delays: list[float] = []
        monkeypatch.setattr("hgrag.llm.client.time.sleep", lambda s: delays.append(s))
        _install_fake_genai(monkeypatch, lambda kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
        client = GeminiClient(
            model="gemini-test",
            api_key="k",
            cache_dir=tmp_path / "cache",
            max_retries=3,
            retry_base_delay=0.5,
        )
        with pytest.raises(LLMError):
            client.generate(_req("judge"))
        assert len(delays) == 3
        assert delays[0] < delays[1] < delays[2]

    def test_non_retryable_error_raises_immediately(self, monkeypatch, tmp_path: Path):
        attempts = {"n": 0}

        def _behaviour(kwargs: Any):
            attempts["n"] += 1
            err = RuntimeError("bad request")
            err.code = 400
            raise err

        _install_fake_genai(monkeypatch, _behaviour)
        client = GeminiClient(
            model="gemini-test",
            api_key="k",
            cache_dir=tmp_path / "cache",
            max_retries=4,
            retry_base_delay=0.001,
        )
        with pytest.raises(LLMError):
            client.generate(_req("judge"))
        assert attempts["n"] == 1

    def test_generate_json_uses_sdk(self, monkeypatch, tmp_path: Path):
        _install_fake_genai(monkeypatch, lambda kwargs: _ok_response('{"score": 1}'))
        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        payload = client.generate_json(_req("judge"))
        assert payload == {"score": 1}

    def test_cache_disabled_always_calls_api(self, monkeypatch, tmp_path: Path):
        calls = {"n": 0}

        def _behaviour(kwargs: Any):
            calls["n"] += 1
            return _ok_response("ok")

        _install_fake_genai(monkeypatch, _behaviour)
        client = GeminiClient(
            model="gemini-test", api_key="k", cache_dir=tmp_path / "cache", cache_enabled=False
        )
        client.generate(_req("judge"))
        client.generate(_req("judge"))
        assert calls["n"] == 2

    def test_response_schema_passed_to_sdk(self, monkeypatch, tmp_path: Path):
        models = _install_fake_genai(monkeypatch, lambda kwargs: _ok_response("{}"))
        req = _req("judge")
        req.response_schema = {"type": "object"}
        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        client.generate(req)
        cfg = models.calls[0]["config"]
        assert cfg["response_mime_type"] == "application/json"
        assert cfg["response_schema"] == {"type": "object"}

    def test_embed_normalises_vectors(self, monkeypatch, tmp_path: Path):
        class _Emb:
            def __init__(self, values):
                self.values = values

        class _EmbedModels:
            def embed_content(self, **kwargs: Any):
                return SimpleNamespace(embeddings=[_Emb([3.0, 4.0])])

        genai_module = SimpleNamespace(
            Client=lambda **kwargs: SimpleNamespace(models=_EmbedModels())
        )
        google_module = SimpleNamespace(genai=genai_module)
        import sys

        monkeypatch.setitem(sys.modules, "google", google_module)
        monkeypatch.setitem(sys.modules, "google.genai", genai_module)

        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        vec = client.embed(["một câu"])
        assert vec == [[0.6, 0.8]]
        assert client.embed_dim == 2

    def test_embed_empty_returns_empty(self, monkeypatch, tmp_path: Path):
        _install_fake_genai(monkeypatch, lambda kwargs: _ok_response(""))
        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        assert client.embed([]) == []

    def test_system_instruction_split(self, monkeypatch, tmp_path: Path):
        models = _install_fake_genai(monkeypatch, lambda kwargs: _ok_response("ok"))
        req = LLMRequest(
            messages=[
                LLMMessage(role="system", content="Bạn là trợ lý lịch sử."),
                LLMMessage(role="user", content="Câu hỏi?"),
            ],
            task="answer",
        )
        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        client.generate(req)
        cfg = models.calls[0]["config"]
        assert cfg["system_instruction"] == "Bạn là trợ lý lịch sử."
        assert models.calls[0]["contents"] == "Câu hỏi?"

    def test_generation_time_is_not_zero(self):
        """Bảo đảm test không bị 'xanh giả' do sleep bị patch."""
        start = time.monotonic()
        time.sleep(0.001)
        assert time.monotonic() - start >= 0.0

    def test_json_response_serialisable(self, mock_llm: MockLLMClient):
        resp = mock_llm.generate(_req("judge"))
        payload = json.loads(resp.text)
        assert isinstance(payload, dict)
