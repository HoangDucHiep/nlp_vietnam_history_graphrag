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
    LLMClient,
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

    def test_recovers_json_wrapped_in_prose(self):
        class _Prose(MockLLMClient):
            def generate(self, request: LLMRequest):  # type: ignore[override]
                from hgrag.llm.schemas import LLMResponse

                return LLMResponse(
                    text='Kết quả: {"score": 1, "correct": true}', model="prose"
                )

        payload = _Prose(embed_dim=8).generate_json(_req("judge"))
        assert payload == {"score": 1, "correct": True}

    def test_recovers_a_truncated_json(self):
        """Gemma đôi khi cắt ngang JSON -> phải sửa lại được."""

        class _Truncated(MockLLMClient):
            def generate(self, request: LLMRequest):  # type: ignore[override]
                from hgrag.llm.schemas import LLMResponse

                return LLMResponse(
                    text='{"score": 2, "correct": true, "reason": "Đáp án đúng và đầy',
                    model="truncated",
                )

        payload = _Truncated(embed_dim=8).generate_json(_req("judge"))
        assert payload["score"] == 2
        assert payload["correct"] is True


class TestExtractJson:
    """The JSON repair helper of the LLM layer."""

    def test_pure_json_is_returned_unchanged(self):
        assert LLMClient._extract_json('{"a": 1}') == '{"a": 1}'

    def test_markdown_fence_is_stripped(self):
        assert LLMClient._extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_prose_around_the_object_is_stripped(self):
        assert LLMClient._extract_json('Kết quả: {"a": 1}') == '{"a": 1}'

    def test_truncated_object_is_closed(self):
        repaired = LLMClient._extract_json('{"score": 2, "reason": "cắt ngang')
        assert repaired is not None
        assert json.loads(repaired)["score"] == 2

    def test_no_json_at_all_returns_none(self):
        assert LLMClient._extract_json("không có json ở đây") is None

    def test_empty_text_returns_none(self):
        assert LLMClient._extract_json("") is None

    def test_nested_objects_are_handled(self):
        payload = {"a": {"b": [1, 2]}, "c": "x"}
        assert json.loads(LLMClient._extract_json(json.dumps(payload))) == payload


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

    def test_embed_is_cached_on_disk(self, monkeypatch, tmp_path: Path):
        calls = {"n": 0}

        class _Emb:
            def __init__(self, values):
                self.values = values

        class _EmbedModels:
            def embed_content(self, **kwargs: Any):
                calls["n"] += 1
                return SimpleNamespace(embeddings=[_Emb([3.0, 4.0])])

        genai_module = SimpleNamespace(
            Client=lambda **kwargs: SimpleNamespace(models=_EmbedModels())
        )
        google_module = SimpleNamespace(genai=genai_module)
        import sys

        monkeypatch.setitem(sys.modules, "google", google_module)
        monkeypatch.setitem(sys.modules, "google.genai", genai_module)

        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        first = client.embed(["một câu"])
        second = client.embed(["một câu"])
        assert first == second
        assert calls["n"] == 1  # lần hai lấy từ cache đĩa

    def test_embed_cache_is_keyed_by_model(self, monkeypatch, tmp_path: Path):
        """Vector của model khác phải nằm ở khoá cache khác."""
        client = GeminiClient(
            model="gemini-test", api_key="k", cache_dir=tmp_path / "cache"
        )
        key_a = client._embed_cache_key("một câu")
        client.embed_model = "model-khác"
        key_b = client._embed_cache_key("một câu")
        assert key_a != key_b

    def test_embed_rotates_to_fallback_model_on_daily_quota(self, monkeypatch, tmp_path: Path):
        """Khi model chính hết quota, client chuyển sang model dự phòng."""
        used: list[str] = []

        class _Emb:
            def __init__(self, values):
                self.values = values

        class _EmbedModels:
            def embed_content(self, **kwargs: Any):
                model = kwargs.get("model")
                used.append(model)
                if model == "model-chính":
                    err = RuntimeError("quota hết")
                    err.code = 429
                    err.response = SimpleNamespace(
                        json=lambda: {
                            "details": [
                                {"retryInfo": [{"retryDelay": "72000s"}]}
                            ]
                        }
                    )
                    raise err
                return SimpleNamespace(embeddings=[_Emb([3.0, 4.0])])

        genai_module = SimpleNamespace(
            Client=lambda **kwargs: SimpleNamespace(models=_EmbedModels())
        )
        google_module = SimpleNamespace(genai=genai_module)
        import sys

        monkeypatch.setitem(sys.modules, "google", google_module)
        monkeypatch.setitem(sys.modules, "google.genai", genai_module)

        client = GeminiClient(
            model="gemini-test",
            api_key="k",
            cache_dir=tmp_path / "cache",
            embed_model="model-chính",
            fallback_embed_models=["model-dự-phòng"],
        )
        vec = client.embed(["một câu"])
        assert vec == [[0.6, 0.8]]
        assert "model-dự-phòng" in used
        assert client.embed_model == "model-dự-phòng"

    def test_embed_splits_batch_when_model_collapses_it(self, monkeypatch, tmp_path: Path):
        """Một số model trả về 1 vector cho cả batch -> client phải tách."""

        class _Emb:
            def __init__(self, values):
                self.values = values

        class _EmbedModels:
            def embed_content(self, **kwargs: Any):
                contents = kwargs.get("contents") or []
                if len(contents) > 1:
                    return SimpleNamespace(embeddings=[_Emb([1.0, 0.0])])
                return SimpleNamespace(embeddings=[_Emb([1.0, 0.0])])

        genai_module = SimpleNamespace(
            Client=lambda **kwargs: SimpleNamespace(models=_EmbedModels())
        )
        google_module = SimpleNamespace(genai=genai_module)
        import sys

        monkeypatch.setitem(sys.modules, "google", google_module)
        monkeypatch.setitem(sys.modules, "google.genai", genai_module)

        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        vec = client.embed(["a", "b"])
        assert len(vec) == 2
        assert all(len(v) == 2 for v in vec)

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

    def test_count_tokens_used_when_available(self, monkeypatch, tmp_path: Path):
        """Số token được đếm chính xác bằng ``count_tokens``."""
        calls = {"n": 0}

        class _Models(_FakeModels):
            def count_tokens(self, **kwargs: Any):
                calls["n"] += 1
                return SimpleNamespace(total_tokens=4321)

        models = _Models(lambda kwargs: _ok_response("ok"))
        genai_module = SimpleNamespace(Client=lambda **kwargs: _FakeClient(models))
        google_module = SimpleNamespace(genai=genai_module)
        import sys

        monkeypatch.setitem(sys.modules, "google", google_module)
        monkeypatch.setitem(sys.modules, "google.genai", genai_module)

        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        tokens = client._count_prompt_tokens("gemini-test", "", "câu hỏi")
        assert tokens == 4321
        assert calls["n"] == 1

    def test_count_tokens_falls_back_to_estimate(self, monkeypatch, tmp_path: Path):
        """Khi ``count_tokens`` lỗi, client ước lượng theo độ dài prompt."""
        models = _install_fake_genai(monkeypatch, lambda kwargs: _ok_response("ok"))

        def _boom(**kwargs: Any):
            raise RuntimeError("không hỗ trợ")

        models.count_tokens = _boom  # type: ignore[attr-defined]
        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        tokens = client._count_prompt_tokens("gemini-test", "", "x" * 400)
        assert tokens == 200

    def test_token_limiter_books_the_prompt_size(self, monkeypatch, tmp_path: Path):
        """Token của prompt được đặt trước trong hạn mức dùng chung."""
        models = _install_fake_genai(monkeypatch, lambda kwargs: _ok_response("ok"))
        models.count_tokens = lambda **kw: SimpleNamespace(total_tokens=1000)  # type: ignore[attr-defined]
        client = GeminiClient(
            model="gemini-test",
            api_key="k",
            cache_dir=tmp_path / "cache",
            model_tpm={"gemini-test": 5000},
        )
        client.generate(_req("judge"))
        limiter_file = tmp_path / "rate_limits" / "gemini-test.tokens.json"
        assert limiter_file.exists()
        import json as _json

        pairs = _json.loads(limiter_file.read_text(encoding="utf-8"))
        assert sum(p[1] for p in pairs) == 1000


class TestGeminiClientRateLimitHandling:
    """Xử lý 429: quota theo *phút* phải chờ rồi thử lại cùng model."""

    def _install_throttled(self, monkeypatch, fail_times: int, models_used: list[str]):
        state = {"n": 0}

        def _behaviour(kwargs: Any):
            state["n"] += 1
            models_used.append(kwargs["model"])
            if state["n"] <= fail_times:
                err = RuntimeError("429 RESOURCE_EXHAUSTED")
                err.code = 429
                err.response = SimpleNamespace(
                    json=lambda: {"details": [{"retryInfo": [{"retryDelay": "3s"}]}]}
                )
                raise err
            return _ok_response("ok")

        _install_fake_genai(monkeypatch, _behaviour)
        monkeypatch.setattr("hgrag.llm.client.time.sleep", lambda s: None)

    def test_per_minute_quota_retries_same_model(self, monkeypatch, tmp_path: Path):
        """429 kèm RetryInfo ngắn -> chờ rồi thử lại CHÍNH model đó.

        Xoay sang model dự phòng sẽ đốt quota *ngày* của Gemini, không thể
        hoàn lại trong một lượt chạy (xem docs/10_quota_gemini_free_tier.md).
        """
        used: list[str] = []
        self._install_throttled(monkeypatch, fail_times=1, models_used=used)
        client = GeminiClient(
            model="model-chính",
            api_key="k",
            cache_dir=tmp_path / "cache",
            fallback_models=["model-dự-phòng"],
            max_retries=3,
        )
        resp = client.generate(_req("judge"))
        assert resp.text == "ok"
        assert used == ["model-chính", "model-chính"]

    def test_daily_quota_rotates_to_fallback_model(self, monkeypatch, tmp_path: Path):
        """429 không kèm RetryInfo ngắn -> quota ngày cạn -> xoay model."""
        used: list[str] = []

        def _behaviour(kwargs: Any):
            used.append(kwargs["model"])
            if kwargs["model"] == "model-chính":
                err = RuntimeError("429 RESOURCE_EXHAUSTED")
                err.code = 429
                raise err
            return _ok_response("ok")

        _install_fake_genai(monkeypatch, _behaviour)
        monkeypatch.setattr("hgrag.llm.client.time.sleep", lambda s: None)
        client = GeminiClient(
            model="model-chính",
            api_key="k",
            cache_dir=tmp_path / "cache",
            fallback_models=["model-dự-phòng"],
        )
        resp = client.generate(_req("judge"))
        assert resp.text == "ok"
        assert used == ["model-chính", "model-dự-phòng"]

    def test_cache_key_includes_the_model(self, tmp_path: Path):
        """Cache phải phân vùng theo model, nếu không kết quả của hai model
        khác nhau sẽ bị trộn lẫn trong cùng một bảng so sánh."""
        a = GeminiClient(model="model-a", api_key="k", cache_dir=tmp_path / "cache")
        b = GeminiClient(model="model-b", api_key="k", cache_dir=tmp_path / "cache")
        assert a._cache_key(_req("judge")) != b._cache_key(_req("judge"))


# --------------------------------------------------------------------------
# Gemma: suy luận ăn hết ngân sách output
# --------------------------------------------------------------------------


def _thinking_overrun_response() -> SimpleNamespace:
    """Mô phỏng câu trả lời bị cắt vì token suy luận dùng hết ngân sách."""
    return SimpleNamespace(
        text="",
        candidates=[SimpleNamespace(finish_reason="FinishReason.MAX_TOKENS")],
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=None),
    )


class TestGeminiThinkingOverrun:
    """Gemma 4 tính token suy luận vào `max_output_tokens`.

    Khi phần suy luận không kết thúc, API trả ``text`` rỗng và
    ``finish_reason=MAX_TOKENS``: câu trả lời bị mất hoàn toàn.  Client phải
    phát hiện tình huống này và thử lại với ngân sách lớn hơn.
    """

    def test_detects_max_tokens_finish_reason(self):
        assert GeminiClient._is_thinking_overrun(_thinking_overrun_response()) is True

    def test_normal_response_is_not_an_overrun(self):
        resp = SimpleNamespace(
            text="ok",
            candidates=[SimpleNamespace(finish_reason="FinishReason.STOP")],
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
        )
        assert GeminiClient._is_thinking_overrun(resp) is False

    def test_expanded_budget_doubles_and_caps(self):
        assert GeminiClient._expanded_output_budget(256) == 512
        assert GeminiClient._expanded_output_budget(2048) == 4096
        assert GeminiClient._expanded_output_budget(4096) == 8192
        assert GeminiClient._expanded_output_budget(8192) == 8192

    def test_retries_with_a_larger_budget(self, monkeypatch, tmp_path: Path):
        models = _install_fake_genai(
            monkeypatch,
            lambda kwargs: (
                _thinking_overrun_response()
                if len(models.calls) == 1
                else _ok_response("câu trả lời")
            ),
        )
        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        resp = client.generate(_req("judge"))
        assert resp.text == "câu trả lời"
        assert len(models.calls) == 2
        first_budget = models.calls[0]["config"]["max_output_tokens"]
        second_budget = models.calls[1]["config"]["max_output_tokens"]
        assert second_budget == first_budget * 2

    def test_does_not_retry_when_the_answer_is_present(self, monkeypatch, tmp_path: Path):
        models = _install_fake_genai(monkeypatch, lambda kwargs: _ok_response("ok"))
        client = GeminiClient(model="gemini-test", api_key="k", cache_dir=tmp_path / "cache")
        resp = client.generate(_req("judge"))
        assert resp.text == "ok"
        assert len(models.calls) == 1
