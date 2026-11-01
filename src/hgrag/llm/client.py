"""LLM clients: Gemini (real) and a deterministic offline mock.

Design rules (see docs/06_quyet_dinh_thiet_ke.md, ADR-05):

* One protocol (:class:`LLMClient`) for every module -- no module may import a
  vendor SDK directly.
* Every call is cached on disk by a hash of the request payload, so re-running
  the pipeline is cheap and reproducible.
* When ``GEMINI_API_KEY`` is missing the factory returns :class:`MockLLMClient`
  which produces *deterministic* output.  This keeps the whole pipeline
  runnable offline (used by tests, CI and the smoke test).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from ..data.records import stable_hash
from .schemas import LLMMessage, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when a request fails in a non-retryable way."""


class LLMClient:
    """Protocol implemented by every backend."""

    name: str = "base"

    def generate(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def generate_json(self, request: LLMRequest) -> Any:
        """Generate and parse a JSON payload (retries once on parse failure)."""
        last_exc: Exception | None = None
        for _ in range(2):
            resp = self.generate(request)
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                last_exc = exc
                logger.warning("LLM returned invalid JSON (%s), retrying once", exc)
        raise LLMError(f"LLM did not return valid JSON: {last_exc}")

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class DiskCache:
    """Content-addressed JSON cache stored under ``artifacts/llm_cache``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiClient(LLMClient):
    """Google Gemini backend via the ``google-genai`` SDK."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        api_key: str | None = None,
        embed_model: str = "gemini-embedding-001",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        request_timeout: float = 120.0,
        max_retries: int = 4,
        retry_base_delay: float = 2.0,
        cache_dir: Path | str | None = None,
        cache_enabled: bool = True,
        output_dim: int | None = None,
    ):
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "google-genai chưa được cài. Chạy: uv pip install 'google-genai>=1.0.0'"
            ) from exc

        self._genai = genai
        self.model = model
        self.embed_model = embed_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.cache = DiskCache(cache_dir) if cache_enabled and cache_dir else None
        self.output_dim = output_dim
        self._client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._embed_dim = output_dim or 768

    # -- helpers ---------------------------------------------------------
    def _cache_key(self, request: LLMRequest) -> str:
        return stable_hash(request.to_dict())

    def _split_system(self, messages: list[LLMMessage]) -> tuple[str, str]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        rest = "\n\n".join(m.content for m in messages if m.role != "system")
        return system, rest

    # -- API -------------------------------------------------------------
    def generate(self, request: LLMRequest) -> LLMResponse:
        key = self._cache_key(request)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return LLMResponse(
                    text=hit["text"], model=hit.get("model", self.model), cached=True, raw=hit
                )

        system, prompt = self._split_system(request.messages)
        config: dict[str, Any] = {
            "temperature": request.temperature or self.temperature,
            "max_output_tokens": request.max_output_tokens or self.max_output_tokens,
        }
        if request.response_schema:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = request.response_schema

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {"model": request.model or self.model, "contents": prompt}
                if system:
                    kwargs["config"] = {
                        **config,
                        "system_instruction": system,
                    }
                else:
                    kwargs["config"] = config
                resp = self._client.models.generate_content(**kwargs)
                text = getattr(resp, "text", "") or ""
                usage = getattr(resp, "usage_metadata", None)
                payload = {
                    "text": text,
                    "model": request.model or self.model,
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_token_count", None),
                        "output_tokens": getattr(usage, "candidates_token_count", None),
                    }
                    if usage
                    else {},
                }
                if self.cache is not None:
                    self.cache.put(key, payload)
                return LLMResponse(text=text, model=payload["model"], cached=False, raw=payload)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status = getattr(getattr(exc, "code", None), "value", None) or getattr(
                    exc, "code", None
                )
                code = int(status) if isinstance(status, int) else None
                if code is not None and code not in RETRYABLE_STATUS:
                    raise LLMError(f"Gemini lỗi không retryable ({code}): {exc}") from exc
                delay = self.retry_base_delay * (2**attempt)
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s -- retry sau %.1fs",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise LLMError(f"Gemini call thất bại sau {self.max_retries} lần: {last_error}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            kwargs: dict[str, Any] = {"model": self.embed_model, "contents": batch}
            if self.output_dim:
                kwargs["config"] = {"output_dimensionality": self.output_dim}
            resp = self._client.models.embed_content(**kwargs)
            for item in resp.embeddings:
                vec = list(item.values)
                norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                out.append([x / norm for x in vec])
        if out:
            self._embed_dim = len(out[0])
        return out

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


# ---------------------------------------------------------------------------
# Mock (offline, deterministic)
# ---------------------------------------------------------------------------


class MockLLMClient(LLMClient):
    """Deterministic offline stub.

    It inspects ``request.task`` and returns a *schema-valid* payload so that
    the whole downstream pipeline can be exercised without network access.
    Output is deterministic (seeded by the request hash), never random.
    """

    name = "mock"

    def __init__(self, embed_dim: int = 768, seed: int = 42):
        self._embed_dim = embed_dim
        self.seed = seed
        self.calls: list[LLMRequest] = []

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _prompt_text(request: LLMRequest) -> str:
        return "\n".join(m.content for m in request.messages if m.role != "system")

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        rng = random.Random(
            int(hashlib.sha256(request.to_dict().__repr__().encode()).hexdigest()[:8], 16)
            ^ self.seed
        )
        task = request.task.lower()
        text = self._dispatch(task, request, rng)
        return LLMResponse(text=text, model="mock", cached=False, raw={"task": task})

    def _dispatch(self, task: str, request: LLMRequest, rng: random.Random) -> str:
        prompt = self._prompt_text(request)
        if "extract" in task or "event" in task:
            return json.dumps(self._mock_extraction(prompt), ensure_ascii=False)
        if "understand" in task or "query" in task:
            return json.dumps(self._mock_query_understanding(prompt), ensure_ascii=False)
        if "judge" in task:
            return json.dumps(self._mock_judge(prompt), ensure_ascii=False)
        if "verify" in task or "alias" in task or "disambig" in task:
            return json.dumps(self._mock_verify(prompt), ensure_ascii=False)
        if "entity" in task and "link" in task:
            return json.dumps(self._mock_entity_link(prompt), ensure_ascii=False)
        return self._mock_answer(prompt, rng)

    # -- task-specific mocks --------------------------------------------
    @staticmethod
    def _mock_extraction(prompt: str) -> dict[str, Any]:
        title = prompt.strip().splitlines()[0][:120] if prompt.strip() else "Sự kiện"
        return {
            "events": [
                {
                    "event_type": "Conflict",
                    "trigger": title[:60],
                    "description": f"Diễn biến được trích xuất (mock) từ: {title[:120]}",
                    "year": 0,
                    "arguments": [
                        {"role": "Agent", "name": "Nhân vật lịch sử", "type": "Person"},
                        {"role": "Location", "name": "Địa điểm", "type": "Place"},
                    ],
                }
            ]
        }

    @staticmethod
    def _mock_query_understanding(prompt: str) -> dict[str, Any]:
        q = prompt.strip().splitlines()[-1][:200]
        return {
            "intent": "factual",
            "entities": [w for w in q.split() if len(w) > 3][:3],
            "time_refs": [],
            "relation_plan": [],
            "expected_hops": 1,
            "needs_graph": True,
        }

    @staticmethod
    def _mock_judge(prompt: str) -> dict[str, Any]:
        return {"score": 1, "correct": True, "grounded": True, "reason": "mock judge"}

    @staticmethod
    def _mock_verify(prompt: str) -> dict[str, Any]:
        return {"is_same_entity": True, "confidence": 0.9, "reason": "mock verification"}

    @staticmethod
    def _mock_entity_link(prompt: str) -> dict[str, Any]:
        return {"wikidata_id": "", "canonical_name": "", "confidence": 0.5, "reason": "mock link"}

    @staticmethod
    def _mock_answer(prompt: str, rng: random.Random) -> str:
        cites = [f"[E{i}]" for i in range(1, min(3, max(1, prompt.count("[E"))) + 1)]
        cites = cites or ["[E1]"]
        body = (
            "Dựa trên ngữ cảnh lịch sử được cung cấp, sự kiện trong câu hỏi diễn ra theo "
            "trình tự thời gian sau (chế độ mock, không có API key):\n"
            "1. Bước khởi đầu: " + cites[0] + " nêu bối cảnh.\n"
            "2. Diễn biến: " + (cites[1] if len(cites) > 1 else cites[0]) + " mô tả hành động chính.\n"
            f"Kết luận: {', '.join(cites)}."
        )
        return body

    # -- embeddings ------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self._embed_dim
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            for i in range(self._embed_dim):
                vec[i] = ((digest[i % len(digest)] / 255.0) - 0.5) * 2.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_llm_client(settings: Any | None = None) -> LLMClient:
    """Return the best available LLM client for the current environment."""
    if settings is None:
        from ..config import get_settings

        settings = get_settings()

    provider = settings.llm_provider
    llm_cfg = settings.llm

    if provider == "mock":
        return MockLLMClient(embed_dim=settings.embed_dim, seed=settings.seed)
    if provider in ("gemini", "auto") and settings.has_gemini_key:
        try:
            return GeminiClient(
                model=settings.llm_model,
                api_key=settings.gemini_api_key,
                embed_model=settings.embed_model,
                temperature=llm_cfg.get("temperature", 0.0),
                max_output_tokens=llm_cfg.get("max_output_tokens", 2048),
                request_timeout=llm_cfg.get("request_timeout", 120),
                max_retries=llm_cfg.get("max_retries", 4),
                retry_base_delay=llm_cfg.get("retry_base_delay", 2.0),
                cache_dir=settings.cache_dir,
                cache_enabled=llm_cfg.get("cache_enabled", True),
                output_dim=settings.embed_dim,
            )
        except LLMError as exc:
            logger.warning("Không khởi tạo được GeminiClient (%s) -> dùng mock", exc)
    return MockLLMClient(embed_dim=settings.embed_dim, seed=settings.seed)
