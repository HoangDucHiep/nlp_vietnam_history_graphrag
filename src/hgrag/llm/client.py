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
import re
import time
from pathlib import Path
from typing import Any

from ..data.records import stable_hash
from .schemas import LLMMessage, LLMRequest, LLMResponse

_YEAR_RE = re.compile(r"\b(9\d{2}|1[0-9]{3})\b")

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _retry_after_seconds(exc: Exception) -> float | None:
    """The wait time suggested by the API in ``RetryInfo`` (if any)."""
    payload = getattr(exc, "response", None)
    text = str(exc)
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    details = getattr(payload, "json", None)
    if callable(details):
        try:
            data = details()
        except Exception:  # noqa: BLE001
            return None
        for violation in (data.get("details") or []):
            for info in violation.get("retryInfo") or []:
                delay = info.get("retryDelay")
                if isinstance(delay, str) and delay.endswith("s"):
                    try:
                        return float(delay[:-1])
                    except ValueError:
                        continue
    return None


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
        fallback_models: list[str] | None = None,
        embed_batch_size: int = 16,
        embed_max_retries: int = 8,
        max_rate_limit_wait: float = 90.0,
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
        # Free-tier API keys have a per-model daily quota (20 req/day for the
        # Flash models).  When a model is exhausted we transparently rotate to
        # the next one instead of failing the whole pipeline.
        self.fallback_models = [m for m in (fallback_models or []) if m and m != model]
        self.embed_batch_size = max(1, int(embed_batch_size))
        self.embed_max_retries = max(1, int(embed_max_retries))
        self.max_rate_limit_wait = float(max_rate_limit_wait)
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
    @staticmethod
    def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Make a JSON schema acceptable by the Gemini structured-output API.

        Gemini rejects union types (``"type": ["integer", "null"]``) and a few
        other JSON-Schema-only constructs.  We keep the first concrete type and
        let the caller handle missing/``null`` values itself.
        """
        if not isinstance(schema, dict):
            return schema
        out = {k: v for k, v in schema.items() if k != "additionalProperties"}
        if "type" in out and isinstance(out["type"], list):
            concrete = [x for x in out["type"] if x != "null"]
            out["type"] = concrete[0] if concrete else "string"
        if isinstance(out.get("properties"), dict):
            out["properties"] = {
                k: GeminiClient._sanitize_schema(v) if isinstance(v, dict) else v
                for k, v in out["properties"].items()
            }
        if isinstance(out.get("items"), dict):
            out["items"] = GeminiClient._sanitize_schema(out["items"])
        return out

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
            config["response_schema"] = self._sanitize_schema(request.response_schema)

        last_error: Exception | None = None
        models_to_try = [request.model or self.model, *self.fallback_models]
        for model_name in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    kwargs: dict[str, Any] = {"model": model_name, "contents": prompt}
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
                        "model": model_name,
                        "usage": {
                            "prompt_tokens": getattr(usage, "prompt_token_count", None),
                            "output_tokens": getattr(usage, "candidates_token_count", None),
                        }
                        if usage
                        else {},
                    }
                    if self.cache is not None:
                        self.cache.put(key, payload)
                    return LLMResponse(text=text, model=model_name, cached=False, raw=payload)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    status = getattr(getattr(exc, "code", None), "value", None) or getattr(
                        exc, "code", None
                    )
                    code = int(status) if isinstance(status, int) else None
                    if code == 429:
                        # Free tier: quota theo *phút* (hoặc theo ngày).  Nếu API
                        # gợi ý thời gian chờ thì tôn trọng nó; nếu hết hẳn quota
                        # ngày thì chuyển sang model dự phòng.
                        wait = _retry_after_seconds(exc)
                        if wait is not None and wait <= self.max_rate_limit_wait:
                            logger.warning(
                                "Gemini bị giới hạn tốc độ (model=%s) -- chờ %.1fs rồi thử lại",
                                model_name,
                                wait,
                            )
                            time.sleep(wait)
                            continue
                        logger.warning(
                            "Gemini quota hết cho model %s -> chuyển model khác", model_name
                        )
                        break
                    if code is not None and code not in RETRYABLE_STATUS:
                        raise LLMError(
                            f"Gemini lỗi không retryable ({code}) với model {model_name}: {exc}"
                        ) from exc
                    delay = self.retry_base_delay * (2**attempt)
                    logger.warning(
                        "Gemini call failed (model=%s, attempt %d/%d): %s -- retry sau %.1fs",
                        model_name,
                        attempt + 1,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        raise LLMError(f"Gemini call thất bại sau khi thử {models_to_try}: {last_error}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` (batched), retrying on rate-limit errors.

        The free tier of ``gemini-embedding-001`` allows a limited number of
        embed requests *per minute*; the method therefore backs off and
        retries instead of failing the whole run.
        """
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            vectors = self._embed_batch(batch)
            out.extend(vectors)
        if out:
            self._embed_dim = len(out[0])
        return out

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed one batch, retrying with the delay suggested by the API.

        The free tier of ``gemini-embedding-001`` limits the number of embed
        *tokens* per minute (30 000).  When the limit is hit the API returns a
        ``RetryInfo`` telling how long to wait -- we honour it instead of
        failing the whole run.
        """
        last_error: Exception | None = None
        for attempt in range(self.embed_max_retries):
            try:
                kwargs: dict[str, Any] = {"model": self.embed_model, "contents": batch}
                if self.output_dim:
                    kwargs["config"] = {"output_dimensionality": self.output_dim}
                resp = self._client.models.embed_content(**kwargs)
                vectors: list[list[float]] = []
                for item in resp.embeddings:
                    vec = list(item.values)
                    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                    vectors.append([x / norm for x in vec])
                return vectors
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                delay = _retry_after_seconds(exc) or (self.retry_base_delay * (2**attempt))
                logger.warning(
                    "Embed call failed (attempt %d/%d): %s -- retry sau %.1fs",
                    attempt + 1,
                    self.embed_max_retries,
                    str(exc)[:160],
                    delay,
                )
                time.sleep(delay)
        raise LLMError(
            f"Embed call thất bại sau {self.embed_max_retries} lần thử: {last_error}"
        )

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
        question = ""
        for line in prompt.splitlines():
            if line.strip().upper().startswith("CÂU HỎI:"):
                question = line.split(":", 1)[1].strip()
        entities = [
            w
            for w in question.replace("?", " ")
            .replace(",", " ")
            .split()
            if len(w) > 2 and w[:1].isupper()
        ]
        years = [int(y) for y in _YEAR_RE.findall(question)]
        return {
            "intent": "factual",
            "entities": entities[:4] or ["Trần Hưng Đạo"],
            "time_refs": years,
            "relation_plan": [],
            "expected_hops": 1 if len(entities) <= 1 else 2,
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
                fallback_models=llm_cfg.get("fallback_models", []),
            )
        except LLMError as exc:
            logger.warning("Không khởi tạo được GeminiClient (%s) -> dùng mock", exc)
    return MockLLMClient(embed_dim=settings.embed_dim, seed=settings.seed)
