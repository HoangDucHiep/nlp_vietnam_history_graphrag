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
import threading
import time
from pathlib import Path
from typing import Any

from ..data.records import stable_hash
from .rate_limit import SharedRateLimiter, SharedTokenLimiter
from .schemas import LLMMessage, LLMRequest, LLMResponse

_YEAR_RE = re.compile(r"\b(9\d{2}|1[0-9]{3})\b")


def _slug(text: str) -> str:
    """A file-system safe name for a model id."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text) or "model"

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

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Recover a JSON object from a response that is not pure JSON.

        Small models do not always honour the structured-output contract:
        they may wrap the JSON in prose or in a markdown fence, and a
        truncated response may cut the object in the middle of a string.
        Three repairs are applied, in this order:

        1. strip a markdown fence / the surrounding prose;
        2. cut the object at its **last complete top-level pair** and close
           the braces -- this keeps the fields that were written entirely
           (the rubric fields come first, so they survive);
        3. if the object was cut in the middle of a string, close the string
           first.

        The result is returned as ``None`` when nothing can be recovered.
        """
        if not text:
            return None
        candidate = text.strip()
        if candidate.startswith("```"):
            parts = candidate.split("```")
            candidate = parts[1] if len(parts) > 1 else candidate
            if candidate.lower().startswith("json"):
                candidate = candidate[4:]
            candidate = candidate.strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
        # Find the outermost JSON object and repair a truncated tail.
        start = candidate.find("{")
        if start == -1:
            return None
        body = candidate[start:]
        end = LLMClient._scan_object_end(body)
        if end is not None:
            snippet = body[: end + 1]
            try:
                json.loads(snippet)
                return snippet
            except json.JSONDecodeError:
                pass
        return LLMClient._repair_truncated_object(body)

    @staticmethod
    def _repair_truncated_object(body: str) -> str | None:
        """Repair an object that was cut short.

        The object is cut at the end of its last **complete** top-level
        key/value pair; the open braces are then closed.  A field that was
        only partially written (``"reason": "cắt ngang...``) is dropped
        instead of being guessed.
        """
        depth = 0
        in_string = False
        escaped = False
        # The index just after the last complete top-level pair.
        last_complete = -1
        for index, char in enumerate(body):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in "{[":
                depth += 1
                continue
            if char in "}]":
                depth -= 1
                continue
            if char == "," and depth == 1:
                # The pair before the comma is complete.
                last_complete = index
        if last_complete > 0:
            repaired = body[:last_complete]
            open_braces = repaired.count("{") - repaired.count("}")
            if open_braces > 0:
                repaired += "}" * open_braces
            try:
                json.loads(repaired)
                return repaired
            except json.JSONDecodeError:
                pass
        # Nothing complete: close the open string and the open braces.
        in_string = False
        escaped = False
        for char in body:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
        repaired = body
        if in_string:
            repaired += '"'
        repaired = repaired.rstrip().rstrip(",")
        repaired = re.sub(r",\s*$", "", repaired)
        open_braces = repaired.count("{") - repaired.count("}")
        if open_braces > 0:
            repaired += "}" * open_braces
        try:
            json.loads(repaired)
        except json.JSONDecodeError:
            return None
        return repaired

    @staticmethod
    def _scan_object_end(body: str) -> int | None:
        """The index of the ``}`` that closes the object starting at ``body[0]``.

        ``None`` is returned when the object is never closed (a truncated
        response).
        """
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(body):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def generate_json(self, request: LLMRequest) -> Any:
        """Generate and parse a JSON payload (retries once on parse failure)."""
        last_exc: Exception | None = None
        for attempt in range(2):
            resp = self.generate(request)
            text = resp.text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                last_exc = exc
                repaired = self._extract_json(text)
                if repaired is not None:
                    logger.info(
                        "LLM trả về JSON không chuẩn nhưng đã sửa được (%s)", exc
                    )
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        pass
                if attempt == 0:
                    logger.warning(
                        "LLM returned invalid JSON (%s), retrying once", exc
                    )
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
        # The temporary name carries the PID so that several evaluation
        # workers can write the same cache directory concurrently without
        # clobbering each other's partial file.
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
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
        fallback_embed_models: list[str] | None = None,
        model_rpm: dict[str, int] | None = None,
        model_tpm: dict[str, int] | None = None,
        count_tokens: bool = True,
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
        # The embedding model has its *own* quota, separate from the chat
        # models.  When it is exhausted we rotate to a sibling embedding
        # model -- but the corpus must then be re-embedded with the same
        # model, otherwise the vector space no longer matches.
        self.fallback_embed_models = [
            m for m in (fallback_embed_models or []) if m and m != embed_model
        ]
        self._client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self._embed_dim = output_dim or 768
        # Circuit breaker: set once every embedding model refused with a
        # "quota exhausted" error (see :meth:`_embed_batch`).
        self._embed_quota_exhausted = False
        # Chat models whose daily quota is exhausted.  They are skipped
        # until every model is exhausted (then the list is reset, so a
        # quota reset is picked up automatically).
        self._exhausted_models: set[str] = set()
        # Models that answered "429, retry after N seconds": they are only
        # rate limited for a few seconds, so they get a short cooldown
        # instead of being dropped like an exhausted model.
        self._throttled: dict[str, float] = {}
        # Per-model requests-per-minute limiter.  The free tier applies an
        # RPM quota *per model* (30/min for the Gemma models, 15/min for the
        # Flash-Lite ones).  Proactively spacing the calls avoids the 429
        # round-trip, which costs far more than the wait itself.
        self._model_rpm = dict(model_rpm or {})
        self._call_times: dict[str, list[float]] = {}
        self._rpm_lock = threading.Lock()
        # The *token* quota is the real limit of the Gemma family: the free
        # tier allows 16 000 input tokens per minute, which a request-count
        # limiter cannot express (one long prompt may use a large part of
        # the budget).  See docs/10_quota_gemini_free_tier.md.
        self._model_tpm = dict(model_tpm or {})
        self._count_tokens_enabled = bool(count_tokens)
        self._token_cache: dict[tuple[str, str], int] = {}
        self._token_cache_lock = threading.Lock()
        # The quota is global per project+model, so the limiter is shared
        # between the worker processes through a lock file.
        self._rate_limit_dir = (
            Path(cache_dir).parent / "rate_limits"
            if cache_dir
            else Path(os.getenv("HGRAG_RATE_LIMIT_DIR", "artifacts/rate_limits"))
        )
        self._shared_limiters: dict[str, SharedRateLimiter] = {}
        self._shared_token_limiters: dict[str, SharedTokenLimiter] = {}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _is_thinking_overrun(resp: Any) -> bool:
        """Whether the response was cut because thinking ate the budget.

        Gemma 4 counts the *reasoning* tokens against ``max_output_tokens``.
        When the reasoning does not terminate, the API answers with
        ``finish_reason=MAX_TOKENS`` and an empty ``text``: the answer is lost
        even though the call succeeded.  Measured in docs/10.
        """
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            reason = str(getattr(cand, "finish_reason", "") or "")
            if "MAX_TOKENS" in reason:
                return True
        return False

    @staticmethod
    def _expanded_output_budget(current: int) -> int:
        """The budget to retry with when thinking overran ``current``.

        The measured thinking length of the Gemma family is 250–1 100 tokens
        for a question it can answer and unbounded for one it cannot, so the
        retry doubles the budget up to 8 192 tokens.
        """
        return min(8192, max(1, int(current)) * 2)

    def _cache_key(self, request: LLMRequest) -> str:
        # The *primary* model is part of the key: when the model of the run
        # changes (e.g. from Gemini to Gemma because the Gemini free-tier
        # quota is exhausted) the answers of the previous model must not be
        # reused, otherwise the comparison between the systems would mix two
        # different generators.
        return stable_hash({**request.to_dict(), "model": self.model})

    def _split_system(self, messages: list[LLMMessage]) -> tuple[str, str]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        rest = "\n\n".join(m.content for m in messages if m.role != "system")
        return system, rest

    def _wait_for_rpm(self, model_name: str) -> None:
        """Block until a slot is free under the RPM quota of ``model_name``.

        The free tier counts the requests of the *last minute*; spacing the
        calls out is much cheaper than eating a 429 and rotating away (the
        rotation burns the quota of the sibling models for nothing).

        The quota is *global* (per project and per model), so the limiter is
        shared between the worker processes through a small lock file --
        otherwise every worker would apply the limit on its own and the sum
        would exceed the quota.
        """
        rpm = self._model_rpm.get(model_name)
        if not rpm or rpm <= 0:
            return
        shared = self._shared_limiters.get(model_name)
        if shared is None:
            shared = SharedRateLimiter(
                self._rate_limit_dir / f"{_slug(model_name)}.json", rpm
            )
            self._shared_limiters[model_name] = shared
        shared.acquire()
        with self._rpm_lock:
            now = time.monotonic()
            times = [t for t in self._call_times.get(model_name, []) if now - t < 60.0]
            times.append(now)
            self._call_times[model_name] = times

    def _count_prompt_tokens(self, model_name: str, system: str, prompt: str) -> int:
        """The number of input tokens of a request (exact, via ``count_tokens``).

        ``count_tokens`` does *not* consume the generation quota (verified in
        docs/10_quota_gemini_free_tier.md), so it can be used to book the
        right amount of the token budget before the real call.

        The Developer API does not accept a ``system_instruction`` in
        ``count_tokens``, so the system prompt is counted **separately** and
        added.  Ignoring it would under-count the real request by ~30 % and
        the workers would then exceed the per-minute quota.

        The results are cached per ``(model, text)``: the same context is sent
        over and over during an evaluation run.
        """
        if not self._count_tokens_enabled:
            # Rough fallback: Vietnamese text is roughly 2.5 characters per
            # token for the Gemma tokenizer.
            return max(1, (len(system) + len(prompt)) // 2)
        prompt_tokens = self._count_text_tokens(model_name, prompt)
        system_tokens = self._count_text_tokens(model_name, system) if system else 0
        return max(1, prompt_tokens + system_tokens)

    def _count_text_tokens(self, model_name: str, text: str) -> int:
        """The token count of a piece of text (cached)."""
        if not text.strip():
            return 0
        key = (model_name, text)
        with self._token_cache_lock:
            hit = self._token_cache.get(key)
        if hit is not None:
            return hit
        tokens = 0
        try:
            resp = self._client.models.count_tokens(
                model=model_name, contents=text
            )
            tokens = int(getattr(resp, "total_tokens", 0) or 0)
        except Exception as exc:  # noqa: BLE001 - counting is best effort
            logger.debug("count_tokens thất bại (%s) -- dùng ước lượng", exc)
            tokens = 0
        if tokens <= 0:
            tokens = max(1, len(text) // 2)
        with self._token_cache_lock:
            self._token_cache[key] = tokens
        return tokens

    def _wait_for_tokens(self, model_name: str, tokens: int) -> None:
        """Book ``tokens`` input tokens in the shared per-minute budget."""
        tpm = self._model_tpm.get(model_name)
        if not tpm or tpm <= 0:
            return
        shared = self._shared_token_limiters.get(model_name)
        if shared is None:
            shared = SharedTokenLimiter(
                self._rate_limit_dir / f"{_slug(model_name)}.tokens.json", tpm
            )
            self._shared_token_limiters[model_name] = shared
        shared.acquire(tokens)

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
        # The models whose quota is exhausted are remembered: a full
        # evaluation makes ~3 000 calls and each of them would otherwise
        # pay the ~5 s of the failed request before rotating to the next
        # model.
        models_to_try = [m for m in models_to_try if m not in self._exhausted_models]
        if not models_to_try:
            self._exhausted_models.clear()
            models_to_try = [request.model or self.model, *self.fallback_models]
        # A model that answered "429, retry after N s" is only *rate limited*
        # for a few seconds -- the per-minute budget.  It must NOT be rotated
        # away: the fallback models of the free tier are capped by requests
        # **per day**, so rotating would burn their daily budget for nothing.
        # Instead the call waits and is retried on the *same* model.
        now = time.monotonic()
        cooled = [m for m in models_to_try if self._throttled.get(m, 0.0) <= now]
        if cooled:
            models_to_try = cooled
        else:
            # Every model is cooling down: wait for the first one to free up.
            wait_for = min(self._throttled.get(m, 0.0) for m in models_to_try) - now
            if wait_for > 0:
                logger.info("Mọi model đều bị giới hạn tốc độ -- chờ %.1fs", wait_for)
                time.sleep(min(wait_for + 0.5, self.max_rate_limit_wait))
                models_to_try = [
                    m
                    for m in models_to_try
                    if self._throttled.get(m, 0.0) <= time.monotonic()
                ] or models_to_try
        # The token budget of the request is booked *before* the call, so that
        # the parallel workers of an evaluation share one global budget
        # instead of all hitting the per-minute limit at the same time.
        # The tokens are booked once per request: a retry after a 429 re-uses
        # the same reservation (the failed attempt did not consume the budget
        # in a meaningful way, and double-booking would starve the run).
        prompt_tokens = self._count_prompt_tokens(
            models_to_try[0] if models_to_try else self.model, system, prompt
        )
        booked = False
        for model_name in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    if not booked:
                        self._wait_for_tokens(model_name, prompt_tokens)
                        booked = True
                    self._wait_for_rpm(model_name)
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
                    # Gemma 4 (và các model suy luận khác) tính token suy
                    # luận vào `max_output_tokens`.  Khi phần suy luận dùng hết
                    # ngân sách, API trả `finish_reason=MAX_TOKENS` với `text`
                    # **rỗng** -- câu trả lời bị mất hoàn toàn.  Đo đạc ở
                    # docs/10 cho thấy đây là nguyên nhân 65/110 câu closed-book
                    # rỗng.  Ta phát hiện tình huống này và **thử lại với
                    # ngân sách lớn hơn** trước khi bỏ cuộc.
                    if not text.strip() and self._is_thinking_overrun(resp):
                        retry_budget = self._expanded_output_budget(
                            request.max_output_tokens or self.max_output_tokens
                        )
                        if retry_budget > (request.max_output_tokens or self.max_output_tokens):
                            logger.warning(
                                "Mô hình %s hết ngân sách token khi suy luận "
                                "(text rỗng) -- thử lại với max_output_tokens=%d",
                                model_name,
                                retry_budget,
                            )
                            config = {**config, "max_output_tokens": retry_budget}
                            kwargs = {"model": model_name, "contents": prompt}
                            kwargs["config"] = (
                                {**config, "system_instruction": system}
                                if system
                                else config
                            )
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
                        # Free tier: quota theo *phút* (token/phút) hoặc theo
                        # *ngày*.  Nếu API gợi ý thời gian chờ ngắn thì đó là
                        # quota theo phút: chờ rồi thử lại **cùng model**.
                        # Xoay sang model dự phòng sẽ đốt quota *ngày* của
                        # Gemini -- điều mà không thể hoàn lại trong một lượt
                        # chạy.  Xem docs/10_quota_gemini_free_tier.md.
                        wait = _retry_after_seconds(exc)
                        if wait is not None and wait <= self.max_rate_limit_wait:
                            self._throttled[model_name] = time.monotonic() + wait
                            if attempt + 1 < self.max_retries:
                                logger.warning(
                                    "Gemini giới hạn tốc độ (model=%s, %.0fs) -- "
                                    "chờ rồi thử lại",
                                    model_name,
                                    wait,
                                )
                                time.sleep(wait + 0.5)
                                continue
                            # Hết số lần thử của model này: thử model kế tiếp.
                            logger.warning(
                                "Gemini giới hạn tốc độ (model=%s) sau %d lần thử "
                                "-- chuyển model khác",
                                model_name,
                                self.max_retries,
                            )
                            break
                        # Quota *ngày* đã cạn: model này không dùng được cho
                        # tới ngày mai.
                        self._exhausted_models.add(model_name)
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

        Results are cached on disk (keyed by model + text) so that repeated
        evaluation runs over the same questions do not consume quota twice.
        """
        if not texts:
            return []
        results: list[list[float] | None] = []
        pending: list[str] = []
        # text -> every position in ``results`` that needs its vector
        pending_slots: dict[str, list[int]] = {}
        for text in texts:
            hit = self._embed_cache_get(text)
            if hit is not None:
                results.append(hit)
                continue
            pending_slots.setdefault(text, []).append(len(results))
            if text not in pending_slots or len(pending_slots[text]) == 1:
                pending.append(text)
            results.append(None)
        for start in range(0, len(pending), self.embed_batch_size):
            batch = pending[start : start + self.embed_batch_size]
            vectors = self._embed_batch(batch)
            for text, vec in zip(batch, vectors, strict=False):
                self._embed_cache_put(text, vec)
                for slot in pending_slots[text]:
                    results[slot] = vec
        out = [r if r is not None else [] for r in results]
        if out:
            self._embed_dim = len(out[0])
        return out

    def _embed_cache_key(self, text: str) -> str:
        return stable_hash({"model": self.embed_model, "text": text, "dim": self.output_dim})

    def _embed_cache_get(self, text: str) -> list[float] | None:
        if self.cache is None:
            return None
        payload = self.cache.get(self._embed_cache_key(text))
        if not payload:
            return None
        vec = payload.get("vector")
        if isinstance(vec, list) and vec:
            return [float(x) for x in vec]
        return None

    def _embed_cache_put(self, text: str, vector: list[float]) -> None:
        if self.cache is None:
            return
        self.cache.put(
            self._embed_cache_key(text),
            {"vector": list(vector), "model": self.embed_model},
        )

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed one batch, retrying with the delay suggested by the API.

        The free tier of the embedding models limits the number of embed
        *tokens* per minute (and per day).  When the limit is hit the API
        returns a ``RetryInfo`` telling how long to wait -- we honour it, but
        only up to ``max_rate_limit_wait`` seconds: a *daily* quota reset
        (which can be ~20 hours away) must fail fast instead of freezing the
        whole run.  In that case we rotate to a sibling embedding model.

        Some embedding models (``gemini-embedding-2``) only accept a *single*
        input per request, so the batch is split into one call per text
        whenever the API returns fewer vectors than requested.

        **Circuit breaker.**  Once every embedding model has refused with a
        "quota exhausted" error, the client remembers it and fails *fast*
        on the next calls.  The alternative is catastrophic: the KG build
        resolves thousands of mentions, and every single one of them would
        wait for the full retry ladder of every embedding model.
        """
        if self._embed_quota_exhausted:
            raise LLMError(
                "Quota embedding của Gemini đã cạn (đã ghi nhớ) -> "
                "dùng model embedding cục bộ"
            )
        last_error: Exception | None = None
        models_to_try = [self.embed_model, *self.fallback_embed_models]
        quota_hit = False
        for model_name in models_to_try:
            for attempt in range(self.embed_max_retries):
                try:
                    vectors = self._embed_once(model_name, batch)
                    if len(vectors) != len(batch):
                        # The model collapsed the batch into a single vector:
                        # embed the texts one by one instead.
                        vectors = []
                        for text in batch:
                            single = self._embed_once(model_name, [text])
                            vectors.extend(single)
                    if model_name != self.embed_model:
                        logger.warning(
                            "Đã chuyển sang model embedding %s (quota của %s đã cạn)",
                            model_name,
                            self.embed_model,
                        )
                        self.embed_model = model_name
                    return vectors
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    suggested = _retry_after_seconds(exc)
                    if suggested is not None and suggested > self.max_rate_limit_wait:
                        logger.warning(
                            "Embed quota của %s đã cạn (cần chờ %.0fs) -> thử model khác",
                            model_name,
                            suggested,
                        )
                        quota_hit = True
                        break
                    delay = suggested or (self.retry_base_delay * (2**attempt))
                    logger.warning(
                        "Embed call failed (attempt %d/%d, model=%s): %s -- retry sau %.1fs",
                        attempt + 1,
                        self.embed_max_retries,
                        model_name,
                        str(exc)[:160],
                        delay,
                    )
                    time.sleep(delay)
        if quota_hit:
            self._embed_quota_exhausted = True
        raise LLMError(
            f"Embed call thất bại sau {self.embed_max_retries} lần thử: {last_error}"
        )

    def _embed_once(self, model_name: str, batch: list[str]) -> list[list[float]]:
        """One API call; returns the (normalised) vectors it produced.

        The batch is sent as a list of ``Content`` objects when the SDK
        exposes them: some embedding models (e.g. ``gemini-embedding-2``)
        collapse a list of plain strings into a *single* vector, while a list
        of ``Content`` keeps one vector per input.
        """
        types = getattr(self._genai, "types", None)
        contents: list[Any] = batch
        if types is not None:
            contents = [
                types.Content(parts=[types.Part(text=text)]) for text in batch
            ]
        kwargs: dict[str, Any] = {"model": model_name, "contents": contents}
        if self.output_dim:
            kwargs["config"] = {"output_dimensionality": self.output_dim}
        resp = self._client.models.embed_content(**kwargs)
        vectors: list[list[float]] = []
        for item in resp.embeddings:
            vec = list(item.values)
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

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
                fallback_embed_models=llm_cfg.get("fallback_embed_models", []),
                model_rpm=llm_cfg.get("model_rpm", {}),
                model_tpm=llm_cfg.get("model_tpm", {}),
                count_tokens=llm_cfg.get("count_tokens", True),
            )
        except LLMError as exc:
            logger.warning("Không khởi tạo được GeminiClient (%s) -> dùng mock", exc)
    return MockLLMClient(embed_dim=settings.embed_dim, seed=settings.seed)
