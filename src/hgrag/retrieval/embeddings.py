"""Embedder protocol + Gemini / local / mock implementations."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Text -> dense vector."""

    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...

    @property
    def dim(self) -> int: ...


class LLMEmbedder:
    """Adapter around an :class:`~hgrag.llm.client.LLMClient` (Gemini or mock)."""

    def __init__(self, client: Any):
        self.client = client
        self.name = f"llm:{getattr(client, 'name', 'unknown')}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.client.embed(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return int(self.client.embed_dim)


class LocalEmbedder:
    """sentence-transformers backend (used on Kaggle GPU, or as an offline fallback).

    The model is loaded lazily and its dimension is read from the model
    itself, so the same code path works for a 384-dim multilingual MiniLM
    (CPU) and a 1024-dim Qwen3-Embedding (GPU).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name, device=device)
        self.name = f"local:{model_name}"
        self._dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=len(texts) > 32
        )
        return [list(map(float, v)) for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self._dim


class MockEmbedder:
    """Deterministic hash-based embedder for offline runs and tests.

    It is *not* semantically meaningful, but it is stable, normalised and
    gives a slightly higher score to texts sharing surface tokens -- enough to
    exercise the retrieval plumbing offline.
    """

    name = "mock"

    def __init__(self, dim: int = 768, seed: int = 42):
        self._dim = dim
        self.seed = seed

    def _vector(self, text: str) -> list[float]:
        import hashlib
        import math
        import re

        vec = [0.0] * self._dim
        tokens = re.findall(r"\w+", text.lower())
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            for k in range(4):
                idx = int.from_bytes(digest[k * 2 : k * 2 + 2], "big") % self._dim
                sign = 1.0 if digest[k * 2 + 1] % 2 == 0 else -1.0
                vec[idx] += sign
        # add a small deterministic global component so unrelated texts are not orthogonal
        gdigest = hashlib.sha256(text.encode("utf-8")).digest()
        for i in range(self._dim):
            vec[i] += (gdigest[i % len(gdigest)] / 255.0 - 0.5) * 0.05
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vector(text)

    @property
    def dim(self) -> int:
        return self._dim


def build_embedder(settings: Any | None = None, client: Any | None = None) -> Embedder:
    """Pick the best available embedder for the environment.

    Order of preference:

    1. ``mock``   -- deterministic offline stub (tests, CI);
    2. ``local``  -- a sentence-transformers model (offline fallback, no quota);
    3. ``gemini`` -- the hosted embedding model (best quality, but the free
       tier has a *daily* quota that is easy to exhaust);
    4. ``auto``   -- Gemini first, then a **local** model, then the mock.

    The local fallback matters: when the Gemini embedding quota is exhausted
    the pipeline must still be able to embed the query, otherwise the vector
    search silently degrades to zero hits and the system refuses to answer.
    """
    if settings is None:
        from ..config import get_settings

        settings = get_settings()

    provider = settings.embed_provider
    if provider == "mock":
        return MockEmbedder(dim=settings.embed_dim, seed=settings.seed)

    if provider == "local":
        embedder = _try_local(settings)
        if embedder is not None:
            return embedder
        return MockEmbedder(dim=settings.embed_dim, seed=settings.seed)

    if provider in ("gemini", "auto") and settings.has_gemini_key:
        if client is None:
            from ..llm.client import build_llm_client

            client = build_llm_client(settings)
        if getattr(client, "name", "") == "gemini":
            if provider == "gemini":
                return LLMEmbedder(client)
            # provider == "auto": Gemini is preferred, but a local model is
            # used when the Gemini *embedding* quota is exhausted.
            if _gemini_embedding_available(client):
                return LLMEmbedder(client)
            logger.warning(
                "Quota embedding của Gemini đã cạn -> dùng model embedding cục bộ"
            )
            embedder = _try_local(settings)
            if embedder is not None:
                return embedder
            return LLMEmbedder(client)

    embedder = _try_local(settings)
    if embedder is not None:
        return embedder
    return MockEmbedder(dim=settings.embed_dim, seed=settings.seed)


def _try_local(settings: Any) -> Embedder | None:
    """Build the local embedder, or return ``None`` when it is unavailable."""
    model_name = str(settings.embedding.get("local_model", "") or "")
    if not model_name:
        return None
    try:
        return LocalEmbedder(model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không dùng được local embedder %r (%s)", model_name, exc)
        return None


def _gemini_embedding_available(client: Any) -> bool:
    """Whether the Gemini embedding endpoint still has quota left.

    A single tiny embedding request is enough to detect the daily-quota
    exhaustion (``HTTP 429``) of the free tier without spending real quota.
    """
    try:
        vectors = client.embed(["ping"])
    except Exception as exc:  # noqa: BLE001
        logger.info("Embedding Gemini không khả dụng: %s", str(exc)[:160])
        return False
    return bool(vectors)
