"""Reranking of the vector candidates (docs/03 §4).

The reference system uses a cross-encoder (``bge-reranker-v2-m3`` or
``ViRanker``) to reorder the top-30 vector hits down to the top-10.  Those
models need a GPU, so they are **optional**: :class:`CrossEncoderReranker`
uses one when a local model is available, and :class:`HeuristicReranker` is
the always-available fallback that works offline.

The heuristic score combines four signals:

* ``similarity``  -- cosine similarity between the embedding of the query and
  the embedding of the candidate (falls back to a token overlap when no
  embedder is given);
* ``keyword``     -- a BM25-ish overlap between the (accent-free) query
  tokens and the candidate tokens, with the inverse-document-frequency
  weighting estimated on the candidate list itself;
* ``temporal``    -- 1 when the candidate mentions a year of the question;
* ``length_penalty`` -- a mild penalty for very long candidates (they eat the
  context budget without adding information).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "CrossEncoderReranker",
    "HeuristicReranker",
    "Reranker",
    "build_reranker",
]

_YEAR_RE = re.compile(r"\b(\d{3,4})\b")
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN_RE.findall(text or "")]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b, strict=False):
        num += x * y
        da += x * x
        db += y * y
    if da <= 0 or db <= 0:
        return 0.0
    return num / ((da**0.5) * (db**0.5))


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    """Extract the searchable text of a candidate."""
    for key in ("text", "content", "body"):
        value = candidate.get(key)
        if value:
            return str(value)
    payload = candidate.get("payload")
    if isinstance(payload, Mapping):
        for key in ("text", "title"):
            value = payload.get(key)
            if value:
                return str(value)
    meta = candidate.get("meta")
    if isinstance(meta, Mapping):
        for key in ("text", "title"):
            value = meta.get(key)
            if value:
                return str(value)
    return ""


def _candidate_title(candidate: Mapping[str, Any]) -> str:
    payload = candidate.get("payload")
    if isinstance(payload, Mapping) and payload.get("title"):
        return str(payload["title"])
    if candidate.get("title"):
        return str(candidate["title"])
    return ""


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------


@runtime_checkable
class Reranker(Protocol):
    """Anything able to reorder a list of candidates."""

    name: str

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]: ...


# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------


class HeuristicReranker:
    """Offline reranker: similarity + BM25-ish keyword overlap + temporal."""

    name = "heuristic"

    def __init__(
        self,
        embedder: Any = None,
        weights: Mapping[str, float] | None = None,
        length_penalty: float = 0.15,
        max_chars: int = 1200,
    ) -> None:
        self.embedder = embedder
        self.weights = dict(
            weights
            or {
                "similarity": 0.40,
                "keyword": 0.40,
                "temporal": 0.20,
            }
        )
        self.length_penalty = float(length_penalty)
        self.max_chars = int(max_chars)

    # -- public ---------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        items = [dict(c) for c in (candidates or [])]
        if not items:
            return []
        query_tokens = _tokens(query)
        query_years = {
            int(m.group(1)) for m in _YEAR_RE.finditer(query or "") if m.group(1)
        }
        idf = self._idf(items)
        query_vec, cand_vecs = self._embed_pair(query, items)

        scored: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            text = _candidate_text(item)
            tokens = _tokens(text)
            similarity = self._similarity(query_vec, cand_vecs, index)
            keyword = self._keyword_overlap(query_tokens, tokens, idf)
            temporal = self._temporal(text, query_years)
            penalty = self._length_penalty(text)
            score = (
                self.weights["similarity"] * similarity
                + self.weights["keyword"] * keyword
                + self.weights["temporal"] * temporal
                - self.length_penalty * penalty
            )
            item["rerank_score"] = score
            item["rerank_features"] = {
                "similarity": round(similarity, 4),
                "keyword": round(keyword, 4),
                "temporal": round(temporal, 4),
                "length_penalty": round(penalty, 4),
            }
            item["score"] = score
            scored.append(item)
        scored.sort(key=lambda e: (-float(e.get("rerank_score") or 0.0), str(e.get("id") or "")))
        limit = max(1, int(top_k))
        return scored[:limit]

    # -- internals ------------------------------------------------------
    def _embed(self, text: str) -> list[float] | None:
        embedder = self.embedder
        if embedder is None or not (text or "").strip():
            return None
        try:
            return list(embedder.embed_one(text))
        except Exception as exc:  # noqa: BLE001
            logger.debug("embed failed: %s", exc)
            return None

    def _embed_pair(
        self, query: str, items: Sequence[Mapping[str, Any]]
    ) -> tuple[list[float] | None, list[list[float]]]:
        """Embed the query and every candidate in **one** batched call.

        Embedding the candidates one by one costs one API request per
        candidate (30 per question), which exhausts the free-tier embedding
        quota after a handful of questions.  When the embedder is unavailable
        the reranker keeps working -- the ``similarity`` signal is then 0 and
        the BM25-ish keyword overlap carries the ranking.
        """
        embedder = self.embedder
        if embedder is None or not (query or "").strip():
            return None, []
        texts = [query, *[_candidate_text(i) for i in items]]
        try:
            vectors = list(embedder.embed(texts))
        except Exception as exc:  # noqa: BLE001
            logger.info("Không embed được candidate (%s) -> bỏ qua similarity", exc)
            return None, []
        if not vectors or len(vectors) != len(texts):
            logger.info("Embed trả về %d/%d vector -> bỏ qua similarity", len(vectors), len(texts))
            return None, []
        return list(vectors[0]), [list(v) for v in vectors[1:]]

    def _similarity(
        self,
        query_vec: list[float] | None,
        cand_vecs: Sequence[Sequence[float]],
        index: int,
    ) -> float:
        if query_vec is None or index >= len(cand_vecs):
            return 0.0
        return max(0.0, min(1.0, _cosine(query_vec, cand_vecs[index])))

    @staticmethod
    def _idf(items: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Inverse document frequency estimated on the candidate list."""
        n_docs = max(1, len(items))
        df: dict[str, int] = {}
        for item in items:
            for token in set(_tokens(_candidate_text(item))):
                df[token] = df.get(token, 0) + 1
        return {token: math.log(1 + n_docs / count) for token, count in df.items()}

    @staticmethod
    def _keyword_overlap(
        query_tokens: Sequence[str],
        candidate_tokens: Sequence[str],
        idf: Mapping[str, float],
    ) -> float:
        if not query_tokens or not candidate_tokens:
            return 0.0
        candidate_set = set(candidate_tokens)
        total = 0.0
        weight = 0.0
        for token in set(query_tokens):
            w = float(idf.get(token, math.log(1 + 1)))
            weight += w
            if token in candidate_set:
                total += w
        if weight <= 0:
            return 0.0
        return max(0.0, min(1.0, total / weight))

    @staticmethod
    def _temporal(text: str, query_years: set[int]) -> float:
        if not query_years:
            return 0.0
        years = {int(m.group(1)) for m in _YEAR_RE.finditer(text or "") if m.group(1)}
        return 1.0 if years & query_years else 0.0

    def _length_penalty(self, text: str) -> float:
        length = len(text or "")
        if length <= self.max_chars:
            return 0.0
        return min(1.0, (length - self.max_chars) / float(self.max_chars))


class CrossEncoderReranker:
    """Reranker backed by a local cross-encoder (optional dependency).

    The model is loaded lazily; if ``sentence_transformers`` or the model
    weights are unavailable, :meth:`rerank` falls back to the heuristic
    reranker so that the pipeline never breaks.
    """

    name = "cross-encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        fallback: Any = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self._model: Any = None
        self._load_error: str = ""
        self.fallback = fallback if fallback is not None else HeuristicReranker()

    def _ensure_model(self) -> Any:
        if self._model is not None or self._load_error:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.warning(
                "Không nạp được cross-encoder %s (%s) -> dùng heuristic reranker",
                self.model_name,
                exc,
            )
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        items = [dict(c) for c in (candidates or [])]
        if not items:
            return []
        model = self._ensure_model()
        if model is None:
            return self.fallback.rerank(query, items, top_k=top_k)
        pairs = [(query or "", _candidate_text(item)) for item in items]
        try:
            scores = model.predict(pairs, max_length=self.max_length)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cross-encoder predict failed (%s) -> heuristic", exc)
            return self.fallback.rerank(query, items, top_k=top_k)
        for item, score in zip(items, scores, strict=True):
            item["rerank_score"] = float(score)
            item["score"] = float(score)
        items.sort(key=lambda e: -float(e.get("rerank_score") or 0.0))
        return items[: max(1, int(top_k))]


def build_reranker(settings: Any = None, embedder: Any = None) -> Reranker:
    """Build the best reranker available in this environment."""
    heuristic = HeuristicReranker(embedder=embedder)
    if settings is None:
        return heuristic
    cfg = getattr(settings, "retrieval", None)
    if cfg is None and isinstance(settings, dict):
        cfg = settings.get("retrieval")
    cfg = dict(cfg or {})
    if str(cfg.get("rerank_backend") or "auto").lower() == "heuristic":
        return heuristic
    model_name = str(cfg.get("rerank_model") or "BAAI/bge-reranker-v2-m3")
    return CrossEncoderReranker(model_name=model_name, fallback=heuristic)
