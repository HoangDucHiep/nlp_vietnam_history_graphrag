"""End-to-end QA pipeline: question in, cited answer + subgraph out.

The pipeline follows docs/03:

1. query understanding
2. entity linking (anchors)
3. graph search (reasoning paths)
4. vector search (chunks)
5. rerank
6. RRF fusion
7. context building (token budget)
8. grounded answer generation
9. explanation

Every stage can be switched off through the ``ablation`` argument -- this is
what the evaluation harness (Wave 5) uses to measure the contribution of
each component.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..data.records import Evidence
from .answer import AnswerGenerator, AnswerResult
from .explain import Explainer
from .query_understanding import QueryUnderstanding, QueryUnderstandingModule

logger = logging.getLogger(__name__)

#: Ablation switches understood by :meth:`QAPipeline.ask`.
ABLATION_FLAGS = (
    "no_entity_linking",
    "no_resolution",
    "no_temporal",
    "no_graph",
    "no_rerank",
    "no_context_budget",
    "no_fusion",
)

REFUSAL_ANSWER = (
    "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này."
)


@dataclass
class QAResult:
    """The full result of one QA turn."""

    question: str = ""
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    reasoning_chain: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    explanation: dict[str, Any] = field(default_factory=dict)
    understanding: dict[str, Any] = field(default_factory=dict)
    refused: bool = False
    elapsed: float = 0.0
    system: str = "ours"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "reasoning_chain": list(self.reasoning_chain),
            "evidence": list(self.evidence),
            "explanation": dict(self.explanation),
            "understanding": dict(self.understanding),
            "refused": self.refused,
            "elapsed": round(float(self.elapsed), 4),
            "system": self.system,
        }


class QAPipeline:
    """The hybrid GraphRAG QA pipeline."""

    def __init__(
        self,
        llm_client: Any = None,
        embedder: Any = None,
        graph_store: Any = None,
        vector_store: Any = None,
        entity_linker: Any = None,
        graph_searcher: Any = None,
        context_builder: Any = None,
        reranker: Any = None,
        settings: Any = None,
    ) -> None:
        from ..config import get_settings

        self.settings = settings if settings is not None else get_settings()
        cfg = dict(getattr(self.settings, "retrieval", {}) or {})

        self.llm_client = llm_client
        self.embedder = embedder
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.entity_linker = entity_linker
        self.graph_searcher = graph_searcher
        self.context_builder = context_builder
        self.reranker = reranker

        self.top_k = int(cfg.get("vector_top_k", 30))
        self.rerank_top_k = int(cfg.get("rerank_top_k", 10))
        self.max_hops = int(cfg.get("graph_max_hops", 3))
        self.max_paths = int(cfg.get("graph_max_paths", 20))
        self.rrf_k = int(cfg.get("rrf_k", 60))
        self.max_tokens = int(cfg.get("context_max_tokens", 6000))
        self.max_evidence = int(cfg.get("max_evidence", 12))
        self.max_evidence_per_doc = int(cfg.get("max_evidence_per_doc", 3))
        self.temporal_boost = float(cfg.get("temporal_boost", 0.15))
        self.chunk_collection = str(
            (getattr(self.settings, "qdrant", {}) or {}).get("collection_chunks", "chunks")
        )
        self._answer_generator: AnswerGenerator | None = None
        self._explainer: Explainer | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: int | None = None,
        max_hops: int | None = None,
        ablation: Mapping[str, bool] | None = None,
        system: str = "ours",
    ) -> QAResult:
        """Answer ``question``."""
        started = time.perf_counter()
        flags = {name: bool((ablation or {}).get(name, False)) for name in ABLATION_FLAGS}
        top_k = int(top_k or self.top_k)
        max_hops = int(max_hops or self.max_hops)

        understanding = self._understand(question)
        if self._flag(flags, "no_temporal"):
            understanding.time_refs = []
            understanding.relation_plan = [
                r for r in understanding.relation_plan if "OCCURRED" not in r.upper()
            ]

        linked = self._link(understanding, flags)
        graph_result = self._graph_search(linked, understanding, max_hops, flags)
        vector_hits = self._vector_search(question, top_k, understanding, flags)
        vector_hits = self._rerank(question, vector_hits, flags)
        evidence = self._fuse(vector_hits, graph_result, understanding, flags)
        context_pack = self._build_context(evidence, flags)

        if context_pack is None or not context_pack.text.strip():
            result = QAResult(
                question=question,
                answer=REFUSAL_ANSWER,
                refused=True,
                understanding=understanding.to_dict(),
                system=system,
            )
            result.elapsed = time.perf_counter() - started
            return result

        answer = self._answer(question, context_pack, evidence, understanding)
        explanation = self._explain(
            question,
            linked,
            graph_result,
            {
                "n_vector_hits": len(vector_hits),
                "n_graph_paths": len(getattr(graph_result, "paths", []) or []),
                "n_fused": len(evidence),
                "n_evidence": len(context_pack.evidence_ids),
                "n_tokens_estimate": context_pack.n_tokens_estimate,
                "context_truncated": context_pack.truncated,
                "ablation": {k: v for k, v in flags.items() if v},
            },
        )

        result = QAResult(
            question=question,
            answer=answer.answer,
            citations=list(answer.citations),
            reasoning_chain=list(answer.reasoning_chain),
            evidence=[e.to_dict() for e in evidence],
            explanation=explanation.to_dict(),
            understanding=understanding.to_dict(),
            refused=answer.refused,
            system=system,
        )
        result.elapsed = time.perf_counter() - started
        return result

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def _understand(self, question: str) -> QueryUnderstanding:
        module = QueryUnderstandingModule(self.llm_client)
        return module.understand(question)

    def _flag(self, flags: Mapping[str, bool], name: str) -> bool:
        """Read an ablation flag, defaulting to ``False``."""
        return bool(flags.get(name, False))

    def _link(
        self, understanding: QueryUnderstanding, flags: Mapping[str, bool]
    ) -> list[Any]:
        if self._flag(flags, "no_entity_linking") or self.entity_linker is None:
            return []
        linker = self.entity_linker
        # A2: bỏ entity resolution -> chỉ giữ tier 0 (QID chính xác).
        if self._flag(flags, "no_resolution"):
            previous = getattr(linker, "use_resolution", True)
            try:
                linker.use_resolution = False
                return linker.link_from_query(
                    understanding.question,
                    entities=understanding.entities or None,
                )
            finally:
                linker.use_resolution = previous
        try:
            return linker.link_from_query(
                understanding.question,
                entities=understanding.entities or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Entity linking thất bại: %s", exc)
            return []

    def _graph_search(
        self,
        linked: Sequence[Any],
        understanding: QueryUnderstanding,
        max_hops: int,
        flags: Mapping[str, bool],
    ) -> Any:
        if self._flag(flags, "no_graph") or self.graph_searcher is None or not linked:
            return None
        anchors = [e.uid for e in linked if getattr(e, "uid", "")]
        if not anchors:
            return None
        try:
            return self.graph_searcher.search(
                anchors,
                relation_plan=understanding.relation_plan,
                max_hops=max_hops,
                max_paths=self.max_paths,
                query_text=understanding.question,
                year_refs=understanding.time_refs,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Graph search thất bại: %s", exc)
            return None

    def _vector_search(
        self,
        question: str,
        top_k: int,
        understanding: QueryUnderstanding,
        flags: Mapping[str, bool],
    ) -> list[dict[str, Any]]:
        if self.vector_store is None or self.embedder is None:
            return []
        try:
            query_vec = self.embedder.embed_one(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Embedding câu hỏi thất bại: %s", exc)
            return []
        filters = self._filters(understanding, flags)
        try:
            hits = self.vector_store.search(
                self.chunk_collection, query_vec, top_k=top_k, filters=filters or None
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector search thất bại: %s", exc)
            return []
        return list(hits or [])

    def _filters(
        self, understanding: QueryUnderstanding, flags: Mapping[str, bool]
    ) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if understanding.time_refs and not self._flag(flags, "no_temporal"):
            filters["year_min"] = understanding.time_refs
        return filters

    def _rerank(
        self, question: str, hits: list[dict[str, Any]], flags: Mapping[str, bool]
    ) -> list[dict[str, Any]]:
        if self._flag(flags, "no_rerank") or self.reranker is None or not hits:
            return hits[: self.top_k]
        try:
            return list(self.reranker.rerank(question, hits, top_k=self.rerank_top_k))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rerank thất bại: %s", exc)
            return hits[: self.top_k]

    def _fuse(
        self,
        vector_hits: list[dict[str, Any]],
        graph_result: Any,
        understanding: QueryUnderstanding,
        flags: Mapping[str, bool],
    ) -> list[Evidence]:
        from ..retrieval.fusion import apply_diversity_cap, apply_temporal_boost, fuse

        paths = list(getattr(graph_result, "paths", []) or [])
        if self._flag(flags, "no_fusion"):
            items: list[dict[str, Any]] = [
                {
                    "id": h.get("id"),
                    "score": float(h.get("score", 0.0)),
                    "payload": h.get("payload") or {},
                    "source": "vector",
                }
                for h in vector_hits
            ]
        else:
            fused = fuse(
                vector_hits,
                paths,
                k=self.rrf_k,
                year_refs=None
                if self._flag(flags, "no_temporal")
                else understanding.time_refs,
                max_per_doc=self.max_evidence_per_doc,
            )
            items = list(fused.items)

        if understanding.time_refs and not self._flag(flags, "no_temporal"):
            items = apply_temporal_boost(items, understanding.time_refs, self.temporal_boost)
        items = apply_diversity_cap(items, self.max_evidence_per_doc)

        return self._to_evidence(items, paths)[: self.max_evidence]

    def _to_evidence(
        self, items: list[dict[str, Any]], paths: Sequence[Any]
    ) -> list[Evidence]:
        """Turn the fused items into :class:`Evidence` objects."""
        path_by_text: dict[str, Any] = {}
        for path in paths:
            to_text = getattr(path, "to_text", None)
            if callable(to_text):
                key = to_text()
                if key:
                    path_by_text[key] = path

        evidence: list[Evidence] = []
        for index, item in enumerate(items, start=1):
            payload = item.get("payload") or {}
            meta = item.get("meta") or {}
            source = str(item.get("source", "vector"))
            text = ""
            chunk_id = ""
            title = ""
            url = ""
            for holder in (payload, meta, item):
                if not isinstance(holder, Mapping):
                    continue
                text = text or str(holder.get("text") or "")
                chunk_id = chunk_id or str(holder.get("chunk_id") or "")
                title = title or str(holder.get("title") or "")
                url = url or str(holder.get("url") or "")
            if not text:
                text = str(item.get("text") or "")
            path = path_by_text.get(str(item.get("id") or ""))
            path_payload = []
            if path is not None:
                path_payload = [
                    {
                        "src": e.get("src"),
                        "rel": e.get("rel_type"),
                        "dst": e.get("dst"),
                    }
                    for e in (getattr(path, "edges", []) or [])
                ]
            evidence.append(
                Evidence(
                    evidence_id=f"E{index}",
                    kind=source,
                    text=text,
                    score=float(item.get("rrf_score", item.get("score", 0.0)) or 0.0),
                    source_chunk_ids=[chunk_id] if chunk_id else [],
                    path=path_payload,
                    meta={
                        "title": title,
                        "url": url,
                        "chunk_id": chunk_id,
                        "rank_vector": item.get("vector_rank"),
                        "rank_graph": item.get("graph_rank"),
                    },
                )
            )
        return evidence

    def _build_context(
        self, evidence: list[Evidence], flags: Mapping[str, bool]
    ) -> Any:
        from ..retrieval.context import ContextBuilder

        builder = self.context_builder
        if builder is None:
            builder = ContextBuilder(self.settings)
            self.context_builder = builder
        budget = self.max_tokens
        # A6: bỏ context budget -> đưa toàn bộ evidence vào prompt.
        if self._flag(flags, "no_context_budget"):
            budget = 10_000_000
        try:
            return builder.build_context(evidence, max_tokens=budget)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Build context thất bại: %s", exc)
            return None

    def _answer(
        self,
        question: str,
        context_pack: Any,
        evidence: list[Evidence],
        understanding: QueryUnderstanding,
    ) -> AnswerResult:
        generator = self._answer_generator
        if generator is None:
            generator = AnswerGenerator(self.llm_client, self.settings)
            self._answer_generator = generator
        return generator.generate(
            question,
            context_pack.text,
            evidence=evidence,
            understanding=understanding,
        )

    def _explain(
        self,
        question: str,
        linked: Sequence[Any],
        graph_result: Any,
        trace: Mapping[str, Any],
    ) -> Any:
        explainer = self._explainer
        if explainer is None:
            explainer = Explainer(self.graph_searcher)
            self._explainer = explainer
        return explainer.explain(question, linked, graph_result, trace)

    # ------------------------------------------------------------------
    # baselines (used by the evaluation harness)
    # ------------------------------------------------------------------
    def answer_llm_only(self, question: str, system: str = "B0") -> QAResult:
        """B0 baseline: the LLM answers without any retrieval."""
        started = time.perf_counter()
        generator = AnswerGenerator(self.llm_client, self.settings)
        answer = generator.generate(question, "", evidence=[], understanding=None)
        result = QAResult(
            question=question,
            answer=answer.answer,
            citations=list(answer.citations),
            reasoning_chain=list(answer.reasoning_chain),
            refused=answer.refused,
            system=system,
        )
        result.elapsed = time.perf_counter() - started
        return result

    def answer_vanilla_rag(self, question: str, top_k: int = 5, system: str = "B1") -> QAResult:
        """B1 baseline: plain vector RAG (no graph, no rerank, no fusion)."""
        started = time.perf_counter()
        hits = self._vector_search(question, top_k, QueryUnderstanding(question), {})
        evidence: list[Evidence] = []
        for index, hit in enumerate(hits[:top_k], start=1):
            payload = hit.get("payload") or {}
            evidence.append(
                Evidence(
                    evidence_id=f"E{index}",
                    kind="chunk",
                    text=str(payload.get("text") or ""),
                    score=float(hit.get("score", 0.0) or 0.0),
                    source_chunk_ids=[str(payload.get("chunk_id") or "")],
                    meta={"title": payload.get("title", ""), "url": payload.get("url", "")},
                )
            )
        from ..retrieval.context import ContextBuilder

        builder = self.context_builder or ContextBuilder(self.settings)
        context = builder.build_context(evidence, max_tokens=self.max_tokens)
        generator = AnswerGenerator(self.llm_client, self.settings)
        answer = generator.generate(question, context.text, evidence=evidence)
        result = QAResult(
            question=question,
            answer=answer.answer,
            citations=list(answer.citations),
            reasoning_chain=list(answer.reasoning_chain),
            evidence=[e.to_dict() for e in evidence],
            refused=answer.refused,
            system=system,
        )
        result.elapsed = time.perf_counter() - started
        return result

    def answer_rag_plus_graph(
        self, question: str, top_k: int = 10, system: str = "B2"
    ) -> QAResult:
        """B2 baseline: RAG + graph context but *without* the rerank/fusion stage."""
        started = time.perf_counter()
        understanding = self._understand(question)
        linked = self._link(understanding, {})
        graph_result = self._graph_search(linked, understanding, self.max_hops, {})
        vector_hits = self._vector_search(question, top_k, understanding, {})
        evidence = self._fuse(vector_hits, graph_result, understanding, {"no_fusion": True})
        from ..retrieval.context import ContextBuilder

        builder = self.context_builder or ContextBuilder(self.settings)
        context = builder.build_context(evidence, max_tokens=self.max_tokens)
        generator = AnswerGenerator(self.llm_client, self.settings)
        answer = generator.generate(
            question, context.text, evidence=evidence, understanding=understanding
        )
        explainer = Explainer(self.graph_searcher)
        explanation = explainer.explain(question, linked, graph_result, {})
        result = QAResult(
            question=question,
            answer=answer.answer,
            citations=list(answer.citations),
            reasoning_chain=list(answer.reasoning_chain),
            evidence=[e.to_dict() for e in evidence],
            explanation=explanation.to_dict(),
            understanding=understanding.to_dict(),
            refused=answer.refused,
            system=system,
        )
        result.elapsed = time.perf_counter() - started
        return result
