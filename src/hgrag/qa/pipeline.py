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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
            "subgraph": dict(self.explanation.get("subgraph") or {}),
            "grounded": bool(self.citations),
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
        lexical_index: Any = None,
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
        self.lexical_index = lexical_index

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
        # The VHE gold sentences live in their own collection: they are
        # classical Vietnamese and must not pollute the Wikipedia vector
        # space.  The pipeline searches *all* of them and merges the hits.
        self.chunk_collections: list[str] = [self.chunk_collection]
        # The vector collection depends on the *active* embedder: the Gemini
        # vectors are 768-dim, a local sentence-transformers model is usually
        # 384-dim.  Mixing them in the same collection would break the search.
        self._collection_overrides: dict[str, str] = {}
        self._answer_generator: AnswerGenerator | None = None
        self._explainer: Explainer | None = None
        self._understanding_cache: dict[str, QueryUnderstanding] = {}

    def set_chunk_collection(self, name: str) -> None:
        """Override the Qdrant collection used for the chunk vectors."""
        if name:
            self.chunk_collection = str(name)
            self.chunk_collections = [self.chunk_collection]

    def set_chunk_collections(self, names: Sequence[str]) -> None:
        """Search several Qdrant collections at once (merged by score)."""
        cleaned = [str(n) for n in (names or []) if str(n).strip()]
        if not cleaned:
            return
        self.chunk_collections = cleaned
        self.chunk_collection = cleaned[0]

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
        """Understand ``question`` (cached: the plan only depends on the text)."""
        cached = self._understanding_cache.get(question)
        if cached is not None:
            return cached
        module = QueryUnderstandingModule(self.llm_client)
        understanding = module.understand(question)
        if len(self._understanding_cache) >= 512:
            self._understanding_cache.clear()
        self._understanding_cache[question] = understanding
        return understanding

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
            return self._lexical_search(question, top_k, understanding, flags)
        try:
            query_vec = self.embedder.embed_one(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Embedding câu hỏi thất bại (%s) -> dùng tìm kiếm từ khoá", exc
            )
            return self._lexical_search(question, top_k, understanding, flags)
        filters = self._filters(understanding, flags)
        hits = self._search_collections(query_vec, top_k, filters)
        if not hits:
            # Không có kết quả vector -> thử tìm kiếm từ khoá trước khi kết luận.
            lexical = self._lexical_search(question, top_k, understanding, flags)
            if lexical:
                return lexical
        return list(hits or [])

    def _search_collections(
        self,
        query_vec: Sequence[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Search every chunk collection and merge the hits.

        The VHE gold corpus and the Wikipedia corpus are indexed in two
        separate collections (different vector dimensions are not the only
        reason: the two corpora have a very different register).  The hits of
        the two collections are merged and re-sorted by score, so a VHE
        sentence can compete with a Wikipedia paragraph on equal terms.
        """
        collections = list(self.chunk_collections or [self.chunk_collection])
        merged: list[dict[str, Any]] = []
        errors: list[str] = []
        for name in collections:
            try:
                hits = self.vector_store.search(
                    name, list(query_vec), top_k=top_k, filters=filters or None
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Vector search thất bại ở collection %r: %s", name, exc)
                errors.append(name)
                continue
            merged.extend(hits or [])
        if errors and len(errors) == len(collections):
            logger.warning("Không collection nào tìm kiếm được: %s", ", ".join(errors))
            return []
        merged.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)
        return merged[:top_k]

    def _lexical_search(
        self,
        question: str,
        top_k: int,
        understanding: QueryUnderstanding,
        flags: Mapping[str, bool],
    ) -> list[dict[str, Any]]:
        """Tìm kiếm từ khoá (BM25) -- dự phòng khi không embed được câu hỏi."""
        index = self.lexical_index
        if index is None:
            return []
        filters = self._filters(understanding, flags)
        try:
            hits = index.search(question, top_k=top_k, filters=filters or None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lexical search thất bại: %s", exc)
            return []
        return list(hits or [])

    def _filters(
        self, understanding: QueryUnderstanding, flags: Mapping[str, bool]
    ) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if understanding.time_refs and not self._flag(flags, "no_temporal"):
            filters["year_overlap"] = list(understanding.time_refs)
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


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------


def build_pipeline(settings: Any = None, mock: bool = False) -> QAPipeline:
    """Build a fully wired :class:`QAPipeline` from the project settings.

    The Streamlit app and the CLI both use this factory, so the demo and the
    evaluation harness always run the *same* pipeline.
    """
    from ..config import get_settings
    from ..kg.alias import AliasResolver
    from ..kg.store import Neo4jGraphStore
    from ..llm.client import MockLLMClient, build_llm_client
    from ..retrieval.embeddings import MockEmbedder, build_embedder
    from ..retrieval.entity_linker import EntityLinker
    from ..retrieval.graph_search import GraphSearcher
    from ..retrieval.rerank import build_reranker
    from ..retrieval.vector_store import NumpyVectorStore, QdrantVectorStore

    settings = settings if settings is not None else get_settings()

    if mock:
        llm_client = MockLLMClient(embed_dim=settings.embed_dim, seed=settings.seed)
        embedder = MockEmbedder(dim=settings.embed_dim)
    else:
        llm_client = build_llm_client(settings)
        embedder = build_embedder(settings, llm_client)

    ncfg = settings.neo4j
    try:
        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=str(ncfg.get("user", "neo4j")),
            password=settings.neo4j_password,
            database=str(ncfg.get("database", "neo4j")),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không kết nối được Neo4j (%s) -> dùng InMemoryGraphStore", exc)
        from ..kg.store import InMemoryGraphStore

        store = InMemoryGraphStore()

    qcfg = settings.qdrant
    embed_dim = int(getattr(embedder, "dim", settings.embed_dim) or settings.embed_dim)
    # The Gemini vectors are 768-dim; a local sentence-transformers model is
    # 384-dim.  Each embedder therefore uses its *own* collection so that the
    # query vector always has the same dimension as the stored corpus.
    is_local = str(getattr(embedder, "name", "")).startswith("local:")
    if is_local:
        collection = str(qcfg.get("collection_chunks_local", "chunks_local"))
    else:
        collection = str(qcfg.get("collection_chunks", "chunks"))
    try:
        vector_store = QdrantVectorStore(
            url=str(qcfg.get("url", "http://localhost:6333")),
            vector_size=embed_dim,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không kết nối được Qdrant (%s) -> dùng NumpyVectorStore", exc)
        vector_store = NumpyVectorStore(vector_size=embed_dim)

    resolver = AliasResolver(
        store=store,
        llm_client=llm_client,
        embedder=embedder,
        settings=settings,
    )
    # The fuzzy tiers (T1 alias, T2 redirect, T4 name_key, T5 embedding) need
    # the alias index; without it the resolver can only do the exact-name
    # lookup of tier 0.
    try:
        entities = store.list_entities()
        resolver.build_alias_index(entities)
        logger.info("Alias index: %d entity", len(entities))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không dựng được alias index: %s", exc)

    linker = EntityLinker(resolver=resolver, settings=settings)
    searcher = GraphSearcher(store=store, settings=settings, embedder=embedder)

    # Chỉ mục từ khoá: dự phòng khi quota embedding của Gemini cạn.
    lexical_index = None
    try:
        from ..retrieval.lexical import build_lexical_index

        lexical_index = build_lexical_index(
            corpus_path=settings.data_processed / "corpus.jsonl",
            cache_path=settings.artifacts / "lexical_index.json",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không dựng được lexical index: %s", exc)

    pipeline = QAPipeline(
        llm_client=llm_client,
        embedder=embedder,
        graph_store=store,
        vector_store=vector_store,
        entity_linker=linker,
        graph_searcher=searcher,
        reranker=build_reranker(settings, embedder),
        lexical_index=lexical_index,
        settings=settings,
    )
    # The chunk corpus is split over several collections: the Wikipedia
    # chunks and the VHE gold sentences.  Both are searched and merged.
    collections = [collection]
    vhe_collection = str(
        qcfg.get(
            "collection_chunks_vhe_local" if is_local else "collection_chunks_vhe",
            "chunks_vhe_local" if is_local else "chunks_vhe",
        )
    )
    if vhe_collection != collection:
        collections.append(vhe_collection)
    pipeline.set_chunk_collections(collections)
    return pipeline


