"""Explainability: entity linking trace, reasoning paths and the subgraph.

The explanation is what makes a GraphRAG answer *verifiable*: the user can
see which entity of the question was anchored in the KG, which chains of
relations were traversed and which nodes/edges make up the final subgraph.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Everything needed to audit a single answer."""

    linked_entities: list[dict[str, Any]] = field(default_factory=list)
    paths: list[dict[str, Any]] = field(default_factory=list)
    subgraph: dict[str, Any] = field(default_factory=lambda: {"nodes": [], "edges": []})
    retrieval_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "linked_entities": list(self.linked_entities),
            "paths": list(self.paths),
            "subgraph": dict(self.subgraph),
            "retrieval_trace": dict(self.retrieval_trace),
        }


class Explainer:
    """Build the explanation of a QA turn."""

    def __init__(self, graph_searcher: Any = None) -> None:
        self.graph_searcher = graph_searcher

    # ------------------------------------------------------------------
    def explain(
        self,
        question: str,
        linked_entities: Sequence[Any] | None = None,
        graph_result: Any = None,
        retrieval_trace: Mapping[str, Any] | None = None,
    ) -> Explanation:
        """Assemble the explanation of one answer."""
        entities = [_linked_to_dict(e) for e in (linked_entities or [])]
        paths = _paths_to_dicts(graph_result)
        subgraph = self.build_subgraph(paths)
        trace = dict(retrieval_trace or {})
        trace.setdefault("n_linked_entities", len(entities))
        trace.setdefault("n_paths", len(paths))
        trace.setdefault("n_subgraph_nodes", len(subgraph.get("nodes", [])))
        return Explanation(
            linked_entities=entities,
            paths=paths,
            subgraph=subgraph,
            retrieval_trace=trace,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def build_subgraph(paths: Sequence[Any], max_nodes: int = 40) -> dict[str, Any]:
        """Deduplicate the nodes/edges of the retrieved paths.

        ``paths`` may be :class:`~hgrag.retrieval.graph_search.GraphPath`
        objects or plain dicts with ``nodes``/``edges`` keys.
        """
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()
        seen_edges: set[str] = set()

        for path in paths or []:
            path_nodes = _get(path, "nodes") or []
            path_edges = _get(path, "edges") or []
            for node in path_nodes:
                uid = str(node.get("uid") or node.get("id") or "")
                if not uid or uid in seen_nodes:
                    continue
                if len(nodes) >= max_nodes:
                    break
                seen_nodes.add(uid)
                nodes.append(
                    {
                        "id": uid,
                        "name": str(node.get("name") or uid),
                        "type": str(node.get("entity_type") or node.get("type") or "Entity"),
                        "qid": str(node.get("wikidata_id") or node.get("qid") or ""),
                        "year_min": node.get("year_min"),
                        "year_max": node.get("year_max"),
                    }
                )
            for edge in path_edges:
                src = str(edge.get("src") or edge.get("source") or "")
                dst = str(edge.get("dst") or edge.get("target") or "")
                rel = str(edge.get("rel_type") or edge.get("label") or edge.get("type") or "")
                if not src or not dst:
                    continue
                key = f"{src}|{rel}|{dst}"
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(
                    {
                        "source": src,
                        "target": dst,
                        "label": rel,
                        "type": rel,
                        "text": str(edge.get("text") or ""),
                        "confidence": edge.get("confidence"),
                        "source_chunk_ids": list(edge.get("source_chunk_ids") or []),
                    }
                )
        return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _linked_to_dict(entity: Any) -> dict[str, Any]:
    if isinstance(entity, Mapping):
        return dict(entity)
    to_dict = getattr(entity, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {
        "mention": str(getattr(entity, "mention", "") or ""),
        "uid": str(getattr(entity, "uid", "") or ""),
        "name": str(getattr(entity, "name", "") or ""),
        "wikidata_id": str(getattr(entity, "wikidata_id", "") or ""),
        "confidence": float(getattr(entity, "confidence", 0.0) or 0.0),
        "tier": int(getattr(entity, "tier", 5) or 5),
    }


def _paths_to_dicts(graph_result: Any) -> list[dict[str, Any]]:
    """Normalise a graph-search result into a list of path dicts."""
    if graph_result is None:
        return []
    raw_paths = _get(graph_result, "paths") or []
    out: list[dict[str, Any]] = []
    for path in raw_paths:
        if isinstance(path, Mapping):
            out.append(dict(path))
            continue
        to_dict = getattr(path, "to_dict", None)
        if callable(to_dict):
            out.append(dict(to_dict()))
            continue
        out.append(
            {
                "nodes": list(_get(path, "nodes") or []),
                "edges": list(_get(path, "edges") or []),
                "score": float(_get(path, "score") or 0.0),
                "hops": int(_get(path, "hops") or 0),
            }
        )
    return out


def format_paths(paths: Iterable[Any]) -> list[str]:
    """Human-readable rendering of the paths (``A -REL-> B``)."""
    out: list[str] = []
    for path in paths or []:
        nodes = _get(path, "nodes") or []
        edges = _get(path, "edges") or []
        if not nodes:
            continue
        parts = [str(nodes[0].get("name") or nodes[0].get("uid") or "")]
        for index, edge in enumerate(edges):
            rel = str(edge.get("rel_type") or edge.get("label") or "")
            nxt = nodes[index + 1] if index + 1 < len(nodes) else {}
            parts.append(f" -{rel}-> ")
            parts.append(str(nxt.get("name") or nxt.get("uid") or ""))
        out.append("".join(parts))
    return out
