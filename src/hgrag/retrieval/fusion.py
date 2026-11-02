"""Fusion of the vector and graph retrieval lists (docs/03 §6).

Three mechanisms are combined here:

1. **Reciprocal Rank Fusion (RRF)** -- the standard formula

   .. math::

       score(d) = \\sum_{list \\in L} \\frac{1}{k + rank_{list}(d)}

   with ``k = 60``.  RRF only needs the *ranking*, not the scores, which is
   exactly why it can merge a cosine-similarity list (vector) with a
   path-scoring list (graph) whose scales are incomparable.

2. **Temporal boost** -- evidence whose text mentions a year of the question
   gets a small additive bonus.

3. **Diversity cap** -- at most ``max_per_doc`` evidence of the same
   ``doc_id`` survive (docs/03 §6: "prompt bị nhiễu" is a real cause of
   accuracy loss).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "FusionResult",
    "apply_diversity_cap",
    "apply_temporal_boost",
    "dedupe_by_text",
    "fuse",
    "reciprocal_rank_fusion",
]

_YEAR_RE = re.compile(r"\b(\d{3,4})\b")


def _years_in(text: str) -> set[int]:
    out: set[int] = set()
    for match in _YEAR_RE.finditer(text or ""):
        try:
            out.add(int(match.group(1)))
        except ValueError:
            continue
    return out


def _doc_id_of(item: Mapping[str, Any]) -> str:
    payload = item.get("payload") or {}
    if isinstance(payload, Mapping):
        doc_id = payload.get("doc_id") or payload.get("source_id")
        if doc_id:
            return str(doc_id)
    meta = item.get("meta") or {}
    if isinstance(meta, Mapping):
        doc_id = meta.get("doc_id")
        if doc_id:
            return str(doc_id)
    for key in ("doc_id", "source_doc_id"):
        if item.get(key):
            return str(item[key])
    return ""


def _diversity_key(item: Mapping[str, Any]) -> str:
    """The bucket an item is counted against by the diversity cap.

    A graph path has no ``doc_id``: it is not a passage of a document but a
    chain of KG edges.  Counting *all* graph paths against the same bucket
    (``"__nodoc__"``) would let the first few paths evict every later one --
    with ``max_per_doc=3`` only three graph paths ever reached the context,
    no matter how many the traversal had found.  Each path therefore counts
    against its own bucket, keyed by its chain of nodes.
    """
    payload = item.get("payload") or {}
    if isinstance(payload, Mapping) and payload.get("kind") == "graph_path":
        path = payload.get("path") or {}
        nodes = path.get("nodes") if isinstance(path, Mapping) else None
        if nodes:
            return "graph:" + "|".join(
                str(n.get("uid") or n.get("name") or "") for n in nodes
            )
        return f"graph:{item.get('id') or ''}"
    return _doc_id_of(item) or "__nodoc__"


def _normalise_text(text: str) -> str:
    """A comparison key for two pieces of context text."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _text_of(item: Mapping[str, Any]) -> str:
    for key in ("text", "title", "name"):
        value = item.get(key)
        if value:
            return str(value)
    payload = item.get("payload") or {}
    if isinstance(payload, Mapping):
        for key in ("text", "title"):
            value = payload.get(key)
            if value:
                return str(value)
    meta = item.get("meta") or {}
    if isinstance(meta, Mapping):
        for key in ("text", "title"):
            value = meta.get(key)
            if value:
                return str(value)
    return ""


# --------------------------------------------------------------------------
# RRF
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse several ranked lists with the RRF formula.

    Parameters
    ----------
    rankings:
        ``{source_name: [{"id": ..., "score": ...}, ...]}`` -- the lists must
        already be sorted by decreasing relevance.
    k:
        The RRF constant (60 in the paper; smaller values favour the top of
        each list more aggressively).

    Returns
    -------
    list[dict]
        ``[{"id", "rrf_score", "vector_rank", "graph_rank", "source",
        "payload"}]`` sorted by decreasing ``rrf_score``.
    """
    fused: dict[str, dict[str, Any]] = {}
    for source, items in (rankings or {}).items():
        for rank, item in enumerate(items or [], start=1):
            if not isinstance(item, Mapping):
                continue
            identifier = str(item.get("id") or "")
            if not identifier:
                continue
            entry = fused.get(identifier)
            if entry is None:
                entry = {
                    "id": identifier,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "graph_rank": None,
                    "source": "",
                    "payload": item.get("payload") or {},
                    "scores": {},
                    "ranks": {},
                }
                fused[identifier] = entry
            entry["rrf_score"] += 1.0 / (k + rank)
            entry["scores"][source] = float(item.get("score") or 0.0)
            entry["ranks"][source] = rank
            entry["source"] = source
            if source == "vector":
                entry["vector_rank"] = rank
            elif source == "graph":
                entry["graph_rank"] = rank
            if item.get("payload"):
                entry["payload"] = item["payload"]
    out = list(fused.values())
    out.sort(key=lambda e: (-float(e["rrf_score"]), str(e["id"])))
    return out


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class FusionResult:
    """The fused candidate list."""

    items: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {"items": list(self.items), "n_items": len(self.items)}


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------


def apply_temporal_boost(
    items: list[dict[str, Any]],
    year_refs: Iterable[int] | None,
    boost: float = 0.15,
) -> list[dict[str, Any]]:
    """Add ``boost`` to the score of the items mentioning a year of the question.

    The list is re-sorted afterwards (the boost is small but can flip two
    adjacent items).
    """
    years = {int(y) for y in (year_refs or [])}
    if not years:
        return items
    out: list[dict[str, Any]] = []
    for item in items or []:
        entry = dict(item)
        item_years = _years_in(_text_of(entry))
        if item_years & years:
            entry["temporal_match"] = True
            entry["rrf_score"] = float(entry.get("rrf_score") or 0.0) + float(boost)
        else:
            entry["temporal_match"] = False
        out.append(entry)
    out.sort(key=lambda e: (-float(e.get("rrf_score") or 0.0), str(e.get("id") or "")))
    return out


def dedupe_by_text(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop items whose text is identical to an item already kept.

    The traversal can reach the same chain of edges through several routes
    (``Nhà Tần - ABOUT -> giải thể`` appeared twice for ``vhe_00013``), and
    the KG stores one node per *mention*, so two chains can also render
    identically while having different ids.  Repeating the same sentence in
    the context only burns the token budget and makes the answer generator
    believe the fact is stated twice.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items or []:
        key = _normalise_text(_text_of(item))
        if not key:
            out.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def apply_diversity_cap(
    items: list[dict[str, Any]],
    max_per_doc: int = 3,
) -> list[dict[str, Any]]:
    """Keep at most ``max_per_doc`` items per document.

    Graph paths are counted against their *own* bucket (see
    :func:`_diversity_key`), so the cap limits repeated passages without
    evicting the graph paths.
    """
    cap = max(1, int(max_per_doc))
    counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for item in items or []:
        doc_id = _diversity_key(item)
        if counts.get(doc_id, 0) >= cap:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        out.append(item)
    return out


# --------------------------------------------------------------------------
# High-level entry point
# --------------------------------------------------------------------------


def fuse(
    vector_hits: Sequence[Mapping[str, Any]] | None,
    graph_paths: Sequence[Any] | None,
    k: int = 60,
    year_refs: Iterable[int] | None = None,
    max_per_doc: int = 3,
) -> FusionResult:
    """Fuse the vector hits and the graph paths into a single ranking.

    ``vector_hits`` are the raw hits of the vector store
    (``{"id", "score", "payload"}``) while ``graph_paths`` are
    :class:`~hgrag.retrieval.graph_search.GraphPath` objects (or plain
    dicts with an ``id``/``text``).  Both are converted into the RRF input
    format before being fused.
    """
    vector_list = [
        {
            "id": str(hit.get("id") or ""),
            "score": float(hit.get("score") or 0.0),
            "payload": hit.get("payload") or {},
        }
        for hit in (vector_hits or [])
        if str(hit.get("id") or "")
    ]

    graph_list: list[dict[str, Any]] = []
    for index, path in enumerate(graph_paths or []):
        if isinstance(path, Mapping):
            identifier = str(path.get("id") or f"graph:{index}")
            text = str(path.get("text") or "")
            score = float(path.get("score") or 0.0)
            payload = dict(path.get("payload") or {})
        else:
            identifier = str(getattr(path, "path_id", "") or f"graph:{index}")
            # The *readable* rendering is what goes into the prompt: the
            # compact ``A -REL-> B`` chain is kept as ``path_text`` for the
            # logs and for the path signature.  Without the readable form the
            # answer generator treats a graph path as an opaque list of
            # relation labels and refuses to answer (measured: 42 % of the
            # VHE template questions).
            natural = getattr(path, "to_natural_text", None)
            text = str(natural() or "") if callable(natural) else ""
            if not text:
                text = str(getattr(path, "to_text", lambda: "")() or "")
            score = float(getattr(path, "score", 0.0) or 0.0)
            payload = {
                "kind": "graph_path",
                "text": text,
                "path_text": str(getattr(path, "to_text", lambda: "")() or ""),
                "hops": int(getattr(path, "hops", 0) or 0),
                "source_chunk_ids": list(getattr(path, "source_chunk_ids", []) or []),
                "path": getattr(path, "to_dict", lambda: {})(),
            }
        graph_list.append(
            {"id": identifier, "score": score, "payload": {**payload, "text": text}}
        )

    fused = reciprocal_rank_fusion({"vector": vector_list, "graph": graph_list}, k=k)
    fused = apply_temporal_boost(fused, year_refs)
    fused = dedupe_by_text(fused)
    fused = apply_diversity_cap(fused, max_per_doc=max_per_doc)
    return FusionResult(items=fused)
