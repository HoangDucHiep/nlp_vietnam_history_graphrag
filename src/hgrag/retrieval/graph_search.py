"""Graph retrieval: BFS traversal + path scoring (docs/03 §5).

The KG store protocol (:class:`~hgrag.kg.store.GraphStore`) only exposes
``neighbors()``.  Instead of writing a Neo4j-only Cypher traversal (which
would break the in-memory fallback used by the tests and by the offline
demo), the traversal is implemented as a **breadth-first search on top of
``neighbors()``**.  It is portable, works with every backend and is easy to
explain in the report.

Path scoring follows ``docs/03`` §5.2::

    score = w1 * relation_affinity
          + w2 * path_similarity
          + w3 * temporal_overlap
          - w4 * degree_penalty

* ``relation_affinity`` -- how well the relation types of the path match the
  *relation plan* produced by the query-understanding step.  Causal
  relations (``CAUSE_OF``/``PRECONDITION_OF``) and temporal relations
  (``OCCURRED_IN``/``HAPPENED_AT``) get a bonus when the question is causal
  or temporal.
* ``path_similarity`` -- cosine similarity between the embedding of the
  rendered path and the embedding of the question (0.5 when no embedder is
  available).
* ``temporal_overlap`` -- 1.0 when a year of the path is mentioned in the
  question, 0.5 when the question has no year at all.
* ``degree_penalty`` -- normalised ``log(1 + degree)``; it keeps *hub* nodes
  (``Nhà Lý`` connected to hundreds of edges) from dominating the ranking.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "GraphPath",
    "GraphSearchResult",
    "GraphSearcher",
    "PATH_SCORE_WEIGHTS",
    "CAUSAL_RELATIONS",
    "TEMPORAL_RELATIONS",
]

#: Weights of the path-scoring formula (see the module docstring).
PATH_SCORE_WEIGHTS: dict[str, float] = {
    "relation_affinity": 0.30,
    "path_similarity": 0.30,
    "temporal_overlap": 0.25,
    "degree_penalty": 0.15,
}

#: Relations that express causality.
CAUSAL_RELATIONS: frozenset[str] = frozenset(
    {"CAUSE_OF", "PRECONDITION_OF", "CAUSED_BY", "ENABLED_BY"}
)

#: Relations that express time.
TEMPORAL_RELATIONS: frozenset[str] = frozenset(
    {"OCCURRED_IN", "HAPPENED_AT", "PRECEDED_BY", "SIMULTANEOUS_WITH", "OCCURRED_AT"}
)

#: Relations that carry provenance -- they are useful but should not be the
#: backbone of a reasoning chain.
PROVENANCE_RELATIONS: frozenset[str] = frozenset({"SUPPORTED_BY"})

_CAUSAL_CUE_RE = re.compile(
    r"(vì sao|tại sao|nguyên nhân|lý do|do đâu|vì|làm sao|cause)", re.IGNORECASE
)
_TEMPORAL_CUE_RE = re.compile(
    r"(khi nào|thời gian|năm|nào trước|nào sau|trước|sau|kỷ|kỳ|mốc|thời đại)",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(\d{3,4})\b")


def _extract_years(text: str) -> list[int]:
    """Extract plausible historical years (905-1975) mentioned in ``text``."""
    out: list[int] = []
    for match in _YEAR_RE.finditer(text or ""):
        try:
            year = int(match.group(1))
        except ValueError:
            continue
        if 905 <= year <= 1975:
            out.append(year)
    return out


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


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------


@dataclass
class GraphPath:
    """A chain of KG nodes joined by typed relations."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    hops: int = 0

    def to_text(self) -> str:
        """Render the path as ``A -REL-> B -REL-> C``."""
        if not self.nodes:
            return ""
        parts = [str(self.nodes[0].get("name") or self.nodes[0].get("uid") or "")]
        for index, edge in enumerate(self.edges):
            rel = str(edge.get("rel_type") or "")
            nxt = self.nodes[index + 1] if index + 1 < len(self.nodes) else {}
            name = str(nxt.get("name") or nxt.get("uid") or "")
            parts.append(f" -{rel}-> ")
            parts.append(name)
        return "".join(parts)

    @property
    def node_uids(self) -> list[str]:
        return [str(n.get("uid") or "") for n in self.nodes]

    @property
    def source_chunk_ids(self) -> list[str]:
        out: list[str] = []
        for edge in self.edges:
            for cid in edge.get("source_chunk_ids") or []:
                if cid not in out:
                    out.append(str(cid))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "score": round(float(self.score), 4),
            "hops": self.hops,
            "text": self.to_text(),
            "source_chunk_ids": self.source_chunk_ids,
        }


@dataclass
class GraphSearchResult:
    """Everything a graph retrieval step produced."""

    paths: list[GraphPath] = field(default_factory=list)
    anchor_uids: list[str] = field(default_factory=list)
    n_expanded: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": [p.to_dict() for p in self.paths],
            "anchor_uids": list(self.anchor_uids),
            "n_expanded": self.n_expanded,
        }

    def subgraph(self) -> dict[str, Any]:
        """Node/edge view of the result (for the Streamlit "Đồ thị" tab)."""
        return extract_subgraph(self.paths)


# --------------------------------------------------------------------------
# Searcher
# --------------------------------------------------------------------------


class GraphSearcher:
    """Retrieve reasoning paths around a set of anchor nodes.

    Parameters
    ----------
    store:
        A :class:`~hgrag.kg.store.GraphStore` (Neo4j or in-memory).
    ontology:
        Optional VHE ontology (kept for API symmetry with the rest of the
        pipeline; the traversal itself is ontology-agnostic).
    embedder:
        Optional embedder used for the *path similarity* term.
    settings:
        Optional project settings (``retrieval.graph_max_hops`` & friends).
    """

    def __init__(
        self,
        store: Any,
        ontology: Any = None,
        embedder: Any = None,
        settings: Any = None,
    ) -> None:
        self.store = store
        self.ontology = ontology
        self.embedder = embedder
        self.settings = settings
        cfg: dict[str, Any] = {}
        if settings is not None:
            raw = getattr(settings, "retrieval", None)
            if raw is None and isinstance(settings, dict):
                raw = settings.get("retrieval")
            cfg = dict(raw or {})
        self.default_max_hops = int(cfg.get("graph_max_hops", 3))
        self.default_max_paths = int(cfg.get("graph_max_paths", 20))
        self.max_paths_per_hop = int(cfg.get("graph_max_paths_per_hop", 30))
        self._neighbor_cache: dict[str, list[dict[str, Any]]] = {}
        self._degree_cache: dict[str, int] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def search(
        self,
        anchors: list[str],
        relation_plan: list[str] | None = None,
        max_hops: int = 3,
        max_paths: int = 20,
        query_text: str = "",
        year_refs: Iterable[int] | None = None,
    ) -> GraphSearchResult:
        """Find the best paths that start at ``anchors``.

        Parameters
        ----------
        anchors:
            uids of the anchor nodes (output of the entity linker).
        relation_plan:
            Relation types the question is about.
        max_hops:
            Maximum number of edges of a path.
        max_paths:
            Maximum number of paths returned.
        query_text:
            The question (used for the path-similarity term and to detect
            the causal/temporal cues).
        year_refs:
            Years mentioned in the question.  When ``None`` they are
            extracted from ``query_text``.
        """
        anchor_uids = [str(a) for a in (anchors or []) if str(a)]
        result = GraphSearchResult(anchor_uids=anchor_uids)
        if not anchor_uids:
            return result

        hops = max(1, min(int(max_hops or self.default_max_hops), self.default_max_hops))
        limit = max(1, int(max_paths or self.default_max_paths))

        raw_paths = self._traverse(anchor_uids, hops)
        result.n_expanded = len(raw_paths)

        years = list(year_refs) if year_refs is not None else _extract_years(query_text)
        scored = [self._score_path(p, relation_plan, query_text, years) for p in raw_paths]
        scored.sort(key=lambda p: (-p.score, p.hops))
        result.paths = scored[:limit]
        return result

    def extract_subgraph(self, paths: Sequence[GraphPath]) -> dict[str, Any]:
        """Node/edge view of ``paths`` (for the UI)."""
        return extract_subgraph(paths)

    # ------------------------------------------------------------------
    # traversal
    # ------------------------------------------------------------------
    def _neighbors(self, uid: str) -> list[dict[str, Any]]:
        """The *outgoing* neighbours of ``uid`` (directed edges only).

        :meth:`GraphStore.neighbors` is undirected, which loses the true
        direction of a relation and produces bogus paths such as
        ``A -CAUSE_OF-> B`` where the edge really is ``B -CAUSE_OF-> A``.
        The docs/03 §5.1 Cypher is directed (``-[r*1..$maxHops]->``), so
        this helper keeps only the forward direction:

        * when the backend exposes a Neo4j runner, a directed Cypher query
          is used;
        * otherwise the undirected ``neighbors()`` result is *oriented* by
          comparing the uids of both endpoints (the in-memory store keeps
          the true ``src``/``dst`` of the relation).
        """
        cached = self._neighbor_cache.get(uid)
        if cached is not None:
            return cached
        rows = self._directed_rows(uid)
        self._neighbor_cache[uid] = rows
        return rows

    def _directed_rows(self, uid: str) -> list[dict[str, Any]]:
        runner = getattr(self.store, "_run", None)
        if callable(runner):
            try:
                records = runner(
                    """
                    MATCH (a:Entity {uid: $uid})-[r]->(b:Entity)
                    RETURN a{.*} AS src, type(r) AS rel, properties(r) AS rprops, b{.*} AS dst
                    LIMIT 500
                    """,
                    uid=uid,
                )
                return [
                    {
                        "src": dict(r["src"]),
                        "rel": r["rel"],
                        "rel_props": dict(r["rprops"]),
                        "dst": dict(r["dst"]),
                    }
                    for r in records
                ]
            except Exception as exc:  # noqa: BLE001
                logger.debug("directed neighbors(%r) failed: %s", uid, exc)
        try:
            raw_rows = self.store.neighbors(uid) or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("neighbors(%r) failed: %s", uid, exc)
            raw_rows = []
        out: list[dict[str, Any]] = []
        for row in raw_rows:
            src = row.get("src") or {}
            dst = row.get("dst") or {}
            src_uid = str(src.get("uid") or "")
            dst_uid = str(dst.get("uid") or "")
            if src_uid == uid:
                out.append(row)
            elif dst_uid == uid:
                # orient the edge away from the expanded node
                out.append(
                    {
                        "src": dst,
                        "rel": row.get("rel"),
                        "rel_props": row.get("rel_props") or {},
                        "dst": src,
                    }
                )
        return out

    def _traverse(self, anchors: list[str], max_hops: int) -> list[GraphPath]:
        """Enumerate the simple paths that start at an anchor node."""
        out: list[GraphPath] = []
        seen_signatures: set[tuple[str, ...]] = set()
        # hard cap so that a dense graph cannot explode the enumeration
        budget = max(1, self.max_paths_per_hop) * max(1, max_hops)

        for anchor in anchors:
            if len(out) >= budget:
                break
            anchor_props = self._entity(anchor)
            if anchor_props is None:
                logger.debug("Anchor %r không tồn tại trong KG", anchor)
                continue
            base_nodes = [self._node_dict(anchor_props)]
            self._expand(
                nodes=base_nodes,
                edges=[],
                visited={anchor},
                depth=0,
                max_hops=max_hops,
                out=out,
                seen_signatures=seen_signatures,
                budget=budget,
            )
        return out

    def _expand(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        visited: set[str],
        depth: int,
        max_hops: int,
        out: list[GraphPath],
        seen_signatures: set[tuple[str, ...]],
        budget: int = 1000,
    ) -> None:
        current_uid = str(nodes[-1].get("uid") or "")
        if depth >= max_hops or len(out) >= budget:
            return
        for row in self._neighbors(current_uid):
            src = row.get("src") or {}
            dst = row.get("dst") or {}
            src_uid = str(src.get("uid") or "")
            dst_uid = str(dst.get("uid") or "")
            # ``neighbors()`` is undirected -> orient the edge away from the
            # node we are expanding.  Only the *forward* direction of a
            # relation is followed, so that a path is always a genuine
            # directed reasoning chain (like the Cypher of docs/03 §5.1:
            # ``MATCH path = (a)-[r*1..$maxHops]->(b)``).
            if src_uid != current_uid or not dst_uid:
                continue
            other = dst
            other_uid = dst_uid
            if other_uid in visited:
                continue
            rel_type = str(row.get("rel") or "")
            if rel_type in PROVENANCE_RELATIONS:
                continue
            rel_props = row.get("rel_props") or {}
            edge = {
                "src": current_uid,
                "rel_type": rel_type,
                "dst": other_uid,
                "text": str(rel_props.get("text") or ""),
                "confidence": float(rel_props.get("confidence") or 0.0),
                "source_chunk_ids": list(rel_props.get("source_chunk_ids") or []),
                "year": rel_props.get("year"),
            }
            new_nodes = [*nodes, self._node_dict(other)]
            new_edges = [*edges, edge]
            signature = self._signature(new_nodes, new_edges)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                out.append(
                    GraphPath(nodes=new_nodes, edges=new_edges, hops=len(new_edges))
                )
            self._expand(
                nodes=new_nodes,
                edges=new_edges,
                visited={*visited, other_uid},
                depth=depth + 1,
                max_hops=max_hops,
                out=out,
                seen_signatures=seen_signatures,
                budget=budget,
            )

    @staticmethod
    def _signature(
        nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]
    ) -> tuple[str, ...]:
        """Canonical signature of a path.

        The traversal is undirected (``neighbors()`` returns both
        directions), so the same chain can be discovered twice with the
        edges reversed.  The signature is therefore built from the *set* of
        node uids and the *set* of (rel_type, {src, dst}) pairs, which makes
        the two views collapse into one.
        """
        node_ids = tuple(sorted(str(n.get("uid") or "") for n in nodes))
        edge_ids = tuple(
            sorted(
                f"{e.get('rel_type')}|{min(str(e.get('src') or ''), str(e.get('dst') or ''))}"
                f"|{max(str(e.get('src') or ''), str(e.get('dst') or ''))}"
                for e in edges
            )
        )
        return node_ids + edge_ids

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _score_path(
        self,
        path: GraphPath,
        relation_plan: list[str] | None,
        query_text: str,
        year_refs: list[int],
    ) -> GraphPath:
        affinity = self._relation_affinity(path, relation_plan, query_text)
        similarity = self._path_similarity(path, query_text)
        temporal = self._temporal_overlap(path, year_refs)
        penalty = self._degree_penalty(path)
        weights = PATH_SCORE_WEIGHTS
        path.score = (
            weights["relation_affinity"] * affinity
            + weights["path_similarity"] * similarity
            + weights["temporal_overlap"] * temporal
            - weights["degree_penalty"] * penalty
        )
        return path

    def _relation_affinity(
        self,
        path: GraphPath,
        relation_plan: list[str] | None,
        query_text: str,
    ) -> float:
        if not path.edges:
            return 0.0
        plan = {str(p).upper() for p in (relation_plan or []) if str(p).strip()}
        causal_cue = bool(_CAUSAL_CUE_RE.search(query_text or ""))
        temporal_cue = bool(_TEMPORAL_CUE_RE.search(query_text or ""))
        scores: list[float] = []
        for edge in path.edges:
            rel = str(edge.get("rel_type") or "").upper()
            if plan and rel in plan:
                scores.append(1.0)
            elif rel in CAUSAL_RELATIONS and causal_cue:
                scores.append(0.9)
            elif rel in TEMPORAL_RELATIONS and temporal_cue:
                scores.append(0.9)
            elif not plan and not causal_cue and not temporal_cue:
                # neutral: the question says nothing about relations
                scores.append(0.5)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores)

    def _path_similarity(self, path: GraphPath, query_text: str) -> float:
        embedder = self.embedder
        if embedder is None or not (query_text or "").strip():
            return 0.5
        try:
            q_vec = embedder.embed_one(query_text)
            p_vec = embedder.embed_one(path.to_text())
        except Exception as exc:  # noqa: BLE001
            logger.debug("embedding failed for path %s: %s", path.to_text(), exc)
            return 0.5
        # embeddings are normalised in practice -> map cosine to [0, 1]
        return max(0.0, min(1.0, _cosine(q_vec, p_vec)))

    def _temporal_overlap(self, path: GraphPath, year_refs: list[int]) -> float:
        if not year_refs:
            return 0.5
        path_years = self._path_years(path)
        if not path_years:
            return 0.0
        if path_years & set(year_refs):
            return 1.0
        # partial credit when the path is close to a mentioned year
        best = min(abs(y - r) for y in path_years for r in year_refs)
        if best <= 20:
            return 0.5
        return 0.0

    @staticmethod
    def _path_years(path: GraphPath) -> set[int]:
        years: set[int] = set()
        for edge in path.edges:
            year = edge.get("year")
            if year is None:
                continue
            try:
                years.add(int(year))
            except (TypeError, ValueError):
                continue
        for node in path.nodes:
            for key in ("year_min", "year_max"):
                value = node.get(key)
                if value is None:
                    continue
                try:
                    years.add(int(value))
                except (TypeError, ValueError):
                    continue
        return years

    def _degree_penalty(self, path: GraphPath) -> float:
        degrees = [self._degree(str(n.get("uid") or "")) for n in path.nodes]
        degrees = [d for d in degrees if d > 0]
        if not degrees:
            return 0.0
        max_degree = max(self._degree(str(n.get("uid") or "")) for n in path.nodes) or 1
        denom = math.log(1 + max_degree) or 1.0
        penalty = sum(math.log(1 + d) / denom for d in degrees) / len(degrees)
        return max(0.0, min(1.0, penalty))

    def _degree(self, uid: str) -> int:
        """Undirected degree of a node (used by the hub penalty)."""
        if not uid:
            return 0
        cached = self._degree_cache.get(uid)
        if cached is not None:
            return cached
        try:
            rows = self.store.neighbors(uid) or []
        except Exception as exc:  # noqa: BLE001
            logger.debug("neighbors(%r) failed: %s", uid, exc)
            rows = []
        degree = len(rows)
        self._degree_cache[uid] = degree
        return degree

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _entity(self, uid: str) -> Mapping[str, Any] | None:
        getter = getattr(self.store, "get_entity", None)
        if not callable(getter):
            return None
        try:
            return getter(uid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_entity(%r) failed: %s", uid, exc)
            return None

    @staticmethod
    def _node_dict(props: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "uid": str(props.get("uid") or ""),
            "name": str(props.get("name") or props.get("uid") or ""),
            "entity_type": str(props.get("entity_type") or "Entity"),
            "wikidata_id": str(props.get("wikidata_id") or ""),
            "year_min": props.get("year_min"),
            "year_max": props.get("year_max"),
        }


# --------------------------------------------------------------------------
# Subgraph extraction (shared with the UI)
# --------------------------------------------------------------------------


def extract_subgraph(paths: Sequence[GraphPath]) -> dict[str, Any]:
    """Build a ``{"nodes": [...], "edges": [...]}`` view of ``paths``.

    The structure is directly usable by ``pyvis``/``plotly`` in the
    Streamlit "Đồ thị" tab.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    edge_ids: set[tuple[str, str, str]] = set()
    for path in paths or []:
        for node in path.nodes:
            uid = str(node.get("uid") or "")
            if not uid:
                continue
            nodes.setdefault(uid, dict(node))
        for edge in path.edges:
            src = str(edge.get("src") or "")
            dst = str(edge.get("dst") or "")
            rel = str(edge.get("rel_type") or "")
            key = (src, rel, dst)
            if key in edge_ids:
                continue
            edge_ids.add(key)
            edges.append(dict(edge))
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }
