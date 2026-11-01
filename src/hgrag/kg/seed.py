"""Seed entities: the canonical anchor nodes of the knowledge graph.

The LLM extraction produces *argument mentions* ("Trần Quốc Tuấn", "nhà Trần",
"thành Đại Việt") but it never produces the canonical entity itself.  Without
a canonical node the graph is a bag of anonymous names, the Wikidata QID of a
person is never attached and the entity linking has nothing to link *to*.

This module derives the canonical entities from the **document metadata**
that the ingest step already collected:

* ``chunk.wikidata_id`` -- the QID of the Wikipedia article;
* ``chunk.extra["redirects"]`` -- the Wikipedia redirect titles, i.e. the
  alternative names of the subject ("Trần Quốc Tuấn", "Hưng Đạo Vương");
* ``chunk.title`` -- the canonical Vietnamese label.

The result is a small set of *anchor* entities that:

1. carry the Wikidata QID, the aliases and the domain properties
   (``P31``/``P27``) so that the alias resolver can work at tier 0/1;
2. give the KG a stable "subject" node that every chunk of the document
   links to (``ABOUT`` edge), which is what makes the multi-hop retrieval
   meaningful.

The type of the anchor is resolved from Wikidata ``P31`` when the network is
available; otherwise it falls back to a title-based heuristic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from ..data.records import Chunk, Entity
from .alias import P31_TYPE_MAP

logger = logging.getLogger(__name__)

__all__ = [
    "seed_entities_from_chunks",
    "guess_entity_type",
    "anchor_relation",
]

#: Titles that clearly denote a person (used when Wikidata is unreachable).
_PERSON_HINTS = (
    "vương",
    "vua",
    "hoàng đế",
    "hoàng tử",
    "công chúa",
    "thái tử",
    "tướng",
    "quốc công",
    "võ nguyên",
    "hồ chí minh",
)

_ORG_HINTS = ("đảng", "quốc hội", "chính phủ", "nhà ")
_PLACE_HINTS = ("thành phố", "tỉnh", "huyện", "làng", "thành")


def guess_entity_type(title: str, description: str = "") -> str:
    """Guess the coarse entity type of a document subject."""
    text = f"{title} {description}".lower()
    for hint in _ORG_HINTS:
        if hint in text:
            return "Organization"
    for hint in _PLACE_HINTS:
        if hint in text:
            return "Place"
    for hint in _PERSON_HINTS:
        if hint in text:
            return "Person"
    return "Person"


def _type_from_p31(p31: Sequence[str]) -> str:
    for pid in p31 or []:
        if pid in P31_TYPE_MAP:
            return P31_TYPE_MAP[pid]
    return ""


def seed_entities_from_chunks(
    chunks: Iterable[Chunk],
    wikidata: Any = None,
) -> list[Entity]:
    """Build the canonical anchor entities of the corpus.

    Parameters
    ----------
    chunks:
        The corpus chunks.  Only the *first* chunk of every document is used
        (they all carry the same document metadata).
    wikidata:
        Optional :class:`~hgrag.data.wikidata.WikidataClient` used to fetch
        ``P31``/``P27``/``aliases`` of the QIDs.  Without it the entities
        are still created, only without the domain properties.
    """
    by_doc: dict[str, Chunk] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, chunk)

    qids: list[str] = []
    for chunk in by_doc.values():
        qid = str(chunk.wikidata_id or "").strip().upper()
        if qid:
            qids.append(qid)

    props_by_qid: dict[str, dict[str, Any]] = {}
    if wikidata is not None and qids:
        try:
            entities = wikidata.get_entities(sorted(set(qids)))
            for qid, payload in entities.items():
                props_by_qid[qid.upper()] = payload
        except Exception as exc:  # noqa: BLE001 - seeding must never break the build
            logger.warning("Không lấy được props Wikidata: %s", exc)

    out: list[Entity] = []
    for chunk in by_doc.values():
        title = (chunk.title or "").strip()
        if not title:
            continue
        qid = str(chunk.wikidata_id or "").strip().upper()
        # A document without a QID does not get an anchor node: such a node
        # would carry the title of the article but no identity, and would
        # therefore *shadow* the real entity that has the same name.
        if not qid:
            continue
        redirects = [str(r) for r in (chunk.extra.get("redirects") or []) if r]

        aliases: list[str] = []
        alias_sources: dict[str, str] = {}
        p31: list[str] = []
        p27: list[str] = []
        description = ""
        if qid and qid in props_by_qid:
            payload = props_by_qid[qid]
            labels = payload.get("labels") or {}
            alias_group = payload.get("aliases") or {}
            for lang in ("vi", "en"):
                lab = labels.get(lang) or {}
                if lab.get("value"):
                    description = description or str(lab["value"])
                for item in alias_group.get(lang) or []:
                    val = str(item.get("value") or "")
                    if val and val not in aliases:
                        aliases.append(val)
                        alias_sources[val] = "wikidata"
            claims = payload.get("claims") or {}
            for pid, target in (("P31", p31), ("P27", p27)):
                for claim in claims.get(pid) or []:
                    snak = claim.get("mainsnak") or {}
                    if snak.get("snaktype") != "value":
                        continue
                    value = (snak.get("datavalidue") or {}).get("value")
                    if isinstance(value, dict) and value.get("id"):
                        target.append(str(value["id"]))

        # Wikipedia redirect titles are aliases of the *subject*.
        for red in redirects:
            if red != title and red not in aliases:
                aliases.append(red)
                alias_sources[red] = "wiki_redirect"

        entity_type = _type_from_p31(p31) or guess_entity_type(title, description)
        uid = f"pe:{qid}"

        out.append(
            Entity(
                uid=uid,
                name=title,
                entity_type=entity_type,
                wikidata_id=qid,
                aliases=aliases,
                name_key="",
                description=description,
                period=chunk.period,
                year_min=chunk.year_min,
                year_max=chunk.year_max,
                source="wikipedia",
                confidence=1.0,
                source_chunk_ids=[chunk.chunk_id],
                extra={
                    "doc_id": chunk.doc_id,
                    "anchor": True,
                    "alias_sources": alias_sources,
                    "url": chunk.url,
                },
            )
        )
    logger.info("Seed entities: %d anchor node(s)", len(out))
    return out


def anchor_relation(chunk: Chunk, anchor_uid: str) -> dict[str, Any]:
    """The ``ABOUT`` relation between a chunk and its document anchor."""
    return {
        "uid": f"rel_about_{chunk.chunk_id}",
        "src": anchor_uid,
        "dst": chunk.chunk_id,
        "rel_type": "ABOUT",
        "text": f"{chunk.title} là chủ đề của chunk {chunk.chunk_id}",
        "year": chunk.year_min,
        "source": chunk.source,
        "confidence": 1.0,
        "source_chunk_ids": [chunk.chunk_id],
        "extra": {
            "anchor": True,
            "evidence_text": f"[{chunk.title}] {chunk.text[:300]}".strip(),
        },
    }
