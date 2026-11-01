"""Knowledge-graph construction from chunks.

The builder turns the events extracted from a chunk into a small bipartite
graph::

    (:Event)-[:PARTICIPATED_IN]->(:Person)
    (:Event)-[:LOCATED_IN]->(:Place)
    (:Event)-[:OCCURRED_IN]->(:Period)
    (:Event)-[:SUPPORTED_BY]->(:Chunk)      # provenance

Design constraints (docs/02 §3, docs/03):

* every node has a **stable uid** -- running the builder twice must not
  create duplicates (the store merges on ``uid``);
* every relation carries ``source_chunk_ids`` and the evidence text;
* every node that came from a chunk is linked to its chunk with a
  ``SUPPORTED_BY`` edge so provenance can be traversed in Neo4j.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..data.normalize import name_key
from ..data.records import Chunk, Entity, Relation, stable_uid
from .alias import AliasResolver
from .ontology import ONTOLOGY as _DEFAULT_ONTOLOGY
from .ontology import Ontology, normalize_role, relation_for_role
from .seed import anchor_relation, seed_entities_from_chunks

logger = logging.getLogger(__name__)

__all__ = ["BuildStats", "KGBuilder"]

#: Argument type -> entity type used for the argument node.
ARG_TYPE_TO_ENTITY_TYPE: dict[str, str] = {
    "Person": "Person",
    "Place": "Place",
    "Organization": "Organization",
    "Period": "Period",
    "Event": "Event",
    "Artifact": "Artifact",
    "Entity": "Entity",
}

#: Argument types that become a *participant* of the event.
PARTICIPANT_ROLES = ("AGENT", "BENEFICIARY", "EXPERIENCER", "FORCE", "INSTRUMENT")

#: Trigger phrases that carry no information on their own.  An event named
#: only after such a trigger ("sinh", "mất", "đánh bại") is useless for the
#: entity linker -- hundreds of them share the same name.  When the trigger
#: is one of these, the event is renamed after its *agent* instead.
GENERIC_TRIGGERS: frozenset[str] = frozenset(
    {
        "sinh",
        "mất",
        "chết",
        "cưới",
        "gả",
        "đánh",
        "đánh bại",
        "đánh tan",
        "đánh đuổi",
        "đánh dẹp",
        "đánh phá",
        "thắng trận",
        "thua trận",
        "xây dựng",
        "lập ra",
        "thành lập",
        "giải thể",
        "chia",
        "hợp nhất",
        "ký",
        "kết",
        "bắt đầu",
        "kết thúc",
        "diễn ra",
        "tham gia",
        "chỉ huy",
        "ra lệnh",
        "tập kích",
        "giao",
        "bị",
        "được",
        "mang",
        "giữ",
        "giết",
        "cầu",
        "dựng",
        "mở",
        "đóng",
        "tổ chức",
        "di chuyển",
        "trở lại",
        "trở về",
        "rút",
        "thăm",
        "dặn",
        "phong",
        "phong tặng",
        "tặng",
        "ban",
        "săn",
        "trồng",
        "lập",
        "thành",
        "giải phóng",
        "thống nhất",
        "chia rẽ",
    }
)


def event_display_name(
    trigger: str,
    description: str,
    agent_name: str,
    year: int | None,
) -> str:
    """A readable, *linkable* name for an event node.

    The raw LLM trigger is often a bare verb ("sinh", "mất"), which makes
    hundreds of events share the same name and wrecks the entity linking.
    The rule is:

    * a descriptive trigger (``"lập ra nhà Lý"``) is kept as is;
    * a generic trigger is combined with the agent -> ``"Lý Công Uẩn sinh"``;
    * when there is no agent, the description (truncated) is used;
    * the year is appended when it is known, so that two events of the same
      type but different years stay distinguishable.
    """
    trigger = (trigger or "").strip()
    description = (description or "").strip()
    agent = (agent_name or "").strip()

    generic = trigger.casefold() in GENERIC_TRIGGERS or len(trigger) <= 2
    if generic:
        if agent:
            name = f"{agent} {trigger}" if trigger else agent
        elif description:
            name = description
        else:
            name = trigger or "sự kiện"
    else:
        name = trigger

    if year and str(year) not in name:
        name = f"{name} ({year})"
    return name[:180].strip()


@dataclass
class BuildStats:
    """Counters returned by :meth:`KGBuilder.build_from_chunks`."""

    n_entities: int = 0
    n_relations: int = 0
    n_skipped: int = 0
    n_invalid: int = 0
    elapsed: float = 0.0
    n_chunks: int = 0
    n_events: int = 0
    n_linked: int = 0
    relation_types: dict[str, int] = field(default_factory=dict)
    entity_types: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def export(self) -> dict[str, Any]:
        """JSON-serialisable summary (used by the CLI)."""
        return {
            "n_entities": self.n_entities,
            "n_relations": self.n_relations,
            "n_skipped": self.n_skipped,
            "n_invalid": self.n_invalid,
            "n_chunks": self.n_chunks,
            "n_events": self.n_events,
            "n_linked": self.n_linked,
            "elapsed": round(self.elapsed, 3),
            "relation_types": dict(self.relation_types),
            "entity_types": dict(self.entity_types),
        }


class KGBuilder:
    """Build the knowledge graph out of chunks.

    Parameters
    ----------
    store:
        The :class:`~hgrag.kg.store.GraphStore` where nodes/edges are written.
    ontology:
        The VHE ontology (event types + role -> relation mapping).
    extractor:
        The :class:`~hgrag.kg.extractor.EventExtractor`.
    resolver:
        The :class:`~hgrag.kg.alias.AliasResolver` used to attach Wikidata
        QIDs to the argument nodes.
    settings:
        The project settings (optional).
    """

    def __init__(
        self,
        store: Any,
        ontology: Ontology | None = None,
        extractor: Any = None,
        resolver: AliasResolver | None = None,
        settings: Any = None,
        llm_client: Any = None,
        wikidata: Any = None,
    ) -> None:
        self.store = store
        self.ontology = ontology or _DEFAULT_ONTOLOGY
        self.extractor = extractor
        self.resolver = resolver
        self.settings = settings
        self.llm_client = llm_client
        self.wikidata = wikidata
        self._entity_cache: dict[str, Entity] = {}

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def build_from_chunks(
        self,
        chunks: Iterable[Chunk],
        max_chunks: int | None = None,
        progress: bool = True,
    ) -> BuildStats:
        """Extract events from ``chunks`` and write the graph.

        The operation is **idempotent**: the uids are derived from stable
        inputs, so building the same chunks twice yields the same number of
        nodes and edges.
        """
        started = time.time()
        stats = BuildStats()
        entities: dict[str, Entity] = {}
        relations: dict[str, Relation] = {}
        chunk_nodes: dict[str, Entity] = {}

        chunk_list = list(chunks)
        if max_chunks is not None:
            chunk_list = chunk_list[: max(0, int(max_chunks))]

        # -- anchor nodes -------------------------------------------------
        # The canonical subject of every document (with its Wikidata QID and
        # its Wikipedia redirect aliases) is seeded *before* the extraction
        # so that the resolver can link the extracted mentions to it.
        anchors = seed_entities_from_chunks(chunk_list, wikidata=self.wikidata)
        anchor_by_uid = {a.uid: a for a in anchors}
        for anchor in anchors:
            entities.setdefault(anchor.uid, anchor)
        if self.resolver is not None and anchors:
            try:
                # Only the anchors that carry a QID are indexed: an anchor
                # without one would shadow the real entity that has the same
                # name (e.g. the Wikipedia subject without its QID would hide
                # the Wikidata-identified person).
                self.resolver.add_to_alias_index([a for a in anchors if a.wikidata_id])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Không dựng được alias index từ anchor: %s", exc)

        for index, chunk in enumerate(chunk_list, start=1):
            stats.n_chunks += 1
            events = self._extract(chunk)
            if not events:
                stats.n_skipped += 1
                if progress:
                    logger.info("[%d/%d] %s -- không có event", index, len(chunk_list), chunk.chunk_id)
                continue
            stats.n_events += len(events)
            chunk_entity = self._chunk_entity(chunk)
            chunk_nodes[chunk.chunk_id] = chunk_entity
            for event in events:
                self._add_event(
                    event=event,
                    chunk=chunk,
                    entities=entities,
                    relations=relations,
                    stats=stats,
                )
            if progress:
                logger.info(
                    "[%d/%d] %s -- %d event(s)",
                    index,
                    len(chunk_list),
                    chunk.chunk_id,
                    len(events),
                )
            # the document subject is linked to its chunk
            anchor_uid = _anchor_uid_for_chunk(chunk, anchors)
            if anchor_uid:
                rel = anchor_relation(chunk, anchor_uid)
                relations[rel["uid"]] = Relation(**rel)

        # write the chunk nodes (provenance anchors)
        if chunk_nodes:
            self.store.upsert_entities(list(chunk_nodes.values()))

        if entities:
            self.store.upsert_entities(list(entities.values()))
        if relations:
            self.store.upsert_relations(list(relations.values()))

        stats.n_entities = len(entities) + len(chunk_nodes)
        stats.n_relations = len(relations)
        stats.elapsed = time.time() - started
        stats.relation_types = _count(r.rel_type for r in relations.values())
        stats.entity_types = _count(e.entity_type for e in entities.values())
        return stats

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _extract(self, chunk: Chunk) -> list[Any]:
        if self.extractor is None:
            return []
        try:
            return list(self.extractor.extract_from_chunk(chunk) or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extraction failed for %s: %s", chunk.chunk_id, exc)
            return []

    def _chunk_entity(self, chunk: Chunk) -> Entity:
        return Entity(
            uid=chunk.chunk_id,
            name=chunk.title or chunk.doc_id or chunk.chunk_id,
            entity_type="Chunk",
            description=(chunk.text or "")[:400],
            source=chunk.source,
            source_chunk_ids=[chunk.chunk_id],
            extra={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "url": chunk.url,
                "year_min": chunk.year_min,
                "year_max": chunk.year_max,
                "wikidata_id": chunk.wikidata_id,
            },
        )

    def _add_event(
        self,
        event: Any,
        chunk: Chunk,
        entities: dict[str, Entity],
        relations: dict[str, Relation],
        stats: BuildStats,
    ) -> None:
        event_type = event.event_type
        trigger = event.trigger or event_type
        agent_name = _agent_name(event.arguments)
        display_name = event_display_name(
            trigger, getattr(event, "description", "") or "", agent_name, event.year
        )
        # The uid must stay stable across rebuilds *and* must not collide
        # between two different events that share the same trigger, hence the
        # agent and the year are part of the key.
        event_uid = stable_uid(
            event_type, trigger, agent_name, str(event.year or ""), prefix="evt_"
        )
        event_entity = Entity(
            uid=event_uid,
            name=display_name,
            entity_type="Event",
            description=event.description or "",
            year_min=event.year,
            year_max=event.year,
            period=chunk.period,
            source="llm",
            confidence=1.0,
            source_chunk_ids=[chunk.chunk_id],
            extra={
                "event_type": event_type,
                "trigger": trigger,
                "agent": agent_name,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
            },
        )
        if event_uid in entities:
            stats.n_skipped += 1
            return
        entities[event_uid] = event_entity

        for arg in event.arguments or []:
            role = normalize_role(str(arg.get("role") or ""))
            name = str(arg.get("name") or "").strip()
            if not role or not name:
                stats.n_skipped += 1
                continue
            arg_type = str(arg.get("type") or "Entity")
            entity_type = ARG_TYPE_TO_ENTITY_TYPE.get(arg_type, "Entity")
            arg_entity, linked = self._argument_entity(name, entity_type, chunk, stats)
            existing = entities.get(arg_entity.uid)
            if existing is not None:
                # The node already exists (e.g. it is an anchor node with a
                # canonical name and its Wikidata aliases): keep the
                # canonical data and only merge the provenance.
                for cid in arg_entity.source_chunk_ids:
                    if cid not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(cid)
                if not existing.wikidata_id and arg_entity.wikidata_id:
                    existing.wikidata_id = arg_entity.wikidata_id
                if not existing.description and arg_entity.description:
                    existing.description = arg_entity.description
                arg_entity = existing
            else:
                entities[arg_entity.uid] = arg_entity
            if linked:
                stats.n_linked += 1

            rel_type = relation_for_role(role, self.ontology.relation_types)
            rel_uid = stable_uid(event_uid, rel_type, arg_entity.uid, prefix="rel_")
            relation = Relation(
                uid=rel_uid,
                src=event_uid,
                dst=arg_entity.uid,
                rel_type=rel_type,
                text=f"{event_type}: {trigger} --{role.lower()}--> {name}",
                year=event.year,
                source="llm",
                confidence=arg_entity.confidence,
                source_chunk_ids=[chunk.chunk_id],
                extra={
                    "evidence_text": self._evidence(event, chunk, name, role),
                    "role": role,
                    "event_type": event_type,
                    "argument_type": arg_type,
                },
            )
            relations[rel_uid] = relation

            # provenance: entity -[:SUPPORTED_BY]-> chunk
            sup_uid = stable_uid(arg_entity.uid, "SUPPORTED_BY", chunk.chunk_id, prefix="rel_")
            relations[sup_uid] = Relation(
                uid=sup_uid,
                src=arg_entity.uid,
                dst=chunk.chunk_id,
                rel_type="SUPPORTED_BY",
                text=f"{arg_entity.name} được hỗ trợ bởi chunk {chunk.chunk_id}",
                year=event.year,
                source=chunk.source,
                confidence=1.0,
                source_chunk_ids=[chunk.chunk_id],
                extra={"evidence_text": self._evidence(event, chunk, name, role), "role": role},
            )

        # provenance: event -[:SUPPORTED_BY]-> chunk
        sup_uid = stable_uid(event_uid, "SUPPORTED_BY", chunk.chunk_id, prefix="rel_")
        relations[sup_uid] = Relation(
            uid=sup_uid,
            src=event_uid,
            dst=chunk.chunk_id,
            rel_type="SUPPORTED_BY",
            text=f"{trigger} được hỗ trợ bởi chunk {chunk.chunk_id}",
            year=event.year,
            source=chunk.source,
            confidence=1.0,
            source_chunk_ids=[chunk.chunk_id],
            extra={
                "evidence_text": self._evidence(event, chunk, trigger, "event"),
                "event_type": event_type,
            },
        )

    def _evidence(self, event: Any, chunk: Chunk, name: str, role: str) -> str:
        description = getattr(event, "description", "") or ""
        trigger = getattr(event, "trigger", "") or ""
        return (
            f"[{chunk.title}] {trigger} ({role}: {name}) "
            f"{description} :: {chunk.text[:300]}"
        ).strip()

    def _argument_entity(
        self, name: str, entity_type: str, chunk: Chunk, stats: BuildStats
    ) -> tuple[Entity, bool]:
        """Create (or reuse) the node of an event argument."""
        linked = False
        wikidata_id = ""
        aliases: list[str] = []
        p31: list[str] = []
        p27: list[str] = []
        confidence = 0.6
        year_min: int | None = None
        year_max: int | None = None

        if self.resolver is not None:
            hints = {"entity_type": entity_type}
            try:
                candidate = self.resolver.resolve_one(name, hints)
            except Exception as exc:  # noqa: BLE001
                logger.debug("resolve failed for %r: %s", name, exc)
                candidate = None
            if candidate is not None:
                wikidata_id = candidate.wikidata_id
                aliases = list(candidate.aliases)
                p31 = list(candidate.p31)
                p27 = list(candidate.p27)
                confidence = candidate.score
                year_min = candidate.year_min
                year_max = candidate.year_max
                linked = bool(wikidata_id)
                # The canonical name of the resolved entity is preferred over
                # the raw mention ("Trần Quốc Tuấn" -> "Trần Hưng Đạo").
                if candidate.name and candidate.tier <= 1:
                    name = candidate.name
                    if name not in aliases:
                        aliases.insert(0, name)

        if wikidata_id:
            uid = f"pe:{wikidata_id.upper()}"
        else:
            uid = stable_uid(name_key(name) or name, entity_type, prefix="ent_")

        existing = self._entity_cache.get(uid)
        if existing is not None:
            if chunk.chunk_id not in existing.source_chunk_ids:
                existing.source_chunk_ids.append(chunk.chunk_id)
            return existing, linked

        entity = Entity(
            uid=uid,
            name=name,
            entity_type=entity_type,
            wikidata_id=wikidata_id,
            aliases=aliases,
            name_key=name_key(name),
            p31=p31,
            p27=p27,
            year_min=year_min,
            year_max=year_max,
            source="llm",
            confidence=confidence,
            source_chunk_ids=[chunk.chunk_id],
            extra={
                "mention": name,
                "resolution": "wikidata" if linked else "local",
            },
        )
        self._entity_cache[uid] = entity
        return entity, linked

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def export_stats(self) -> dict[str, Any]:
        """Return the statistics of the underlying store plus the ontology."""
        payload: dict[str, Any] = {
            "ontology": {
                "source": self.ontology.source,
                "n_event_types": len(self.ontology.event_types),
                "n_roles": len(self.ontology.roles),
            }
        }
        try:
            payload["store"] = self.store.stats()
        except Exception as exc:  # noqa: BLE001
            logger.warning("store.stats() failed: %s", exc)
            payload["store"] = {}
        return payload


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _anchor_uid_for_chunk(chunk: Chunk, anchors: Sequence[Entity]) -> str:
    """The uid of the anchor entity that corresponds to ``chunk``."""
    for anchor in anchors:
        if anchor.extra.get("doc_id") == chunk.doc_id:
            return anchor.uid
    return ""


def _agent_name(arguments: Sequence[Mapping[str, Any]] | None) -> str:
    """The name of the first ``AGENT`` argument of an event (empty when none)."""
    for arg in arguments or []:
        if not isinstance(arg, Mapping):
            continue
        if normalize_role(str(arg.get("role") or "")) == "AGENT":
            name = str(arg.get("name") or "").strip()
            if name:
                return name
    return ""


def entities_from_mappings(rows: Sequence[Mapping[str, Any]]) -> list[Entity]:
    """Helper: build :class:`Entity` objects from plain dicts."""
    out: list[Entity] = []
    for row in rows:
        try:
            out.append(Entity.from_dict(dict(row)))
        except (TypeError, ValueError):
            continue
    return out
