"""Import the VHE *gold* annotations as a dedicated knowledge-graph subgraph.

Why
---
ADR-01 decided that VHE is the **ontology + gold** of the project, and that
the Wikipedia corpus is the main retrieval corpus.  But the VHE gold
annotations are also what the *evaluation set* is built from
(docs/04 §2.1, source A): 88 of the 150 evaluation questions are template
questions over the VHE gold events.

Those questions can only be answered if the corresponding facts live in the
knowledge graph.  Without this importer the graph only contains the Wikipedia
events, so 73 of the 88 VHE questions ask about entities (``Đế Minh``,
``Hùng Vương``, ``Sơn Tinh`` ...) that the system has never seen -- the
evaluation would measure the model's luck instead of the retrieval
architecture.

Design
------
The VHE gold is imported into **its own subgraph** and never mixed with the
LLM-extracted graph:

* every VHE event is a node ``(:Event) {source: "vhe"}`` with a stable uid
  derived from the instance id + the event index;
* every argument becomes an entity node (``source: "vhe"``), reused across
  events through the same stable uid;
* the VHE *sentence* becomes a provenance chunk (``source: "vhe"``) and every
  edge points at it with ``SUPPORTED_BY``, so the whole subgraph remains
  traceable to its source text -- the very same rule the Wikipedia subgraph
  follows;
* the relation types come from the VHE ontology (the role -> relation map),
  so the two subgraphs are comparable.

The VHE text is classical Vietnamese, hence the retrieval layer applies a
*source filter* (``vhe`` chunks are only retrieved for questions that ask
about a VHE entity) -- see :mod:`hgrag.retrieval.entity_linker`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from ..data.normalize import name_key
from ..data.records import Chunk, Entity, Relation, stable_uid
from .labels import clean_mention, event_label, event_verb, pick_mention
from .ontology import (
    Ontology,
    edge_is_participant_to_event,
    normalize_role,
    relation_for_role,
)

logger = logging.getLogger(__name__)

__all__ = [
    "VHE_SOURCE",
    "import_vhe_gold",
    "vhe_event_uid",
    "vhe_entity_uid",
]

#: The ``source`` value of every node/edge/chunk of the VHE subgraph.
VHE_SOURCE = "vhe"

#: Relation types whose edge points **from the participant to the event**.
#: The traversal of the evaluation set starts at a named entity (the
#: question anchor) and has to reach the event first, then the context
#: arguments of the event.  See :func:`hgrag.kg.ontology.edge_is_participant_to_event`.
PARTICIPANT_TO_EVENT_RELATIONS: frozenset[str] = frozenset(
    {
        "PARTICIPATED_IN",
        "AGENT_OF",
        "INVOLVES",
        "CAUSE_OF",
        "PRECONDITION_OF",
    }
)

#: Relation types whose edge points **from the event to the argument**.
EVENT_TO_ARGUMENT_RELATIONS: frozenset[str] = frozenset(
    {
        "ABOUT",
        "LOCATED_IN",
        "OCCURRED_IN",
        "OCCURRED_AT",
        "HAPPENED_AT",
    }
)

_YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|[1-9][0-9]{2})(?!\d)")


def vhe_event_uid(instance_id: str, index: int, event_type: str, trigger: str) -> str:
    """Stable uid of a VHE gold event."""
    return stable_uid(
        str(instance_id), str(index), str(event_type), str(trigger), prefix="vhe_evt_"
    )


def vhe_entity_uid(mention: str, entity_type: str = "") -> str:
    """Stable uid of a VHE gold entity.

    The uid is derived from the **normalised mention only** -- the entity
    type is deliberately *not* part of the key.

    Why?  The VHE annotations mention the same real-world entity with
    different roles in different events: ``Sơn Tinh`` is the *agent* of
    ``Sơn Tinh gặp gỡ`` (type ``Person``) and the *theme* of ``Mị Nương
    cưới Sơn Tinh`` (type ``Entity``).  With the type in the key the two
    mentions became **two different nodes** (1 056 names, 2 819 nodes and
    12 914 relations were affected), so the graph traversal that starts at
    the anchor of a template question could not reach the gold event: the
    question "Ai là người cưới Sơn Tinh?" linked the *Person* node while
    the answer sat behind the *Entity* node, and the answerer refused
    (measured: 42 % of the VHE template questions were refused).

    With a type-independent uid the two mentions collapse into one node and
    the event hub (``theme <-[ABOUT]- event <-[PARTICIPATED_IN]- agent``)
    connects the whole chain.  The coarse type is still stored on the node
    (it is merged afterwards), it just does not split the identity.

    >>> vhe_entity_uid("Sơn Tinh", "Person") == vhe_entity_uid("Sơn Tinh", "Entity")
    True
    """
    return stable_uid(name_key(mention) or mention, prefix="vhe_ent_")


def _arg_type(role: str, mention: str) -> str:
    """Coarse entity type of a VHE argument."""
    if role in ("AGENT", "BENEFICIARY", "EXPERIENCER", "FORCE"):
        return "Person" if mention and mention[0].isupper() else "Entity"
    if role == "LOCATION":
        return "Place"
    if role == "TEMPORAL":
        return "Period"
    return "Entity"


def _years(text: str) -> list[int]:
    out: list[int] = []
    for raw in _YEAR_RE.findall(text or ""):
        year = int(raw)
        if 905 <= year <= 1975 and year not in out:
            out.append(year)
    return out


def import_vhe_gold(
    instances: Iterable[dict[str, Any]],
    ontology: Ontology,
    max_instances: int | None = None,
) -> tuple[list[Entity], list[Relation], list[Chunk]]:
    """Convert the VHE gold annotations into entities, relations and chunks.

    Returns ``(entities, relations, chunks)``.  The chunks carry the VHE
    sentence so that every edge of the subgraph has its provenance.
    """
    entities: dict[str, Entity] = {}
    relations: dict[str, Relation] = {}
    chunks: dict[str, Chunk] = {}
    chunk_ids_by_instance: dict[str, str] = {}

    rows = list(instances)
    if max_instances is not None:
        rows = rows[: max(0, int(max_instances))]

    for instance in rows:
        instance_id = str(instance.get("id") or "")
        text = str(instance.get("text") or "")
        events = instance.get("events") or []
        if not instance_id or not text or not events:
            continue

        chunk_id = f"vhe_chunk_{instance_id}"
        chunk_ids_by_instance[instance_id] = chunk_id
        years = _years(text)
        chunks[chunk_id] = Chunk(
            chunk_id=chunk_id,
            doc_id=f"vhe:{instance_id}",
            title=f"VHE {instance_id}",
            text=text,
            source=VHE_SOURCE,
            period="",
            year_min=years[0] if years else None,
            year_max=years[-1] if years else None,
            extra={"vhe_id": instance_id, "n_events": len(events)},
        )

        for index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "")
            trigger = str(event.get("trigger_word") or "").strip()
            if not event_type:
                continue
            arguments = event.get("arguments") or []
            by_role: dict[str, list[str]] = {}
            for arg in arguments:
                if not isinstance(arg, dict):
                    continue
                role = normalize_role(str(arg.get("role") or ""))
                mention = clean_mention(str(arg.get("mention") or ""))
                if not role or not mention:
                    continue
                by_role.setdefault(role, []).append(mention)

            # The agent and the theme are picked with the *same* rule as the
            # evaluation dataset builder (:func:`hgrag.kg.labels.pick_mention`)
            # so that the name of the event node and the gold path of the
            # question refer to the very same event.
            agent = pick_mention(by_role.get("AGENT", []))
            theme = pick_mention(by_role.get("THEME", []))

            event_uid = vhe_event_uid(instance_id, index, event_type, trigger)
            if event_uid in entities:
                continue
            entities[event_uid] = Entity(
                uid=event_uid,
                name=event_label(agent, event_verb(event_type, trigger), theme),
                entity_type="Event",
                description=text[:400],
                period="",
                year_min=years[0] if years else None,
                year_max=years[-1] if years else None,
                source=VHE_SOURCE,
                confidence=1.0,
                source_chunk_ids=[chunk_id],
                extra={
                    "event_type": event_type,
                    "trigger": trigger,
                    "verb": event_verb(event_type, trigger),
                    "agent": agent,
                    "theme": theme,
                    "vhe_id": instance_id,
                    "vhe_index": index,
                },
            )

            for role, mentions in by_role.items():
                for mention in mentions:
                    entity_type = _arg_type(role, mention)
                    ent_uid = vhe_entity_uid(mention, entity_type)
                    existing = entities.get(ent_uid)
                    if existing is None:
                        existing = Entity(
                            uid=ent_uid,
                            name=mention,
                            entity_type=entity_type,
                            name_key=name_key(mention),
                            description="",
                            source=VHE_SOURCE,
                            confidence=0.7,
                            source_chunk_ids=[chunk_id],
                            extra={
                                "mention": mention,
                                "roles": [role],
                                "entity_types": [entity_type],
                                "resolution": "vhe_gold",
                            },
                        )
                        entities[ent_uid] = existing
                    else:
                        if chunk_id not in existing.source_chunk_ids:
                            existing.source_chunk_ids.append(chunk_id)
                        roles = list(existing.extra.get("roles") or [])
                        if role not in roles:
                            roles.append(role)
                            existing.extra["roles"] = roles
                        # The same mention can be typed differently in two
                        # events (see :func:`vhe_entity_uid`).  The node keeps
                        # every observed type so that the retrieval layer can
                        # still use it as a hint.
                        types = list(existing.extra.get("entity_types") or [])
                        if entity_type not in types:
                            types.append(entity_type)
                            existing.extra["entity_types"] = types

                    rel_type = relation_for_role(role, ontology.relation_types)
                    if edge_is_participant_to_event(rel_type):
                        src_uid, dst_uid = ent_uid, event_uid
                    else:
                        src_uid, dst_uid = event_uid, ent_uid
                    rel_uid = stable_uid(src_uid, rel_type, dst_uid, prefix="vhe_rel_")
                    relations[rel_uid] = Relation(
                        uid=rel_uid,
                        src=src_uid,
                        dst=dst_uid,
                        rel_type=rel_type,
                        text=f"{event_type}: {trigger} --{role.lower()}--> {mention}",
                        year=years[0] if years else None,
                        source=VHE_SOURCE,
                        confidence=0.9,
                        source_chunk_ids=[chunk_id],
                        extra={
                            "role": role,
                            "event_type": event_type,
                            "evidence_text": text[:400],
                        },
                    )

                    sup_uid = stable_uid(ent_uid, "SUPPORTED_BY", chunk_id, prefix="vhe_rel_")
                    relations[sup_uid] = Relation(
                        uid=sup_uid,
                        src=ent_uid,
                        dst=chunk_id,
                        rel_type="SUPPORTED_BY",
                        text=f"{mention} được hỗ trợ bởi VHE {instance_id}",
                        year=years[0] if years else None,
                        source=VHE_SOURCE,
                        confidence=1.0,
                        source_chunk_ids=[chunk_id],
                        extra={"evidence_text": text[:400], "role": role},
                    )

            # the event itself is supported by the sentence
            sup_uid = stable_uid(event_uid, "SUPPORTED_BY", chunk_id, prefix="vhe_rel_")
            relations[sup_uid] = Relation(
                uid=sup_uid,
                src=event_uid,
                dst=chunk_id,
                rel_type="SUPPORTED_BY",
                text=f"{event_type} được hỗ trợ bợi VHE {instance_id}",
                year=years[0] if years else None,
                source=VHE_SOURCE,
                confidence=1.0,
                source_chunk_ids=[chunk_id],
                extra={"evidence_text": text[:400], "event_type": event_type},
            )

    logger.info(
        "VHE gold: %d entity, %d relation, %d chunk",
        len(entities),
        len(relations),
        len(chunks),
    )
    return list(entities.values()), list(relations.values()), list(chunks.values())
