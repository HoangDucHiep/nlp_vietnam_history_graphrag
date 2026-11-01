"""Knowledge-graph store protocol and Neo4j implementation.

The rest of the system only depends on the :class:`GraphStore` protocol, so
the Neo4j backend can be swapped for an in-memory/SQLite fallback without
touching the KG builder or the retrieval layer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from ..data.records import Entity, Relation

logger = logging.getLogger(__name__)


def _restore_extra(props: dict[str, Any]) -> dict[str, Any]:
    """Neo4j cannot store nested maps: ``extra`` is persisted as JSON text."""
    raw = props.pop("extra_json", None)
    if raw is not None:
        try:
            props["extra"] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            props["extra"] = {}
    if not isinstance(props.get("extra"), dict):
        props["extra"] = {}
    return props


@runtime_checkable
class GraphStore(Protocol):
    """Minimal graph persistence interface."""

    def upsert_entities(self, entities: Iterable[Entity]) -> int: ...

    def upsert_relations(self, relations: Iterable[Relation]) -> int: ...

    def get_entity(self, uid: str) -> dict[str, Any] | None: ...

    def find_entity_by_name(self, name: str) -> list[dict[str, Any]]: ...

    def neighbors(self, uid: str, rel_types: list[str] | None = None) -> list[dict[str, Any]]: ...

    def stats(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def entity_props(entity: Entity) -> dict[str, Any]:
    """Flatten an :class:`Entity` into Neo4j-friendly properties."""
    return {
        "uid": entity.uid,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "wikidata_id": entity.wikidata_id,
        "aliases": entity.aliases,
        "name_key": entity.name_key,
        "description": entity.description,
        "real_name": entity.real_name,
        "title": entity.title,
        "posthumous_name": entity.posthumous_name,
        "period": entity.period,
        "year_min": entity.year_min,
        "year_max": entity.year_max,
        "roles": entity.roles,
        "p31": entity.p31,
        "p27": entity.p27,
        "wiki_categories": entity.wiki_categories,
        "source": entity.source,
        "confidence": entity.confidence,
        "source_chunk_ids": entity.source_chunk_ids,
        "extra": entity.extra,
    }


class Neo4jGraphStore:
    """Neo4j-backed :class:`GraphStore`.

    Node label: ``Entity`` with property ``uid`` as the merge key.
    Relationship type: the relation ``rel_type`` (upper-snake-case).
    Every relationship carries provenance: ``source``, ``confidence``,
    ``source_chunk_ids`` and ``text``.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "history2025",
        database: str = "neo4j",
        batch_size: int = 500,
    ):
        from neo4j import GraphDatabase  # imported lazily

        self.batch_size = batch_size
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._driver.verify_connectivity()
        self.database = database
        logger.info("Đã kết nối Neo4j tại %s", uri)

    # -- internals -------------------------------------------------------
    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Run a Cypher query and materialise all records inside the session.

        The result MUST be consumed before the session closes, otherwise the
        driver raises ``ResultConsumedError``.
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    @staticmethod
    def _entity_props(entity: Entity) -> dict[str, Any]:
        props = entity_props(entity)
        # Neo4j properties must be primitives/lists -> serialise the nested map
        props["extra_json"] = json.dumps(props.pop("extra", {}), ensure_ascii=False)
        return props

    # -- writes ----------------------------------------------------------
    def upsert_entities(self, entities: Iterable[Entity]) -> int:
        rows = [self._entity_props(e) for e in entities]
        if not rows:
            return 0
        query = """
        UNWIND $rows AS row
        MERGE (e:Entity {uid: row.uid})
        SET e += row
        RETURN count(*) AS n
        """
        total = 0
        for i in range(0, len(rows), self.batch_size):
            records = self._run(query, rows=rows[i : i + self.batch_size])
            total += int(records[0]["n"]) if records else 0
        return total

    def upsert_relations(self, relations: Iterable[Relation]) -> int:
        rels = list(relations)
        if not rels:
            return 0
        # group by rel_type so the Cypher type is a literal (no injection)
        by_type: dict[str, list[dict[str, Any]]] = {}
        for r in rels:
            by_type.setdefault(r.rel_type, []).append(
                {
                    "uid": r.uid,
                    "src": r.src,
                    "dst": r.dst,
                    "text": r.text,
                    "year": r.year,
                    "source": r.source,
                    "confidence": r.confidence,
                    "source_chunk_ids": r.source_chunk_ids,
                    "extra_json": json.dumps(r.extra, ensure_ascii=False),
                }
            )
        total = 0
        for rel_type, rows in by_type.items():
            if not rel_type.replace("_", "").isalnum():
                raise ValueError(f"Loại quan hệ không hợp lệ: {rel_type}")
            query = f"""
            UNWIND $rows AS row
            MATCH (a:Entity {{uid: row.src}})
            MATCH (b:Entity {{uid: row.dst}})
            MERGE (a)-[r:{rel_type} {{uid: row.uid}}]->(b)
            SET r += row
            RETURN count(r) AS n
            """
            for i in range(0, len(rows), self.batch_size):
                records = self._run(query, rows=rows[i : i + self.batch_size])
                total += int(records[0]["n"]) if records else 0
        return total

    # -- reads -----------------------------------------------------------
    def get_entity(self, uid: str) -> dict[str, Any] | None:
        records = self._run(
            "MATCH (e:Entity {uid: $uid}) RETURN e{.*} AS props", uid=uid
        )
        if not records:
            return None
        return _restore_extra(dict(records[0]["props"]))

    def find_entity_by_name(self, name: str) -> list[dict[str, Any]]:
        records = self._run(
            """
            MATCH (e:Entity)
            WHERE e.name = $name OR $name IN e.aliases OR e.name_key = $name
            RETURN e{.*} AS props LIMIT 20
            """,
            name=name,
        )
        return [_restore_extra(dict(r["props"])) for r in records]

    def neighbors(self, uid: str, rel_types: list[str] | None = None) -> list[dict[str, Any]]:
        rel_filter = ""
        if rel_types:
            safe = [t for t in rel_types if t.replace("_", "").isalnum()]
            if safe:
                rel_filter = ":" + "|".join(safe)
        records = self._run(
            f"""
            MATCH (a:Entity {{uid: $uid}})-[r{rel_filter}]-(b:Entity)
            RETURN a{{.*}} AS src, type(r) AS rel, properties(r) AS rprops, b{{.*}} AS dst
            LIMIT 500
            """,
            uid=uid,
        )
        return [
            {
                "src": _restore_extra(dict(r["src"])),
                "rel": r["rel"],
                "rel_props": _restore_extra(dict(r["rprops"])),
                "dst": _restore_extra(dict(r["dst"])),
            }
            for r in records
        ]

    def stats(self) -> dict[str, Any]:
        records = self._run(
            """
            MATCH (e:Entity)
            WITH count(e) AS nodes, collect(DISTINCT e.entity_type) AS types
            MATCH ()-[r]->()
            WITH nodes, types, count(r) AS rels
            RETURN nodes, types, rels
            """
        )
        if not records:
            return {"nodes": 0, "relations": 0, "entity_types": []}
        rec = records[0]
        return {
            "nodes": int(rec["nodes"]),
            "relations": int(rec["rels"]),
            "entity_types": sorted(t for t in rec["types"] if t),
        }

    def close(self) -> None:
        self._driver.close()


class InMemoryGraphStore:
    """Fallback graph store used when Neo4j is unavailable (tests, offline)."""

    def __init__(self) -> None:
        self.entities: dict[str, dict[str, Any]] = {}
        self.relations: dict[str, dict[str, Any]] = {}

    def upsert_entities(self, entities: Iterable[Entity]) -> int:
        n = 0
        for e in entities:
            self.entities[e.uid] = entity_props(e)
            n += 1
        return n

    def upsert_relations(self, relations: Iterable[Relation]) -> int:
        n = 0
        for r in relations:
            if r.src not in self.entities or r.dst not in self.entities:
                continue
            self.relations[r.uid] = {
                "uid": r.uid,
                "src": r.src,
                "dst": r.dst,
                "rel_type": r.rel_type,
                "text": r.text,
                "year": r.year,
                "source": r.source,
                "confidence": r.confidence,
                "source_chunk_ids": r.source_chunk_ids,
                "extra": r.extra,
            }
            n += 1
        return n

    def get_entity(self, uid: str) -> dict[str, Any] | None:
        return self.entities.get(uid)

    def find_entity_by_name(self, name: str) -> list[dict[str, Any]]:
        out = []
        for props in self.entities.values():
            if (
                props.get("name") == name
                or name in (props.get("aliases") or [])
                or props.get("name_key") == name
            ):
                out.append(props)
        return out[:20]

    def neighbors(self, uid: str, rel_types: list[str] | None = None) -> list[dict[str, Any]]:
        out = []
        for rel in self.relations.values():
            if rel_types and rel["rel_type"] not in rel_types:
                continue
            if uid in (rel["src"], rel["dst"]):
                other = rel["dst"] if rel["src"] == uid else rel["src"]
                out.append(
                    {
                        "src": self.entities.get(rel["src"], {}),
                        "rel": rel["rel_type"],
                        "rel_props": {k: v for k, v in rel.items() if k not in ("src", "dst")},
                        "dst": self.entities.get(other, {}),
                    }
                )
        return out

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": len(self.entities),
            "relations": len(self.relations),
            "entity_types": sorted({e.get("entity_type", "") for e in self.entities.values()} - {""}),
        }

    def close(self) -> None:  # pragma: no cover - trivial
        return None
