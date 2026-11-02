"""Quantify duplicate entity names in the KG (same name, different uid)."""
from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.kg.store import Neo4jGraphStore


def main() -> None:
    settings = get_settings()
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str((getattr(settings, "neo4j", {}) or {}).get("user", "neo4j")),
        password=settings.neo4j_password,
    )
    rows = store._run(
        """
        MATCH (e:Entity)
        WHERE e.entity_type <> 'Chunk'
        RETURN e.name AS name, collect(DISTINCT e.uid) AS uids,
               collect(DISTINCT e.entity_type) AS types,
               count(*) AS n
        """
    )
    by_name: dict[str, list] = defaultdict(list)
    for row in rows:
        by_name[row["name"]].append(row)
    dup = {k: v for k, v in by_name.items() if len(v[0]["uids"]) > 1}
    print("total distinct names:", len(by_name))
    print("names with >1 uid  :", len(dup))
    for name, group in list(dup.items())[:25]:
        print(f"  {name!r}: uids={group[0]['uids']} types={group[0]['types']}")
    print()
    # how many relations touch a duplicated name
    dup_names = list(dup)
    if dup_names:
        r = store._run(
            """
            MATCH (a:Entity)-[rel]->(b:Entity)
            WHERE a.name IN $names OR b.name IN $names
            RETURN count(rel) AS n
            """,
            names=dup_names,
        )
        print("relations touching a duplicated name:", r[0]["n"] if r else 0)


if __name__ == "__main__":
    main()
