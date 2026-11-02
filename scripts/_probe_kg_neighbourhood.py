"""Inspect the KG neighbourhood of a VHE entity (raw edge text included)."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.kg.store import Neo4jGraphStore

TARGETS = ["Sơn Tinh", "Nam Hải", "Tần Thuỷ Hoàng", "Nam Chiếu"]


def main() -> None:
    settings = get_settings()
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str((getattr(settings, "neo4j", {}) or {}).get("user", "neo4j")),
        password=settings.neo4j_password,
    )
    for name in TARGETS:
        print("=" * 78)
        print("ENTITY:", name)
        rows = store._run(
            """
            MATCH (e:Entity {name: $name})-[r]-(o)
            WITH startNode(r) AS s, endNode(r) AS d, type(r) AS t, properties(r) AS p
            RETURN s.name AS src, t AS rel, d.name AS dst,
                   p.text AS text, p.source_chunk_ids AS chunks, p.role AS role
            LIMIT 60
            """,
            name=name,
        )
        for row in rows:
            print(
                f"  {row['src']} -{row['rel']}-> {row['dst']}"
                f" | role={row.get('role')} | chunks={row.get('chunks')}"
            )
            if row.get("text"):
                print("      TEXT:", str(row["text"])[:220])


if __name__ == "__main__":
    main()
