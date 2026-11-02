"""Check for duplicate entities sharing the same name (uid mismatch)."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.kg.store import Neo4jGraphStore

NAMES = ["Sơn Tinh", "Nam Hải", "Tần Thuỷ Hoàng", "Nam Chiếu", "Mị Nương"]


def main() -> None:
    settings = get_settings()
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str((getattr(settings, "neo4j", {}) or {}).get("user", "neo4j")),
        password=settings.neo4j_password,
    )
    for name in NAMES:
        print("=" * 78)
        print("NAME:", name)
        rows = store._run(
            """
            MATCH (e:Entity {name: $name})
            RETURN e.uid AS uid, e.entity_type AS type, e.description AS desc,
                   e.source_chunk_ids AS chunks
            """,
            name=name,
        )
        for row in rows:
            print(f"  uid={row['uid']} type={row['type']} chunks={row['chunks']}")
            print(f"     desc: {str(row['desc'] or '')[:160]}")


if __name__ == "__main__":
    main()
