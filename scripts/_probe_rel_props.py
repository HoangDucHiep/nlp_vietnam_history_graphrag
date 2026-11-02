"""Dump the full properties of the VHE relations and their chunk nodes."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.kg.store import Neo4jGraphStore

REL_UIDS = []


def main() -> None:
    settings = get_settings()
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str((getattr(settings, "neo4j", {}) or {}).get("user", "neo4j")),
        password=settings.neo4j_password,
    )
    rows = store._run(
        """
        MATCH (a:Entity)-[r:ABOUT]->(b:Entity)
        WHERE a.name IN ['Mị Nương cưới Sơn Tinh', 'Nhà Tần thành lập Nam Hải',
                          'Cao Vương tấn công Nam Chiếu', 'qua đời Tần Thuỷ Hoàng']
        RETURN a.name AS src, b.name AS dst, properties(r) AS props
        """
    )
    for row in rows:
        print("=" * 78)
        print(row["src"], "->", row["dst"])
        for key, value in sorted(row["props"].items()):
            print(f"   {key}: {str(value)[:300]}")

    print()
    print("#" * 78)
    print("SUPPORTED_BY chunk node properties")
    rows = store._run(
        """
        MATCH (a:Entity)-[r:SUPPORTED_BY]->(c:Entity)
        WHERE a.name IN ['Mị Nương cưới Sơn Tinh', 'Nhà Tần thành lập Nam Hải']
        RETURN a.name AS src, c.name AS chunk, properties(c) AS cprops
        """
    )
    for row in rows:
        print("=" * 78)
        print(row["src"], "->", row["chunk"])
        for key in ("uid", "title", "description", "entity_type", "source_chunk_ids"):
            print(f"   {key}: {str(row['cprops'].get(key))[:300]}")


if __name__ == "__main__":
    main()
