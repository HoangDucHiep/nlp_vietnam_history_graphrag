"""How many nodes would merge if the VHE entity uid ignored the role type?"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.data.normalize import name_key
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
        WHERE e.source = 'vhe' AND e.entity_type <> 'Chunk'
        RETURN e.uid AS uid, e.name AS name, e.entity_type AS t
        """
    )
    by_key: dict[str, list] = defaultdict(list)
    for row in rows:
        by_key[name_key(row["name"]) or row["name"]].append(row)

    multi = {k: v for k, v in by_key.items() if len(v) > 1}
    print("VHE nodes          :", len(rows))
    print("distinct name_keys :", len(by_key))
    print("keys with >1 node  :", len(multi))
    nodes_merged = sum(len(v) - 1 for v in multi.values())
    print("nodes that merge   :", nodes_merged)

    type_conflicts = Counter()
    for key, group in multi.items():
        types = sorted({r["t"] for r in group})
        type_conflicts["+".join(types)] += 1
    print()
    print("type-combination of the merged keys:")
    for combo, count in type_conflicts.most_common(20):
        print(f"   {combo:40s} {count}")

    print()
    print("examples (same name, different type):")
    shown = 0
    for key, group in multi.items():
        types = sorted({r["t"] for r in group})
        if len(types) > 1 and shown < 15:
            print(f"   {group[0]['name']!r}: {[(r['uid'], r['t']) for r in group]}")
            shown += 1

    # also: how many relations would be affected
    names = [g[0]["name"] for g in multi.values()]
    if names:
        r = store._run(
            """
            MATCH (a:Entity)-[rel]->(b:Entity)
            WHERE a.name IN $names OR b.name IN $names
            RETURN count(rel) AS n
            """,
            names=names,
        )
        print()
        print("relations touching a merged key:", r[0]["n"] if r else 0)


if __name__ == "__main__":
    main()
