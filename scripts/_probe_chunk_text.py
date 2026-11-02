"""Dump the text of the chunks that back the VHE edges."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.kg.store import Neo4jGraphStore

CHUNKS = [
    "vhe_chunk_dvsktt-s60",
    "vhe_chunk_dvsktt-s121",
    "vhe_chunk_dvsktt-s125",
    "vhe_chunk_dvsktt-s117",
    "vhe_chunk_dvsktt-s56",
]


def main() -> None:
    settings = get_settings()
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str((getattr(settings, "neo4j", {}) or {}).get("user", "neo4j")),
        password=settings.neo4j_password,
    )
    for cid in CHUNKS:
        print("=" * 78)
        print("CHUNK:", cid)
        rows = store._run(
            """
            MATCH (c) WHERE c.uid = $cid OR c.chunk_id = $cid
            RETURN labels(c) AS labels, properties(c) AS props
            LIMIT 3
            """,
            cid=cid,
        )
        for row in rows:
            props = row["props"]
            print("labels:", row["labels"])
            print("keys:", sorted(props))
            for key in ("text", "title", "doc_id", "section", "url", "description"):
                if props.get(key):
                    print(f"  {key}: {str(props[key])[:400]}")


if __name__ == "__main__":
    main()
