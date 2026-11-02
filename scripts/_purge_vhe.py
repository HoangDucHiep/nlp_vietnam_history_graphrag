"""Drop the stale VHE subgraph (old type-split uids) and re-import it.

The VHE entity uid no longer contains the entity type, so the nodes created
by the previous scheme are orphans: they keep the old names but the new
relations point at the canonical uids.  The whole ``source = 'vhe'`` subgraph
is therefore deleted and rebuilt from the gold annotations.

The VHE *chunk* nodes keep their uid (``vhe_chunk_<instance_id>``), so the
Qdrant index of the VHE sentences stays valid.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.kg.store import Neo4jGraphStore


def main() -> None:
    settings = get_settings()
    ncfg = getattr(settings, "neo4j", {}) or {}
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str(ncfg.get("user", "neo4j")),
        password=settings.neo4j_password,
        database=str(ncfg.get("database", "neo4j")),
    )

    before = store._run(
        """
        MATCH (e:Entity) WHERE e.source = 'vhe'
        RETURN count(e) AS nodes
        """
    )
    print("vhe nodes before:", before)

    # 1) drop the relations of the VHE subgraph
    rels = store._run(
        """
        MATCH ()-[r]->()
        WHERE r.source = 'vhe'
        RETURN count(r) AS n
        """
    )
    print("vhe relations before:", rels)

    # 2) drop the nodes of the VHE subgraph (the chunk nodes included: they
    #    are re-created with the same uid)
    store._run(
        """
        MATCH (e:Entity) WHERE e.source = 'vhe'
        DETACH DELETE e
        """
    )
    store._run(
        """
        MATCH (e:Entity) WHERE e.source = 'vhe'
        RETURN count(e) AS n
        """
    )

    # 3) drop the orphan relations (their source node is gone)
    store._run(
        """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.uid IS NULL OR b.uid IS NULL
        DELETE r
        """
    )

    after = store._run(
        """
        MATCH (e:Entity) WHERE e.source = 'vhe'
        RETURN count(e) AS nodes
        """
    )
    print("vhe nodes after purge:", after)


if __name__ == "__main__":
    main()
