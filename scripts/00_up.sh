#!/usr/bin/env bash
# Bật Neo4j + Qdrant và chờ cả hai service healthy.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; source .env; set +a
fi

COMPOSE="docker compose"
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE="docker-compose"
fi

echo "==> docker compose up -d (neo4j + qdrant)"
$COMPOSE up -d

echo "==> Chờ healthcheck..."
for i in $(seq 1 60); do
  NEO=$(docker inspect --format='{{.State.Health.Status}}' hgrag-neo4j 2>/dev/null || echo "none")
  QDR=$(docker inspect --format='{{.State.Health.Status}}' hgrag-qdrant 2>/dev/null || echo "none")
  if [ "$NEO" = "healthy" ] && [ "$QDR" = "healthy" ]; then
    echo "==> OK: neo4j=$NEO qdrant=$QDR"
    exit 0
  fi
  sleep 2
done

echo "!! Healthcheck chưa sạch sau 120s: neo4j=$NEO qdrant=$QDR" >&2
$COMPOSE ps
exit 1
