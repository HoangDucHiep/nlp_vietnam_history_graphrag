#!/usr/bin/env bash
# Dừng Neo4j + Qdrant (giữ volume).
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE="docker compose"
if ! docker compose version >/dev/null 2>&1; then
  COMPOSE="docker-compose"
fi

if [ "${1:-}" = "--volumes" ] || [ "${1:-}" = "-v" ]; then
  echo "==> docker compose down -v (XÓA DỮ LIỆU)"
  $COMPOSE down -v
else
  echo "==> docker compose down (giữ volume)"
  $COMPOSE down
fi
