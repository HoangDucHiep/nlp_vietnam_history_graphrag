#!/usr/bin/env bash
# Step 8 of the pipeline: smoke test of the whole system.
#
# The script is *fast* and *offline-friendly*: it uses the mock LLM/embedder
# when the real services are not available, and it never fails the build for
# a non-critical step -- every step prints PASS/FAIL and the script exits
# with a non-zero code only when a *critical* step fails.
#
# Usage:
#     bash scripts/08_smoke_test.sh
#     bash scripts/08_smoke_test.sh --strict     # fail on any FAIL

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

STRICT=0
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
    esac
done

FAILED=0
CRITICAL_FAILED=0

step() {
    local name="$1"; shift
    echo ""
    echo "=== $name ==="
    if "$@"; then
        echo "PASS: $name"
    else
        echo "FAIL: $name"
        FAILED=$((FAILED + 1))
    fi
}

step_critical() {
    local name="$1"; shift
    echo ""
    echo "=== $name (critical) ==="
    if "$@"; then
        echo "PASS: $name"
    else
        echo "FAIL: $name"
        FAILED=$((FAILED + 1))
        CRITICAL_FAILED=$((CRITICAL_FAILED + 1))
    fi
}

echo "Vietnamese History GraphRAG — smoke test"
echo "repo: $REPO_ROOT"
echo "python: $PY"
"$PY" --version || true

# ---------------------------------------------------------------- 1. imports
step_critical "import các module chính" "$PY" -c "
import hgrag.config, hgrag.data.records, hgrag.llm.client
import hgrag.kg.ontology, hgrag.kg.alias, hgrag.kg.builder
import hgrag.retrieval.entity_linker, hgrag.retrieval.graph_search
import hgrag.retrieval.fusion, hgrag.retrieval.context, hgrag.retrieval.rerank
import hgrag.qa.pipeline, hgrag.qa.answer, hgrag.qa.explain
import hgrag.eval.metrics, hgrag.eval.judge, hgrag.eval.run_eval, hgrag.eval.report
print('imports OK')
"

# ---------------------------------------------------------------- 2. tests
step_critical "pytest (unit tests)" "$PY" -m pytest -q

# ---------------------------------------------------------------- 3. services
step "Neo4j kết nối được" "$PY" -c "
from hgrag.config import get_settings
from hgrag.kg.store import Neo4jGraphStore
s = get_settings()
store = Neo4jGraphStore(uri=s.neo4j_uri, user=str(s.neo4j.get('user','neo4j')), password=s.neo4j_password)
print('neo4j stats:', store.stats())
"

step "Qdrant kết nối được" "$PY" -c "
from hgrag.config import get_settings
from hgrag.retrieval.vector_store import QdrantVectorStore
s = get_settings()
vs = QdrantVectorStore(url=s.qdrant_url, vector_size=int(s.qdrant.get('vector_size',768)))
print('chunks:', vs.count('chunks'))
"

# ---------------------------------------------------------------- 4. artefacts
step "corpus tồn tại" "$PY" -c "
import json, pathlib
p = pathlib.Path('data/processed/corpus.jsonl')
assert p.exists(), 'thiếu data/processed/corpus.jsonl'
n = sum(1 for line in p.open(encoding='utf-8') if line.strip())
print('chunks:', n)
assert n > 0
"

step "bộ câu hỏi đánh giá tồn tại" "$PY" -c "
import json, pathlib
p = pathlib.Path('data/processed/eval_questions.jsonl')
assert p.exists(), 'thiếu data/processed/eval_questions.jsonl'
qs = [json.loads(l) for l in p.open(encoding='utf-8') if l.strip()]
print('câu hỏi:', len(qs))
assert len(qs) >= 100, f'chỉ có {len(qs)} câu (cần >= 100)'
"

# ---------------------------------------------------------------- 5. pipeline
step "QA pipeline (mock, offline)" "$PY" scripts/06_ask.py \
    "Trần Hưng Đạo đánh bại quân Mông Cổ vào năm nào?" --mock --json

step "đánh giá mini (mock, 3 câu)" "$PY" scripts/07_run_eval.py \
    --mock --limit 3 --no-ablations --reports /tmp/hgrag_smoke_reports

echo ""
echo "=============================================="
echo "smoke test: $FAILED step(s) failed"
if [ "$STRICT" -eq 1 ] && [ "$FAILED" -gt 0 ]; then
    echo "STRICT mode -> exit 1"
    exit 1
fi
if [ "$CRITICAL_FAILED" -gt 0 ]; then
    exit 1
fi
echo "OK"
exit 0
