#!/usr/bin/env bash
# Run the whole pipeline end to end.
#
#   bash scripts/run_all.sh              # full pipeline (dùng LLM thật)
#   bash scripts/run_all.sh --mock       # offline, nhanh (dùng cho demo/CI)
#   bash scripts/run_all.sh --quick      # bỏ bước ingest (dùng lại corpus sẵn có)
#
# Mọi bước đều idempotent và có cache: chạy lại không tốn tiền LLM.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$REPO_ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

MOCK=""
QUICK=0
EVAL_LIMIT=""
KG_LIMIT=""

for arg in "$@"; do
    case "$arg" in
        --mock) MOCK="--mock" ;;
        --quick) QUICK=1 ;;
        --eval-limit=*) EVAL_LIMIT="--limit ${arg#*=}" ;;
        --kg-limit=*) KG_LIMIT="--limit ${arg#*=}" ;;
    esac
done

banner() {
    echo ""
    echo "############################################################"
    echo "# $1"
    echo "############################################################"
}

banner "0/8 — bật Neo4j + Qdrant"
bash scripts/00_up.sh || echo "CẢNH BÁO: không bật được dịch vụ (sẽ fallback sang in-memory)"

if [ "$QUICK" -eq 0 ]; then
    banner "1/8 — ingest dữ liệu (VHE + Wikipedia + History Chat)"
    "$PY" scripts/01_ingest.py $MOCK || echo "BỎ QUA: ingest lỗi"
fi

banner "2/8 — build knowledge graph (Neo4j)"
"$PY" scripts/03_build_kg.py $MOCK $KG_LIMIT --out artifacts/kg_stats.json || echo "BỎ QUA: build KG lỗi"

banner "3/8 — embed corpus (Qdrant)"
"$PY" scripts/04_embed.py $MOCK --out artifacts/embed_stats.json || echo "BỎ QUA: embed lỗi"

banner "4/8 — sinh bộ câu hỏi đánh giá"
"$PY" scripts/05_eval_dataset.py $MOCK --out artifacts/eval_stats.json || echo "BỎ QUA: dataset lỗi"

banner "5/8 — smoke test (offline)"
bash scripts/08_smoke_test.sh || echo "BỎ QUA: smoke test lỗi"

banner "6/8 — live check (dịch vụ thật)"
"$PY" scripts/09_live_check.py || echo "BỎ QUA: live check lỗi"

banner "7/8 — chạy đánh giá (B0/B1/B2/Ours + ablation)"
"$PY" scripts/07_run_eval.py $MOCK $EVAL_LIMIT --out artifacts/eval_run.json || echo "BỎ QUA: eval lỗi"

banner "8/8 — sinh báo cáo"
ls -la reports/ 2>/dev/null || true

echo ""
echo "Xong. Báo cáo nằm ở reports/results.md và reports/results.csv"
