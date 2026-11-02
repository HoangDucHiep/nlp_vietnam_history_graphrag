#!/bin/bash
# Launch the full evaluation with 4 parallel workers.
#
# The workers share ONE global quota budget: the token limiter (16 000 input
# tokens/minute for gemma-4-26b) is coordinated through a lock file, so the
# four processes together stay below the free-tier limit.
#
# Each worker handles a disjoint set of systems so that no (system, qid) pair
# is produced twice.  Because ``--systems`` is given explicitly, the seven
# ablations are NOT added automatically (see scripts/07_run_eval.py): the
# ablation systems are listed explicitly and each of them really disables the
# corresponding component of the pipeline.
#
# The embedder is pinned to the *local* sentence-transformers model: the
# Gemini embedding quota of the free tier is exhausted (1000 requests/day),
# and the corpus in Qdrant was indexed with the 384-dim local model
# (`chunks_local` + `chunks_vhe_local`).  Pinning it makes the run
# reproducible and avoids a silent switch to a different vector space.
set -u
cd /media/hoanghiep/All/Codes/NPL/vietnam_history_graphrag
set -a
. ./.env
set +a

export HGRAG_EMBED_PROVIDER=local

LOG_DIR=artifacts
mkdir -p "$LOG_DIR"

run_worker() {
  local name="$1"; shift
  local systems="$*"
  .venv/bin/python -u scripts/07_run_eval.py \
    --questions data/processed/eval_questions.jsonl \
    --checkpoint "$LOG_DIR/eval_ckpt_${name}.json" \
    --checkpoint-every 5 \
    --out "$LOG_DIR/eval_run_${name}.json" \
    --concurrency 6 \
    --systems $systems \
    > "$LOG_DIR/log_eval_${name}.txt" 2>&1
}

run_worker main B0 B1 B2 &
run_worker b ours A1_no_entity_linking A2_no_resolution &
run_worker c A3_no_temporal A4_no_graph A5_no_rerank &
run_worker d A6_no_context_budget A7_no_fusion &
wait
echo "ALL WORKERS DONE"
