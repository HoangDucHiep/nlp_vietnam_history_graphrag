#!/usr/bin/env bash
# Chờ 4 worker đánh giá chạy xong, rồi hợp nhất và sinh báo cáo.
#
# Dùng nền:  `setsid nohup bash scripts/wait_and_report.sh > artifacts/wait_report.log 2>&1 &`
set -u
cd /media/hoanghiep/All/Codes/NPL/vietnam_history_graphrag
set -a
. ./.env
set +a
export HGRAG_EMBED_PROVIDER=local

echo "[$(date +%T)] chờ worker kết thúc..."

while true; do
  running=$(pgrep -fc "07_run_eval.py" || true)
  if [ "${running:-0}" -eq 0 ]; then
    echo "[$(date +%T)] tất cả worker đã xong"
    break
  fi
  rows=$(.venv/bin/python - <<'EOF'
import json, glob
rows = []
for f in glob.glob("artifacts/eval_ckpt_*.json"):
    try:
        rows += json.load(open(f)).get("rows", [])
    except Exception:
        pass
print(len(rows))
EOF
)
  echo "[$(date +%T)] còn $running worker, $rows/1650 dòng"
  sleep 120
done

echo "[$(date +%T)] chấm lại các dòng bị degrade..."
.venv/bin/python scripts/14_rejudge_degraded.py --in-place || true

echo "[$(date +%T)] hợp nhất checkpoint..."
.venv/bin/python scripts/12_merge_eval.py artifacts/eval_ckpt_*.json \
  --out artifacts/eval_run.json --reports reports || true

echo "[$(date +%T)] xong"
