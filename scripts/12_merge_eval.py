#!/usr/bin/env python
"""Step 7b: merge the checkpoints of several evaluation workers into one report.

A full evaluation is 150 questions x 11 systems = 1 650 scored answers.  On the
free tier of the Gemini API the throughput is limited by the requests-per-minute
quota, so the run can be split over several workers (each with its own
``--systems`` list and its own checkpoint file).  This script merges those
checkpoints, de-duplicates the rows by ``(system, qid)`` and writes the report.

Usage
-----
    python scripts/12_merge_eval.py artifacts/eval_checkpoint*.json \\
        --out artifacts/eval_run.json --reports reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.eval.report import build_report  # noqa: E402
from hgrag.eval.run_eval import EvalRow, EvalRun, compare_systems  # noqa: E402

logger = logging.getLogger("12_merge_eval")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gộp các checkpoint đánh giá và sinh báo cáo")
    p.add_argument("checkpoints", type=str, nargs="+", help="các file checkpoint JSON")
    p.add_argument("--out", type=str, default="artifacts/eval_run.json")
    p.add_argument("--reports", type=str, default="reports")
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--error-systems", type=str, nargs="+", default=["ours"])
    return p


def load_rows(paths: list[str]) -> EvalRun:
    """Load every checkpoint and merge the rows (first writer wins)."""
    run = EvalRun(meta={"n_questions": 0, "systems": []})
    seen: set[tuple[str, str]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            logger.warning("Không có checkpoint %s -- bỏ qua", path)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("meta") or {}
        run.meta.setdefault("config", meta.get("config") or {})
        for key, value in meta.items():
            if key not in ("n_questions", "systems") and key not in run.meta:
                run.meta[key] = value
        for raw in payload.get("rows") or []:
            row = EvalRow(**{k: v for k, v in raw.items() if k in EvalRow.__annotations__})
            key = (row.system, row.qid)
            if key in seen:
                continue
            seen.add(key)
            run.rows.append(row)
        logger.info("%s: %d dòng", path.name, len(payload.get("rows") or []))
    run.meta["n_questions"] = len({r.qid for r in run.rows})
    run.meta["systems"] = sorted({r.system for r in run.rows})
    return run


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run = load_rows(args.checkpoints)
    if not run.rows:
        print(json.dumps({"error": "Không có dòng kết quả nào"}, ensure_ascii=False))
        return 4

    comparisons = [
        compare_systems(run, baseline=baseline, target="ours", n_samples=args.bootstrap, seed=args.seed)
        for baseline in ("B0", "B1", "B2")
        if baseline in (run.meta.get("systems") or [])
    ]

    reports_dir = Path(args.reports)
    if not reports_dir.is_absolute():
        reports_dir = REPO_ROOT / reports_dir
    artefacts = build_report(run, reports_dir, comparisons)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    run.save(out_path)

    summary = {
        "n_questions": run.meta.get("n_questions"),
        "n_rows": len(run.rows),
        "systems": run.meta.get("systems"),
        "artifacts": artefacts,
        "raw_rows": str(out_path),
        "comparisons": comparisons,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
