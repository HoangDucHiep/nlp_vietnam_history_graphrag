"""Automatic error analysis for the evaluation run (docs/04 §5.4).

The module classifies every failed answer of a system into the error
categories of docs/04 §5.4 and aggregates them, so the report can point at
the *root cause* instead of only listing wrong answers.

Usage
-----
    python scripts/07_run_eval.py            # writes reports/error_analysis.md
    python -m hgrag.eval.report --help       # (module entry point)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import metrics as M
from .run_eval import EvalRun, EvalRow

__all__ = ["classify_error", "analyze_errors", "write_error_analysis"]

#: The error taxonomy of docs/04 §5.4.
ERROR_TYPES: tuple[str, ...] = (
    "entity_linking_wrong",
    "wrong_path",
    "missing_evidence",
    "generation_wrong_despite_evidence",
    "judge_mismatch",
    "out_of_kb_hallucination",
    "refused_when_answerable",
    "citation_invalid",
    "other",
)

ERROR_LABELS: dict[str, str] = {
    "entity_linking_wrong": "Entity linking sai",
    "wrong_path": "Đường đi sai",
    "missing_evidence": "Thiếu evidence",
    "generation_wrong_despite_evidence": "Sinh sai dù có evidence",
    "judge_mismatch": "Judge lệch",
    "out_of_kb_hallucination": "Bịa ở câu ngoài KB",
    "refused_when_answerable": "Từ chối nhầm câu trả lời được",
    "citation_invalid": "Trích dẫn không hợp lệ",
    "other": "Lỗi khác",
}

ERROR_ROOTS: dict[str, str] = {
    "entity_linking_wrong": "Alias thiếu / trùng tên → anchor node sai",
    "wrong_path": "Ontology thiếu quan hệ cần thiết / path scoring chưa tốt",
    "missing_evidence": "Chunk quá dài / embedding kém cho câu hỏi lịch sử",
    "generation_wrong_despite_evidence": "Prompt chưa ràng buộc đủ / context quá dài",
    "judge_mismatch": "Rubric chưa rõ / đáp án tham chiếu chưa sắc",
    "out_of_kb_hallucination": "Thiếu cơ chế từ chối khi không có evidence",
    "refused_when_answerable": "Ngân sách context quá chặt / evidence bị lọc hết",
    "citation_invalid": "Mô hình sinh [E#] không tồn tại / evidence thiếu text",
    "other": "Cần xem thủ công",
}


def classify_error(row: EvalRow, question_type: str = "") -> str:
    """Classify one failed answer into an error category."""
    markers = M.extract_citation_markers(row.answer)
    has_evidence = row.n_evidence > 0
    chunk_recall = row.chunk_recall_at_10

    if question_type == "out_of_kb" and not row.refused and row.answer:
        return "out_of_kb_hallucination"
    if row.refused and question_type != "out_of_kb":
        return "refused_when_answerable"
    if markers and not row.citation_precision:
        return "citation_invalid"
    if not has_evidence:
        return "missing_evidence"
    if chunk_recall < 0.5 and row.path_hit == 0:
        # gold chunk không có trong top-k và không tìm được đường đi
        return "missing_evidence" if row.entity_recall < 0.5 else "wrong_path"
    if row.entity_recall < 0.5:
        return "entity_linking_wrong"
    if row.path_hit == 0 and question_type == "multi_hop":
        return "wrong_path"
    if has_evidence and chunk_recall >= 0.5:
        return "generation_wrong_despite_evidence"
    return "other"


def analyze_errors(run: EvalRun, system: str = "ours") -> dict[str, Any]:
    """Aggregate the error categories of one system."""
    rows = run.rows_of(system)
    failures = [r for r in rows if not r.judge_correct]
    counts: dict[str, int] = {name: 0 for name in ERROR_TYPES}
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in ERROR_TYPES}
    for row in failures:
        category = classify_error(row, row.question_type)
        counts[category] = counts.get(category, 0) + 1
        if len(examples[category]) < 5:
            examples[category].append(
                {
                    "qid": row.qid,
                    "question": row.question,
                    "question_type": row.question_type,
                    "answer": row.answer[:300],
                    "reference": row.reference[:200],
                    "judge_reason": row.judge_reason,
                    "n_evidence": row.n_evidence,
                    "citations": M.extract_citation_markers(row.answer),
                }
            )
    total = len(failures)
    return {
        "system": system,
        "n_questions": len(rows),
        "n_failures": total,
        "failure_rate": (total / len(rows)) if rows else 0.0,
        "counts": counts,
        "ratios": {
            k: (v / total if total else 0.0) for k, v in counts.items()
        },
        "examples": examples,
    }


def write_error_analysis(
    run: EvalRun,
    path: str | Path,
    systems: Sequence[str] = ("ours",),
) -> Path:
    """Write ``reports/error_analysis.md``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    analyses = {system: analyze_errors(run, system) for system in systems}

    lines = [
        "# Phân tích lỗi — Vietnamese History GraphRAG",
        "",
        "Phân loại lỗi theo docs/04 §5.4. Mỗi lỗi được gán *dấu hiệu trong log* và "
        "*nguyên nhân gốc* để dẫn tới hướng khắc phục cụ thể.",
        "",
    ]

    for system, analysis in analyses.items():
        lines.append(f"## Hệ thống: {system}")
        lines.append("")
        lines.append(
            f"- Tổng câu: **{analysis['n_questions']}** · sai: **{analysis['n_failures']}** "
            f"({analysis['failure_rate'] * 100:.1f}%)"
        )
        lines.append("")
        lines.append("| Loại lỗi | Số lỗi | Tỉ lệ | Nguyên nhân gốc |")
        lines.append("|---|---|---|---|")
        for name in ERROR_TYPES:
            count = analysis["counts"].get(name, 0)
            if count == 0:
                continue
            lines.append(
                f"| {ERROR_LABELS[name]} | {count} | "
                f"{analysis['ratios'].get(name, 0.0) * 100:.1f}% | {ERROR_ROOTS[name]} |"
            )
        lines.append("")

        examples_found = False
        for name in ERROR_TYPES:
            items = analysis["examples"].get(name) or []
            if not items:
                continue
            if not examples_found:
                lines.append("### Ví dụ")
                lines.append("")
                examples_found = True
            lines.append(f"**{ERROR_LABELS[name]}**")
            lines.append("")
            for item in items:
                lines.append(
                    f"- `{item['qid']}` ({item['question_type']}) — *{item['question']}*"
                )
                lines.append(
                    f"  - Đáp án: {item['answer'].strip() or '(trống)'}"
                )
                lines.append(f"  - Tham chiếu: {item['reference'].strip() or '(không có)'}")
                if item["judge_reason"]:
                    lines.append(f"  - Judge: {item['judge_reason']}")
            lines.append("")
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _main() -> int:  # pragma: no cover - CLI helper
    import argparse

    parser = argparse.ArgumentParser(description="Phân tích lỗi từ artifacts/eval_run.json")
    parser.add_argument("--run", default="artifacts/eval_run.json")
    parser.add_argument("--out", default="reports/error_analysis.md")
    parser.add_argument("--systems", nargs="+", default=["ours"])
    args = parser.parse_args()

    payload = json.loads(Path(args.run).read_text(encoding="utf-8"))
    rows = []
    for raw in payload.get("rows") or []:
        rows.append(EvalRow(**{k: v for k, v in raw.items() if k in EvalRow.__dataclass_fields__}))
    run = EvalRun(rows=rows, meta=payload.get("meta") or {})
    path = write_error_analysis(run, args.out, args.systems)
    print(f"Đã ghi {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
