"""Report generation for the evaluation run (docs/04 §5).

The report module produces:

* ``reports/results.csv``   -- one row per (system, question group)
* ``reports/results.md``    -- the main / ablation tables
* ``reports/figures/*.png`` -- the figures of docs/04 §5.3
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .error_analysis import write_error_analysis
from .run_eval import ABLATIONS, EvalRun, compare_systems, summarize_run

logger = logging.getLogger(__name__)

__all__ = [
    "write_results_csv",
    "write_results_markdown",
    "make_figures",
    "build_report",
    "write_error_analysis",
]

#: Columns of ``reports/results.csv``.
CSV_COLUMNS = (
    "system",
    "group",
    "n",
    "accuracy",
    "judge_score",
    "groundedness",
    "hallucination_rate",
    "token_f1",
    "rouge_l",
    "citation_precision",
    "citation_recall",
    "no_citation_rate",
    "chunk_recall_at_10",
    "ndcg_at_10",
    "entity_recall",
    "path_hit",
    "temporal_accuracy",
    "gold_fact_accuracy",
    "latency_p50",
    "latency_p95",
)


def _rows_for(run: EvalRun) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summarize_run(run):
        system = summary["system"]
        rows.append(_summary_row(system, "overall", summary))
        for qtype, stats in (summary.get("by_type") or {}).items():
            rows.append(
                {
                    "system": system,
                    "group": qtype,
                    "n": stats.get("n", 0),
                    "accuracy": round(float(stats.get("accuracy", 0.0)), 6),
                    "judge_score": "",
                    "groundedness": (
                        round(float(stats.get("groundedness", 0.0)), 6)
                        if stats.get("groundedness_applicable", True)
                        else ""
                    ),
                    "hallucination_rate": (
                        round(1.0 - float(stats.get("groundedness", 0.0)), 6)
                        if stats.get("groundedness_applicable", True)
                        else ""
                    ),
                    "token_f1": round(float(stats.get("token_f1", 0.0)), 6),
                    "rouge_l": "",
                    "citation_precision": "",
                    "citation_recall": "",
                    "no_citation_rate": "",
                    "chunk_recall_at_10": "",
                    "ndcg_at_10": "",
                    "entity_recall": "",
                    "path_hit": round(float(stats.get("path_hit", 0.0)), 6),
                    "temporal_accuracy": "",
                    "gold_fact_accuracy": "",
                    "latency_p50": "",
                    "latency_p95": "",
                }
            )
    return rows


def _summary_row(system: str, group: str, summary: dict[str, Any]) -> dict[str, Any]:
    grounded_applicable = bool(summary.get("groundedness_applicable", True))
    grounded = round(float(summary.get("groundedness", 0.0)), 6) if grounded_applicable else ""
    return {
        "system": system,
        "group": group,
        "n": summary.get("n", 0),
        "accuracy": round(float(summary.get("accuracy", 0.0)), 6),
        "judge_score": round(float(summary.get("judge_score", 0.0)), 6),
        "groundedness": grounded,
        "hallucination_rate": round(1.0 - float(summary.get("groundedness", 0.0)), 6)
        if grounded_applicable
        else "",
        "token_f1": round(float(summary.get("token_f1", 0.0)), 6),
        "rouge_l": round(float(summary.get("rouge_l", 0.0)), 6),
        "citation_precision": round(float(summary.get("citation_precision", 0.0)), 6),
        "citation_recall": round(float(summary.get("citation_recall", 0.0)), 6),
        "no_citation_rate": round(float(summary.get("no_citation_rate", 0.0)), 6),
        "chunk_recall_at_10": round(float(summary.get("chunk_recall_at_10", 0.0)), 6),
        "ndcg_at_10": round(float(summary.get("ndcg_at_10", 0.0)), 6),
        "entity_recall": round(float(summary.get("entity_recall", 0.0)), 6),
        "path_hit": round(float(summary.get("path_hit", 0.0)), 6),
        "temporal_accuracy": round(float(summary.get("temporal_accuracy", 0.0)), 6),
        "gold_fact_accuracy": round(float(summary.get("gold_fact_accuracy", 0.0)), 6),
        "latency_p50": round(float(summary.get("latency_p50", 0.0)), 4),
        "latency_p95": round(float(summary.get("latency_p95", 0.0)), 4),
    }


def write_results_csv(run: EvalRun, path: str | Path) -> Path:
    """Write ``results.csv``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = _rows_for(run)
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "–"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_signed(value: Any, digits: int = 3) -> str:
    """Format a *difference* with an explicit sign.

    The ablation table shows ``Δ so với Ours``.  Without the sign a value
    like ``0.080`` is ambiguous: it reads as "Ours tốt hơn 8 %" while it
    actually means "ablation này **tốt hơn** Ours 8 %".
    """
    if value is None or value == "":
        return "–"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _groundedness(summary: dict[str, Any]) -> Any:
    """The groundedness of a summary, or ``None`` when it is not applicable.

    The B0 baseline answers closed-book: there is no context to check the
    claims against, so its groundedness must be reported as "–" instead of
    a misleading ``0.000``.
    """
    if not summary.get("groundedness_applicable", True):
        return None
    return summary.get("groundedness")


def _main_table(run: EvalRun) -> str:
    summaries = {s["system"]: s for s in summarize_run(run)}
    groups = sorted(
        {
            qtype
            for s in summaries.values()
            for qtype in (s.get("by_type") or {})
        }
    )
    header = (
        "| Hệ thống | n | Overall acc | Groundedness | Token-F1 | "
        "Cite-P | Cite-R | Path hit | p50 (s) | p95 (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for system in ("B0", "B1", "B2", "ours"):
        s = summaries.get(system)
        if not s:
            continue
        lines.append(
            "| {name} | {n} | {acc} | {gr} | {f1} | {cp} | {cr} | {ph} | {p50} | {p95} |".format(
                name=_SYSTEM_LABELS.get(system, system),
                n=s.get("n", 0),
                acc=_fmt(s.get("accuracy")),
                gr=_fmt(_groundedness(s)),
                f1=_fmt(s.get("token_f1")),
                cp=_fmt(s.get("citation_precision")),
                cr=_fmt(s.get("citation_recall")),
                ph=_fmt(s.get("path_hit")),
                p50=_fmt(s.get("latency_p50"), 2),
                p95=_fmt(s.get("latency_p95"), 2),
            )
        )
    if groups:
        lines.append("")
        lines.append("### Theo nhóm câu hỏi (accuracy)")
        lines.append("")
        lines.append("| Hệ thống | " + " | ".join(groups) + " |")
        lines.append("|---" * (len(groups) + 1) + "|")
        for system in ("B0", "B1", "B2", "ours"):
            s = summaries.get(system)
            if not s:
                continue
            cells = [
                _fmt((s.get("by_type") or {}).get(g, {}).get("accuracy"))
                for g in groups
            ]
            lines.append(
                f"| {_SYSTEM_LABELS.get(system, system)} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def _ablation_table(run: EvalRun) -> str:
    summaries = {s["system"]: s for s in summarize_run(run)}
    base = summaries.get("ours")
    lines = [
        "| Cấu hình | Overall acc | Δ so với Ours (− = kém hơn) | Groundedness | Token-F1 | Path hit |",
        "|---|---|---|---|---|---|",
    ]
    if base:
        lines.append(
            "| Ours (đầy đủ) | {acc} | 0 | {gr} | {f1} | {ph} |".format(
                acc=_fmt(base.get("accuracy")),
                gr=_fmt(_groundedness(base)),
                f1=_fmt(base.get("token_f1")),
                ph=_fmt(base.get("path_hit")),
            )
        )
    for name in ABLATIONS:
        s = summaries.get(name)
        if not s:
            continue
        delta = ""
        if base:
            delta = _fmt_signed(
                float(s.get("accuracy", 0.0)) - float(base.get("accuracy", 0.0))
            )
        lines.append(
            "| {label} | {acc} | {delta} | {gr} | {f1} | {ph} |".format(
                label=_ABLATION_LABELS.get(name, name),
                acc=_fmt(s.get("accuracy")),
                delta=delta,
                gr=_fmt(_groundedness(s)),
                f1=_fmt(s.get("token_f1")),
                ph=_fmt(s.get("path_hit")),
            )
        )
    return "\n".join(lines)


def _stats_section(run: EvalRun) -> str:
    lines = [
        "| So sánh | n | Acc baseline | Acc target | Δ = Ours − baseline (95% CI) | McNemar p |",
        "|---|---|---|---|---|---|",
    ]
    for baseline in ("B0", "B1", "B2"):
        comparison = compare_systems(run, baseline=baseline, target="ours")
        if not comparison.get("n_questions"):
            continue
        ci = comparison["accuracy_diff_ci"]
        lines.append(
            "| Ours vs {base} | {n} | {b} | {t} | {mean} [{lo}, {hi}] | {p} |".format(
                base=baseline,
                n=comparison["n_questions"],
                b=_fmt(comparison["baseline_accuracy"]),
                t=_fmt(comparison["target_accuracy"]),
                mean=_fmt_signed(ci["mean"]),
                lo=_fmt_signed(ci["ci_low"]),
                hi=_fmt_signed(ci["ci_high"]),
                p=_fmt(comparison["mcnemar"]["p_value"], 4),
            )
        )
    return "\n".join(lines)


def write_results_markdown(
    run: EvalRun,
    path: str | Path,
    comparisons: Sequence[dict[str, Any]] | None = None,
) -> Path:
    """Write ``results.md``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = run.meta or {}
    parts = [
        "# Kết quả đánh giá — Vietnamese History GraphRAG",
        "",
        f"- Số câu hỏi: **{meta.get('n_questions', len({r.qid for r in run.rows}))}**",
        f"- Số hệ thống: **{len({r.system for r in run.rows})}**",
        f"- Tổng số lượt chấm: **{len(run.rows)}**",
        "",
        "## 0. Cấu hình lượt chạy",
        "",
    ]
    config = meta.get("config") or {}
    if config:
        parts += [
            "| Tham số | Giá trị |",
            "|---|---|",
        ]
        for key, value in config.items():
            parts.append(f"| {key} | {value} |")
        parts.append("")
    n_degraded = int(meta.get("n_degraded", 0) or 0)
    if n_degraded:
        parts += [
            "> ⚠️ **Cảnh báo**: "
            f"{n_degraded}/{len(run.rows)} lượt chấm rơi về heuristic vì LLM judge "
            "không truy cập được (hết quota). Các điểm số của những dòng đó là "
            "token-F1 overlap, **không so sánh được** với điểm LLM của các dòng "
            "còn lại — cần chạy lại để có kết quả đáng tin cậy.",
            "",
        ]
    parts += [
        "> Các số liệu dưới đây được sinh tự động từ `artifacts/eval_run.json` "
        "bằng `scripts/07_run_eval.py`; có thể tái lập bằng "
        "`python scripts/07_run_eval.py`.",
        "",
        "## 1. Bảng chính (baseline vs Ours)",
        "",
        _main_table(run),
        "",
        "## 2. Bảng ablation",
        "",
        _ablation_table(run),
        "",
        "## 3. Thống kê (bootstrap CI 95% + McNemar)",
        "",
        _stats_section(run),
    ]
    if comparisons:
        parts += [
            "",
            "### Chi tiết so sánh",
            "",
            "```json",
            json.dumps(list(comparisons), ensure_ascii=False, indent=2),
            "```",
        ]
    parts.append("")
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def make_figures(run: EvalRun, out_dir: str | Path) -> list[Path]:
    """Draw the figures of docs/04 §5.3 (matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - figures are optional
        logger.warning("Không vẽ được biểu đồ (matplotlib thiếu?): %s", exc)
        return []

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = {s["system"]: s for s in summarize_run(run)}
    paths: list[Path] = []

    # -- 1. accuracy theo hệ thống x nhóm câu hỏi -------------------------
    systems = [s for s in ("B0", "B1", "B2", "ours") if s in summaries]
    groups = sorted(
        {g for s in summaries.values() for g in (s.get("by_type") or {})}
    )
    if systems and groups:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.8 / max(1, len(systems))
        for index, system in enumerate(systems):
            values = [
                float((summaries[system].get("by_type") or {}).get(g, {}).get("accuracy", 0.0))
                for g in groups
            ]
            positions = [i + index * width for i in range(len(groups))]
            ax.bar(positions, values, width=width, label=_SYSTEM_LABELS.get(system, system))
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(groups))])
        ax.set_xticklabels(groups)
        ax.set_ylabel("Accuracy (judge score / 2)")
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy theo hệ thống và nhóm câu hỏi")
        ax.legend()
        fig.tight_layout()
        path = out / "accuracy_by_group.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # -- 2. groundedness vs hallucination --------------------------------
    if systems:
        fig, ax = plt.subplots(figsize=(7, 4))
        grounded = [
            float(summaries[s].get("groundedness", 0.0))
            if summaries[s].get("groundedness_applicable", True)
            else float("nan")
            for s in systems
        ]
        halluc = [1.0 - g for g in grounded]
        positions = range(len(systems))
        ax.bar([p - 0.2 for p in positions], grounded, width=0.4, label="Groundedness")
        ax.bar([p + 0.2 for p in positions], halluc, width=0.4, label="Hallucination rate")
        ax.set_xticks(list(positions))
        ax.set_xticklabels([_SYSTEM_LABELS.get(s, s) for s in systems])
        ax.set_ylim(0, 1)
        ax.set_title("Groundedness vs hallucination rate")
        ax.legend()
        fig.tight_layout()
        path = out / "groundedness.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # -- 3. heatmap ablation ------------------------------------------------
    abl = [name for name in ABLATIONS if name in summaries]
    if abl and "ours" in summaries:
        base_acc = float(summaries["ours"].get("accuracy", 0.0))
        deltas = [
            float(summaries[name].get("accuracy", 0.0)) - base_acc for name in abl
        ]
        fig, ax = plt.subplots(figsize=(7, 2.6))
        image = ax.imshow([deltas], cmap="RdYlGn", vmin=-0.4, vmax=0.4, aspect="auto")
        ax.set_yticks([])
        ax.set_xticks(range(len(abl)))
        ax.set_xticklabels([_ABLATION_LABELS.get(a, a) for a in abl], rotation=20, ha="right")
        for index, delta in enumerate(deltas):
            ax.text(index, 0, f"{delta:+.3f}", ha="center", va="center")
        ax.set_title("Δ accuracy của ablation so với Ours")
        fig.colorbar(image, ax=ax, fraction=0.03)
        fig.tight_layout()
        path = out / "ablation_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # -- 4. phân bố số evidence --------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    systems_present = [s for s in systems]
    if systems_present:
        data = [
            [
                r.n_evidence
                for r in run.rows_of(s)
            ]
            for s in systems_present
        ]
        ax.boxplot(data, tick_labels=[_SYSTEM_LABELS.get(s, s) for s in systems_present])
        ax.set_ylabel("Số evidence dùng / câu")
        ax.set_title("Phân bố số evidence theo hệ thống")
    fig.tight_layout()
    path = out / "evidence_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


def build_report(
    run: EvalRun,
    reports_dir: str | Path,
    comparisons: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write every artefact of the report and return their paths."""
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = write_results_csv(run, reports / "results.csv")
    md_path = write_results_markdown(run, reports / "results.md", comparisons)
    figures = make_figures(run, reports / "figures")
    error_path = write_error_analysis(
        run,
        reports / "error_analysis.md",
        systems=[s for s in ("ours", "B1", "B0") if s in {r.system for r in run.rows}],
    )
    return {
        "results_csv": str(csv_path),
        "results_md": str(md_path),
        "error_analysis": str(error_path),
        "figures": [str(f) for f in figures],
    }


_SYSTEM_LABELS: dict[str, str] = {
    "B0": "B0 LLM-only",
    "B1": "B1 Vanilla RAG",
    "B2": "B2 RAG+Graph",
    "ours": "**Ours (Hybrid GraphRAG)**",
}

_ABLATION_LABELS: dict[str, str] = {
    "A1_no_entity_linking": "A1 bỏ entity linking",
    "A2_no_resolution": "A2 bỏ entity resolution",
    "A3_no_temporal": "A3 bỏ temporal boost/filter",
    "A4_no_graph": "A4 bỏ graph traversal",
    "A5_no_rerank": "A5 bỏ rerank",
    "A6_no_context_budget": "A6 bỏ context budget",
    "A7_no_fusion": "A7 bỏ fusion",
}
