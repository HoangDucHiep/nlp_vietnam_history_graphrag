"""Evaluation harness (docs/04 §4–§5).

The harness runs every system (B0 / B1 / B2 / Ours + the 7 ablations) over the
evaluation dataset, scores every answer with the metric layer, and produces
the statistical comparisons (bootstrap CI + McNemar).

The design keeps the *same* LLM, the *same* prompt and the *same* corpus for
every system, so the difference measured is the retrieval architecture
(docs/04 §1.4).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import metrics as M
from .dataset import EvalDataset, EvalQuestion
from .judge import JudgeResult, build_judge, is_out_of_kb

logger = logging.getLogger(__name__)

__all__ = [
    "SYSTEMS",
    "ABLATIONS",
    "EvalRow",
    "EvalRun",
    "run_evaluation",
    "compare_systems",
]

#: The four systems compared in the report.
SYSTEMS: tuple[str, ...] = ("B0", "B1", "B2", "ours")

#: The seven ablations (docs/04 §4.2).
ABLATIONS: dict[str, str] = {
    "A1_no_entity_linking": "no_entity_linking",
    "A2_no_resolution": "no_resolution",
    "A3_no_temporal": "no_temporal",
    "A4_no_graph": "no_graph",
    "A5_no_rerank": "no_rerank",
    "A6_no_context_budget": "no_context_budget",
    "A7_no_fusion": "no_fusion",
}


@dataclass
class EvalRow:
    """One scored answer."""

    qid: str
    system: str
    question: str
    question_type: str = "single_hop"
    source: str = "unknown"
    answer: str = ""
    reference: str = ""
    refused: bool = False
    elapsed: float = 0.0

    judge_score: float = 0.0
    judge_correct: bool = False
    grounded: bool = False
    judge_reason: str = ""

    token_f1: float = 0.0
    rouge_l: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    chunk_recall_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    entity_recall: float = 0.0
    path_hit: float = 0.0
    temporal_accuracy: float = 0.0
    gold_fact_accuracy: float = 0.0
    n_evidence: int = 0
    n_citations: int = 0
    #: Whether the retrieval metrics of this row are *applicable*: a question
    #: without a gold path cannot be hit, and a question without a gold chunk
    #: cannot be recalled.  Such rows are excluded from the average of the
    #: corresponding metric instead of being counted as a miss.
    path_applicable: bool = False
    chunk_applicable: bool = False
    #: Whether the *groundedness* of this row is measurable.  The B0
    #: baseline answers closed-book: there is no context to check the claims
    #: against, so the groundedness of B0 is reported as "–" (n/a).
    grounded_applicable: bool = True
    linked_entities: list[str] = field(default_factory=list)
    retrieved_paths: list[list[str]] = field(default_factory=list)
    retrieved_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "system": self.system,
            "question": self.question,
            "question_type": self.question_type,
            "source": self.source,
            "answer": self.answer,
            "reference": self.reference,
            "refused": self.refused,
            "elapsed": round(self.elapsed, 4),
            "judge_score": self.judge_score,
            "judge_correct": self.judge_correct,
            "grounded": self.grounded,
            "judge_reason": self.judge_reason,
            "token_f1": round(self.token_f1, 6),
            "rouge_l": round(self.rouge_l, 6),
            "citation_precision": round(self.citation_precision, 6),
            "citation_recall": round(self.citation_recall, 6),
            "chunk_recall_at_10": round(self.chunk_recall_at_10, 6),
            "ndcg_at_10": round(self.ndcg_at_10, 6),
            "entity_recall": round(self.entity_recall, 6),
            "path_hit": round(self.path_hit, 6),
            "temporal_accuracy": round(self.temporal_accuracy, 6),
            "gold_fact_accuracy": round(self.gold_fact_accuracy, 6),
            "n_evidence": self.n_evidence,
            "n_citations": self.n_citations,
            "path_applicable": self.path_applicable,
            "chunk_applicable": self.chunk_applicable,
            "grounded_applicable": self.grounded_applicable,
            "linked_entities": list(self.linked_entities),
            "retrieved_paths": list(self.retrieved_paths),
            "retrieved_chunks": list(self.retrieved_chunks),
        }


@dataclass
class EvalRun:
    """The full result of an evaluation run."""

    rows: list[EvalRow] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def rows_of(self, system: str) -> list[EvalRow]:
        return [r for r in self.rows if r.system == system]

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": dict(self.meta),
            "rows": [r.to_dict() for r in self.rows],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def _answer_system(pipeline: Any, question: str, system: str, ablation: dict[str, bool]) -> Any:
    if system == "B0":
        return pipeline.answer_llm_only(question, system="B0")
    if system == "B1":
        return pipeline.answer_vanilla_rag(question, system="B1")
    if system == "B2":
        return pipeline.answer_rag_plus_graph(question, system="B2")
    return pipeline.ask(question, ablation=ablation, system=system)


def _score_row(
    question: EvalQuestion,
    system: str,
    result: Any,
    judge: Any,
) -> EvalRow:
    answer = str(getattr(result, "answer", "") or "")
    evidence = list(getattr(result, "evidence", []) or [])
    explanation = dict(getattr(result, "explanation", {}) or {})
    retrieval_trace = dict(explanation.get("retrieval_trace") or {})
    subgraph = dict(explanation.get("subgraph") or {})
    linked_entities = [
        str(e.get("name") or e.get("mention") or "")
        for e in (explanation.get("linked_entities") or [])
    ]
    retrieved_paths = [
        [str(n.get("name") or n.get("uid") or "") for n in (p.get("nodes") or [])]
        for p in (explanation.get("paths") or [])
    ]
    retrieved_chunks: list[str] = []
    evidence_chunk_map: dict[str, str] = {}
    for item in evidence:
        meta = item.get("meta") or {}
        chunk_id = str(meta.get("chunk_id") or "")
        eid = str(item.get("evidence_id") or "")
        if chunk_id:
            retrieved_chunks.append(chunk_id)
            if eid:
                evidence_chunk_map[eid] = chunk_id

    verdict: JudgeResult = judge.grade(
        question=question.question,
        answer=answer,
        reference=question.answer,
        gold_facts=list(question.gold_relations),
        evidence=evidence,
        out_of_kb=is_out_of_kb(question),
        closed_book=(system == "B0"),
    )

    markers = M.extract_citation_markers(answer)
    row = EvalRow(
        qid=question.qid,
        system=system,
        question=question.question,
        question_type=question.question_type,
        source=question.source,
        answer=answer,
        reference=question.answer,
        refused=bool(getattr(result, "refused", False)),
        elapsed=float(getattr(result, "elapsed", 0.0) or 0.0),
        judge_score=verdict.score,
        judge_correct=verdict.correct,
        grounded=verdict.grounded,
        judge_reason=verdict.reason,
        token_f1=M.token_f1(answer, question.answer),
        rouge_l=M.rouge_l(answer, question.answer),
        citation_precision=M.citation_precision(
            answer, list(evidence_chunk_map.keys()), verdict.supported_evidence_ids
        ),
        citation_recall=M.citation_recall(
            answer, question.gold_chunk_ids, evidence_chunk_map
        ),
        chunk_recall_at_10=M.chunk_recall_at_k(
            question.gold_chunk_ids, retrieved_chunks, k=10
        ),
        ndcg_at_10=M.ndcg_at_k(question.gold_chunk_ids, retrieved_chunks, k=10),
        entity_recall=M.entity_recall(question.gold_entities, linked_entities),
        path_hit=M.path_hit_rate(question.gold_paths, retrieved_paths),
        temporal_accuracy=M.temporal_accuracy(
            question.year_refs, [str(e.get("text") or "") for e in evidence]
        ),
        gold_fact_accuracy=M.gold_fact_accuracy(question.gold_relations, answer),
        n_evidence=len(evidence),
        n_citations=len(markers),
        path_applicable=bool(question.gold_paths),
        chunk_applicable=bool(question.gold_chunk_ids),
        grounded_applicable=(system != "B0"),
        linked_entities=linked_entities,
        retrieved_paths=retrieved_paths,
        retrieved_chunks=retrieved_chunks,
    )
    row.retrieved_chunks = retrieved_chunks
    _ = retrieval_trace, subgraph
    return row


def run_evaluation(
    pipeline: Any,
    dataset: EvalDataset | Sequence[EvalQuestion],
    systems: Sequence[str] = SYSTEMS,
    ablations: bool = True,
    judge: Any = None,
    limit: int | None = None,
    progress: bool = True,
    skip_existing: bool = True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 5,
) -> EvalRun:
    """Run every system over the dataset and score the answers.

    ``checkpoint_path`` makes the run resumable: after every
    ``checkpoint_every`` scored questions the partial rows are written to
    disk, and a re-run with the same path skips the (system, qid) pairs
    that are already present.  This matters a lot on free-tier LLM APIs
    where a long run can die on a 429 half-way through.
    """
    questions = list(dataset.questions if isinstance(dataset, EvalDataset) else dataset)
    if limit:
        questions = questions[: int(limit)]

    judge = judge or build_judge(getattr(pipeline, "llm_client", None))
    run = EvalRun(meta={"n_questions": len(questions), "systems": list(systems)})

    plan: list[tuple[str, dict[str, bool]]] = [(s, {}) for s in systems]
    if ablations:
        plan += [(name, {flag: True}) for name, flag in ABLATIONS.items()]

    done_keys: set[tuple[str, str]] = set()
    if skip_existing and checkpoint_path:
        done_keys = _load_checkpoint_rows(checkpoint_path, run)

    total = len(questions) * len(plan)
    done = 0
    for system, ablation in plan:
        for question in questions:
            done += 1
            key = (system, question.qid)
            if key in done_keys:
                continue
            if progress and done % 10 == 0:
                logger.info("[eval] %d/%d (%s)", done, total, system)
            try:
                result = _answer_system(pipeline, question.question, system, ablation)
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
                logger.warning("Câu %s (%s) lỗi: %s", question.qid, system, exc)
                continue
            try:
                row = _score_row(question, system, result, judge)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Chấm câu %s (%s) lỗi: %s", question.qid, system, exc)
                continue
            run.rows.append(row)
            if checkpoint_path and len(run.rows) % max(1, checkpoint_every) == 0:
                _save_checkpoint(checkpoint_path, run)
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, run)
    return run


def _load_checkpoint_rows(path: str | Path, run: EvalRun) -> set[tuple[str, str]]:
    """Load previously scored rows from ``path`` into ``run``."""
    source = Path(path)
    if not source.exists():
        return set()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Không đọc được checkpoint %s -- bắt đầu lại", source)
        return set()
    rows = payload.get("rows") or []
    # NB: the key must be (system, qid) -- a question is answered by every
    # system, so de-duplicating by qid alone would silently drop rows.
    known = {(r.system, r.qid) for r in run.rows}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        qid = str(raw.get("qid") or "")
        system = str(raw.get("system") or "")
        if not qid or not system or (system, qid) in known:
            continue
        run.rows.append(EvalRow(**{k: v for k, v in raw.items() if k in EvalRow.__annotations__}))
        known.add((system, qid))
    logger.info("Checkpoint: đã có %d dòng kết quả", len(run.rows))
    return {(str(r.get("system")), str(r.get("qid"))) for r in rows}


def _save_checkpoint(path: str | Path, run: EvalRun) -> None:
    """Persist the rows of ``run`` so an interrupted run can be resumed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"meta": run.meta, "rows": [r.to_dict() for r in run.rows]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(target)


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def _system_scores(run: EvalRun, system: str, metric: str = "judge_correct") -> dict[str, float]:
    return {r.qid: (1.0 if getattr(r, metric) else 0.0) for r in run.rows_of(system)}


def compare_systems(
    run: EvalRun,
    baseline: str = "B1",
    target: str = "ours",
    n_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI + McNemar between ``baseline`` and ``target``."""
    qids = sorted(
        {r.qid for r in run.rows_of(baseline)} & {r.qid for r in run.rows_of(target)}
    )
    base_rows = {r.qid: r for r in run.rows_of(baseline)}
    target_rows = {r.qid: r for r in run.rows_of(target)}

    base_binary = [1.0 if base_rows[q].judge_correct else 0.0 for q in qids]
    target_binary = [1.0 if target_rows[q].judge_correct else 0.0 for q in qids]

    diff_ci = M.paired_bootstrap_ci(
        target_binary,
        base_binary,
        n_samples=n_samples,
        confidence=confidence,
        seed=seed,
    )
    mcnemar = M.mcnemar_test(
        [bool(v) for v in target_binary],
        [bool(v) for v in base_binary],
    )
    acc_ci = M.bootstrap_ci(
        target_binary, n_samples=n_samples, confidence=confidence, seed=seed
    )
    base_acc_ci = M.bootstrap_ci(
        base_binary, n_samples=n_samples, confidence=confidence, seed=seed
    )
    return {
        "baseline": baseline,
        "target": target,
        "n_questions": len(qids),
        "baseline_accuracy": (sum(base_binary) / len(base_binary)) if base_binary else 0.0,
        "target_accuracy": (sum(target_binary) / len(target_binary)) if target_binary else 0.0,
        "baseline_accuracy_ci": base_acc_ci.to_dict(),
        "target_accuracy_ci": acc_ci.to_dict(),
        "accuracy_diff_ci": diff_ci.to_dict(),
        "mcnemar": mcnemar.to_dict(),
    }


def summarize_system(run: EvalRun, system: str) -> dict[str, Any]:
    """Aggregate the metrics of one system (overall + per question group)."""
    rows = run.rows_of(system)
    if not rows:
        return {"system": system, "n": 0}
    metrics = (
        "judge_score",
        "token_f1",
        "rouge_l",
        "citation_precision",
        "citation_recall",
        "chunk_recall_at_10",
        "ndcg_at_10",
        "entity_recall",
        "path_hit",
        "temporal_accuracy",
        "gold_fact_accuracy",
    )
    payload: dict[str, Any] = {
        "system": system,
        "n": len(rows),
        "accuracy": sum(1.0 if r.judge_correct else 0.0 for r in rows) / len(rows),
        "groundedness": _mean_applicable(rows, "grounded", "grounded_applicable"),
        "groundedness_applicable": any(r.grounded_applicable for r in rows),
        "refusal_rate": sum(1.0 if r.refused else 0.0 for r in rows) / len(rows),
        "no_citation_rate": M.no_citation_rate([r.answer for r in rows]),
        "latency_p50": M.percentile([r.elapsed for r in rows], 50),
        "latency_p95": M.percentile([r.elapsed for r in rows], 95),
        "avg_n_evidence": sum(r.n_evidence for r in rows) / len(rows),
    }
    payload.update(M.aggregate([r.to_dict() for r in rows], metrics))
    # The retrieval metrics only apply to the questions that *have* a gold
    # path / a gold chunk.  Averaging them over all the rows would count
    # every inapplicable question as a miss and deflate the metric.
    payload["path_hit"] = _mean_applicable(rows, "path_hit", "path_applicable")
    payload["chunk_recall_at_10"] = _mean_applicable(
        rows, "chunk_recall_at_10", "chunk_applicable"
    )
    payload["ndcg_at_10"] = _mean_applicable(rows, "ndcg_at_10", "chunk_applicable")
    payload["n_path_applicable"] = sum(1 for r in rows if r.path_applicable)
    payload["n_chunk_applicable"] = sum(1 for r in rows if r.chunk_applicable)

    by_type: dict[str, Any] = {}
    for qtype in sorted({r.question_type for r in rows}):
        subset = [r for r in rows if r.question_type == qtype]
        by_type[qtype] = {
            "n": len(subset),
            "accuracy": sum(1.0 if r.judge_correct else 0.0 for r in subset) / len(subset),
            "groundedness": _mean_applicable(subset, "grounded", "grounded_applicable"),
            "groundedness_applicable": any(r.grounded_applicable for r in subset),
            "token_f1": sum(r.token_f1 for r in subset) / len(subset),
            "path_hit": _mean_applicable(subset, "path_hit", "path_applicable"),
        }
    payload["by_type"] = by_type
    return payload


def _mean_applicable(rows: Sequence[EvalRow], metric: str, flag: str) -> float:
    """Mean of ``metric`` over the rows where ``flag`` is true."""
    values = [float(getattr(r, metric)) for r in rows if getattr(r, flag, False)]
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_run(run: EvalRun) -> list[dict[str, Any]]:
    """Summaries of every system present in the run (ordered)."""
    order = list(SYSTEMS) + [name for name in ABLATIONS if name not in SYSTEMS]
    seen = {r.system for r in run.rows}
    systems = [s for s in order if s in seen] + sorted(seen - set(order))
    return [summarize_system(run, s) for s in systems]
