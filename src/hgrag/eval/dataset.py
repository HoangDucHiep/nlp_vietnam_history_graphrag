"""Evaluation dataset contracts (docs/04_danh_gia.md).

This module defines the *stable* data contract shared with the Wave-5 metric
code.  Everything the metric layer needs (gold answer, gold entities, gold
paths, gold chunks, gold relations, year references) lives on
:class:`EvalQuestion`; the container :class:`EvalDataset` groups the
questions and can report the distribution used by ``scripts/05_eval_dataset.py``.

The contract is intentionally plain (dataclass + JSONL) so that a human can
review / edit the dataset by hand (see docs/04 §2.3 "human-verify").
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "QUESTION_TYPES",
    "DIFFICULTIES",
    "QUESTION_SOURCES",
    "EvalQuestion",
    "EvalDataset",
    "save_jsonl",
    "load_jsonl",
]

#: The four question groups of the evaluation set (docs/04 §2.1).
QUESTION_TYPES: tuple[str, ...] = ("single_hop", "multi_hop", "temporal", "out_of_kb")

#: Difficulty labels attached to a question.
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

#: Provenance of a question (which generator produced it).
QUESTION_SOURCES: tuple[str, ...] = (
    "vhe_template",
    "llm_from_wiki",
    "history_chat",
    "out_of_kb",
)


@dataclass
class EvalQuestion:
    """One evaluation question with its gold evidence.

    Attributes
    ----------
    qid:
        Stable identifier, unique inside a dataset.
    question:
        The question text (Vietnamese).
    answer:
        Reference (gold) answer.  Empty for ``out_of_kb`` questions -- those
        are scored on the *refusal behaviour* instead (docs/04 §3.2).
    question_type:
        One of :data:`QUESTION_TYPES`.
    gold_entities:
        Surface names of the entities that must appear in a correct answer.
    gold_qids:
        Wikidata QIDs of ``gold_entities`` when known.
    gold_paths:
        Expected traversal paths, each a list of node names (>= 2 nodes for a
        meaningful multi-hop path).
    gold_chunk_ids:
        Chunk ids that support the answer.
    gold_relations:
        Triples written as ``"A -REL-> B"``.
    year_refs:
        Years mentioned by / relevant for the question.
    difficulty:
        ``"easy"`` | ``"medium"`` | ``"hard"``.
    source:
        Which generator produced the question (see :data:`QUESTION_SOURCES`).
    metadata:
        Free-form payload (excluded chunks, prompts, notes, ...).
    """

    qid: str
    question: str
    answer: str = ""
    question_type: str = "single_hop"
    gold_entities: list[str] = field(default_factory=list)
    gold_qids: list[str] = field(default_factory=list)
    gold_paths: list[list[str]] = field(default_factory=list)
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_relations: list[str] = field(default_factory=list)
    year_refs: list[int] = field(default_factory=list)
    difficulty: str = "medium"
    source: str = "vhe_template"
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the question."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalQuestion:
        """Rebuild a question from :meth:`to_dict` output."""
        payload = dict(d or {})
        known = {
            "qid",
            "question",
            "answer",
            "question_type",
            "gold_entities",
            "gold_qids",
            "gold_paths",
            "gold_chunk_ids",
            "gold_relations",
            "year_refs",
            "difficulty",
            "source",
            "metadata",
        }
        kwargs = {k: v for k, v in payload.items() if k in known}
        extra = {k: v for k, v in payload.items() if k not in known}
        obj = cls(**kwargs)
        if extra:
            meta = dict(obj.metadata or {})
            meta.update(extra)
            obj.metadata = meta
        return obj

    # -- helpers --------------------------------------------------------
    @property
    def is_out_of_kb(self) -> bool:
        """Whether the question lies outside the knowledge base."""
        return self.question_type == "out_of_kb"

    @property
    def has_evidence(self) -> bool:
        """Whether the question carries at least one gold evidence anchor."""
        return bool(self.gold_entities or self.gold_chunk_ids)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> EvalQuestion:
        return cls.from_dict(json.loads(line))


@dataclass
class EvalDataset:
    """A collection of :class:`EvalQuestion` objects."""

    questions: list[EvalQuestion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- container protocol ---------------------------------------------
    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def __getitem__(self, index: int) -> EvalQuestion:
        return self.questions[index]

    def add(self, question: EvalQuestion) -> None:
        self.questions.append(question)

    def extend(self, questions: list[EvalQuestion]) -> None:
        self.questions.extend(questions)

    # -- persistence ----------------------------------------------------
    def save_jsonl(self, path: str | Path) -> int:
        """Write the dataset as JSONL; returns the number of lines written."""
        return save_jsonl(self.questions, path)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> EvalDataset:
        """Read a dataset previously written with :meth:`save_jsonl`."""
        questions = load_jsonl(path)
        return cls(questions=questions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": [q.to_dict() for q in self.questions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalDataset:
        questions = [EvalQuestion.from_dict(q) for q in (d.get("questions") or [])]
        return cls(questions=questions, metadata=dict(d.get("metadata") or {}))

    # -- statistics -----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Distribution of the dataset by type / source / difficulty."""
        by_type = Counter(q.question_type for q in self.questions)
        by_source = Counter(q.source for q in self.questions)
        by_difficulty = Counter(q.difficulty for q in self.questions)
        total = len(self.questions)
        out_of_kb = by_type.get("out_of_kb", 0)
        with_evidence = sum(1 for q in self.questions if q.has_evidence)
        multi_hop = by_type.get("multi_hop", 0)
        multi_hop_with_path = sum(
            1
            for q in self.questions
            if q.question_type == "multi_hop"
            and any(len(p) >= 2 for p in q.gold_paths)
        )
        answer_lengths = [len(q.answer) for q in self.questions if q.answer]
        return {
            "total": total,
            "by_type": dict(sorted(by_type.items())),
            "by_source": dict(sorted(by_source.items())),
            "by_difficulty": dict(sorted(by_difficulty.items())),
            "by_type_ratio": {
                k: (v / total if total else 0.0) for k, v in sorted(by_type.items())
            },
            "out_of_kb": out_of_kb,
            "out_of_kb_ratio": (out_of_kb / total) if total else 0.0,
            "with_evidence": with_evidence,
            "without_evidence": total - with_evidence,
            "multi_hop": multi_hop,
            "multi_hop_with_path": multi_hop_with_path,
            "avg_answer_chars": (sum(answer_lengths) / len(answer_lengths))
            if answer_lengths
            else 0.0,
        }


def save_jsonl(questions: list[EvalQuestion], path: str | Path) -> int:
    """Write questions to ``path`` (one JSON object per line)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with target.open("w", encoding="utf-8") as fh:
        for question in questions:
            fh.write(question.to_json())
            fh.write("\n")
            n += 1
    return n


def load_jsonl(path: str | Path) -> list[EvalQuestion]:
    """Read questions from a JSONL file (malformed lines are skipped)."""
    source = Path(path)
    if not source.exists():
        return []
    questions: list[EvalQuestion] = []
    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            questions.append(EvalQuestion.from_dict(payload))
    return questions
