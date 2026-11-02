"""LLM judge for the evaluation harness (docs/04 §3.1–§3.2).

The judge grades an answer against the reference answer and the gold facts
with a fixed rubric:

* ``score``       -- 0 (sai), 1 (đúng một phần), 2 (đúng và đầy đủ)
* ``correct``     -- ``score >= 2`` (the binary metric used by McNemar)
* ``grounded``    -- every claim of the answer is supported by the evidence
* ``supported_evidence_ids`` -- evidence ids that actually back a claim
* ``reason``      -- a short justification (used for the error analysis)

The judge is *cached* through the LLM client, so re-running the evaluation
costs nothing.  A deterministic heuristic fallback exists for offline runs.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import metrics as M

logger = logging.getLogger(__name__)

__all__ = [
    "JudgeResult",
    "LLMJudge",
    "HeuristicJudge",
    "build_judge",
    "is_out_of_kb",
    "looks_like_refusal",
]

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "correct": {"type": "boolean"},
        "grounded": {"type": "boolean"},
        "supported_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["score", "correct", "grounded", "reason"],
}

JUDGE_SYSTEM = """Bạn là giám khảo khách quan cho bài đánh giá hệ thống hỏi đáp lịch sử Việt Nam.

Bạn được cho: CÂU HỎI, ĐÁP ÁN THAM CHIẾU, GOLD FACTS, CÁC ĐÁP ÁN ỨNG VIÊN (có kèm trích dẫn [E#]) và CONTEXT.

Quy tắc chấm điểm (score):
- 2 = đáp án đúng VỀ NGHĨA và đầy đủ so với đáp án tham chiếu / gold facts.
- 1 = đáp án đúng một phần (thiếu chi tiết quan trọng, hoặc sai một phần nhỏ).
- 0 = đáp án sai, hoặc không liên quan, hoặc bịa đặt (hallucinate) thông tin không có trong CONTEXT.

Quy tắc grounded:
- grounded = true nếu mọi mệnh đề khẳng định trong đáp án đều được CONTEXT hỗ trợ.
- Nếu đáp án có thông tin không có trong CONTEXT thì grounded = false.

Quy tắc supported_evidence_ids:
- Liệt kê các mã evidence (ví dụ "E1") thực sự chứng minh mệnh đề trong đáp án.

Trả về DUY NHẤT một JSON object hợp lệ, không markdown fence.
"""

JUDGE_TEMPLATE = """CÂU HỎI:
{question}

ĐÁP ÁN THAM CHIẾU:
{reference}

GOLD FACTS:
{gold_facts}

CONTEXT (evidence được trích dẫn):
{context}

ĐÁP ÁN ỨNG VIÊN:
{answer}

Hãy chấm điểm theo rubric trên.
"""

#: Marker of a question that lies **outside** the knowledge base.  Such a
#: question has no reference answer; it is graded by *behaviour* instead:
#: the correct behaviour is to refuse, the wrong behaviour is to answer
#: (i.e. to hallucinate).  See docs/04 §2.1 (nguồn D).
OUT_OF_KB_MARKERS = ("out_of_kb", "out-of-kb", "ngoài phạm vi")

#: Phrases that signal a refusal in a Vietnamese answer.  The list is kept
#: deliberately **specific**: a generic phrase such as "không có thông tin"
#: also occurs in legitimate answers ("chưa có thông tin nào về năm này") and
#: would wrongly count them as a refusal.
REFUSAL_PHRASES = (
    "không tìm thấy đủ thông tin",
    "không đủ thông tin trong kho kiến thức",
    "không có trong kho kiến thức",
    "ngoài phạm vi",
    "không thuộc phạm vi",
    "tài liệu hiện có không",
    "không đủ căn cứ",
    "không đủ dữ liệu để trả lời",
)


def is_out_of_kb(question: Any) -> bool:
    """Whether ``question`` is an out-of-knowledge-base question.

    The flag is read from ``question_type`` first and from the source
    prefix second, so both the JSONL dataset and a bare string work.
    """
    if question is None:
        return False
    qtype = str(getattr(question, "question_type", "") or "").lower()
    if any(marker in qtype for marker in OUT_OF_KB_MARKERS):
        return True
    source = str(getattr(question, "source", "") or "").lower()
    if any(marker in source for marker in OUT_OF_KB_MARKERS):
        return True
    if isinstance(question, str):
        return any(marker in question.lower() for marker in OUT_OF_KB_MARKERS)
    return False


def looks_like_refusal(answer: str) -> bool:
    """Whether ``answer`` refuses to answer instead of inventing an answer.

    The detection reuses :func:`hgrag.qa.answer.is_refusal` so that the
    judge and the pipeline agree on what a refusal is.
    """
    from ..qa.answer import is_refusal

    if is_refusal(answer):
        return True
    text = (answer or "").lower()
    if not text.strip():
        return True
    return any(phrase in text for phrase in REFUSAL_PHRASES)


@dataclass
class JudgeResult:
    """The verdict of the judge on a single answer."""

    score: float = 0.0
    correct: bool = False
    grounded: bool = False
    supported_evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""
    source: str = "heuristic"

    @property
    def accuracy(self) -> float:
        """The 0-1 accuracy used in the comparison tables (``score / 2``)."""
        return float(self.score) / 2.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "correct": self.correct,
            "grounded": self.grounded,
            "accuracy": self.accuracy,
            "supported_evidence_ids": list(self.supported_evidence_ids),
            "reason": self.reason,
            "source": self.source,
        }


class LLMJudge:
    """Judge backed by an :class:`~hgrag.llm.client.LLMClient`."""

    def __init__(self, llm_client: Any = None) -> None:
        self.llm_client = llm_client
        self.calls = 0

    def grade(
        self,
        question: str,
        answer: str,
        reference: str = "",
        gold_facts: Sequence[str] | None = None,
        evidence: Sequence[Any] | None = None,
        out_of_kb: bool = False,
    ) -> JudgeResult:
        """Grade ``answer``.

        ``out_of_kb`` switches to the *behaviour* rubric: the correct
        answer is a refusal, and any attempt to answer the question is
        scored as a hallucination (docs/04 §2.1, nguồn D).
        """
        if out_of_kb:
            return _grade_out_of_kb(answer)
        context = _render_evidence(evidence)
        prompt = JUDGE_TEMPLATE.format(
            question=question,
            reference=reference or "(không có đáp án tham chiếu)",
            gold_facts="\n".join(f"- {f}" for f in (gold_facts or [])) or "(không có)",
            context=context or "(không có context)",
            answer=answer or "(không có đáp án)",
        )
        payload = self._ask(prompt)
        if payload is None:
            return HeuristicJudge().grade(
                question=question,
                answer=answer,
                reference=reference,
                gold_facts=gold_facts,
                evidence=evidence,
                out_of_kb=out_of_kb,
            )
        return _result_from_payload(payload)

    # ------------------------------------------------------------------
    def _ask(self, prompt: str) -> dict[str, Any] | None:
        if self.llm_client is None:
            return None
        from ..llm.schemas import LLMMessage, LLMRequest

        self.calls += 1
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=JUDGE_SYSTEM),
                LLMMessage(role="user", content=prompt),
            ],
            task="judge_answer",
            temperature=0.0,
            response_schema=JUDGE_SCHEMA,
        )
        try:
            payload = self.llm_client.generate_json(request)
        except Exception as exc:  # noqa: BLE001 - the judge must never crash the run
            logger.info("Judge thất bại (%s) -> dùng heuristic", exc)
            return None
        return payload if isinstance(payload, dict) else None


class HeuristicJudge:
    """Deterministic offline judge (token-F1 based)."""

    def grade(
        self,
        question: str,
        answer: str,
        reference: str = "",
        gold_facts: Sequence[str] | None = None,
        evidence: Sequence[Any] | None = None,
        out_of_kb: bool = False,
    ) -> JudgeResult:
        if out_of_kb:
            return _grade_out_of_kb(answer)
        f1 = M.token_f1(answer, reference)
        if reference:
            score = 2.0 if f1 >= 0.75 else (1.0 if f1 >= 0.35 else 0.0)
        else:
            score = 0.0
        markers = M.extract_citation_markers(answer)
        evidence_ids = [str(_evidence_id(e)) for e in (evidence or [])]
        supported = [m for m in markers if m in evidence_ids]
        grounded = bool(supported) or not markers
        return JudgeResult(
            score=score,
            correct=score >= 2,
            grounded=grounded,
            supported_evidence_ids=supported,
            reason=f"heuristic token-F1={f1:.3f}",
            source="heuristic",
        )


def build_judge(llm_client: Any = None, mock: bool = False) -> Any:
    """Pick the judge implementation for the environment."""
    if mock or llm_client is None:
        return HeuristicJudge()
    return LLMJudge(llm_client)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _grade_out_of_kb(answer: str) -> JudgeResult:
    """Grade an out-of-knowledge-base question by *behaviour*.

    The question has no reference answer: what matters is whether the
    system refused (correct) or invented an answer (hallucination).  A
    refusal is scored ``2`` (fully correct behaviour), an answer that
    clearly invents content is scored ``0``.
    """
    refused = looks_like_refusal(answer)
    if refused:
        return JudgeResult(
            score=2.0,
            correct=True,
            grounded=True,
            supported_evidence_ids=[],
            reason="out-of-KB: hệ thống từ chối trả lời (hành vi đúng)",
            source="behaviour",
        )
    return JudgeResult(
        score=0.0,
        correct=False,
        grounded=False,
        supported_evidence_ids=[],
        reason="out-of-KB: hệ thống cố trả lời câu hỏi ngoài phạm vi (hallucination)",
        source="behaviour",
    )


def _evidence_id(evidence: Any) -> str:
    if isinstance(evidence, dict):
        raw = evidence.get("evidence_id") or evidence.get("id") or ""
    else:
        raw = getattr(evidence, "evidence_id", "") or getattr(evidence, "id", "")
    return str(raw).strip().upper().replace("[", "").replace("]", "")


def _render_evidence(evidence: Sequence[Any] | None) -> str:
    if not evidence:
        return ""
    lines: list[str] = []
    for item in evidence:
        if isinstance(item, dict):
            eid = str(item.get("evidence_id") or item.get("id") or "")
            text = str(item.get("text") or "")
        else:
            eid = _evidence_id(item)
            text = str(getattr(item, "text", "") or "")
        text = re.sub(r"\s+", " ", text).strip()
        lines.append(f"[{eid}] {text[:600]}")
    return "\n\n".join(lines)


def _result_from_payload(payload: dict[str, Any]) -> JudgeResult:
    try:
        score = float(payload.get("score", 0) or 0)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(2.0, score))
    supported = [str(s).strip().upper() for s in (payload.get("supported_evidence_ids") or [])]
    return JudgeResult(
        score=score,
        correct=bool(payload.get("correct", score >= 2)),
        grounded=bool(payload.get("grounded", False)),
        supported_evidence_ids=supported,
        reason=str(payload.get("reason") or ""),
        source="llm",
    )


def _json_or_none(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None
