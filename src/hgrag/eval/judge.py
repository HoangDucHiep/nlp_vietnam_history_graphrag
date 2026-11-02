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
import time
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

Quy tắc về độ dài:
- Trường "reason" chỉ ghi TỐI ĐA 2 câu ngắn (khoảng 40 từ). Không lặp lại câu hỏi, không trích lại CONTEXT.

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

#: Rubric cho baseline **B0 (closed-book)**: không có CONTEXT nên không thể
#: chấm "grounded" (không có nguồn để đối chiếu).  Chỉ chấm độ đúng so với
#: đáp án tham chiếu / gold facts.  Cột groundedness của B0 sẽ được báo cáo
#: là "–" (không áp dụng).
CLOSED_BOOK_JUDGE_TEMPLATE = """CÂU HỎI:
{question}

ĐÁP ÁN THAM CHIẾU:
{reference}

GOLD FACTS:
{gold_facts}

ĐÁP ÁN ỨNG VIÊN (trả lời closed-book, KHÔNG có tài liệu tham chiếu):
{answer}

Hãy chấm điểm theo rubric trên (chỉ chấm độ đúng; đặt grounded = true).
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
    #: ``True`` when the judge could *not* reach the LLM and the verdict fell
    #: back to the deterministic heuristic.  Such a verdict is much weaker
    #: (it is a token-F1 overlap, not a semantic grading), so the evaluation
    #: harness reports it instead of silently mixing the two kinds of score.
    degraded: bool = False

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

    def __init__(
        self,
        llm_client: Any = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ) -> None:
        self.llm_client = llm_client
        self.calls = 0
        self.max_retries = max(1, int(max_retries))
        self.retry_base_delay = float(retry_base_delay)

    def grade(
        self,
        question: str,
        answer: str,
        reference: str = "",
        gold_facts: Sequence[str] | None = None,
        evidence: Sequence[Any] | None = None,
        out_of_kb: bool = False,
        closed_book: bool = False,
    ) -> JudgeResult:
        """Grade ``answer``.

        ``out_of_kb`` switches to the *behaviour* rubric: the correct
        answer is a refusal, and any attempt to answer the question is
        scored as a hallucination (docs/04 §2.1, nguồn D).

        ``closed_book`` marks the B0 baseline: there is no context to check
        the answer against, so only the correctness is graded.
        """
        if out_of_kb:
            return _grade_out_of_kb(answer)
        context = _render_evidence(evidence)
        template = CLOSED_BOOK_JUDGE_TEMPLATE if closed_book else JUDGE_TEMPLATE
        prompt = template.format(
            question=question,
            reference=reference or "(không có đáp án tham chiếu)",
            gold_facts="\n".join(f"- {f}" for f in (gold_facts or [])) or "(không có)",
            context=context or "(không có context)",
            answer=answer or "(không có đáp án)",
        )
        payload = self._ask(prompt)
        if payload is None:
            fallback = HeuristicJudge().grade(
                question=question,
                answer=answer,
                reference=reference,
                gold_facts=gold_facts,
                evidence=evidence,
                out_of_kb=out_of_kb,
                closed_book=closed_book,
            )
            fallback.degraded = True
            return fallback
        result = _result_from_payload(payload)
        if result is None:
            # The JSON was repaired but the rubric fields are missing: the
            # verdict would be meaningless, so the row is marked as degraded.
            fallback = HeuristicJudge().grade(
                question=question,
                answer=answer,
                reference=reference,
                gold_facts=gold_facts,
                evidence=evidence,
                out_of_kb=out_of_kb,
                closed_book=closed_book,
            )
            fallback.degraded = True
            return fallback
        if closed_book:
            result.grounded = True
        return result

    # ------------------------------------------------------------------
    def _ask(self, prompt: str) -> dict[str, Any] | None:
        """Ask the judge, retrying on transient failures.

        A single failed call must not silently degrade the verdict to the
        token-F1 heuristic: mixing the two kinds of score in one table makes
        the comparison between the systems meaningless.  The client already
        handles the per-minute quota internally (it waits and retries on the
        *same* model), so a failure here is a real one -- but a couple of
        retries still absorb the transient 500s of the free tier.
        """
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
            # The verdict only needs a handful of fields.  The budget is kept
            # small on purpose: with a ``response_schema`` the Gemini API
            # enforces the schema and **does not** run the model's reasoning
            # channel, so the whole budget is available for the JSON body
            # (measured in docs/10_quota_gemini_free_tier.md).  Without the
            # schema, Gemma would spend the budget on reasoning and the
            # response would come back empty.
            max_output_tokens=512,
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                # The ``task`` is part of the cache key.  Varying it on a
                # retry makes sure a *cached* bad verdict is not returned
                # again and again (the same request would hit the cache).
                request.task = "judge_answer" if attempt == 0 else f"judge_answer#{attempt}"
                payload = self.llm_client.generate_json(request)
            except Exception as exc:  # noqa: BLE001 - the judge must never crash the run
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    delay = self.retry_base_delay * (2**attempt)
                    logger.info(
                        "Judge thất bại (lần %d/%d): %s -- thử lại sau %.1fs",
                        attempt + 1,
                        self.max_retries,
                        str(exc)[:120],
                        delay,
                    )
                    time.sleep(delay)
                    continue
                logger.warning("Judge thất bại (%s) -> dùng heuristic", exc)
                return None
            if isinstance(payload, dict):
                if not _is_self_consistent(payload):
                    # The verdict contradicts itself (see the helper): the
                    # score cannot be trusted, so the row is judged again.
                    logger.warning(
                        "Judge trả về payload tự mâu thuẫn (lần %d/%d) -- thử lại",
                        attempt + 1,
                        self.max_retries,
                    )
                    if attempt + 1 < self.max_retries:
                        time.sleep(self.retry_base_delay * (2**attempt))
                        continue
                    return None
                return payload
            logger.info("Judge trả về payload không phải dict -> thử lại")
        _ = last_exc
        return None


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
        closed_book: bool = False,
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
        if closed_book:
            grounded = True
        return JudgeResult(
            score=score,
            correct=score >= 2,
            grounded=grounded,
            supported_evidence_ids=supported,
            reason=f"heuristic token-F1={f1:.3f}",
            source="heuristic",
            degraded=True,
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


def _result_from_payload(payload: dict[str, Any]) -> JudgeResult | None:
    """Build a verdict from the judge's payload.

    ``None`` is returned when the payload does not carry the fields of the
    rubric: a partially repaired JSON must not silently become a score of 0.
    """
    if not isinstance(payload, dict):
        return None
    if "score" not in payload:
        return None
    try:
        score = float(payload.get("score", 0) or 0)
    except (TypeError, ValueError):
        return None
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


#: Phrases with which the judge *praises* the candidate answer.
_PRAISE_MARKERS = (
    "đúng về nghĩa",
    "đúng hoàn toàn",
    "đúng và đầy đủ",
    "đúng và chính xác",
    "chính xác và đầy đủ",
    "trả lời đúng",
)

_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def _is_self_consistent(payload: dict[str, Any]) -> bool:
    """Whether a judge verdict is internally consistent.

    Gemma occasionally emits a verdict that contradicts itself: the ``reason``
    praises the answer ("Đáp án đúng về nghĩa và đầy đủ...") while ``score``
    is ``0`` -- measured on 4 of 471 cached verdicts (0.85 %).  Such a verdict
    cannot be used: it would silently put a correct answer into the "wrong"
    bucket of the report.

    Two kinds of defect are detected:

    * **self-contradiction** -- the reason praises the answer but the score is
      below 1 (or the other way round: the reason condemns it but the score
      is 2);
    * **degenerate repetition** -- the reason repeats the same sentence, the
      typical symptom of a model that ran out of budget mid-sentence.
    """
    reason = str(payload.get("reason") or "")
    if not reason.strip():
        return False
    try:
        score = float(payload.get("score", 0) or 0)
    except (TypeError, ValueError):
        return False
    lowered = reason.lower()
    praise = any(marker in lowered for marker in _PRAISE_MARKERS)
    if praise and score < 1.0:
        return False
    if not praise and score >= 2.0 and "sai" not in lowered:
        # A perfect score without any praise in the reason is suspicious but
        # not contradictory on its own -- it is allowed.
        pass
    # Degenerate repetition: the same sentence appears at least twice.
    sentences = [s.strip().lower() for s in _SENTENCE_RE.findall(reason) if len(s.strip()) > 12]
    if sentences:
        counts: dict[str, int] = {}
        for sentence in sentences:
            counts[sentence] = counts.get(sentence, 0) + 1
        if max(counts.values()) >= 2:
            return False
    return True
