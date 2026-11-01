"""Grounded answer generation with citation markers and reasoning chain.

The generator is *grounded*: it may only use the evidence handed to it, and
every factual statement must carry a ``[E#]`` marker.  When the evidence is
insufficient the model is instructed to refuse instead of hallucinating.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: ``[E1]``, ``[E12]`` -- citation markers.
CITATION_RE = re.compile(r"\[E(\d+)\]", re.IGNORECASE)

#: Phrases that signal a refusal.
REFUSAL_MARKERS = (
    "không đủ thông tin",
    "không tìm thấy đủ thông tin",
    "không có thông tin",
    "tôi không thể trả lời",
    "không thể trả lời",
    "ngoài phạm vi",
    "insufficient information",
)

#: Headings the model may use for the reasoning chain.
_REASONING_HEADINGS = (
    "## suy luận",
    "### suy luận",
    "## reasoning",
    "### reasoning",
    "## lập luận",
    "## chain of thought",
)

_STEP_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)\s*$")


SYSTEM_PROMPT = """Bạn là chuyên gia lịch sử Việt Nam, trả lời câu hỏi dựa TRÊN DUY NHẤT ngữ cảnh được cung cấp.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong CONTEXT. TUYỆT ĐỐI không bịa thông tin ngoài CONTEXT.
2. Sau mỗi mệnh đề khẳng định phải gắn nguồn trích dẫn dạng [E1], [E2]... trúng với số hiệu evidence trong CONTEXT.
3. Nếu CONTEXT không đủ để trả lời, hãy trả lời đúng câu:
   "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này."
   và KHÔNG suy đoán.
4. Trả lời bằng tiếng Việt, súc tích, đúng mốc thời gian và địa danh.
5. Kết thúc bằng một mục "## Suy luận" gồm các bước suy luận đánh số (1., 2., 3.) ngắn gọn, mỗi bước có trích dẫn.
"""

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

CÂU HỎI: {question}

Hãy trả lời theo quy tắc ở trên.
{plan}"""


@dataclass
class AnswerResult:
    """The generated answer and its metadata."""

    answer: str = ""
    citations: list[str] = field(default_factory=list)
    used_evidence_ids: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    refused: bool = False
    reasoning_chain: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "used_evidence_ids": list(self.used_evidence_ids),
            "invalid_citations": list(self.invalid_citations),
            "refused": self.refused,
            "reasoning_chain": list(self.reasoning_chain),
            "raw_response": self.raw_response,
        }


def validate_citations(
    answer: str, evidence_ids: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Split the citation markers of ``answer`` into (valid, invalid).

    ``evidence_ids`` are the identifiers actually available (``E1``, ``E2``...).
    """
    available = {str(e).strip().upper() for e in evidence_ids or []}
    valid: list[str] = []
    invalid: list[str] = []
    for match in CITATION_RE.finditer(answer or ""):
        marker = f"E{match.group(1)}"
        if marker in available:
            if marker not in valid:
                valid.append(marker)
        elif marker not in invalid:
            invalid.append(marker)
    return valid, invalid


def is_refusal(text: str) -> bool:
    """Whether the answer is a refusal."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def parse_reasoning_chain(response: str) -> list[str]:
    """Extract the reasoning steps from a model response."""
    text = response or ""
    lowered = text.lower()
    cut = len(text)
    for heading in _REASONING_HEADINGS:
        idx = lowered.find(heading)
        if idx != -1:
            cut = min(cut, idx)
    tail = text[cut:]
    steps: list[str] = []
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _STEP_RE.match(line)
        if match:
            steps.append(match.group(1).strip())
    if steps:
        return steps
    # Fallback: numbered lines anywhere in the response.
    for line in text.splitlines():
        match = _STEP_RE.match(line.strip())
        if match and match.group(1).strip():
            steps.append(match.group(1).strip())
    return steps


class AnswerGenerator:
    """Generate a grounded, cited answer."""

    def __init__(self, llm_client: Any = None, settings: Any = None) -> None:
        self.llm_client = llm_client
        self.settings = settings

    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        context_text: str,
        evidence: Sequence[Any] | None = None,
        understanding: Any = None,
    ) -> AnswerResult:
        """Answer ``question`` using ``context_text`` only."""
        evidence_ids = [_evidence_id(e) for e in (evidence or [])]
        evidence_ids = [e for e in evidence_ids if e]

        if not context_text.strip():
            return AnswerResult(
                answer=(
                    "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời "
                    "câu hỏi này."
                ),
                refused=True,
                used_evidence_ids=[],
            )

        prompt = self._build_prompt(question, context_text, understanding)
        raw = ""
        if self.llm_client is not None:
            from ..llm.schemas import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[
                    LLMMessage(role="system", content=SYSTEM_PROMPT),
                    LLMMessage(role="user", content=prompt),
                ],
                task="answer_grounded",
                temperature=0.0,
            )
            try:
                raw = self.llm_client.generate(request).text
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sinh câu trả lời thất bại: %s", exc)
                raw = ""

        if not raw.strip():
            return AnswerResult(
                answer=(
                    "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời "
                    "câu hỏi này."
                ),
                refused=True,
                used_evidence_ids=[],
            )

        answer_text = self._strip_reasoning(raw)
        valid, invalid = validate_citations(answer_text, evidence_ids)
        refused = is_refusal(answer_text)
        chain = parse_reasoning_chain(raw)
        if not chain:
            chain = parse_reasoning_chain(answer_text)
        return AnswerResult(
            answer=answer_text.strip(),
            citations=valid,
            used_evidence_ids=valid,
            invalid_citations=invalid,
            refused=refused,
            reasoning_chain=chain,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(question: str, context_text: str, understanding: Any) -> str:
        plan = ""
        if understanding is not None:
            intent = getattr(understanding, "intent", "")
            hops = getattr(understanding, "expected_hops", 0)
            if intent:
                plan = (
                    f"\n(Gợi ý phân tích: loại câu hỏi = {intent}, "
                    f"số bước suy luận dự kiến = {hops}.)\n"
                )
        return USER_PROMPT_TEMPLATE.format(
            context=context_text.strip(),
            question=question.strip(),
            plan=plan,
        )

    @staticmethod
    def _strip_reasoning(response: str) -> str:
        """Remove the reasoning block from the answer body."""
        text = response or ""
        lowered = text.lower()
        cut = len(text)
        for heading in _REASONING_HEADINGS:
            idx = lowered.find(heading)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut].strip()


def _evidence_id(evidence: Any) -> str:
    """The ``E#`` identifier of an evidence object."""
    if isinstance(evidence, dict):
        raw = evidence.get("evidence_id") or evidence.get("id") or ""
    else:
        raw = getattr(evidence, "evidence_id", "") or getattr(evidence, "id", "")
    return str(raw).strip().upper().replace("[", "").replace("]", "")
