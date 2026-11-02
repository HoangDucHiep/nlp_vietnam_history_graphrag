"""Grounded answer generation with citation markers and reasoning chain.

The generator is *grounded*: it may only use the evidence handed to it, and
every factual statement must carry a ``[E#]`` marker.  When the evidence is
insufficient the model is instructed to refuse instead of hallucinating.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..llm.client import LLMClient

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

#: A word of the Vietnamese text (used by the repetition detector).
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: The exact refusal sentence the pipeline uses everywhere.
REFUSAL_ANSWER = (
    "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này."
)


def is_degenerate(text: str, min_words: int = 12) -> bool:
    """Whether ``text`` is a degenerate (repetitive) model output.

    Gemma 4 sometimes falls into a repetition loop when it is pushed into a
    narrow output format: it repeats a word ("phất phất phất ..."), a phrase
    ("theo truyền thuyết, theo truyền thuyết, ...") or a whole sentence
    until the budget is exhausted.  Such an output is *not* an answer -- it
    would be graded as a hallucination by the judge, so it must be detected
    and replaced by an honest refusal.

    Three signals are used:

    * the same **word** repeated at least 6 times in a row;
    * the same **n-gram** (2 to 5 words) repeated at least 4 times;
    * the same **line** repeated at least 3 times.

    >>> is_degenerate("phất phất phất phất phất phất")
    True
    >>> is_degenerate("Trần Hưng Đạo chỉ huy quân tại trận Bạch Đằng năm 1285.")
    False
    """
    body = (text or "").strip()
    if not body:
        return False
    words = [w.casefold() for w in _WORD_RE.findall(body)]
    # (1) the same word repeated in a row
    run = 1
    for index in range(1, len(words)):
        if words[index] == words[index - 1]:
            run += 1
            if run >= 6:
                return True
        else:
            run = 1
    # (2) the same n-gram repeated
    if len(words) >= min_words:
        for size in (5, 4, 3, 2):
            if len(words) < size * 4:
                continue
            # The shorter the n-gram, the more repetitions are needed: a
            # two-word repetition happens naturally in a list of citations.
            threshold = 4 if size >= 4 else 5
            counts: dict[tuple[str, ...], int] = {}
            for index in range(len(words) - size + 1):
                gram = tuple(words[index : index + size])
                counts[gram] = counts.get(gram, 0) + 1
                if counts[gram] >= threshold:
                    return True
    # (3) the same line repeated
    lines = [line.strip().casefold() for line in body.splitlines() if line.strip()]
    if lines:
        line_counts: dict[str, int] = {}
        for line in lines:
            line_counts[line] = line_counts.get(line, 0) + 1
            if line_counts[line] >= 3:
                return True
    return False


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


#: Prompt của baseline **B0 (closed-book)**: không có CONTEXT, mô hình phải
#: trả lời bằng kiến thức tham số của chính nó.  Đây là baseline bắt buộc
#: theo docs/04 §4.1 (B0 = LLM-only, "chỉ kiến thức mô hình").
#:
#: Đo đạc (`scripts/_probe_closedbook_escape.py`, xem docs/10) cho thấy Gemma 4
#: **không kết thúc** phần suy luận khi được hỏi một câu mà nó không biết:
#: token suy luận dùng hết toàn bộ ngân sách (256 → 8192) và phần trả lời
#: trả về **rỗng**.  Nguyên nhân là bản prompt cũ *không* nói rằng "không
#: biết" là một câu trả lời hợp lệ -- mô hình cứ suy luận để tìm câu trả lời
#: mà không tồn tại.  Vì vậy prompt mới cho mô hình một **lối thoát** rõ
#: ràng, đồng thời giữ nguyên ngữ nghĩa của baseline: không CONTEXT, không
#: trích dẫn, không bịa đặt.
CLOSED_BOOK_SYSTEM_PROMPT = """Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp.

Quy tắc bắt buộc:
1. Trả lời trực tiếp, súc tích, bằng tiếng Việt, đúng mốc thời gian và địa danh.
2. KHÔNG gắn trích dẫn [E#] vì không có nguồn nào được cung cấp.
3. Nếu bạn KHÔNG CHẮC CHẮN hoặc không biết, hãy trả lời đúng câu: "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này." Đây là một câu trả lời HỢP LỆ, không phải lỗi.
4. KHÔNG suy luận dài, KHÔNG tự hỏi lại người dùng. Chỉ viết câu trả lời cuối cùng.
5. Kết thúc bằng một mục "## Suy luận" gồm các bước suy luận đánh số (1., 2., 3.) ngắn gọn.
"""

#: Prompt **JSON mode** của baseline B0.  Được dùng thay cho
#: :data:`CLOSED_BOOK_SYSTEM_PROMPT` khi client hỗ trợ ``response_schema``:
#: khi có schema thì API tắt hoàn toàn kênh suy luận
#: (``thoughts_token_count: null``) nên **không bao giờ** rơi vào
#: ``MAX_TOKENS`` với câu trả lời rỗng -- đây chính là nguyên nhân 98/150
#: câu B0 rỗng trong lượt đánh giá trước.  Đo đạc: docs/10 §2.
#:
#: Hai ràng buộc đo được từ đoạc đạc: (a) ``answer`` tối đa 4 câu -- câu dài
#: làm mô hình rơi vào lặp; và (b) ``reasoning_steps`` gồm đúng 3 bước ngắn,
#: mỗi bước là **một** câu tiếng Việt độc lập -- mô hình hay sao chép cụm từ
#: của câu hỏi vào bước suy luận khi không bị ràng buộc.
CLOSED_BOOK_JSON_SYSTEM_PROMPT = """Bạn là chuyên gia lịch sử Việt Nam. Bạn trả lời câu hỏi dựa trên KIẾN THỨC CỦA CHÍNH BẠN (closed-book), không có tài liệu tham chiếu nào được cung cấp.

Quy tắc bắt buộc:
1. `answer`: trả lời trực tiếp, súc tích, bằng tiếng Việt, đúng mốc thời gian và địa danh. Tối đa 4 câu.
2. KHÔNG gắn trích dẫn [E#] vì không có nguồn nào được cung cấp.
3. Nếu bạn KHÔNG CHẮC CHẮN hoặc không biết, `answer` phải đúng câu: "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này." Đây là một câu trả lời HỢP LỆ, không phải lỗi.
4. `reasoning_steps`: đúng 3 bước, mỗi bước là MỘT CÂU TIẾNG VIỆT ngắn (tối đa 20 từ), không sao chép văn bản, không viết tiếng Anh, không giải thích quy trình nội bộ.
5. KHÔNG tự hỏi lại người dùng.
"""

#: Schema JSON của baseline B0 (xem :data:`CLOSED_BOOK_JSON_SYSTEM_PROMPT`).
CLOSED_BOOK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "reasoning_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer"],
}

CLOSED_BOOK_USER_TEMPLATE = """CÂU HỎI: {question}

Hãy trả lời theo quy tắc ở trên."""


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
    def generate_closed_book(self, question: str) -> AnswerResult:
        """Answer ``question`` from the model's own knowledge (baseline B0).

        Không có CONTEXT và không có evidence: mô hình phải trả lời bằng kiến
        thức tham số.  Vì không có nguồn, đáp án không mang trích dẫn ``[E#]``
        và không thể bị chấm "grounded" -- chấm điểm groundedness chỉ áp dụng
        cho các hệ thống có truy xuất (xem ``grounded_applicable``).

        **Vì sao dùng JSON mode?**  Gemma 4 chạy một kênh suy luận riêng và
        tính token của kênh đó vào ``max_output_tokens``.  Với câu hỏi
        closed-book mà mô hình *không biết*, kênh suy luận cứ quay vòng tìm
        kiến thức tham số cho tới khi hết ngân sách, khiến API trả về
        ``finish_reason=MAX_TOKENS`` với **câu trả lời rỗng** (đo đạc:
        ``scripts/_probe_closedbook_cause.py``; 98/150 câu B0 của lượt đánh
        giá trước rơi vào tình huống này).  Khi dùng ``response_schema`` thì
        API **tắt hoàn toàn** kênh suy luận (``thoughts_token_count: null``)
        và lời gọi luôn kết thúc trong 3–9 giây.

        Vì vậy baseline B0 dùng JSON mode, và:
        * nếu client không hỗ trợ schema -> fallback sang prompt văn bản thuần;
        * nếu JSON hỏng -> fallback sang prompt văn bản thuần;
        * nếu câu trả lời **lặp** (degenerate) -> coi như không có câu trả lời
          và trả lời bằng câu từ chối chuẩn.
        """
        if self.llm_client is None:
            return AnswerResult(answer=REFUSAL_ANSWER, refused=True)

        from ..llm.schemas import LLMMessage, LLMRequest

        user = CLOSED_BOOK_USER_TEMPLATE.format(question=question.strip())

        # -- 1) JSON mode (tắt thinking -> luôn kết thúc) ------------------
        json_request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=CLOSED_BOOK_JSON_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user),
            ],
            task="answer_closed_book",
            temperature=0.0,
            response_schema=dict(CLOSED_BOOK_SCHEMA),
        )
        raw = ""
        try:
            raw = self.llm_client.generate(json_request).text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sinh câu trả lời closed-book (JSON) thất bại: %s", exc)
            raw = ""

        answer_text, chain = self._parse_closed_book_json(raw)
        if answer_text and not is_degenerate(answer_text):
            return AnswerResult(
                answer=answer_text.strip(),
                citations=[],
                used_evidence_ids=[],
                invalid_citations=[],
                refused=is_refusal(answer_text),
                reasoning_chain=chain,
                raw_response=raw,
            )

        if raw.strip():
            # The JSON was unusable (broken or degenerate): fall back to the
            # plain-text prompt, which is slower but sometimes works.
            logger.info("Closed-book JSON không dùng được -> thử prompt văn bản")
            answer_text, chain = self._generate_closed_book_text(question)
            if answer_text and not is_degenerate(answer_text):
                return AnswerResult(
                    answer=answer_text.strip(),
                    citations=[],
                    used_evidence_ids=[],
                    invalid_citations=[],
                    refused=is_refusal(answer_text),
                    reasoning_chain=chain,
                    raw_response=raw,
                )

        logger.info("Closed-book không sinh được câu trả lời -> từ chối")
        return AnswerResult(answer=REFUSAL_ANSWER, refused=True)

    # ------------------------------------------------------------------
    def _generate_closed_book_text(self, question: str) -> tuple[str, list[str]]:
        """The plain-text fallback of the closed-book baseline."""
        from ..llm.schemas import LLMMessage, LLMRequest

        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=CLOSED_BOOK_SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=CLOSED_BOOK_USER_TEMPLATE.format(question=question.strip()),
                ),
            ],
            task="answer_closed_book_text",
            temperature=0.0,
        )
        raw = ""
        try:
            raw = self.llm_client.generate(request).text
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sinh câu trả lời closed-book thất bại: %s", exc)
            raw = ""
        if not raw.strip():
            return "", []
        answer_text = self._strip_reasoning(raw)
        chain = parse_reasoning_chain(raw) or parse_reasoning_chain(answer_text)
        return answer_text.strip(), chain

    @staticmethod
    def _parse_closed_book_json(raw: str) -> tuple[str, list[str]]:
        """Parse the JSON payload of the closed-book baseline.

        Returns ``(answer, reasoning_chain)``.  The answer keeps the
        ``## Suy luận`` block so that the downstream parsing (and the UI)
        sees the same shape as the other systems.
        """
        if not (raw or "").strip():
            return "", []
        payload: Any = None
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # The model sometimes wraps the JSON in prose or a markdown fence.
            repaired = LLMClient._extract_json(raw)
            if repaired is not None:
                try:
                    payload = json.loads(repaired)
                except json.JSONDecodeError:
                    payload = None
        if not isinstance(payload, dict):
            return "", []
        answer = str(payload.get("answer") or "").strip()
        steps = payload.get("reasoning_steps") or []
        chain = [str(step).strip() for step in steps if str(step).strip()]
        return answer, chain

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
