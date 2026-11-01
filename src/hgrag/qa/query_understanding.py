"""Query understanding: parse a Vietnamese question into a retrieval plan.

The module asks the LLM for a structured JSON plan and *always* keeps a
heuristic fallback: if the LLM fails (no key, quota, malformed JSON) the
question is still understood well enough to run the retrieval pipeline.

This is the first stage of the pipeline described in docs/03 §3.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Years outside this range are not considered historical references.
YEAR_MIN = 905
YEAR_MAX = 1975

_YEAR_RE = re.compile(r"\b(9\d{2}|1[0-9]{3})\b")

_CAUSAL_CUES = (
    "vì sao",
    "vì sao",
    "tại sao",
    "nguyên nhân",
    "do đâu",
    "lý do",
    "tại sao",
    "gây ra",
    "dẫn đến",
    "hậu quả",
)
_TEMPORAL_CUES = (
    "khi nào",
    "năm nào",
    "thời gian nào",
    "trước hay sau",
    "diễn ra vào",
    "mốc thời gian",
    "thứ tự",
)
_COMPARISON_CUES = ("so sánh", "khác nhau", "giống nhau", "khác biệt", "trong khi")
_MULTIHOP_CUES = (
    "và",
    "kết hợp",
    "qua",
    "thông qua",
    "dẫn đến",
    "liên quan",
    "từ",
    "đến",
)

#: Relation types to look for, per intent.
RELATION_PLAN: dict[str, list[str]] = {
    "causal": ["CAUSE_OF", "PRECONDITION_OF", "PRECEDED_BY", "PARTICIPATED_IN"],
    "temporal": ["OCCURRED_IN", "HAPPENED_AT", "PRECEDED_BY", "SUCCEEDED_BY"],
    "comparison": ["PARTICIPATED_IN", "OCCURRED_IN", "LOCATED_IN", "ABOUT"],
    "multi_hop": ["PARTICIPATED_IN", "CAUSE_OF", "LOCATED_IN", "OCCURRED_IN", "ABOUT"],
    "factual": ["PARTICIPATED_IN", "LOCATED_IN", "OCCURRED_IN", "ABOUT"],
    "unknown": [],
}

UNDERSTANDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "time_refs": {"type": "array", "items": {"type": "integer"}},
        "relation_plan": {"type": "array", "items": {"type": "string"}},
        "expected_hops": {"type": "integer"},
        "needs_graph": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "entities"],
}


@dataclass
class QueryUnderstanding:
    """The structured plan extracted from a question."""

    question: str
    intent: str = "factual"
    entities: list[str] = field(default_factory=list)
    time_refs: list[int] = field(default_factory=list)
    relation_plan: list[str] = field(default_factory=list)
    expected_hops: int = 1
    needs_graph: bool = True
    confidence: float = 0.5
    source: str = "heuristic"  # "llm" | "heuristic"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "entities": list(self.entities),
            "time_refs": list(self.time_refs),
            "relation_plan": list(self.relation_plan),
            "expected_hops": self.expected_hops,
            "needs_graph": self.needs_graph,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "raw": dict(self.raw),
        }


def extract_years(text: str) -> list[int]:
    """Historical years (905-1975) mentioned in ``text``."""
    out: list[int] = []
    for match in _YEAR_RE.finditer(text or ""):
        try:
            year = int(match.group(1))
        except ValueError:
            continue
        if YEAR_MIN <= year <= YEAR_MAX and year not in out:
            out.append(year)
    return out


def detect_intent(question: str) -> str:
    """Classify the question intent with keyword cues."""
    q = (question or "").lower()
    if any(cue in q for cue in _COMPARISON_CUES):
        return "comparison"
    if any(cue in q for cue in _CAUSAL_CUES):
        return "causal"
    if any(cue in q for cue in _TEMPORAL_CUES):
        return "temporal"
    return "factual"


class QueryUnderstandingModule:
    """Turn a question into a :class:`QueryUnderstanding`."""

    def __init__(self, llm_client: Any = None, ontology: Any = None) -> None:
        self.llm_client = llm_client
        self.ontology = ontology

    # ------------------------------------------------------------------
    def understand(self, question: str) -> QueryUnderstanding:
        """Understand ``question`` (LLM first, heuristic fallback)."""
        question = (question or "").strip()
        payload = self._ask_llm(question)
        if payload:
            understanding = self._from_payload(question, payload)
            if understanding is not None:
                return understanding
        return self.heuristic(question)

    # ------------------------------------------------------------------
    def heuristic(self, question: str) -> QueryUnderstanding:
        """Rule-based plan used when the LLM is unavailable."""
        from ..retrieval.entity_linker import extract_vietnamese_mentions

        intent = detect_intent(question)
        years = extract_years(question)
        entities = extract_vietnamese_mentions(question)
        hops = 1 if len(entities) <= 1 else (2 if len(entities) == 2 else 3)
        if intent in ("causal", "comparison", "multi_hop"):
            hops = max(hops, 2)
        return QueryUnderstanding(
            question=question,
            intent=intent,
            entities=entities,
            time_refs=years,
            relation_plan=list(RELATION_PLAN.get(intent, [])),
            expected_hops=hops,
            needs_graph=True,
            confidence=0.4,
            source="heuristic",
            raw={"detected_intent": intent},
        )

    # ------------------------------------------------------------------
    def _ask_llm(self, question: str) -> dict[str, Any] | None:
        if self.llm_client is None:
            return None
        from ..llm.schemas import LLMMessage, LLMRequest

        entity_hint = ""
        if self.ontology is not None:
            types = getattr(self.ontology, "event_types", None) or []
            entity_hint = ", ".join(list(types)[:20])

        prompt = (
            "Bạn là bộ phân tích câu hỏi cho hệ thống hỏi đáp lịch sử Việt Nam.\n"
            "Phân tích câu hỏi sau và trả về JSON theo đúng schema.\n\n"
            f"CÂU HỎI: {question}\n\n"
            "Quy ước:\n"
            "- intent: một trong factual | temporal | causal | comparison | multi_hop\n"
            "- entities: các thực thể (người, địa điểm, sự kiện) có trong câu hỏi\n"
            f"- time_refs: các năm lịch sử trong khoảng 905-1975 (có thể rỗng)\n"
            "- relation_plan: các loại quan hệ KG cần ưu tiên, trong đó có thể dùng: "
            f"{entity_hint or 'PARTICIPATED_IN, LOCATED_IN, OCCURRED_IN, CAUSE_OF, PRECONDITION_OF, ABOUT'}\n"
            "- expected_hops: số bước suy luận cần thiết (1-3)\n"
            "- needs_graph: có cần tra cứu đồ thị kiến thức hay không\n"
        )
        request = LLMRequest(
            messages=[LLMMessage(role="user", content=prompt)],
            task="query_understanding",
            temperature=0.0,
            response_schema=UNDERSTANDING_SCHEMA,
        )
        try:
            payload = self.llm_client.generate_json(request)
        except Exception as exc:  # noqa: BLE001 - never fail the pipeline
            logger.info("Query understanding qua LLM thất bại (%s) -> dùng heuristic", exc)
            return None
        return payload if isinstance(payload, dict) else None

    # ------------------------------------------------------------------
    def _from_payload(
        self, question: str, payload: dict[str, Any]
    ) -> QueryUnderstanding | None:
        raw_entities = payload.get("entities")
        entities = [str(e).strip() for e in (raw_entities or []) if str(e).strip()]
        if not entities:
            return None
        intent = str(payload.get("intent") or "factual").strip().lower()
        if intent not in RELATION_PLAN:
            intent = detect_intent(question)

        years: list[int] = []
        for value in payload.get("time_refs") or []:
            try:
                year = int(value)
            except (TypeError, ValueError):
                continue
            if YEAR_MIN <= year <= YEAR_MAX and year not in years:
                years.append(year)
        if not years:
            years = extract_years(question)

        plan = [str(r).strip().upper() for r in (payload.get("relation_plan") or []) if str(r).strip()]
        if not plan:
            plan = list(RELATION_PLAN.get(intent, []))

        try:
            hops = int(payload.get("expected_hops") or 0)
        except (TypeError, ValueError):
            hops = 0
        if hops <= 0:
            hops = 1 if len(entities) <= 1 else (2 if len(entities) == 2 else 3)

        try:
            confidence = float(payload.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8

        return QueryUnderstanding(
            question=question,
            intent=intent,
            entities=entities,
            time_refs=years,
            relation_plan=plan,
            expected_hops=max(1, min(3, hops)),
            needs_graph=bool(payload.get("needs_graph", True)),
            confidence=confidence,
            source="llm",
            raw=dict(payload),
        )
