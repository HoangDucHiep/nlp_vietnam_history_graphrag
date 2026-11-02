"""Typed LLM request/response contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JSON_SCHEMA_GUIDE = (
    "Trả về DUY NHẤT một JSON object hợp lệ, không markdown fence, không giải thích thêm."
)


@dataclass(slots=True)
class LLMMessage:
    role: str  # system | user | assistant
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class LLMRequest:
    messages: list[LLMMessage]
    model: str = ""
    temperature: float = 0.0
    #: Output-token budget.  ``0`` means "use the budget of the client"
    #: (``llm.max_output_tokens`` in the configuration).  The Gemma family
    #: counts its *reasoning* tokens against this budget, so a value that is
    #: too small makes the model return an **empty** answer; see
    #: docs/10_quota_gemini_free_tier.md.
    max_output_tokens: int = 0
    response_schema: dict[str, Any] | None = None
    response_mime_type: str | None = None
    task: str = "general"  # used as cache namespace

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_schema": self.response_schema,
            "response_mime_type": self.response_mime_type,
            "task": self.task,
        }


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str = ""
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_json(self) -> bool:
        return self.text.strip().startswith("{") or self.text.strip().startswith("[")

    def json(self) -> Any:
        import json

        return json.loads(self.text)
