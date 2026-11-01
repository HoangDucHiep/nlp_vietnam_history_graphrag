"""Context assembly under a token budget (docs/03 §6).

The context builder is the last step before the answer generator: it takes
the fused evidence, formats it with stable citation labels (``E1``, ``E2``,
...) and **truncates** it so that the prompt never exceeds the token budget
(6 000 tokens by default).

Vietnamese text is tokenised by the Gemini tokenizer at roughly three
characters per token, which is the default estimate used here; the ratio is
configurable (``retrieval.chars_per_token``).

The truncation is *greedy but fair*: the evidence are sorted by decreasing
score and dropped from the tail until the budget fits.  At least one
evidence is always kept -- an empty context would make the answer generator
refuse to answer, which is a different behaviour from "the document does
not contain the answer".
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..data.records import Evidence

logger = logging.getLogger(__name__)

__all__ = ["ContextBuilder", "ContextPack", "estimate_tokens", "render_for_prompt"]

_WS_RE = re.compile(r"\s+")


def estimate_tokens(text: str, chars_per_token: int = 3) -> int:
    """Estimate the number of LLM tokens of ``text``.

    Vietnamese text packs roughly three characters per token.

    >>> estimate_tokens("a" * 30)
    10
    """
    if not text:
        return 0
    ratio = max(1, int(chars_per_token))
    return max(1, len(text) // ratio)


@dataclass
class ContextPack:
    """The formatted context handed to the answer generator."""

    text: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    n_tokens_estimate: int = 0
    truncated: bool = False
    citations_map: dict[str, Evidence] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
            "n_tokens_estimate": self.n_tokens_estimate,
            "truncated": self.truncated,
            "citations_map": {k: v.to_dict() for k, v in self.citations_map.items()},
        }


def render_for_prompt(context_pack: ContextPack) -> str:
    """Render a :class:`ContextPack` as the ``CONTEXT`` block of the prompt."""
    if context_pack is None:
        return ""
    header = "=== CONTEXT ===\n"
    footer = "\n=== END CONTEXT ===\n"
    return f"{header}{context_pack.text}{footer}"


class ContextBuilder:
    """Format and truncate the evidence list under a token budget."""

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings
        cfg: dict[str, Any] = {}
        if settings is not None:
            raw = getattr(settings, "retrieval", None)
            if raw is None and isinstance(settings, dict):
                raw = settings.get("retrieval")
            cfg = dict(raw or {})
        self.max_tokens = int(cfg.get("context_max_tokens", 6000))
        self.max_evidence = int(cfg.get("max_evidence", 12))
        self.chars_per_token = int(cfg.get("chars_per_token", 3))
        self.max_evidence_per_doc = int(cfg.get("max_evidence_per_doc", 3))

    # ------------------------------------------------------------------
    def build_context(
        self,
        evidence: Sequence[Evidence],
        max_tokens: int = 6000,
    ) -> ContextPack:
        """Build the context pack of ``evidence``.

        The citation labels are always ``E1..En`` and contiguous, even when
        some evidence were dropped by the budget.
        """
        items = [e for e in (evidence or []) if e is not None]
        budget = int(max_tokens or self.max_tokens)
        if budget <= 0:
            budget = self.max_tokens
        if self.max_evidence > 0:
            items = items[: self.max_evidence]

        # Sort by decreasing score but keep the original order of equal scores
        indexed = list(enumerate(items))
        indexed.sort(key=lambda pair: (-float(pair[1].score or 0.0), pair[0]))

        kept: list[tuple[int, Evidence]] = []
        used = 0
        truncated = False
        for index, item in indexed:
            rendered = self._render_evidence(item, len(kept) + 1)
            cost = estimate_tokens(rendered, self.chars_per_token)
            if kept and used + cost > budget:
                truncated = True
                continue
            if not kept and used + cost > budget:
                # always keep at least one evidence, even if it is too long
                truncated = True
            kept.append((index, item))
            used += cost

        kept.sort(key=lambda pair: pair[0])
        texts: list[str] = []
        citations: dict[str, Evidence] = {}
        ids: list[str] = []
        for position, (_index, item) in enumerate(kept, start=1):
            label = f"E{position}"
            texts.append(self._render_evidence(item, position))
            citations[label] = item
            ids.append(label)

        body = "\n\n".join(texts)
        return ContextPack(
            text=body,
            evidence_ids=ids,
            n_tokens_estimate=estimate_tokens(body, self.chars_per_token),
            truncated=truncated or len(kept) < len(items),
            citations_map=citations,
        )

    # ------------------------------------------------------------------
    def _render_evidence(self, evidence: Evidence, position: int) -> str:
        label = f"E{position}"
        source = self._source_label(evidence)
        header = f"[{label}] (Nguồn: {source})"
        text = (evidence.text or "").strip()
        return f"{header}\n{text}"

    @staticmethod
    def _source_label(evidence: Evidence) -> str:
        meta = evidence.meta if isinstance(evidence.meta, Mapping) else {}
        title = str(meta.get("title") or "").strip()
        section = str(meta.get("section") or "").strip()
        source = str(meta.get("source") or "").strip() or "Wikipedia"
        if evidence.kind == "graph_path":
            label = f"{source} - đường đi đồ thị"
            if title:
                label = f"{source} - \"{title}\" (đường đi đồ thị)"
            return label
        if title and section:
            return f'{source} - "{title}", mục "{section}"'
        if title:
            return f'{source} - "{title}"'
        return source
