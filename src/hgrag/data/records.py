"""Shared data contracts between the data, KG, retrieval and QA layers.

These dataclasses are the *interface* between modules.  Every module that
produces or consumes data must use them, so that the pipeline stays
composable and the sub-modules can be developed independently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# Entity / relation types
# --------------------------------------------------------------------------

ENTITY_TYPES = ("Person", "Place", "Organization", "Period", "Event", "Artifact")


@dataclass(slots=True)
class Chunk:
    """A unit of retrievable text with full provenance."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    url: str = ""
    section: str = ""
    source: str = "wikipedia"  # wikipedia | vhe | history_chat
    wikidata_id: str = ""
    period: str = ""
    year_min: int | None = None
    year_max: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        known = {f for f in cls.__slots__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        obj = cls(**payload)
        obj.extra.update(extra)
        return obj


@dataclass(slots=True)
class Entity:
    """A historical entity, canonically identified by a Wikidata QID when known."""

    uid: str
    name: str
    entity_type: str = "Event"
    wikidata_id: str = ""
    aliases: list[str] = field(default_factory=list)
    name_key: str = ""
    description: str = ""
    real_name: str = ""
    title: str = ""
    posthumous_name: str = ""
    period: str = ""
    year_min: int | None = None
    year_max: int | None = None
    roles: list[str] = field(default_factory=list)
    p31: list[str] = field(default_factory=list)
    p27: list[str] = field(default_factory=list)
    wiki_categories: list[str] = field(default_factory=list)
    source: str = "wikipedia"  # wikipedia | vhe | llm
    confidence: float = 1.0
    source_chunk_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Entity:
        known = {f for f in cls.__slots__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        obj = cls(**payload)
        obj.extra.update(extra)
        return obj


@dataclass(slots=True)
class Relation:
    """A typed edge between two entities, always traceable to a source chunk."""

    uid: str
    src: str  # Entity.uid
    dst: str  # Entity.uid
    rel_type: str  # e.g. PARTICIPATED_IN, CAUSE_OF, SUPPORTED_BY
    text: str = ""  # natural-language description used for embedding
    year: int | None = None
    source: str = "llm"  # vhe | llm | wiki
    confidence: float = 1.0
    source_chunk_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Relation:
        known = {f for f in cls.__slots__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        obj = cls(**payload)
        obj.extra.update(extra)
        return obj


@dataclass(slots=True)
class EventExtraction:
    """One extracted event with its arguments, as produced by the LLM."""

    chunk_id: str
    event_type: str
    trigger: str = ""
    arguments: list[dict[str, str]] = field(default_factory=list)
    description: str = ""
    year: int | None = None
    uid: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventExtraction:
        known = {f for f in cls.__slots__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        obj = cls(**payload)
        obj.extra.update(extra)
        return obj


@dataclass(slots=True)
class Evidence:
    """A single retrievable piece of evidence handed to the answer generator."""

    evidence_id: str  # E1, E2, ...
    kind: str  # chunk | graph_path | node
    text: str
    score: float = 0.0
    source_chunk_ids: list[str] = field(default_factory=list)
    path: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_uid(*parts: str, prefix: str = "") -> str:
    """Deterministic short uid derived from the given parts."""
    raw = "\x1f".join(p for p in parts if p)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}" if prefix else digest


def stable_hash(payload: Any) -> str:
    """Deterministic hash of any JSON-serialisable payload (used for LLM cache)."""
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
