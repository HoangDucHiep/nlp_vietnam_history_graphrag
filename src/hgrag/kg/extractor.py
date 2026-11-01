"""LLM-based event extraction against the VHE ontology.

The extractor asks the LLM for a JSON payload that follows the VHE schema
(35 event types / 11 argument roles) and validates the result with
:func:`hgrag.kg.ontology.validate_event`.  Events that do not satisfy the
ontology are dropped with a warning -- they must never reach the KG.

Results are cached on disk (``artifacts/extractions/<chunk_id>.json``) so a
re-run of the pipeline costs no LLM call at all.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..data.records import Chunk, EventExtraction
from .ontology import Ontology, normalize_event_type, normalize_role

logger = logging.getLogger(__name__)

__all__ = ["EventExtractor", "EXTRACTION_CACHE_DIR", "build_extraction_prompt"]

#: Default cache directory for the extractions (relative to the repo root).
EXTRACTION_CACHE_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "extractions"

_YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|[1-9][0-9]{2})(?!\d)")

#: JSON schema handed to the LLM (kept in sync with the VHE ontology).
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_type": {"type": "string"},
                    "trigger": {"type": "string"},
                    "description": {"type": "string"},
                    "year": {"type": ["integer", "null"]},
                    "arguments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                            },
                            "required": ["role", "name"],
                        },
                    },
                },
                "required": ["event_type", "trigger", "arguments"],
            },
        }
    },
    "required": ["events"],
}

ARGUMENT_TYPES = ("Person", "Place", "Organization", "Period", "Event", "Artifact")


def _event_from_mapping(item: Any) -> EventExtraction | None:
    """Build an :class:`EventExtraction` from a plain mapping.

    ``EventExtraction.from_dict`` cannot be used here: the dataclass has no
    ``extra`` field, so ``from_dict`` raises for unknown keys.
    """
    if not isinstance(item, Mapping):
        return None
    arguments = []
    for arg in item.get("arguments") or []:
        if isinstance(arg, Mapping):
            arguments.append({str(k): str(v) for k, v in arg.items()})
    try:
        return EventExtraction(
            chunk_id=str(item.get("chunk_id") or ""),
            event_type=str(item.get("event_type") or ""),
            trigger=str(item.get("trigger") or ""),
            arguments=arguments,
            description=str(item.get("description") or ""),
            year=item.get("year"),
            uid=str(item.get("uid") or ""),
        )
    except (TypeError, ValueError):
        return None


def build_extraction_prompt(chunk: Chunk, ontology: Ontology) -> str:
    """Build the user prompt for the extraction request."""
    event_types = ", ".join(ontology.event_types)
    roles = ", ".join(ontology.roles)
    arg_types = ", ".join(ARGUMENT_TYPES)
    header = f"Tiêu đề bài: {chunk.title}"
    if chunk.section:
        header += f" | Mục: {chunk.section}"
    if chunk.year_min:
        header += f" | Năm: {chunk.year_min}"
    return (
        f"{header}\n"
        "Trích xuất các SỰ KIỆN lịch sử có trong đoạn văn bản dưới đây.\n\n"
        f"Đoạn văn bản:\n{chunk.text}\n\n"
        "Quy tắc:\n"
        f"- event_type phải là một trong 35 loại sau: {event_types}\n"
        f"- role phải là một trong 11 role sau (viết HOA, dấu gạch dưới): {roles}\n"
        f"- type của argument phải là một trong: {arg_types}\n"
        "- Mỗi event phải có ít nhất một argument có role AGENT.\n"
        "- trigger là cụm từ kích hoạt sự kiện trong văn bản.\n"
        "- year là năm (số nguyên) hoặc null nếu không rõ.\n"
        "- Trả về DUY NHẤT một JSON object hợp lệ, không markdown fence.\n"
        'Cú pháp: {"events": [{"event_type": ..., "trigger": ..., "description": ..., '
        '"year": ..., "arguments": [{"role": ..., "name": ..., "type": ...}]}]}'
    )


class EventExtractor:
    """Extract VHE-conformant events out of a :class:`~hgrag.data.records.Chunk`.

    Parameters
    ----------
    llm_client:
        Any :class:`~hgrag.llm.client.LLMClient`.  The request ``task`` is
        ``"extract_events"`` -- the mock client dispatches on ``"extract"``.
    ontology:
        The VHE ontology used to validate the LLM output.
    cache_dir:
        Where the per-chunk extraction cache lives.  ``None`` disables caching.
    max_events:
        Maximum number of events kept per chunk.
    """

    def __init__(
        self,
        llm_client: Any,
        ontology: Ontology | None = None,
        cache_dir: str | Path | None = None,
        max_events: int = 4,
        use_cache: bool = True,
    ) -> None:
        self.llm_client = llm_client
        self.ontology = ontology or Ontology()
        if not self.ontology.event_types:
            from .ontology import ONTOLOGY

            self.ontology = ONTOLOGY
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_events = int(max_events)
        self.use_cache = bool(use_cache and self.cache_dir is not None)
        self.n_cache_hits = 0
        self.n_llm_calls = 0
        self.n_invalid = 0

    # ------------------------------------------------------------------
    # cache
    # ------------------------------------------------------------------
    def _cache_path(self, chunk_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", chunk_id)
        return self.cache_dir / f"{safe}.json"

    def _load_cache(self, chunk: Chunk) -> list[EventExtraction] | None:
        if not self.use_cache:
            return None
        path = self._cache_path(chunk.chunk_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Cannot read extraction cache %s: %s", path, exc)
            return None
        events = payload.get("events")
        if not isinstance(events, list):
            return None
        out: list[EventExtraction] = []
        for item in events:
            event = _event_from_mapping(item)
            if event is not None:
                out.append(event)
        self.n_cache_hits += 1
        return out

    def _save_cache(self, chunk: Chunk, events: Sequence[EventExtraction]) -> None:
        if not self.use_cache:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_path(chunk.chunk_id)
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "events": [e.to_dict() for e in events],
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Cannot write extraction cache: %s", exc)

    # ------------------------------------------------------------------
    # extraction
    # ------------------------------------------------------------------
    def extract_from_chunk(self, chunk: Chunk) -> list[EventExtraction]:
        """Extract the events of one chunk (cached)."""
        cached = self._load_cache(chunk)
        if cached is not None:
            return cached

        payload = self._call_llm(chunk)
        events = self._parse_events(payload, chunk)
        self._save_cache(chunk, events)
        return events

    def _call_llm(self, chunk: Chunk) -> Any:
        from ..llm.schemas import LLMMessage, LLMRequest

        prompt = build_extraction_prompt(chunk, self.ontology)
        request = LLMRequest(
            messages=[LLMMessage(role="user", content=prompt)],
            task="extract_events",
            temperature=0.0,
            response_schema=RESPONSE_SCHEMA,
        )
        self.n_llm_calls += 1
        try:
            return self.llm_client.generate_json(request)
        except Exception as exc:  # noqa: BLE001 - extraction must never crash the build
            logger.warning("LLM extraction failed for %s: %s", chunk.chunk_id, exc)
            return None

    def _parse_events(self, payload: Any, chunk: Chunk) -> list[EventExtraction]:
        if not isinstance(payload, Mapping):
            logger.warning("Extraction payload không phải object (chunk %s)", chunk.chunk_id)
            return []
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            logger.warning("Thiếu trường 'events' (chunk %s)", chunk.chunk_id)
            return []

        out: list[EventExtraction] = []
        for raw in raw_events:
            if not isinstance(raw, Mapping):
                continue
            event = self._build_event(raw, chunk)
            if event is None:
                continue
            errors = self.ontology.validate_event(event.event_type, event.arguments)
            if errors:
                self.n_invalid += 1
                logger.warning(
                    "Bỏ event không hợp lệ (%s) trong chunk %s: %s",
                    event.event_type,
                    chunk.chunk_id,
                    "; ".join(errors),
                )
                continue
            out.append(event)
            if len(out) >= self.max_events:
                break
        return out

    def _build_event(self, raw: Mapping[str, Any], chunk: Chunk) -> EventExtraction | None:
        event_type = normalize_event_type(str(raw.get("event_type") or ""))
        if not event_type:
            return None
        arguments: list[dict[str, str]] = []
        for arg in raw.get("arguments") or []:
            if not isinstance(arg, Mapping):
                continue
            role = normalize_role(str(arg.get("role") or ""))
            name = str(arg.get("name") or "").strip()
            if not role or not name:
                continue
            arg_type = str(arg.get("type") or "").strip()
            if arg_type not in ARGUMENT_TYPES:
                arg_type = "Entity"
            arguments.append({"role": role, "name": name, "type": arg_type})
        year = raw.get("year")
        try:
            year_int = int(year) if year not in (None, "", 0) else None
        except (TypeError, ValueError):
            year_int = None
        if year_int is None:
            match = _YEAR_RE.search(chunk.text or "")
            if match:
                year_int = int(match.group(1))
        return EventExtraction(
            chunk_id=chunk.chunk_id,
            event_type=event_type,
            trigger=str(raw.get("trigger") or "").strip(),
            arguments=arguments,
            description=str(raw.get("description") or "").strip(),
            year=year_int,
        )
