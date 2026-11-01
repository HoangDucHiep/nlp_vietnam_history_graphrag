"""Section-aware chunking of long text (Wikipedia articles, chat records).

Design goals (docs/01 §3, docs/02 §2):

* Keep *sections* (``== ... ==``) as hard boundaries -- never cut a heading
  away from its body.
* Never cut a sentence in the middle when a sentence boundary is available.
* Produce **stable** ``chunk_id`` values so that a re-run of the pipeline
  yields the same ids (needed for provenance / Qdrant payload updates).
"""

from __future__ import annotations

import logging
import re

from .records import Chunk, stable_uid

logger = logging.getLogger(__name__)

__all__ = [
    "split_sections",
    "split_sentences",
    "chunk_text",
    "chunk_document",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "DEFAULT_MIN_CHARS",
]

DEFAULT_MAX_CHARS = 1400
DEFAULT_OVERLAP_CHARS = 200
DEFAULT_MIN_CHARS = 120

#: Headings inside an article ("== Thân thế ==") and their level.
_HEADING_RE = re.compile(r"^(={1,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
#: Sentence terminator followed by whitespace (Vietnamese + latin punctuation).
_SENTENCE_RE = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a plain-text article into ``(heading, body)`` pairs.

    The leading text before the first heading is returned under the empty
    heading ``""``.  Headings themselves are *not* part of the body but the
    heading text is returned so callers can label chunks.

    >>> split_sections("intro\\n== A ==\\nbody")
    [('', 'intro'), ('A', 'body')]
    """
    if not text:
        return []
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text.strip())]
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        sections.append((m.group(2).strip(), body))
    return [(h, b) for h, b in sections if b]


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences (keeping the trailing punctuation)."""
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_RE.split(text.strip())]
    return [p for p in parts if p]


def _pack(
    units: list[str],
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[str]:
    """Greedily pack text units into chunks, never splitting a unit."""
    chunks: list[str] = []
    chunk_units: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal chunk_units, buf_len
        if chunk_units:
            chunks.append(" ".join(chunk_units).strip())
        chunk_units = []
        buf_len = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if chunk_units and buf_len + len(unit) + 1 > max_chars:
            tail: list[str] = []
            tail_len = 0
            if overlap_chars > 0:
                for prev in reversed(chunk_units):
                    if tail_len + len(prev) + 1 > overlap_chars:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev) + 1
            flush()
            if tail:
                chunk_units = tail
                buf_len = tail_len
        chunk_units.append(unit)
        buf_len += len(unit) + 1
    flush()
    # Merge a trailing fragment that is too short into the previous chunk.
    if len(chunks) >= 2 and len(chunks[-1]) < min_chars:
        tail = chunks.pop()
        chunks[-1] = (chunks[-1] + " " + tail).strip()
    return chunks


def chunk_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    section: str = "",
) -> list[tuple[str, str]]:
    """Chunk a single section body.

    Returns a list of ``(section_heading, chunk_text)``.  Sentences are the
    atomic unit; a sentence longer than ``max_chars`` is hard-split on word
    boundaries (rare, but avoids unbounded chunks).
    """
    if not text or not text.strip():
        return []
    units: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for sent in split_sentences(para):
            if len(sent) <= max_chars:
                units.append(sent)
                continue
            # Long sentence -> split on words.
            cur: list[str] = []
            cur_len = 0
            for word in sent.split():
                if cur and cur_len + len(word) + 1 > max_chars:
                    units.append(" ".join(cur))
                    cur, cur_len = [], 0
                cur.append(word)
                cur_len += len(word) + 1
            if cur:
                units.append(" ".join(cur))
    if not units:
        return []
    return [(section, c) for c in _pack(units, max_chars, overlap_chars, min_chars)]


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    sections: list[dict] | None = None,
    source: str = "wikipedia",
    url: str = "",
    wikidata_id: str = "",
    period: str = "",
    year_min: int | None = None,
    year_max: int | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    extra: dict | None = None,
) -> list[Chunk]:
    """Chunk a whole document and return :class:`Chunk` objects.

    Parameters
    ----------
    sections:
        Optional pre-parsed sections, each ``{"heading": str, "text": str}``.
        When ``None`` the sections are derived from ``text`` (Wikipedia
        ``== ... ==`` markup or plain paragraphs).
    """
    chunks: list[Chunk] = []
    base_extra = dict(extra or {})

    if sections:
        parsed: list[tuple[str, str]] = []
        for sec in sections:
            if isinstance(sec, str):
                parsed.append(("", sec))
            else:
                parsed.append((str(sec.get("heading", "") or ""), str(sec.get("text", "") or "")))
    else:
        parsed = split_sections(text or "")

    idx = 0
    for heading, body in parsed:
        if not body.strip():
            continue
        for sub_heading, chunk_body in chunk_text(
            body, max_chars=max_chars, overlap_chars=overlap_chars, min_chars=min_chars
        ):
            chunk_body = chunk_body.strip()
            if not chunk_body:
                continue
            label = sub_heading or heading
            cid = stable_uid(doc_id, label, str(idx), chunk_body[:80], prefix="ck_")
            chunk_extra = dict(base_extra)
            chunk_extra.setdefault("section_index", idx)
            if heading:
                chunk_extra.setdefault("article_section", heading)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    title=title,
                    text=chunk_body,
                    url=url,
                    section=label,
                    source=source,
                    wikidata_id=wikidata_id,
                    period=period,
                    year_min=year_min,
                    year_max=year_max,
                    extra=chunk_extra,
                )
            )
            idx += 1
    return chunks


def chunk_from_settings(
    doc_id: str,
    title: str,
    text: str,
    settings: object,
    **kwargs,
) -> list[Chunk]:
    """Convenience wrapper reading chunk sizes from a :class:`Settings`."""
    cfg = getattr(settings, "chunking", {}) or {}
    max_chars = int(cfg.get("max_chars", DEFAULT_MAX_CHARS))
    overlap_chars = int(cfg.get("overlap_chars", DEFAULT_OVERLAP_CHARS))
    min_chars = int(cfg.get("min_chars", DEFAULT_MIN_CHARS))
    return chunk_document(
        doc_id,
        title,
        text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
        **kwargs,
    )
