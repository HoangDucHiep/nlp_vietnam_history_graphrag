"""Ingest orchestrator: raw sources -> ``data/processed/corpus.jsonl``.

Outputs
-------
``data/processed/corpus.jsonl``
    One JSON object per :class:`~hgrag.data.records.Chunk`.

``data/processed/corpus_meta.json``
    Run statistics (documents, chunks, per-source counters, warnings).

The module never crashes because of a single failing source: every source is
wrapped in ``try/except`` and a warning is recorded instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from . import chunking, history_chat, vhe, wikipedia
from .records import Chunk

logger = logging.getLogger(__name__)

__all__ = ["ingest_all", "write_corpus", "IngestStats"]


class IngestStats(dict):
    """A plain dict with a couple of convenience accessors."""

    @property
    def n_chunks(self) -> int:
        return int(self.get("n_chunks", 0))

    @property
    def n_docs(self) -> int:
        return int(self.get("n_docs", 0))


def _doc_id(source: str, key: str) -> str:
    return f"{source}:{key}"


def _year_from_text(text: str) -> tuple[int | None, int | None]:
    """Extract a plausible year range from an article lead.

    Only 4-digit years in ``[900, 2100]`` are considered so that small
    numbers (Hán-Việt era numbers, counts, ...) do not pollute the range.
    """
    import re

    years = [
        int(y)
        for y in re.findall(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)", text or "")
        if 900 <= int(y) <= 2100
    ]
    if not years:
        return None, None
    return min(years), max(years)


def _ingest_wikipedia(
    settings: Settings,
    limit: int | None,
    seed_titles: list[str] | None,
    warnings: list[str],
    offline: bool = False,
) -> list[Chunk]:
    cfg = settings.ingest
    cache_dir = settings.data_raw / "wiki"
    client = wikipedia.WikipediaClient(
        delay=float(cfg.get("request_delay", 0.5)),
        user_agent=str(cfg.get("user_agent", wikipedia.DEFAULT_USER_AGENT)),
        cache_dir=cache_dir,
        offline=offline,
    )
    max_articles = int(limit) if limit else int(cfg.get("wikipedia_max_articles", 300))
    rounds = int(cfg.get("wikipedia_expand_rounds", 1))
    min_chars = int(cfg.get("wikipedia_min_chars", 400))
    max_chars = int(cfg.get("wikipedia_max_chars", 120000))

    seeds = seed_titles if seed_titles else list(wikipedia.DEFAULT_SEED_TITLES)
    try:
        articles = client.expand_from_seeds(
            seed_titles=seeds,
            max_articles=max_articles,
            rounds=0 if seed_titles else rounds,
            min_chars=min_chars,
        )
    except Exception as exc:  # noqa: BLE001 - never break the pipeline
        warnings.append(f"wikipedia: expansion failed ({exc})")
        logger.warning("wikipedia: expansion failed: %s", exc)
        return []

    ck = settings.chunking
    chunks: list[Chunk] = []
    for page in articles:
        extract = page.get("extract") or ""
        if len(extract) > max_chars:
            extract = extract[:max_chars]
        if len(extract) < min_chars:
            continue
        title = page.get("title") or ""
        doc_id = _doc_id("wiki", wikipedia.title_slug(title))
        # The lead paragraph carries the birth/death years -> use it for the
        # temporal metadata instead of the whole article.
        lead = (page.get("sections") or [{}])[0].get("text", "") if page.get("sections") else ""
        year_min, year_max = _year_from_text(lead[:1500] or extract[:1500])
        try:
            doc_chunks = chunking.chunk_document(
                doc_id=doc_id,
                title=title,
                text=extract,
                sections=page.get("sections") or None,
                source="wikipedia",
                url=page.get("url", ""),
                wikidata_id=page.get("wikidata_id", ""),
                year_min=year_min,
                year_max=year_max,
                max_chars=int(ck.get("max_chars", chunking.DEFAULT_MAX_CHARS)),
                overlap_chars=int(ck.get("overlap_chars", chunking.DEFAULT_OVERLAP_CHARS)),
                min_chars=int(ck.get("min_chars", chunking.DEFAULT_MIN_CHARS)),
                extra={
                    "pageid": page.get("pageid", 0),
                    "categories": page.get("categories", [])[:20],
                    "redirects": page.get("redirects", [])[:20],
                },
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"wikipedia: chunking failed for {title} ({exc})")
            continue
        chunks.extend(doc_chunks)
    return chunks


def _ingest_vhe(
    settings: Settings,
    limit: int | None,
    warnings: list[str],
    offline: bool = False,
) -> tuple[list[Chunk], dict[str, Any]]:
    """Load the VHE ontology + gold instances.

    Per ADR-01 VHE is **not** part of the retrieval corpus; the instances are
    returned separately so the caller can store them as gold annotations.
    """
    cache_dir = settings.data_raw / "vhe"
    ontology: dict[str, Any] = {}
    instances: list[dict[str, Any]] = []
    try:
        ontology = vhe.load_ontology(cache_dir=cache_dir, offline=offline)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"vhe: ontology load failed ({exc})")
        ontology = {
            "event_types": list(vhe.DEFAULT_EVENT_TYPES),
            "roles": list(vhe.DEFAULT_ROLES),
            "schema": {},
            "source": "builtin-fallback",
            "n_event_types": len(vhe.DEFAULT_EVENT_TYPES),
            "n_roles": len(vhe.DEFAULT_ROLES),
        }
    try:
        instances = vhe.load_instances(limit=limit, cache_dir=cache_dir, offline=offline)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"vhe: instances load failed ({exc})")
        instances = []
    # VHE is gold/ontology -> no chunks in the corpus (ADR-01).
    return [], {"ontology": ontology, "instances": instances}


def _ingest_history_chat(
    settings: Settings,
    limit: int | None,
    warnings: list[str],
    offline: bool = False,
) -> list[Chunk]:
    cfg = settings.ingest
    chat_limit = int(limit) if limit else int(cfg.get("history_chat_limit", 15000))
    min_year = int(cfg.get("history_chat_min_year", 905))
    max_year = int(cfg.get("history_chat_max_year", 1975))
    cache_dir = settings.data_raw / "history_chat"
    try:
        records = history_chat.load_history_chat(
            limit=chat_limit,
            cache_dir=cache_dir,
            offline=offline,
            min_year=min_year,
            max_year=max_year,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"history_chat: load failed ({exc})")
        return []
    if not records:
        warnings.append(
            "history_chat: no records loaded (dataset unreachable or empty) -- "
            "không có dữ liệu history_chat trong lần chạy này"
        )
        return []

    ck = settings.chunking
    chunks: list[Chunk] = []
    for i, rec in enumerate(records):
        question = rec.get("question", "")
        answer = rec.get("answer", "")
        text = f"Câu hỏi: {question}\nTrả lời: {answer}".strip()
        if len(text) < 30:
            continue
        year = rec.get("year")
        doc_id = _doc_id("history_chat", f"chat{i:06d}")
        try:
            doc_chunks = chunking.chunk_document(
                doc_id=doc_id,
                title=question[:120],
                text=text,
                sections=None,
                source="history_chat",
                url="",
                year_min=year,
                year_max=year,
                max_chars=int(ck.get("max_chars", chunking.DEFAULT_MAX_CHARS)),
                overlap_chars=int(ck.get("overlap_chars", chunking.DEFAULT_OVERLAP_CHARS)),
                min_chars=int(ck.get("min_chars", chunking.DEFAULT_MIN_CHARS)),
                extra={"year": year, "row_index": i},
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"history_chat: chunking failed for row {i} ({exc})")
            continue
        chunks.extend(doc_chunks)
    return chunks


def write_corpus(chunks: Iterable[Chunk], out_path: str | Path) -> int:
    """Write chunks to a JSONL file; returns the number of lines written."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.to_json())
            fh.write("\n")
            n += 1
    return n


def ingest_all(
    settings: Settings | None = None,
    limit: int | None = None,
    sources: Iterable[str] = ("wikipedia", "vhe", "history_chat"),
    seed_titles: list[str] | None = None,
    out_dir: str | Path | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    """Run the full ingest and write the corpus + metadata files.

    Returns the statistics dict (also written to ``corpus_meta.json``).
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    wanted = {s.strip().lower() for s in sources if s and s.strip()}
    warnings: list[str] = []

    chunks: list[Chunk] = []
    vhe_payload: dict[str, Any] = {}

    if "wikipedia" in wanted:
        logger.info("Ingesting Wikipedia...")
        try:
            chunks.extend(_ingest_wikipedia(settings, limit, seed_titles, warnings, offline=offline))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"wikipedia: source skipped ({exc})")
            logger.warning("wikipedia source skipped: %s", exc)

    if "vhe" in wanted:
        logger.info("Ingesting VHE (ontology + gold)...")
        try:
            vhe_chunks, vhe_payload = _ingest_vhe(settings, limit, warnings, offline=offline)
            chunks.extend(vhe_chunks)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"vhe: source skipped ({exc})")

    if "history_chat" in wanted:
        logger.info("Ingesting history chat...")
        try:
            chunks.extend(_ingest_history_chat(settings, limit, warnings, offline=offline))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"history_chat: source skipped ({exc})")

    out_root = Path(out_dir) if out_dir else settings.data_processed
    out_root.mkdir(parents=True, exist_ok=True)
    corpus_path = out_root / "corpus.jsonl"
    meta_path = out_root / "corpus_meta.json"

    n_lines = write_corpus(chunks, corpus_path)

    per_source: dict[str, int] = {}
    docs_per_source: dict[str, set[str]] = {}
    for c in chunks:
        per_source[c.source] = per_source.get(c.source, 0) + 1
        docs_per_source.setdefault(c.source, set()).add(c.doc_id)
    doc_ids = {c.doc_id for c in chunks}

    ontology = vhe_payload.get("ontology") or {}
    stats: dict[str, Any] = {
        "n_docs": len(doc_ids),
        "n_chunks": n_lines,
        "n_docs_by_source": {k: len(v) for k, v in docs_per_source.items()},
        "n_chunks_by_source": per_source,
        "sources_requested": sorted(wanted),
        "limit": limit,
        "corpus_path": str(corpus_path),
        "meta_path": str(meta_path),
        "vhe": {
            "event_types": ontology.get("event_types", []),
            "n_event_types": ontology.get("n_event_types", 0),
            "roles": ontology.get("roles", []),
            "n_roles": ontology.get("n_roles", 0),
            "ontology_source": ontology.get("source", ""),
            "n_instances": len(vhe_payload.get("instances", []) or []),
        },
        "warnings": warnings,
    }
    try:
        meta_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot write %s: %s", meta_path, exc)
    logger.info("Ingest done: %d docs / %d chunks", len(doc_ids), n_lines)
    return stats
