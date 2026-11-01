"""Loader for the "Vietnam History Chat" datasets on HuggingFace.

Primary target: ``minhxthanh/Vietnam-History-15k`` (and the larger
``-200k`` / ``-1M`` variants when they are reachable).  Rows have a single
``messages`` column with ChatML turns:

``system`` / ``user`` / ``assistant(analysis)`` / ``assistant(final)``

Because the datasets are hosted on the HuggingFace hub, the loader prefers
the parquet export (``/api/datasets/<id>/parquet/...``) and falls back to the
datasets-server rows API.  Nothing raises on failure -- a warning is logged
and an empty list is returned.

Caveat from docs/01 §1.3: the answers are LLM-generated and may be wrong, so
this source is used for *question samples* and supplementary QA text, never
as a ground-truth source.
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "HF_DATASETS",
    "load_history_chat",
    "load_raw",
    "row_to_record",
    "MIN_YEAR",
    "MAX_YEAR",
]

#: Candidate dataset ids, ordered by preference.
HF_DATASETS: tuple[str, ...] = (
    "minhxthanh/Vietnam-History-15k",
    "minhxthanh/Vietnam-History-200k",
    "minhxthanh/Vietnam-History-1M",
)

MIN_YEAR = 905
MAX_YEAR = 1975

USER_AGENT = "VietnamHistoryGraphRAG/0.1 (research; course project)"
_ROWS_API = "https://datasets-server.huggingface.co/rows"
_SPLITS_API = "https://datasets-server.huggingface.co/splits"
_PARQUET_API = "https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"

_YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|[1-9][0-9]{2})(?!\d)")


def _get(url: str, params: dict[str, Any] | None = None, timeout: int = 60) -> requests.Response | None:
    try:
        resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        logger.warning("History-chat request failed (%s): %s", url, exc)
        return None
    if resp.status_code != 200:
        logger.warning("History-chat HTTP %s for %s", resp.status_code, url)
        return None
    return resp


def _dataset_variants(dataset: str) -> list[tuple[str, str, str]]:
    """Return ``(dataset, config, split)`` candidates."""
    out: list[tuple[str, str, str]] = []
    resp = _get(_SPLITS_API, params={"dataset": dataset})
    if resp is not None:
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        for split in payload.get("splits", []) or []:
            out.append((split.get("dataset", dataset), split.get("config", "default"), split.get("split", "train")))
    if not out:
        out.append((dataset, "default", "train"))
    return out


def _rows_via_server(dataset: str, config: str, split: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page = 100
    while len(rows) < limit:
        length = min(page, limit - len(rows))
        resp = _get(
            _ROWS_API,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
        )
        if resp is None:
            break
        try:
            payload = resp.json()
        except ValueError:
            break
        batch = payload.get("rows") or []
        if not batch:
            break
        for item in batch:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                rows.append(row)
        if len(batch) < length:
            break
        offset += len(batch)
    return rows[:limit]


def _rows_via_parquet(dataset: str, config: str, split: str, limit: int) -> list[dict[str, Any]]:
    """Download the parquet export and read it with pandas (if available)."""
    resp = _get(_PARQUET_API.format(dataset=dataset, config=config, split=split))
    if resp is None:
        return []
    try:
        urls = resp.json()
    except ValueError:
        return []
    if not isinstance(urls, list) or not urls:
        return []
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas unavailable -> cannot read parquet export")
        return []
    rows: list[dict[str, Any]] = []
    for url in urls:
        pr = _get(url, timeout=180)
        if pr is None:
            continue
        try:
            df = pd.read_parquet(io.BytesIO(pr.content))
        except Exception as exc:  # noqa: BLE001 - defensive, third-party parser
            logger.warning("Cannot read parquet %s: %s", url, exc)
            continue
        for rec in df.head(limit).to_dict(orient="records"):
            rows.append(_json_safe(rec))
        if rows:
            break
    return rows[:limit]


def load_raw(
    limit: int = 1000,
    dataset: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> list[dict[str, Any]]:
    """Return raw rows (``{"messages": [...]}``) from the first reachable dataset."""
    if offline:
        cached = _load_cache(cache_dir)
        return cached[:limit] if cached else []
    candidates = [dataset] if dataset else list(HF_DATASETS)
    for ds in candidates:
        for d, config, split in _dataset_variants(ds):
            rows = _rows_via_parquet(d, config, split, limit)
            if not rows:
                rows = _rows_via_server(d, config, split, limit)
            if rows:
                logger.info("Loaded %d history-chat rows from %s/%s/%s", len(rows), d, config, split)
                _save_cache(cache_dir, rows)
                return rows[:limit]
    logger.warning("No history-chat dataset could be reached")
    return []


def _cache_path(cache_dir: str | Path | None) -> Path | None:
    if not cache_dir:
        return None
    return Path(cache_dir) / "history_chat.json"


def _load_cache(cache_dir: str | Path | None) -> list[dict[str, Any]]:
    path = _cache_path(cache_dir)
    if path and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (OSError, ValueError) as exc:
            logger.warning("Cannot read history-chat cache: %s", exc)
    return []


def _json_safe(value: Any) -> Any:
    """Convert numpy / pandas scalars into plain Python objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:  # noqa: BLE001 - defensive
            return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001 - defensive
            return str(value)
    return str(value)


def _save_cache(cache_dir: str | Path | None, rows: list[dict[str, Any]]) -> None:
    path = _cache_path(cache_dir)
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_json_safe(rows), ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Cannot write history-chat cache: %s", exc)


def _extract_year(text: str) -> int | None:
    years = [int(y) for y in _YEAR_RE.findall(text or "")]
    if not years:
        return None
    return min(years)


def row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw row into a flat record with ``question`` / ``answer``."""
    messages = row.get("messages") or []
    question = ""
    answer = ""
    analysis = ""
    system = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content") or ""
        channel = msg.get("channel")
        if role == "system" and not system:
            system = content
        elif role == "user" and not question:
            question = content
        elif role == "assistant":
            # The ``final`` channel holds the real answer, ``analysis`` the CoT.
            if channel == "final":
                if not answer:
                    answer = content
            elif not analysis:
                analysis = content
    if not answer:
        answer = analysis
    return {
        "question": question,
        "answer": answer,
        "system": system,
        "year": _extract_year(question),
        "n_messages": len(messages),
    }


def load_history_chat(
    limit: int = 15000,
    dataset: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    min_year: int = MIN_YEAR,
    max_year: int = MAX_YEAR,
    filter_years: bool = True,
) -> list[dict[str, Any]]:
    """Load history-chat Q/A pairs, optionally filtered to 905-1975.

    Rows without any year mention are kept (they may still be in scope);
    rows whose year falls outside ``[min_year, max_year]`` are dropped.
    """
    rows = load_raw(limit=limit * 2, dataset=dataset, cache_dir=cache_dir, offline=offline)
    out: list[dict[str, Any]] = []
    for row in rows:
        rec = row_to_record(row)
        if not rec["question"]:
            continue
        year = rec["year"]
        if filter_years and year is not None and not (min_year <= year <= max_year):
            continue
        out.append(rec)
        if len(out) >= limit:
            break
    logger.info("history-chat: kept %d records (limit=%d)", len(out), limit)
    return out
