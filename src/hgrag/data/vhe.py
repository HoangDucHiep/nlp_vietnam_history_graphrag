"""Loader for the VHE dataset (PACLIC 2024).

``hoangthuytruc/vhe-dataset`` is the canonical home of the dataset (35 event
types / 11 argument roles / 4.111 annotated sentences).  Depending on how the
environment is set up the files can be reached in three ways:

1. the HuggingFace dataset repo (when publicly reachable),
2. the GitHub repo ``hoangthuytruc/vhe-dataset`` (raw files),
3. a local copy under ``data/raw/vhe/``.

Every loader returns ``[]`` / empty structures (with a warning) instead of
raising, so the ingest pipeline degrades gracefully.

Per ADR-01 VHE is used as **ontology + gold**, not as the main corpus.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "HF_DATASET",
    "GITHUB_REPO",
    "VHE_FILES",
    "load_ontology",
    "load_instances",
    "load_raw",
    "load_event_relations",
    "DEFAULT_EVENT_TYPES",
    "DEFAULT_ROLES",
]

HF_DATASET = "hoangthuytruc/vhe-dataset"
GITHUB_REPO = "hoangthuytruc/vhe-dataset"
GITHUB_RAW = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

#: Relative path -> role in the dataset.
VHE_FILES: dict[str, str] = {
    "event-extraction/event.json": "instances",
    "event-extraction/vhe_event_schema.json": "schema",
    "event-extraction/role2id.json": "roles",
    "event-extraction/label2id.json": "event_types",
    "event-relation-extraction/event-relation.json": "relations",
    "event-relation-extraction/type2id.json": "relation_types",
}

#: Fallback ontology (identical to the dataset's own schema) -- used when the
#: network is unavailable so that the pipeline always has a usable ontology.
DEFAULT_EVENT_TYPES: tuple[str, ...] = (
    "be-born",
    "marry",
    "divorce",
    "injure",
    "die",
    "transport",
    "start-position",
    "end-position",
    "elect",
    "nominate",
    "arrest-jail",
    "release-parole",
    "trial-hearing",
    "charge-indict",
    "sue",
    "convict",
    "sentence",
    "fine",
    "execute",
    "extradite",
    "acquit",
    "appeal",
    "pardon",
    "transfer-ownership",
    "transfer-money",
    "start-org",
    "end-org",
    "merge-org",
    "declare-bankruptcy",
    "attack",
    "demonstrate",
    "meet",
    "phone-write",
    "natural-disaster",
    "natural-phenomenon",
)

DEFAULT_ROLES: tuple[str, ...] = (
    "temporal",
    "location",
    "agent",
    "theme",
    "beneficiary",
    "instrument",
    "source",
    "goal",
    "content",
    "force",
    "experiencer",
)

USER_AGENT = "VietnamHistoryGraphRAG/0.1 (research; course project)"


def _candidate_dirs(cache_dir: str | Path | None) -> list[Path]:
    dirs: list[Path] = []
    if cache_dir:
        dirs.append(Path(cache_dir))
    return dirs


def _local_path(cache_dir: str | Path | None, rel: str) -> Path | None:
    for d in _candidate_dirs(cache_dir):
        p = d / rel
        if p.exists():
            return p
    return None


def _download_to(url: str, dest: Path | None, timeout: int = 90) -> bytes | None:
    """Download ``url``; persist to ``dest`` when given."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        logger.warning("VHE download failed (%s): %s", url, exc)
        return None
    if resp.status_code != 200:
        logger.info("VHE download HTTP %s for %s", resp.status_code, url)
        return None
    if dest is not None:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
        except OSError as exc:
            logger.warning("Cannot write %s: %s", dest, exc)
    return resp.content


def _parse_json_or_jsonl(payload: bytes) -> Any:
    """Parse a JSON document or a JSON-Lines document."""
    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    # JSON-Lines: one JSON object per line.
    items: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except ValueError:
            continue
    return items or None


def _fetch_file(rel: str, cache_dir: str | Path | None = None, offline: bool = False) -> Any:
    """Return the parsed JSON of a VHE file, or ``None``.

    Resolution order: local cache -> HuggingFace -> GitHub raw.
    ``vhe_event_schema.json`` is JSON-Lines, the others are plain JSON.
    """
    local = _local_path(cache_dir, rel)
    if local is not None:
        try:
            return _parse_json_or_jsonl(local.read_bytes())
        except OSError as exc:
            logger.warning("Cannot read local VHE file %s: %s", local, exc)
    if offline:
        return None

    cache_root = Path(cache_dir) if cache_dir else None
    dest = cache_root / rel if cache_root is not None else None
    for prefix in (HF_RESOLVE, GITHUB_RAW):
        payload = _download_to(f"{prefix}/{rel}", dest)
        if payload is None:
            continue
        parsed = _parse_json_or_jsonl(payload)
        if parsed is not None:
            return parsed
        logger.warning("Cannot parse downloaded VHE file %s", rel)
    logger.warning("VHE file unavailable: %s", rel)
    return None


def load_raw(
    key: str = "instances",
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> Any:
    """Load a raw VHE file by logical key.

    ``key`` is one of ``instances``, ``schema``, ``roles``, ``event_types``,
    ``relations``, ``relation_types``.
    """
    rel = None
    for r, k in VHE_FILES.items():
        if k == key:
            rel = r
            break
    if rel is None:
        raise KeyError(f"Unknown VHE key: {key!r} (known: {sorted(set(VHE_FILES.values()))})")
    return _fetch_file(rel, cache_dir=cache_dir, offline=offline)


def load_ontology(
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    """Return the VHE ontology.

    Shape::

        {
          "event_types": ["be-born", ...],        # 35 entries
          "roles": ["temporal", ...],             # 11 entries
          "schema": {"be-born": ["temporal", ...]},
          "relation_types": {...},
          "source": "vhe-dataset" | "builtin-fallback",
        }
    """
    event_types: list[str] = []
    roles: list[str] = []
    schema: dict[str, list[str]] = {}
    relation_types: dict[str, Any] = {}
    source = "builtin-fallback"

    raw_types = load_raw("event_types", cache_dir=cache_dir, offline=offline)
    if isinstance(raw_types, dict):
        event_types = [k for k in raw_types if k and k != "NA"]

    raw_roles = load_raw("roles", cache_dir=cache_dir, offline=offline)
    if isinstance(raw_roles, dict):
        roles = [k for k in raw_roles if k and k != "NA"]

    raw_schema = load_raw("schema", cache_dir=cache_dir, offline=offline)
    if isinstance(raw_schema, list):
        for item in raw_schema:
            if not isinstance(item, dict):
                continue
            et = item.get("event-type")
            if not et:
                continue
            schema[et] = [r.get("role") for r in item.get("role-list", []) if r.get("role")]

    raw_rel_types = load_raw("relation_types", cache_dir=cache_dir, offline=offline)
    if isinstance(raw_rel_types, dict):
        relation_types = raw_rel_types

    if not event_types:
        event_types = list(DEFAULT_EVENT_TYPES)
        source = "builtin-fallback"
    else:
        source = "vhe-dataset"
    if not roles:
        roles = list(DEFAULT_ROLES)
    if not schema:
        schema = {et: list(roles) for et in event_types}

    return {
        "event_types": event_types,
        "roles": roles,
        "schema": schema,
        "relation_types": relation_types,
        "source": source,
        "n_event_types": len(event_types),
        "n_roles": len(roles),
    }


def load_instances(
    limit: int | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> list[dict[str, Any]]:
    """Load the annotated VHE sentences.

    Each item is ``{"id", "text", "events": [...]}`` where every event has
    ``type``, ``trigger_word``, ``offset`` and ``arguments``
    (``mention``, ``offset``, ``role``).
    """
    raw = load_raw("instances", cache_dir=cache_dir, offline=offline)
    if not isinstance(raw, list):
        logger.warning("VHE instances unavailable")
        return []
    items: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        text = row.get("text") or ""
        events = row.get("events") or []
        if not text or not events:
            continue
        items.append(
            {
                "id": row.get("id", ""),
                "text": text,
                "events": events,
            }
        )
    if limit is not None and limit >= 0:
        items = items[:limit]
    return items


def load_event_relations(
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> list[dict[str, Any]]:
    """Load the event-relation annotations (temporal / causal / subevent)."""
    raw = load_raw("relations", cache_dir=cache_dir, offline=offline)
    if not isinstance(raw, list):
        logger.warning("VHE event relations unavailable")
        return []
    out: list[dict[str, Any]] = []
    for row in raw:
        if isinstance(row, dict):
            out.append(row)
    return out


def extract_mentions(instances: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map ``mention -> [roles]`` over a list of instances."""
    out: dict[str, list[str]] = {}
    for inst in instances:
        for ev in inst.get("events", []):
            for arg in ev.get("arguments", []):
                mention = arg.get("mention")
                role = arg.get("role")
                if not mention:
                    continue
                out.setdefault(mention, [])
                if role and role not in out[mention]:
                    out[mention].append(role)
    return out


_YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|[1-9][0-9]{2})(?!\d)")
