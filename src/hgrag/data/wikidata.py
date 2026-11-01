"""Wikidata access via the Action API (``wbgetentities`` / ``wbsearchentities``).

Per ADR-08 the SPARQL endpoint is **not** used (it timed out at 40 s during
prototyping).  ``wbgetentities`` returns aliases, P31, P27, P569, P570 and
the description for a batch of 50 QIDs in a single request, which is fast
enough for the whole pipeline.

See docs/07 §2 and §5 (tầng T0/T1) for how the returned data is used.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "WikidataClient",
    "get_entities",
    "search_entity",
    "entity_aliases",
    "entity_props",
    "PROP_P31",
    "PROP_P27",
    "PROP_P569",
    "PROP_P570",
]

API_BASE = "https://www.wikidata.org/w/api.php"
DEFAULT_USER_AGENT = "VietnamHistoryGraphRAG/0.1 (research; course project)"
DEFAULT_DELAY = 0.5
REQUEST_TIMEOUT = 30
BATCH_SIZE = 50

PROP_P31 = "P31"  # instance of
PROP_P27 = "P27"  # country of citizenship
PROP_P569 = "P569"  # date of birth
PROP_P570 = "P570"  # date of death
PROP_DESCRIPTION = "description"
PROP_ALIASES = "aliases"
PROP_LABELS = "labels"

#: Languages whose aliases/labels are kept.
_ALIAS_LANGS = ("vi", "en")


class WikidataClient:
    """Small client around the Wikidata Action API."""

    def __init__(
        self,
        delay: float = DEFAULT_DELAY,
        user_agent: str = DEFAULT_USER_AGENT,
        cache_dir: str | Path | None = None,
        timeout: int = REQUEST_TIMEOUT,
        offline: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.delay = max(0.0, float(delay))
        self.timeout = timeout
        self.offline = offline
        self.api_base = API_BASE
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self._last_request = 0.0

    # ------------------------------------------------------------------
    # disk cache
    # ------------------------------------------------------------------
    def _cache_file(self, kind: str, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:120] or "_"
        return self.cache_dir / "wikidata" / f"{kind}_{safe}.json"

    def _cache_load(self, kind: str, key: str) -> Any | None:
        path = self._cache_file(kind, key)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _cache_save(self, kind: str, key: str, payload: Any) -> None:
        path = self._cache_file(kind, key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot write Wikidata cache: %s", exc)

    def _throttle(self, delay: float | None = None) -> None:
        gap = self.delay if delay is None else delay
        if gap <= 0:
            return
        elapsed = time.monotonic() - self._last_request
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_request = time.monotonic()

    def _request(self, params: dict[str, Any], retries: int = 4) -> dict[str, Any]:
        """Perform an API request, retrying on 429/5xx with exponential backoff."""
        if self.offline:
            return {}
        delay = max(self.delay, 1.5)
        for attempt in range(retries):
            self._throttle(delay)
            try:
                resp = self.session.get(self.api_base, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                logger.warning("Wikidata request failed (%s): %s", params.get("action"), exc)
                return {}
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    logger.warning("Wikidata returned non-JSON: %s", exc)
                    return {}
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = delay * (2**attempt)
                logger.warning(
                    "Wikidata HTTP %s -> retry in %.1fs (%s)",
                    resp.status_code,
                    wait,
                    params.get("ids", params.get("search", params.get("action"))),
                )
                time.sleep(wait)
                delay = max(delay, 1.0)
                continue
            logger.warning("Wikidata HTTP %s", resp.status_code)
            return {}
        logger.warning("Wikidata request gave up after %d retries", retries)
        return {}

    # ------------------------------------------------------------------
    def get_entities(self, qids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Fetch entities by QID (batched 50 QIDs per request).

        The responses are cached on disk (``<cache_dir>/wikidata``) so that a
        re-run does not hammer the API -- the Wikimedia endpoints answer
        429 quickly when the request rate is too high.
        """
        wanted = [q.strip().upper() for q in qids if q and q.strip()]
        # de-duplicate, keep order
        seen: set[str] = set()
        unique: list[str] = []
        for q in wanted:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        entities: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        for qid in unique:
            cached = self._cache_load("entity", qid)
            if isinstance(cached, dict) and cached:
                entities[qid] = cached
            else:
                pending.append(qid)
        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            params = {
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(batch),
                "props": "labels|aliases|descriptions|claims",
                "languages": "|".join(_ALIAS_LANGS),
            }
            data = self._request(params)
            got = data.get("entities") or {}
            for qid, payload in got.items():
                if qid in batch:
                    entities[qid] = payload
                    self._cache_save("entity", qid, payload)
            missing = [q for q in batch if q not in got]
            if missing:
                logger.warning("Wikidata: %d entity/entities not found", len(missing))
        return entities

    def search_entity(self, name: str, language: str = "vi", limit: int = 5) -> dict[str, Any] | None:
        """Search an entity by label/alias.  Returns the best hit or ``None``."""
        name = (name or "").strip()
        if not name:
            return None
        cache_key = f"{name}|{language}|{limit}"
        cached = self._cache_load("search", cache_key)
        if cached is not None:
            return cached or None
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "search": name,
            "language": language,
            "uselang": language,
            "type": "item",
            "limit": max(1, min(int(limit), 50)),
        }
        data = self._request(params)
        hits = data.get("search") or []
        if not hits:
            self._cache_save("search", cache_key, {})
            return None
        self._cache_save("search", cache_key, hits[0])
        return hits[0]

    # ------------------------------------------------------------------
    @staticmethod
    def _labels(payload: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for lang in _ALIAS_LANGS:
            lab = (payload.get("labels") or {}).get(lang) or {}
            val = lab.get("value")
            if val:
                out.append(val)
        return out

    @staticmethod
    def _aliases(payload: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for lang in _ALIAS_LANGS:
            group = (payload.get("aliases") or {}).get(lang) or []
            for item in group:
                val = item.get("value")
                if val:
                    out.append(val)
        return out

    @staticmethod
    def _description(payload: dict[str, Any]) -> str:
        for lang in _ALIAS_LANGS:
            desc = (payload.get("descriptions") or {}).get(lang) or {}
            if desc.get("value"):
                return desc["value"]
        return ""

    @staticmethod
    def _claim_values(payload: dict[str, Any], prop: str) -> list[str]:
        claims = (payload.get("claims") or {}).get(prop) or []
        out: list[str] = []
        for claim in claims:
            snak = claim.get("mainsnak") or {}
            if snak.get("snaktype") != "value":
                continue
            value = (snak.get("datavalue") or {}).get("value")
            if isinstance(value, dict) and value.get("id"):
                out.append(str(value["id"]))
            elif isinstance(value, dict) and value.get("time"):
                out.append(str(value["time"]))
            elif value is not None:
                out.append(str(value))
        return out

    def entity_aliases(self, qid: str) -> list[str]:
        """All aliases (vi + en) of an entity, de-duplicated, order preserved."""
        qid = (qid or "").strip().upper()
        if not qid:
            return []
        payload = self.get_entities([qid]).get(qid)
        if not payload:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for val in self._labels(payload) + self._aliases(payload):
            if val not in seen:
                seen.add(val)
                out.append(val)
        return out

    def entity_props(self, qid: str) -> dict[str, Any]:
        """Return ``{p31, p27, p569, p570, description, label, aliases}``."""
        qid = (qid or "").strip().upper()
        empty = {
            "qid": qid,
            "label": "",
            "aliases": [],
            "description": "",
            "p31": [],
            "p27": [],
            "p569": [],
            "p570": [],
        }
        if not qid:
            return empty
        payload = self.get_entities([qid]).get(qid)
        if not payload:
            return empty
        labels = self._labels(payload)
        return {
            "qid": qid,
            "label": labels[0] if labels else "",
            "aliases": self._aliases(payload),
            "description": self._description(payload),
            "p31": self._claim_values(payload, PROP_P31),
            "p27": self._claim_values(payload, PROP_P27),
            "p569": self._claim_values(payload, PROP_P569),
            "p570": self._claim_values(payload, PROP_P570),
        }


_DEFAULT_CLIENT: WikidataClient | None = None


def get_client(**kwargs) -> WikidataClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None or kwargs:
        _DEFAULT_CLIENT = WikidataClient(**kwargs)
    return _DEFAULT_CLIENT


def get_entities(qids: Iterable[str], **kwargs) -> dict[str, dict[str, Any]]:
    return get_client().get_entities(qids)


def search_entity(name: str, **kwargs) -> dict[str, Any] | None:
    return get_client().search_entity(name, **kwargs)


def entity_aliases(qid: str, **kwargs) -> list[str]:
    return get_client().entity_aliases(qid, **kwargs)


def entity_props(qid: str, **kwargs) -> dict[str, Any]:
    return get_client().entity_props(qid, **kwargs)
