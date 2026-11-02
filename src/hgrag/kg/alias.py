"""Entity resolution -- the 5-tier "QID-first" strategy of ADR-07.

The strategy is described in ``docs/07_xac_dinh_thuc_the.md``:

===  ==========================================================================
T0   The mention is a QID that is already known in the store (``wikidata_id``)
     -> ``score = 1.00``
T1   A Vietnamese Wikidata alias (``skos:altLabel[vi]``) of a known QID
     -> ``score = 0.97``
T2   A Wikipedia redirect (``prop=redirects``) pointing at a known entity
     -> ``score = 0.93``
T3   LLM verification of a *dubious* redirect (docs/07 §3)
     -> ``score = 0.85``
T4   String normalisation (``name_key`` from :mod:`hgrag.data.normalize`)
     -> ``score = 0.85``
T5   Dense embedding top-k + LLM disambiguation constrained by the historical
     domain (``P31`` human vs. ward, ``P27`` Vietnamese polity, time overlap)
     -> ``score = 0.60 .. 0.90``
===  ==========================================================================

The resolver is deliberately **conservative**: it never merges two entities
just because they share a label.  The two hard cases of docs/07 are handled
explicitly:

* ``Trần Quốc Tuấn`` / ``Hưng Đạo Vương`` / ``Trần Hưng Đạo`` /
  ``Hứng Đạo đại vương`` all resolve to **Q511375** (T1 exact alias, T1
  near-exact alias for the typo, T4 normalised key).
* ``Thụy Ba Công chúa`` is **not** Trần Hưng Đạo -- it is his adoptive aunt.
  Wikipedia redirects it to his article, so the resolver has to reject it.
  This is done with the *kinship guard* (:meth:`AliasResolver._kinship_mismatch`)
  and, when an LLM is available, with the T3 verification prompt.
* ``Hùng Vương`` (the legendary king, ``Q1320050``) and ``Hùng Vương``
  (a modern ward, ``Q10771249``) are **never** merged: they are separate
  nodes and the domain filter (``P31``/``P27``/requested type) keeps them apart.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.normalize import name_key, strip_diacritics
from ..data.records import stable_uid
from .ontology import ONTOLOGY, Ontology

logger = logging.getLogger(__name__)

__all__ = [
    "AliasResolver",
    "EntityCandidate",
    "TIER_SCORES",
    "MODERN_PLACE_P31",
    "KINSHIP_TITLES",
    "VN_HISTORY_QIDS",
    "P31_TYPE_MAP",
]

# --------------------------------------------------------------------------
# Tier scores (docs/07 §5)
# --------------------------------------------------------------------------

#: Score assigned to a candidate found at each tier.
TIER_SCORES: dict[int, float] = {
    0: 1.00,
    1: 0.97,
    2: 0.93,
    3: 0.85,
    4: 0.85,
    5: 0.60,
}

#: ``P31`` -> coarse entity type used by the domain filter.
P31_TYPE_MAP: dict[str, str] = {
    "Q5": "Person",  # human
    "Q486972": "Place",  # human settlement
    "Q515": "Place",  # city
    "Q532": "Place",  # capital
    "Q3957": "Place",  # locality
    "Q687188": "Place",  # xã / phường
    "Q166118": "Place",  # địa danh hành chính
    "Q164950": "Period",  # giai đoạn lịch sử
    "Q3024240": "Period",  # thời kỳ lịch sử
    "Q43229": "Organization",  # tổ chức
    "Q1661180": "Organization",
    "Q783794": "Organization",  # chính trị
    "Q783794x": "Organization",
    "Q327333": "Event",  # sự kiện lịch sử
    "Q13474373": "Event",  # sự kiện
    "Q163740": "Artifact",  # tác phẩm
    "Q179957": "Artifact",  # văn bản
    "Q860861": "Artifact",
}

#: Polities that mark an entity as belonging to Vietnamese history.
VN_HISTORY_QIDS: frozenset[str] = frozenset(
    {
        "Q881",  # Việt Nam
        "Q430309",  # Đại Việt
        "Q1763996",  # Văn Lang
        "Q878276",  # nhà Lê
        "Q10841085",  # nhà Đinh
        "Q1253644",  # nhà Trần
        "Q392238",  # nhà Lý
        "Q1005947",  # nhà Nguyễn
        "Q8714",  # Lý Thái Tổ (dynasty member)
    }
)

#: ``P31`` values of *modern* administrative divisions (xã/phường).
#: Such entities are never merged into the historical graph unless the caller
#: explicitly asks for a ``Place`` (docs/07 §4).
MODERN_PLACE_P31: frozenset[str] = frozenset(
    {
        "Q687188",  # xã / phường
        "Q1661180",  # đơn vị hành chính
        "Q200126",  # phường
    }
)

#: Titles/phrases that denote a *relative* of the article subject.  A
#: redirect title carrying one of them points to a **different person**
#: (docs/07 §3).  Matching is done on word boundaries of the accent-free
#: surface form, so multi-word phrases such as ``"cong chua"`` are matched
#: as a whole.
KINSHIP_TITLES: tuple[str, ...] = (
    "cong chua",
    "hoang cong chua",
    "nu vuong",
    "hoang tu",
    "thai tu",
    "thai phu",
    "thuong phu",
    "truong vuong",
    "tien nu",
    "ba co",
    "ong noi",
    "ba noi",
)

_QID_RE = re.compile(r"^Q\d+$", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _surface(s: str) -> str:
    """Lower-case, diacritic-free surface form that keeps the token order."""
    if not s:
        return ""
    text = unicodedata.normalize("NFC", str(s)).lower()
    text = text.replace("ð", "đ").replace("Ð", "Đ")
    text = _PUNCT_RE.sub(" ", text)
    text = strip_diacritics(text)
    return _WS_RE.sub(" ", text).strip()


def _tokens(s: str) -> list[str]:
    return [t for t in _surface(s).split(" ") if t]


def _contains_phrase(text: str, phrase: str) -> bool:
    """Word-boundary containment test on the accent-free surface form."""
    if not text or not phrase:
        return False
    return f" {phrase} " in f" {text} "


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance (two-row dynamic programming)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def _token_sim(a: str, b: str) -> float:
    """Similarity of two tokens in ``[0, 1]`` (edit distance based)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - (_levenshtein(a, b) / max(len(a), len(b)))


def _name_sim(a: str, b: str) -> float:
    """Token-aligned similarity of two names in ``[0, 1]``.

    Tokens of the shorter name are greedily matched against the best token
    of the longer one; the result is the mean of the matched similarities.
    This tolerates typos such as ``"Hứng Đạo"`` vs ``"Hưng Đạo"`` (0.8 per
    token) while rejecting unrelated names.
    """
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    if len(ta) > len(tb):
        ta, tb = tb, ta
    used: set[int] = set()
    total = 0.0
    for token in ta:
        best = 0.0
        best_idx = -1
        for idx, other in enumerate(tb):
            if idx in used:
                continue
            sim = _token_sim(token, other)
            if sim > best:
                best = sim
                best_idx = idx
        if best_idx >= 0 and best > 0:
            used.add(best_idx)
        total += best
    return total / len(tb)


def _alias_source_to_tier(source: str) -> int:
    """Map an alias provenance label onto a tier number."""
    src = (source or "").strip().lower().replace("-", "_")
    if src in ("wikidata", "wikidata_alias", "altlabel", "label", "exact"):
        return 1
    if src in ("wiki_redirect", "redirect", "redirects"):
        return 2
    if src in ("llm_verified", "verified"):
        return 3
    if src in ("normalized", "name_key", "fuzzy"):
        return 4
    return 1


# --------------------------------------------------------------------------
# Candidate
# --------------------------------------------------------------------------


@dataclass
class EntityCandidate:
    """A possible resolution of a mention to a KG entity."""

    uid: str
    name: str
    wikidata_id: str = ""
    aliases: list[str] = field(default_factory=list)
    entity_type: str = "Entity"
    score: float = 0.0
    tier: int = 5
    matched_alias: str = ""
    reason: str = ""
    p31: list[str] = field(default_factory=list)
    p27: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    #: Confidence reported by the LLM disambiguation (tier 5 only).
    extra_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "wikidata_id": self.wikidata_id,
            "aliases": list(self.aliases),
            "entity_type": self.entity_type,
            "score": round(self.score, 4),
            "tier": self.tier,
            "matched_alias": self.matched_alias,
            "reason": self.reason,
            "p31": list(self.p31),
            "p27": list(self.p27),
            "year_min": self.year_min,
            "year_max": self.year_max,
        }


@dataclass
class _Record:
    """Internal per-entity record kept in the alias index."""

    uid: str
    name: str
    name_key: str
    aliases: list[str] = field(default_factory=list)
    alias_tiers: dict[str, int] = field(default_factory=dict)
    wikidata_id: str = ""
    entity_type: str = "Entity"
    p31: list[str] = field(default_factory=list)
    p27: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    description: str = ""

    def alias_tier(self, alias: str) -> int:
        return self.alias_tiers.get(alias, 1)

    def has_surface(self, needle: str) -> bool:
        needle = _surface(needle)
        if not needle:
            return False
        for value in [self.name, *self.aliases]:
            if _surface(value) == needle:
                return True
        return False


# --------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------


class AliasResolver:
    """QID-first entity resolver with the 5 tiers of docs/07.

    Parameters
    ----------
    store:
        The KG store.  Used to look up entities that are already known
        (``find_entity_by_name``) -- this is what makes tier 0 possible.
    llm_client:
        Optional LLM used by tier 3 (redirect verification) and tier 5
        (disambiguation).  Without it those tiers are skipped.
    embedder:
        Optional embedder for tier 5.
    wikidata:
        Optional Wikidata client (anything exposing ``search_entity`` and
        ``entity_props``).
    settings:
        The project :class:`~hgrag.config.Settings` -- used to read the
        confidence thresholds from ``configs/default.yaml``.
    """

    def __init__(
        self,
        store: Any = None,
        llm_client: Any = None,
        embedder: Any = None,
        wikidata: Any = None,
        settings: Any = None,
        wikipedia: Any = None,
        offline: bool = False,
        ontology: Ontology | None = None,
    ) -> None:
        self.store = store
        self.llm_client = llm_client
        self.embedder = embedder
        self.wikidata = wikidata
        self.wikipedia = wikipedia
        self.offline = offline
        self.ontology = ontology or ONTOLOGY

        if settings is None:
            from ..config import get_settings

            settings = get_settings()
        self.settings = settings
        retrieval = {}
        try:
            retrieval = settings.retrieval or {}
        except Exception:  # noqa: BLE001 - a plain dict is fine too
            retrieval = dict(settings.get("retrieval", {})) if isinstance(settings, dict) else {}
        self.min_confidence = float(retrieval.get("entity_link_min_confidence", 0.7))
        self.alias_threshold = float(retrieval.get("alias_confidence", 0.95))
        self.redirect_threshold = float(retrieval.get("redirect_confidence", 0.9))
        self.normalized_threshold = float(retrieval.get("normalized_confidence", 0.85))
        self.embedding_min_score = float(retrieval.get("embedding_link_min_score", 0.86))

        # index
        self.records: dict[str, _Record] = {}
        self._exact_index: dict[str, list[str]] = {}
        self._key_index: dict[str, list[str]] = {}
        self._qid_index: dict[str, str] = {}
        self._embed_cache: dict[str, list[float]] = {}
        # The tier-5 scorer needs the vector of *every* indexed entity.  With
        # ~14 000 entities that is a 4-minute CPU embedding pass on every
        # start of the evaluation harness, so the vectors are cached on disk
        # (keyed by the embedder name + the text) and reused.
        self._embed_cache_path: Path | None = None
        self._embed_cache_dirty = False
        # memoisation of the (expensive) network-backed lookups
        self._redirect_cache: dict[str, list[str]] = {}
        self._search_cache: dict[str, Any] = {}
        self._resolve_cache: dict[tuple[str, str], list[EntityCandidate]] = {}

    # ------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------
    def enable_embed_cache(self, path: str | Path | None) -> None:
        """Persist the entity vectors on disk (see ``_embed_cache_path``)."""
        self._embed_cache_path = Path(path) if path else None
        if self._embed_cache_path and self._embed_cache_path.exists():
            try:
                payload = json.loads(self._embed_cache_path.read_text(encoding="utf-8"))
                vectors = payload.get("vectors") or {}
                self._embed_cache = {
                    str(k): [float(x) for x in v] for k, v in vectors.items()
                }
                logger.info(
                    "Alias embed cache: %d vector từ %s",
                    len(self._embed_cache),
                    self._embed_cache_path.name,
                )
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                logger.warning("Không đọc được alias embed cache (%s)", exc)

    def save_embed_cache(self) -> None:
        """Write the entity vectors to disk (if a cache path was set)."""
        path = self._embed_cache_path
        if path is None or not self._embed_cache:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(
                json.dumps(
                    {"embedder": str(getattr(self.embedder, "name", "")),
                     "vectors": self._embed_cache},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError as exc:  # pragma: no cover - best effort
            logger.warning("Không ghi được alias embed cache: %s", exc)

    def build_alias_index(self, entities: Iterable[Any]) -> None:
        """(Re)build the alias index from a collection of entities.

        Each entity may be an :class:`~hgrag.data.records.Entity` or a plain
        dict with the same keys.  Alias provenance is read from
        ``extra["alias_sources"]`` (``{alias: "wiki_redirect" | "wikidata"}``);
        unknown aliases default to the Wikidata-alias tier (1).
        """
        self.records.clear()
        self._exact_index.clear()
        self._key_index.clear()
        self._qid_index.clear()
        self.add_to_alias_index(entities)

    def add_to_alias_index(self, entities: Iterable[Any]) -> int:
        """Add entities to the alias index **without** clearing it.

        Used by the KG builder: the anchor nodes (document subjects) are
        added on top of the index that was built from the store, so that the
        mentions extracted from the corpus can be linked to them.
        """
        added = 0
        for entity in entities:
            record = self._record_from(entity)
            if record is None:
                continue
            # ``Chunk`` nodes are *provenance anchors*, not entities: they
            # carry the title of the document they come from, so a chunk named
            # "Trần Hưng Đạo" would shadow the real person node and hijack
            # every mention.  They are therefore excluded from the index.
            if record.entity_type == "Chunk":
                continue
            self.records[record.uid] = record
            self._index_record(record)
            added += 1
        logger.info(
            "Alias index: %d entities, %d surface keys, %d normalised keys",
            len(self.records),
            len(self._exact_index),
            len(self._key_index),
        )
        return added

    def _record_from(self, entity: Any) -> _Record | None:
        if entity is None:
            return None
        if isinstance(entity, Mapping):
            get = entity.get
            extra = entity.get("extra") or {}
        else:
            get = lambda k, d=None: getattr(entity, k, d)  # noqa: E731
            extra = getattr(entity, "extra", {}) or {}
        uid = str(get("uid", "") or "")
        name = str(get("name", "") or "")
        if not uid or not name:
            return None
        aliases = [str(a) for a in (get("aliases", []) or []) if a]
        alias_sources = {}
        if isinstance(extra, Mapping):
            alias_sources = dict(extra.get("alias_sources") or {})
            extra_aliases = list(extra.get("aliases") or [])
            for alias in extra_aliases:
                if alias not in aliases:
                    aliases.append(str(alias))
        record = _Record(
            uid=uid,
            name=name,
            name_key=name_key(name),
            aliases=aliases,
            wikidata_id=str(get("wikidata_id", "") or ""),
            entity_type=str(get("entity_type", "Entity") or "Entity"),
            p31=[str(p) for p in (get("p31", []) or [])],
            p27=[str(p) for p in (get("p27", []) or [])],
            year_min=get("year_min", None),
            year_max=get("year_max", None),
            description=str(get("description", "") or ""),
        )
        for alias in aliases:
            record.alias_tiers[alias] = _alias_source_to_tier(alias_sources.get(alias, "wikidata"))
        return record

    def _index_record(self, record: _Record) -> None:
        surfaces = {record.name: 1}
        for alias, tier in record.alias_tiers.items():
            surfaces[alias] = min(surfaces.get(alias, 9), tier)
        for surface, _tier in surfaces.items():
            key = _surface(surface)
            if not key:
                continue
            bucket = self._exact_index.setdefault(key, [])
            if record.uid not in bucket:
                bucket.append(record.uid)
        # normalised key index
        for surface in surfaces:
            key = name_key(surface)
            if not key:
                continue
            bucket = self._key_index.setdefault(key, [])
            if record.uid not in bucket:
                bucket.append(record.uid)
        if record.wikidata_id:
            self._qid_index[record.wikidata_id.upper()] = record.uid

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def resolve(self, mention: str, hints: dict[str, Any] | None = None) -> list[EntityCandidate]:
        """Resolve ``mention`` to a ranked list of candidates."""
        mention = (mention or "").strip()
        if not mention:
            return []
        hints = hints or {}
        cache_key = (mention, str(hints.get("entity_type") or hints.get("type") or ""))
        cached = self._resolve_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        candidates: dict[str, EntityCandidate] = {}

        def offer(cand: EntityCandidate | None) -> None:
            if cand is None:
                return
            existing = candidates.get(cand.uid)
            if existing is None or (cand.tier, cand.score) < (existing.tier, existing.score):
                candidates[cand.uid] = cand

        # -- T0: the mention *is* a QID, or a QID is already known in the store
        offer(self._tier0_qid(mention))
        offer(self._tier0_store(mention))

        # -- T1: Wikidata aliases
        for cand in self._tier1_wikidata_alias(mention):
            offer(cand)

        # -- T2/T3: Wikipedia redirects (+ LLM verification)
        for cand in self._tier2_redirect(mention):
            offer(cand)

        # -- T4: normalised key
        for cand in self._tier4_normalized(mention):
            offer(cand)

        # -- T5: embedding + LLM disambiguation
        for cand in self._tier5_embedding(mention, hints):
            offer(cand)

        out = [c for c in candidates.values() if c.score > 0]
        out = self._filter_domain(out, hints)
        out.sort(key=lambda c: (-c.score, c.tier, c.name))
        self._resolve_cache[cache_key] = list(out)
        return out

    def resolve_one(
        self, mention: str, hints: dict[str, Any] | None = None
    ) -> EntityCandidate | None:
        """Return the best candidate above the confidence threshold."""
        for cand in self.resolve(mention, hints):
            if cand.score >= self.min_confidence:
                return cand
        return None

    def resolve_batch(
        self, mentions: Iterable[str], hints: dict[str, Any] | None = None
    ) -> dict[str, list[EntityCandidate]]:
        """Resolve many mentions at once."""
        out: dict[str, list[EntityCandidate]] = {}
        for mention in mentions:
            key = (mention or "").strip()
            if key and key not in out:
                out[key] = self.resolve(key, hints)
        return out

    # ------------------------------------------------------------------
    # tier 0
    # ------------------------------------------------------------------
    def _tier0_qid(self, mention: str) -> EntityCandidate | None:
        if not _QID_RE.match(mention):
            return None
        qid = mention.upper()
        uid = self._qid_index.get(qid) or f"pe:{qid}"
        record = self.records.get(uid)
        return EntityCandidate(
            uid=uid,
            name=record.name if record else mention,
            wikidata_id=qid,
            aliases=list(record.aliases) if record else [],
            entity_type=record.entity_type if record else "Entity",
            score=TIER_SCORES[0],
            tier=0,
            matched_alias=mention,
            reason="T0: mention là QID đã biết",
            p31=list(record.p31) if record else [],
            p27=list(record.p27) if record else [],
            year_min=record.year_min if record else None,
            year_max=record.year_max if record else None,
        )

    def _tier0_store(self, mention: str) -> EntityCandidate | None:
        """The mention is already an entity of the KG with a known QID."""
        if self.store is None:
            return None
        try:
            rows = self.store.find_entity_by_name(mention)
        except Exception as exc:  # noqa: BLE001
            logger.debug("store.find_entity_by_name failed: %s", exc)
            return None
        for row in rows or []:
            qid = str(row.get("wikidata_id") or "")
            if not qid:
                continue
            uid = str(row.get("uid") or f"pe:{qid}")
            return EntityCandidate(
                uid=uid,
                name=str(row.get("name") or mention),
                wikidata_id=qid,
                aliases=[str(a) for a in (row.get("aliases") or [])],
                entity_type=str(row.get("entity_type") or "Entity"),
                score=TIER_SCORES[0],
                tier=0,
                matched_alias=str(row.get("name") or mention),
                reason="T0: QID đã biết trong KG",
                p31=[str(p) for p in (row.get("p31") or [])],
                p27=[str(p) for p in (row.get("p27") or [])],
                year_min=row.get("year_min"),
                year_max=row.get("year_max"),
            )
        return None

    # ------------------------------------------------------------------
    # tier 1
    # ------------------------------------------------------------------
    def _tier1_wikidata_alias(self, mention: str) -> list[EntityCandidate]:
        out: list[EntityCandidate] = []
        surf = _surface(mention)

        # (a) local index: exact surface match on a name / Wikidata alias
        for uid in self._exact_index.get(surf, []):
            record = self.records.get(uid)
            if record is None:
                continue
            # The tier of the *matching* surface form: the canonical name is
            # always tier 1, an alias carries its own provenance tier.
            tiers: list[int] = []
            matched = record.name
            if _surface(record.name) == surf:
                tiers.append(1)
            for alias, atier in record.alias_tiers.items():
                if _surface(alias) == surf:
                    tiers.append(atier)
                    if atier <= 1:
                        matched = alias
            if not tiers:
                continue
            tier = min(tiers)
            if tier > 1:
                # a Wikipedia redirect alias is tier 2 -> handled below
                continue
            out.append(self._candidate_from_record(record, tier, matched, "T1: alias Wikidata"))

        # (b) near-exact alias (typos such as "Hứng Đạo đại vương")
        if not out:
            for record in self.records.values():
                best_ratio = 0.0
                best_alias = ""
                for alias, tier in record.alias_tiers.items():
                    if tier != 1:
                        continue
                    ratio = _name_sim(mention, alias)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_alias = alias
                if best_ratio >= 0.85:
                    out.append(
                        self._candidate_from_record(
                            record, 1, best_alias, f"T1: alias Wikidata gần đúng ({best_ratio:.2f})"
                        )
                    )

        # (c) live Wikidata search: ``match.type == "alias"``
        if not out and self.wikidata is not None and not self.offline:
            if mention in self._search_cache:
                hit = self._search_cache[mention]
            else:
                try:
                    hit = self.wikidata.search_entity(mention)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("wikidata.search_entity failed: %s", exc)
                    hit = None
                self._search_cache[mention] = hit
            if hit:
                qid = str(hit.get("id") or "")
                match_type = str((hit.get("match") or {}).get("type") or "")
                label = str(hit.get("label") or (hit.get("match") or {}).get("text") or mention)
                description = str(hit.get("description") or "")
                if qid and match_type in ("alias", "label"):
                    # ``wbsearchentities`` returns ``match.type == "label"``
                    # when the mention *is* the canonical label of the item --
                    # an exact-label hit is as strong as a Wikidata alias.
                    tier = 1 if match_type == "alias" else 1
                    cand = self._candidate_for_qid(
                        qid, label, tier, mention, f"T1: Wikidata wbsearchentities ({match_type})"
                    )
                    if description and not cand.reason:
                        cand.reason += f" -- {description}"
                    out.append(cand)
        return out

    # ------------------------------------------------------------------
    # tier 2 / 3
    # ------------------------------------------------------------------
    def _tier2_redirect(self, mention: str) -> list[EntityCandidate]:
        out: list[EntityCandidate] = []
        redirects = self._known_redirects(mention)
        for target in redirects:
            record = self._record_for_target(target)
            if record is None:
                continue
            if self._kinship_mismatch(mention, record):
                logger.info(
                    "T2 rejected: %r là quan hệ huyết thống của %s, không phải alias",
                    mention,
                    record.name,
                )
                continue
            out.append(self._candidate_from_record(record, 2, mention, "T2: Wikipedia redirect"))
        if not out:
            return []

        # Tier 3: verify the dubious redirect with the LLM.
        if self.llm_client is None:
            return out
        verified: list[EntityCandidate] = []
        for cand in out:
            if self._llm_same_entity(mention, cand):
                cand.tier = 3
                cand.score = TIER_SCORES[3]
                cand.reason = f"T3: LLM xác nhận redirect ({cand.reason})"
                verified.append(cand)
            else:
                logger.info("T3 rejected alias %r -> %s", mention, cand.name)
        return verified

    def _known_redirects(self, mention: str) -> list[str]:
        """Titles that the mention is a redirect to (offline + online)."""
        cached = self._redirect_cache.get(mention)
        if cached is not None:
            return list(cached)
        out: list[str] = []

        # (a) locally known: an alias whose provenance is a wiki redirect
        surf = _surface(mention)
        for uid in self._exact_index.get(surf, []):
            record = self.records.get(uid)
            if record is None:
                continue
            for alias, tier in record.alias_tiers.items():
                if tier == 2 and _surface(alias) == surf:
                    out.append(record.name)

        # (b) live Wikipedia
        if self.wikipedia is not None and not self.offline:
            try:
                page = self.wikipedia.fetch_page(mention)
            except Exception as exc:  # noqa: BLE001
                logger.debug("wikipedia.fetch_page failed: %s", exc)
                page = None
            if page and page.get("exists"):
                title = str(page.get("title") or "")
                if title and _surface(title) != surf:
                    out.append(title)
        out = list(dict.fromkeys(out))
        self._redirect_cache[mention] = list(out)
        return out

    def _record_for_target(self, target: str) -> _Record | None:
        surf = _surface(target)
        for uid in self._exact_index.get(surf, []):
            record = self.records.get(uid)
            if record is not None:
                return record
        return None

    def _own_surfaces(self, record: _Record) -> set[str]:
        """Surface forms that are *own* names of the entity.

        Only the canonical name and the Wikidata aliases (tier 1) count --
        a Wikipedia redirect that carries a kinship title ("Thụy Ba Công
        chúa") is a name of **another** person and must not be treated as
        one of the entity's own titles.
        """
        out = {_surface(record.name)}
        for alias, tier in record.alias_tiers.items():
            if tier <= 1:
                out.add(_surface(alias))
        return out

    def _kinship_mismatch(self, mention: str, record: _Record) -> bool:
        """``True`` when the mention names a *relative* of the entity.

        ``Thụy Ba Công chúa`` redirects to ``Trần Hưng Đạo`` but denotes his
        adoptive aunt -- the title ``công chúa`` is not one of his own names.
        """
        mention_surf = _surface(mention)
        if not mention_surf:
            return False
        own = self._own_surfaces(record)
        for title in KINSHIP_TITLES:
            if not _contains_phrase(mention_surf, title):
                continue
            if any(_contains_phrase(surface, title) for surface in own):
                continue
            return True
        return False

    def _llm_same_entity(self, mention: str, cand: EntityCandidate) -> bool:
        """Ask the LLM whether ``mention`` and ``cand`` are the same entity."""
        client = self.llm_client
        if client is None:
            return False
        prompt = (
            "Cho cặp (tên gốc, tên biến thể) của một thực thể lịch sử Việt Nam. "
            "Chúng có cùng một thực thể không?\n"
            f"Tên gốc: {cand.name}\n"
            f"Tên biến thể: {mention}\n"
            'Trả về JSON: {"same": true/false, "reason": "...", "confidence": 0-1}\n'
            "Tiêu chí same=true: cùng tên thật/tước hiệu/thụy hiệu, cùng triều đại, "
            "cùng khoảng thời gian sống, và tên biến thể là cách gọi *của chính* thực thể "
            "đó -- không phải tên của người khác có quan hệ họ hàng, thầy trò, đồng liệu."
        )
        try:
            from ..llm.schemas import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                task="verify_alias",
                temperature=0.0,
                response_schema={
                    "type": "object",
                    "properties": {
                        "same": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["same"],
                },
            )
            payload = client.generate_json(request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM verify_alias failed (%s) -> giữ như T2", exc)
            return True
        if not isinstance(payload, Mapping):
            return True
        same = payload.get("same")
        if same is None:
            same = payload.get("is_same_entity")
        if same is None:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
            same = confidence >= 0.7
        return bool(same)

    # ------------------------------------------------------------------
    # tier 4
    # ------------------------------------------------------------------
    def _tier4_normalized(self, mention: str) -> list[EntityCandidate]:
        out: list[EntityCandidate] = []
        key = name_key(mention)
        if not key:
            return out
        for uid in self._key_index.get(key, []):
            record = self.records.get(uid)
            if record is None:
                continue
            if self._kinship_mismatch(mention, record):
                continue
            out.append(
                self._candidate_from_record(
                    record, 4, mention, "T4: khớp khóa chuẩn hóa (name_key)"
                )
            )
        return out

    # ------------------------------------------------------------------
    # tier 5
    # ------------------------------------------------------------------
    def _tier5_embedding(
        self, mention: str, hints: Mapping[str, Any]
    ) -> list[EntityCandidate]:
        embedder = self.embedder
        if embedder is None or not self.records:
            return []
        try:
            query = self._embed(embedder, mention)
        except Exception as exc:  # noqa: BLE001
            logger.debug("embed failed: %s", exc)
            return []
        if not query:
            return []
        scored = self._score_records(embedder, query)
        if not scored:
            return []
        top = [item for item in scored[:10] if item[0] >= self.embedding_min_score]
        if not top:
            return []
        top = self._filter_domain(
            [self._candidate_from_record(rec, 5, mention, "T5: embedding top-k") for _, rec in top],
            hints,
        )
        if not top:
            return []
        if self.llm_client is not None:
            choice = self._llm_disambiguate(mention, top)
            if choice is not None:
                top = [choice]
        out: list[EntityCandidate] = []
        for cand in top:
            cand.tier = 5
            cand.score = max(0.6, min(0.9, cand.score))
            if cand.score < self.min_confidence:
                continue
            out.append(cand)
        return out

    def _llm_disambiguate(
        self, mention: str, candidates: Sequence[EntityCandidate]
    ) -> EntityCandidate | None:
        client = self.llm_client
        if client is None:
            return candidates[0] if candidates else None
        options = "\n".join(
            f"- {c.uid} | {c.name} | {c.entity_type} | P31={','.join(c.p31)} | P27={','.join(c.p27)}"
            for c in candidates
        )
        prompt = (
            "Chọn thực thể lịch sử Việt Nam phù hợp nhất với chuỗi tên trong văn bản.\n"
            f"Chuỗi: {mention}\n"
            f"Ứng viên:\n{options}\n"
            "Chỉ chọn ứng viên cùng miền lịch sử Việt Nam, cùng loại thực thể, "
            "cùng triều đại và không mâu thuẫn thời gian.\n"
            'Trả về JSON: {"uid": "...", "confidence": 0-1, "reason": "..."}'
        )
        try:
            from ..llm.schemas import LLMMessage, LLMRequest

            request = LLMRequest(
                messages=[LLMMessage(role="user", content=prompt)],
                task="entity_link_select",
                temperature=0.0,
                # A response schema makes the model return the JSON object
                # directly instead of a fenced code block that the parser
                # would have to strip.
                response_schema={
                    "type": "object",
                    "properties": {
                        "uid": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["uid", "confidence"],
                },
            )
            payload = client.generate_json(request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM disambiguate failed (%s)", exc)
            return candidates[0] if candidates else None
        if not isinstance(payload, Mapping):
            return candidates[0] if candidates else None
        uid = str(payload.get("uid") or "")
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        chosen = next((c for c in candidates if c.uid == uid), None) if uid else None
        if chosen is None:
            chosen = candidates[0] if candidates else None
        if chosen is not None:
            chosen.extra_confidence = confidence
            if confidence:
                chosen.score = max(0.6, min(0.9, 0.5 * chosen.score + 0.5 * confidence))
        return chosen

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _embed(self, embedder: Any, text: str) -> list[float]:
        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached
        vector = embedder.embed_one(text)
        vector = [float(x) for x in vector]
        self._embed_cache[text] = vector
        return vector

    def _score_records(
        self, embedder: Any, query: list[float]
    ) -> list[tuple[float, _Record]]:
        """Cosine score of every indexed record against ``query``.

        The vectors of the records are embedded **in one batch** the first
        time they are needed and then cached in memory, so that resolving N
        mentions costs N + 1 embedding calls instead of N * len(records).
        """
        records = list(self.records.values())
        missing = [
            r
            for r in records
            if self._embed_text(r) not in self._embed_cache
        ]
        if missing:
            texts = [self._embed_text(r) for r in missing]
            try:
                vectors = embedder.embed(texts)
            except Exception as exc:  # noqa: BLE001
                logger.debug("batch embed failed: %s", exc)
                vectors = []
            for text, vec in zip(texts, vectors, strict=False):
                self._embed_cache[text] = [float(x) for x in vec]
            self.save_embed_cache()
        scored: list[tuple[float, _Record]] = []
        for record in records:
            vector = self._embed_cache.get(self._embed_text(record))
            if not vector:
                continue
            scored.append((_cosine(query, vector), record))
        scored.sort(key=lambda item: -item[0])
        return scored

    @staticmethod
    def _embed_text(record: _Record) -> str:
        parts = [record.name, *record.aliases[:8]]
        if record.description:
            parts.append(record.description)
        return ". ".join(p for p in parts if p)

    def _candidate_from_record(
        self, record: _Record, tier: int, matched: str, reason: str
    ) -> EntityCandidate:
        return EntityCandidate(
            uid=record.uid,
            name=record.name,
            wikidata_id=record.wikidata_id,
            aliases=list(record.aliases),
            entity_type=record.entity_type,
            score=TIER_SCORES.get(tier, 0.6),
            tier=tier,
            matched_alias=matched,
            reason=reason,
            p31=list(record.p31),
            p27=list(record.p27),
            year_min=record.year_min,
            year_max=record.year_max,
        )

    def _candidate_for_qid(
        self, qid: str, label: str, tier: int, matched: str, reason: str
    ) -> EntityCandidate:
        uid = self._qid_index.get(qid.upper()) or f"pe:{qid.upper()}"
        record = self.records.get(uid)
        if record is not None:
            return self._candidate_from_record(record, tier, matched, reason)
        p31: list[str] = []
        p27: list[str] = []
        if self.wikidata is not None and not self.offline:
            try:
                props = self.wikidata.entity_props(qid)
            except Exception as exc:  # noqa: BLE001
                logger.debug("wikidata.entity_props failed: %s", exc)
                props = {}
            p31 = [str(p) for p in (props.get("p31") or [])]
            p27 = [str(p) for p in (props.get("p27") or [])]
        return EntityCandidate(
            uid=uid,
            name=label or qid,
            wikidata_id=qid.upper(),
            entity_type=self._type_from_p31(p31),
            score=TIER_SCORES.get(tier, 0.6),
            tier=tier,
            matched_alias=matched,
            reason=reason,
            p31=p31,
            p27=p27,
        )

    # -- domain filter ---------------------------------------------------
    @staticmethod
    def _type_from_p31(p31: Sequence[str]) -> str:
        for pid in p31 or []:
            if pid in P31_TYPE_MAP:
                return P31_TYPE_MAP[pid]
        return "Entity"

    def _filter_domain(
        self, candidates: list[EntityCandidate], hints: Mapping[str, Any]
    ) -> list[EntityCandidate]:
        """Keep only candidates compatible with the requested type/domain."""
        wanted_type = str(hints.get("entity_type") or hints.get("type") or "").strip()
        year = hints.get("year")
        out: list[EntityCandidate] = []
        for cand in candidates:
            cand_type = cand.entity_type
            if cand_type in ("Entity", ""):
                cand_type = self._type_from_p31(cand.p31)
            if wanted_type:
                if cand_type != wanted_type and cand_type != "Entity":
                    # a place/period mismatch is a hard filter (Hùng Vương)
                    continue
            if not wanted_type and set(cand.p31) & MODERN_PLACE_P31:
                # A modern administrative unit must not silently take the place
                # of a historical person/period with the same label.
                cand.score = min(cand.score, 0.6)
            if cand.p27 and not (set(cand.p27) & VN_HISTORY_QIDS):
                # outside the Vietnamese historical domain -> heavy penalty
                cand.score = min(cand.score, 0.6)
            if year is not None and cand.year_min is not None and cand.year_max is not None:
                try:
                    y = int(year)
                except (TypeError, ValueError):
                    y = None
                if y is not None and not (cand.year_min - 20 <= y <= cand.year_max + 20):
                    cand.score = min(cand.score, 0.6)
            out.append(cand)
        return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b, strict=False):
        num += x * y
        da += x * x
        db += y * y
    if da <= 0 or db <= 0:
        return 0.0
    return num / ((da**0.5) * (db**0.5))


def uid_for_qid(qid: str, prefix: str = "pe:") -> str:
    """Stable uid for a Wikidata QID."""
    qid = (qid or "").strip().upper()
    if not qid:
        return ""
    if qid.startswith(prefix):
        return qid
    return f"{prefix}{qid}"


def uid_for_name(name: str, entity_type: str = "Entity") -> str:
    """Stable uid for a QID-less entity (normalised key + hash)."""
    key = name_key(name)
    return stable_uid(key or name, entity_type, prefix="ent_")
