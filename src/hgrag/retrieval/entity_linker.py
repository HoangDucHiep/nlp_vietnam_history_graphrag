"""Entity linking for the QA layer (a thin, opinionated wrapper around
:class:`~hgrag.kg.alias.AliasResolver`).

The :class:`~hgrag.kg.alias.AliasResolver` implements the *QID-first* 5-tier
strategy described in ``docs/07_xac_dinh_thuc_the.md`` (ADR-07).  That module
is designed for the **offline KG build** step: it may hit Wikidata/Wikipedia,
it caches aggressively and it returns a full candidate list.

The QA layer needs something slightly different:

* a *single* best anchor per mention (so that the graph traversal starts from
  a reliable node);
* a hard confidence threshold (``entity_link_min_confidence``) so that a bad
  link never poisons the traversal -- if nothing matches, an **empty** list is
  returned (we never invent an entity);
* the *other* candidates must stay visible so the UI can explain why a link
  was chosen;
* an **ablation switch** (``use_resolution=False``) that keeps only tier 0
  (an exact Wikidata QID) so that ablation A1/A2 can measure how much the
  fuzzy tiers contribute.

The module also extracts Vietnamese proper nouns from a free-form question
with a conservative regex so that the QA pipeline works even when the
query-understanding step (Wave 4) is not available.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "EntityLinker",
    "LinkedEntity",
    "extract_vietnamese_mentions",
    "VIETNAMESE_STOPWORDS",
]

#: Common Vietnamese words that are capitalised by mistake (or that are pure
#: function words) and must never be treated as an entity mention.
VIETNAMESE_STOPWORDS: frozenset[str] = frozenset(
    {
        "ai",
        "bao",
        "bao gio",
        "bao lau",
        "bay",
        "bi",
        "biet",
        "boi",
        "cai",
        "can",
        "chuyen",
        "co",
        "coi",
        "cu",
        "cuoc",
        "da",
        "dang",
        "day",
        "de",
        "den",
        "dieu",
        "do",
        "duoc",
        "gia",
        "gi",
        "giang",
        "hay",
        "ho",
        "hien",
        "hieu",
        "khi",
        "khong",
        "la",
        "lai",
        "lan",
        "len",
        "lien",
        "loi",
        "luc",
        "mien",
        "mot",
        "muon",
        "nay",
        "nen",
        "neu",
        "nhung",
        "nhu",
        "o",
        "phai",
        "qua",
        "ra",
        "rang",
        "roi",
        "sao",
        "se",
        "so",
        "sau",
        "song",
        "su",
        "tai",
        "the",
        "theo",
        "thi",
        "thoi",
        "thu",
        "toi",
        "tren",
        "trong",
        "truoc",
        "tu",
        "va",
        "ve",
        "vi",
        "vi sao",
        "voi",
        "von",
        "vu",
        "vua",
        "xay",
        "viem",
        "gioi",
        "khu",
        "moi",
        "rat",
        "tung",
        "vai",
    }
)

#: A "word" is a run of unicode letters (no digits, no underscore).
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: An explicit QID (``Q511375``).
_QID_RE = re.compile(r"^Q\d+$", re.IGNORECASE)


# --------------------------------------------------------------------------
# Mention extraction
# --------------------------------------------------------------------------


def extract_vietnamese_mentions(text: str, max_words: int = 4) -> list[str]:
    """Extract candidate Vietnamese proper nouns from ``text``.

    The heuristic is deliberately *conservative*: a mention is a run of
    capitalised words (2 to ``max_words`` of them).  Vietnamese proper nouns
    are capitalised (``Trần Hưng Đạo``, ``Trận Bạch Đằng``), function words
    are not, so the rule has a very low false-positive rate.

    >>> extract_vietnamese_mentions("Vì sao nhà Trần gả Trần Khánh Dư?")
    ['Trần Khánh Dư']
    """
    if not text:
        return []
    words = list(_WORD_RE.finditer(text))
    groups: list[list[str]] = []
    current: list[str] = []
    for match in words:
        token = match.group(0)
        starts_upper = token[0].isupper()
        if starts_upper and len(token) > 1:
            current.append(token)
            if len(current) > max_words:
                current = current[-max_words:]
        else:
            if current:
                groups.append(current)
            current = []
    if current:
        groups.append(current)

    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for size in range(min(len(group), max_words), 1, -1):
            for start in range(0, len(group) - size + 1):
                candidate = " ".join(group[start : start + size])
                key = candidate.casefold()
                if key in VIETNAMESE_STOPWORDS or key in seen:
                    continue
                seen.add(key)
                out.append(candidate)
    # keep the original order but drop mentions contained in a longer one
    ordered: list[str] = []
    for mention in out:
        if any(mention != other and mention in other for other in out):
            continue
        ordered.append(mention)
    return ordered


# --------------------------------------------------------------------------
# Linked entity
# --------------------------------------------------------------------------


@dataclass
class LinkedEntity:
    """A mention resolved to a KG entity (with its losing candidates)."""

    mention: str
    uid: str
    name: str
    wikidata_id: str = ""
    entity_type: str = "Entity"
    confidence: float = 0.0
    tier: int = 5
    matched_alias: str = ""
    reason: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention": self.mention,
            "uid": self.uid,
            "name": self.name,
            "wikidata_id": self.wikidata_id,
            "entity_type": self.entity_type,
            "confidence": round(float(self.confidence), 4),
            "tier": self.tier,
            "matched_alias": self.matched_alias,
            "reason": self.reason,
            "candidates": list(self.candidates),
        }


# --------------------------------------------------------------------------
# Linker
# --------------------------------------------------------------------------


def _retrieval_config(settings: Any) -> dict[str, Any]:
    """Read the ``retrieval`` block from a settings object (or a plain dict)."""
    if settings is None:
        return {}
    raw = getattr(settings, "retrieval", None)
    if raw is None and isinstance(settings, dict):
        raw = settings.get("retrieval")
    return dict(raw or {})


class EntityLinker:
    """Resolve mentions to KG anchors for the QA pipeline.

    Parameters
    ----------
    resolver:
        An :class:`~hgrag.kg.alias.AliasResolver` (or anything exposing
        ``resolve(mention, hints) -> list[EntityCandidate]``).
    settings:
        Optional project settings; ``retrieval.entity_link_min_confidence``
        provides the confidence threshold (default ``0.7``).
    use_resolution:
        ``False`` disables the fuzzy tiers (T1-T5) and keeps only tier 0
        (an exact Wikidata QID).  Used by the ablation A1/A2.
    """

    def __init__(
        self,
        resolver: Any,
        settings: Any = None,
        use_resolution: bool = True,
    ) -> None:
        self.resolver = resolver
        self.settings = settings
        self.use_resolution = bool(use_resolution)
        cfg = _retrieval_config(settings)
        self.min_confidence = float(cfg.get("entity_link_min_confidence", 0.7))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def link(
        self,
        mentions: list[str],
        hints: dict[str, Any] | None = None,
    ) -> list[LinkedEntity]:
        """Link a list of mentions; duplicated uids are dropped."""
        out: list[LinkedEntity] = []
        seen_uids: set[str] = set()
        for mention in mentions or []:
            text = (mention or "").strip()
            if not text:
                continue
            linked = self.link_one(text, hints)
            if linked is None:
                continue
            if linked.uid in seen_uids:
                continue
            seen_uids.add(linked.uid)
            out.append(linked)
        return out

    def link_one(
        self, mention: str, hints: dict[str, Any] | None = None
    ) -> LinkedEntity | None:
        """Link a single mention (``None`` when nothing matches confidently)."""
        mention = (mention or "").strip()
        if not mention:
            return None
        if not self.use_resolution:
            candidates = self._tier0_candidates(mention)
        else:
            candidates = self._resolve(mention, hints)
        if not candidates:
            return None
        best = candidates[0]
        if float(best.get("score", 0.0)) < self.min_confidence:
            logger.debug(
                "Bỏ qua mention %r: confidence %.2f < %.2f",
                mention,
                float(best.get("score", 0.0)),
                self.min_confidence,
            )
            return None
        others = [dict(c) for c in candidates[1:]]
        return LinkedEntity(
            mention=mention,
            uid=str(best.get("uid") or ""),
            name=str(best.get("name") or mention),
            wikidata_id=str(best.get("wikidata_id") or ""),
            entity_type=str(best.get("entity_type") or "Entity"),
            confidence=float(best.get("score") or 0.0),
            tier=int(best.get("tier", 5) if best.get("tier") is not None else 5),
            matched_alias=str(best.get("matched_alias") or ""),
            reason=str(best.get("reason") or ""),
            candidates=others,
        )

    def link_from_query(
        self,
        question: str,
        entities: list[str] | None = None,
        hints: dict[str, Any] | None = None,
    ) -> list[LinkedEntity]:
        """Link the entities of a question.

        ``entities`` are the mentions extracted by the query-understanding
        module (Wave 4).  When it is ``None`` (or empty) the mentions are
        extracted from the question itself with :func:`extract_vietnamese_mentions`.
        """
        mentions = [str(e).strip() for e in (entities or []) if str(e).strip()]
        if not mentions:
            mentions = extract_vietnamese_mentions(question)
        return self.link(mentions, hints)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _resolve(self, mention: str, hints: dict[str, Any] | None) -> list[dict[str, Any]]:
        resolver = self.resolver
        if resolver is None:
            return []
        try:
            raw = resolver.resolve(mention, hints)
        except TypeError:
            try:
                raw = resolver.resolve(mention)
            except Exception as exc:  # noqa: BLE001
                logger.debug("resolve(%r) failed: %s", mention, exc)
                return []
        except Exception as exc:  # noqa: BLE001
            logger.debug("resolve(%r) failed: %s", mention, exc)
            return []
        return [self._to_dict(c) for c in raw or []]

    @staticmethod
    def _to_dict(candidate: Any) -> dict[str, Any]:
        if isinstance(candidate, dict):
            return dict(candidate)
        to_dict = getattr(candidate, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        return {
            "uid": str(getattr(candidate, "uid", "") or ""),
            "name": str(getattr(candidate, "name", "") or ""),
            "wikidata_id": str(getattr(candidate, "wikidata_id", "") or ""),
            "entity_type": str(getattr(candidate, "entity_type", "Entity") or "Entity"),
            "score": float(getattr(candidate, "score", 0.0) or 0.0),
            "tier": int(getattr(candidate, "tier", 5) or 5),
            "matched_alias": str(getattr(candidate, "matched_alias", "") or ""),
            "reason": str(getattr(candidate, "reason", "") or ""),
        }

    def _tier0_candidates(self, mention: str) -> list[dict[str, Any]]:
        """Only tier 0: the mention *is* a QID, or a QID already in the store."""
        resolver = self.resolver
        if resolver is None:
            return []
        out: list[dict[str, Any]] = []

        # (a) the mention is literally a QID
        if _QID_RE.match(mention):
            qid = mention.upper()
            uid = ""
            name = mention
            entity_type = "Entity"
            aliases: list[str] = []
            records = getattr(resolver, "records", None)
            if isinstance(records, dict):
                for candidate_uid, record in records.items():
                    if str(getattr(record, "wikidata_id", "") or "").upper() == qid:
                        uid = candidate_uid
                        name = str(getattr(record, "name", mention) or mention)
                        entity_type = str(getattr(record, "entity_type", "Entity") or "Entity")
                        aliases = list(getattr(record, "aliases", []) or [])
                        break
            if not uid:
                uid = f"pe:{qid}"
            out.append(
                {
                    "uid": uid,
                    "name": name,
                    "wikidata_id": qid,
                    "entity_type": entity_type,
                    "aliases": aliases,
                    "score": 1.0,
                    "tier": 0,
                    "matched_alias": mention,
                    "reason": "T0: mention là QID đã biết (ablation: chỉ dùng tầng T0)",
                }
            )

        # (b) the mention is an entity of the KG whose QID is known
        store = getattr(resolver, "store", None)
        if store is not None:
            try:
                rows = store.find_entity_by_name(mention) or []
            except Exception as exc:  # noqa: BLE001
                logger.debug("find_entity_by_name(%r) failed: %s", mention, exc)
                rows = []
            for row in rows:
                qid = str(row.get("wikidata_id") or "")
                if not qid:
                    continue
                out.append(
                    {
                        "uid": str(row.get("uid") or f"pe:{qid.upper()}"),
                        "name": str(row.get("name") or mention),
                        "wikidata_id": qid.upper(),
                        "entity_type": str(row.get("entity_type") or "Entity"),
                        "aliases": list(row.get("aliases") or []),
                        "score": 1.0,
                        "tier": 0,
                        "matched_alias": str(row.get("name") or mention),
                        "reason": "T0: QID đã biết trong KG (ablation: chỉ dùng tầng T0)",
                    }
                )
                break
        return out
