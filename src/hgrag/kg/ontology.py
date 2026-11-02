"""VHE ontology: 35 event types, 11 argument roles, role -> KG relation map.

The ontology is loaded from the VHE dataset (``hoangthuytruc/vhe-dataset``)
through :func:`hgrag.data.vhe.load_ontology`.  When the dataset cannot be
reached (offline CI, no cached copy) the module falls back to the built-in
copy of the very same schema -- the pipeline must always have a usable
ontology (ADR-01).

Roles
-----
VHE spells its argument roles in lower case (``agent``, ``beneficiary`` ...).
The KG layer uses UPPER_SNAKE everywhere, so every role is normalised on
load::

    agent        -> AGENT
    beneficiary  -> BENEFICIARY
    experiencer  -> EXPERIENCER
    instrument   -> INSTRUMENT
    content      -> CONTENT
    ...

The 11 roles are split into four semantic groups (see :data:`ROLE_GROUPS`):

``TEMPORAL``   (2) -- ``TEMPORAL``, ``SOURCE``
    *When* / *from where the information comes* -- the temporal context of the
    event.  ``SOURCE`` (where the information is taken from) is grouped here
    because it also answers "where does this claim come from in time".
``LOCATION``   (1) -- ``LOCATION``
    *Where* the event happened.
``PARTICIPANT`` (8) -- ``AGENT``, ``BENEFICIARY``, ``EXPERIENCER``, ``FORCE``,
    ``INSTRUMENT``, ``CONTENT``, ``GOAL``, ``THEME``
    *Who / what takes part* in the event (the actor, the affected party, the
    means, the goal, the subject matter ...).

Relation types
--------------
:data:`RELATION_TYPES` maps an argument role onto the KG edge type used to
link the *argument node* with the *event node* (docs/02 §3.2)::

    AGENT / BENEFICIARY / EXPERIENCER / FORCE / INSTRUMENT -> PARTICIPATED_IN
    LOCATION                                              -> LOCATED_IN
    TEMPORAL                                              -> OCCURRED_IN
    SOURCE                                                -> SUPPORTED_BY
    THEME / GOAL / CONTENT                                -> ABOUT
    CAUSE                                                 -> CAUSE_OF
    PRECONDITION                                           -> PRECONDITION_OF
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.vhe import DEFAULT_EVENT_TYPES, DEFAULT_ROLES, load_ontology

logger = logging.getLogger(__name__)

__all__ = [
    "ARGUMENT_ROLES",
    "EVENT_TYPES",
    "ONTOLOGY",
    "Ontology",
    "RELATION_TYPES",
    "ROLE_GROUPS",
    "EVENT_TYPE_ALIASES",
    "build_ontology",
    "normalize_event_type",
    "normalize_role",
    "relation_for_role",
    "validate_event",
    "edge_is_participant_to_event",
    "PARTICIPANT_TO_EVENT_RELATIONS",
    "EVENT_TO_ARGUMENT_RELATIONS",
]

#: Repository root (used to locate the cached VHE copy under ``data/raw/vhe``).
#: ``<repo>/src/hgrag/kg/ontology.py`` -> parents[3] == ``<repo>``.
REPO_ROOT = Path(__file__).resolve().parents[3]
VHE_CACHE_DIR = REPO_ROOT / "data" / "raw" / "vhe"

# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------

#: The four role groups (ADR-07 / docs/02 §3.1).
ROLE_GROUPS: dict[str, tuple[str, ...]] = {
    "TEMPORAL": ("TEMPORAL", "SOURCE"),
    "LOCATION": ("LOCATION",),
    "PARTICIPANT": (
        "AGENT",
        "BENEFICIARY",
        "EXPERIENCER",
        "FORCE",
        "INSTRUMENT",
        "CONTENT",
        "GOAL",
        "THEME",
    ),
}

#: Role -> KG relation type.
RELATION_TYPES: dict[str, str] = {
    # participants
    "AGENT": "PARTICIPATED_IN",
    "BENEFICIARY": "PARTICIPATED_IN",
    "EXPERIENCER": "PARTICIPATED_IN",
    "FORCE": "PARTICIPATED_IN",
    "INSTRUMENT": "PARTICIPATED_IN",
    # context
    "LOCATION": "LOCATED_IN",
    "TEMPORAL": "OCCURRED_IN",
    "SOURCE": "SUPPORTED_BY",
    # topic
    "THEME": "ABOUT",
    "GOAL": "ABOUT",
    "CONTENT": "ABOUT",
    # causality (VHE relation types)
    "CAUSE": "CAUSE_OF",
    "PRECONDITION": "PRECONDITION_OF",
}

#: Fallback for roles that carry no explicit mapping.
DEFAULT_RELATION = "ABOUT"

#: Relation types whose edge points **from the participant to the event**
#: (``person -[:PARTICIPATED_IN]-> event``).  The traversal of the evaluation
#: set starts at a named entity (the question anchor) and has to reach the
#: event first, then the context arguments of the event -- so the
#: participant edge is oriented towards the event (docs/02 §3.2).
PARTICIPANT_TO_EVENT_RELATIONS: frozenset[str] = frozenset(
    {
        "PARTICIPATED_IN",
        "AGENT_OF",
        "INVOLVES",
        "CAUSE_OF",
        "PRECONDITION_OF",
    }
)

#: Relation types whose edge points **from the event to the argument**
#: (``event -[:LOCATED_IN]-> place``).
EVENT_TO_ARGUMENT_RELATIONS: frozenset[str] = frozenset(
    {
        "ABOUT",
        "LOCATED_IN",
        "OCCURRED_IN",
        "OCCURRED_AT",
        "HAPPENED_AT",
    }
)


def edge_is_participant_to_event(rel_type: str) -> bool:
    """Whether an argument edge is oriented ``participant -> event``.

    >>> edge_is_participant_to_event("PARTICIPATED_IN")
    True
    >>> edge_is_participant_to_event("LOCATED_IN")
    False
    """
    rel = str(rel_type or "").strip().upper()
    if rel in PARTICIPANT_TO_EVENT_RELATIONS:
        return True
    if rel in EVENT_TO_ARGUMENT_RELATIONS:
        return False
    # Unknown relation type: fall back to the participant direction so that
    # the entity anchor can still reach the event.
    return True


_ROLE_RE = re.compile(r"[^A-Z0-9]+")

#: Free-text roles an LLM commonly produces mapped onto the 11 VHE roles.
#: Without this, an extracted event would be rejected by
#: :func:`validate_event` and the chunk would yield no relation at all.
ROLE_ALIASES: dict[str, str] = {
    # participants
    "COMMANDER": "AGENT",
    "LEADER": "AGENT",
    "ACTOR": "AGENT",
    "ACTOR_1": "AGENT",
    "ACTOR_2": "AGENT",
    "PARTICIPANT": "AGENT",
    "PERSON": "AGENT",
    "PROTAGONIST": "AGENT",
    "WINNER": "AGENT",
    "OPPORTUNIST": "AGENT",
    "INITIATOR": "AGENT",
    "VICTIM": "EXPERIENCER",
    "AFFECTED": "EXPERIENCER",
    "PATIENT": "EXPERIENCER",
    "RECIPIENT": "BENEFICIARY",
    "RECIPIENT_1": "BENEFICIARY",
    "RECIPIENT_2": "BENEFICIARY",
    "OPPOSITION": "FORCE",
    "OPPOSING_FORCE": "FORCE",
    "ENEMY": "FORCE",
    "ADVERSARY": "FORCE",
    "MILITARY_FORCE": "FORCE",
    "ARMY": "FORCE",
    "TROOPS": "FORCE",
    "WEAPON": "INSTRUMENT",
    "TOOL": "INSTRUMENT",
    "MEANS": "INSTRUMENT",
    "VEHICLE": "INSTRUMENT",
    "DOCUMENT": "CONTENT",
    "TEXT": "CONTENT",
    "SUBJECT_MATTER": "THEME",
    "TOPIC": "THEME",
    "PURPOSE": "GOAL",
    "OBJECTIVE": "GOAL",
    # context
    "PLACE": "LOCATION",
    "LOCATION_1": "LOCATION",
    "LOCATION_2": "LOCATION",
    "SITE": "LOCATION",
    "SPOT": "LOCATION",
    "TIME": "TEMPORAL",
    "DATE": "TEMPORAL",
    "PERIOD": "TEMPORAL",
    "YEAR": "TEMPORAL",
    "WHEN": "TEMPORAL",
    "SOURCE_OF_INFORMATION": "SOURCE",
    "REFERENCE": "SOURCE",
    "CITATION": "SOURCE",
}


def normalize_role(role: str) -> str:
    """Normalise an argument role to UPPER_SNAKE.

    >>> normalize_role("agent")
    'AGENT'
    >>> normalize_role("Beneficiary")
    'BENEFICIARY'
    """
    if not role:
        return ""
    text = _ROLE_RE.sub("_", str(role).strip().upper()).strip("_")
    # Free-text LLM roles ("Commander", "Opponent", ...) map onto a VHE role.
    return ROLE_ALIASES.get(text, text)


#: Common English / capitalised spellings mapped onto the VHE event types.
EVENT_TYPE_ALIASES: dict[str, str] = {
    "be-born": "be-born",
    "birth": "be-born",
    "born": "be-born",
    "birthday": "be-born",
    "marry": "marry",
    "marriage": "marry",
    "wedding": "marry",
    "divorce": "divorce",
    "injure": "injure",
    "injury": "injure",
    "wound": "injure",
    "die": "die",
    "death": "die",
    "deceased": "die",
    "transport": "transport",
    "transportation": "transport",
    "start-position": "start-position",
    "start_position": "start-position",
    "end-position": "end-position",
    "end_position": "end-position",
    "elect": "elect",
    "election": "elect",
    "nominate": "nominate",
    "nomination": "nominate",
    "appoint": "nominate",
    "arrest-jail": "arrest-jail",
    "arrest": "arrest-jail",
    "jail": "arrest-jail",
    "release-parole": "release-parole",
    "release": "release-parole",
    "trial-hearing": "trial-hearing",
    "trial": "trial-hearing",
    "hearing": "trial-hearing",
    "charge-indict": "charge-indict",
    "charge": "charge-indict",
    "indict": "charge-indict",
    "sue": "sue",
    "lawsuit": "sue",
    "convict": "convict",
    "conviction": "convict",
    "sentence": "sentence",
    "sentencing": "sentence",
    "fine": "fine",
    "execute": "execute",
    "execution": "execute",
    "extradite": "extradite",
    "extradition": "extradite",
    "acquit": "acquit",
    "acquittal": "acquit",
    "appeal": "appeal",
    "pardon": "pardon",
    "transfer-ownership": "transfer-ownership",
    "transfer money": "transfer-money",
    "transfer-money": "transfer-money",
    "start-org": "start-org",
    "start organization": "start-org",
    "found": "start-org",
    "establish": "start-org",
    "end-org": "end-org",
    "dissolve": "end-org",
    "merge-org": "merge-org",
    "merge": "merge-org",
    "declare-bankruptcy": "declare-bankruptcy",
    "bankruptcy": "declare-bankruptcy",
    "attack": "attack",
    "battle": "attack",
    "war": "attack",
    "conflict": "attack",
    "military-conflict": "attack",
    "demonstrate": "demonstrate",
    "protest": "demonstrate",
    "strike": "demonstrate",
    "meet": "meet",
    "meeting": "meet",
    "phone-write": "phone-write",
    "correspondence": "phone-write",
    "natural-disaster": "natural-disaster",
    "disaster": "natural-disaster",
    "flood": "natural-disaster",
    "earthquake": "natural-disaster",
    "natural-phenomenon": "natural-phenomenon",
    "phenomenon": "natural-phenomenon",
}


def normalize_event_type(event_type: str) -> str:
    """Normalise an event type to the VHE spelling (``kebab-case``).

    Unknown types are returned in their normalised kebab-case form so that
    :func:`validate_event` can report them as invalid.

    >>> normalize_event_type("Conflict")
    'attack'
    >>> normalize_event_type("Start Position")
    'start-position'
    """
    if not event_type:
        return ""
    text = str(event_type).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if text in EVENT_TYPE_ALIASES:
        return EVENT_TYPE_ALIASES[text]
    return text


# --------------------------------------------------------------------------
# Ontology container
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Ontology:
    """Immutable description of the VHE ontology used by the KG builder."""

    event_types: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    relation_types: Mapping[str, str] = field(default_factory=dict)
    role_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    schema: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source: str = "fallback"

    # -- helpers ---------------------------------------------------------
    def is_valid_event_type(self, event_type: str) -> bool:
        return normalize_event_type(event_type) in self.event_types

    def is_valid_role(self, role: str) -> bool:
        return normalize_role(role) in self.roles

    def group_of(self, role: str) -> str:
        """Return the semantic group of ``role`` (``""`` when unknown)."""
        norm = normalize_role(role)
        for group, members in self.role_groups.items():
            if norm in members:
                return group
        return ""

    def relation_for_role(self, role: str) -> str:
        """KG relation type used for an argument role."""
        return relation_for_role(role, self.relation_types)

    def allowed_roles(self, event_type: str) -> tuple[str, ...]:
        """Roles allowed by the VHE schema for ``event_type``."""
        et = normalize_event_type(event_type)
        return tuple(self.schema.get(et, self.roles))

    def validate_event(
        self, event_type: str, arguments: Sequence[Mapping[str, Any]]
    ) -> list[str]:
        """Return the list of problems of an extracted event (empty = valid)."""
        return validate_event(event_type, arguments, self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_types": list(self.event_types),
            "roles": list(self.roles),
            "relation_types": dict(self.relation_types),
            "role_groups": {k: list(v) for k, v in self.role_groups.items()},
            "source": self.source,
            "n_event_types": len(self.event_types),
            "n_roles": len(self.roles),
        }


def _schema_roles(raw: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for et, roles in (raw or {}).items():
        if not isinstance(roles, Iterable) or isinstance(roles, (str, bytes)):
            continue
        norm = [normalize_role(r) for r in roles if r]
        out[normalize_event_type(et)] = tuple(dict.fromkeys(norm))
    return out


def build_ontology(
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> Ontology:
    """Load the VHE ontology (with a built-in fallback).

    ``source`` is ``"vhe-dataset"`` when the real dataset was loaded and
    ``"fallback"`` otherwise.
    """
    raw: dict[str, Any] = {}
    try:
        raw = load_ontology(cache_dir=cache_dir or VHE_CACHE_DIR, offline=offline) or {}
    except Exception as exc:  # noqa: BLE001 - ontology must never break the pipeline
        logger.warning("Không nạp được ontology VHE (%s) -> dùng fallback", exc)
        raw = {}

    event_types = tuple(
        dict.fromkeys(normalize_event_type(e) for e in (raw.get("event_types") or ()) if e)
    )
    roles = tuple(
        dict.fromkeys(normalize_role(r) for r in (raw.get("roles") or ()) if r)
    )
    if not event_types:
        event_types = tuple(dict.fromkeys(DEFAULT_EVENT_TYPES))
    if not roles:
        roles = tuple(dict.fromkeys(normalize_role(r) for r in DEFAULT_ROLES))

    source = str(raw.get("source") or "fallback")
    if source != "vhe-dataset":
        source = "fallback"

    return Ontology(
        event_types=event_types,
        roles=roles,
        relation_types=dict(RELATION_TYPES),
        role_groups={k: tuple(v) for k, v in ROLE_GROUPS.items()},
        schema=_schema_roles(raw.get("schema") or {}),
        source=source,
    )


#: Module-level ontology (loaded once at import time).
ONTOLOGY: Ontology = build_ontology()

#: The 35 VHE event types.
EVENT_TYPES: tuple[str, ...] = ONTOLOGY.event_types
#: The 11 VHE argument roles (UPPER_SNAKE).
ARGUMENT_ROLES: tuple[str, ...] = ONTOLOGY.roles


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def relation_for_role(
    role: str, relation_types: Mapping[str, str] | None = None
) -> str:
    """Return the KG relation type for an argument role.

    >>> relation_for_role("agent")
    'PARTICIPATED_IN'
    >>> relation_for_role("beneficiary")
    'PARTICIPATED_IN'
    >>> relation_for_role("location")
    'LOCATED_IN'
    >>> relation_for_role("temporal")
    'OCCURRED_IN'
    >>> relation_for_role("source")
    'SUPPORTED_BY'
    """
    table = dict(relation_types or RELATION_TYPES)
    return table.get(normalize_role(role), DEFAULT_RELATION)


def validate_event(
    event_type: str,
    arguments: Sequence[Mapping[str, Any]],
    ontology: Ontology | None = None,
) -> list[str]:
    """Validate one extracted event.

    Returns a list of human readable error strings; an empty list means the
    event is valid.  Checks:

    * the event type belongs to the 35 VHE event types;
    * every argument role belongs to the 11 VHE roles and carries a name;
    * the event has an ``AGENT`` argument (VHE events are agent-centric).
    """
    onto = ontology or ONTOLOGY
    errors: list[str] = []
    et = normalize_event_type(event_type)
    if et not in onto.event_types:
        errors.append(f"event_type không hợp lệ: {event_type!r}")

    args = list(arguments or [])
    seen_roles: list[str] = []
    for index, arg in enumerate(args):
        if not isinstance(arg, Mapping):
            errors.append(f"argument #{index} không phải object")
            continue
        role = normalize_role(str(arg.get("role") or ""))
        if not role:
            errors.append(f"argument #{index} thiếu role")
            continue
        if role not in onto.roles:
            errors.append(f"role không hợp lệ: {arg.get('role')!r}")
        if not str(arg.get("name") or "").strip():
            errors.append(f"argument #{index} ({role}) thiếu name")
        if role not in seen_roles:
            seen_roles.append(role)

    allowed = onto.allowed_roles(et)
    if "AGENT" in allowed and "AGENT" not in seen_roles:
        errors.append(f"event '{et}' thiếu argument AGENT")
    return errors


def event_types() -> tuple[str, ...]:
    """Return the 35 VHE event types."""
    return ONTOLOGY.event_types


def argument_roles() -> tuple[str, ...]:
    """Return the 11 VHE argument roles."""
    return ONTOLOGY.roles
