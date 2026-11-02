"""Shared Vietnamese labels for the VHE event types and argument roles.

Why a shared module
-------------------
Two very different parts of the project need to agree *character by
character* on how an event is written in Vietnamese:

* the **knowledge graph** (:mod:`hgrag.kg.vhe_gold`) names every VHE gold
  event node after its arguments (``"Đế Minh sinh ra Đế Nghi"``);
* the **evaluation dataset** (:mod:`hgrag.eval.dataset_builder`) builds the
  template questions *and* the gold paths out of the very same labels.

If the two modules use different verbs, the graph path found at retrieval
time is called ``"Đế Minh sinh Đế Nghi"`` while the gold path says
``"Đế Minh sinh ra Đế Nghi"`` -- the ``path_hit_rate`` metric then reports a
miss for a fact the system retrieved perfectly.  Sharing the mapping removes
that whole class of silent errors.

The VHE trigger words (``"sinh"``, ``"đánh"``, ``"chết"`` ...) are short and
therefore ambiguous (``"sinh"`` vs ``"sinh ra"``), so the *canonical* label of
an event type is the :data:`EVENT_TYPE_VN` verb, not the raw trigger word.
The trigger word is only used as a fallback for the (rare) event types that
have no canonical Vietnamese verb.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = [
    "EVENT_TYPE_VN",
    "ROLE_VN",
    "clean_mention",
    "event_verb",
    "event_label",
    "role_vn",
    "is_proper_mention",
    "is_usable_entity",
    "pick_mention",
    "GENERIC_MENTIONS",
]

#: Vietnamese label (verb) of every VHE event type.
EVENT_TYPE_VN: dict[str, str] = {
    "be-born": "sinh ra",
    "marry": "cưới",
    "divorce": "ly hôn",
    "injure": "làm bị thương",
    "die": "qua đời",
    "transport": "di chuyển",
    "start-position": "đảm nhiệm chức vụ",
    "end-position": "thôi giữ chức vụ",
    "elect": "được bầu",
    "nominate": "được tiến cử",
    "arrest-jail": "bị bắt giam",
    "release-parole": "được thả tự do",
    "trial-hearing": "ra tòa xét xử",
    "charge-indict": "bị khởi tứ",
    "sue": "kiện tụng",
    "convict": "bị kết án",
    "sentence": "bị tuyên án",
    "fine": "bị phạt tiền",
    "execute": "bị thi hành án",
    "extradite": "bị dẫn đội",
    "acquit": "được tuyên vô tội",
    "appeal": "kêu án",
    "pardon": "được tha tội",
    "transfer-ownership": "chuyển nhượng",
    "transfer-money": "chuyển tiền",
    "start-org": "thành lập",
    "end-org": "giải thể",
    "merge-org": "sáp nhập",
    "declare-bankruptcy": "phá sản",
    "attack": "tấn công",
    "demonstrate": "biểu tình",
    "meet": "gặp gỡ",
    "phone-write": "gửi thư",
    "natural-disaster": "thiên tai",
    "natural-phenomenon": "hiện tượng tự nhiên",
}

#: Vietnamese label of every VHE argument role.
ROLE_VN: dict[str, str] = {
    "agent": "chủ thể",
    "theme": "đối tượng",
    "beneficiary": "người thụ hưởng",
    "experiencer": "người trải nghiệm",
    "instrument": "phương tiện",
    "content": "nội dung",
    "goal": "mục đích",
    "force": "lực lượng",
    "location": "địa điểm",
    "temporal": "thời gian",
    "source": "nguồn",
}

_BRACKET_RE = re.compile(r"^\w+\s*\[([^\]]+)\]$")

#: Maximum length of an event label (the label becomes a Neo4j node name).
MAX_LABEL_LEN = 180


def clean_mention(mention: str) -> str:
    """Strip the ``vua [Kinh Dương Vương]`` bracket notation of VHE.

    >>> clean_mention("vua [Kinh Dương Vương]")
    'Kinh Dương Vương'
    """
    text = (mention or "").strip()
    match = _BRACKET_RE.match(text)
    if match:
        text = match.group(1).strip()
    return text


def event_verb(event_type: str, trigger: str = "") -> str:
    """The canonical Vietnamese verb of an event type.

    The VHE trigger word (``"sinh"``) is only a fallback: the canonical
    mapping keeps the KG node names and the gold paths identical.
    """
    canonical = EVENT_TYPE_VN.get(str(event_type or "").strip())
    if canonical:
        return canonical
    return str(trigger or "").strip() or str(event_type or "").strip()


def event_label(
    agent: str,
    verb: str,
    theme: str,
    trigger: str = "",
    event_type: str = "",
) -> str:
    """Build the canonical Vietnamese label of an event node.

    The label never starts with ``"sự kiện"`` -- the question templates add
    that prefix themselves, so the label can be embedded directly into a
    question or a gold path.

    >>> event_label("Đế Minh", "sinh ra", "Đế Nghi")
    'Đế Minh sinh ra Đế Nghi'
    >>> event_label("", "", "")
    ''
    """
    agent = (agent or "").strip()
    theme = (theme or "").strip()
    if not verb:
        verb = event_verb(event_type, trigger)
    if agent and theme:
        base = f"{agent} {verb} {theme}"
    elif agent:
        base = f"{agent} {verb}"
    elif theme:
        base = f"{verb} {theme}"
    else:
        base = trigger or verb
    return base[:MAX_LABEL_LEN]


def role_vn(role: str) -> str:
    """The Vietnamese label of an argument role (``""`` when unknown)."""
    return ROLE_VN.get(str(role or "").strip().lower(), "")


# --------------------------------------------------------------------------
# mention selection
# --------------------------------------------------------------------------

#: Mentions that are too generic to be used as a gold entity.
GENERIC_MENTIONS: frozenset[str] = frozenset(
    {
        "con trai",
        "con gái",
        "con",
        "quân giặc",
        "quân",
        "người nhà giàu",
        "người",
        "vua",
        "nữ vương",
        "đứa trẻ",
        "trẻ",
        "sứ",
        "nước",
        "đất",
        "giặc",
        "vợ",
        "chồng",
        "anh",
        "chị",
        "em",
        "cháu",
        "ông",
        "bà",
        "cha",
        "mẹ",
        "vương",
        "công chúa",
        "hoàng tử",
        "hoàng tộc",
        "triều thần",
        "thần",
        "công nhân",
        "nông dân",
        "đại đoàn",
        "đội quân",
        "người dân",
        "dân",
        "phường",
        "làng",
        "xã",
        "huyện",
        "tỉnh",
        "thành",
        "đô",
    }
)

#: Normalised keys of :data:`GENERIC_MENTIONS` (computed once).
_GENERIC_KEYS: frozenset[str] = frozenset(
    " ".join(m.lower().split()) for m in GENERIC_MENTIONS
)


def is_proper_mention(mention: str) -> bool:
    """Whether a mention looks like a named entity (not a generic noun).

    >>> is_proper_mention("Đế Minh")
    True
    >>> is_proper_mention("con gái")
    False
    """
    text = clean_mention(mention)
    if not text or len(text) < 2:
        return False
    if not text[0].isupper():
        return False
    if " ".join(text.lower().split()) in _GENERIC_KEYS:
        return False
    tokens = text.split()
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return False
    return True


def is_usable_entity(mention: str) -> bool:
    """Stricter entity test used for gold answers.

    A single generic-looking capitalised word (``"Giáp"``, ``"Các"``) is a
    poor gold entity: it cannot be matched reliably by the entity linker.
    Multi-word names and long single words are accepted.

    >>> is_usable_entity("Đế Minh")
    True
    >>> is_usable_entity("Vua")
    False
    """
    if not is_proper_mention(mention):
        return False
    tokens = clean_mention(mention).split()
    if len(tokens) >= 2:
        return True
    return len(tokens[0]) >= 5


def pick_mention(mentions: Sequence[str], proper_only: bool = True) -> str:
    """Pick the first *usable* mention of a list ("" when there is none).

    The knowledge graph and the evaluation set must pick the **same**
    argument of an event, otherwise the node name of the event and the gold
    path of the question refer to two different things.
    """
    for value in mentions or []:
        text = clean_mention(str(value or ""))
        if not text:
            continue
        if not proper_only:
            return text
        if is_usable_entity(text):
            return text
    return ""
