"""Vietnamese name normalisation helpers (docs/07, tầng T4).

The entity-resolution strategy described in ``docs/07_xac_dinh_thuc_the.md``
relies on a *normalised key* to match names that differ only by diacritics,
punctuation, word order, or Vietnamese honorifics.

Three public helpers are provided:

``normalize_name(s)``
    Light normalisation: Unicode NFC, lower-case, punctuation removed,
    whitespace collapsed.  **Accented characters are preserved** (only
    Unicode-composed), because callers often want a readable, comparable
    surface form of the name.

``strip_honorifics(s)``
    Removes Vietnamese honorifics / titles (``vua``, ``Đại vương``,
    ``Tiên Đế`` ...) while keeping the remaining words (and their diacritics)
    untouched.

``name_key(s)``
    The canonical *matching key*: honorifics stripped, diacritics removed,
    lower-cased, punctuation removed, tokens sorted and ordinal prefixes
    dropped.  Two names that denote the same entity should map to the same
    key (``"Quốc Tuấn Trần"`` and ``"Trần Quốc Tuấn"`` do).
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "HONORIFICS",
    "normalize_name",
    "strip_honorifics",
    "name_key",
    "strip_diacritics",
    "split_name_parts",
    "remove_parenthetical",
]

#: Honorifics / titles that must not take part in the matching key.
#: Written with their normal Vietnamese spelling.
HONORIFICS: tuple[str, ...] = (
    # kings / emperors
    "vua",
    "nữ vương",
    "vương",
    "đại vương",
    "tiên đế",
    "tiên vương",
    "thái tổ",
    "thái tông",
    "thái bình chương đế",
    "thượng đế",
    "đế",
    "hoàng đế",
    "hoàng tử",
    "công chúa",
    "hoàng công chúa",
    "thái tử",
    "quốc công",
    "quốc vương",
    "thái phụ",
    "thượng phụ",
    "trưởng vương",
    # generals / officials
    "tướng",
    "đại tướng",
    "tướng quân",
    "đại tướng quân",
    "tướng đại đại quân",
    "đại quân",
    "nguyên soái",
    "hiển tướng quân",
    "chinh phụ tướng quân",
    "đại công thần",
    "công thần",
    # religious titles
    "chúa",
    "đức chúa",
    "đại đạo",
    "tiên nữ",
    # kinship / courtesy
    "ông",
    "bà",
    "cụ",
    # NB: "vợ"/"chồng"/"anh"/"chị" are deliberately *not* listed -- "Võ"
    # (a very common given name) would otherwise be stripped as "vợ".
    # nobility
    "bá hầu",
    "bá tướng",
    "hầu tướng",
    "khanh tướng",
    "trung sẩn",
    "tả thị lang",
    "hữu thị lang",
    "thượng thư",
    "đại thượng thư",
    "bộ trinh",
    "thủy tướng",
    "lục tướng",
)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_ORDINAL_RE = re.compile(r"^\s*(?:\d+|[ivxlcdm]+)\s*[\.\-)]?\s+", re.IGNORECASE)
_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*")
_DIACRITIC_RE = re.compile(r"[^a-z0-9]+")

#: Legacy Vietnamese letters that Unicode NFC does not fold onto the modern
#: ones (``Ð``/``ð`` are separate code points from ``Đ``/``đ``).
_LEGACY_MAP = {
    "Ð": "Đ",  # LATIN CAPITAL LETTER ETH -> LATIN CAPITAL LETTER D WITH STROKE
    "ð": "đ",  # LATIN SMALL LETTER ETH -> LATIN SMALL LETTER D WITH STROKE
}


def strip_diacritics(s: str) -> str:
    """Remove Vietnamese diacritics, keeping only ``a-z0-9`` and spaces.

    Uses NFD decomposition followed by dropping combining marks, which is
    the reliable way to strip Vietnamese tone marks (unlike
    ``unicodedata.normalize("NFD")`` alone, which also alters ``đ``/``Đ``).
    """
    if not s:
        return ""
    text = unicodedata.normalize("NFD", s)
    stripped = "".join(ch for ch in text if not unicodedata.combining(ch))
    # The letter đ/Đ decomposes into d + combining marks that vanish above.
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return stripped


def _honorific_key(s: str) -> str:
    """Accent-free, lower-case key of an honorific phrase."""
    return strip_diacritics(s.lower())


#: Accent-free honorific keys -- multi-word entries must be matched first.
_HONORIFIC_SET = frozenset(_honorific_key(h) for h in HONORIFICS)
_HONORIFIC_MAX_LEN = max(len(h.split()) for h in HONORIFICS)


def _tokens(s: str) -> list[str]:
    return [t for t in _WS_RE.split(s.strip()) if t]


def remove_parenthetical(s: str) -> str:
    """Drop a trailing disambiguation bracket, e.g. ``"Lê Lợi (kỹ thuật)"``.

    Wikipedia uses brackets to disambiguate homonyms; per docs/07 §4 such
    names must *not* be merged automatically.
    """
    return _PAREN_RE.sub(" ", s or "").strip()


def _is_honorific_gram(gram: str) -> bool:
    return _honorific_key(gram) in _HONORIFIC_SET


def _strip_edge_honorifics(text: str) -> str:
    """Remove honorific phrases sitting at the *edges* of the name.

    Only prefixes and suffixes are considered, and a removal is rejected
    when it would leave fewer than two tokens -- otherwise legitimate names
    such as "Lý Thái Tổ" would be reduced to "Lý".
    """
    toks = _tokens(text)
    changed = True
    while changed and len(toks) >= 3:
        changed = False
        for n in range(min(_HONORIFIC_MAX_LEN, len(toks) - 1), 0, -1):
            # prefix
            if _is_honorific_gram(" ".join(toks[:n])):
                candidate = toks[n:]
                if len(candidate) >= 2:
                    toks = candidate
                    changed = True
                    break
            # suffix
            if _is_honorific_gram(" ".join(toks[-n:])):
                candidate = toks[:-n]
                if len(candidate) >= 2:
                    toks = candidate
                    changed = True
                    break
    return " ".join(toks)


def strip_honorifics(s: str) -> str:
    """Remove Vietnamese honorifics/titles from ``s``.

    The surface form (diacritics, case) is preserved; only the honorific
    tokens are dropped.  Honours embedded in parentheses are removed as well.

    >>> strip_honorifics("Đại vương Trần Hưng Đạo")
    'Trần Hưng Đạo'
    >>> strip_honorifics("vua Lê Lợi")
    'Lê Lợi'
    """
    if not s:
        return ""
    text = remove_parenthetical(s)
    text = _strip_edge_honorifics(text)
    # Pass 2: drop remaining single-token honorifics anywhere in the name.
    out = [tok for tok in _tokens(text) if not _is_honorific_gram(tok)]
    result = _WS_RE.sub(" ", " ".join(out)).strip()
    return result or _WS_RE.sub(" ", text).strip()


def normalize_name(s: str) -> str:
    """Light-weight normalisation that **keeps** Vietnamese diacritics.

    Steps: legacy letter mapping -> Unicode NFC -> lower-case -> punctuation
    -> whitespace collapse.

    >>> normalize_name("Trần  Quốc Tuấn, (1228-1300)")
    'trần quốc tuấn 1228 1300'
    """
    if not s:
        return ""
    text = unicodedata.normalize("NFC", s)
    text = "".join(_LEGACY_MAP.get(ch, ch) for ch in text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def name_key(s: str) -> str:
    """Canonical matching key for an entity name.

    Pipeline (docs/07 §5, T4): remove parenthetical -> strip honorifics ->
    drop diacritics -> lower-case -> remove punctuation -> drop ordinal
    prefixes -> sort tokens (so "Quốc Tuấn Trần" == "Trần Quốc Tuấn").

    >>> name_key("Quốc Tuấn Trần") == name_key("Trần Quốc Tuấn")
    True
    """
    if not s:
        return ""
    text = remove_parenthetical(s)
    text = strip_honorifics(text)
    text = _ORDINAL_RE.sub("", text)
    text = strip_diacritics(text.lower())
    text = _DIACRITIC_RE.sub(" ", text)
    toks = sorted(t for t in _WS_RE.split(text) if t)
    return " ".join(toks)


#: Well-known multi-token Vietnamese surnames.
_MULTI_TOKEN_SURNAMES = {
    "đinh bộ",
    "lê thánh",
    "ngô quyền",
    "trần công",
}


def split_name_parts(s: str) -> tuple[str, str]:
    """Split a Vietnamese personal name into ``(family_name, given_name)``.

    Vietnamese names are written family-name-first; the family name is the
    first token (``"Trần Hưng Đạo"`` -> ``("Trần", "Hưng Đạo")``).

    >>> split_name_parts("Trần Hưng Đạo")
    ('Trần', 'Hưng Đạo')
    """
    text = strip_honorifics(s)
    # Keep the original case (normalize_name lower-cases).
    text = unicodedata.normalize("NFC", text)
    text = "".join(_LEGACY_MAP.get(ch, ch) for ch in text)
    toks = _tokens(text)
    if not toks:
        return "", ""
    two = _honorific_key(" ".join(toks[:2]))
    if len(toks) > 2 and two in _MULTI_TOKEN_SURNAMES:
        return " ".join(toks[:2]), " ".join(toks[2:])
    return toks[0], " ".join(toks[1:])
