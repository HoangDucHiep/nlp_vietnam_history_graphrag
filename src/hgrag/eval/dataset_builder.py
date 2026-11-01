"""Build the evaluation question set (docs/04 §2).

Four generators feed one dataset:

``vhe_template``
    Template questions built from the *gold* VHE annotations (gold event +
    argument roles).  The answer is always derivable from the annotation, so
    the gold evidence is guaranteed correct.  The templates never mention the
    answer in the question itself (no leakage).

``llm_from_wiki``
    An LLM reads one Wikipedia chunk and writes a question plus a short
    reference answer.  Only information present in the chunk is allowed; the
    chunk id is recorded as gold evidence.

``history_chat``
    Question/answer pairs taken from ``minhxthanh/Vietnam-History-15k``,
    restricted to the 905-1975 scope (the answers are LLM-generated, so they
    only serve as *candidate* references -- docs/01 §1.3).

``out_of_kb``
    Hand-written questions outside the knowledge base; the expected behaviour
    is a refusal (gold answer empty).

The builder is *quota driven*: it computes the target distribution
(40 % single-hop / 30 % multi-hop / 15 % temporal / 15 % out-of-KB) and fills
each bucket from the sources that can supply it, then falls back to the
richest remaining source if a bucket cannot be filled.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from ..data.normalize import name_key
from ..data.records import Chunk, Entity
from ..llm.schemas import LLMMessage, LLMRequest
from .dataset import EvalDataset, EvalQuestion

logger = logging.getLogger(__name__)

__all__ = [
    "EvalDatasetBuilder",
    "TARGET_TYPE_RATIOS",
    "OUT_OF_KB_QUESTIONS",
    "EVENT_TYPE_VN",
    "ROLE_VN",
]

#: Target distribution of the question groups (docs/04 §2.1).
TARGET_TYPE_RATIOS: dict[str, float] = {
    "single_hop": 0.40,
    "multi_hop": 0.30,
    "temporal": 0.15,
    "out_of_kb": 0.15,
}

#: Hand-written questions that lie *outside* the knowledge base (docs/04 §2.1 D).
OUT_OF_KB_QUESTIONS: tuple[str, ...] = (
    "Thủ tướng Việt Nam hiện nay là ai?",
    "Ai đoạt giải Nobel Văn học năm 2024?",
    "Cầu Sài Gòn dài bao nhiêu km?",
    "Tỷ giá USD/VND hiện tại là bao nhiêu?",
    "Danh sách những người đã trúng giải Oscar năm 2023 gồm những ai?",
    "Mùa hè ở Đà Lạt năm nay nhiệt độ trung bình là bao nhiêu?",
    "Đội tuyển bóng đá Việt Nam đã thắng trận gần đây nhất với tỷ số bao nhiêu?",
    "Google ra mắt mô hình AI mới nhất năm 2025 tên gì?",
    "Ai là người phát minh ra bóng đá ở Việt Nam?",
    "Giá vàng thế giới hôm nay bao nhiêu một lượng?",
    "Cầu Võng Đài nối giữa hai quận nào của Hà Nội?",
    "Tổng thống Mỹ hiện tại sinh năm nào?",
    "Nguyễn Huy Tăng sáng tác tác phẩm nào nổi tiếng nhất?",
    "Học kỳ học chính quy của năm học 2025-2026 kéo dài từ tháng nào đến tháng nào?",
    "Thành phố nào lớn thứ hai của Việt Nam sau Hà Nội?",
    "Bộ phim nào của Việt Nam đoạt giải tại Liên hoan phim quốc tế Cannes 2023?",
    "Dân số Việt Nam tính đến năm 2025 là bao nhiêu người?",
    "Ai là người sáng lập tập đoàn VinFast?",
    "Đội bóng đá nào vô địch UEFA Champions League mùa giải 2024-2025?",
    "Tốc độ giới hạn tối đa trên đường cao tốc tại Việt Nam là bao nhiêu km/h?",
    "Trận chung kết World Cup 2022 diễn ra ở sân vận động nào?",
    "Ai là người được bầu làm Chủ tịch nước Việt Nam nhiệm kỳ 2026-2031?",
    "Quốc gia nào vừa giành vô địch bóng bàn World Cup 2024?",
    "Giá một chiếc iPhone mới nhất tại Việt Nam là bao nhiêu?",
    "Nhà hát lớn nhất Việt Nam hiện nay nằm ở thành phố nào?",
    "Loài hoa nào được chọn là hoa quốc gia của Việt Nam?",
    "Bãi biển nào được xếp hạng đẹp nhất thế giới năm 2024?",
)

#: Vietnamese label of every VHE event type (used to build natural questions).
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

_YEAR_RE = re.compile(r"(?<!\d)(9[0-9]{2}|1[0-9]{3})(?!\d)")
_CAP_RE = re.compile(
    r"[ĐA-ZÀ-ÖØ-Þ][\wÀ-ỹĐđ]*(?:[ \t]+[ĐA-ZÀ-ÖØ-Þ][\wÀ-ỹĐđ]*){0,3}"
)

#: Explicit set of *upper-case* code points.  The range ``À-ỹ`` also contains
#: lower-case accented letters (``á``, ``è`` ...), so it cannot be used to
#: detect the start of a new name.
_UPPER_CHARS: frozenset[str] = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "AÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ"
)

#: Capitalised words that are *not* entities (Vietnamese sentence starters).
_NON_ENTITY_WORDS: frozenset[str] = frozenset(
    {
        "sau",
        "trước",
        "năm",
        "các",
        "những",
        "một",
        "hai",
        "ba",
        "vào",
        "khi",
        "từ",
        "đến",
        "với",
        "theo",
        "là",
        "có",
        "đã",
        "đang",
        "vì",
        "nếu",
        "nhưng",
        "và",
        "của",
        "cho",
        "do",
        "tại",
        "trên",
        "dưới",
        "về",
        "hán",
        "chữ",
        "tên",
        "làm",
        "được",
        "ông",
        "bà",
        "sự",
        "việc",
        "thời",
        "nơi",
        "đất",
        "nước",
        "nhà",
        "triều",
        "quân",
        "giặc",
        "thành",
        "đô",
        "hoàng",
        "đế",
        "vua",
        "công",
        "chúa",
        "lão",
        "người",
        "con",
        "vợ",
        "chồng",
        "anh",
        "chị",
        "em",
        "đi",
        "ra",
        "lên",
        "xuống",
        "sang",
        "gửi",
        "mở",
        "lập",
        "đặt",
        "ban",
        "cử",
        "sai",
        "giữ",
        "nôm",
        "việt",
        "nam",
    }
)

#: Mentions that are too generic to be used as a gold entity.
_GENERIC_MENTIONS: frozenset[str] = frozenset(
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
        "người hầu",
        "triều thần",
        "thái tử",
        "hoàng tử",
        "vương",
        "công chúa",
        "phu nhân",
        "giáo sư",
        "thầy giáo",
        "người dân",
        "dân",
    }
)

#: Temporal mentions that are *too vague* to be a good gold answer.
_VAGUE_TEMPORALS: frozenset[str] = frozenset(
    {
        "mùa xuân",
        "mùa hạ",
        "mùa thu",
        "mùa đông",
        "lúc đó",
        "khi đó",
        "sau đó",
        "trước đó",
        "hiện nay",
        "ngày",
        "hôm",
        "ban đầu",
        "về sau",
        "lúc đầu",
        "thời gian sau",
    }
)

#: A temporal mention is *informative* when it names an era / dynasty / year.
_ERA_TEMPORAL_RE = re.compile(
    r"(đời|thời|hậu|tiền|niên hiệu|năm thứ|\b\d{3}\b)", re.IGNORECASE
)

#: Quota weight of each question type per source.
_SOURCE_WEIGHTS: dict[str, dict[str, float]] = {
    "vhe": {"single_hop": 0.45, "multi_hop": 0.40, "temporal": 0.60},
    "llm": {"single_hop": 0.30, "multi_hop": 0.35, "temporal": 0.25},
    "chat": {"single_hop": 0.25, "multi_hop": 0.25, "temporal": 0.15},
}

#: Aliases accepted on the CLI / in :meth:`EvalDatasetBuilder.build`.
SOURCE_ALIASES: dict[str, str] = {
    "vhe": "vhe",
    "vhe_template": "vhe",
    "llm": "llm",
    "llm_from_wiki": "llm",
    "wiki": "llm",
    "chat": "chat",
    "history_chat": "chat",
    "outkb": "outkb",
    "out_of_kb": "outkb",
    "out-of-kb": "outkb",
}

_MIN_ANSWER_CHARS = 12
_MAX_ANSWER_CHARS = 1200
_MIN_CHAT_ANSWER_CHARS = 40
_MIN_CHAT_QUESTION_CHARS = 15


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clean_mention(mention: str) -> str:
    """Strip the ``vua [Kinh Dương Vương]`` bracket notation of VHE."""
    text = (mention or "").strip()
    match = re.match(r"^\w+\s*\[([^\]]+)\]$", text)
    if match:
        text = match.group(1).strip()
    return text


def _is_proper(mention: str) -> bool:
    """Whether a mention looks like a named entity (not a generic noun)."""
    text = _clean_mention(mention)
    if not text or len(text) < 2:
        return False
    if not text[0].isupper():
        return False
    if name_key(text) in {name_key(g) for g in _GENERIC_MENTIONS}:
        return False
    tokens = text.split()
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return False
    return True


def _is_usable_entity(mention: str) -> bool:
    """Stricter entity test used for gold answers.

    A single generic-looking capitalised word (``"Giáp"``, ``"Các"``) is a
    poor gold entity: it cannot be matched reliably by the entity linker.
    Multi-word names and long single words are accepted.
    """
    if not _is_proper(mention):
        return False
    tokens = _clean_mention(mention).split()
    if len(tokens) >= 2:
        return True
    return len(tokens[0]) >= 5


def _extract_years(text: str, lo: int = 905, hi: int = 1975) -> list[int]:
    """Years mentioned in ``text`` inside the KB scope."""
    out: list[int] = []
    for raw in _YEAR_RE.findall(text or ""):
        year = int(raw)
        if lo <= year <= hi and year not in out:
            out.append(year)
    return out


def _capitals(text: str, limit: int = 8) -> list[str]:
    """Capitalised n-grams of a Vietnamese text (candidate entities).

    Sentence starters and generic nouns are filtered out at both ends, and
    multi-word names are preferred over their first word (``"Lê Đại Hành"``
    instead of ``"Lê"``).
    """
    seen: list[str] = []
    blocked = {name_key(w) for w in _NON_ENTITY_WORDS}
    for match in _CAP_RE.finditer(text or ""):
        phrase = match.group(0).strip()
        if len(phrase) < 3 or phrase[0] not in _UPPER_CHARS:
            continue
        tokens = phrase.split()
        # drop leading / trailing non-entity words ("Năm 1005, Lê Đại Hành",
        # "Nguyễn Ái Quốc đến" -> "Nguyễn Ái Quốc")
        while tokens and name_key(tokens[0]) in blocked:
            tokens = tokens[1:]
        while len(tokens) > 1 and name_key(tokens[-1]) in blocked:
            tokens = tokens[:-1]
        if not tokens:
            continue
        phrase = " ".join(tokens)
        if len(tokens) == 1 and name_key(phrase) in blocked:
            continue
        if phrase in seen:
            continue
        seen.append(phrase)
        if len(seen) >= limit:
            break
    return seen


def _split_sentences(text: str) -> list[str]:
    """Split a text into sentences (Vietnamese-friendly)."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def _token_set(text: str) -> set[str]:
    return {t for t in name_key(text).split() if len(t) > 1}


def _answer_supported(answer: str, chunk_text: str, threshold: float = 0.5) -> bool:
    """Whether the answer only uses information available in the chunk."""
    tokens = _token_set(answer)
    if not tokens:
        return False
    haystack = _token_set(chunk_text)
    overlap = len(tokens & haystack) / len(tokens)
    return overlap >= threshold


def _leaks(answer: str, question: str) -> bool:
    """Whether the answer is (almost) spelled out inside the question."""
    answer_tokens = _token_set(answer)
    if not answer_tokens:
        return False
    question_tokens = _token_set(question)
    return len(answer_tokens & question_tokens) / len(answer_tokens) >= 0.8


def _agent_of(event: dict[str, Any]) -> str:
    """Agent of a prepared VHE event."""
    return str(event.get("agent") or "")


def _theme_of(event: dict[str, Any]) -> str:
    """Theme of a prepared VHE event."""
    return str(event.get("theme") or "")


def _label_has_entity(label: str) -> bool:
    """Whether an event label names a person / place / organisation.

    A label made only of a trigger word (``"đánh"``, ``"phóng lửa"``) does not
    identify an event, so it cannot be used as the subject of a question.
    """
    tokens = label.split()
    if len(tokens) < 2:
        return False
    # A multi-word label is fine as soon as one of its parts is a real name.
    return any(_is_usable_entity(token) for token in tokens)


def _label_has_entity_parts(label: str, agent: str, theme: str) -> bool:
    """Whether the label is built on top of a named entity.

    The agent / theme are checked as *whole strings* -- Vietnamese names are
    multi-word (``"Đế Nghi"``), so a token-wise check would fail.
    """
    for part in (agent, theme):
        if part and _is_usable_entity(part):
            return True
    return False


def _quota(target: int) -> dict[str, int]:
    """Split ``target`` questions across the four types (largest remainder)."""
    raw = {k: target * v for k, v in TARGET_TYPE_RATIOS.items()}
    base = {k: int(v) for k, v in raw.items()}
    remainder = target - sum(base.values())
    order = sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)
    for i in range(remainder):
        base[order[i % len(order)]] += 1
    return base


@dataclass
class _Candidate:
    """A question candidate produced by one of the generators."""

    question: EvalQuestion
    priority: int = 0


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


class EvalDatasetBuilder:
    """Build an :class:`~hgrag.eval.dataset.EvalDataset` from four sources."""

    def __init__(
        self,
        llm_client: Any,
        settings: Settings | None = None,
        corpus_chunks: list[Chunk] | None = None,
        kg_entities: list[Entity] | None = None,
        max_llm_chunks: int = 60,
    ) -> None:
        self.llm = llm_client
        self.settings = settings
        self.corpus_chunks = list(corpus_chunks or [])
        self.kg_entities = list(kg_entities or [])
        self._max_chunks = max(0, int(max_llm_chunks))
        self._llm_broken = False
        self._entity_index: dict[str, Entity] = {}
        for entity in self.kg_entities:
            for name in [entity.name, *entity.aliases]:
                key = name_key(name)
                if key and key not in self._entity_index:
                    self._entity_index[key] = entity
        self._seen: set[str] = set()
        self._corpus_loaded = bool(self.corpus_chunks)
        self._chunk_keys: dict[str, str] = {}
        self._chunk_key_cache: dict[str, str] = {}

    # -- public API -----------------------------------------------------
    def build(
        self,
        target_size: int = 150,
        sources: Sequence[str] | None = None,
        offline: bool = False,
    ) -> EvalDataset:
        """Build the dataset, honouring the type quotas of ``target_size``."""
        wanted = self._normalize_sources(sources)
        quotas = _quota(max(0, int(target_size)))

        pools: dict[str, dict[str, list[_Candidate]]] = {
            src: {"single_hop": [], "multi_hop": [], "temporal": []} for src in wanted
        }

        if "vhe" in wanted:
            pools["vhe"] = self._collect(self._gen_vhe_candidates, offline)
        if "llm" in wanted:
            pools["llm"] = self._collect(self._gen_llm_candidates, offline)
        if "chat" in wanted:
            pools["chat"] = self._collect(self._gen_chat_candidates, offline)

        selected: list[EvalQuestion] = []
        self._seen = set()

        # 1. out-of-KB questions are fixed -> they fill their quota first.
        outkb_quota = quotas.get("out_of_kb", 0)
        if "outkb" in wanted and outkb_quota > 0:
            for question in self._gen_out_of_kb(outkb_quota):
                self._seen.add(name_key(question.question))
                selected.append(question)

        # 2. fill the remaining buckets from the sources that can supply them.
        for qtype in ("single_hop", "multi_hop", "temporal"):
            need = quotas.get(qtype, 0)
            selected.extend(self._select_for_type(pools, qtype, need))

        # 3. top-up if a bucket was under-filled (rare).
        if len(selected) < target_size:
            selected.extend(self._topup(pools, target_size - len(selected)))

        selected = selected[:target_size]
        self._renumber(selected)
        self._attach_excluded_chunks(selected)
        dataset = EvalDataset(questions=selected)
        dataset.metadata = {
            "target_size": target_size,
            "sources": sorted(wanted),
            "quota": quotas,
            "offline": offline,
            "llm_client": getattr(self.llm, "name", "unknown"),
            "n_corpus_chunks": len(self.corpus_chunks),
            "n_kg_entities": len(self.kg_entities),
        }
        return dataset

    # -- source normalisation -------------------------------------------
    @staticmethod
    def _normalize_sources(sources: Sequence[str] | None) -> list[str]:
        if not sources:
            return ["vhe", "llm", "chat", "outkb"]
        out: list[str] = []
        for raw in sources:
            key = str(raw).strip().lower()
            if not key:
                continue
            canonical = SOURCE_ALIASES.get(key)
            if canonical and canonical not in out:
                out.append(canonical)
        return out or ["vhe", "llm", "chat", "outkb"]

    # -- candidate collection -------------------------------------------
    def _collect(self, generator, offline: bool) -> dict[str, list[_Candidate]]:
        pools: dict[str, list[_Candidate]] = {
            "single_hop": [],
            "multi_hop": [],
            "temporal": [],
        }
        try:
            produced = generator(offline=offline)
        except Exception as exc:  # noqa: BLE001 - one source must never break the build
            logger.warning("Generator %s failed: %s", getattr(generator, "__name__", "?"), exc)
            return pools
        for candidate in produced:
            pools.setdefault(candidate.question.question_type, []).append(candidate)
        return pools

    def _select_for_type(
        self,
        pools: dict[str, dict[str, list[_Candidate]]],
        qtype: str,
        need: int,
    ) -> list[EvalQuestion]:
        """Take ``need`` questions of type ``qtype`` respecting source weights."""
        if need <= 0:
            return []
        weights = {src: _SOURCE_WEIGHTS.get(src, {}).get(qtype, 0.2) for src in pools}
        total_weight = sum(weights.values()) or 1.0
        shares = {src: int(need * w / total_weight) for src, w in weights.items()}

        chosen: list[EvalQuestion] = []
        for src, share in shares.items():
            if share <= 0:
                continue
            taken = 0
            for candidate in pools.get(src, {}).get(qtype, []):
                if len(chosen) >= need or taken >= share:
                    break
                key = name_key(candidate.question.question)
                if not key or key in self._seen:
                    continue
                self._seen.add(key)
                chosen.append(candidate.question)
                taken += 1
            if len(chosen) >= need:
                break

        # Second pass: fill the rest from any source (ignoring the weights).
        if len(chosen) < need:
            for _src, by_type in pools.items():
                for candidate in by_type.get(qtype, []):
                    if len(chosen) >= need:
                        break
                    key = name_key(candidate.question.question)
                    if not key or key in self._seen:
                        continue
                    self._seen.add(key)
                    chosen.append(candidate.question)
                if len(chosen) >= need:
                    break
        return chosen

    def _topup(
        self,
        pools: dict[str, dict[str, list[_Candidate]]],
        need: int,
    ) -> list[EvalQuestion]:
        """Fill a residual gap with any still-unused candidate."""
        chosen: list[EvalQuestion] = []
        for qtype in ("single_hop", "multi_hop", "temporal"):
            for _src, by_type in pools.items():
                for candidate in by_type.get(qtype, []):
                    if len(chosen) >= need:
                        return chosen
                    key = name_key(candidate.question.question)
                    if not key or key in self._seen:
                        continue
                    self._seen.add(key)
                    chosen.append(candidate.question)
        return chosen

    def _attach_excluded_chunks(self, questions: list[EvalQuestion]) -> None:
        """Fill ``metadata['excluded_chunks']`` for the selected questions only.

        Done *after* the selection so the (expensive) lookup runs on the ~150
        kept questions instead of on every candidate.
        """
        for question in questions:
            if question.question_type == "out_of_kb":
                continue
            keep = question.gold_chunk_ids[0] if question.gold_chunk_ids else ""
            question.metadata["excluded_chunks"] = self._excluded_chunks(
                question.gold_entities, keep=keep
            )

    @staticmethod
    def _renumber(questions: list[EvalQuestion]) -> None:
        """Give every question a unique, stable qid."""
        counters: dict[str, int] = {}
        prefixes = {
            "vhe_template": "vhe",
            "llm_from_wiki": "llm",
            "history_chat": "chat",
            "out_of_kb": "oob",
        }
        for question in questions:
            prefix = prefixes.get(question.source, "q")
            counters[prefix] = counters.get(prefix, 0) + 1
            question.qid = f"{prefix}_{counters[prefix]:05d}"

    # -- source 4: out-of-KB --------------------------------------------
    def _gen_out_of_kb(self, count: int) -> list[EvalQuestion]:
        questions: list[EvalQuestion] = []
        for text in OUT_OF_KB_QUESTIONS:
            if len(questions) >= count:
                break
            questions.append(
                EvalQuestion(
                    qid="",
                    question=text,
                    answer="",
                    question_type="out_of_kb",
                    gold_entities=[],
                    gold_qids=[],
                    gold_paths=[],
                    gold_chunk_ids=[],
                    gold_relations=[],
                    year_refs=[],
                    difficulty="easy",
                    source="out_of_kb",
                    metadata={
                        "expected_behaviour": "refuse",
                        "excluded_chunks": [],
                        "reason": "ngoài phạm vi KB lịch sử Việt Nam 905-1975",
                    },
                )
            )
        return questions

    # -- source 1: VHE templates ---------------------------------------
    def _gen_vhe_candidates(self, offline: bool = False) -> list[_Candidate]:
        from ..data import vhe

        cache_dir = self._cache_dir("vhe")
        instances = vhe.load_instances(cache_dir=cache_dir, offline=offline)
        if not instances:
            logger.warning("VHE instances unavailable -> no vhe_template question")
            return []

        candidates: list[_Candidate] = []
        counter = 0
        for instance in instances:
            text = instance.get("text", "")
            events = instance.get("events") or []
            if not events:
                continue
            prepared = [self._prepare_event(ev) for ev in events]
            prepared = [p for p in prepared if p is not None]
            if not prepared:
                continue

            for event in prepared:
                agent = event["agent"]
                theme = event["theme"]
                location = event["location"]
                temporal = event["temporal"]
                goal = event["goal"]
                verb = event["verb"]
                # An event without a named entity has no usable question
                # subject ("Sự kiện nhật thực diễn ra ở đâu?").
                if not (_is_usable_entity(agent) or _is_usable_entity(theme)):
                    continue

                # -- single hop: who performed the action? -----------------
                if agent and theme:
                    counter += 1
                    candidates.append(
                        self._vhe_question(
                            counter=counter,
                            qtype="single_hop",
                            question=f"Ai là người {verb} {theme}?",
                            answer=agent,
                            entities=[agent],
                            path=[agent, event["label"]],
                            relations=[f"{agent} -PARTICIPATED_IN-> {event['label']}"],
                            instance=text,
                            role="agent",
                        )
                    )
                # -- single hop: which entity is the action about? ---------
                if agent and theme:
                    counter += 1
                    candidates.append(
                        self._vhe_question(
                            counter=counter,
                            qtype="single_hop",
                            question=(
                                f"Trong sự kiện do {agent} thực hiện, "
                                f"đối tượng được nhắc đến là gì?"
                            ),
                            answer=theme,
                            entities=[theme],
                            path=[agent, event["label"], theme],
                            relations=[f"{agent} -ABOUT-> {theme}"],
                            instance=text,
                            role="theme",
                        )
                    )
                # -- single hop: where did it happen? -----------------------
                if location and (agent or theme):
                    counter += 1
                    subject = agent or theme
                    label = self._label(event, exclude=location)
                    if label and _is_usable_entity(location):
                        candidates.append(
                            self._vhe_question(
                                counter=counter,
                                qtype="single_hop",
                                question=f"Sự kiện {label} diễn ra ở đâu?",
                                answer=location,
                                entities=[location],
                                path=[subject, label, location],
                                relations=[f"{label} -LOCATED_IN-> {location}"],
                                instance=text,
                                role="location",
                            )
                        )
                # -- temporal: when did it happen? -------------------------
                if temporal:
                    informative = self._temporal_is_informative(temporal)
                    label = self._label(event, exclude=temporal)
                    if label:
                        counter += 1
                        year_refs = _extract_years(temporal)
                        if informative:
                            question = f"Sự kiện {label} diễn ra vào thời điểm nào?"
                        else:
                            question = f"Sự kiện {label} diễn ra vào mùa hay tháng nào?"
                        candidates.append(
                            self._vhe_question(
                                counter=counter,
                                qtype="temporal",
                                question=question,
                                answer=temporal,
                                entities=[temporal],
                                path=[label, temporal],
                                relations=[f"{label} -OCCURRED_IN-> {temporal}"],
                                instance=text,
                                role="temporal",
                                year_refs=year_refs,
                            )
                        )
                # -- temporal: the goal / destination of an event -----------
                if (
                    goal
                    and agent
                    and _is_usable_entity(goal)
                    and event["type"] in ("transport", "start-position")
                ):
                    label = self._label(event, exclude=goal)
                    if label:
                        counter += 1
                        year_refs = _extract_years(text)
                        candidates.append(
                            self._vhe_question(
                                counter=counter,
                                qtype="temporal",
                                question=(
                                    f"Sự kiện {label} nhằm mục đích gì, "
                                    f"và diễn ra vào khoảng thời gian nào?"
                                ),
                                answer=(
                                    f"Mục đích: {goal}."
                                    + (
                                        f" Thời điểm: {', '.join(str(y) for y in year_refs)}."
                                        if year_refs
                                        else ""
                                    )
                                ),
                                entities=[goal],
                                path=[label, goal],
                                relations=[f"{label} -ABOUT-> {goal}"],
                                instance=text,
                                role="goal",
                                year_refs=year_refs,
                            )
                        )

            # -- multi hop: two events sharing the same entity --------------
            for candidate in self._multi_hop_pairs(prepared):
                counter += 1
                candidate.question.qid = f"vhe_{counter:05d}"
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _temporal_is_informative(temporal: str) -> bool:
        """Whether a temporal mention names an era / dynasty / year.

        ``"Mùa xuân"`` alone is too vague to be a good gold answer, while
        ``"Đời Hùng Vương thứ 6"`` or ``"năm 1285"`` are precise.
        """
        if not temporal:
            return False
        if name_key(temporal) in {name_key(v) for v in _VAGUE_TEMPORALS}:
            return False
        return bool(_ERA_TEMPORAL_RE.search(temporal))

    def _prepare_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Normalise one VHE event into a dict the templates can use."""
        etype = str(event.get("type") or "")
        if not etype:
            return None
        by_role: dict[str, list[str]] = {}
        for arg in event.get("arguments") or []:
            role = str(arg.get("role") or "").strip().lower()
            mention = _clean_mention(arg.get("mention") or "")
            if not role or not mention:
                continue
            by_role.setdefault(role, []).append(mention)

        def pick(role: str, proper_only: bool = True) -> str:
            for value in by_role.get(role, []):
                if not proper_only:
                    return value
                if _is_usable_entity(value):
                    return value
            return ""

        agent = pick("agent")
        theme = pick("theme")
        location = pick("location", proper_only=False)
        temporal = pick("temporal", proper_only=False)
        goal = pick("goal", proper_only=False)
        trigger = str(event.get("trigger_word") or "").strip()
        verb = EVENT_TYPE_VN.get(etype, trigger or etype)
        return {
            "type": etype,
            "agent": agent,
            "theme": theme,
            "location": location,
            "temporal": temporal,
            "goal": goal,
            "trigger": trigger,
            "verb": verb,
            "label": self._label_from(agent, verb, theme, trigger),
        }

    @staticmethod
    def _label_from(agent: str, verb: str, theme: str, trigger: str) -> str:
        """Build a natural Vietnamese noun phrase for an event node.

        The result never starts with ``"sự kiện"`` -- the question templates
        add that prefix themselves, so the label can be embedded directly.
        A label without a named entity (``"nhật thực"``) is unusable as a
        question subject, so it falls back to the trigger word.
        """
        if agent and theme:
            return f"{agent} {verb} {theme}"
        if agent:
            return f"{agent} {verb}"
        if theme:
            return f"{verb} {theme}"
        return trigger or verb

    def _label(self, event: dict[str, Any], exclude: str = "") -> str:
        """Event label that never mentions ``exclude`` (no answer leakage)."""
        agent = event["agent"]
        theme = event["theme"]
        if exclude:
            if agent == exclude:
                agent = ""
            if theme == exclude:
                theme = ""
        label = self._label_from(agent, event["verb"], theme, event["trigger"])
        # A label must name an entity, otherwise the question has no
        # identifiable subject ("Sự kiện nhật thực diễn ra ở đâu?").
        if not _label_has_entity_parts(label, agent, theme):
            label = ""
        return label

    def _vhe_question(
        self,
        counter: int,
        qtype: str,
        question: str,
        answer: str,
        entities: list[str],
        path: list[str],
        relations: list[str],
        instance: str,
        role: str,
        year_refs: list[int] | None = None,
    ) -> _Candidate:
        """Assemble a VHE template candidate."""
        qids: list[str] = []
        for entity in entities:
            matched = self._entity_index.get(name_key(entity))
            if matched is not None and matched.wikidata_id:
                qids.append(matched.wikidata_id)
        years = list(year_refs or []) or _extract_years(question)
        difficulty = "easy" if qtype == "single_hop" else "medium"
        q = EvalQuestion(
            qid=f"vhe_{counter:05d}",
            question=question,
            answer=answer,
            question_type=qtype,
            gold_entities=entities,
            gold_qids=qids,
            gold_paths=[path] if path else [],
            gold_chunk_ids=[],
            gold_relations=relations,
            year_refs=years,
            difficulty=difficulty,
            source="vhe_template",
            metadata={
                "role": role,
                "vhe_sentence": instance,
                "excluded_chunks": [],
            },
        )
        return _Candidate(question=q, priority=0)

    def _multi_hop_pairs(self, prepared: list[dict[str, Any]]) -> list[_Candidate]:
        """Multi-hop questions from two events of the same sentence.

        Two shapes are used:

        * *chain* -- the agent of one event becomes the theme of another
          (``X sinh ra Y`` -> ``Y làm chủ``): the answer is the shared entity
          and the gold path visits three nodes.
        * *fork* -- two events share the same agent: the answer is that agent
          and the gold path connects both events through it.
        """
        candidates: list[_Candidate] = []
        for i, first in enumerate(prepared):
            for j, second in enumerate(prepared):
                if i == j:
                    continue
                shared = ""
                shape = ""
                # chain: first.theme == second.agent
                if first["theme"] and first["theme"] == second["agent"]:
                    shared = first["theme"]
                    shape = "chain"
                # fork: same agent on both events
                elif first["agent"] and first["agent"] == second["agent"]:
                    shared = first["agent"]
                    shape = "fork"
                if not shared:
                    continue
                label1 = self._label(first, exclude=shared)
                label2 = self._label(second, exclude=shared)
                if not label1 or not label2 or label1 == label2:
                    continue
                # Both labels must identify an event unambiguously, otherwise
                # the question ("sự kiện đánh và sự kiện làm") is meaningless.
                if not _label_has_entity_parts(label1, _agent_of(first), _theme_of(first)):
                    continue
                if not _label_has_entity_parts(label2, _agent_of(second), _theme_of(second)):
                    continue
                if shape == "chain":
                    question = (
                        f"Nhân vật trung gian giữa sự kiện {label1} và sự kiện {label2} "
                        f"là ai?"
                    )
                    path = [label1, shared, label2]
                else:
                    question = (
                        f"Sự kiện {label1} và sự kiện {label2} "
                        f"cùng do nhân vật nào thực hiện?"
                    )
                    path = [label1, shared, label2]
                candidates.append(
                    self._vhe_question(
                        counter=len(candidates) + 1,
                        qtype="multi_hop",
                        question=question,
                        answer=shared,
                        entities=[shared],
                        path=path,
                        relations=[
                            f"{label1} -INVOLVES-> {shared}",
                            f"{label2} -INVOLVES-> {shared}",
                        ],
                        instance="",
                        role="agent",
                    )
                )
        return candidates

    # -- source 2: LLM from Wikipedia ----------------------------------
    def _gen_llm_candidates(self, offline: bool = False) -> list[_Candidate]:
        chunks = self._select_chunks()
        if not chunks:
            logger.warning("No corpus chunk available -> no llm_from_wiki question")
            return []
        # Every chunk costs one LLM call -> cap the number of chunks so the
        # build stays affordable (the LLM cache makes re-runs cheap).
        limit = self._max_chunks
        if limit and len(chunks) > limit:
            step = max(1, len(chunks) // limit)
            chunks = chunks[::step][:limit]
        candidates: list[_Candidate] = []
        use_llm = self._llm_available(offline)
        for chunk in chunks:
            generated = self._llm_questions(chunk) if use_llm else []
            if not generated:
                generated = self._heuristic_questions(chunk)
            for qtype, payload in generated:
                candidates.append(self._llm_candidate(chunk, qtype, payload))
        return candidates

    def _llm_available(self, offline: bool) -> bool:
        if offline or self._llm_broken:
            return False
        if getattr(self.llm, "name", "") == "mock":
            return False
        return hasattr(self.llm, "generate_json")

    def _select_chunks(self) -> list[Chunk]:
        """Chunks usable as a source: prefer Wikipedia chunks with enough text."""
        if not self.corpus_chunks:
            self.corpus_chunks = self._load_corpus()
        usable = [
            c
            for c in self.corpus_chunks
            if len((c.text or "").strip()) >= 200
            and c.source != "history_chat"
            # Wikipedia image placeholders ("260x260px|thumb|...") carry no
            # prose -> they would produce garbage questions.
            and "px|thumb|" not in (c.text or "")
            # Nam-phí rosters ("* Thần phi ...") are lists, not prose.
            and "* Thần phi" not in (c.text or "")
        ]
        if not usable:
            usable = [
                c
                for c in self.corpus_chunks
                if len((c.text or "").strip()) >= 200
                and "px|thumb|" not in (c.text or "")
                and "* Thần phi" not in (c.text or "")
            ]
        return usable

    def _load_corpus(self) -> list[Chunk]:
        if self._corpus_loaded:
            return self.corpus_chunks
        self._corpus_loaded = True
        if self.settings is None:
            return []
        path = Path(self.settings.data_processed) / "corpus.jsonl"
        if not path.exists():
            return []
        chunks: list[Chunk] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(Chunk.from_dict(json.loads(line)))
                except ValueError:
                    continue
        return chunks

    def _llm_questions(self, chunk: Chunk) -> list[tuple[str, dict[str, Any]]]:
        """Ask the LLM for grounded questions about one chunk."""
        prompt = (
            "Bạn đọc một đoạn tài liệu lịch sử Việt Nam và viết câu hỏi đánh giá.\n"
            "QUY TẮC:\n"
            "- Chỉ được dùng thông tin có trong đoạn tài liệu, KHÔNG bịa thêm.\n"
            "- Câu hỏi bằng tiếng Việt, tự nhiên, trả lời được từ đoạn tài liệu.\n"
            "- KHÔNG đưa đáp án vào trong câu hỏi.\n"
            "- Đáp án ngắn gọn (1-2 câu), chứa đáp án cụ thể.\n"
            "- Độ khó: single_hop (một bước), multi_hop (cần ghép 2 thông tin), "
            "temporal (câu hỏi về mốc thời gian).\n"
            "Trả về JSON đúng dạng:\n"
            '{"questions": [{"question": "...", "answer": "...", "type": "single_hop", '
            '"entities": ["..."], "years": [1234]}]}\n\n'
            f"--- ĐOẠN TÀI LIỆU (doc_id={chunk.doc_id}) ---\n{chunk.text}\n"
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content="Bạn là chuyên gia tạo câu hỏi đánh giá."),
                LLMMessage(role="user", content=prompt),
            ],
            temperature=0.3,
            max_output_tokens=1500,
            task="eval_generate_questions",
        )
        try:
            payload = self.llm.generate_json(request)
        except Exception as exc:  # noqa: BLE001 - LLM failure must not break the build
            logger.warning("LLM question generation failed for %s: %s", chunk.chunk_id, exc)
            self._llm_broken = True
            return []
        if not isinstance(payload, dict):
            return []
        raw_questions = payload.get("questions")
        if not isinstance(raw_questions, list):
            return []
        out: list[tuple[str, dict[str, Any]]] = []
        for item in raw_questions:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if not question or not answer:
                continue
            if len(answer) > _MAX_ANSWER_CHARS:
                continue
            if not _answer_supported(answer, chunk.text):
                # The LLM invented something -> drop the question.
                continue
            if _leaks(answer, question):
                continue
            qtype = str(item.get("type") or "").strip()
            if qtype not in ("single_hop", "multi_hop", "temporal"):
                qtype = self._classify(question)
            entities = [str(e).strip() for e in (item.get("entities") or []) if str(e).strip()]
            years = []
            for y in item.get("years") or []:
                try:
                    years.append(int(y))
                except (TypeError, ValueError):
                    continue
            out.append(
                (
                    qtype,
                    {
                        "question": question,
                        "answer": answer,
                        "entities": entities,
                        "years": years,
                    },
                )
            )
        return out

    def _heuristic_questions(self, chunk: Chunk) -> list[tuple[str, dict[str, Any]]]:
        """Deterministic fallback questions derived from the chunk itself.

        Used when no LLM is available (``--mock`` / ``--offline``) so the
        pipeline still produces a usable evaluation set offline.
        """
        out: list[tuple[str, dict[str, Any]]] = []
        sentences = _split_sentences(chunk.text)
        if not sentences:
            return out
        title = chunk.title or chunk.doc_id

        # temporal: a sentence carrying a year -> ask about that year.
        for sentence in sentences:
            years = _extract_years(sentence)
            if not years:
                continue
            entities = _capitals(sentence, limit=4)
            out.append(
                (
                    "temporal",
                    {
                        "question": (
                            f"Theo tài liệu về {title}, điều gì đã xảy ra vào năm "
                            f"{years[0]}?"
                        ),
                        "answer": sentence.strip(),
                        "entities": entities or [title],
                        "years": years,
                    },
                )
            )
            break

        # single hop: a factual sentence about the title entity.
        for sentence in sentences:
            if len(sentence) < _MIN_ANSWER_CHARS:
                continue
            entities = _capitals(sentence, limit=4) or [title]
            years = _extract_years(sentence)
            question = f"Theo tài liệu về {title}, {title} là ai và có vai trò gì?"
            out.append(
                (
                    "single_hop",
                    {
                        "question": question,
                        "answer": sentence.strip(),
                        "entities": entities,
                        "years": years,
                    },
                )
            )
            break

        # multi hop: a sentence mentioning two distinct entities.
        for sentence in sentences:
            entities = _capitals(sentence, limit=4)
            if len(entities) >= 2:
                out.append(
                    (
                        "multi_hop",
                        {
                            "question": (
                                f"Theo tài liệu về {title}, {entities[0]} có mối liên hệ "
                                f"gì với {entities[1]}?"
                            ),
                            "answer": sentence.strip(),
                            "entities": entities[:2],
                            "years": _extract_years(sentence),
                        },
                    )
                )
                break
        return out

    def _llm_candidate(
        self,
        chunk: Chunk,
        qtype: str,
        payload: dict[str, Any],
    ) -> _Candidate:
        entities = [e for e in payload.get("entities") or [] if e]
        title = chunk.title or chunk.doc_id
        if not entities:
            entities = [title]
        qids = []
        for entity in entities:
            matched = self._entity_index.get(name_key(entity))
            if matched is not None and matched.wikidata_id:
                qids.append(matched.wikidata_id)
        years = [int(y) for y in payload.get("years") or []] or _extract_years(payload["question"])
        path = list(entities)
        if len(path) < 2:
            path = [title] + path
        difficulty = {"single_hop": "easy", "multi_hop": "medium", "temporal": "medium"}.get(
            qtype, "medium"
        )
        if qtype == "multi_hop" and len(entities) < 2:
            # Without two anchors the question is not really multi-hop.
            qtype = "single_hop"
            difficulty = "easy"
        question = EvalQuestion(
            qid="",
            question=payload["question"],
            answer=payload["answer"],
            question_type=qtype,
            gold_entities=entities,
            gold_qids=qids,
            gold_paths=[path],
            gold_chunk_ids=[chunk.chunk_id],
            gold_relations=[],
            year_refs=years,
            difficulty=difficulty,
            source="llm_from_wiki",
            metadata={
                "doc_id": chunk.doc_id,
                "title": title,
                "excluded_chunks": [],
            },
        )
        return _Candidate(question=question, priority=1)

    @staticmethod
    def _classify(question: str) -> str:
        """Heuristic classification of a generated question."""
        text = question.lower()
        if any(k in text for k in ("khi nào", "năm nào", "thời đại nào", "mấy năm", "bao lâu")):
            return "temporal"
        if any(
            k in text for k in ("so sánh", "mối liên hệ", "liên quan đến cả hai", "vai trò")
        ):
            return "multi_hop"
        return "single_hop"

    # -- source 3: history chat -----------------------------------------
    def _gen_chat_candidates(self, offline: bool = False) -> list[_Candidate]:
        from ..data import history_chat

        cache_dir = self._cache_dir("history_chat")
        records = history_chat.load_history_chat(
            limit=500,
            cache_dir=cache_dir,
            offline=offline,
            min_year=905,
            max_year=1975,
        )
        candidates: list[_Candidate] = []
        for record in records:
            question = str(record.get("question") or "").strip()
            answer = str(record.get("answer") or "").strip()
            if not question or not answer:
                continue
            if len(answer) > _MAX_ANSWER_CHARS:
                continue
            if len(answer) < _MIN_CHAT_ANSWER_CHARS:
                continue
            if len(question) < _MIN_CHAT_QUESTION_CHARS:
                continue
            years = _extract_years(question) or _extract_years(answer)
            entities = self._chat_entities(question)
            chunks = self._match_chunks(question)
            qtype = self._classify(question)
            if qtype == "multi_hop" and len(entities) < 2:
                # A multi-hop question needs at least two anchors -> otherwise
                # it is really a single-hop question.
                qtype = "single_hop"
            difficulty = "hard" if qtype == "multi_hop" else "medium"
            candidates.append(
                _Candidate(
                    question=EvalQuestion(
                        qid="",
                        question=question,
                        answer=answer,
                        question_type=qtype,
                        gold_entities=entities,
                        gold_qids=[],
                        gold_paths=[entities] if len(entities) >= 2 else [],
                        gold_chunk_ids=chunks,
                        gold_relations=[],
                        year_refs=years,
                        difficulty=difficulty,
                        source="history_chat",
                        metadata={
                            "excluded_chunks": [],
                            "note": (
                                "Đáp án do LLM tạo trong dataset gốc -> cần human-verify "
                                "(docs/04 §2.3)"
                            ),
                        },
                    ),
                    priority=2,
                )
            )
        return candidates

    def _chunk_token_cache(self) -> dict[str, set[str]]:
        """Token set of every corpus chunk (computed once, then cached)."""
        cache = getattr(self, "_chunk_tokens", None)
        if cache is None:
            if not self.corpus_chunks:
                self.corpus_chunks = self._load_corpus()
            cache = {c.chunk_id: _token_set(c.text) for c in self.corpus_chunks}
            self._chunk_tokens = cache
        return cache

    def _chat_entities(self, question: str) -> list[str]:
        """Entities mentioned in a history-chat question.

        The raw capitalised n-grams of a chat question are noisy
        (``"Phân tích"`` -> ``"Phân"``), so the corpus titles mentioned in the
        question are preferred, then multi-word names.
        """
        entities: list[str] = []
        # 1. corpus titles explicitly mentioned in the question.
        for chunk in self.corpus_chunks:
            title = chunk.title
            if title and title in question and title not in entities:
                entities.append(title)
        if entities:
            return entities[:5]
        # 2. multi-word capitalised names.
        raw = _capitals(question, limit=10)
        entities = [name for name in raw if _is_usable_entity(name)]
        return entities[:5]

    def _match_chunks(self, question: str) -> list[str]:
        """Find corpus chunks that mention the question's key words."""
        tokens = {t for t in _token_set(question) if len(t) > 3}
        if not tokens:
            return []
        cache = self._chunk_token_cache()
        hits: list[tuple[int, str]] = []
        for chunk_id, haystack in cache.items():
            overlap = len(tokens & haystack)
            if overlap >= max(2, len(tokens) // 2):
                hits.append((overlap, chunk_id))
        hits.sort(reverse=True)
        return [chunk_id for _, chunk_id in hits[:2]]

    # -- shared helpers -------------------------------------------------
    def _cache_dir(self, name: str) -> Path | None:
        if self.settings is None:
            return None
        return Path(self.settings.data_raw) / name

    def _excluded_chunks(self, entities: Iterable[str], keep: str = "") -> list[str]:
        """Chunks that look related but must NOT be used as the answer.

        They are the *decoy* documents for the exclusive-document evaluation
        (docs/04 §1.1): the answer must not be found in them.
        """
        if not self.corpus_chunks:
            self.corpus_chunks = self._load_corpus()
        if not self.corpus_chunks:
            return []
        keys = {name_key(e) for e in entities if e}
        keys.discard("")
        if not keys:
            return []
        # A mention matches a chunk when all its tokens occur in the chunk.
        token_sets = [set(key.split()) for key in keys]
        excluded: list[str] = []
        for chunk_id, haystack in self._chunk_token_cache().items():
            if keep and chunk_id == keep:
                continue
            if any(tokens and tokens.issubset(haystack) for tokens in token_sets):
                excluded.append(chunk_id)
            if len(excluded) >= 10:
                break
        return excluded
