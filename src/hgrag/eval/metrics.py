"""Metric implementations for the evaluation harness (docs/04 §3).

The module is deliberately **pure**: every function takes plain Python data
and returns plain Python data, so the metrics can be unit-tested without any
LLM, Neo4j or Qdrant dependency.

Families of metrics
-------------------
* ``token_f1`` / ``rouge_l``      -- lexical overlap with the reference answer
* ``citation_precision`` / ``citation_recall`` -- grounding of the answer
* ``chunk_recall_at_k`` / ``ndcg_at_k``      -- retrieval quality
* ``path_hit_rate`` / ``entity_recall`` / ``temporal_accuracy``
* ``bootstrap_ci`` / ``mcnemar_test``         -- statistics (docs/04 §3.5)
"""

from __future__ import annotations

import math
import random
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "normalize_text",
    "tokenize",
    "token_f1",
    "rouge_l",
    "extract_citation_markers",
    "citation_precision",
    "citation_recall",
    "no_citation_rate",
    "chunk_recall_at_k",
    "ndcg_at_k",
    "entity_recall",
    "path_hit_rate",
    "temporal_accuracy",
    "percentile",
    "BootstrapResult",
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "mcnemar_test",
    "McNemarResult",
    "gold_fact_accuracy",
    "aggregate",
]

#: Vietnamese stop-words removed before computing lexical overlap.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "ai",
        "bao",
        "bao_giờ",
        "bao_nhiêu",
        "bị",
        "bởi",
        "cả",
        "các",
        "cái",
        "cần",
        "càng",
        "chỉ",
        "chiếc",
        "cho",
        "chứ",
        "chưa",
        "có",
        "cùng",
        "cũng",
        "đã",
        "đang",
        "đây",
        "để",
        "đến",
        "đều",
        "đi",
        "điều",
        "do",
        "đó",
        "được",
        "dưới",
        "gì",
        "hay",
        "hơn",
        "khi",
        "không",
        "là",
        "lại",
        "lên",
        "lúc",
        "mà",
        "mỗi",
        "một",
        "nào",
        "này",
        "nên",
        "nếu",
        "ngay",
        "nhiều",
        "như",
        "những",
        "nơi",
        "nữa",
        "phải",
        "qua",
        "ra",
        "rằng",
        "rất",
        "rồi",
        "sau",
        "sẽ",
        "so",
        "sự",
        "tại",
        "theo",
        "thì",
        "trong",
        "trên",
        "trước",
        "từ",
        "từng",
        "và",
        "vẫn",
        "vào",
        "vậy",
        "vì",
        "việc",
        "với",
        "vừa",
        "của",
        "về",
        "năm",
        "thế",
        "sao",
        "nhiêu",
    }
)

_CITATION_RE = re.compile(r"\[E(\d+)\]", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|[1-9][0-9]{2})(?!\d)")


# --------------------------------------------------------------------------
# text normalisation
# --------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Lower-case, strip diacritics and collapse whitespace."""
    if not text:
        return ""
    lowered = str(text).lower().strip()
    decomposed = unicodedata.normalize("NFD", lowered)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = stripped.replace("đ", "d").replace("Đ", "d")
    stripped = re.sub(r"[^\w\s]", " ", stripped, flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip()


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenise a Vietnamese text for lexical metrics."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens = [t for t in normalized.split() if t]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# --------------------------------------------------------------------------
# lexical metrics
# --------------------------------------------------------------------------


def token_f1(prediction: str, reference: str) -> float:
    """F1 between the token multisets of ``prediction`` and ``reference``."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 1.0 if not pred_tokens and not ref_tokens else 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            ref_counts[token] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure (LCS based)."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 1.0 if not pred_tokens and not ref_tokens else 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# citation metrics
# --------------------------------------------------------------------------


def extract_citation_markers(text: str) -> list[str]:
    """The ``E#`` markers used in ``text`` (``["E1", "E2"]``)."""
    out: list[str] = []
    for match in _CITATION_RE.finditer(text or ""):
        marker = f"E{match.group(1)}"
        if marker not in out:
            out.append(marker)
    return out


def citation_precision(
    answer: str,
    cited_evidence: Sequence[Any] | None = None,
    supported: Sequence[str] | None = None,
) -> float:
    """Fraction of cited evidence that actually supports a claim.

    ``supported`` is the list of evidence ids the judge marked as *supporting*
    the answer.  When it is ``None`` the precision falls back to ``1.0`` for
    every *valid* citation (i.e. a citation pointing at an unknown evidence id
    counts as a violation).
    """
    markers = extract_citation_markers(answer)
    if not markers:
        return 0.0
    if supported is not None:
        good = {str(s).strip().upper() for s in supported}
        hits = sum(1 for m in markers if m in good)
        return hits / len(markers)
    valid = {str(e).strip().upper() for e in (cited_evidence or [])}
    if not valid:
        return 0.0
    hits = sum(1 for m in markers if m in valid)
    return hits / len(markers)


def citation_recall(
    answer: str,
    gold_chunk_ids: Sequence[str] | None = None,
    evidence_chunk_ids: Sequence[str] | None = None,
) -> float:
    """Fraction of the gold evidence that the answer actually cites.

    ``gold_chunk_ids`` are the gold chunk ids; ``evidence_chunk_ids`` maps an
    evidence id (``E1``) to its chunk id.  When the mapping is unavailable the
    metric degrades to "did the answer cite anything at all".
    """
    markers = set(extract_citation_markers(answer))
    gold = {str(g) for g in (gold_chunk_ids or [])}
    if not gold:
        return 1.0 if markers else 0.0
    mapping = {
        str(k).strip().upper(): str(v)
        for k, v in (evidence_chunk_ids or {}).items()
    } if isinstance(evidence_chunk_ids, dict) else {}
    if mapping:
        cited_chunks = {mapping[m] for m in markers if m in mapping}
        return len(gold & cited_chunks) / len(gold)
    return 1.0 if markers else 0.0


def no_citation_rate(answers: Iterable[str]) -> float:
    """Share of answers that carry no valid citation marker."""
    answers = list(answers)
    if not answers:
        return 0.0
    bad = sum(1 for a in answers if not extract_citation_markers(a))
    return bad / len(answers)


# --------------------------------------------------------------------------
# retrieval metrics
# --------------------------------------------------------------------------


def chunk_recall_at_k(
    gold_chunk_ids: Sequence[str], retrieved_chunk_ids: Sequence[str], k: int = 10
) -> float:
    """Share of gold chunks present in the top-``k`` retrieved chunks."""
    gold = {str(g) for g in (gold_chunk_ids or [])}
    if not gold:
        return 0.0
    top = [str(r) for r in (retrieved_chunk_ids or [])[: max(0, k)]]
    return len(gold & set(top)) / len(gold)


def ndcg_at_k(
    gold_chunk_ids: Sequence[str], retrieved_chunk_ids: Sequence[str], k: int = 10
) -> float:
    """nDCG@k with a binary relevance judgement."""
    gold = {str(g) for g in (gold_chunk_ids or [])}
    if not gold:
        return 0.0
    retrieved = [str(r) for r in (retrieved_chunk_ids or [])][: max(0, k)]
    dcg = 0.0
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            dcg += 1.0 / math.log2(index + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def entity_recall(
    gold_entities: Sequence[str], linked_entities: Sequence[str]
) -> float:
    """Share of gold entities that were linked during retrieval."""
    gold = [normalize_text(g) for g in (gold_entities or []) if str(g).strip()]
    if not gold:
        return 0.0
    linked = {normalize_text(item) for item in (linked_entities or [])}
    linked |= {normalize_text(part) for name in linked for part in name.split()}
    hits = sum(1 for g in gold if g and (g in linked or g in " ".join(linked)))
    return hits / len(gold)


def path_hit_rate(
    gold_paths: Sequence[Sequence[str]], retrieved_paths: Sequence[Sequence[str]]
) -> float:
    """Share of gold paths that appear among the retrieved paths.

    A path is considered hit when the gold node sequence is a *subsequence*
    of a retrieved path (same order, consecutive nodes not required).
    """
    gold_list = [list(p) for p in (gold_paths or []) if p]
    if not gold_list:
        return 0.0
    retrieved = [list(p) for p in (retrieved_paths or []) if p]
    if not retrieved:
        return 0.0
    normalized = [[normalize_text(n) for n in path] for path in retrieved]
    hits = 0
    for gold_path in gold_list:
        target = [normalize_text(n) for n in gold_path]
        found = False
        for path in normalized:
            it = iter(path)
            if all(any(node == g for node in it) for g in target):
                found = True
                break
        if found:
            hits += 1
    return hits / len(gold_list)


def temporal_accuracy(
    gold_years: Sequence[int], evidence_texts: Sequence[str]
) -> float:
    """Whether the evidence mentions (at least) one of the gold years."""
    gold = {int(y) for y in (gold_years or [])}
    if not gold:
        return 0.0
    for text in evidence_texts or []:
        years = {int(m) for m in _YEAR_RE.findall(text or "")}
        if gold & years:
            return 1.0
    return 0.0


def gold_fact_accuracy(
    gold_relations: Sequence[str], answer: str, judge=None
) -> float:
    """Share of gold triples that are reflected by the answer.

    When a ``judge`` callable is provided it is used (``judge(triple, answer)
    -> bool``); otherwise a lexical containment check is applied.
    """
    triples = [str(t) for t in (gold_relations or []) if str(t).strip()]
    if not triples:
        return 0.0
    normalized_answer = normalize_text(answer)
    hits = 0
    for triple in triples:
        if judge is not None:
            try:
                if judge(triple, answer):
                    hits += 1
                    continue
            except Exception:  # noqa: BLE001 - metric must never crash
                pass
        parts = [normalize_text(p) for p in re.split(r"\s*-\w+->\s*", triple) if p]
        if parts and all(p and p in normalized_answer for p in parts):
            hits += 1
    return hits / len(triples)


# --------------------------------------------------------------------------
# operational metrics
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100) of ``values`` (linear interpolation)."""
    data = sorted(float(v) for v in values if v is not None)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    q = min(max(q, 0.0), 100.0)
    position = (len(data) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[int(position)]
    return data[lower] + (data[upper] - data[lower]) * (position - lower)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """The outcome of a bootstrap resampling."""

    mean: float = 0.0
    std: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    n_samples: int = 0
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "n_samples": self.n_samples,
            "confidence": self.confidence,
        }

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low


def bootstrap_ci(
    values: Sequence[float],
    n_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI of the mean of ``values``."""
    data = [float(v) for v in values if v is not None]
    if not data:
        return BootstrapResult(n_samples=0, confidence=confidence)
    rng = random.Random(seed)
    n = len(data)
    means: list[float] = []
    for _ in range(max(1, n_samples)):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low = means[max(0, int(alpha * len(means)) - 1)]
    high = means[min(len(means) - 1, int((1 - alpha) * len(means)))]
    mean = sum(data) / n
    variance = sum((v - mean) ** 2 for v in data) / n if n > 1 else 0.0
    return BootstrapResult(
        mean=mean,
        std=math.sqrt(variance),
        ci_low=low,
        ci_high=high,
        n_samples=len(means),
        confidence=confidence,
    )


def paired_bootstrap_ci(
    system_a: Sequence[float],
    system_b: Sequence[float],
    n_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Paired bootstrap CI of ``mean(a - b)`` (same questions, same order)."""
    if len(system_a) != len(system_b):
        raise ValueError("Hai hệ thống phải được đánh giá trên cùng bộ câu hỏi")
    diffs = [float(a) - float(b) for a, b in zip(system_a, system_b, strict=True)]
    return bootstrap_ci(diffs, n_samples=n_samples, confidence=confidence, seed=seed)


@dataclass
class McNemarResult:
    """The outcome of a McNemar test on paired binary outcomes."""

    b: int = 0  # system A correct, system B wrong
    c: int = 0  # system A wrong, system B correct
    n: int = 0
    chi2: float = 0.0
    p_value: float = 1.0
    significant: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "b": self.b,
            "c": self.c,
            "n": self.n,
            "chi2": round(self.chi2, 6),
            "p_value": round(self.p_value, 6),
            "significant": self.significant,
        }


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 degree of freedom.

    For 1 dof: ``P(X > x) = erfc(sqrt(x / 2))``.
    """
    if x <= 0:
        return 1.0
    return max(0.0, min(1.0, math.erfc(math.sqrt(x / 2.0))))


def mcnemar_test(
    system_a: Sequence[bool | int],
    system_b: Sequence[bool | int],
    alpha: float = 0.05,
) -> McNemarResult:
    """McNemar's exact-ish test on two systems scored on the same questions."""
    if len(system_a) != len(system_b):
        raise ValueError("Hai hệ thống phải được đánh giá trên cùng bộ câu hỏi")
    b = 0
    c = 0
    for a, bb in zip(system_a, system_b, strict=True):
        a_ok = bool(a)
        b_ok = bool(bb)
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
    n = b + c
    if n == 0:
        return McNemarResult(b=0, c=0, n=0, chi2=0.0, p_value=1.0, significant=False)
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_value = _chi2_sf_1df(chi2)
    return McNemarResult(
        b=b,
        c=c,
        n=n,
        chi2=chi2,
        p_value=p_value,
        significant=p_value < alpha,
    )


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def aggregate(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    """Average the numeric ``keys`` of ``rows`` (non-numeric values are ignored)."""
    out: dict[str, Any] = {}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and isinstance(row.get(key), (int, float))
        ]
        out[key] = (sum(values) / len(values)) if values else 0.0
    return out


@dataclass
class MetricBundle:
    """All metrics computed for a single question."""

    values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)
