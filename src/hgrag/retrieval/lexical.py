"""Lexical (BM25) retrieval over the corpus -- fallback khi không có embedding.

Free-tier Gemini có quota embedding riêng và có thể cạn trong lúc một lần
chạy đánh giá dài đang diễn ra.  Khi đó vector search trả về rỗng và hệ
thống mất khả năng trả lời, dù corpus vẫn còn đầy đủ.

Module này cung cấp một *index từ khoá* độc lập với embedding:

* :class:`LexicalIndex` -- chỉ mục BM25 trên các chunk trong corpus;
* :func:`build_lexical_index` -- dựng chỉ mục từ ``corpus.jsonl`` (hoặc từ
  payload của Qdrant) và cache nó trên đĩa.

Khi pipeline không embed được câu hỏi, nó dùng chỉ mục này để lấy
candidate, nhờ vậy hệ thống vẫn trả lời được (chỉ là chất lượng vector
search bị thay bằng BM25).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["LexicalIndex", "build_lexical_index", "tokenize_for_search"]

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_YEAR_RE = re.compile(r"\b(\d{3,4})\b")

#: Stop-words tiếng Việt rút gọn (đủ để BM25 không bị nhiễu).
STOPWORDS: frozenset[str] = frozenset(
    {
        "va", "hoac", "cua", "cho", "voi", "trong", "ngoai", "mot", "nhung",
        "la", "co", "khong", "duoc", "da", "de", "nay", "do", "ve", "theo", "khi",
        "bao", "gio", "nhu", "ra", "roi", "bi", "ma", "thanh", "o", "tai",
        "nen", "hay", "neu", "thi", "se", "boi", "vay", "day", "qua",
        "gi", "nao", "ai", "toi", "chung", "nhom", "cac",
    }
)


def tokenize_for_search(text: str) -> list[str]:
    """Token hoá văn bản tiếng Việt (bỏ dấu, bỏ stop-word, giữ số năm)."""
    from ..data.normalize import strip_diacritics

    out: list[str] = []
    for raw in _TOKEN_RE.findall(strip_diacritics(text or "").casefold()):
        if raw in STOPWORDS:
            continue
        out.append(raw)
    for year in _YEAR_RE.findall(text or ""):
        out.append(f"y{year}")
    return out


def _payload_matches(payload: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    """Kiểm tra một payload có khớp bộ lọc hay không.

    Hỗ trợ hai kiểu lọc đặc biệt:
    * ``year_overlap`` -- chunk có khoảng năm ``[year_min, year_max]`` chồng
      lấn một trong các năm được yêu cầu;
    * các khoá còn lại -- so sánh bằng ``==`` (hoặc membership nếu giá trị là
      một tập).
    """
    for key, value in filters.items():
        if key == "year_overlap":
            years = [int(y) for y in (value or [])]
            if not years:
                continue
            y_min = payload.get("year_min")
            y_max = payload.get("year_max")
            if y_min is None or y_max is None:
                return False
            if not any(int(y_min) <= year <= int(y_max) for year in years):
                return False
            continue
        actual = payload.get(key)
        if isinstance(value, (list, tuple, set)):
            if actual not in value:
                return False
        elif actual != value:
            return False
    return True


class LexicalIndex:
    """Chỉ mục BM25 đơn giản trên một tập văn bản."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids: list[str] = []
        self.doc_tokens: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.doc_freq: Counter[str] = Counter()
        self.payloads: list[dict[str, Any]] = []
        self._avg_len: float = 0.0

    # ------------------------------------------------------------------
    def add(self, doc_id: str, text: str, payload: Mapping[str, Any] | None = None) -> None:
        tokens = tokenize_for_search(text)
        counts = Counter(tokens)
        self.doc_ids.append(doc_id)
        self.doc_tokens.append(counts)
        self.doc_len.append(len(tokens))
        self.doc_freq.update(counts.keys())
        self.payloads.append(dict(payload or {}))

    def finalize(self) -> None:
        total = sum(self.doc_len)
        self._avg_len = (total / len(self.doc_len)) if self.doc_len else 0.0

    # ------------------------------------------------------------------
    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        n = max(1, self.n_docs)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Trả về các chunk khớp nhất với ``query`` (điểm BM25 đã chuẩn hoá)."""
        if not self.doc_ids:
            return []
        q_tokens = tokenize_for_search(query)
        if not q_tokens:
            return []
        q_counts = Counter(q_tokens)
        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self.doc_tokens):
            length = self.doc_len[i] or 1
            score = 0.0
            for term, qtf in q_counts.items():
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * (length / (self._avg_len or 1.0))
                )
                score += self._idf(term) * (tf * (self.k1 + 1) / denom) * min(qtf, 3)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: -item[0])
        out: list[dict[str, Any]] = []
        best = scored[0][0] if scored else 1.0
        for score, i in scored:
            payload = self.payloads[i]
            if filters and not _payload_matches(payload, filters):
                continue
            out.append(
                {
                    "id": self.doc_ids[i],
                    "score": round(float(score / (best or 1.0)), 6),
                    "payload": payload,
                }
            )
            if len(out) >= top_k:
                break
        return out

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "payloads": self.payloads,
            "doc_freq": dict(self.doc_freq),
            "doc_tokens": [dict(c) for c in self.doc_tokens],
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LexicalIndex:
        index = cls(
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )
        index.doc_ids = [str(x) for x in payload.get("doc_ids") or []]
        index.doc_len = [int(x) for x in payload.get("doc_len") or []]
        index.payloads = [dict(x) for x in payload.get("payloads") or []]
        index.doc_freq = Counter(payload.get("doc_freq") or {})
        index.doc_tokens = [Counter(x) for x in payload.get("doc_tokens") or []]
        index.finalize()
        return index


def build_lexical_index(
    corpus_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    limit: int | None = None,
) -> LexicalIndex:
    """Dựng (hoặc nạp từ cache) chỉ mục từ khoá cho corpus."""
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        try:
            return LexicalIndex.from_dict(json.loads(cache.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("Không đọc được lexical cache (%s) -> dựng lại", exc)

    index = LexicalIndex()
    if corpus_path and Path(corpus_path).exists():
        n = 0
        with Path(corpus_path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                chunk_id = str(row.get("chunk_id") or "")
                text = "\n".join(
                    str(row.get(k) or "")
                    for k in ("title", "section", "text")
                ).strip()
                if not chunk_id or not text:
                    continue
                index.add(
                    chunk_id,
                    text,
                    {
                        "chunk_id": chunk_id,
                        "doc_id": row.get("doc_id"),
                        "title": row.get("title"),
                        "section": row.get("section"),
                        "source": row.get("source"),
                        "url": row.get("url"),
                        "year_min": row.get("year_min"),
                        "year_max": row.get("year_max"),
                        "text": row.get("text"),
                    },
                )
                n += 1
                if limit and n >= limit:
                    break
    index.finalize()
    logger.info("Lexical index: %d chunk", index.n_docs)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
    return index


def iter_payload_texts(rows: Iterable[Mapping[str, Any]]) -> Iterable[tuple[str, str]]:
    """Yield ``(chunk_id, text)`` from raw store payloads."""
    for row in rows:
        payload = row.get("payload") or row
        chunk_id = str(payload.get("chunk_id") or row.get("id") or "")
        text = "\n".join(
            str(payload.get(k) or "")
            for k in ("title", "section", "text")
        ).strip()
        if chunk_id and text:
            yield chunk_id, text
