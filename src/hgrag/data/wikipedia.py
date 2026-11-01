"""Wikipedia (vi) ingestion via the MediaWiki Action API.

Only the public ``api.php`` endpoint is used -- no scraping, no SPARQL.

Endpoints used
--------------
``action=query&prop=extracts``      -> plain-text article body (with headings)
``action=query&prop=pageprops``     -> ``wikibase_item`` (Wikidata QID)
``action=query&prop=categories``    -> category names (used for domain filters)
``action=query&prop=redirects``     -> redirect titles (alias candidates)
``action=query&prop=links``         -> outgoing links (graph expansion)
``action=opensearch`` / ``list=search`` -> title search

Every request is wrapped in ``try/except`` and every failure only produces a
warning -- the ingest pipeline must never crash because one article failed.
Successful articles are cached on disk under ``data/raw/wiki/<slug>.json``.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SEED_TITLES",
    "WikipediaClient",
    "fetch_article",
    "fetch_page",
    "search_titles",
    "expand_from_seeds",
    "title_slug",
    "is_in_history_scope",
]

API_BASE = "https://vi.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "VietnamHistoryGraphRAG/0.1 (research; course project)"
DEFAULT_DELAY = 0.5
REQUEST_TIMEOUT = 30

# --------------------------------------------------------------------------
# Seed articles -- Vietnamese history 905-1975 (see docs/01 §1.2).
#
# The order is *interleaved on purpose*: the first entries span the whole
# timeline (Early -> Lý -> Trần -> Lê -> Nguyễn -> colonial -> modern), so a
# small ``--limit`` still yields a chronologically diverse corpus.
# Every title was verified to exist on vi.wikipedia.org.
# --------------------------------------------------------------------------
DEFAULT_SEED_TITLES: tuple[str, ...] = (
    # -- interleave: Lý / Trần / Lê / Nguyễn / hiện đại ---------------
    "Lý Thái Tổ",
    "Trần Hưng Đạo",
    "Lê Lợi",
    "Gia Long",
    "Hồ Chí Minh",
    # -- tiền Lý ------------------------------------------------------
    "Đinh Bộ Lĩnh",
    "Lê Hoàn",
    "Ngô Quyền",
    "Bà Triệu",
    "Trưng Trắc",
    "Trưng Nhị",
    # -- nhà Lý -------------------------------------------------------
    "Lý Thường Kiệt",
    "Lý Thánh Tông",
    "Lý Nhân Tông",
    "Lý Thái Tông",
    # -- nhà Trần -----------------------------------------------------
    "Trần Cảnh",
    "Trần Thái Tông",
    "Trần Thánh Tông",
    "Trần Anh Tông",
    "Trần Minh Tông",
    "Trần Nghệ Tông",
    "Trần Quang Khải",
    "Hồ Quý Ly",
    # -- nhà Lê -------------------------------------------------------
    "Lê Thái Tổ",
    "Lê Thánh Tông",
    "Lê Hiến Tông",
    "Nguyễn Trãi",
    "Lam Sơn",
    # -- nhà Nguyễn ---------------------------------------------------
    "Nguyễn Ánh",
    "Minh Mạng",
    "Tự Đức",
    "Nguyễn Trung Trực",
    "Nguyễn Du",
    "Nguyễn Văn Thiệu",
    "Ngô Đình Diệm",
    # -- thời kỳ Pháp thuộc & hiện đại --------------------------------
    "Sài Gòn",
    "Chiến dịch Điện Biên Phủ",
    "Việt Nam Dân chủ Cộng hòa",
    "Việt Nam Cộng hòa",
    "Việt Nam",
    "Hiến pháp Việt Nam Dân chủ Cộng hòa năm 1946",
    "Quốc hội",
    "Đảng Cộng sản Việt Nam",
    "Võ Nguyên Giáp",
    "Lê Đức Thọ",
    "Phan Bội Châu",
    "Trường Chinh",
    "Nguyễn Chí Thanh",
    "Bảo Đại",
    # -- nhân vật / sự kiện tiêu biểu ---------------------------------
    "Văn Cao",
    "Kim Đồng",
    "Đỗ Hành",
    "Đỗ Mười",
    "Chiến tranh Việt Nam",
    "Cách mạng Tháng Tám",
    "Phong trào Đồng khởi",
)

#: Categories that mark an article as being about Vietnamese history.
_HISTORY_CATEGORY_HINTS = (
    "lịch sử việt nam",
    "nhà lý",
    "nhà trần",
    "nhà hậu trần",
    "nhà lê",
    "nhà nguyễn",
    "vương triều",
    "người việt nam",
    "anh hùng dân tộc việt nam",
    "thành phố việt nam",
    "tỉnh",
    "thời việt nam",
    "chiến tranh việt nam",
    "mất năm",
    "sinh thế kỷ",
    "quân đội việt nam",
    "hoàng đế việt nam",
    "nhà vua việt nam",
    "thị tướng việt nam",
    "nhà sử học việt nam",
    "nhà văn việt nam",
)

#: Links whose titles start with these prefixes are almost never useful for
#: expanding the Vietnamese-history sub-graph.
_SKIP_LINK_PREFIXES = (
    "Thể loại:",
    "Wikipedia:",
    "Chủ đề:",
    "Phụ lục:",
    "Tập tin:",
    "Tin tức:",
    "Cổng thông tin:",
    "Bản mẫu:",
    "Danh sách",
    "Bảng",
    "Chuyên mục",
)

_SECTION_STRIP_RE = re.compile(r"^=+\s*(.+?)\s*=+$", re.MULTILINE)
_CATEGORY_RE = re.compile(r"\[\[(?:Thể loại|Category):([^\]|]+)(?:\|[^\]]*)?\]\]")
_REF_RE = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_REF_SECTION_RE = re.compile(
    r"^==\s*(Xem thêm|Chú thích|Tham khảo|Thư mục|Tài liệu tham khảo|Nguồn|External links|References)\s*==",
    re.IGNORECASE | re.MULTILINE,
)
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|[1-9][0-9]{2})\b")


def title_slug(title: str) -> str:
    """Filesystem-safe slug for an article title."""
    text = unicodedata.normalize("NFC", title or "").strip()
    text = re.sub(r"[\s]+", "_", text)
    text = re.sub(r'[<>:"/\\|?*]', "", text)
    return text[:180] or "untitled"


def _remove_nested(text: str, open_tok: str, close_tok: str) -> str:
    """Remove balanced ``open_tok ... close_tok`` blocks (e.g. ``{{...}}``)."""
    out: list[str] = []
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        if text.startswith(open_tok, i):
            depth += 1
            i += len(open_tok)
            continue
        if depth and text.startswith(close_tok, i):
            depth -= 1
            i += len(close_tok)
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def _strip_markup(text: str) -> str:
    """Remove wiki markup that would pollute a plain-text chunk."""
    if not text:
        return ""
    text = _COMMENT_RE.sub("", text)
    text = _REF_RE.sub("", text)
    # Remove the trailing reference-ish sections entirely.
    m = _REF_SECTION_RE.search(text)
    if m:
        text = text[: m.start()]
    # Infoboxes, tables and templates carry no narrative text.
    text = _remove_nested(text, "<table", "</table>")
    text = _remove_nested(text, "{{", "}}")
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]|]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\]", "", text)
    text = re.sub(r"'''?", "", text)
    text = _TAG_RE.sub("", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&quot;", '"')
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_wiki_sections(text: str) -> list[dict[str, str]]:
    """Parse ``== Heading ==`` markup into ``[{"heading":..., "text":...}]``."""
    sections: list[dict[str, str]] = []
    current_heading = ""
    current: list[str] = []
    for line in text.splitlines():
        m = _SECTION_STRIP_RE.match(line.strip())
        if m:
            body = "\n".join(current).strip()
            if body:
                sections.append({"heading": current_heading, "text": body})
            current_heading = m.group(1).strip()
            current = []
        else:
            current.append(line)
    body = "\n".join(current).strip()
    if body:
        sections.append({"heading": current_heading, "text": body})
    return sections


class WikipediaClient:
    """Thin, well-behaved wrapper around the MediaWiki Action API."""

    def __init__(
        self,
        lang: str = "vi",
        delay: float = DEFAULT_DELAY,
        user_agent: str = DEFAULT_USER_AGENT,
        cache_dir: str | Path | None = None,
        timeout: int = REQUEST_TIMEOUT,
        offline: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.lang = lang
        self.delay = max(0.0, float(delay))
        self.user_agent = user_agent
        self.timeout = timeout
        self.offline = offline
        self.api_base = f"https://{lang}.wikipedia.org/w/api.php"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self._last_request = 0.0

    # ------------------------------------------------------------------
    # low-level
    # ------------------------------------------------------------------
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
        delay = max(self.delay, 0.5)
        for attempt in range(retries):
            self._throttle(delay)
            try:
                resp = self.session.get(self.api_base, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                logger.warning("Wikipedia request failed (%s): %s", params.get("action"), exc)
                return {}
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    logger.warning("Wikipedia returned non-JSON: %s", exc)
                    return {}
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = delay * (2**attempt)
                logger.warning(
                    "Wikipedia HTTP %s -> retry in %.1fs (%s)",
                    resp.status_code,
                    wait,
                    params.get("titles", params.get("action")),
                )
                time.sleep(wait)
                delay = max(delay, 1.0)
                continue
            logger.warning("Wikipedia HTTP %s for %s", resp.status_code, params.get("titles"))
            return {}
        logger.warning("Wikipedia request gave up after %d retries", retries)
        return {}

    # ------------------------------------------------------------------
    # cache
    # ------------------------------------------------------------------
    def _cache_path(self, title: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{title_slug(title)}.json"

    def _cache_load(self, title: str) -> dict[str, Any] | None:
        path = self._cache_path(title)
        if path and path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Cannot read cache %s: %s", path, exc)
        return None

    def _cache_save(self, title: str, payload: dict[str, Any]) -> None:
        path = self._cache_path(title)
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot write cache %s: %s", path, exc)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fetch_page(self, title: str, use_cache: bool = True) -> dict[str, Any]:
        """Fetch one article.

        Returns a dict with the keys ``title``, ``pageid``, ``wikidata_id``,
        ``extract``, ``url``, ``categories``, ``redirects``, ``sections``,
        ``exists``.  Never raises: on failure ``exists`` is ``False``.
        """
        title = (title or "").strip()
        empty: dict[str, Any] = {
            "title": title,
            "pageid": 0,
            "wikidata_id": "",
            "extract": "",
            "url": "",
            "categories": [],
            "redirects": [],
            "sections": [],
            "exists": False,
        }
        if not title:
            return empty
        if use_cache:
            cached = self._cache_load(title)
            if cached is not None:
                return cached

        params = {
            "action": "query",
            "format": "json",
            "formatversion": "1",
            "titles": title,
            "prop": "extracts|pageprops|categories|redirects",
            "explaintext": "1",
            "exsectionformat": "wiki",
            "cllimit": "max",
            "rdlimit": "max",
            "redirects": "1",
        }
        data = self._request(params)
        pages = (data.get("query") or {}).get("pages") or {}
        if not pages:
            return empty
        page = next(iter(pages.values()))
        resolved_title = page.get("title", title)
        record = self._page_to_record(page, resolved_title)
        if not record["exists"]:
            logger.warning("Wikipedia article not found: %s", title)
            if use_cache:
                self._cache_save(title, {**empty, "title": title})
            return {**empty, "title": title}
        if use_cache:
            self._cache_save(title, record)
            self._cache_save(resolved_title, record)
        return record

    def fetch_pages(
        self, titles: Iterable[str], use_cache: bool = True
    ) -> list[dict[str, Any]]:
        """Fetch several articles, batching up to 20 titles per API request.

        Batching is much cheaper than one request per article and keeps the
        request rate low enough to avoid HTTP 429 (see docs/01 §4).
        """
        wanted = [t.strip() for t in titles if t and t.strip()]
        results: dict[str, dict[str, Any]] = {}
        pending: list[str] = []
        for title in wanted:
            cached = self._cache_load(title) if use_cache else None
            if cached is not None:
                results[title] = cached
            else:
                pending.append(title)

        for i in range(0, len(pending), 20):
            batch = pending[i : i + 20]
            pages = self._fetch_batch(batch)
            for title in batch:
                page = pages.get(title)
                if page is None:
                    page = self.fetch_page(title, use_cache=use_cache)
                results[title] = page
        return [results[t] for t in wanted if t in results]

    def _fetch_batch(self, titles: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch up to 20 articles in one request; returns ``{title: page}``.

        Uses ``prop=revisions`` (wikitext) because the ``extracts`` extension
        only ever returns the extract of a *single* page per request.
        """
        out: dict[str, dict[str, Any]] = {}
        if not titles:
            return out
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "1",
            "titles": "|".join(titles),
            "prop": "revisions|pageprops|categories|redirects",
            "rvprop": "content",
            "rvslots": "main",
            "cllimit": "max",
            "rdlimit": "max",
            "redirects": "1",
        }
        data = self._request(params)
        query = data.get("query") or {}
        if not query.get("pages"):
            # The whole request failed (e.g. 429) -- fall back to one-by-one.
            logger.warning("Wikipedia batch request failed, falling back to single requests")
            return {}
        # ``redirects``/``normalized`` map the requested title to the real one.
        alias: dict[str, str] = {}
        for key in ("normalized", "redirects"):
            for item in query.get(key) or []:
                frm = item.get("from")
                to = item.get("to")
                if frm and to:
                    alias[frm] = to
        for page in (query.get("pages") or {}).values():
            resolved = page.get("title", "")
            record = self._page_to_record(page, resolved)
            if not record["exists"]:
                logger.warning("Wikipedia article not found: %s", resolved)
            else:
                self._cache_save(resolved, record)
            # Map the requested (possibly redirected) title onto the record.
            for req in titles:
                if req == resolved or alias.get(req) == resolved:
                    out[req] = record
        for req in titles:
            out.setdefault(req, self._empty_record(req))
        return out

    @staticmethod
    def _empty_record(title: str) -> dict[str, Any]:
        return {
            "title": title,
            "pageid": 0,
            "wikidata_id": "",
            "extract": "",
            "url": "",
            "categories": [],
            "redirects": [],
            "sections": [],
            "exists": False,
        }

    def _page_to_record(self, page: dict[str, Any], title: str) -> dict[str, Any]:
        if page.get("missing") or page.get("invalid") or not page.get("pageid"):
            return self._empty_record(title)
        resolved_title = page.get("title", title)
        pageid = int(page.get("pageid") or 0)
        pageprops = page.get("pageprops") or {}
        wikidata_id = pageprops.get("wikibase_item", "") or ""
        categories = [c.get("title", "") for c in (page.get("categories") or []) if c.get("title")]
        redirects = [r.get("title", "") for r in (page.get("redirects") or []) if r.get("title")]

        raw_extract = page.get("extract") or ""
        sections: list[dict[str, str]] = []
        extract = ""
        if raw_extract:
            # ``prop=extracts`` keeps the "== Heading ==" lines -> parse them.
            sections = _parse_wiki_sections(raw_extract)
            extract = "\n\n".join(s["text"] for s in sections if s["text"]).strip()
            if not extract:
                extract = _strip_markup(raw_extract)
        else:
            # ``prop=revisions`` -> wikitext; strip the markup ourselves.
            wikitext = self._wikitext_of(page)
            if wikitext:
                sections = _parse_wiki_sections(wikitext)
                for sec in sections:
                    sec["text"] = _strip_markup(sec["text"])
                extract = "\n\n".join(s["text"] for s in sections if s["text"]).strip()
        return {
            "title": resolved_title,
            "pageid": pageid,
            "wikidata_id": wikidata_id,
            "extract": extract,
            "url": f"https://{self.lang}.wikipedia.org/wiki/{title_slug(resolved_title)}",
            "categories": categories,
            "redirects": redirects,
            "sections": sections,
            "exists": True,
        }

    @staticmethod
    def _wikitext_of(page: dict[str, Any]) -> str:
        """Extract the wikitext body of a page from ``prop=revisions``."""
        revisions = page.get("revisions") or []
        if not revisions:
            return ""
        rev = revisions[0]
        slots = rev.get("slots") or {}
        main = slots.get("main") or {}
        # formatversion=1 puts the body under "*", formatversion=2 under "content".
        content = main.get("content") or main.get("*")
        if content:
            return str(content)
        return str(rev.get("content") or rev.get("*") or "")

    def fetch_article(self, title: str, use_cache: bool = True) -> dict[str, Any]:
        """Alias of :meth:`fetch_page` kept for API compatibility."""
        return self.fetch_page(title, use_cache=use_cache)

    def search_titles(self, query: str, limit: int = 10) -> list[str]:
        """Search article titles (MediaWiki ``list=search``)."""
        query = (query or "").strip()
        if not query:
            return []
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srnamespace": "0",
            "srlimit": max(1, min(int(limit), 50)),
        }
        data = self._request(params)
        hits = ((data.get("query") or {}).get("search")) or []
        return [h.get("title", "") for h in hits if h.get("title")]

    def outgoing_links(self, title: str, limit: int = 500) -> list[str]:
        """Titles of the internal links of an article (for graph expansion)."""
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "links",
            "plnamespace": "0",
            "pllimit": max(1, min(int(limit), 500)),
        }
        data = self._request(params)
        pages = (data.get("query") or {}).get("pages") or {}
        if not pages:
            return []
        page = next(iter(pages.values()))
        links = [
            link.get("title", "") for link in (page.get("links") or []) if link.get("title")
        ]
        return [t for t in links if not t.startswith(_SKIP_LINK_PREFIXES)]

    def expand_from_seeds(
        self,
        seed_titles: Iterable[str] | None = None,
        max_articles: int = 300,
        rounds: int = 1,
        min_chars: int = 400,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Breadth-first expansion from seed articles.

        Returns the list of fetched articles (each a dict as produced by
        :meth:`fetch_page`) that pass the scope filter, capped at
        ``max_articles``.  Seed articles are always included first.
        """
        seeds = list(seed_titles) if seed_titles is not None else list(DEFAULT_SEED_TITLES)
        seen: set[str] = set()
        articles: list[dict[str, Any]] = []
        frontier: list[str] = list(seeds)

        for round_idx in range(max(1, int(rounds) + 1)):
            if not frontier or len(articles) >= max_articles:
                break
            # Batched fetch: 20 articles per API request.
            budget = max_articles - len(articles)
            batch = [t for t in frontier if t not in seen][: max(budget, 1)]
            seen.update(batch)
            pages = self.fetch_pages(batch, use_cache=use_cache)
            next_frontier: list[str] = []
            for title, page in zip(batch, pages, strict=False):
                if len(articles) >= max_articles:
                    break
                if not page.get("exists"):
                    continue
                if round_idx == 0:
                    # Seeds are trusted; keep them even if short.
                    articles.append(page)
                    continue
                if len(page.get("extract") or "") < min_chars:
                    continue
                if not is_in_history_scope(page):
                    continue
                articles.append(page)
                if round_idx < rounds:
                    next_frontier.extend(self.outgoing_links(title))
            frontier = [t for t in next_frontier if t not in seen]

        return articles[:max_articles]


def is_in_history_scope(page: dict[str, Any]) -> bool:
    """Heuristic: does the article look like Vietnamese history?"""
    cats = " | ".join(page.get("categories") or []).lower()
    for hint in _HISTORY_CATEGORY_HINTS:
        if hint in cats:
            return True
    title = (page.get("title") or "").lower()
    if title.startswith(_SKIP_LINK_PREFIXES):
        return False
    extract = (page.get("extract") or "")[:4000].lower()
    markers = (
        "việt nam",
        "đại việt",
        "nhà trần",
        "nhà lê",
        "nhà nguyễn",
        "nhà lý",
        "triều đại",
        "thế kỷ",
        "thành phố",
        "tỉnh",
    )
    return any(m in extract for m in markers)


# --------------------------------------------------------------------------
# module-level convenience wrappers (default client)
# --------------------------------------------------------------------------
_DEFAULT_CLIENT: WikipediaClient | None = None


def get_client(**kwargs) -> WikipediaClient:
    """Return a lazily-created default client."""
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None or kwargs:
        _DEFAULT_CLIENT = WikipediaClient(**kwargs)
    return _DEFAULT_CLIENT


def fetch_page(title: str, **kwargs) -> dict[str, Any]:
    return get_client().fetch_page(title, **kwargs)


def fetch_article(title: str, **kwargs) -> dict[str, Any]:
    return get_client().fetch_article(title, **kwargs)


def search_titles(query: str, limit: int = 10, **kwargs) -> list[str]:
    return get_client().search_titles(query, limit=limit, **kwargs)


def expand_from_seeds(
    seed_titles: Iterable[str] | None = None,
    max_articles: int = 300,
    rounds: int = 1,
    **kwargs,
) -> list[dict[str, Any]]:
    return get_client().expand_from_seeds(
        seed_titles=seed_titles, max_articles=max_articles, rounds=rounds, **kwargs
    )
