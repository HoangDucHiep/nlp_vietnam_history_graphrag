"""GraphRAG Lịch sử Việt Nam -- giao diện Streamlit (5 tab).

Thiết kế
--------
* **Không side-effect khi import**: module này chỉ định nghĩa hàm; mọi lệnh
  ``st.*`` nằm trong :func:`main` (được ``streamlit run`` gọi khi
  ``__name__ == "__main__"``).  Vì vậy ``import hgrag.app.streamlit_app``
  không gọi ``st.set_page_config`` và có thể chạy trong test.
* **Lazy/adapter cho QA pipeline**: vòng lặp QA (``hgrag.qa.pipeline``) chưa
  tồn tại ở Wave này.  Lớp :class:`QAService` import pipeline *bên trong
  hàm* và báo "pipeline chưa sẵn sàng" nếu thiếu -- app không bao giờ crash.
* **Mọi I/O bọc try/except**: thiếu Neo4j/Qdrant chỉ khiến app hiển thị
  thông báo, không làm sập trang.

Các hàm thuần tuý (không gọi ``st.*``) nằm ở nửa trên file để test được:
:func:`format_citations`, :func:`parse_reasoning`,
:func:`build_knowledge_graph_table`, :func:`load_results_table`,
:func:`subgraph_from_neighbors`, :func:`build_results_chart_data`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

PIPELINE_NOT_READY = "pipeline chưa sẵn sàng"

CITATION_PATTERN = re.compile(r"\[E(\d+)\]")
REASONING_HEADERS = ("chuỗi suy luận", "chuoi suy luan", "reasoning", "chuỗi lập luận")
NUMBERED_PATTERN = re.compile(r"^(\d+)[.)]\s+(.*)$")

SYSTEM_VARIANTS = ("B0", "B1", "B2", "Ours")
SYSTEM_COLUMN_CANDIDATES = ("system", "variant", "method", "config", "baseline")


# ==========================================================================
# Hàm thuần tuý (test được, không phụ thuộc streamlit runtime)
# ==========================================================================


def extract_citation_markers(text: str) -> list[str]:
    """Trả về danh sách marker ``E1, E2, ...`` theo thứ tự xuất hiện (không lặp)."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for num in CITATION_PATTERN.findall(text):
        marker = f"E{num}"
        if marker not in seen:
            seen.add(marker)
            out.append(marker)
    return out


def _evidence_id(evidence: dict[str, Any]) -> str:
    raw = evidence.get("evidence_id") or evidence.get("id") or evidence.get("marker") or ""
    return str(raw).strip().upper().replace("[", "").replace("]", "")


def format_citations(answer: str, evidence: list[dict[str, Any]] | None = None) -> str:
    """Ghép câu trả lời với danh sách nguồn trích dẫn tương ứng.

    Evidence không có trong danh sách vẫn được hiển thị (đánh dấu thiếu) để
    phát hiện citation sai -- đây là bước post-processing quan trọng của
    docs/03 §7.
    """
    answer = answer or ""
    evidence = evidence or []
    markers = extract_citation_markers(answer)
    by_id: dict[str, dict[str, Any]] = {}
    for item in evidence:
        eid = _evidence_id(item)
        if eid:
            by_id[eid] = item

    lines: list[str] = []
    for marker in markers:
        item = by_id.get(marker)
        if item is None:
            lines.append(f"**[{marker}]** — ⚠️ không tìm thấy evidence tương ứng")
            continue
        kind = str(item.get("kind") or item.get("type") or "evidence")
        text = str(item.get("text") or "").strip()
        srcs = item.get("source_chunk_ids") or []
        src_txt = f" — nguồn: {', '.join(str(s) for s in srcs)}" if srcs else ""
        lines.append(f"**[{marker}]** ({kind}{src_txt}): {text}")

    if not lines:
        return answer + "\n\n> ⚠️ Câu trả lời không có citation nào (grounded = False)."
    return answer + "\n\n**Nguồn trích dẫn**\n\n" + "\n\n".join(lines)


def _strip_bullet(line: str) -> str:
    """Bỏ ký hiệu đầu dòng (``-``, ``*``, ``•``) nếu có."""
    return line.lstrip("-*• \t").strip()


def parse_reasoning(text: str) -> list[str]:
    """Tách *chuỗi suy luận* thành danh sách bước.

    Nhận diện theo tiêu đề ("Chuỗi suy luận:") nếu có -- khi đó lấy mọi dòng
    sau tiêu đề.  Nếu không có tiêu đề thì chỉ lấy các dòng đánh số
    ``1.`` / ``1)`` để không nuốt nhầm văn bản thường.
    """
    lines = [(ln or "").strip() for ln in (text or "").splitlines()]
    header_idx: int | None = None
    for i, line in enumerate(lines):
        low = line.lower()
        if any(h in low for h in REASONING_HEADERS):
            header_idx = i
            break

    steps: list[str] = []
    if header_idx is not None:
        for line in lines[header_idx + 1 :]:
            if not line:
                continue
            if line.startswith("#"):
                break
            match = NUMBERED_PATTERN.match(line)
            steps.append(match.group(2).strip() if match else _strip_bullet(line))
    else:
        for line in lines:
            match = NUMBERED_PATTERN.match(line)
            if match:
                steps.append(match.group(2).strip())
    return [s for s in steps if s]


def normalize_link_records(raw: Any) -> list[dict[str, Any]]:
    """Chuẩn hoá output entity-linking của pipeline về list[dict]."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("links") or raw.get("entities") or raw.get("items") or []
    out: list[dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, dict):
            out.append(dict(item))
        elif is_dataclass(item):
            out.append(asdict(item))
        else:
            out.append({"name": str(item)})
    return out


def build_knowledge_graph_table(links: Any) -> list[dict[str, Any]]:
    """Bảng "entity linking": tên → QID → điểm tin cậy → ứng viên."""
    rows: list[dict[str, Any]] = []
    for item in normalize_link_records(links):
        candidates = item.get("candidates") or []
        names: list[str] = []
        for cand in candidates:
            if isinstance(cand, dict):
                names.append(str(cand.get("name") or cand.get("uid") or ""))
            else:
                names.append(str(cand))
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        rows.append(
            {
                "mention": str(item.get("mention") or item.get("name") or item.get("query") or ""),
                "qid": str(item.get("wikidata_id") or item.get("qid") or ""),
                "uid": str(item.get("uid") or ""),
                "entity_type": str(item.get("entity_type") or item.get("type") or ""),
                "confidence": confidence,
                "candidates": ", ".join(n for n in names if n),
                "n_candidates": len(candidates),
            }
        )
    return rows


def _node_label(props: dict[str, Any]) -> str:
    return str(props.get("name") or props.get("uid") or "?")


def subgraph_from_neighbors(neighbors: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Chuyển danh sách quan hệ lấy từ graph store thành payload node/edge."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for item in neighbors or []:
        src = item.get("src") or {}
        dst = item.get("dst") or {}
        rel = item.get("rel") or item.get("rel_type") or "REL"
        src_id = str(src.get("uid") or "")
        dst_id = str(dst.get("uid") or "")
        if not src_id or not dst_id:
            continue
        for uid, props in ((src_id, src), (dst_id, dst)):
            if uid not in nodes:
                nodes[uid] = {
                    "id": uid,
                    "label": _node_label(props),
                    "group": str(props.get("entity_type") or "Entity"),
                    "title": str(props.get("description") or ""),
                }
        edges.append(
            {
                "from": src_id,
                "to": dst_id,
                "label": str(rel),
                "title": str((item.get("rel_props") or {}).get("text") or ""),
            }
        )
    return {"nodes": list(nodes.values()), "edges": edges}


def load_results_table(path: str | Path | None) -> pd.DataFrame:
    """Đọc ``reports/results.csv`` (nếu có) thành DataFrame; không có -> rỗng."""
    if path is None:
        return pd.DataFrame()
    file = Path(path)
    if not file.exists() or not file.is_file():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(file)
    except Exception as exc:  # noqa: BLE001 - file hỏng không được làm sập app
        logger.warning("Không đọc được %s: %s", file, exc)
        return pd.DataFrame()
    return frame


def _system_column(frame: pd.DataFrame) -> str | None:
    for col in SYSTEM_COLUMN_CANDIDATES:
        if col in frame.columns:
            return col
    return None


def build_results_chart_data(
    frame: pd.DataFrame, metric: str | None = None
) -> dict[str, list[Any]]:
    """Chuẩn bị dữ liệu biểu đồ so sánh B0/B1/B2/Ours từ DataFrame kết quả."""
    if frame is None or frame.empty:
        return {"labels": [], "values": [], "metric": ""}
    sys_col = _system_column(frame)
    if sys_col is None:
        return {"labels": [], "values": [], "metric": ""}

    numeric_cols = [
        c
        for c in frame.columns
        if c != sys_col and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if metric is None or metric not in frame.columns:
        if not numeric_cols:
            return {"labels": [], "values": [], "metric": ""}
        metric = numeric_cols[0]

    grouped = frame.groupby(sys_col)[metric].mean()
    order = [s for s in SYSTEM_VARIANTS if s in grouped.index]
    labels = order + [str(i) for i in grouped.index if str(i) not in order]
    values = [float(grouped[i]) for i in labels]
    return {"labels": labels, "values": values, "metric": str(metric)}


def summarize_confidence(confidence: float) -> str:
    """Diễn giải điểm tin cậy entity linking."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "không rõ"
    if value >= 0.9:
        return "cao"
    if value >= 0.7:
        return "trung bình"
    if value > 0:
        return "thấp"
    return "không rõ"


# ==========================================================================
# Lớp dịch vụ (adapter cho pipeline chưa có)
# ==========================================================================


class QAService:
    """Bọc pipeline hỏi đáp (Wave 4) theo kiểu adapter -- dễ thay thế.

    Pipeline được import *bên trong* :meth:`_ensure_pipeline`.  Nếu module
    ``hgrag.qa.pipeline`` chưa tồn tại, service vẫn khởi tạo được và trả về
    thông báo "pipeline chưa sẵn sàng" thay vì ném exception.
    """

    def __init__(
        self,
        settings: Any | None = None,
        llm: Any | None = None,
        graph_store: Any | None = None,
        vector_store: Any | None = None,
        mode: str = "auto",
    ):
        self.settings = settings
        self.llm = llm
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.mode = mode
        self._pipeline: Any | None = None
        self._error: str = ""
        self._loaded = False

    # -- trạng thái -----------------------------------------------------
    @property
    def error(self) -> str:
        return self._error

    @property
    def ready(self) -> bool:
        self._ensure_pipeline()
        return self._pipeline is not None

    def _ensure_pipeline(self) -> Any | None:
        if self._loaded:
            return self._pipeline
        self._loaded = True
        try:
            from hgrag.qa import pipeline as qa_pipeline  # noqa: PLC0415
        except ImportError as exc:
            self._error = f"{PIPELINE_NOT_READY} ({exc})"
            logger.info("QA pipeline chưa sẵn sàng: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - lỗi import lạ cũng không được crash
            self._error = f"{PIPELINE_NOT_READY} ({exc})"
            return None

        factory = getattr(qa_pipeline, "build_pipeline", None)
        pipeline: Any | None = None
        try:
            if callable(factory):
                pipeline = factory(self.settings)
            else:
                cls = getattr(qa_pipeline, "QAPipeline", None) or getattr(
                    qa_pipeline, "QAPipelineRunner", None
                )
                if cls is not None:
                    pipeline = self._instantiate(cls)
        except Exception as exc:  # noqa: BLE001
            self._error = f"{PIPELINE_NOT_READY} ({exc})"
            logger.warning("Không khởi tạo được QA pipeline: %s", exc)
            return None

        if pipeline is None:
            self._error = f"{PIPELINE_NOT_READY} (không tìm thấy entrypoint trong hgrag.qa.pipeline)"
            return None
        self._pipeline = pipeline
        self._error = ""
        return pipeline

    def _instantiate(self, cls: Any) -> Any:
        """Thử lần lượt các tổ hợp tham số phổ biến, bỏ qua tham số không nhận."""
        candidates = [
            {"settings": self.settings, "llm": self.llm, "graph_store": self.graph_store},
            {"settings": self.settings, "llm": self.llm},
            {"settings": self.settings},
            {},
        ]
        for kwargs in candidates:
            try:
                return cls(**kwargs)
            except TypeError:
                continue
        return cls()

    # -- truy vấn --------------------------------------------------------
    def ask(self, question: str, hops: int = 2, token_budget: int = 6000) -> dict[str, Any]:
        """Hỏi câu hỏi; luôn trả về dict (không bao giờ ném exception)."""
        question = (question or "").strip()
        if not question:
            return {"ok": False, "message": "Câu hỏi rỗng.", "answer": "", "evidence": []}

        pipeline = self._ensure_pipeline()
        if pipeline is None:
            return {
                "ok": False,
                "message": self._error or PIPELINE_NOT_READY,
                "answer": "",
                "evidence": [],
                "reasoning": [],
                "subgraph": {"nodes": [], "edges": []},
            }

        try:
            raw = self._call_pipeline(pipeline, question, hops, token_budget)
        except Exception as exc:  # noqa: BLE001 - lỗi pipeline không được làm sập app
            logger.exception("QA pipeline lỗi")
            return {"ok": False, "message": f"Lỗi khi chạy pipeline: {exc}", "answer": ""}

        return self._normalize_result(raw, question)

    def _call_pipeline(self, pipeline: Any, question: str, hops: int, token_budget: int) -> Any:
        for name in ("answer", "ask", "run", "answer_question", "query"):
            fn = getattr(pipeline, name, None)
            if not callable(fn):
                continue
            for kwargs in (
                {"question": question, "hops": hops, "token_budget": token_budget},
                {"question": question, "max_hops": hops, "max_context_tokens": token_budget},
                {"question": question},
            ):
                try:
                    return fn(**kwargs)
                except TypeError:
                    continue
            try:
                return fn(question)
            except TypeError:
                continue
        raise AttributeError("QA pipeline không có phương thức trả lời nào hợp lệ")

    @staticmethod
    def _normalize_result(raw: Any, question: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if isinstance(raw, dict):
            payload = dict(raw)
        elif is_dataclass(raw):
            payload = asdict(raw)
        elif hasattr(raw, "to_dict") and callable(raw.to_dict):
            payload = dict(raw.to_dict())
        else:
            payload = {"answer": str(raw)}

        answer = str(
            payload.get("answer")
            or payload.get("response")
            or payload.get("text")
            or payload.get("final_answer")
            or ""
        )
        reasoning_raw = payload.get("reasoning") or payload.get("reasoning_chain") or ""
        reasoning = reasoning_raw if isinstance(reasoning_raw, list) else parse_reasoning(
            str(reasoning_raw)
        )

        evidence: list[dict[str, Any]] = []
        for item in payload.get("evidence") or payload.get("evidences") or []:
            if isinstance(item, dict):
                evidence.append(dict(item))
            elif is_dataclass(item):
                evidence.append(asdict(item))
            elif hasattr(item, "to_dict") and callable(item.to_dict):
                evidence.append(dict(item.to_dict()))
            else:
                evidence.append({"evidence_id": "", "text": str(item), "kind": "evidence"})

        subgraph = payload.get("subgraph") or {}
        if not isinstance(subgraph, dict):
            subgraph = {}

        return {
            "ok": True,
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "reasoning": [str(r) for r in reasoning],
            "subgraph": subgraph,
            "grounded": bool(payload.get("grounded", bool(extract_citation_markers(answer)))),
            "intent": str(payload.get("intent") or ""),
            "message": "",
            "raw": payload,
        }


# ==========================================================================
# Hỗ trợ môi trường / DB
# ==========================================================================


class GraphReadAdapter:
    """Đọc KG an toàn, không phụ thuộc cách store quản lý session.

    ``Neo4jGraphStore`` (kg/store.py) trả về ``Result`` lazy *ra khỏi* block
    ``with session`` nên đọc sẽ ném ``ResultConsumedError``.  Adapter này tự
    chạy query và consume kết quả trong cùng scope, đồng thời fallback sang
    InMemory store nếu Neo4j không dùng được.
    """

    def __init__(self, store: Any):
        self.store = store
        self.backend = getattr(store, "backend", "unknown")

    # -- generic ---------------------------------------------------------
    def _driver(self) -> Any:
        return getattr(self.store, "_driver", None)

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        driver = self._driver()
        if driver is None:
            return []
        database = getattr(self.store, "database", "neo4j")
        with driver.session(database=database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    # -- reads -----------------------------------------------------------
    def find_entity_by_name(self, name: str) -> list[dict[str, Any]]:
        try:
            rows = self._run(
                """
                MATCH (e:Entity)
                WHERE e.name = $name OR $name IN e.aliases OR e.name_key = $name
                RETURN e{.*} AS props LIMIT 20
                """,
                name=name,
            )
            return [dict(r["props"]) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j find_entity_by_name lỗi (%s) -> fallback", exc)
            return []

    def neighbors(self, uid: str, rel_types: list[str] | None = None) -> list[dict[str, Any]]:
        rel_filter = ""
        if rel_types:
            safe = [t for t in rel_types if t.replace("_", "").isalnum()]
            if safe:
                rel_filter = ":" + "|".join(safe)
        try:
            rows = self._run(
                f"""
                MATCH (a:Entity {{uid: $uid}})-[r{rel_filter}]-(b:Entity)
                RETURN a{{.*}} AS src, type(r) AS rel, properties(r) AS rprops, b{{.*}} AS dst
                LIMIT 500
                """,
                uid=uid,
            )
            return [
                {
                    "src": dict(r["src"]),
                    "rel": r["rel"],
                    "rel_props": dict(r["rprops"]),
                    "dst": dict(r["dst"]),
                }
                for r in rows
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j neighbors lỗi (%s) -> fallback", exc)
            return []

    def stats(self) -> dict[str, Any]:
        try:
            rows = self._run(
                """
                MATCH (e:Entity)
                WITH count(e) AS nodes, collect(DISTINCT e.entity_type) AS types
                MATCH ()-[r]->()
                WITH nodes, types, count(r) AS rels
                RETURN nodes, types, rels
                """
            )
            if not rows:
                return {"nodes": 0, "relations": 0, "entity_types": []}
            rec = rows[0]
            return {
                "nodes": int(rec["nodes"]),
                "relations": int(rec["rels"]),
                "entity_types": sorted(t for t in rec["types"] if t),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j stats lỗi (%s)", exc)
            return {"nodes": 0, "relations": 0, "entity_types": []}


def build_graph_store_for_app(settings: Any) -> Any:
    """Dựng graph store: Neo4j nếu kết nối được, không thì InMemory fallback.

    Không dùng ``hgrag.retrieval.vector_store.build_graph_store`` vì hàm đó
    import sai module (``hgrag.retrieval.kg``) nên luôn rơi về fallback.
    """
    try:
        from hgrag.kg.store import Neo4jGraphStore

        nc = settings.neo4j
        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=nc.get("user", "neo4j"),
            password=settings.neo4j_password,
            database=nc.get("database", "neo4j"),
            batch_size=nc.get("batch_size", 500),
        )
        store.backend = "neo4j"
        return GraphReadAdapter(store)
    except Exception as exc:  # noqa: BLE001 - Neo4j chưa chạy -> fallback
        logger.warning("Không kết nối được Neo4j (%s) -> dùng InMemory fallback", exc)
        from hgrag.kg.store import InMemoryGraphStore

        store = InMemoryGraphStore()
        store.backend = "memory"  # type: ignore[attr-defined]
        return store


def safe_stats(store: Any) -> dict[str, Any]:
    """``store.stats()`` an toàn (Neo4j rỗng có thể ném lỗi)."""
    if store is None:
        return {"nodes": 0, "relations": 0, "entity_types": []}
    try:
        return dict(store.stats())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không lấy được stats: %s", exc)
        return {"nodes": 0, "relations": 0, "entity_types": []}


def build_client_for_mode(settings: Any, mode: str) -> Any:
    """Chọn LLM client theo chế độ người dùng chọn ở sidebar."""
    from hgrag.llm.client import MockLLMClient, build_llm_client

    mode = (mode or "auto").lower()
    if mode == "mock":
        return MockLLMClient(embed_dim=getattr(settings, "embed_dim", 768))
    if mode == "gemini":
        if not getattr(settings, "has_gemini_key", False):
            st.warning("Chế độ Gemini được chọn nhưng chưa có GEMINI_API_KEY -- dùng mock.")
            return MockLLMClient(embed_dim=getattr(settings, "embed_dim", 768))
        try:
            from hgrag.llm.client import GeminiClient

            llm_cfg = settings.llm
            return GeminiClient(
                model=settings.llm_model,
                api_key=settings.gemini_api_key,
                embed_model=settings.embed_model,
                temperature=llm_cfg.get("temperature", 0.0),
                max_output_tokens=llm_cfg.get("max_output_tokens", 2048),
                cache_dir=settings.cache_dir,
                cache_enabled=llm_cfg.get("cache_enabled", True),
                output_dim=settings.embed_dim,
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Không khởi tạo được GeminiClient ({exc}) -- dùng mock.")
            return MockLLMClient(embed_dim=getattr(settings, "embed_dim", 768))
    return build_llm_client(settings)


def check_neo4j(settings: Any) -> dict[str, Any]:
    """Kiểm tra kết nối Neo4j mà không crash."""
    try:
        from neo4j import GraphDatabase  # noqa: PLC0415

        driver = GraphDatabase.driver(settings.neo4j_uri, auth=("neo4j", settings.neo4j_password))
        driver.verify_connectivity()
        driver.close()
        return {"ok": True, "detail": settings.neo4j_uri}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{settings.neo4j_uri} -- {exc}"}


def check_qdrant(settings: Any) -> dict[str, Any]:
    """Kiểm tra kết nối Qdrant mà không crash."""
    try:
        import requests  # noqa: PLC0415

        resp = requests.get(settings.qdrant_url, timeout=3)
        return {"ok": resp.status_code < 500, "detail": f"{settings.qdrant_url} ({resp.status_code})"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": f"{settings.qdrant_url} -- {exc}"}


def cache_info(settings: Any) -> dict[str, Any]:
    """Thống kê cache LLM trên đĩa."""
    cache_dir = Path(settings.cache_dir)
    n_files = 0
    size = 0
    if cache_dir.exists():
        for f in cache_dir.rglob("*.json"):
            n_files += 1
            try:
                size += f.stat().st_size
            except OSError:
                pass
    return {"path": str(cache_dir), "n_files": n_files, "size_bytes": size}


def find_ingest_logs(settings: Any) -> list[Path]:
    """Tìm các file log ingest trong artifacts/ và reports/."""
    roots = [Path(settings.artifacts), Path(settings.reports)]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.log", "logs/*.log", "**/*ingest*.log"):
            found.extend(p for p in root.glob(pattern) if p.is_file())
    return sorted(set(found))


# ==========================================================================
# Render các tab
# ==========================================================================


def _sidebar(settings: Any) -> dict[str, Any]:
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        mode = st.selectbox(
            "Chế độ LLM", ["Auto", "Gemini", "Mock"], index=0, help="Auto: dùng key nếu có, không thì mock."
        )
        hops = st.slider("Số hop đồ thị", min_value=1, max_value=3, value=2)
        token_budget = st.slider(
            "Ngân sách token context", min_value=1000, max_value=12000, value=6000, step=500
        )
        st.divider()
        st.subheader("Trạng thái dịch vụ")
        neo4j = check_neo4j(settings)
        qdrant = check_qdrant(settings)
        st.write("Neo4j:", "🟢" if neo4j["ok"] else "🔴")
        st.write("Qdrant:", "🟢" if qdrant["ok"] else "🔴")
        st.caption(f"LLM provider: `{settings.llm_provider}`")
        st.caption(f"Model: `{settings.llm_model}`")
        st.caption(f"Cache: `{settings.cache_dir}`")
    return {
        "mode": mode.lower(),
        "hops": hops,
        "token_budget": token_budget,
        "neo4j": neo4j,
        "qdrant": qdrant,
    }


def _get_service(settings: Any, mode: str) -> QAService:
    """Lấy (hoặc tạo) QAService trong session_state."""
    key = f"qa_service::{mode}"
    service = st.session_state.get(key)
    if service is None:
        llm = build_client_for_mode(settings, mode)
        graph_store = None
        vector_store = None
        try:
            from hgrag.retrieval.vector_store import build_vector_store

            graph_store = build_graph_store_for_app(settings)
            vector_store = build_vector_store(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không dựng được store: %s", exc)
        service = QAService(
            settings=settings,
            llm=llm,
            graph_store=graph_store,
            vector_store=vector_store,
            mode=mode,
        )
        st.session_state[key] = service
    return service


def render_tab_ask(settings: Any, service: QAService, hops: int, token_budget: int) -> None:
    """Tab 1 -- Hỏi đáp."""
    st.subheader("💬 Hỏi đáp lịch sử Việt Nam")
    question = st.text_input(
        "Câu hỏi (tiếng Việt)",
        placeholder="Ví dụ: Vì sao nhà Trần phải gả Hoàng thái chúa Trần Khánh Dư cho vua Đinh?",
    )
    if st.button("Trả lời", type="primary"):
        if not question.strip():
            st.warning("Hãy nhập câu hỏi trước khi bấm 'Trả lời'.")
        else:
            with st.spinner("Đang truy vấn đồ thị và sinh câu trả lời..."):
                result = service.ask(question, hops=hops, token_budget=token_budget)

            st.session_state["last_result"] = result
            if not result.get("ok"):
                st.warning(result.get("message") or PIPELINE_NOT_READY)
                st.info(
                    "Hệ thống hỏi đáp đầy đủ sẽ có ở Wave 4 (`hgrag.qa.pipeline`). "
                    "Hiện tại các tab truy vết KG / đồ thị vẫn hoạt động nếu đã nạp KG."
                )
                return

            st.markdown("#### Câu trả lời")
            st.markdown(format_citations(result.get("answer", ""), result.get("evidence", [])))

            if result.get("grounded") is False:
                st.error("Câu trả lời không có căn cứ truy vết (grounded = False).")

            reasoning = result.get("reasoning") or []
            if reasoning:
                st.markdown("#### 🧠 Chuỗi suy luận")
                for i, step in enumerate(reasoning, start=1):
                    st.markdown(f"{i}. {step}")

            if result.get("evidence"):
                st.markdown("#### 📚 Evidence")
                st.dataframe(pd.DataFrame(result["evidence"]), use_container_width=True)

            st.download_button(
                "⬇️ Tải kết quả JSON",
                data=json.dumps(result.get("raw", result), ensure_ascii=False, indent=2),
                file_name="hgrag_answer.json",
                mime="application/json",
            )

    if "last_result" in st.session_state:
        prev = st.session_state["last_result"]
        if prev.get("ok"):
            with st.expander("Kết quả trước đó"):
                st.markdown(format_citations(prev.get("answer", ""), prev.get("evidence", [])))


def render_tab_trace(settings: Any, service: QAService) -> None:
    """Tab 2 -- Truy vết KG (entity linking + đường đi)."""
    st.subheader("🔍 Truy vết Knowledge Graph")
    name = st.text_input("Tên thực thể cần tra", placeholder="Ví dụ: Trần Hưng Đạo")
    if not st.button("Tra cứu"):
        return

    store = service.graph_store
    if store is None:
        st.info("Chưa có graph store khả dụng (Neo4j chưa chạy).")
        return

    stats = safe_stats(store)
    st.caption(
        f"KG hiện có: **{stats.get('nodes', 0)}** node / **{stats.get('relations', 0)}** quan hệ "
        f"({', '.join(stats.get('entity_types') or []) or 'chưa có loại thực thể'})"
    )

    try:
        candidates = store.find_entity_by_name(name.strip())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không tra cứu được KG: {exc}")
        return

    if not candidates:
        st.warning(f"Không tìm thấy thực thể nào khớp với '{name}'.")
        return

    links = [
        {
            "mention": name,
            "wikidata_id": c.get("wikidata_id", ""),
            "uid": c.get("uid", ""),
            "entity_type": c.get("entity_type", ""),
            "confidence": float(c.get("confidence") or 0.0),
            "candidates": [{"name": c.get("name", ""), "uid": c.get("uid", "")}],
        }
        for c in candidates
    ]
    st.markdown("**Entity linking**")
    st.dataframe(pd.DataFrame(build_knowledge_graph_table(links)), use_container_width=True)

    for cand in candidates[:5]:
        conf = float(cand.get("confidence") or 0.0)
        st.markdown(
            f"- **{cand.get('name', '?')}** — QID `{cand.get('wikidata_id') or '-'}` — "
            f"điểm tin cậy {conf:.2f} ({summarize_confidence(conf)})"
        )

    st.markdown("**Đường đi (path)**")
    for cand in candidates[:3]:
        uid = str(cand.get("uid") or "")
        try:
            neighbors = store.neighbors(uid)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Không lấy được neighbors: {exc}")
            continue
        if not neighbors:
            st.write(f"- `{cand.get('name')}`: không có quan hệ nào.")
            continue
        rows = []
        for nb in neighbors:
            src = nb.get("src") or {}
            dst = nb.get("dst") or {}
            rows.append(
                {
                    "từ": src.get("name", src.get("uid", "")),
                    "quan hệ": nb.get("rel", ""),
                    "đến": dst.get("name", dst.get("uid", "")),
                    "năm": (nb.get("rel_props") or {}).get("year", ""),
                    "confidence": (nb.get("rel_props") or {}).get("confidence", ""),
                    "mô tả": (nb.get("rel_props") or {}).get("text", ""),
                }
            )
        st.markdown(f"- **{cand.get('name', '?')}**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.session_state["last_neighbors"] = neighbors


def _render_subgraph(subgraph: dict[str, Any], key: str = "subgraph") -> None:
    """Vẽ subgraph bằng pyvis, fallback graphviz rồi bảng."""
    nodes = subgraph.get("nodes") or []
    edges = subgraph.get("edges") or []
    if not nodes and not edges:
        st.info("Không có subgraph để vẽ.")
        return

    try:
        from pyvis.network import Network  # noqa: PLC0415

        net = Network(height="600px", width="100%", notebook=False, directed=True)
        for node in nodes:
            net.add_node(
                node["id"],
                label=node.get("label", node["id"]),
                title=node.get("title", ""),
                group=node.get("group", ""),
            )
        for edge in edges:
            net.add_edge(edge["from"], edge["to"], label=edge.get("label", ""))
        html = net.generate_html(notebook=False)
        st.components.v1.html(html, height=620)
        st.download_button(
            "⬇️ Tải đồ thị HTML",
            data=html,
            file_name="hgrag_subgraph.html",
            mime="text/html",
            key=f"dl_{key}",
        )
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Không dùng được pyvis ({exc}) -- chuyển sang bảng.")

    try:
        import graphviz  # noqa: F401 - kiểm tra graphviz đã cài chưa

        dot = ["digraph G {", '  node [shape=box style=rounded];']
        for node in nodes:
            dot.append(f'  "{node["id"]}" [label="{node.get("label", node["id"])}"];')
        for edge in edges:
            dot.append(f'  "{edge["from"]}" -> "{edge["to"]}" [label="{edge.get("label", "")}"];')
        dot.append("}")
        st.graphviz_chart("\n".join(dot))
        return
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Không dùng được graphviz ({exc}) -- hiển thị dạng bảng.")

    st.dataframe(pd.DataFrame(nodes), use_container_width=True)
    st.dataframe(pd.DataFrame(edges), use_container_width=True)


def render_tab_graph(settings: Any, service: QAService) -> None:
    """Tab 3 -- Đồ thị."""
    st.subheader("🕸️ Đồ thị quan hệ")
    name = st.text_input("Nhập tên thực thể để vẽ subgraph", value="Trần Hưng Đạo")
    if st.button("Vẽ đồ thị"):
        store = service.graph_store
        if store is None:
            st.info("Chưa có graph store khả dụng (Neo4j chưa chạy).")
            return
        try:
            candidates = store.find_entity_by_name(name.strip())
            if not candidates:
                st.warning(f"Không tìm thấy thực thể nào khớp với '{name}'.")
                return
            uid = str(candidates[0].get("uid") or "")
            neighbors = store.neighbors(uid)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Không vẽ được đồ thị: {exc}")
            return
        st.session_state["last_subgraph"] = subgraph_from_neighbors(neighbors)

    subgraph = st.session_state.get("last_subgraph")
    if subgraph:
        _render_subgraph(subgraph, key="manual")
    else:
        st.info("Chưa có dữ liệu đồ thị. Hãy tra cứu một thực thể hoặc hỏi một câu hỏi ở tab đầu.")


def render_tab_eval(settings: Any) -> None:
    """Tab 4 -- Đánh giá."""
    st.subheader("📊 Đánh giá và so sánh")
    results_path = Path(settings.reports) / "results.csv"
    frame = load_results_table(results_path)
    if frame.empty:
        st.info(
            f"Chưa có `{results_path}`. Chạy `scripts/07_run_eval.py` để sinh kết quả "
            "(Baseline B0/B1/B2 + Ours + ablation)."
        )
        return

    st.dataframe(frame, use_container_width=True)

    chart_data = build_results_chart_data(frame)
    if chart_data["labels"]:
        st.markdown(f"**So sánh theo metric `{chart_data['metric']}`**")
        try:
            import plotly.express as px  # noqa: PLC0415

            fig = px.bar(
                x=chart_data["labels"],
                y=chart_data["values"],
                labels={"x": "Hệ thống", "y": chart_data["metric"]},
                title=f"So sánh {chart_data['metric']}",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Không vẽ được plotly ({exc}) -- dùng biểu đồ cột mặc định.")
            st.bar_chart(
                pd.DataFrame(
                    {"Hệ thống": chart_data["labels"], chart_data["metric"]: chart_data["values"]}
                ).set_index("Hệ thống")
            )
    else:
        st.caption("Không tìm thấy cột hệ thống / metric số để so sánh.")


def render_tab_system(settings: Any, service: QAService) -> None:
    """Tab 5 -- Hệ thống."""
    st.subheader("⚙️ Trạng thái hệ thống")

    st.markdown("**Dịch vụ**")
    neo4j = check_neo4j(settings)
    qdrant = check_qdrant(settings)
    kg_stats = safe_stats(service.graph_store) if service.graph_store is not None else {}
    st.dataframe(
        pd.DataFrame(
            [
                {"dịch vụ": "Neo4j", "trạng thái": "ok" if neo4j["ok"] else "lỗi", "chi tiết": neo4j["detail"]},
                {"dịch vụ": "Qdrant", "trạng thái": "ok" if qdrant["ok"] else "lỗi", "chi tiết": qdrant["detail"]},
                {
                    "dịch vụ": "LLM",
                    "trạng thái": "ok" if getattr(service.llm, "name", "") else "chưa có",
                    "chi tiết": str(getattr(service.llm, "name", "")) or "chưa khởi tạo",
                },
                {
                    "dịch vụ": "Knowledge Graph",
                    "trạng thái": "ok" if kg_stats.get("nodes", 0) else "trống",
                    "chi tiết": f"{kg_stats.get('nodes', 0)} node / {kg_stats.get('relations', 0)} quan hệ",
                },
            ]
        ),
        use_container_width=True,
    )

    st.markdown("**Cấu hình**")
    st.json(
        {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embed_model": settings.embed_model,
            "embed_provider": settings.embed_provider,
            "embed_dim": settings.embed_dim,
            "neo4j_uri": settings.neo4j_uri,
            "qdrant_url": settings.qdrant_url,
            "retrieval": settings.retrieval,
        }
    )

    st.markdown("**Cache**")
    st.json(cache_info(settings))

    st.markdown("**Log ingest**")
    logs = find_ingest_logs(settings)
    if not logs:
        st.caption("Không tìm thấy file log ingest nào.")
        return
    for log_file in logs:
        st.caption(str(log_file))
        try:
            text = Path(log_file).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            st.warning(f"Không đọc được {log_file}: {exc}")
            continue
        st.code(text[-20000:], language="text")
        st.download_button(
            "⬇️ Tải log",
            data=text,
            file_name=Path(log_file).name,
            mime="text/plain",
            key=f"dl_log_{Path(log_file).name}",
        )


# ==========================================================================
# Entry point
# ==========================================================================


def main() -> None:
    """Điểm vào của Streamlit (được ``streamlit run`` gọi)."""
    st.set_page_config(page_title="GraphRAG Lịch sử Việt Nam", layout="wide")

    from hgrag.config import get_settings

    settings = get_settings()
    try:
        settings.ensure_dirs()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không tạo được thư mục: %s", exc)

    st.title("📚 GraphRAG — Hỏi đáp Lịch sử Việt Nam (905–1975)")

    cfg = _sidebar(settings)
    service = _get_service(settings, cfg["mode"])

    tab_ask, tab_trace, tab_graph, tab_eval, tab_system = st.tabs(
        ["💬 Hỏi đáp", "🔍 Truy vết KG", "🕸️ Đồ thị", "📊 Đánh giá", "⚙️ Hệ thống"]
    )

    with tab_ask:
        render_tab_ask(settings, service, cfg["hops"], cfg["token_budget"])
    with tab_trace:
        render_tab_trace(settings, service)
    with tab_graph:
        render_tab_graph(settings, service)
    with tab_eval:
        render_tab_eval(settings)
    with tab_system:
        render_tab_system(settings, service)

    st.caption(
        f"Phiên bản 0.1 · `HGRAG_LLM_PROVIDER={os.getenv('HGRAG_LLM_PROVIDER', settings.llm_provider)}`"
    )


if __name__ == "__main__":
    main()
