#!/usr/bin/env python
"""Chuẩn hoá lại ``gold_paths`` của bộ câu hỏi đánh giá theo đồ thị KG.

Vì sao cần script này
--------------------
``gold_paths`` là thứ mà metric ``path_hit_rate`` so với các đường đi mà
bộ truy xuất đi trên đồ thị.  Nếu một node trong gold path không tồn tại
trong KG (ví dụ ``"Vua Hán"`` trong khi KG chỉ có ``"vua Hán"``), hoặc nếu
gold path chỉ là một *danh sách thực thể* chứ không phải đường đi, thì
metric đo sai: hệ thống bị trừ điểm cho những sự kiện mà nó **không thể**
truy xuất được.

Cách sửa
--------
1. Nạp các node của KG cùng chỉ mục tên -> node.
2. Với mỗi node của gold path, thay bằng **tên node thật** của KG:
   * khớp chính xác (hoặc qua alias) -> dùng tên node;
   * chỉ khác hoa/thường -> dùng tên node;
   * node là tiền tố của một node khác (các node sự kiện có dạng
     ``"<chủ thể> <động từ> <đối tượng>"``) -> dùng node dài hơn;
   * không tìm thấy -> **loại bỏ** node đó khỏi gold path.
3. Nếu loại bỏ làm gold path còn dưới 2 node thì giữ nguyên đường đi cũ
   (một đường đi dưới 2 node không mang ý nghĩa nào).

Usage
-----
    python scripts/11_fix_eval_paths.py --dry-run
    python scripts/11_fix_eval_paths.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.data.normalize import strip_diacritics  # noqa: E402
from hgrag.eval.dataset import EvalDataset  # noqa: E402

logger = logging.getLogger("11_fix_eval_paths")

DEFAULT_QUESTIONS = "data/processed/eval_questions.jsonl"

_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chuẩn hoá gold_paths theo KG")
    p.add_argument("--questions", type=str, default=DEFAULT_QUESTIONS)
    p.add_argument("--out", type=str, default=None, help="mặc định ghi đè file input")
    p.add_argument("--store", type=str, default="neo4j", help="neo4j | none")
    p.add_argument("--dry-run", action="store_true", help="chỉ báo cáo, không ghi file")
    return p


def load_kg_index(store_kind: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(name_key -> node name, name_key -> normalised node name)``."""
    if store_kind == "none":
        return {}, {}
    from hgrag.config import get_settings
    from hgrag.kg.store import Neo4jGraphStore

    settings = get_settings()
    ncfg = settings.neo4j
    store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=str(ncfg.get("user", "neo4j")),
        password=settings.neo4j_password,
        database=str(ncfg.get("database", "neo4j")),
    )
    by_key: dict[str, str] = {}
    normalised: dict[str, str] = {}
    for props in store.list_entities():
        name = str(props.get("name") or "")
        if not name:
            continue
        key = _exact_key(name)
        if not key:
            continue
        normalised.setdefault(key, name)
        by_key.setdefault(key, name)
        for alias in props.get("aliases") or []:
            alias_key = _exact_key(str(alias))
            if alias_key:
                by_key.setdefault(alias_key, name)
    logger.info("KG: %d node, %d khóa tên", len(normalised), len(by_key))
    return by_key, normalised


def _exact_key(name: str) -> str:
    """A *strict* key of a node name.

    :func:`hgrag.data.normalize.name_key` strips the honorifics on purpose
    ("Đế Minh" and "Vua Minh" share the key ``"minh"``), which is right for
    entity *matching* but wrong for *node identity*: the two nodes are
    different people.  The gold path must point at the exact node, so the
    key used here only folds the case and the diacritics.
    """
    text = strip_diacritics(str(name or "").lower())
    text = _NON_WORD_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def resolve(
    name: str,
    by_key: dict[str, str],
    normalised: dict[str, str],
) -> tuple[str | None, str]:
    """Resolve ``name`` to a KG node.

    Returns ``(node_name, reason)``; ``node_name`` is ``None`` when the KG
    does not know the name.  The reasons are:

    * ``exact``   -- the name (or one of its aliases) is a node;
    * ``case``    -- the name only differs in the case;
    * ``prefix``  -- the name is contained in a longer node name (the event
      nodes are ``"<agent> <verb> <theme>"``, so ``"sinh ra Đế Nghi"`` is
      contained in ``"Đế Minh sinh ra Đế Nghi"``);
    * ``missing`` -- the KG does not know the name at all.
    """
    clean = str(name or "").strip()
    if not clean:
        return None, "missing"
    key = _exact_key(clean)
    if not key:
        return None, "missing"
    if key in by_key:
        return by_key[key], "exact"
    lowered = key.lower()
    for candidate in normalised:
        if candidate.lower() == lowered:
            return normalised[candidate], "case"
    # The longest containing node wins: the event node carries the whole
    # phrase, the generic one only a part of it.
    best: str | None = None
    for candidate in normalised:
        if key and key in candidate and (best is None or len(candidate) > len(best)):
            best = candidate
    if best is not None:
        return normalised[best], "prefix"
    return None, "missing"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = REPO_ROOT / questions_path
    dataset = EvalDataset.load_jsonl(questions_path)
    if not len(dataset):
        print(json.dumps({"error": f"Không đọc được {questions_path}"}, ensure_ascii=False))
        return 2

    by_key, normalised = load_kg_index(args.store)

    stats: Counter[str] = Counter()
    changed = 0
    for question in dataset.questions:
        paths = [list(p) for p in (question.gold_paths or []) if p]
        if not paths:
            continue
        # The gold path of a Wikipedia / history-chat question starts at the
        # document title (the node the chunk is attached to), so it is a
        # *real* path of the graph and not just a list of entities.
        anchor = ""
        if question.source in ("llm_from_wiki", "history_chat"):
            anchor = str((question.metadata or {}).get("title") or "")
        new_paths: list[list[str]] = []
        for path in paths:
            resolved: list[str] = []
            for node in ([anchor, *path] if anchor else path):
                node_name, reason = resolve(node, by_key, normalised)
                stats[reason] += 1
                if node_name is None:
                    continue
                if node_name not in resolved:
                    resolved.append(node_name)
            if len(resolved) >= 2:
                new_paths.append(resolved)
            else:
                # A one-node path carries no edge: keep the original.
                original = [str(n).strip() for n in path if str(n).strip()]
                new_paths.append(original)
                stats["kept_original_short"] += 1
        if new_paths != paths:
            changed += 1
        question.gold_paths = new_paths

    report = {
        "n_questions": len(dataset.questions),
        "n_changed": changed,
        "node_resolution": dict(stats),
    }
    if not args.dry_run:
        out_path = Path(args.out) if args.out else questions_path
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        dataset.save_jsonl(out_path)
        report["out"] = str(out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
