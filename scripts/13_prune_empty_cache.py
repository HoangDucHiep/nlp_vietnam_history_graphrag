#!/usr/bin/env python
"""Xoá các entry rỗng khỏi cache LLM trên đĩa.

Gemma 4 đã trả về `text` **rỗng** cho nhiều request (token suy luận ăn hết
ngân sách output).  Những response rỗng ấy đã được ghi vào cache, nên mọi
lượt chạy sau sẽ *đọc lại* chúng và không hề gọi lại API -- kết quả là
các dòng đánh giá vẫn rỗng dù đã sửa nguyên nhân gốc.

Script này xoá các entry có ``text`` rỗng (hoặc thiếu ``text``) khỏi
``artifacts/llm_cache`` và báo cáo số lượng đã xoá.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Xoá entry rỗng khỏi cache LLM")
    parser.add_argument("--cache", default="artifacts/llm_cache", help="thư mục cache")
    parser.add_argument("--dry-run", action="store_true", help="chỉ báo cáo, không xoá")
    args = parser.parse_args(argv)

    root = Path(args.cache)
    if not root.is_absolute():
        root = REPO_ROOT / root
    if not root.exists():
        print(json.dumps({"cache": str(root), "error": "không tồn tại"}, ensure_ascii=False))
        return 1

    removed = 0
    kept = 0
    by_model: dict[str, int] = {}
    for path in sorted(root.glob("*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            removed += 1
            if not args.dry_run:
                path.unlink(missing_ok=True)
            continue
        text = payload.get("text")
        # Entry rỗng: câu trả lời bị mất do token suy luận ăn hết ngân sách.
        # Entry embedding không có "text" nhưng có "vector" -> giữ lại.
        if isinstance(text, str) and not text.strip() and "vector" not in payload:
            model = str(payload.get("model") or "?")
            by_model[model] = by_model.get(model, 0) + 1
            removed += 1
            if not args.dry_run:
                path.unlink(missing_ok=True)
        else:
            kept += 1

    print(
        json.dumps(
            {
                "cache": str(root),
                "dry_run": args.dry_run,
                "removed": removed,
                "kept": kept,
                "removed_by_model": by_model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
