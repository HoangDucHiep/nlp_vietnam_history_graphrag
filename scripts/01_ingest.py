#!/usr/bin/env python
"""Step 1 of the pipeline: ingest raw sources into ``data/processed/corpus.jsonl``.

Usage
-----
    python scripts/01_ingest.py --limit 5 --sources wikipedia
    python scripts/01_ingest.py --limit 5 --sources vhe
    python scripts/01_ingest.py --limit 5 --sources history_chat
    python scripts/01_ingest.py --sources wikipedia,vhe,history_chat --seed-file seeds.txt

The script prints a JSON summary on stdout and never crashes because of a
failing data source (warnings are reported inside the JSON payload).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgrag.config import get_settings  # noqa: E402
from hgrag.data.ingest import ingest_all  # noqa: E402

DEFAULT_SOURCES = "wikipedia,vhe,history_chat"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest Wikipedia / VHE / History-Chat into data/processed/corpus.jsonl",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="giới hạn số bài (wikipedia) / số instance (vhe) / số câu (history_chat)",
    )
    p.add_argument(
        "--seed-file",
        type=str,
        default=None,
        help="file txt, mỗi dòng một tên bài Wikipedia seed",
    )
    p.add_argument(
        "--sources",
        type=str,
        default=DEFAULT_SOURCES,
        help=f"danh sách nguồn, phân tách bởi dấu phẩy (default: {DEFAULT_SOURCES})",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="thư mục output (default: data/processed theo settings)",
    )
    p.add_argument(
        "--no-network",
        action="store_true",
        help="không gọi mạng (chỉ dùng cache cục bộ)",
    )
    p.add_argument(
        "--print-chunks",
        type=int,
        default=3,
        help="số chunk mẫu in ra (0 = không in)",
    )
    p.add_argument("--verbose", action="store_true", help="bật log DEBUG")
    return p


def read_seed_file(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Seed file not found: {p}")
    seeds: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seeds.append(line)
    return seeds


def show_sample_chunks(corpus_path: Path, n: int) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    if not corpus_path.exists() or n <= 0:
        return samples
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            samples.append(
                {
                    "chunk_id": obj.get("chunk_id", ""),
                    "title": obj.get("title", ""),
                    "section": obj.get("section", ""),
                    "source": obj.get("source", ""),
                    "url": obj.get("url", ""),
                    "text_preview": (obj.get("text", "") or "")[:200],
                }
            )
            if len(samples) >= n:
                break
    return samples


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    settings.ensure_dirs()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in {"wikipedia", "vhe", "history_chat"}]
    if unknown:
        print(json.dumps({"error": f"Unknown sources: {unknown}"}, ensure_ascii=False, indent=2))
        return 2

    seed_titles = None
    if args.seed_file:
        try:
            seed_titles = read_seed_file(args.seed_file)
        except FileNotFoundError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
            return 2

    stats = ingest_all(
        settings=settings,
        limit=args.limit,
        sources=sources,
        seed_titles=seed_titles,
        out_dir=args.out,
        offline=args.no_network,
    )
    corpus_path = Path(stats["corpus_path"])
    stats["sample_chunks"] = show_sample_chunks(corpus_path, args.print_chunks)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())