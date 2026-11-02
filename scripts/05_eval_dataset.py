#!/usr/bin/env python
"""Step 5: build the evaluation question set (docs/04 §2).

Usage
-----
    python scripts/05_eval_dataset.py --target 150 --mock --offline
    python scripts/05_eval_dataset.py --target 180 --sources vhe,llm,chat,outkb
    python scripts/05_eval_dataset.py --target 150 --corpus data/processed/corpus.jsonl

The script prints a JSON summary on stdout and writes the questions to
``data/processed/eval_questions.jsonl``.
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
from hgrag.data.records import Chunk  # noqa: E402
from hgrag.eval.dataset import EvalDataset, EvalQuestion  # noqa: E402
from hgrag.eval.dataset_builder import EvalDatasetBuilder  # noqa: E402
from hgrag.llm.client import MockLLMClient, build_llm_client  # noqa: E402

DEFAULT_SOURCES = "vhe,llm,chat,outkb"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sinh bộ câu hỏi đánh giá (VHE template + LLM + history chat + out-of-KB)",
    )
    p.add_argument(
        "--target",
        type=int,
        default=150,
        help="số câu hỏi mục tiêu (default: 150)",
    )
    p.add_argument(
        "--sources",
        type=str,
        default=DEFAULT_SOURCES,
        help=(
            "các nguồn sinh câu hỏi, phân tách bằng dấu phẩy: "
            f"vhe,llm,chat,outkb (default: {DEFAULT_SOURCES})"
        ),
    )
    p.add_argument(
        "--corpus",
        type=str,
        default="data/processed/corpus.jsonl",
        help="đường dẫn corpus chunk (default: data/processed/corpus.jsonl)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/processed/eval_questions.jsonl",
        help="file JSONL đầu ra (default: data/processed/eval_questions.jsonl)",
    )
    p.add_argument(
        "--stats-out",
        type=str,
        default="artifacts/eval_stats.json",
        help="file JSON thống kê (default: artifacts/eval_stats.json)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="dùng MockLLMClient (offline, deterministic) thay cho Gemini",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="không gọi mạng: chỉ dùng cache VHE / history-chat cục bộ",
    )
    p.add_argument(
        "--max-llm-chunks",
        type=int,
        default=60,
        help="số chunk tối đa dùng để sinh câu hỏi bằng LLM (default: 60)",
    )
    p.add_argument(
        "--kg",
        type=str,
        default="neo4j",
        help="nơi lấy entity của KG để dựng gold path (neo4j | inmemory | none)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=5,
        help="số câu hỏi mẫu in ra (0 = không in)",
    )
    p.add_argument("--verbose", action="store_true", help="bật log DEBUG")
    return p


def load_corpus(path: str | Path) -> list[Chunk]:
    """Read a corpus JSONL file into :class:`Chunk` objects."""
    source = Path(path)
    if not source.is_absolute():
        source = REPO_ROOT / source
    if not source.exists():
        return []
    chunks: list[Chunk] = []
    with source.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except ValueError:
                continue
    return chunks


def load_kg_entities(store_kind: str) -> list:
    """Read the entities of the knowledge graph (used for the gold paths).

    The gold path of a question must consist of nodes that the knowledge
    graph really has, otherwise the metric ``path_hit_rate`` penalises the
    system for facts it could never retrieve.  The builder resolves the
    mentioned names against this index (see
    :meth:`hgrag.eval.dataset_builder.EvalDatasetBuilder._kg_path`).
    """
    if store_kind == "none":
        return []
    from hgrag.config import get_settings as _get_settings
    from hgrag.data.records import Entity

    settings = _get_settings()
    try:
        if store_kind == "inmemory":
            from hgrag.kg.store import InMemoryGraphStore

            store = InMemoryGraphStore()
        else:
            from hgrag.kg.store import Neo4jGraphStore

            ncfg = settings.neo4j
            store = Neo4jGraphStore(
                uri=settings.neo4j_uri,
                user=str(ncfg.get("user", "neo4j")),
                password=settings.neo4j_password,
                database=str(ncfg.get("database", "neo4j")),
            )
        props = store.list_entities()
    except Exception as exc:  # noqa: BLE001 - the dataset can be built without the KG
        logging.getLogger(__name__).warning("Không đọc được entity KG: %s", exc)
        return []
    out: list = []
    for item in props:
        entity = Entity(
            uid=str(item.get("uid") or ""),
            name=str(item.get("name") or ""),
            entity_type=str(item.get("entity_type") or "Entity"),
            aliases=list(item.get("aliases") or []),
            name_key=str(item.get("name_key") or ""),
            wikidata_id=str(item.get("wikidata_id") or ""),
        )
        if entity.name:
            out.append(entity)
    logging.getLogger(__name__).info("KG: %d entity dùng để dựng gold path", len(out))
    return out


def sample_payload(dataset: EvalDataset, n: int) -> list[dict[str, object]]:
    """A few representative questions with their gold fields.

    Chọn tối đa `n` câu: ưu tiên một câu cho mỗi `question_type` khác nhau,
    sau đó lấp đầy các chỗ trống còn lại bằng câu từ các nguồn chưa xuất hiện.
    """

    def to_payload(question: EvalQuestion) -> dict[str, object]:
        return {
            "qid": question.qid,
            "question": question.question,
            "answer": question.answer,
            "question_type": question.question_type,
            "gold_entities": question.gold_entities,
            "gold_qids": question.gold_qids,
            "gold_chunk_ids": question.gold_chunk_ids,
            "gold_relations": question.gold_relations,
            "gold_paths": question.gold_paths,
            "year_refs": question.year_refs,
            "difficulty": question.difficulty,
            "source": question.source,
        }

    if n <= 0:
        return []

    picked: list[EvalQuestion] = []
    seen_types: set[str] = set()
    seen_sources: set[str] = set()

    # Vòng 1: một câu đại diện cho mỗi loại câu hỏi.
    for question in dataset.questions:
        if question.question_type in seen_types:
            continue
        seen_types.add(question.question_type)
        seen_sources.add(question.source)
        picked.append(question)
        if len(picked) >= n:
            return [to_payload(q) for q in picked]

    # Vòng 2: lấp đầy bằng câu từ nguồn chưa dùng.
    for question in dataset.questions:
        if question in picked:
            continue
        if question.source in seen_sources:
            continue
        seen_sources.add(question.source)
        picked.append(question)
        if len(picked) >= n:
            break

    # Vòng 3: nếu vẫn thiếu thì lấp bằng bất kỳ câu nào chưa chọn.
    if len(picked) < n:
        for question in dataset.questions:
            if question in picked:
                continue
            picked.append(question)
            if len(picked) >= n:
                break

    return [to_payload(q) for q in picked[:n]]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    settings.ensure_dirs()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    chunks = load_corpus(args.corpus)
    if not chunks:
        logging.getLogger(__name__).warning(
            "Không đọc được corpus từ %s -> nguồn llm_from_wiki sẽ rỗng", args.corpus
        )

    if args.mock:
        llm = MockLLMClient(embed_dim=settings.embed_dim, seed=settings.seed)
    else:
        llm = build_llm_client(settings)

    builder = EvalDatasetBuilder(
        llm_client=llm,
        settings=settings,
        corpus_chunks=chunks,
        kg_entities=load_kg_entities(args.kg),
        max_llm_chunks=args.max_llm_chunks,
    )
    dataset = builder.build(
        target_size=args.target,
        sources=sources,
        offline=args.offline,
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    n_written = dataset.save_jsonl(out_path)

    stats = dataset.stats()
    stats.update(
        {
            "out_path": str(out_path),
            "n_written": n_written,
            "corpus_path": str(args.corpus),
            "n_corpus_chunks": len(chunks),
            "sources_requested": sources,
            "llm_client": getattr(llm, "name", "unknown"),
            "offline": bool(args.offline),
        }
    )
    stats["samples"] = sample_payload(dataset, args.samples)

    stats_path = Path(args.stats_out)
    if not stats_path.is_absolute():
        stats_path = REPO_ROOT / stats_path
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["stats_out"] = str(stats_path)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
