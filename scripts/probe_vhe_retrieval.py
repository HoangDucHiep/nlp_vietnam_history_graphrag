#!/usr/bin/env python
"""Truy xét vì sao hệ thống `ours` từ chối các câu hỏi VHE.

Câu hỏi mẫu: "Ai là người thành lập Văn Lang?" -- đáp án nằm ngay trong câu
VHE "Hùng Vương lên ngôi, đặt quốc hiệu là Văn Lang ...".

Script đo cho từng câu hỏi VHE:
1. chunk VHE chứa đáp án có nằm trong top-k của tìm kiếm vector không;
2. hệ số cosine của nó;
3. vị trí của nó trong danh sách xếp hạng;
4. so sánh với tìm kiếm từ khoá (lexical).

Mục đích: phân biệt "mô hình sinh câu trả lời sai" với "hệ thống không
truy xuất được tài liệu chứa đáp án".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hgrag.config import get_settings  # noqa: E402
from hgrag.retrieval.embeddings import build_embedder  # noqa: E402
from hgrag.retrieval.vector_store import QdrantVectorStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Truy xét truy xuất cho câu hỏi VHE")
    parser.add_argument("--qids", nargs="*", default=["vhe_00006", "vhe_00010", "vhe_00012"])
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args(argv)

    settings = get_settings()
    embedder = build_embedder(settings)
    qcfg = settings.qdrant
    is_local = str(getattr(embedder, "name", "")).startswith("local:")
    collection = str(
        qcfg.get("collection_chunks_local", "chunks_local")
        if is_local
        else qcfg.get("collection_chunks", "chunks")
    )
    vhe_collection = str(
        qcfg.get(
            "collection_chunks_vhe_local" if is_local else "collection_chunks_vhe",
            "chunks_vhe_local" if is_local else "chunks_vhe",
        )
    )
    store = QdrantVectorStore(
        url=str(qcfg.get("url", "http://localhost:6333")),
        vector_size=int(getattr(embedder, "dim", settings.embed_dim)),
    )
    print(f"embedder={getattr(embedder, 'name', '?')} dim={getattr(embedder, 'dim', '?')}")
    print(f"collections={collection}, {vhe_collection}\n")

    questions = {}
    with open(REPO_ROOT / "data/processed/eval_questions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            questions[row["qid"]] = row

    # Các chunk VHE có chứa đáp án: tìm theo tên thực thể và của câu hỏi.
    corpus: dict[str, str] = {}
    vhe_path = REPO_ROOT / "data/processed/vhe_corpus.jsonl"
    if vhe_path.exists():
        with open(vhe_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                corpus[row["chunk_id"]] = row.get("text") or ""

    for qid in args.qids:
        question = questions.get(qid)
        if question is None:
            continue
        text = question["question"]
        answer = question.get("answer") or ""
        vec = embedder.embed([text])[0]
        merged: list[dict] = []
        for name in (collection, vhe_collection):
            try:
                merged.extend(store.search(name, list(vec), top_k=args.top_k) or [])
            except Exception as exc:  # noqa: BLE001
                print(f"  ! search {name}: {exc}")
        merged.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)

        # Chunk VHE chứa cả câu hỏi lẫn đáp án -> "đáp án nằm trong corpus".
        gold_haystack = [
            (cid, txt)
            for cid, txt in corpus.items()
            if answer and answer.lower() in txt.lower()
        ]
        gold_ids = {cid for cid, _ in gold_haystack}

        ranks = []
        for rank, hit in enumerate(merged, 1):
            payload = hit.get("payload") or {}
            chunk_id = str(payload.get("chunk_id") or "")
            if chunk_id in gold_ids:
                ranks.append((rank, chunk_id, round(float(hit.get("score", 0.0)), 4)))

        print(f"--- {qid}: {text}")
        print(f"    đáp án: {answer!r}  | chunk VHE chứa đáp án: {len(gold_ids)}")
        if ranks:
            print(f"    xếp hạng của chunk chứa đáp án: {ranks[:5]}")
        else:
            print("    !! KHÔNG chunk nào chứa đáp án nằm trong top-k")
        print("    top-5 truy xuất:")
        for rank, hit in enumerate(merged[:5], 1):
            payload = hit.get("payload") or {}
            snippet = str(payload.get("text") or "")[:110].replace("\n", " ")
            print(
                f"      {rank}. score={float(hit.get('score', 0.0)):.4f} "
                f"chunk={payload.get('chunk_id')} :: {snippet}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
