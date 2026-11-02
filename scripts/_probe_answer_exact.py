"""Reproduce the exact eval context and run the answerer on it."""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from hgrag.config import get_settings
from hgrag.llm.client import build_llm_client
from hgrag.llm.schemas import LLMMessage, LLMRequest
from hgrag.qa.answer import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from hgrag.qa.pipeline import build_pipeline

QUESTIONS = [
    "Ai là người thành lập Nam Hải?",
    "Ai là người cưới Sơn Tinh?",
    "Ai là người tấn công Nam Chiếu?",
]


def main() -> None:
    settings = get_settings()
    pipeline = build_pipeline(settings)
    client = build_llm_client(settings)
    for question in QUESTIONS:
        understanding = pipeline._understand(question)
        linked = pipeline._link(understanding, {})
        graph = pipeline._graph_search(linked, understanding, pipeline.max_hops, {})
        vector_hits = pipeline._vector_search(question, pipeline.top_k, understanding, {})
        vector_hits = pipeline._rerank(question, vector_hits, {})
        evidence = pipeline._fuse(vector_hits, graph, understanding, {})
        pack = pipeline._build_context(evidence, {})
        prompt = USER_PROMPT_TEMPLATE.format(
            context=pack.text,
            question=question,
            plan="\n(Gợi ý phân tích: loại câu hỏi = single_hop, số bước suy luận dự kiến = 1.)\n",
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ],
            task="answer_grounded",
            temperature=0.0,
        )
        response = client.generate(request)
        print("=" * 78)
        print("Q:", question)
        print("--- RAW ANSWER ---")
        print(response.text)


if __name__ == "__main__":
    main()
