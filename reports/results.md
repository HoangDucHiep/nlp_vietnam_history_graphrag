# Kết quả đánh giá — Vietnamese History GraphRAG

- Số câu hỏi: **6**
- Số hệ thống: **1**
- Tổng số lượt chấm: **6**

## 0. Cấu hình lượt chạy

| Tham số | Giá trị |
|---|---|
| llm_model | gemma-4-26b-a4b-it |
| embed_model | gemini-embedding-2 |
| rerank_backend | auto |
| vector_top_k | 30 |
| graph_max_hops | 3 |
| context_max_tokens | 6000 |
| max_evidence | 12 |
| reranker | HeuristicReranker |
| embedder | local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| collections | chunks_local, chunks_vhe_local |
| judge | LLM judge (Gemini) |

> Các số liệu dưới đây được sinh tự động từ `artifacts/eval_run.json` bằng `scripts/07_run_eval.py`; có thể tái lập bằng `python scripts/07_run_eval.py`.

## 1. Bảng chính (baseline vs Ours)

| Hệ thống | n | Overall acc | Groundedness | Token-F1 | Cite-P | Cite-R | Path hit | p50 (s) | p95 (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Ours (Hybrid GraphRAG)** | 6 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 14.92 | 57.72 |

### Theo nhóm câu hỏi (accuracy)

| Hệ thống | out_of_kb |
|---|---|
| **Ours (Hybrid GraphRAG)** | 1.000 |

## 2. Bảng ablation

| Cấu hình | Overall acc | Δ so với Ours | Groundedness | Token-F1 | Path hit |
|---|---|---|---|---|---|
| Ours (đầy đủ) | 1.000 | 0 | 1.000 | 0.000 | 0.000 |

## 3. Thống kê (bootstrap CI 95% + McNemar)

| So sánh | n | Acc baseline | Acc target | Δ (95% CI) | McNemar p |
|---|---|---|---|---|---|
