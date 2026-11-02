# Kết quả đánh giá — Vietnamese History GraphRAG

- Số câu hỏi: **150**
- Số hệ thống: **11**
- Tổng số lượt chấm: **1650**

## 0. Cấu hình lượt chạy

| Tham số | Giá trị |
|---|---|
| llm_model | gemma-4-26b-a4b-it |
| rerank_backend | auto |
| vector_top_k | 30 |
| graph_max_hops | 3 |
| context_max_tokens | 6000 |
| max_evidence | 12 |
| reranker | HeuristicReranker |
| embedder | local:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 |
| collections | chunks_local, chunks_vhe_local |
| judge | LLM judge (gemma-4-26b-a4b-it) |

> Các số liệu dưới đây được sinh tự động từ `artifacts/eval_run.json` bằng `scripts/07_run_eval.py`; có thể tái lập bằng `python scripts/07_run_eval.py`.

## 1. Bảng chính (baseline vs Ours)

| Hệ thống | n | Overall acc | Groundedness | Token-F1 | Cite-P | Cite-R | Path hit | p50 (s) | p95 (s) |
|---|---|---|---|---|---|---|---|---|---|
| B0 LLM-only | 150 | 0.240 | – | 0.052 | 0.000 | 0.000 | 0.000 | 0.01 | 273.60 |
| B1 Vanilla RAG | 150 | 0.213 | 0.900 | 0.025 | 0.080 | 0.087 | 0.000 | 0.10 | 0.16 |
| B2 RAG+Graph | 150 | 0.267 | 0.707 | 0.051 | 0.158 | 0.140 | 0.488 | 32.03 | 474.56 |
| **Ours (Hybrid GraphRAG)** | 150 | 0.600 | 0.853 | 0.108 | 0.511 | 0.413 | 0.488 | 72.70 | 643.54 |

### Theo nhóm câu hỏi (accuracy)

| Hệ thống | multi_hop | out_of_kb | single_hop | temporal |
|---|---|---|---|---|
| B0 LLM-only | 0.133 | 0.636 | 0.233 | 0.087 |
| B1 Vanilla RAG | 0.044 | 1.000 | 0.133 | 0.000 |
| B2 RAG+Graph | 0.178 | 1.000 | 0.083 | 0.217 |
| **Ours (Hybrid GraphRAG)** | 0.511 | 1.000 | 0.567 | 0.478 |

## 2. Bảng ablation

| Cấu hình | Overall acc | Δ so với Ours (− = kém hơn) | Groundedness | Token-F1 | Path hit |
|---|---|---|---|---|---|
| Ours (đầy đủ) | 0.600 | 0 | 0.853 | 0.108 | 0.488 |
| A1 bỏ entity linking | 0.320 | -0.280 | 0.673 | 0.058 | 0.000 |
| A2 bỏ entity resolution | 0.353 | -0.247 | 0.700 | 0.070 | 0.008 |
| A3 bỏ temporal boost/filter | 0.587 | -0.013 | 0.873 | 0.103 | 0.512 |
| A4 bỏ graph traversal | 0.307 | -0.293 | 0.653 | 0.058 | 0.000 |
| A5 bỏ rerank | 0.553 | -0.047 | 0.800 | 0.100 | 0.512 |
| A6 bỏ context budget | 0.607 | +0.007 | 0.853 | 0.112 | 0.488 |
| A7 bỏ fusion | 0.300 | -0.300 | 0.660 | 0.056 | 0.488 |

## 3. Thống kê (bootstrap CI 95% + McNemar)

| So sánh | n | Acc baseline | Acc target | Δ = Ours − baseline (95% CI) | McNemar p |
|---|---|---|---|---|---|
| Ours vs B0 | 150 | 0.240 | 0.600 | +0.360 [+0.267, +0.460] | 0.0000 |
| Ours vs B1 | 150 | 0.213 | 0.600 | +0.387 [+0.307, +0.473] | 0.0000 |
| Ours vs B2 | 150 | 0.267 | 0.600 | +0.333 [+0.253, +0.413] | 0.0000 |

### Chi tiết so sánh

```json
[
  {
    "baseline": "B0",
    "target": "ours",
    "n_questions": 150,
    "baseline_accuracy": 0.24,
    "target_accuracy": 0.6,
    "baseline_accuracy_ci": {
      "mean": 0.24,
      "std": 0.427083,
      "ci_low": 0.173333,
      "ci_high": 0.313333,
      "n_samples": 10000,
      "confidence": 0.95
    },
    "target_accuracy_ci": {
      "mean": 0.6,
      "std": 0.489898,
      "ci_low": 0.52,
      "ci_high": 0.68,
      "n_samples": 10000,
      "confidence": 0.95
    },
    "accuracy_diff_ci": {
      "mean": 0.36,
      "std": 0.603103,
      "ci_low": 0.266667,
      "ci_high": 0.46,
      "n_samples": 10000,
      "confidence": 0.95
    },
    "mcnemar": {
      "b": 64,
      "c": 10,
      "n": 74,
      "chi2": 37.959459,
      "p_value": 0.0,
      "significant": true
    }
  },
  {
    "baseline": "B1",
    "target": "ours",
    "n_questions": 150,
    "baseline_accuracy": 0.21333333333333335,
    "target_accuracy": 0.6,
    "baseline_accuracy_ci": {
      "mean": 0.213333,
      "std": 0.409661,
      "ci_low": 0.146667,
      "ci_high": 0.28,
      "n_samples": 10000,
      "confidence": 0.95
    },
    "target_accuracy_ci": {
      "mean": 0.6,
      "std": 0.489898,
      "ci_low": 0.52,
      "ci_high": 0.68,
      "n_samples": 10000,
      "confidence": 0.95
    },
    "accuracy_diff_ci": {
      "mean": 0.386667,
      "std": 0.526456,
      "ci_low": 0.306667,
      "ci_high": 0.473333,
      "n_samples": 10000,
      "confidence": 0.95
    },
    "mcnemar": {
      "b": 61,
      "c": 3,
      "n": 64,
      "chi2": 50.765625,
      "p_value": 0.0,
      "significant": true
    }
  },
  {
    "baseline": "B2",
    "target": "ours",
    "n_questions": 150,
    "baseline_accuracy": 0.26666666666666666,
    "target_accuracy": 0.6,
    "baseline_accuracy_ci": {
      "mean": 0.266667,
      "std": 0.442217,
      "ci_low": 0.2,
      "ci_high": 0.34,
      "n_samples": 10000,
      "confidence": 0.95
    },
    "target_accuracy_ci": {
      "mean": 0.6,
      "std": 0.489898,
      "ci_low": 0.52,
      "ci_high": 0.68,
      "n_samples": 10000,
      "confidence": 0.95
    },
    "accuracy_diff_ci": {
      "mean": 0.333333,
      "std": 0.498888,
      "ci_low": 0.253333,
      "ci_high": 0.413333,
      "n_samples": 10000,
      "confidence": 0.95
    },
    "mcnemar": {
      "b": 52,
      "c": 2,
      "n": 54,
      "chi2": 44.462963,
      "p_value": 0.0,
      "significant": true
    }
  }
]
```
