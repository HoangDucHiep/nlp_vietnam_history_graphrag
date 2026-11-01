# Kết quả đánh giá — Vietnamese History GraphRAG

- Số câu hỏi: **3**
- Số hệ thống: **1**
- Tổng số lượt chấm: **3**

## 1. Bảng chính (baseline vs Ours)

| Hệ thống | n | Overall acc | Groundedness | Token-F1 | Cite-P | Cite-R | Path hit | p50 (s) | p95 (s) |
|---|---|---|---|---|---|---|---|---|---|
| **Ours (Hybrid GraphRAG)** | 3 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.30 | 13.20 |

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
