# Khung báo cáo — Vietnamese History GraphRAG (905–1975)

> File này là **khung** (skeleton) của bài báo cáo đồ án. Các số liệu trong
> báo cáo cuối **phải** lấy từ [`reports/results.csv`](results.csv) (sinh bởi
> `python scripts/07_run_eval.py`) — không ghi tay.
>
> Quy ước: `[...]` = chỗ cần điền số liệu thật.

---

## 1. Tóm tắt

**Bối cảnh.** Hệ thống hỏi đáp lịch sử Việt Nam giai đoạn 905–1975 cần trả lời
các câu hỏi **nhiều bước** (multi-hop) với **căn cứ kiểm chứng được** (mỗi mệnh đề
kèm trích dẫn tới đoạn văn bản nguồn). RAG thuần dựa trên embedding khó trả lời
các câu hỏi cần nối 2–3 sự kiện qua quan hệ nhân quả/thời gian.

**Đóng góp.** Đề xuất kiến trúc **hybrid GraphRAG** cho lịch sử Việt Nam:

1. Knowledge Graph theo ontology **VHE** (35 loại sự kiện, 11 vai trò) với
   **provenance bắt buộc** (`[:SUPPORTED_BY] → :Chunk`) cho mọi cạnh.
2. **Định danh thực thể QID-first 6 tầng** giải quyết bài toán trùng tên/alias.
3. Truy xuất **hybrid** (vector + BFS đồ thị) hợp nhất bằng **RRF**, kèm
   **temporal boost** và **diversity cap** chống nhiễu prompt.
4. Sinh câu trả lời **có ràng buộc căn cứ** (`[E#]`) kèm chuỗi suy luận và subgraph.
5. Hệ thống đánh giá **đầy đủ**: 4 hệ thống, 7 ablation, CI bootstrap 95%,
   McNemar, phân tích lỗi.

**Kết quả (chính).**

| Hệ thống | Overall acc | Groundedness | Token-F1 | Path hit |
|---|---|---|---|---|
| B0 LLM-only | [...] | [...] | [...] | – |
| B1 Vanilla RAG | [...] | [...] | [...] | – |
| B2 RAG+Graph | [...] | [...] | [...] | [...] |
| **Ours** | **[...]** | **[...]** | **[...]** | **[...]** |

*(điền từ `reports/results.md` bảng §1)*

**Kết luận.** [Viết sau khi có số liệu: hệ thống thắng/hoà/thua baseline ở nhóm nào,
và vì sao.]

---

## 2. Bài toán & phạm vi

* Khoảng thời gian: **905–1975** (Đinh → Nguyễn).
* Ngôn ngữ: tiếng Việt.
* Loại câu hỏi: single-hop, multi-hop, temporal, out-of-KB.
* Ràng buộc thiết kế: **không bịa**, mọi mệnh đề phải dẫn được nguồn.

## 3. Dữ liệu

| Nguồn | Số lượng | Ghi chú |
|---|---|---|
| Wikipedia tiếng Việt (seed 59 bài) | **438 chunk** | nguồn chính, có provenance URL |
| VHE (Vietnam Historical Events) | **4.110 câu chú giải** → 12.089 entity, 30.997 relation | nguồn gold cho ontology + câu hỏi template |
| History Chat | **10 câu** trong bộ đánh giá | câu hỏi "ngoài đời" |
| Wikidata | QID cho các entity khớp được | định danh thực thể, alias, P31/P27 |

*(điền từ `artifacts/embed_stats*.json`, `artifacts/kg_stats.json`)*

**Chia tập.** `15%` tài liệu giữ lại làm `eval-exclusive` (không nạp vào KG/vector
store) → câu hỏi sinh từ đó buộc hệ thống phải suy luận từ quan hệ đã học ở tài liệu
khác. Với nguồn VHE, cơ chế tương đương là **đồ thị con VHE gold**: câu hỏi được
sinh từ *quan hệ* của VHE nhưng câu trả lời phải được hỗ trợ bằng *văn bản* VHE
(câu chú giải) → đo được khả năng nối tri thức có cấu trúc với ngôn ngữ tự nhiên.

## 4. Kiến trúc hệ thống

Mô tả ngắn gọn 4 tầng (kèm sơ đồ từ `docs/02_kien_truc.md`):

1. **Ngoại tuyến**: ingest → trích xuất event (Gemini, cache đĩa) → dựng KG (Neo4j) → embed (Qdrant).
2. **Phân tích câu hỏi**: intent, entities, time_refs, relation_plan, expected_hops (LLM + heuristic fallback).
3. **Truy xuất hybrid**: entity linking → BFS đồ thị + vector search → RRF → rerank → diversity cap → ngân sách token.
4. **Sinh câu trả lời**: prompt ràng buộc → kiểm tra citation → tách chuỗi suy luận → dựng subgraph.

### 4.1 Ontology VHE & provenance

* 35 event type, 11 argument role → quan hệ KG.
* Mọi cạnh có `source_chunk_ids` + `[:SUPPORTED_BY] → :Chunk` → **truy vết được nguồn**.

### 4.2 Định danh thực thể QID-first (6 tầng)

| Tầng | Nguồn | Điểm |
|---|---|---|
| T0 | QID đã biết / node đã có trong KG | 1.00 |
| T1 | Wikidata alias vi (`skos:altLabel[vi]`, `wbsearchentities`) | 0.97 |
| T2 | Wikipedia `prop=redirects` | 0.93 |
| T3 | LLM xác minh redirect | 0.90 |
| T4 | Chuẩn hóa chuỗi (bỏ dấu, bỏ honorific, NFC) | 0.85 |
| T5 | Embedding top-k + lọc miền (P31/P27/thời gian) + LLM phân giải | 0.60–0.90 |

*(chi tiết: `docs/07_xac_dinh_thuc_the.md`)*

### 4.3 Chấm điểm đường đi

\[
S(\text{path}) = 0{,}30\,\text{affinity} + 0{,}30\,\text{similarity} + 0{,}25\,\text{temporal} - 0{,}15\,\text{degree\_penalty}
\]

### 4.4 Fusion & ngân sách context

RRF \(k=60\) → temporal boost `+0.15` → diversity cap (≤ 3 evidence/doc) →
ngân sách 6 000 token.

## 5. Thiết lập thực nghiệm

* **Mô hình sinh**: `gemini-flash-lite-latest` (xoay sang `gemini-3.5-flash`, `gemini-3.8-flash`, `gemini-3.1-flash-lite` khi hết quota).
* **Embedding**: quota Gemini đã cạn → dùng model **cục bộ** `paraphrase-multilingual-MiniLM-L12-v2` (384 chiều) cho cả corpus VHE và corpus Wikipedia (`chunks_local`, `chunks_vhe_local`).
* **Rerank**: `bge-reranker-v2-m3` quá chậm trên CPU (~35 s/lần gọi) → `rerank_backend: auto` tự chuyển sang `HeuristicReranker` (BM25 + boost thời gian + cosine). *Đây là hạn chế của môi trường đánh giá và được nêu rõ trong phần phân tích.*
* **Kho**: Neo4j 5.26 + Qdrant 1.12.4.
* **Cố định**: cùng LLM, cùng prompt, cùng corpus cho mọi hệ thống → khác biệt do kiến trúc.
* **Bộ câu hỏi**: 150 câu / 4 nhóm (chi tiết `docs/04_danh_gia.md`).
* **Judge**: Gemini, rubric 0–2 + grounded + supported evidence ids; câu out-of-KB chấm theo **hành vi** (từ chối = đúng).
* **Tái lập**: `bash scripts/run_all.sh` (seed cố định `42`).

## 6. Kết quả

### 6.1 Baseline vs Ours

*(chèn bảng từ `reports/results.md` §1)*

**Nhận xét.** [So sánh từng nhóm câu hỏi; giải thích tại sao đồ thị giúp multi-hop.]

### 6.2 Thống kê

| So sánh | Δ accuracy (95% CI) | McNemar p | Kết luận |
|---|---|---|---|
| Ours vs B0 | [...] | [...] | [...] |
| Ours vs B1 | [...] | [...] | [...] |
| Ours vs B2 | [...] | [...] | [...] |

*(chèn từ `reports/results.md` §3)*

### 6.3 Ablation

*(chèn bảng từ `reports/results.md` §2)*

| Cờ | Giả thuyết | Kiểm chứng |
|---|---|---|
| A1 bỏ entity linking | alias quan trọng | [...] |
| A2 bỏ entity resolution | trùng lặp node làm nhiễu | [...] |
| A3 bỏ temporal | mất ngữ cảnh thời gian | [...] |
| A4 bỏ graph traversal | đồ thị thuần đóng góp gì | [...] |
| A5 bỏ rerank | rerank đáng giá không | [...] |
| A6 bỏ context budget | "prompt bị nhiễu" làm giảm accuracy | [...] |
| A7 bỏ fusion | hợp nhất hai nguồn đáng giá không | [...] |

### 6.4 Chất lượng truy xuất

| Hệ thống | Chunk Recall@10 | nDCG@10 | Entity recall | Path hit | Temporal acc |
|---|---|---|---|---|---|
| B1 | [...] | [...] | [...] | – | [...] |
| Ours | [...] | [...] | [...] | [...] | [...] |

### 6.5 Vận hành

| Hệ thống | p50 (s) | p95 (s) | evidence/câu |
|---|---|---|---|
| B0 | [...] | [...] | – |
| B1 | [...] | [...] | [...] |
| Ours | [...] | [...] | [...] |

## 7. Phân tích lỗi

*(tóm tắt `reports/error_analysis.md`)*

| Loại lỗi | Tần suất | Nguyên nhân gốc | Hướng khắc phục |
|---|---|---|---|
| Entity linking sai | [...] | [...] | [...] |
| Đường đi sai | [...] | [...] | [...] |
| Thiếu evidence | [...] | [...] | [...] |
| Sinh sai dù có evidence | [...] | [...] | [...] |
| Judge sai | [...] | [...] | [...] |
| Bịa (out-of-KB) | [...] | [...] | [...] |

## 8. Kết luận & hướng phát triển

* **Kết luận 1** — [...]
* **Kết luận 2** — [...]
* **Hướng 1**: mở rộng ontology (thêm quan hệ nhân quả `CAUSE_OF` từ LLM suy luận).
* **Hướng 2**: fine-tune embedder (LoRA) trên cặp (query, gold chunk) sinh từ KG.
* **Hướng 3**: mở rộng corpus Wikipedia (hiện 59 bài seed) và thêm nguồn sách lịch sử.
* **Hướng 4**: đánh giá độ đồng thuận judge–người trên mẫu 30 câu.

## 9. Tài liệu tham khảo

1. Gao et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020).
2. Microsoft, *From Local to Global: A Graph RAG Approach to Query-Focused Summarization* (2024).
3. Nguyen et al., *VHE: Vietnamese Historical Events* (benchmark event extraction).
4. Edge et al., *From RAG to GraphRAG* (2024).
5. HisGraphRAG — tác giả cung cấp, đối chiếu ablation A2/A3.

## 10. Phụ lục

* A. Đường dẫn mã nguồn từng module.
* B. Toàn bộ bảng số liệu (`reports/results.csv`).
* C. 30 ví dụ câu hỏi + câu trả lời + subgraph.
* D. Prompt đầy đủ (system + user) cho từng hệ thống.
