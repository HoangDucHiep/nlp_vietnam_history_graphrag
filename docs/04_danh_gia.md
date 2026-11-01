# 04 — Thiết kế đánh giá

## 1. Nguyên tắc

1. **Tách biệt với dữ liệu xây hệ thống**: giữ lại 15% tài liệu làm `eval-exclusive`, **không nạp vào KG hay vector store**. Câu hỏi sinh từ các tài liệu này buộc hệ thống phải suy luận từ quan hệ đã học ở tài liệu khác.
2. **Câu hỏi có gold path / gold evidence**, không chỉ có đáp án → đo được chất lượng truy xuất chứ không chỉ chất lượng sinh.
3. **Phân nhóm rõ**: single-hop / multi-hop / temporal / comparison / out-of-KB — vì kết luận "GraphRAG hơn RAG thuần" chỉ có ý nghĩa nếu tách nhóm.
4. **Cùng LLM, cùng prompt** cho mọi hệ thống → khác biệt đo được là do *kiến trúc truy xuất*, không phải do model.
5. **Đo hallucination trực tiếp**, không chỉ qua LLM-judge chung chung.

## 2. Bộ câu hỏi đánh giá

### 2.1 Ba nguồn sinh câu hỏi

| Nguồn | Cách sinh | Lợi thế | Rủi ro | Số lượng |
|---|---|---|---|---|
| **A. Template từ gold relation VHE** | Duyệt đồ thị quan hệ temporal/causal/subevent của VHE → sinh câu hỏi theo template ("Sự kiện nào xảy ra trước...?", "Điều gì là nguyên nhân của...?") kèm **gold path chính xác** | Tự động kiểm chứng được, gold path chắc chắn, tách được multi-hop 2–3 bước | Câu hỏi có tính "khuôn mẫu" | ~70 |
| **B. Gemini sinh từ chunk Wikipedia** | Cho Gemini đọc 1–3 chunk liên quan và tự viết câu hỏi + đáp án tham chiếu → **human duyệt** trên giao diện Streamlit | Câu hỏi tự nhiên, đúng phong cách người dùng | Cần người duyệt (bạn ~20–30 phút) | ~60 |
| **C. Lấy từ Vietnam History Chat** | Chọn câu hỏi trong phạm vi 905–1975, dùng `final` làm đáp án tham chiếu → human duyệt lọc lỗi | Câu hỏi "ngoài đời", đúng kỳ vọng người dùng | Tác giả cảnh báo dữ liệu do AI tạo có lỗi | ~30 |
| **D. Câu hỏi out-of-KB (viết tay)** | Câu hỏi ngoài phạm vi KG (vd: "Ai là người phát minh ra bóng đá ở Việt Nam?") | Đo hành vi từ chối bịa | Không có đáp án đúng → chấm theo tiêu chí hành vi | ~20 |

**Tổng mục tiêu: ~180 câu.**

### 2.2 Schema một câu hỏi

```json
{
  "qid": "mh_vhe_0042",
  "question": "Sự kiện nào xảy ra trước trận Bạch Đằng năm 938 và do ai làm?",
  "group": "multi_hop",
  "source": "vhe_template",
  "reference_answer": "Trận Bạch Đằng năm 938 do Ngô Quyền chỉ huy.",
  "gold_facts": [
    {"subject": "Trận Bạch Đằng (938)", "predicate": "diễn ra trước", "object": "Trận Bạch Đằng (938)"}
  ],
  "gold_entities": ["Trận Bạch Đằng (938)", "Ngô Quyền"],
  "gold_path": [["Trận Bạch Đằng (938)", "PRECEDED_BY", "..."]],
  "gold_chunk_ids": ["wiki:Tran_Trung_Thang:sec3"],
  "is_out_of_kb": false,
  "verified": true
}
```

### 2.3 Quy trình human-verify

Giao diện Streamlit có tab **"Duyệt câu hỏi"**: hiển thị câu hỏi + đoạn nguồn → nút `Duyệt` / `Sửa` / `Bỏ`. Chỉ câu `verified=true` mới vào bộ đánh giá cuối. Ghi trong báo cáo: "180 câu, 142 được duyệt" → minh bạch.

## 3. Metrics

### 3.1 Độ đúng câu trả lời

| Metric | Cách tính | Ghi chú |
|---|---|---|
| **Token-F1** | F1 trên token đã chuẩn hóa (bỏ dấu, thường) so với đáp án tham chiếu | Rẻ, ổn định, dùng làm chỉ số chính |
| **ROUGE-L** | LCS-based | Bổ sung |
| **LLM-judge (0–2)** | Gemini chấm: 0 = sai, 1 = đúng một phần, 2 = đúng & đầy đủ. Rubric kèm đáp án tham chiếu + gold facts | Đo "đúng về nghĩa", khó đối chiếu tuyệt đối với câu MC |
| **Gold-fact accuracy** | tỉ lệ gold_facts có mặt trong đáp án (kiểm bằng LLM-judge hoặc so khớp thực thể) | Gắn trực tiếp với cấu trúc KG |

**Ngưỡng chính**: `judge_score / 2` → dùng làm `accuracy` trong bảng so sánh.

### 3.2 Bám sát nguồn & hallucination

| Metric | Cách tính |
|---|---|
| **Groundedness** | judge 0/1: mọi mệnh đề trong đáp án đều được evidence hỗ trợ |
| **Citation precision** | trong các evidence được cite, bao nhiêu thực sự chứng minh claim (judge từng cặp claim–evidence) |
| **Citation recall** | các gold evidence có bị cite không |
| **Hallucination rate** | \(1 - \text{groundedness}\) ở mức claim (chia đôi mỗi mệnh đề) |
| **No-citation rate** | tỉ lệ câu trả lời không có citation hợp lệ |

### 3.3 Chất lượng truy xuất

| Metric | Cách tính |
|---|---|
| **Chunk Recall@k** | tỉ lệ gold chunk nằm trong top-k evidence |
| **Chunk nDCG@10** | có thứ hạng của gold chunk |
| **Entity recall** | tỉ lệ gold entity được liên kết đúng |
| **Path hit rate** | đường đi truy xuất có chứa gold_path hay không ← **chỉ số riêng cho multi-hop** |
| **Temporal accuracy** | với câu hỏi có năm: evidence có chứa event đúng năm không |

### 3.4 Vận hành

- Latency p50 / p95 (ms) theo từng giai đoạn (linking / vector / graph / generation).
- Số token input/output, chi phí ước tính mỗi câu.
- Số evidence dùng, tỉ lệ evidence bị loại do ngân sách.

### 3.5 Thống kê

- **Bootstrap CI 95%** (paired, 10.000 lần lặp) cho chênh lệch accuracy giữa hệ thống và baseline.
- **McNemar test** cho so sánh hai hệ thống trên cùng câu hỏi (binary: đúng/sai).
- Báo cáo riêng theo nhóm câu hỏi.

## 4. Baseline & ablation

### 4.1 Baseline (bắt buộc)

| Mã | Hệ thống | Mô tả |
|---|---|---|
| **B0** | LLM-only | Không truy xuất, chỉ kiến thức mô hình |
| **B1** | Vanilla RAG | Vector search top-k trên cùng corpus, cùng rerank, cùng prompt sinh đáp án. **Không dùng đồ thị** ← baseline bắt buộc |
| **B2** | RAG + GraphRAG không có rerank | Để tách đóng góp của rerank |
| **Ours** | Hybrid GraphRAG đầy đủ | Vector + đồ thị + rerank + fusion + context budget + entity resolution |

> B1 phải dùng **cùng chunk, cùng embedding model, cùng LLM, cùng prompt** → khác biệt là do kiến trúc.

### 4.2 Ablation (bỏ từng thành phần của Ours)

| Mã | Bỏ gì | Kiểm chứng giả thuyết |
|---|---|---|
| A1 | Entity linking → dùng node theo tên thô | Tầng alias quan trọng thế nào |
| A2 | Bỏ entity resolution (merge) | Đo trùng lặp node ảnh hưởng bao nhiêu (đối chiếu HisGraphRAG) |
| A3 | Bỏ temporal boost + filter | Đo mất ngữ cảnh thời gian (đối chiếu HisGraphRAG) |
| A4 | Bỏ graph traversal (chỉ vector) | Tác động của đồ thị thuần |
| A5 | Bỏ rerank | Tác động của rerank |
| A6 | Bỏ context budget (đưa hết evidence) | Chứng minh "prompt bị nhiễu" làm giảm accuracy |
| A7 | Bỏ fusion (chỉ dùng vector top-k) | Tác động của việc hợp nhất hai nguồn |

## 5. Báo cáo kết quả

### 5.1 Bảng chính (mẫu)

| Hệ thống | Single-hop acc | Multi-hop acc | Temporal acc | Overall judge | Groundedness | Citation prec. | Path hit | p50 (s) |
|---|---|---|---|---|---|---|---|---|
| B0 LLM-only | | | | | | | – | |
| B1 Vanilla RAG | | | | | | | – | |
| B2 RAG+Graph (no rerank) | | | | | | | | |
| **Ours** | | | | | | | | |

### 5.2 Bảng ablation

| Cấu hình | Overall judge | Δ so với Ours | Nhận xét |
|---|---|---|---|
| Ours (đầy đủ) | | 0 | |
| A1 bỏ entity linking | | | |
| A2 bỏ entity resolution | | | |
| A3 bỏ temporal | | | |
| A4 bỏ graph traversal | | | |
| A5 bỏ rerank | | | |
| A6 bỏ context budget | | | |
| A7 bỏ fusion | | | |

### 5.3 Biểu đồ

- Cột nhóm: accuracy theo từng hệ thống × từng nhóm câu hỏi (có error bar = CI 95%).
- Đường: groundedness vs hallucination rate.
- Heatmap ablation (Δ so với Ours).
- Phân bố độ dài evidence (chứng minh context budget hoạt động).

### 5.4 Phân tích lỗi

Phân loại lỗi tự động (dựa trên log pipeline) + xác nhận tay trên ~30 câu:

| Loại lỗi | Dấu hiệu trong log | Nguyên nhân gốc |
|---|---|---|
| **Entity linking sai** | anchor node sai, `confidence` thấp | Alias thiếu, tên trùng (hai nhân vật trùng tên) |
| **Đường đi sai** | path hit = 0 dù có chunk đúng | Ontology thiếu quan hệ cần thiết, path scoring chưa tốt |
| **Thiếu evidence** | gold chunk không có trong top-k | Chunk quá dài, embedding kém cho câu hỏi lịch sử |
| **Sinh sai dù có evidence** | evidence đúng nhưng đáp án sai | Prompt không ràng buộc đủ, context quá dài |
| **Judge sai** | người đánh giá và judge lệch nhau | Rubric chưa rõ, đáp án tham chiếu chưa sắc |
| **Câu hỏi ngoài KG** | hệ thống bịa | Thiếu cơ chế từ chối |

Kết quả phân tích lỗi là phần **quan trọng nhất** của báo cáo: cho thấy điểm mạnh/yếu và hướng cải thiện.

## 6. Rủi ro của thiết kế đánh giá

| Rủi ro | Xử lý |
|---|---|
| Câu hỏi do Gemini sinh → "điều khoản" mà hệ thống học ngược được | Tách `eval-exclusive` docs + human verify + kiểm tra câu hỏi không chứa từ khóa trùng nguyên văn câu trả lời |
| Chỉ 180 câu → nhiễu thống kê lớn | Báo cáo CI rộng, dùng paired test, tăng lên ~300 câu nếu còn thời gian |
| LLM-judge thiên lệch về câu trả lời dài/dài dòng | Rubric cố định + kèm gold facts + kiểm tra chéo bằng token-F1 |
| Đáp án tham chiếu từ History Chat bị sai | Bắt buộc human verify trước khi đưa vào bộ đánh giá |
| Chênh lệch giữa judge và người đánh giá | Lấy mẫu 30 câu để đo độ đồng thuận, báo cáo trong bài |
