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

### 2.1.1 Rubric chấm câu **ngoài phạm vi KB** (nguồn D)

Câu hỏi ngoài phạm vi **không có** đáp án tham chiếu, nên không thể chấm "đúng/sai"
theo nghĩa.  Tiêu chí chấm là **hành vi**:

| Hành vi của hệ thống | Điểm | Lý do |
|---|---|---|
| **Từ chối trả lời** | `2` (đúng, grounded) | Đúng đắn: không bịa nội dung ngoài KB |
| **Cố trả lời / bịa đặt** | `0` (sai, không grounded) | Lỗi nghiêm trọng nhất của hệ thống có KB |

Cơ chế nhận diện từ chối (`eval/judge.py::looks_like_refusal`) dùng danh sách cụm từ
**đặc hiệu** ("không tìm thấy đủ thông tin", "ngoài phạm vi", "không đủ căn cứ"…)
để không nhầm với câu trả lời hợp lệ có câu "chưa có thông tin nào về năm này".
Đây chính là chỗ mà baseline **closed-book (B0)** thua: không có KB thì không có
cách biết câu hỏi nằm ngoài phạm vi.

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

### 2.4 Bộ câu hỏi đã sinh (thực tế)

| Nguồn | Số câu | Ghi chú |
|---|---|---|
| A. Template từ gold relation VHE | 88 | gold path **tự động kiểm chứng**; 121 câu có gold path, 40 câu có gold chunk |
| B. Gemini sinh từ chunk Wikipedia | 30 | |
| C. Vietnam History Chat | 10 | |
| D. Out-of-KB (viết tay) | 22 | chấm theo **hành vi** (§2.1.1) |
| **Tổng** | **150** | 60 single-hop · 45 multi-hop · 23 temporal · 22 out-of-KB |

> **Lưu ý về `gold_chunk_ids` của nhóm VHE (88 câu)**: các câu hỏi template
> được sinh từ *quan hệ trong đồ thị*, nên câu trả lời nằm trong **câu VHE**
> chứ không nằm trong chunk Wikipedia.  Vì vậy `gold_chunk_ids` của nhóm này
> **rỗng**, và theo quy tắc "applicable" (§3.3) chúng **không tính vào mẫu số**
> của `chunk_recall_at_10` / `ndcg_at_10`.  Ở nhóm này, chỉ số truy xuất có ý
> nghĩa là **path hit rate** và **entity recall**.

**Gold path đã được kiểm chứng lại với KG thật**:

```bash
python scripts/10_validate_eval_paths.py   # đo tỉ lệ gold path khớp node trong Neo4j
python scripts/11_fix_eval_paths.py        # chuẩn hoá gold path theo tên node thật
```

Kết quả: **97,7% exact match** (98,67% khi so khớp chuỗi con) — trước khi chuẩn hoá
chỉ có 68,1%.  Các câu còn lại (4 câu) là node VHE chưa được nạp vào đồ thị con;
chúng vẫn được giữ trong bộ đánh giá nhưng `path_hit` của chúng không tính vào
mẫu số (xem quy tắc "applicable" ở §3.3).

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

> **Quy tắc "applicable" (quan trọng):** câu hỏi **không có** gold path /
> gold chunk không thể bị trượt đường đi / không thể bị sót chunk → những câu
> đó **không được tính vào mẫu số** của `path_hit` và `chunk_recall_at_10`
> (cờ `path_applicable` / `chunk_applicable` trong `eval/run_eval.py`).
> Nếu không áp dụng quy tắc này, chỉ số bị hạ thấp oan vì câu hỏi không có gold.
> Tương tự, **groundedness của B0 (closed-book) là không áp dụng** (không có
> context để đối chiếu) → báo cáo hiển thị "–".

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
| **B0** | LLM-only | Không truy xuất, chỉ kiến thức mô hình. Cài đặt: prompt **closed-book** riêng (`CLOSED_BOOK_SYSTEM_PROMPT`) — không CONTEXT, không citation `[E#]`. Judge chỉ chấm độ đúng (groundedness = không áp dụng) |
| **B1** | Vanilla RAG | Vector search top-k trên cùng corpus, cùng rerank, cùng prompt sinh đáp án. **Không dùng đồ thị** ← baseline bắt buộc |
| **B2** | RAG + GraphRAG không có rerank | Để tách đóng góp của rerank |
| **Ours** | Hybrid GraphRAG đầy đủ | Vector + đồ thị + rerank + fusion + context budget + entity resolution |

> B1 phải dùng **cùng chunk, cùng embedding model, cùng LLM, cùng prompt** → khác biệt là do kiến trúc.
>
> **Lưu ý khi đọc bảng**: B0 là baseline *closed-book*, nên nó **trả lời** (không
> từ chối) cho mọi câu hỏi trong phạm vi lịch sử. Ở nhóm `out_of_kb`, hành vi
> đúng là **từ chối**, nên B0 bị trừ điểm ở nhóm đó (đó chính là điểm mà một
> hệ thống có KB thắng được).

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

## 7. Vận hành khi API free tier bị giới hạn

Lần chạy đánh giá đầy đủ (150 câu × 11 hệ thống = 1 650 lượt chấm, mỗi lượt
cần 1 lời gọi sinh đáp án + 1 lời gọi chấm ≈ 3 300 lời gọi) đã vấp phải giới
hạn của Google AI Studio.  Bốn điểm đã được xử lý trong mã nguồn:

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| **Quota ngày của Gemini cạn** | Các model Gemini (flash / flash-lite) có quota **20 (flash) / 500 (lite) request mỗi ngày**; vài worker chạy song song là cạn hết trong vài phút. Lỗi 429 kèm `RetryInfo.retryDelay ≈ 3 h 30 m` | Model chính chuyển sang **Gemma** (`gemma-4-26b-a4b-it`) — bucket quota riêng, 30 request **mỗi phút**. Các model Gemini giữ làm dự phòng |
| **429 làm chậm toàn bộ lượt chạy** | Client cũ *ngủ* đúng số giây API yêu cầu rồi thử lại **cùng** model | Client giờ **xoay vòng sang model khác** (mỗi model có hạn mức RPM riêng) và **giãn nhịp chủ động** theo hạn mức `model_rpm` |
| **Nhiều worker vượt hạn mức chung** | Hạn mức RPM tính **theo project + model**, tức là *dùng chung* cho mọi tiến trình | `hgrag.llm.rate_limit.SharedRateLimiter` — mỗi worker gọi `flock` trên cùng một file nên tổng số request/phút vẫn nằm trong hạn mức |
| **Judge rơi về heuristic mà không báo** | Khi LLM judge không gọi được, `HeuristicJudge` (token-F1) được dùng thay và **không ai biết** — 722/1 075 dòng bị chấm sai kiểu | `JudgeResult.degraded` / `EvalRow.judge_degraded` được gắn cờ, đếm trong `EvalRun.meta.n_degraded` và **cảnh báo ngay trong `reports/results.md`** |

### 7.1 Ba lỗi phát hiện được ở lượt chạy đầu tiên

Lượt chạy đầu tiên (150 câu × 11 hệ thống) không cho kết quả dùng được, và
nguyên nhân **không nằm ở LLM** mà ở ba lỗi của hệ thống:

| # | Triệu chứng | Nguyên nhân gốc | Cách sửa |
|---|---|---|---|
| **1** | **98/150 câu B0 rỗng**, độ trễ 45–90 s/câu | Gemma 4 dùng *toàn bộ* `max_output_tokens` cho kênh suy luận rồi trả `finish_reason=MAX_TOKENS` với body rỗng | Chuyển B0 sang **JSON mode** (`response_schema` → API tắt thinking) + bộ phát hiện lặp `is_degenerate()` (docs/03 §7) |
| **2** | **60+ câu VHE có 0 đường đi** trong KG | Traversal chỉ đi theo chiều xuôi, trong khi cạnh `ABOUT` nối theme ← event ← agent | Thêm `_event_hub_rows()` đảo chiều các cạnh hub (docs/03 §5.1.1) |
| **3** | `InMemoryGraphStore` và `Neo4jGraphStore` **không thống nhất** về hướng cạnh | `InMemoryGraphStore.neighbors()` hoán đổi `src`/`dst` khi `uid == rel["dst"]` → `src == dst`, mất quan hệ đi vào | Cả hai backend trả **đúng hướng thật** của quan hệ (docs/03 §5.0) |

Ngoài ra còn một lỗi ở **harness** (không phải ở pipeline): khi `--systems` được
chỉ định rõ, `run_evaluation()` vẫn tự thêm cả 7 hệ thống ablation → mỗi
worker của `launch_eval.sh` chạy lại toàn bộ ablation (4 lần số câu hỏi bị đốt
quota vô ích) **và** các hệ thống ablation bị chạy với cấu hình đầy đủ nên
**bảng ablation không đo đúng thứ nó nêu**. Đã sửa ở `src/hgrag/eval/run_eval.py`
(`_answer_system`/`plan`) và `scripts/07_run_eval.py` (`--with-ablations`).

> **Quy tắc bất di bất dịch của thiết kế đánh giá**: cùng LLM, cùng prompt,
> cùng corpus cho mọi hệ thống.  Vì vậy khoá cache của LLM client chứa cả
> **tên model chính** (`_cache_key`), nên đổi model giữa hai lượt chạy không
> thể vô tình dùng lại câu trả lời của model cũ.
>
> **Nếu một lượt chấm rơi về heuristic thì phải chạy lại** — trộn điểm
> token-F1 với điểm LLM trong cùng một bảng làm mất ý nghĩa của so sánh.
> `scripts/12_merge_eval.py` đếm số dòng thiếu và số dòng bị degrade trước
> khi sinh báo cáo.

### 7.2 Bốn lỗi phát hiện ở lượt chạy thứ hai

Lượt chạy thứ hai (sau khi dựng lại đồ thị VHE) vẫn còn hai lỗi, lần này nằm ở
**tầng truy xuất** và ở **bộ chấm điểm**:

| # | Triệu chứng | Nguyên nhân gốc | Cách sửa |
|---|---|---|---|
| **4** | **42 % câu VHE template bị từ chối** dù đáp án vàng nằm trong kho | (a) `vhe_entity_uid` trộn `entity_type` vào khoá hash ⇒ cùng một mention ở hai vai trò tách thành hai node, traversal từ theme không chạm tới event của agent; (b) đường đi được dựng thành chuỗi trừu tượng `A -ABOUT-> B` nên LLM đọc như danh sách nhãn và từ chối | `vhe_entity_uid` bỏ `entity_type` khỏi khoá hash (mọi mention gộp về một node chuẩn); `GraphPath.to_natural_text()` kể lại đường đi bằng câu tiếng Việt kèm câu VHE gốc (`extra.evidence_text`) (docs/03 §5.1.2–§5.1.3) |
| **5** | Dòng `vhe_00001` trả lời **đúng** nhưng báo cáo ghi **sai** (`judge_degraded = true`) | (a) bộ lọc `_is_self_consistent` tìm chuỗi khen trên cả `reason` nên loại oan lý do *chê*; (b) lần chấm lại chỉ đổi cache key, mà `temperature=0` khiến cả ba lần thử trả về payload giống hệt nhau | Bộ lọc làm việc trên **từng mệnh đề** (nhận diện từ phủ định và từ nối đảo chiều); lần chấm lại **đổi prompt** (`JUDGE_RETRY_HINT`) (docs/10 §6) |

Lưu ý quan trọng về lỗi #5: bộ lọc "loại verdict mâu thuẫn" chỉ an toàn khi
heuristic dự phòng **không chặt hơn** rubric.  Ở đây heuristic token-F1 đòi
F1 ≥ 0.75 mới coi là đúng, trong khi rubric LLM chấm theo *ngữ nghĩa* ⇒ rơi về
heuristic đồng nghĩa với việc báo cáo sai một đáp án đúng.  Đo trên 1 198
verdict thật: bản cũ loại 51 verdict (31 loại oan), bản mới loại 20 (0 loại oan).

`scripts/14_rejudge_degraded.py` cho phép chấm lại các dòng bị degrade mà
không phải chạy lại toàn bộ đánh giá (câu trả lời đã cache nên không tốn
quota sinh đáp án).

