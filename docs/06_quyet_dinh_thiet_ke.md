# 06 — Nhật ký quyết định thiết kế & câu hỏi còn mở

## 1. Quyết định đã chốt (ADR)

### ADR-01 — VHE là ontology + gold, không phải corpus chính
- **Trạng thái:** đã chốt
- **Bối cảnh:** VHE chỉ phủ tới thế kỷ 16, văn bản cổ Việt gần như không khớp câu hỏi hiện đại, mention rất nhiễu.
- **Quyết định:** dùng 35 event type + 11 role của VHE làm ontology cố định; VHE gold dùng để chấm extractor và sinh câu hỏi multi-hop tự động kiểm chứng. Corpus chính = Wikipedia tiếng Việt.
- **Hệ quả:** trích xuất phải dùng LLM (không dùng NER huấn luyện sẵn); có thêm 1 chỉ số định lượng (F1 extractor) cho báo cáo.

### ADR-02 — Neo4j + Qdrant qua Docker Compose
- **Trạng thái:** đã chốt (đã xác minh image tồn tại, port trống, daemon chạy)
- **Quyết định:** `neo4j:5.26-community` (bolt 7687, web 7474, plugin APOC) + `qdrant/qdrant:v1.12.4` (6333). Fallback: `SQLiteGraphStore` + `NumpyVectorStore`.
- **Hệ quả:** cần `00_up.sh`; demo cần Docker; nhưng có multi-hop traversal & path query thật → giải thích được chuỗi suy luận.

### ADR-03 — Gemini là LLM chính
- **Trạng thái:** đã chốt
- **Bối cảnh:** free tier, context dài, structured output ổn định, không cần GPU cục bộ.
- **Hệ quả:** cần `GEMINI_API_KEY`; phải có cache đĩa + retry vì quota; `LLMClient` hỗ trợ đổi sang endpoint OpenAI-compatible (Kaggle/vLLM) nếu cần.

### ADR-04 — Có fine-tune embedding trên Kaggle
- **Trạng thái:** đã chốt (đây là hạng mục cắt đầu tiên nếu trễ)
- **Bối cảnh:** cần số liệu before/after cho phần "tác vụ nặng cần GPU"; Qwen3-Embedding-0.6B là điểm khởi đầu hợp lý (VN-MTEB: 0.6B đạt 45.87 NDCG@10, gần bge-m3 45.53).
- **Hệ quả:** cần dataset Kaggle + ~4–6 h GPU; phải cẩn thận rò rỉ (train/dev tách theo doc).

### ADR-05 — 4 nhóm câu hỏi đánh giá, có gold path
- **Trạng thái:** đã chốt
- **Lý do:** kết luận "GraphRAG > RAG thuần" chỉ có nghĩa ở nhóm multi-hop; cần `path hit rate` để đo đóng góp của đồ thị, không chỉ đo chất lượng sinh.

### ADR-06 — Không dùng "RELATE_TO" chung chung
- **Trạng thái:** đã chốt
- **Lý do:** HisGraphRAG dùng quan hệ đơn nhất kèm mô tả tự nhiên → không phân tích được theo loại, không lọc được theo kế hoạch quan hệ. Ta dùng ontology VHE + nhóm temporal/causal.

### ADR-07 — Định danh thực thể theo mô hình "QID-first" (thay đổi lớn so với bản draft đầu)
- **Trạng thái:** đã chốt — chi tiết & bằng chứng ở [`07_xac_dinh_thuc_the.md`](07_xac_dinh_thuc_the.md)
- **Bối cảnh:** bản draft đầu định thiết kế entity resolution kiểu "union-find + cụm embedding + LLM merge" (giống HisGraphRAG). Khi kiểm chứng thực tế với API, phát hiện cơ chế tốt hơn nhiều:
  - Wikipedia `prop=redirects` trả về **toàn bộ** biến thể tên (17 biến thể cho Trần Hưng Đạo, gồm cả bản gõ sai "Hứng Đạo đại vương").
  - Wikidata `skos:altLabel` cho alias đã duyệt, chính xác cao, đã lọc sẵn nhiễu.
  - `pageprops.wikibase_item` cho QID → khoá định danh ổn định vĩnh viễn.
- **Quyết định:** entity = **thực thể Wikidata có QID**; alias table xây từ (QID, Wikidata aliases, Wikipedia redirects đã kiểm chứng, key chuẩn hóa). LLM chỉ dùng để **kiểm chứng** các cặp nghi ngờ, không dùng để "merge" tự do.
- **Hệ quả:**
  - Bỏ bước LLM-merge tốn kém; thay bằng tra cứu QID gần như tuyệt đối.
  - Bổ sung bảng alias + bộ kiểm thử ~100 cặp tên → có metric **entity-linking precision/recall** độc lập cho báo cáo.
  - Thêm module `data/wikidata.py` + `kg/alias.py` vào kế hoạch triển khai.
  - Xác nhận ADR-01: VHE không làm nguồn entity (chỉ 54% mention có bài wiki, phần lớn là khớp giả do tên chung chung).

### ADR-08 — Không dùng SPARQL endpoint trong pipeline
- **Trạng thái:** đã chốt
- **Lý do:** `query.wikidata.org/sparql` đã timeout sau 40 s trong thử nghiệm; `wbgetentities` batch 50 QID/request trả đủ aliases + P31 + P27 trong 1 request → dùng cái này.

## 2. Các quyết định đã chốt ở đợt thảo luận gần nhất

| # | Câu hỏi đã hỏi | Quyết định |
|---|---|---|
| Q1 | Giữ VHE trong KG chính? | **Không** — VHE chỉ làm ontology + gold đánh giá + nguồn câu hỏi (ADR-01) |
| Q2 | Bộ câu hỏi đánh giá lấy từ đâu? | **Lai**: MCQA thi tốt nghiệp (uy tín) + câu hỏi tự do có gold path (đo multi-hop) |
| Q3 | Mức độ UI? | **Đầy đủ 5 tab** — Chat, Đồ thị, So sánh, Đánh giá, Duyệt câu hỏi |
| Q4 | Embedding backend? | **Cả hai**: Gemini mặc định + Qwen3-Embedding-0.6B (Kaggle) để so sánh & fine-tune |
| Q5 | Có đưa SGK THPT vào corpus? | **Không** — bỏ qua OCR, tiết kiệm thời gian |

### Q1. Phạm vi KG: có giữ VHE trong KG chính không?
- **Phương án A (hiện tại):** VHE chỉ dùng làm gold đánh giá + nguồn câu hỏi; KG chính xây từ Wikipedia.
- **Phương án B:** nhập VHE (event + argument + temporal/causal) vào KG như một subgraph `source=vhe` → KG giàu quan hệ nhân quả/temporal hơn, nhưng chứa văn bản cổ và node nhiễu (`vua`, `quân`), có thể làm nhiễu truy xuất.
- **Cần quyết định:** có chấp nhận nhiễu để đổi lấy quan hệ nhân quả nhiều hơn không?
- **Gợi ý:** thử B ở dạng thử nghiệm nhỏ — nếu path hit rate tăng mà precision giảm nhẹ thì giữ B với `confidence` thấp và chỉ dùng để giải thích, không dùng làm evidence chính.

### Q2. Bộ câu hỏi đánh giá lấy từ đâu cho "đúng chuẩn"?
- **Phương án A:** sinh tự động (template VHE + Gemini) + human duyệt ~30 câu. Nhanh nhưng chất lượng đáp án tham chiếu phụ thuộc Gemini.
- **Phương án B:** dùng câu hỏi thi tốt nghiệp THPT (có đáp án chính thức, khách quan) → uy tín cao nhưng là MCQA nên không đo được multi-hop tự do.
- **Phương án C (đề xuất):** **lai** — MCQA thi tốt nghiệp làm phần "độ chính xác kiến thức" (uy tín, có sẵn đáp án) + bộ câu hỏi tự do có gold path làm phần "multi-hop & giải thích".
- **Cần quyết định:** có đưa MCQA vào bộ đánh giá không, và lấy từ nguồn nào (đề thi các năm nào).

### Q3. Mức độ đầu tư cho UI
- **Phương án A:** Streamlit tối giản — ô nhập, câu trả lời, danh sách nguồn, đồ thị subgraph. (đủ để demo, ~2 h)
- **Phương án B:** thêm tab so sánh Ours vs B1 trên cùng câu, tab bảng điểm với biểu đồ, tab duyệt câu hỏi tích hợp. (~4 h, trải nghiệm demo tốt hơn nhiều)
- **Gợi ý:** phương án B, vì chính tab "so sánh" và "bảng điểm" là thứ khiến giảng viên tin rằng hệ thống thực sự có số liệu.

### Q4. Embedding backend cho hệ thống chính
- **Phương án A:** Gemini `gemini-embedding-001` (đơn giản, chạy local, không GPU).
- **Phương án B:** Qwen3-Embedding-0.6B từ Kaggle (miễn phí, mở, có thể fine-tune → khớp ADR-04).
- **Gợi ý:** dùng **cả hai**: Gemini cho hệ thống chạy mặc định (đơn giản, tái lập được không cần Kaggle), Qwen3 fine-tuned cho phần "cải thiện trên domain" và so sánh trong báo cáo.

### Q5. Có cần thêm nguồn dữ liệu nữa không?
- Wikipedia tiếng Việt đã đủ cho phạm vi 905–1975, nhưng có thể thêm:
  - `UVW-2026` (Wikipedia + Wikidata QID) → tăng chất lượng entity linking, có QID để đối chiếu.
  - SGK lịch sử THPT (Cánh diều) → văn bản sạch, có cấu trúc chương mục (nhưng bản PDF kèm hình → cần OCR/VLM, tốn thời gian).
- **Cần quyết định:** có đưa SGK vào không (cân nhắc thời gian OCR).

## 3. Những điểm còn mờ trong thiết kế (đã cập nhật sau đợt thảo luận)

| # | Điểm mờ | Trạng thái | Ghi chú |
|---|---|---|---|
| M1 | **Định nghĩa "entity" trong KG**: lấy từ mention VHE hay từ LLM trích Wikipedia? | ✅ **Đã chốt** | Entity = thực thể **Wikidata có QID**; VHE không làm nguồn entity. Xem [`07_xac_dinh_thuc_the.md`](07_xac_dinh_thuc_the.md) §4, §6 |
| M2 | **Chuỗi suy luận sinh từ đâu** | ✅ **Đã chốt** | Lấy trực tiếp từ `graph_path` evidence (đã resolve về node QID) → mỗi bước có định danh chuẩn + nguồn; LLM chỉ viết lại câu chữ |
| M3 | **Có cho LLM "bịa" quan hệ nhân quả không?** | ✅ **Đã chốt** | Có, nhưng chỉ giữa **hai node đã có QID**, gắn `confidence` thấp hơn, mỗi edge vẫn truy vết được về chunk nguồn |
| M4 | **Định nghĩa multi-hop**: 2 hay 3 bước? | ⚠️ **Còn mở** | Đề xuất: **≥2 bước qua đồ thị** = multi-hop; thống kê riêng 2-hop và 3+-hop trong báo cáo |
| M5 | **Ngân sách context 6.000 token** | ⚠️ **Còn mở** | Đề xuất: thử 3 mức (2k/6k/12k) như một ablation (A6) — rẻ và có giá trị học thuật |
| M6 | **Có đánh giá B0 trên câu hỏi open-ended?** | ⚠️ **Còn mở** | Đề xuất: **có** — B0 trên open-ended rất dễ bịa, cho thấy giá trị của RAG rõ nhất |
| M7 | **Xử lý entity trùng tên** | ✅ **Đã chốt** | QID + P31/P27 + category lịch sử + bài phân giải ngoặc + ràng buộc triều đại/thời gian. Xem [`07_xac_dinh_thuc_the.md`](07_xac_dinh_thuc_the.md) §4 |

### 3.1 Các nhóm vấn đề khác đã được xử lý trong thiết kế

| Vấn đề | Cách xử lý | Tài liệu |
|---|---|---|
| Tên nhập nhằng (Trần Hưng Đạo / Hưng Đạo Vương / Trần Quốc Tuấn / Hứng Đạo đại vương) | 5 tầng: QID → Wikidata alias → Wikipedia redirect (có kiểm chứng) → key chuẩn hóa → embedding+LLM | [`07`](07_xac_dinh_thuc_the.md) §5 |
| Chữ Đ/Ð cũ, bỏ dấu, viết thường/thường hoa, honorific | Chuẩn hóa 6 bước + index cả chuỗi gốc | [`07`](07_xac_dinh_thuc_the.md) §5 T4 |
| Redirect Wikipedia sai (Thụy Ba Công chúa → Trần Hưng Đạo) | Lớp kiểm chứng LLM với tiêu chí "cùng triều đại + cùng thời gian" | [`07`](07_xac_dinh_thuc_the.md) §3, §5 T2 |
| Trùng nhãn (Hùng Vương: người / giai đoạn / phường) | Lọc P31 + P27 + category lịch sử + bài phân giải ngoặc | [`07`](07_xac_dinh_thuc_the.md) §4 |
| Mention VHE nhiễu (`vua` 416, `mùa xuân` 195) | Chỉ giữ mention có từ khóa riêng, đánh dấu `confidence=0.7`, `source=vhe` | [`07`](07_xac_dinh_thuc_the.md) §6 |
| Văn bản cổ VHE không khớp câu hỏi hiện đại | VHE chỉ làm gold, không làm corpus | ADR-01 |
| Prompt bị nhiễu làm giảm accuracy | Context budget + giới hạn ≤3 evidence/doc + diversity | [`03`](03_thuat_toan_truy_xuat.md) §6, ablation A6 |
| Mất ngữ cảnh thời gian | Mọi event có `year`; temporal filter + boost; payload filter Qdrant | [`03`](03_thuat_toan_truy_xuat.md) §5–6 |
| Hệ thống bịa khi không đủ thông tin | Cơ chế từ chối trả lời + post-check citation + 20 câu out-of-KB | [`03`](03_thuat_toan_truy_xuat.md) §9 |

## 4. Nhật ký trao đổi (tóm tắt các quyết định đã bàn)

| # | Quyết định | Chốt bởi |
|---|---|---|
| 1 | LLM: Gemini (free tier) | bạn |
| 2 | Dữ liệu: VHE + Wikipedia tiếng Việt (Wikipedia là corpus chính) | bạn |
| 3 | Phạm vi GraphRAG: thiết kế riêng (không dùng sẵn Microsoft GraphRAG) | bạn |
| 4 | Thời gian: ~2 ngày | bạn |
| 5 | Có GEMINI_API_KEY, sẽ tạo `.env` | bạn |
| 6 | Kaggle: embedding + fine-tune LoRA | bạn |
| 7 | Hạ tầng: Docker (Neo4j + Qdrant) | bạn (nhắc máy có Docker) |
