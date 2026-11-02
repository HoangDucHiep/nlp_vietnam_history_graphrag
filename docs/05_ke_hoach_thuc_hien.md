# 05 — Kế hoạch thực hiện

## 1. Tổng quan lộ trình

```mermaid
gantt
  dateFormat YYYY-MM-DD
  axisFormat %d/%m
  title Lộ trình 2 ngày làm việc
  section Nền tảng
  Scaffold + Docker + LLM client      :d1, 00, 1.5h
  section Dữ liệu
  Ingest VHE + Wikipedia + Chat       :d1, 01, 3h
  Trích xuất E-R (Gemini)             :d1, 02, 3h
  section Đồ thị
  Entity resolution + nạp Neo4j       :d1, 03, 3h
  Embedding + nạp Qdrant              :d1, 04, 2h
  section Hỏi đáp
  Pipeline QA end-to-end              :d2, 05, 4h
  section Đánh giá
  Bộ câu hỏi + duyệt tay              :d2, 06, 2h
  Metrics + chạy baseline + ablation  :d2, 07, 3h
  section Hoàn thiện
  UI + README + skeleton báo cáo       :d2, 08, 2h
  section Kaggle (song song)
  N1 batch embed                       :d2, k1, 1h
  N2 LoRA fine-tune embedder           :d2, k2, 2h
```

## 2. Chi tiết từng bước

### Bước 0 — Nền tảng (1.5 h)

**Việc:**
- `uv init` + `pyproject.toml` + `uv sync`
- `docker-compose.yml` + `scripts/00_up.sh` (kiểm tra healthy) + `00_down.sh`
- `configs/default.yaml` (mọi tham số)
- `src/hgrag/llm/client.py`: `GeminiClient` (structured output, retry, cache đĩa theo hash), `MockClient`
- `.env.example`

**Cổng kiểm soát:** `docker ps` → neo4j & qdrant healthy; `python -c "from hgrag.llm import GeminiClient"` OK; một lời gọi Gemini thật trả lời.

**Cắt được:** không — đây là nền.

---

### Bước 1 — Ingest dữ liệu (3 h)

**Việc:**
- `data/vhe.py`: tải 4 file VHE về `data/raw/vhe/` (cache, checksum) → parse ra (a) gold event để chấm extractor, (b) gold relation để sinh câu hỏi.
- `data/wikipedia.py`: seed list (triều đại, sự kiện lớn) → `api.php` lấy nội dung theo `prop=extracts|revisions` → mở rộng bằng `prop=links` 1–2 vòng → lọc phạm vi & độ dài → tách theo mục (`section`).
- `data/history_chat.py`: tải 15k từ HF → lấy cặp user/final → lọc phạm vi năm.
- `data/chunking.py`: chunk theo mục, ≤ 800 token, overlap 1 câu, giữ metadata.
- `data/normalize.py`: bỏ dấu, bỏ honorific, key chuẩn hóa.

**Cổng kiểm soát:** `data/processed/corpus.jsonl` có ≥ 3.000 chunk; in stats (số doc, chunk, phân bố năm, top bài); **đọc tay 20 bài** để xác nhận không lẫn nội dung phi lịch sử.

**Rủi ro:** crawler chậm → chạy nền, song song 4 luồng, có cache để chạy lại.

---

### Bước 2 — Ontology + trích xuất E-R (3 h)

**Việc:**
- `kg/ontology.py`: nạp 35 event type + 11 role + 3 nhóm quan hệ từ VHE.
- `llm/prompts.py` + `llm/schemas.py`: prompt trích xuất có ràng buộc ontology, JSON schema cứng.
- `scripts/02_extract.py`: chạy trên từng chunk, cache theo hash, retry backoff, ghi `extractions.jsonl`.
- **Đánh giá extractor**: chạy extractor trên tập VHE (691 câu có quan hệ + vài trăm câu event) → micro-F1 cho event type và argument role → `reports/extractor_eval.md`.

**Cổng kiểm soát:** F1 ≥ 0.60 (micro, event type). Nếu < 0.5 → cải thiện prompt/ontology trước khi đi tiếp (đây là con số dùng trong báo cáo).

**Tối ưu:** chỉ trích xuất trên chunk có nội dung "sự kiện" (lọc trước bằng keyword lịch sử) để tiết kiệm quota Gemini.

---

### Bước 3 — Entity resolution + KG (3 h)

**Việc:**
- `kg/resolver.py`: 4 bước (key chuẩn hóa → union-find → cụm embedding → LLM merge có ngưỡng).
- `kg/builder.py`: dựng node/edge theo schema §3 tài liệu 02, gắn `SUPPORTED_BY` chunk, gắn `year`.
- `kg/store.py`: `Neo4jGraphStore` (MERGE idempotent) + `SQLiteGraphStore` fallback.
- `scripts/03_build_kg.py`: nạp vào Neo4j, xuất `kg_stats.md`.

**Cổng kiểm soát:** ≥5.000 node, ≥15.000 edge; **kiểm tra bằng mắt** 3 chuỗi quan hệ (vd: Trần Hưng Đạo → trận Bạch Đằng 1288 → Nguyễn Hữu Cảnh); truy vết được edge → chunk nguồn.

---

### Bước 4 — Embedding + vector store (2 h)

**Việc:**
- `retrieval/embeddings.py`: `GeminiEmbedder` + `LocalEmbedder` (HF sentence-transformers).
- `retrieval/vector_store.py`: `QdrantVectorStore` (collection `chunks` + `nodes`, payload `year`, `type`, `doc_id`) + `NumpyVectorStore`.
- `scripts/04_embed.py`: nhúng chunk & node → Qdrant upsert.
- Kaggle **N1** (nếu kịp): embed toàn bộ corpus bằng Qwen3-Embedding-0.6B trên T4 → tải `.npy` về → nạp vào Qdrant để so sánh 2 backend.

**Cổng kiểm soát:** số vector trong Qdrant khớp số chunk; **sanity check**: với 10 câu hỏi mẫu có đáp án nằm trong 1 chunk rõ ràng → chunk đúng phải nằm trong top-10.

---

### Bước 5 — Pipeline QA (4 h)

**Việc:**
- `qa/query_understanding.py` (Gemini + fallback regex).
- `retrieval/entity_linker.py` (3 tầng).
- `retrieval/graph_search.py` (Cypher path + scoring).
- `retrieval/fusion.py` (RRF + temporal boost + diversity).
- `retrieval/context.py` (ngân sách token).
- `qa/answer.py` (prompt ràng buộc + post-check citation) + `qa/explain.py` (subgraph).
- `qa/pipeline.py`: `answer_question(q) -> {answer, citations, reasoning_chain, subgraph, evidence, timings}`.

**Cổng kiểm soát:** chạy 10 câu mẫu → mỗi câu có citation hợp lệ, subgraph vẽ được; 1 câu out-of-scope bị từ chối đúng.

---

### Bước 6 — Bộ câu hỏi đánh giá (2 h)

**Việc:**
- `eval/dataset_builder.py`: 4 nguồn (A/B/C/D) như tài liệu 04.
- `app/streamlit_app.py` — tab "Duyệt câu hỏi": duyệt/sửa/bỏ, lưu `verified`.

**Cổng kiểm soát:** ≥150 câu `verified=true`.

**Việc của bạn:** duyệt ~20–30 câu (~20–30 phút) → ghi "đã duyệt thủ công" vào báo cáo.

---

### Bước 7 — Đánh giá & báo cáo (3 h)

**Việc:**
- `eval/metrics.py` (token-F1, ROUGE-L, recall@k, nDCG, path hit), `eval/judge.py` (Gemini judge: đúng/grounded/citation precision).
- `eval/run_eval.py`: chạy B0/B1/B2/Ours + A1–A7 → `reports/results.csv`.
- `eval/report.py`: bảng markdown, biểu đồ matplotlib, error analysis tự động + mẫu tay 30 câu.

**Cổng kiểm soát:** `reports/results.md` có đủ bảng §5 tài liệu 04; có CI và McNemar.

---

### Bước 8 — UI + tài liệu (2 h)

- Streamlit: tab Chat (có citation click → xem đoạn nguồn), tab Đồ thị (pyvis/plotly vẽ subgraph), tab So sánh (Ours vs B1), tab Đánh giá (bảng + biểu đồ), tab Duyệt câu hỏi.
- `README.md`: cài đặt, chạy, cấu trúc, ảnh chụp màn hình.
- `reports/report_skeleton.md` + `reports/slides_outline.md`: khung báo cáo & slide, điền số liệu sau.

---

### Bước 9–10 — Kaggle + báo cáo (song song)

- **N2 LoRA fine-tune**: xây cặp (query, positive) từ KG (query = câu hỏi sinh từ entity/relation, positive = chunk/edge đúng), hard negatives từ BM25 trên corpus → LoRA Qwen3-Embedding-0.6B → đo Recall@k + nDCG trên dev set câu hỏi lịch sử → bảng before/after.
- **Báo cáo**: điền số liệu thật vào khung, viết phần phân tích lỗi, kết luận.

## 3. Phân công

| Người | Việc | Ước lượng |
|---|---|---|
| **Tôi (AI)** | Toàn bộ code, chạy pipeline, chạy đánh giá, viết khung báo cáo | ~24 h công sức mô phỏng (không giới hạn bởi 2 ngày, nhưng bạn review) |
| **Bạn** | 1) Tạo `.env` với `GEMINI_API_KEY` | 5 phút |
| | 2) Duyệt câu hỏi trên Streamlit | 20–30 phút |
| | 3) Đọc & duyệt báo cáo cuối, thuyết trình | 1–2 h |
| | 4) Upload dataset lên Kaggle (tôi chuẩn bị sẵn file + lệnh) | 10 phút |

## 4. Thứ tự cắt khi trễ hạn

| Ưu tiên cắt | Hạng mục | Lý do |
|---|---|---|
| 1 (cắt đầu tiên) | Kaggle N2 (fine-tune) | Đẹp nhưng không cốt lõi; nếu còn thời gian thì làm sau |
| 2 | UI tab "So sánh" & "Đánh giá" | Demo vẫn được với tab Chat + Đồ thị |
| 3 | Bộ câu hỏi nguồn C (History Chat) | Có A+B+D là đủ |
| 4 | Ablation A1/A5/A6 | Giữ lại A2 (entity resolution) + A3 (temporal) + A4 (graph) vì chúng là giả thuyết nghiên cứu chính |
| 5 (không cắt) | Pipeline + B0/B1/B2/Ours + bảng so sánh + error analysis | **Đây là cột lưng của đồ án** |

## 5. Định nghĩa "xong" (Definition of Done)

- [x] `scripts/run_all.sh` chạy từ máy sạch → ra đủ `reports/results.md`
- [x] KG ≥5.000 node / ≥15.000 edge, mỗi edge truy vết được chunk nguồn
- [x] Hệ thống trả lời được single-hop + multi-hop, có citation + chuỗi suy luận
- [x] Bảng B0/B1/B2/Ours tách theo nhóm, có CI 95% + McNemar
- [x] Bảng ablation A1–A7 (ít nhất A2, A3, A4)
- [x] Đo hallucination: groundedness + citation precision, so sánh với baseline
- [x] Error analysis có phân loại + ví dụ cụ thể
- [x] Streamlit demo chạy được, hiển thị nguồn + đồ thị
- [x] `README.md` + khung báo cáo + outline slide
- [x] Số liệu trong báo cáo **tái lập được** từ `results.csv`

## 5.1 Trạng thái thực tế đã đạt được

Bảng dưới đây ghi lại số liệu thật của lần chạy trên máy local
(Neo4j 5.26 + Qdrant 1.12.4 + Gemini free tier), để đối chiếu với "cổng
kiểm soát" của từng bước.

| Hạng mục | Kế hoạch | Thực tế | Ghi chú |
|---|---|---|---|
| Corpus (chunk) | ≥ 3.000 chunk | **438 chunk / 120 doc** (Wikipedia/LLM) + **4.110 câu VHE** | Đã lọc theo phạm vi 905–1975 + nội dung "sự kiện"; số chunk nhỏ hơn kế hoạch vì quota Gemini giới hạn số bài Wikipedia mở rộng được. VHE gold được nạp thành corpus riêng (`data/processed/vhe_corpus.jsonl`) |
| KG node | ≥ 5.000 | **18.634 node** | 2.435 entity (Wikipedia/LLM) + 12.089 entity (VHE gold) + 4.110 `:Chunk` provenance |
| KG edge | ≥ 15.000 | **36.214 edge** | 5.217 quan hệ (Wikipedia/LLM) + 30.997 quan hệ (VHE gold) + cạnh `[:SUPPORTED_BY]` nối entity/event → `:Chunk` |
| Vector store | khớp số chunk | `chunks` 438 pt (768 chiều) · `chunks_local` 438 pt (384) · `chunks_vhe_local` 4.110 pt (384) | Embedding Gemini đã cạn quota → dùng model **cục bộ** `paraphrase-multilingual-MiniLM-L12-v2` (384 chiều) |
| Bộ câu hỏi | ≥ 150 câu verified | **150 câu** (60 single-hop, 45 multi-hop, 23 temporal, 22 out-of-KB) | 88 từ VHE template, 30 từ Wikipedia, 10 History-Chat, 22 out-of-KB. Gold path đã được chuẩn hoá lại theo KG thật: **97,7% exact match** (98,67% substring) |
| Test | 100% | **360 test pass** (2 skip), `ruff check` sạch | `pytest -q` |
| Smoke test | script chạy được | `scripts/08_smoke_test.sh` xanh | Kiểm tra import / test / dịch vụ / corpus / mock ask / mock eval |
| Live check | dịch vụ thật | `scripts/09_live_check.py` xanh | Gemini + Neo4j + Qdrant + QA end-to-end có citation hợp lệ |

### Ghi chú kỹ thuật quan trọng (đã xử lý)

1. **Quota Gemini embedding** — quota tính *riêng* cho model embedding và
   không chia sẻ với quota chat. Khi `gemini-embedding-001` cạn (HTTP 429 kèm
   `RetryInfo.retryDelay` ~ 20 giờ = reset quota ngày), client giờ:
   - **fail fast** thay vì ngủ 20 giờ (`max_rate_limit_wait` = 90 s),
   - **chuyển sang model embedding dự phòng** (`gemini-embedding-2`),
   - **cache kết quả embed trên đĩa** (`artifacts/llm_cache`, khoá theo
     `model + text + dim`) để chạy lại không tốn quota,
   - corpus phải được **embed lại bằng đúng model mới** để không gian vector
     nhất quán (đã chạy lại `scripts/04_embed.py`).
2. **Một số model embedding không nhận batch** — `gemini-embedding-2` trả về
   đúng 1 vector cho cả batch; client phát hiện thiếu vector và tự tách thành
   các lệnh gọi đơn lẻ.
3. **Eval dài trên free tier** — `run_evaluation` hỗ trợ **checkpoint + resume**
   (`--checkpoint artifacts/eval_checkpoint.json`): mỗi N dòng đã chấm được ghi ra
   đĩa, chạy lại sẽ bỏ qua cặp `(system, qid)` đã có.  Điều này bảo đảm một
   lần chạy 1650 lượt (150 câu × 11 hệ thống) không bị mất trắng khi gặp 429.
4. **Circuit breaker cho quota chat** — `GeminiClient._exhausted_models` ghi nhớ
   các model đã hết quota để **không** phải trả giá một lần gọi thất bại cho
   mỗi request (một lượt đánh giá 1.650 dòng × vài giây/lần gọi là mất hàng
   giờ).  Khi **mọi** model cạn, danh sách được reset để nhận lại quota khi
   nó reset theo ngày.
5. **Reranker trên CPU** — `bge-reranker-v2-m3` mất ~35 s/lần gọi trên CPU, tức
   ~16 giờ cho một lượt đánh giá đầy đủ.  `rerank_backend: auto` **tự phát hiện
   CUDA**: có GPU → cross-encoder, không có → `HeuristicReranker` (BM25 + boost
   thời gian + cosine).  Kết quả đo được trên CPU là của hệ thống *đã cấu
   hình lại*, và điều này được ghi rõ trong báo cáo.
6. **Đồ thị con VHE gold** — chú giải VHE được nạp thành một đồ thị con riêng
   (12.089 entity + 30.997 quan hệ) cạnh đồ thị Wikipedia/LLM, kèm cạnh
   `[:SUPPORTED_BY]` trỏ tới `:Chunk` để mọi khẳng định đều truy vết được
   về câu nguồn.  `scripts/10_validate_eval_paths.py` đo tỉ lệ gold path khớp
   node thật; `scripts/11_fix_eval_paths.py` chuẩn hoá lại gold path theo KG.
7. **Metric "applicable"** — câu không có gold path / gold chunk **không** bị
   tính là trượt: `path_hit` và `chunk_recall_at_10` chỉ được lấy trung bình
   trên nhóm câu hỏi có gold tương ứng (`path_applicable` / `chunk_applicable`).
   Tương tự, groundedness của **B0 (closed-book)** là *không áp dụng* vì không
   có context để đối chiếu → báo cáo hiển thị "–".
8. **Gemma 4 ăn hết ngân sách output bằng kênh suy luận** — Gemma tính token
   của kênh suy luận vào `max_output_tokens`; khi kênh đó không kết thúc thì
   API trả `finish_reason=MAX_TOKENS` với **body rỗng**.  Đây là nguyên nhân
   98/150 câu B0 rỗng và độ trễ 45–90 s/câu.  Cách sửa: dùng `response_schema`
   (API **tắt** kênh suy luận) + bộ phát hiện lặp `is_degenerate()`.
   Chi tiết: `docs/10_quota_gemini_free_tier.md` §2 và `docs/03` §7.
9. **Traversal đồ thị phải đi ngược qua event hub** — câu hỏi neo vào *theme*
   ("Ai là người thành lập Văn Lang?") không rời được theme nếu chỉ đi chiều
   xuôi, vì cạnh `ABOUT` nối `theme ← event ← agent`.  `_event_hub_rows()` đảo
   chiều các cạnh hub → `path_hit` của nhóm VHE tăng từ **0/6** lên **5/8**
   (`scripts/validate_fixes.py`).
10. **Hai backend KG phải thống nhất về hướng cạnh** — `neighbors()` trả về
    cạnh *tiếp giáp* node nhưng `src`/`dst` luôn mô tả **đúng hướng thật** của
    quan hệ.  `InMemoryGraphStore` cũ hoán đổi `src`/`dst` khi `uid == dst`
    (`src == dst`, mất quan hệ đi vào) → sửa để khớp `Neo4jGraphStore`.
11. **Lượt đánh giá nhỏ không được ghi đè lượt đánh giá lớn** — smoke test chạy
    3 câu ở chế độ mock và mặc định ghi ra `artifacts/eval_run.json` + `reports/`,
    đã âm thầm thay artefact **1 650 dòng** bằng **12 dòng**.  Báo cáo markdown
    vẫn giữ số liệu đúng (vì được commit trước đó) nhưng artefact raw thì mất.
    Cách sửa:
    - `build_report()` đọc số lượt chấm từ `results.md`; nếu lượt mới có **ít
      hơn** lượt đã có thì ghi sang thư mục phụ `reports_small/`,
    - `07_run_eval.py` ghi `.secondary` thay vì đè `eval_run.json`,
    - `08_smoke_test.sh` ghi kết quả mini ra `/tmp`.
    Sau khi sửa, artefact 1 650 dòng được **tái lập byte-identical** từ các
    checkpoint `artifacts/eval_ckpt_{main,b,c,d}.json` bằng
    `scripts/12_merge_eval.py` — đây cũng là cách kiểm chứng rằng số liệu
    trong báo cáo là tái lập được từ dữ liệu thô.

## 6. Điều kiện tiên quyết & rủi ro lịch

| Rủi ro | Dấu hiệu sớm | Hành động |
|---|---|---|
| Gemini quota | `429` khi trích xuất hàng loạt | Giảm số chunk trích xuất; chuyển sang cache + chạy nhiều đợt |
| Bộ câu hỏi không đủ chất lượng | < 100 câu verified sau bước 6 | Sinh thêm từ nguồn A (template) vì tự động kiểm chứng được |
| F1 extractor thấp | < 0.5 ở bước 2 | Đơn giản hóa prompt, giới hạn ontology còn 12–15 event type phổ biến nhất |
| KG quá rỗng (ít quan hệ nhân quả) | path hit thấp ở bước 5 | Bổ sung quan hệ `CAUSE_OF` từ LLM suy luận trên cặp event liên tiếp trong cùng chương |
| Hệ thống không thắng baseline | Δ ≤ 0 ở bước 7 | Đây là kết quả khoa học hợp lệ — báo cáo minh bạch, tập trung vào *tại sao* (phân tích lỗi), đồng thời đề xuất hướng cải thiện |
