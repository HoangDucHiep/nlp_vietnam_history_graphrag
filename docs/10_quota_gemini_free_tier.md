# Ghi chú chẩn đoán quota Gemini (2026-11-02)

## Tóm tắt

Quota free tier của Google AI Studio **không phải** giới hạn "số request/phút"
đơn thuần. Với model Gemma, quota thực sự là:

```
quotaMetric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count
quotaId:     GenerateContentInputTokensPerModelPerMinute-FreeTier
quotaValue:  16000
quotaDimensions: { model: gemma-4-26b, location: global }
```

Tức là **16 000 input token / phút / model**. Mỗi lời gọi ~800 token ⇒
khoảng **20 lời gọi/phút**. Với mỗi câu hỏi cần ~3 lời gọi (hiểu câu hỏi,
sinh đáp án, chấm điểm) ⇒ **~7 câu/phút** ⇒ 1650 dòng (150 câu × 11 hệ thống)
cần ~4 giờ nếu giãn nhịp đúng.

Ngoài quota *đầu vào* còn một giới hạn **ít ai để ý**: quota *đầu ra* của
Gemma bị **token suy luận (thinking) chiếm mất** — xem mục 2.

## 1. Vì sao lượt chạy trước bị "degraded"

1. Bộ giới hạn cũ chỉ giới hạn **số request/phút** (`model_rpm`), không giới
   hạn token. Với `rpm=28` và prompt ~800 token ⇒ 28 × 800 = 22 400 token/phút
   > 16 000 ⇒ vượt quota liên tục.
2. Khi API trả 429 kèm `RetryInfo` ngắn, client **xoay sang model dự phòng**
   (`gemini-flash-lite-latest`, `gemini-3.5-flash`, ...). Mỗi model dự phòng có
   quota **20 request/ngày** (flash) hoặc **500 request/ngày** (flash-lite) —
   quota này bị đốt sạch chỉ trong vài phút.
3. `LLMJudge._ask` bắt exception → rơi về `HeuristicJudge` (token-F1). Kết quả:
   722 dòng bị chấm bằng heuristic, trộn lẫn với chấm LLM trong cùng bảng.

## 2. Nguyên nhân thật của câu trả lời rỗng: token suy luận ăn hết ngân sách

Gemma 4 chạy một **kênh suy luận** và tính token của kênh đó vào
`max_output_tokens`.  Khi phần suy luận không kết thúc, API trả:

```
finish_reason: MAX_TOKENS
text:          ""            <- câu trả lời bị mất hoàn toàn
```

Đo đạc thực nghiệm (`scripts/_probe_thinking.py`, prompt thật của pipeline):

| Prompt | 256 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| `grounded` (có CONTEXT) | rỗng | OK (733 token suy luận) | OK | OK | OK |
| `judge` (có `response_schema`) | OK | OK | OK | OK | OK |
| `closed_book` (câu hỏi khó) | rỗng | rỗng | rỗng | rỗng | **rỗng** |

Hai kết luận quan trọng:

* **Ngân sách 256 token là chắc chắn hỏng** cho mọi prompt không ép schema.
  Đó chính là lý do các dòng của judge bị đánh dấu `judge_degraded`.
* **`response_schema` tắt hoàn toàn kênh suy luận** (`thoughts_token_count: null`).
  Với schema, toàn bộ ngân sách dành cho thân JSON ⇒ judge chạy ổn ở 256 token.

### Vì sao prompt closed-book không bao giờ kết thúc?

Cô lập từng yếu tố (`scripts/_probe_closedbook_cause.py`) cho thấy nguyên nhân
**không phải** cụm "closed-book" mà là **câu hỏi mà mô hình không biết**:

| Trường hợp | Suy luận | Kết quả |
|---|---|---|
| Câu hỏi dễ ("Thủ tướng đầu tiên…") | 203–425 token | kết thúc bình thường |
| Câu hỏi khó ("Ai sinh ra Đế Nghi?") | hết 2046 token | `MAX_TOKENS`, **rỗng** |
| Câu hỏi khó + cụm "closed-book" | hết 2046 token | `MAX_TOKENS`, **rỗng** |
| Câu hỏi khó + cụm "closed-book" + "## Suy luận" | hết 2046 token | `MAX_TOKENS`, **rỗng** |

Prompt cũ yêu cầu *"nếu không chắc chắn, hãy nói rõ"* — nhưng **không** nói
rằng *"không biết"* là một câu trả lời hợp lệ.  Mô hình cứ suy luận để tìm
một câu trả lời mà không tồn tại.

Có CONTEXT thì mọi chuyện ổn: mô hình tìm được câu trả lời trong CONTEXT và
kết thúc ở ~350–1 050 token suy luận.

## 3. Cách khắc phục

1. **`SharedTokenLimiter`** — giới hạn **token/phút**, chia sẻ giữa các tiến trình
   qua `fcntl.flock` trên `artifacts/rate_limits/<model>.tokens.json`.
2. Client đếm token chính xác bằng `count_tokens` (cache theo model+text) trước
   khi gọi, rồi xin slot từ limiter.
3. Khi API trả 429 **kèm `RetryInfo` ngắn** (quota theo phút) → **chờ rồi thử
   lại chính model đó**, KHÔNG xoay sang model dự phòng (xoay vòng làm đốt
   quota ngày của Gemini). Chỉ xoay vòng khi quota *ngày* thực sự cạn.
4. **`LLMJudge` thử lại nhiều lần** trước khi rơi về heuristic; nếu vẫn thất bại
   thì dòng đó được đánh dấu `judge_degraded` và báo cáo cảnh báo rõ ràng.
5. **Ngân sách output đủ lớn** — `llm.max_output_tokens: 4096`.  Suy luận cần
   250–1 100 token cho câu có CONTEXT, nên 2048 là mức tối thiểu và 4096 là mức
   có biên.
6. **Phát hiện và thử lại khi thinking ăn hết ngân sách** —
   `GeminiClient._is_thinking_overrun()` nhận ra `finish_reason=MAX_TOKENS` với
   `text` rỗng và **thử lại với ngân sách gấp đôi** (tối đa 8 192).  Đây là
   lưới an toàn chung cho mọi prompt, không phụ thuộc việc đã đoán đúng nguyên
   nhân hay chưa.
7. **Prompt closed-book có "lối thoát"** — `CLOSED_BOOK_SYSTEM_PROMPT` nói rõ
   rằng *"không chắc chắn / không biết"* là một câu trả lời **hợp lệ** và cấm
   suy luận dài.  Đo đạc (`scripts/_probe_closedbook_escape.py`) cho thấy biến
   thể này kết thúc ổn định ở ~700–840 token suy luận cho cả ba câu hỏi khó.
8. **Dọn cache rỗng** — `scripts/13_prune_empty_cache.py` xoá các entry có
   `text` rỗng khỏi `artifacts/llm_cache`.  Nếu không, các câu trả lời rỗng đã
   bị cache sẽ được *đọc lại* và mọi lượt chạy sau vẫn hỏng.

## 4. Đo đạc thực nghiệm (quota)

| Mô hình | Quota | Ghi chú |
|---|---|---|
| `gemma-4-26b-a4b-it` | 16 000 input token/phút | ổn định, latency ~2–25 s |
| `gemma-4-31b-it` | — | trả `500 INTERNAL` phần lớn, latency 37–63 s ⇒ không dùng |
| `gemini-3.5/3.6/3.7/3.8-flash`, `flash-latest` | 20 request/**ngày** | đã cạn |
| `gemini-3.1-flash-lite`, `3.5-flash-lite`, `flash-lite-latest` | 500 request/**ngày** | đã cạn |
| `gemini-3-flash-preview` | 20 request/ngày | không đủ dùng |
| `gemini-2.5-flash*` | — | 404 (không còn cho user mới) |
| `gemini-embedding-2` | 1 000 request/ngày | đã cạn ⇒ dùng embedder **cục bộ** 384 chiều |

`models.count_tokens` **không** tính vào quota generate_content (đã kiểm chứng:
sau khi đốt hết quota vẫn gọi được, và số token trả về khớp chính xác với
`usage_metadata.prompt_token_count`). Latency ~0.11 s ⇒ dùng được để giãn
nhịp theo token.

`thinking_config` **không** dùng được với Gemma 4 (API trả `400 INVALID_ARGUMENT:
Thinking budget is not supported for this model`), nên không thể tắt kênh suy
luận; cách duy nhất là dành đủ ngân sách cho nó.

## 4b. Prompt không sửa được vấn đề thinking của closed-book

Bốn biến thể prompt (bỏ mục "## Suy luận", yêu cầu trả lời ≤ 3 câu, rút gọn
system prompt, ép trả lời dứt câu) đều cho **cùng một kết quả**:
`thoughts_token_count = 2045`, `text = ""`, `finish_reason = MAX_TOKENS` ở ngân
sách 2048 (`scripts/_probe_prompt_variants.py`).  Kết luận: **nguyên nhân không
nằm ở wording của prompt** mà ở chỗ mô hình không biết câu trả lời và suy luận
vô hạn để tìm nó.  Cách khắc phục duy nhất là (a) `response_schema` (tắt hẳn kênh
suy luận) hoặc (b) dành ngân sách lớn hơn — không phải sửa từ ngữ.

## 5. Tham số vận hành

```yaml
llm:
  model: gemma-4-26b-a4b-it
  model_rpm:
    gemma-4-26b-a4b-it: 26      # 26 × ~800 token < 16 000
  model_tpm:
    gemma-4-26b-a4b-it: 14500   # biên an toàn ~9 %
  count_tokens: true             # đếm chính xác trước khi gọi
  max_output_tokens: 4096        # phải chứa đủ token suy luận
```

Embedder của lượt đánh giá được **ghim cố định** vào model cục bộ
(`HGRAG_EMBED_PROVIDER=local` trong `scripts/launch_eval.sh`): corpus trong
Qdrant đã được index bằng vector 384 chiều (`chunks_local`, `chunks_vhe_local`)
và quota embedding Gemini đã cạn.

## 6. Lỗi chấm điểm: bộ lọc verdict tự mâu thuẫn (2026-11-02)

### 6.1 Triệu chứng

Sau khi sửa lỗi truy xuất VHE, hệ thống **"ours"** trả lời đúng câu
`vhe_00001` (*"Đế Minh là người sinh ra Đế Nghi [E1], [E2]"*, đáp án tham chiếu
*"Đế Minh"*), nhưng báo cáo ghi **sai** (`judge_correct = false`,
`judge_degraded = true`, lý do `heuristic token-F1=0.364`).

### 6.2 Nguyên nhân

Hai lỗi độc lập trong `src/hgrag/eval/judge.py`:

**(a) Bộ lọc `_is_self_consistent` loại oan verdict.**  Bản cũ tìm chuỗi khen
(`"đúng về nghĩa"`, `"trả lời đúng"`, ...) trên **cả** chuỗi `reason`:

| Lý do thật | Chứa cụm khen? | Bản cũ | Đúng ra |
|---|---|---|---|
| *"Đáp án ứng viên **không trả lời đúng** mốc liên hệ..."* | có | loại | **giữ** (là lý do *chê*) |
| *"Thông tin ... là **đúng về nghĩa** nhưng sai chi tiết quan trọng."* | có | loại | **giữ** (bị `"nhưng"` lật ngược) |
| *"Đáp án **đúng hoàn toàn** so với đáp án tham chiếu."* (`score=0`) | có | loại | loại (đúng) |

Đo trên cache thật (1198 verdict): bản cũ loại **51** verdict, trong đó **31**
là loại oan; bản mới loại **20** verdict, trong đó **0** loại oan.

**(b) Lần chấm lại vô dụng vì `temperature=0`.**  Bản cũ chỉ đổi `task` (cache
key) mà **không** đổi prompt, nên cùng một prompt luôn cho cùng một kết quả:
ba lần thử trả về đúng ba payload giống hệt nhau.  Đo trực tiếp trên
`vhe_00001` với CONTEXT thật:

```
lần 1: {"score": 0, "correct": false, ..., "reason": "Đáp án đúng hoàn toàn so với đáp án tham chiếu..."}
lần 2: (giống hệt)
lần 3: (giống hệt)
⇒ rơi về heuristic token-F1 = 0.364 ⇒ báo cáo "sai"
```

### 6.3 Cách khắc phục

1. `_is_self_consistent` làm việc trên **mệnh đề** (`_split_clauses`), không
   phải trên cả chuỗi: phân biệt lý do *khen* với lý do *chê* bằng từ phủ định
   (`không`, `chưa`, `sai`, `thiếu`) đứng trước cụm khen, và bằng từ nối đảo
   chiều (`nhưng`, `tuy nhiên`, `song`) vô hiệu hoá mệnh đề khen.
2. Lần chấm lại **đổi prompt** (thêm `JUDGE_RETRY_HINT`: chấm `score` trước,
   viết `reason` khớp với điểm), không chỉ đổi cache key.

Sau khi sửa, verdict của `vhe_00001` là
`{"score": 2, "correct": true, "grounded": true, "supported_evidence_ids": ["E1","E2"]}`
— đúng như rubric.

### 6.4 Bài học

Bộ lọc "loại verdict mâu thuẫn" **nguy hiểm khi heuristic dự phòng chặt hơn
rubric**.  Trong trường hợp này, heuristic token-F1 yêu cầu F1 ≥ 0.75 mới coi
là đúng, trong khi rubric LLM chấm theo *ngữ nghĩa*.  Rơi về heuristic ⇒ đáp án
đúng bị báo cáo là sai.  Vì vậy bộ lọc phải **bảo thủ**: chỉ loại khi mâu thuẫn
*không nghi ngờ* được, và lần chấm lại phải thực sự khác prompt.

## 6b. Embedder không khớp corpus: mất toàn bộ VHE gold (2026-11-03)

### 6b.1 Triệu chứng

Chạy `scripts/06_ask.py` / `scripts/09_live_check.py` ngoài khung đánh giá thì
log báo:

```
WARNING hgrag.qa.pipeline: Vector search thất bại ở collection 'chunks_vhe':
  Not found: Collection `chunks_vhe` doesn't exist!
```

và câu hỏi về Trần Hưng Đạo **không tìm được câu trả lời có căn cứ** (chỉ còn
bằng chứng Wikipedia, mất phần VHE gold).

### 6b.2 Nguyên nhân

Hai lớp collection trong Qdrant:

| Collection | Số chiều | Nguồn |
|---|---|---|
| `chunks`, `chunks_vhe` | 768 | vector Gemini (`gemini-embedding-2`) |
| `chunks_local`, `chunks_vhe_local` | 384 | vector cục bộ (MiniLM đa ngôn ngữ) |

Lượt đánh giá chạy với `HGRAG_EMBED_PROVIDER=local` nên corpus **thực tế** được
đánh chỉ mục bằng vector 384 chiều.  Nhưng `build_embedder()` ở chế độ `auto`
**dò quota** Gemini trước: nếu quota còn (hoặc bộ đệm còn sống), nó trả về
embedder Gemini 768 chiều.  Pipeline sau đó tìm collection 768 chiều — vốn
**không tồn tại** trong Qdrant của lượt chạy này — nên:

* `chunks_vhe` → 404 (collection không có);
* `chunks` → có, nhưng chỉ chứa 438 đoạn Wikipedia, **không** có 4 110 câu VHE
  gold (nhóm câu hỏi chính của bộ đánh giá).

Nói cách khác: pipeline im lặng đánh đổi *toàn bộ* VHE gold lấy một vài
trăm đoạn Wikipedia.

### 6b.3 Cách khắc phục

`src/hgrag/qa/pipeline.py::_resolve_corpus()` đối chiếu **collection đang tồn tại
thật** trong Qdrant (qua `VectorStore.collection_exists` / `collection_dim` mới
thêm) với số chiều của embedder đang dùng:

1. ưu tiên nhóm collection có **cùng số chiều** với embedder hiện tại;
2. nếu không nhóm nào khớp → **đổi embedder** sang nhóm còn lại (local ↔ Gemini)
   để vector truy vấn luôn khớp corpus;
3. nếu Qdrant không kiểm tra được → giữ tên collection theo cấu hình.

Sau bản vá này log chỉ còn:

```
INFO hgrag.qa.pipeline: Vector search dùng 2 collection: chunks_local, chunks_vhe_local (dim=384)
```

### 6b.4 Bài học

Quota embedding của Gemini cạn **không đồng nghĩa** vector trong kho đã đổi.
Dò quota chỉ nói "API còn trả lời được", còn kho vector là **trạng thái đã
ghi**.  Pipeline phải lấy trạng thái đã ghi làm chuẩn, không dùng trạng thái
API làm chuẩn.

### 6b.5 `FinishReason.RECITATION`: Gemma trả body rỗng dù đã retry

Ngoài `MAX_TOKENS`, Gemma 4 còn trả `finish_reason=RECITATION` với `text` rỗng
(kênh suy luận lặp nội dung huấn luyện rồi bị chặn).  Bản vá trước đây chỉ
*thử lại* với ngân sách lớn hơn rồi trả về body rỗng cho pipeline, khiến bước
disambiguate thất bại (`LLM did not return valid JSON`).

Sửa: nếu body **vẫn rỗng sau lần thử lại** và còn model dự phòng → **chuyển
sang model tiếp theo** thay vì trả về câu rỗng
(`src/hgrag/llm/client.py`, `GeminiClient.generate`).

## 7. Cách tái lập các phép đo

```bash
python scripts/_probe_thinking.py          # ngân sách output vs. token suy luận
python scripts/_probe_closedbook_cause.py  # cô lập nguyên nhân prompt closed-book
python scripts/_probe_closedbook_escape.py # tìm biến thể prompt kết thúc được
python scripts/_probe_prompt_variants.py   # biến thể prompt: không sửa được thinking
python scripts/_probe_judge_filter.py      # đo tác động của bản vá bộ lọc verdict
python scripts/_probe_judge_retry.py        # kiểm chứng lần chấm lại đổi prompt
python scripts/validate_thinking_fix.py    # kiểm chứng trên pipeline thật
python scripts/13_prune_empty_cache.py     # dọn entry rỗng khỏi cache
python scripts/14_rejudge_degraded.py --in-place  # chấm lại dòng judge_degraded
```
