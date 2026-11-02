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

## 6. Cách tái lập các phép đo

```bash
python scripts/_probe_thinking.py          # ngân sách output vs. token suy luận
python scripts/_probe_closedbook_cause.py  # cô lập nguyên nhân prompt closed-book
python scripts/_probe_closedbook_escape.py # tìm biến thể prompt kết thúc được
python scripts/validate_thinking_fix.py    # kiểm chứng trên pipeline thật
python scripts/13_prune_empty_cache.py     # dọn entry rỗng khỏi cache
```
