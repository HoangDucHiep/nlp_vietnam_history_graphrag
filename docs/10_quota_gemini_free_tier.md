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

## Vì sao lượt chạy trước bị "degraded"

1. Bộ giới hạn cũ chỉ giới hạn **số request/phút** (`model_rpm`), không giới
   hạn token. Với `rpm=28` và prompt ~800 token ⇒ 28 × 800 = 22 400 token/phút
   > 16 000 ⇒ vượt quota liên tục.
2. Khi API trả 429 kèm `RetryInfo` ngắn, client **xoay sang model dự phòng**
   (`gemini-flash-lite-latest`, `gemini-3.5-flash`, ...). Mỗi model dự phòng có
   quota **20 request/ngày** (flash) hoặc **500 request/ngày** (flash-lite) —
   quota này bị đốt sạch chỉ trong vài phút.
3. `LLMJudge._ask` bắt exception → rơi về `HeuristicJudge` (token-F1). Kết quả:
   722 dòng bị chấm bằng heuristic, trộn lẫn với chấm LLM trong cùng bảng.

## Đo đạc thực nghiệm

| Mô hình | Quota | Ghi chú |
|---|---|---|
| `gemma-4-26b-a4b-it` | 16 000 input token/phút | ổn định, latency ~2.1 s |
| `gemma-4-31b-it` | — | trả `500 INTERNAL` phần lớn, latency 37–63 s ⇒ không dùng |
| `gemini-3.5/3.6/3.7/3.8-flash`, `flash-latest` | 20 request/**ngày** | đã cạn |
| `gemini-3.1-flash-lite`, `3.5-flash-lite`, `flash-lite-latest` | 500 request/**ngày** | đã cạn |
| `gemini-3-flash-preview` | 20 request/ngày | không đủ dùng |
| `gemini-2.5-flash*` | — | 404 (không còn cho user mới) |

`models.count_tokens` **không** tính vào quota generate_content (đã kiểm chứng:
sau khi đốt hết quota vẫn gọi được, và số token trả về khớp chính xác với
`usage_metadata.prompt_token_count`). Latency ~0.11 s ⇒ dùng được để giãn
nhịp theo token.

## Cách khắc phục

1. `SharedTokenLimiter` — giới hạn **token/phút**, chia sẻ giữa các tiến trình
   qua `fcntl.flock` trên `artifacts/rate_limits/<model>.tokens.json`.
2. Client đếm token chính xác bằng `count_tokens` (cache theo model+text) trước
   khi gọi, rồi xin slot từ limiter.
3. Khi API trả 429 **kèm `RetryInfo` ngắn** (quota theo phút) → **chờ rồi thử
   lại chính model đó**, KHÔNG xoay sang model dự phòng (xoay vòng làm đốt
   quota ngày của Gemini). Chỉ xoay vòng khi quota *ngày* thực sự cạn.
4. `LLMJudge` thử lại nhiều lần trước khi rơi về heuristic; nếu vẫn thất bại
   thì dòng đó được đánh dấu `judge_degraded` và báo cáo cảnh báo rõ ràng.

## Tham số vận hành

```yaml
llm:
  model: gemma-4-26b-a4b-it
  model_rpm:
    gemma-4-26b-a4b-it: 26      # 26 × ~800 token < 16 000
  model_tpm:
    gemma-4-26b-a4b-it: 14000   # biên an toàn ~12 %
```
