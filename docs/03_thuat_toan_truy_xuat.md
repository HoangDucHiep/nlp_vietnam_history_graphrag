# 03 — Thuật toán truy xuất & sinh câu trả lời

## 1. Tổng quan pipeline truy vấn

```mermaid
sequenceDiagram
  participant U as Người dùng
  participant Q as QueryUnderstanding
  participant E as EntityLinker
  participant V as VectorStore
  participant G as GraphSearch
  participant F as Fusion
  participant C as Context
  participant A as Answer
  U->>Q: câu hỏi tiếng Việt
  Q-->>E: {entities, time_refs, relation_plan, expected_hops, intent}
  E->>V: tìm node ứng viên theo embedding
  E->>G: anchor nodes + relation_plan + hops
  par vector
    V-->>F: top-k chunks (rank)
  and graph
    G-->>F: đường đi quan hệ (rank + chuỗi)
  end
  F->>F: RRF + boost thời gian
  F->>C: ứng viên evidence
  C->>A: context đã cắt ngân sách token
  A-->>U: câu trả lời + [E#] + chuỗi suy luận + subgraph
```

## 2. Phân tích câu hỏi (Query Understanding)

Gemini, **JSON schema cứng**, output:

```json
{
  "intent": "multi_hop",
  "entities": ["Ngô Quyền", "Trần Hưng Đạo"],
  "time_refs": [938, 1288],
  "relation_plan": ["PRECEDED_BY", "PARTICIPATED_IN"],
  "expected_hops": 2,
  "answer_type": "event | person | place | year | cause | comparison | explanation",
  "needs_temporal_filter": true
}
```

- `intent` ∈ {`single_hop`, `multi_hop`, `temporal`, `comparison`, `out_of_scope`}.
- `expected_hops` quyết định độ sâu traversal (1–3), có cap cứng.
- **Quy tắc an toàn**: nếu `out_of_scope` (không liên quan lịch sử VN 905–1975) → trả "không thuộc phạm vi" mà không gọi LLM sinh đáp án.
- **Fallback không cần LLM**: regex năm (`\b(9|1[0-9])\d{2}\b`) và so khớp tên trong alias table → dùng khi Gemini lỗi/quota.

## 3. Entity Linking

Tầng chuẩn hoá (tên gọi trong `kg/alias.py`), từ rẻ → đắt, dừng sớm khi đủ tin cậy:

| Tầng | Cách làm | Điểm |
|---|---|---|
| **T0** | mention **là QID** (`Q123`) hoặc node đã biết trong KG | `1.00` |
| **T1** | alias Wikidata (`wbsearchentities`, altLabel) | `0.97` |
| **T2** | redirect Wikipedia (`tier 2`) | `0.93` |
| **T3** | redirect Wikipedia **xác minh bằng LLM** | `0.90` |
| **T4** | khớp `name_key` đã chuẩn hoá (bỏ dấu, bỏ honorific) | `0.85` |
| **T5** | embedding + LLM disambiguation (top-k ứng viên) | `0.60–0.90` |

- `kg/alias.py` chịu trách nhiệm **6 tầng QID-first** (ADR-07) và sinh *nhiều* ứng viên.
- `retrieval/entity_linker.py` là lớp vỏ của tầng QA: mỗi mention chỉ giữ **một anchor tốt nhất**, có ngưỡng cứng `entity_link_min_confidence = 0.7`; không khớp được thì trả danh sách **rỗng** (không bịa entity).
- Cờ ablation: `use_resolution=False` chỉ giữ **T0** → dùng cho ablation A1 (bỏ entity linking) và A2 (bỏ entity resolution).
- `entity_linker` còn trích xuất mention tiếng Việt bằng regex + danh sách stopword (`VIETNAMESE_STOPWORDS`) để pipeline vẫn chạy được khi bước Query Understanding không có kết quả.

> **Lưu ý thực tế**: bảng alias của hệ thống chạy chủ yếu ở **T0/T4** (node đã có trong KG + khớp tên đã chuẩn hoá); các tầng T1–T3 cần mạng và bị bỏ qua khi offline. Vì vậy ablation A1 cho thấy tác động của *tầng khớp mờ*, còn A2 cho thấy tác động của việc gộp node trùng.
>
> **Chi phí của tầng T5**: tầng này cần vector của *mọi* entity trong bảng alias
> (~14 000 entity).  Trên CPU, việc embed 441 batch mất ~4 phút — và phải
> lặp lại ở mỗi lần khởi động của harness đánh giá.  Vì vậy vector được
> cache trên đĩa tại `artifacts/alias_embeddings.json`
> (`AliasResolver.enable_embed_cache` / `save_embed_cache`): lần đầu tiên
> embed rồi ghi đĩa, các lần sau nạp lại trong ~1 giây.

## 4. Truy xuất vector

- Nhúng câu hỏi bằng embedder đang cấu hình (Gemini `gemini-embedding-2` 768 chiều, hoặc model **cục bộ** 384 chiều `paraphrase-multilingual-MiniLM-L12-v2` khi quota Gemini cạn).
- Hai collection Qdrant được tìm **song song** rồi gộp:
  - `chunks` (Wikipedia/LLM, 768 dim) và `chunks_vhe_local` (chú giải VHE, 384 dim) → nhúng câu hỏi vào **cả hai** collection tương ứng.
- `search` cosine với `limit = vector_top_k = 30`; payload filter tùy chọn `year ∈ [năm-20, năm+20]` khi câu hỏi có năm rõ ràng (bộ lọc được dịch sang `Filter(should=…)` của Qdrant 1.12).
- **Rerank**: `retrieval/rerank.py` có 2 backend:
  - `cross-encoder` (`BAAI/bge-reranker-v2-m3`) — chính xác hơn nhưng **~35 s/lần gọi trên CPU**;
  - `heuristic` — BM25 + boost thời gian + cosine similarity, chạy được trên CPU.
  - `rerank_backend: auto` (mặc định) **tự phát hiện CUDA**: có GPU → cross-encoder, không có → heuristic. Đây là lựa chọn bắt buộc cho lượt đánh giá 150 câu × 11 cấu hình trên máy CPU.
  - LLM-rerank (Gemini chọn top-k trong 1 lời gọi) là đường dự phòng khi cả hai backend trên không khả dụng.

## 5. Truy xuất đồ thị (phần lõi)

### 5.0 Hướng của cạnh: quy ước bắt buộc

`GraphStore.neighbors()` trả về **các cạnh *tiếp giáp* với node** (theo cả hai
chiều), nhưng `src`/`dst` **luôn mô tả đúng hướng thật của quan hệ** — chúng
không bao giờ bị hoán đổi để cho `uid` luôn là nguồn. Nhờ vậy phía trên
(`retrieval/graph_search.py`) phân biệt được cạnh *đi vào* với cạnh *đi ra*:

| Backend | Cơ chế |
|---|---|
| `Neo4jGraphStore` | `MATCH (a)-[r]-(b)` + `WITH startNode(r) AS s, endNode(r) AS e` → `src = s`, `dst = e` |
| `InMemoryGraphStore` | `rel["src"]` / `rel["dst"]` nguyên vẹn |

> **Lỗi đã sửa (2026-11-02)**: bản `InMemoryGraphStore` cũ gán
> `dst = entities[other]` với `other = rel["src"]` khi `uid == rel["dst"]`, khiến
> `src == dst` và **mất toàn bộ quan hệ đi vào**. Hậu quả: hai backend không
> thống nhất về hướng cạnh và traversal kiểu "event hub" (mục 5.1.1) không
> dựng được trên bộ nhớ. Test `tests/test_retrieval.py::test_graph_searcher_theme_path_direction`
> chốt lại quy ước này.

### 5.1 Truy vấn Cypher

```cypher
MATCH (a:Entity {uid: $uid})-[r]->(b:Entity)
RETURN a{.*} AS src, type(r) AS rel, properties(r) AS rprops, b{.*} AS dst
LIMIT 500
```

Traversal **có hướng** (`-[r]->-`) được thực hiện từng bước trong Python (`GraphSearcher._traverse`), không dùng `apoc.path.expandConfig` vì biến độ dài (`$maxHops`) không được Cypher cho phép. Mỗi bước:

1. lấy **hàng xuất** của node (giữ đúng hướng quan hệ — `neighbors()` không có hướng sẽ sinh ra các path sai như `A -CAUSE_OF-> B` khi thực tế là `B -CAUSE_OF-> A`);
2. giữ lại cạnh có `rel_type` nằm trong `relation_plan` (nếu có) — cạnh ngoài kế hoạch vẫn được phép nhưng bị chấm điểm thấp ở bước 5.2;
3. dừng ở `graph_max_hops = 3` (cap cứng), tối đa `graph_max_paths_per_hop = 30` path mỗi bước;
4. chống "hub": node có bậc quá cao bị loại khỏi việc mở rộng tiếp (`_degree_penalty`).

### 5.1.1 Đi ngược qua "event hub" (bắt buộc với câu hỏi neo vào theme)

KG mô hình sự kiện là **node riêng**:

```
agent -[PARTICIPATED_IN]-> event -[ABOUT]-> theme
                        └-[LOCATED_IN]-> place
```

Câu hỏi dạng VHE thường neo vào **theme** ("Ai là người thành lập Văn Lang?")
nhưng đáp án là **agent** của sự kiện `ABOUT` theme đó. Vì cạnh `ABOUT` đi theo
hướng ngược lại (theme ← event ← agent), một traversal chỉ đi theo chiều xuôi
**không bao giờ rời được node theme** — đó là lý do **60+ câu VHE có 0 đường đi**
trong lượt đánh giá đầu tiên.

`GraphSearcher._event_hub_rows()` giải quyết việc này: nó lấy các cạnh **đi vào**
node hiện tại thuộc nhóm quan hệ hub

\[
\text{BIDIRECTIONAL} = \{\text{ABOUT, LOCATED\_IN, OCCURRED\_IN, OCCURRED\_AT, HAPPENED\_AT}\} \cup \{\text{PARTICIPATED\_IN, AGENT\_OF, INVOLVES}\}
\]

rồi **đảo chiều** chúng (`theme → event`, `event → agent`) để traversal đi tiếp
được theo chiều xuôi. Cạnh đảo giữ nguyên thuộc tính của cạnh gốc — đặc biệt
`source_chunk_ids`, tức là câu VHE mà sự kiện được trích xuất từ.

Quan hệ `SUPPORTED_BY` **không** bị đảo: nó là quan hệ *provenance*, không phải
quan hệ *suy luận*.

Cơ chế này được kiểm chứng bằng `scripts/validate_fixes.py`:

| Câu hỏi | Số path | `path_hit` |
|---|---|---|
| `vhe_00001` "Ai là người sinh ra Đế Nghi?" | 13 | ✓ |
| `vhe_00004` "Ai là người sinh ra Lạc Long Quân?" | 20 | ✓ |
| `vhe_00006` "Ai là người thành lập Văn Lang?" | 20 | ✓ |
| `vhe_00007` "Trong sự kiện do Hùng Vương thực hiện…" | 1 | ✗ (thiếu node sự kiện trong KG) |

Trước khi sửa: `path_hit = 0/6`; sau khi sửa: `path_hit = 5/8`.

### 5.1.2 Định danh thực thể VHE: uid không chứa *loại* (bắt buộc)

Cơ chế event hub chỉ chạy được nếu **anchor của câu hỏi và node trong sự kiện
là cùng một node**. VHE gold thường nhắc cùng một thực thể với **hai vai trò
khác nhau**:

- `Sơn Tinh` là **agent** của sự kiện `Sơn Tinh gặp gỡ` (loại `Person`);
- `Sơn Tinh` là **theme** của sự kiện `Mị Nương cưới Sơn Tinh` (loại `Entity`).

Bản cũ sinh uid bằng `stable_uid(name_key(mention), entity_type)` — tức **loại
thực thể nằm trong khoá** — nên hai mention hoá thành **hai node khác nhau**.
Hậu quả đo được trên toàn bộ đồ thị VHE:

| Chỉ số | Trước | Sau |
|---|---|---|
| Node VHE (không tính event/chunk) | 9 028 | **6 209** |
| Tên có > 1 node | 1 056 | **0** |
| Node bị tách đôi | 2 819 | **0** |
| Quan hệ chạm node bị tách | 12 914 | — |

Với đồ thị bị tách, câu "Ai là người cưới Sơn Tinh?" neo vào node `Person`,
trong khi đáp án nằm sau node `Entity`: traversal **không chạm tới** sự kiện vàng
→ context không có đáp án → mô hình từ chối trả lời. Đo trên 88 câu VHE
template: **42 % bị từ chối**, và lỗi lớn nhất trong `error_analysis.md` là
nhóm *"Từ chối nhầm câu trả lời được"* (74 %).

Sau khi bỏ `entity_type` khỏi khoá uid (`vhe_entity_uid`), mọi mention trùng
`tên` gộp làm một node; các loại quan sát được vẫn được lưu ở
`extra.entity_types` để tầng truy xuất dùng làm gợi ý. Đo lại
(`scripts/_probe_vhe_reach_offline.py`): **84/88 = 95,5 %** câu VHE template
tới được node vàng của sự kiện (4 câu còn lại là lỗi **entity linking**:
"Ngô" bị neo nhầm vào node Wikipedia thay vì "vua Ngô").

> Test chốt lại quy tắc: `tests/test_kg.py::test_vhe_entity_uid_ignores_entity_type`
> và `::test_import_vhe_gold_merges_the_roles_of_one_mention`.

### 5.1.3 Đường đi phải **đọc được** (bắt buộc với bộ sinh câu trả lời)

Sau khi traversal chạm tới sự kiện vàng, đường đi được đưa vào context dưới
dạng chuỗi quan hệ:

```
Nam Hải -ABOUT-> Nhà Tần thành lập Nam Hải -ABOUT-> Tượng Quận -ABOUT-> gặp gỡ
```

Chuỗi này **quá mờ** với bộ sinh câu trả lời: nó đọc được như danh sách nhãn
quan hệ, không cho biết *ai* làm *gì*, nên mô hình kết luận "không đủ thông
tin" và từ chối — dù đáp án ("Nhà Tần") nằm ngay trong chuỗi. Đo lại: cả ba câu
`vhe_00008` / `vhe_00012` / `vhe_00010` đều bị từ chối với lý do này.

`GraphPath.to_natural_text()` (dùng bởi `fusion.fuse()` khi dựng evidence) vì
vậy trình bày đường đi thành **câu có nghĩa**, kèm **câu nguồn** của mỗi cạnh
(cảnh giác lấy từ `extra.evidence_text` của quan hệ — VHE gold lưu câu gốc
của sự kiện ở đó):

```
Sơn Tinh - ABOUT -> Mị Nương cưới Sơn Tinh
Nguồn: Mị Nương đã lấy Sơn Tinh, Thục Vương tức giận, dặn lại con cháu phải diệt Văn Lang mà chiếm lấy nước.
```

Sau khi sửa, cả ba câu trả lời đúng:

| Câu hỏi | Trước | Sau |
|---|---|---|
| `vhe_00008` "Ai là người cưới Sơn Tinh?" | từ chối | "Mị Nương là người đã lấy Sơn Tinh [E4]" |
| `vhe_00012` "Ai là người thành lập Nam Hải?" | từ chối | "Nhà Tần thành lập Nam Hải [E2], [E4], [E6]" |
| `vhe_00010` "Ai là người tấn công Nam Chiếu?" | từ chối | "… Cao Vương [E4]" |

> Test chốt lại quy tắc: `tests/test_retrieval.py::test_graph_path_natural_text_quotes_the_evidence`
> và `::test_graph_path_edge_carries_the_evidence_sentence`.

### 5.1.4 Diversity cap không được loại đường đi đồ thị (bắt buộc)

`apply_diversity_cap` giới hạn ≤ `max_evidence_per_doc = 3` evidence cho mỗi
`doc_id` để tránh 5 đoạn về cùng một bài.  Nhưng **đường đi đồ thị không phải
đoạn văn** nên không có `doc_id`: chúng rơi vào cùng một nhóm giả `"__nodoc__"`,
và chỉ **ba đường đi đầu tiên** lọt vào context dù traversal tìm ra nhiều hơn.
Đường đi chứa **đáp án vàng** có thể rơi ngoài ba đường đi đó — tức là hệ
thống *đã tìm thấy* đáp án rồi lại bỏ nó khỏi context.

Đo trên `vhe_00013` ("Trong sự kiện do Nhà Tần thực hiện, đối tượng được
nhắc đến là gì?", đáp án vàng **Nam Hải**):

| | evidence | đường đi đồ thị | đáp án vàng trong context |
|---|---|---|---|
| trước | 6 | 3 (E2, E6, E8) | **không** |
| sau | 12 | 6 (E2…E12) | **có** |

Hai thay đổi trong `retrieval/fusion.py`:

1. **`_diversity_key()`** tách đường đi đồ thị ra khỏi nhóm chung: mỗi đường
   đi đếm vào nhóm riêng, khoá bằng chuỗi `uid` của các node ⇒ cap vẫn giới
   hạn đoạn văn trùng lặp mà không đụng tới đường đi.
2. **`dedupe_by_text()`** loại các mục có **text trùng nhau**: traversal có thể
   đi tới cùng một chuỗi cạnh qua nhiều đường (`Nhà Tần - ABOUT -> giải thể`
   xuất hiện hai lần).  Lặp lại câu giống nhau chỉ tốn ngân sách context và
   khiến LLM tưởng sự kiện được nói hai lần.

> Test chốt lại quy tắc: `tests/test_retrieval.py::test_diversity_cap_does_not_evict_the_graph_paths`
> và `::test_dedupe_by_text_drops_identical_items`.

### 5.2 Chấm điểm đường đi

\[
S(\text{path}) = \underbrace{w_a \cdot \text{affinity}(\text{rel\_types}, \text{relation\_plan})}_{\text{đúng kế hoạch}} + \underbrace{w_s \cdot \text{sim}(q, \text{path\_text})}_{\text{ngữ nghĩa}} + \underbrace{w_t \cdot \text{temporal\_overlap}}_{\text{trùng thời gian}} - \underbrace{w_d \cdot \text{deg\_penalty}}_{\text{node độ bậc cao}}
\]

Trọng số nằm trong `PATH_SCORE_WEIGHTS` (`retrieval/graph_search.py`).

- `affinity`: mỗi cạnh được chấm `1.0` nếu `rel_type` có trong `relation_plan`; `0.9` nếu là quan hệ nhân quả/temporal mà câu hỏi có *cue* tương ứng ("vì sao", "trước/sau"); `0.5` khi câu hỏi trung tính; `0.0` nếu lệch kế hoạch. Lấy trung bình các cạnh.
- `sim`: cosine giữa embedding câu hỏi và embedding chuỗi mô tả path. **Tối ưu quota**: `_path_similarities` nhúng *tất cả* path trong **một** lệnh gọi embedding thay vì mỗi path một lệnh.
- `temporal_overlap`: có cạnh/năm trùng `year_refs` của câu hỏi.
- `deg_penalty`: chống node hub ("Nhà Lý" nối hàng trăm cạnh) làm nhiễu.

### 5.3 Đường đi được dùng làm evidence

Mỗi đường đi → 1 evidence với:
- `type = "graph_path"`
- `text` = mô tả tự nhiên của chuỗi triple (dùng cho LLM và để hiển thị)
- `provenance` = danh sách `chunk_id` của các cạnh `SUPPORTED_BY` + `edge_uids`

→ Đây là cơ chế **"chuỗi liên kết logic"** mà yêu cầu 3.4 nói tới.

## 6. Fusion (RRF) và ngân sách context

- **RRF**: \(score(d) = \sum_{list} \frac{1}{k + rank_{list}(d)}\), `k = rrf_k = 60`, giữa list `vector` và list `graph`. RRF chỉ cần *thứ hạng* nên hợp nhất được list cosine (vector) với list điểm path (graph) vốn khác thang đo.
- **Boost thời gian**: `+ temporal_boost = 0.15` nếu evidence chứa năm trùng câu hỏi (dùng `year_refs` của Query Understanding).
- **Ưu tiên đa dạng**: giới hạn ≤ `max_evidence_per_doc = 3` evidence từ cùng một `doc_id` → tránh 5 đoạn về cùng một bài (đây chính là "prompt bị nhiễu" gây giảm accuracy trong HisGraphRAG).  **Đường đi đồ thị không có `doc_id` nên được tính vào nhóm riêng** — xem §5.1.4.
- **Loại trùng lặp**: `dedupe_by_text()` bỏ các evidence có text giống hệt nhau trước khi chọn `max_evidence` (§5.1.4).
- **Ngân sách token**: `max_evidence = 12` evidence, tổng ≤ `context_max_tokens = 6000` → cắt theo điểm (thiếu ngân sách = ablation A6).
- Bỏ fusion (chỉ dùng vector top-k) = ablation A7.

## 7. Sinh câu trả lời

Prompt ràng buộc (tiếng Việt, `qa/answer.py::SYSTEM_PROMPT`):

```
Bạn là chuyên gia lịch sử Việt Nam, trả lời câu hỏi dựa TRÊN DUY NHẤT ngữ cảnh được cung cấp.
1. Chỉ sử dụng thông tin có trong CONTEXT. TUYỆT ĐỐI không bịa thông tin ngoài CONTEXT.
2. Sau mỗi mệnh đề khẳng định phải gắn nguồn trích dẫn dạng [E1], [E2]... trúng với số hiệu evidence.
3. Nếu CONTEXT không đủ để trả lời, hãy trả lời đúng câu:
   "Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này." và KHÔNG suy đoán.
4. Trả lời bằng tiếng Việt, súc tích, đúng mốc thời gian và địa danh.
5. Kết thúc bằng một mục "## Suy luận" gồm các bước suy luận đánh số (1., 2., 3.), mỗi bước có trích dẫn.
```

**Post-processing (quan trọng cho điểm "có căn cứ")**:
1. `validate_citations()` tách citation **hợp lệ** (có `E#` trong evidence) và **sai** (không tồn tại) → loại bỏ citation sai.
2. `refused` được suy ra bằng `is_refusal()` (so khớp các cụm từ từ chối đặc hiệu) — nếu **không còn citation nào** → coi là câu trả lời không có căn cứ, đánh dấu `grounded=False` trong error analysis.
3. `parse_reasoning_chain()` lấy phần sau tiêu đề `## Suy luận` (hoặc `## Reasoning`/`## Lập luận`/`## Chain of thought`), tách từng bước đánh số; nếu mô hình không dùng tiêu đề thì lấy các dòng bắt đầu bằng `1.`/`-`/`*`.
4. Trả về kèm `used_evidence` (đã lọc theo citation thực sự dùng) và `subgraph` (node/edge được dùng) để UI vẽ.

**Baseline B0 (closed-book)** dùng prompt riêng (`CLOSED_BOOK_SYSTEM_PROMPT`): không CONTEXT, không citation, trả lời bằng kiến thức tham số của mô hình → đây là baseline "LLM-only" mà docs/04 §4.1 yêu cầu.

> **B0 phải dùng JSON mode — không dùng prompt văn bản thuần.**
> Gemma 4 chạy một *kênh suy luận* riêng và tính token của kênh đó vào
> `max_output_tokens`.  Với câu hỏi closed-book mà mô hình **không biết**,
> kênh suy luận cứ quay vòng tìm kiến thức tham số cho tới khi hết ngân sách →
> API trả `finish_reason=MAX_TOKENS` với **câu trả lời rỗng**; lượt đánh giá
> đầu tiên vì thế có **98/150 câu B0 rỗng** (và độ trễ 45–90 s/câu).
>
> Khi dùng `response_schema` thì Gemini API **tắt hoàn toàn** kênh suy luận
> (`thoughts_token_count: null`) và lời gọi kết thúc trong **2.9–9 s**.
> Đo đạc: `scripts/_probe_thinking.py`, `scripts/_probe_prompt_variants.py`,
> `docs/10_quota_gemini_free_tier.md`.
>
> Vì vậy `generate_closed_book()` gọi JSON mode (`CLOSED_BOOK_SCHEMA`) với
> fallback theo thứ tự: (1) JSON mode → (2) nếu JSON hỏng **hoặc lặp** thì
> prompt văn bản thuần → (3) nếu vẫn lặp (`is_degenerate()`) thì trả câu từ
> chối chuẩn. Bộ phát hiện lặp (`is_degenerate`) bắt ba dạng: cùng một từ lặp
> ≥ 6 lần, cùng một n-gram (2–5 từ) lặp ≥ 4–5 lần, cùng một dòng lặp ≥ 3 lần.
>
> Số liệu đo được sau khi sửa (8 câu đầu, `scripts/validate_fixes.py`):
> `canned = 3/8`, `refused = 3/8`, độ trễ trung bình **8.3 s** (trước đây
> 88.9 s với 98/150 câu rỗng).

**Sinh chuỗi suy luận** cho multi-hop: phần `## Suy luận` do mô hình sinh **kèm citation** từ evidence đã cung cấp (chính là các `graph_path` evidence) → đảm bảo "chuỗi liên kết logic" có nguồn thật, không phải LLM tự bịa.

## 8. Entity Resolution (offline)

```mermaid
flowchart LR
  A["Danh sách mention<br/>từ cả 2 nguồn"] --> B["Bỏ dấu + bỏ honorific<br/>+ viết thường"]
  B --> C["Union-find theo key chuẩn hóa"]
  C --> D["Cụm embedding<br/>cos ≥ 0.86"]
  D --> E{"Trùng lặp<br/>mơ hồ?<br/>0.86 ≤ sim ≤ 0.97"}
  E -->|có| F["LLM merge<br/>Gemini"]
  E -->|không| G["Giữ riêng"]
  F --> H["Node chuẩn + aliases"]
  G --> H
```

- Node chuẩn giữ `aliases` (mọi biến thể) → entity linking tầng 1 khớp được nhiều cách viết.
- **Chống merge sai**: chỉ merge khi LLM xác nhận; ngoài ngưỡng → giữ riêng, ghi vào báo cáo (minh bạch).
- Mỗi node có `wikidata_id` nếu khớp được → dùng để đối chiếu thủ công khi kiểm tra.

## 9. Xử lý câu hỏi ngoài phạm vi & không đủ căn cứ

| Tình huống | Hành vi |
|---|---|
| Không liên quan lịch sử VN 905–1975 | Trả "Ngoài phạm vi" — tính đúng (không bịa) |
| Liên quan nhưng KG không đủ thông tin | Trả "Tài liệu hiện có không đủ thông tin" |
| Có đáp án nhưng thiếu căn cứ truy vết | Đánh dấu `grounded=False`, vẫn hiển thị để phân tích lỗi |

Bộ đánh giá có **22 câu out-of-KB** để đo hành vi này — đây là điểm mà baseline RAG thuần hay "bịa" và GraphRAG có lợi thế.

## 10. Ví dụ minh hoạ

**Câu hỏi:** "Vì sao nhà Trần phải gả Hoàng thái chúa Trần Khánh Dư cho vua Đinh?"

| Bước | Thành phần | Kết quả |
|---|---|---|
| 1 | Query Understanding | intent=`multi_hop`, entities=[Trần Khánh Dư, Đinh Bộ Lĩnh], time_refs=[1000], expected_hops=2, relation_plan=[`PRECEDED_BY`, `ABOUT`] |
| 2 | Entity Linking | `Trần Khánh Dư` → `:Person` (confidence 0.95), `Đinh Bộ Lĩnh` → `:Person` (0.93) |
| 3 | Vector | top chunk từ bài "Trần Khánh Dư" (đoạn nói về hôn ước) |
| 4 | Graph | path: `Trần Khánh Dư -[PRECEDED_BY]-> Hôn ước 1000 -[ABOUT]-> Đinh Bộ Lĩnh` + path đệ tử Nguyễn Trung Ngạn |
| 5 | Fusion | 3 evidence: 2 chunk + 1 graph_path |
| 6 | Sinh | "Vì nguy cơ cung đình Đinh Bộ Lĩnh bị phế, nhà Trần gả Trần Khánh Dư (con Trần Trọng Tháp) cho Đinh Bộ Lĩnh [E1][E3]. Việc này tạo mối quan hệ hôn nhân giữa hai nhà, làm suy yếu quyền lực riêng của Nguyễn Trung Ngạn [E3]." |
| 7 | Giải thích | Subgraph 5 node / 5 edge hiển thị trên UI |

*(Số liệu trên là ví dụ minh hoạ, chưa phải kết quả chạy thật.)*
