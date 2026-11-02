# 02 — Kiến trúc hệ thống

## 1. Nguyên tắc thiết kế

1. **Thay thế được từng thành phần**: mỗi năng lực (trích xuất, embedding, KG store, vector store, LLM) là một **interface** + nhiều implementation. Đổi model, đổi DB không đụng code gọi.
2. **Tái lập được**: mọi tham số nằm trong `configs/default.yaml`; mọi lời gọi LLM được **cache đĩa theo hash** nên chạy lại không tốn tiền và cho ra kết quả giống nhau.
3. **Có fallback**: nếu Docker lỗi, hệ thống vẫn chạy được với SQLite + numpy (ít tiện nghiệm hơn nhưng đủ demo).
4. **Có căn cứ**: mọi thứ sinh ra cho người dùng đều dẫn được về (chunk nguồn) hoặc (node–edge cụ thể).

## 2. Sơ đồ module

```mermaid
flowchart TB
  subgraph data["hgrag.data"]
    V[vhe.py]
    W[wikipedia.py]
    WD[wikidata.py<br/>QID + alias + P31/P27]
    H[history_chat.py]
    CH[chunking.py]
    NM[normalize.py]
  end
  subgraph llm["hgrag.llm"]
    CL[client.py<br/>Gemini + OpenAI-compat + mock]
    RL[rate_limit.py<br/>giới hạn RPM dùng chung<br/>giữa các tiến trình]
    PR[prompts.py]
    SC[schemas.py]
  end
  subgraph kg["hgrag.kg"]
    ON[ontology.py]
    AL[alias.py<br/>bảng alias QID-first]
    LB[labels.py<br/>tên sự kiện dùng chung]
    VG[vhe_gold.py<br/>import chú giải VHE]
    B[builder.py]
    ST[store.py<br/>Neo4j | SQLite]
  end
  subgraph ret["hgrag.retrieval"]
    EM[embeddings.py<br/>Gemini | LocalHF]
    VS[vector_store.py<br/>Qdrant | Numpy]
    EL[entity_linker.py<br/>5 tầng QID-first]
    GS[graph_search.py]
    FU[fusion.py]
    CX[context.py]
  end
  subgraph qa["hgrag.qa"]
    QU[query_understanding.py]
    PI[pipeline.py]
    AN[answer.py]
    EX[explain.py]
  end
  subgraph ev["hgrag.eval"]
    DB[dataset_builder.py]
    ME[metrics.py]
    JU[judge.py]
    RE[run_eval.py]
    RP[report.py]
  end
  CL --> PR
  CL --> SC
  CL --> RL
  V --> CH
  W --> CH
  CH --> NM
  NM --> B
  ON --> B
  B --> R
  R --> ST
  EM --> VS
  QU --> PI
  EL --> GS
  VS --> GS
  GS --> FU
  FU --> CX
  CX --> AN
  AN --> EX
  DB --> RE
  PI --> RE
  ME --> RE
  JU --> RE
  RE --> RP
```

## 3. Schema Knowledge Graph

### 3.1 Node

| Label | Thuộc tính chính | Ý nghĩa |
|---|---|---|
| `:Event` | `type` (1 trong 35 loại VHE), `year`, `date_text`, `description`, `uid` | Sự kiện lịch sử |
| `:Person` | `name`, `aliases`, `real_name`, `title`, `posthumous_name`, `roles`, `birth`, `death`, `uid` | Nhân vật |
| `:Place` | `name`, `aliases`, `kind` (thành phố/tỉnh/thép), `uid` | Địa điểm |
| `:Organization` | `name`, `aliases`, `kind` (triều đình/quân/chính phủ/tổ chức), `uid` | Tổ chức, chính quyền, quân đội |
| `:Period` | `name` (triều đại), `year_min`, `year_max`, `uid` | Giai đoạn lịch sử (Node ghép từ cây phân cấp trong wiki) |
| `:Chunk` | `chunk_id`, `text`, `title`, `url`, `year` | Đơn vị văn bản, điểm neo cho truy vết nguồn |

Tất cả node có `uid` ổn định. **Định danh ưu tiên theo QID** (chi tiết ở [`07_xac_dinh_thuc_the.md`](07_xac_dinh_thuc_the.md)):

| Loại node | `uid` | Ví dụ |
|---|---|---|
| Có Wikidata ID | `pe:Q511375` | Trần Hưng Đạo |
| Có QID nhưng khác miền | `pl:Q123456` | Địa danh |
| Không có QID | `ev:wiki:<doc>:<chunk>#<n>` | Sự kiện chỉ có trong văn bản |

Mọi node có thêm: `wikidata_id`, `aliases[]`, `name`, `name_key` (khóa chuẩn hóa), `entity_type`, `period`, `year_min`, `year_max`, `p31[]`, `p27[]`, `wiki_categories[]`, `source` (`wikipedia` | `vhe` | `llm`), `confidence`.

### 3.2 Edge

| Quan hệ | Từ → Đến | Nguồn |
|---|---|---|
| `PARTICIPATED_IN` | Person/Organization → Event | role `agent`, `beneficiary`, `instrument`, `force`, `experiencer` |
| `OCCURRED_AT` | Event → Place | role `location` |
| `OCCURRED_IN` | Event → Period | suy ra từ năm của event |
| `HAPPENED_AT` | Event → Event | role `temporal` (năm/nhàm/ngày) |
| `ABOUT` | Event → Person/Organization/Place | role `theme`, `source`, `goal`, `content` |
| `CAUSE_OF` | Event → Event | VHE causal `CAUSE` + LLM suy luận |
| `PRECONDITION_OF` | Event → Event | VHE causal `PRECONDITION` |
| `PRECEDED_BY` | Event → Event | VHE temporal `BEFORE` |
| `SIMULTANEOUS_WITH` | Event ↔ Event | VHE temporal `SIMULTANEOUS` |
| `SUBEVENT_OF` | Event → Event | VHE `SUBEVENT` |
| `BELONGED_TO_DYNASTY` | Person/Event → Period | từ cây phân cấp wiki + suy luận năm |
| `SUPPORTED_BY` | mọi edge → `:Chunk` | **provenance bắt buộc** |

Mọi edge (trừ `SUPPORTED_BY`) đều có thuộc tính: `type`, `text` (mô tả tự nhiên để nhúng), `year` (để lọc thời gian), `source` (`vhe` | `llm` | `wiki`), `confidence`, `uid` ổn định.

**Quy tắc bắt buộc cho edge do LLM suy luận** (chốt ở ADR-07/M3): chỉ tạo edge giữa **hai node đã có Wikidata QID** → quan hệ vẫn truy vết được, và `confidence` thấp hơn (`0.6` vs `0.95` cho edge từ văn bản).

### 3.3 Ontology lấy từ VHE

- **35 event type**: `be-born, marry, divorce, injure, die, transport, start-position, end-position, elect, nominate, arrest-jail, release-parole, trial-hearing, charge-indict, sue, convict, sentence, fine, execute, extradite, acquit, appeal, pardon, transfer-ownership, transfer-money, start-org, end-org, merge-org, declare-bankruptcy, attack, demonstrate, meet, phone-write, natural-disaster, natural-phenomenon`.
- **11 role**: `temporal, location, agent, theme, beneficiary, instrument, source, goal, content, force, experiencer`.
- **Quan hệ event**: `BEFORE, OVERLAP, CONTAINS, SIMULTANEOUS, ENDS-ON, BEGINS-ON` (temporal), `CAUSE, PRECONDITION` (causal), `SUBEVENT`.

Trích xuất từ Wikipedia dùng **đúng schema này** → KG đồng nhất giữa hai nguồn, và extractor có thể chấm điểm bằng VHE gold.

## 4. Hạ tầng Docker

```yaml
# docker-compose.yml (rút gọn)
services:
  neo4j:
    image: neo4j:5.26-community
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-history2025}
      NEO4J_PLUGINS: '["apoc"]'      # apoc.path.expandConfig cho traversal có giới hạn
    volumes: ["neo4j_data:/data"]
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:7474 >/dev/null || exit 1"]
      interval: 10s
      retries: 12
  qdrant:
    image: qdrant/qdrant:v1.12.4
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]
    healthcheck:
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/localhost/6333' || exit 1"]
      interval: 10s
      retries: 12
volumes: { neo4j_data: , qdrant_data: }
```

**Script `scripts/00_up.sh`**: kiểm tra `docker info` → `docker compose up -d` → chờ cả hai healthy → in URL. `scripts/00_down.sh`: `docker compose down` (giữ volume) hoặc `down -v` (xoá sạch).

**Port** đọc từ `.env`: `NEO4J_BOLT_PORT`, `NEO4J_HTTP_PORT`, `QDRANT_PORT` → đổi được nếu bị chiếm mà không sửa code.

## 5. Cấu trúc repo

```
vietnam_history_graphrag/
├── docker-compose.yml
├── pyproject.toml                 # uv; deps: google-genai, neo4j, qdrant-client,
│                                  # numpy, pandas, scikit-learn, streamlit, pyvis,
│                                  # matplotlib, plotly, pyyaml, requests, tenacity
├── .env.example                   # GEMINI_API_KEY, NEO4J_PASSWORD, KAGGLE_*
├── configs/default.yaml
├── docs/                          # (tài liệu này)
├── src/hgrag/
│   ├── config.py
│   ├── llm/       client.py rate_limit.py prompts.py schemas.py
│   ├── data/      vhe.py wikipedia.py history_chat.py chunking.py normalize.py
│   ├── kg/        ontology.py labels.py builder.py alias.py store.py vhe_gold.py
│   ├── retrieval/ embeddings.py vector_store.py entity_linker.py
│   │              graph_search.py fusion.py context.py lexical.py rerank.py
│   ├── qa/        query_understanding.py pipeline.py answer.py explain.py
│   ├── eval/      dataset_builder.py dataset.py metrics.py judge.py run_eval.py report.py
│   └── app/       streamlit_app.py
├── scripts/       00_up.sh 00_down.sh 01_ingest.py 03_build_kg.py 04_embed.py
│                  05_eval_dataset.py 06_ask.py 07_run_eval.py 08_smoke_test.sh
│                  09_live_check.py 10_validate_eval_paths.py 11_fix_eval_paths.py
│                  12_merge_eval.py run_all.sh
├── notebooks/kaggle/ 01_embed_corpus.ipynb 02_finetune_embedder.ipynb
├── data/raw  data/processed  artifacts/  reports/
└── tests/
```

## 6. Interface & implementation

| Interface | Implementation chính | Implementation dự phòng |
|---|---|---|
| `LLMClient` | `GeminiClient` (`google-genai`) | `OpenAICompatClient` (Qwen/Qwen3 qua vLLM trên Kaggle), `MockClient` (offline) |
| `Embedder` | `GeminiEmbedder` (`gemini-embedding-001`) | `LocalEmbedder` (Qwen3-Embedding-0.6B / bge-m3 từ Kaggle) |
| `GraphStore` | `Neo4jGraphStore` | `SQLiteGraphStore` |
| `VectorStore` | `QdrantVectorStore` | `NumpyVectorStore` (brute-force cosine) |

Tất cả đọc cấu hình từ `configs/default.yaml` + biến môi trường.

### 6.1 Lớp LLM và hạn mức free tier

`hgrag.llm.client.GeminiClient` là điểm duy nhất mà code gọi được API.  Nó
đảm bảo bốn điều khi hạn mức của Google AI Studio bị siết:

| Đặc điểm của free tier | Cơ chế trong client |
|---|---|
| Quota **ngày** (20 req/ngày cho Flash, 500 cho Flash-Lite) | `_exhausted_models` ghi nhớ model đã cạn → không phải trả giá một lời gọi thất bại cho mỗi request; khi **mọi** model cạn, danh sách được reset để nhận lại quota theo ngày |
| Quota **phút** (15–30 req/phút) | `_throttled` đưa model vào cooldown ngắn rồi **xoay sang model khác**; `_wait_for_rpm` giãn nhịp chủ động theo `llm.model_rpm` |
| Hạn mức tính **chung cho mọi tiến trình** | `hgrag.llm.rate_limit.SharedRateLimiter` — các worker `flock` trên cùng một file nên tổng số request/phút vẫn trong hạn mức |
| Quota **embedding** tách riêng với quota chat | `_embed_quota_exhausted` (circuit breaker) + `fallback_embed_models`; vì không gian vector phải nhất quán, corpus được **embed lại** bằng model mới |

Ngoài ra, khoá cache của client chứa **tên model chính** → đổi model giữa hai
lượt chạy không thể vô tình dùng lại câu trả lời của model cũ.

### 6.2 Cache tồn tại lâu

Ba tầng cache trên đĩa giúp chạy lại gần như miễn phí:

| Cache | Vị trí | Nội dung |
|---|---|---|
| LLM | `artifacts/llm_cache/` | theo hash của request (+ tên model) |
| Embedding | `artifacts/llm_cache/` | theo hash của `model + text + dim` |
| Vector của entity | `artifacts/alias_embeddings.json` | vector của ~14 000 entity để tầng liên kết tầng 5 không phải embed lại (4 phút CPU) mỗi lần khởi động |
| Chỉ mục từ khoá | `artifacts/lexical_index.json` | dự phòng khi quota embedding cạn |

## 7. Luồng chạy chính (`scripts/run_all.sh`)

```
00_up.sh                      # bật Neo4j + Qdrant, chờ healthy
01_ingest.py                  # VHE + Wikipedia + HistoryChat → corpus.jsonl (cached)
03_build_kg.py --vhe          # Gemini trích xuất event theo ontology VHE + entity resolution
                              #   + nạp Neo4j (kèm provenance SUPPORTED_BY → :Chunk)
                              #   + import chú giải VHE thành đồ thị con riêng
04_embed.py                   # nhúng chunk → Qdrant `chunks_local`
04_embed.py --corpus data/processed/vhe_corpus.jsonl
                              #   → nhúng câu VHE → Qdrant `chunks_vhe_local`
05_eval_dataset.py            # sinh bộ câu hỏi đánh giá (4 nguồn) → eval_questions.jsonl
10_validate_eval_paths.py     # đo bao nhiêu node của gold path có thật trong KG
11_fix_eval_paths.py          # chuẩn hoá gold path theo KG (nếu cần)
06_ask.py                     # CLI hỏi đáp: câu trả lời + [E#] + chuỗi suy luận + subgraph
07_run_eval.py                # chạy B0/B1/B2/Ours + A1–A7 → reports/results.{csv,md} + figures/
08_smoke_test.sh              # kiểm tra offline toàn hệ thống (imports, pytest, services)
09_live_check.py              # kiểm tra dịch vụ thật (Gemini + Neo4j + Qdrant + QA end-to-end)
12_merge_eval.py              # gộp checkpoint của nhiều worker → một báo cáo
run_all.sh                    # chạy toàn bộ pipeline theo đúng thứ tự trên
```

> Ghi chú: bước trích xuất event **không tách riêng** thành `02_extract.py` mà được
> tích hợp vào `03_build_kg.py` (cache theo `artifacts/extractions/<chunk_id>.json`),
> nhờ vậy việc dựng KG là idempotent và chạy lại không tốn quota.

Mỗi bước **idempotent**, có cache, chạy lại được, seed cố định.

## 8. Ranh giới hệ thống & điểm mở rộng

- **Thêm nguồn dữ liệu**: viết loader mới trong `data/` xuất ra cùng schema `corpus.jsonl` → chạy lại từ `01` → KG tự cập nhật (đáp ứng yêu cầu "cập nhật/mở rộng KG").
- **Thêm loại quan hệ mới**: bổ sung vào `kg/ontology.py` → extractor & graph_search dùng chung danh sách → không cần sửa code truy xuất.
- **Thêm model LLM**: cấu hình `llm.provider` trong yaml → không sửa code.
- **Scale lên corpus lớn**: thay `NumpyVectorStore` → FAISS/Qdrant cluster; việc trích xuất đã cache theo chunk nên chạy lại rẻ.

## 9. Trạng thái thực thi (đã chạy thật)

| Hạng mục | Kết quả thực tế |
|---|---|
| Corpus | **438 chunk** từ Wikipedia tiếng Việt (59 bài seed) + **4 110 câu VHE** + History Chat |
| Knowledge Graph | **18 634 node · 36 214 edge** (Wikipedia trích xuất bằng Gemini thật + **VHE gold**: 12 089 entity, 30 997 relation) |
| Provenance | Mọi edge `SUPPORTED_BY` → `:Chunk` (mọi node truy vết được nguồn) |
| Vector store | Qdrant `chunks_local` (438 chunk, 384 chiều, MiniLM đa ngôn ngữ) + `chunks_vhe_local` (4 110 câu VHE, 384 chiều) |
| Bộ câu hỏi | **150 câu**: 88 VHE template · 30 LLM-from-Wikipedia · 10 history chat · 22 out-of-KB |
| Gold path | **97,7%** node của gold path có thật trong đồ thị (`scripts/10_validate_eval_paths.py`) |
| Test | `pytest` 100% pass (offline, không gọi mạng) — **313 test + 2 skip** |
| Báo cáo | `reports/results.md`, `reports/results.csv`, `reports/figures/*.png` |

### 9.1 Hai đồ thị con (subgraph)

Đồ thị gồm hai đồ thị con **không trộn lẫn**, phân biệt bằng thuộc tính `source`:

| Subgraph | Nguồn | Nội dung | Vì sao tách riêng |
|---|---|---|---|
| VHE gold | `vhe` | 12 089 entity + 30 997 relation + 4 110 chunk | Chỉ có ở bộ chú giải VHE; **88/150 câu hỏi đánh giá** dựng từ đây |
| Wikipedia/LLM | `wikipedia`, `llm` | 2 435 entity + 5 217 relation + 438 chunk | Trích xuất bằng Gemini từ văn bản Wikipedia |

Lý do tách: đồ thị Wikipedia là *suy luận* của LLM (sai sót có), đồ thị VHE là
*chú giải vàng* của bộ dữ liệu (đúng hơn) và có câu Vệt cổ — trộn lẫn sẽ làm
mất cả tính truy vết và chất lượng. Chi tiết: [`06_quyet_dinh_thiet_ke.md`](06_quyet_dinh_thiet_ke.md).

### 9.2 Hướng của các edge

Mọi edge có hướng cố định (`hgrag.kg.ontology.edge_is_participant_to_event`):

- **người tham gia → sự kiện**: `PARTICIPATED_IN`, `AGENT_OF`, `INVOLVES`, `CAUSE_OF`, `PRECONDITION_OF`
- **sự kiện → đối tượng ngữ cảnh**: `ABOUT`, `LOCATED_IN`, `OCCURRED_IN`, `OCCURRED_AT`, `HAPPENED_AT`

Nhờ vậy một BFS bắt đầu từ một thực thể được nêu trong câu hỏi ("Đế Minh")
đi tới được node sự kiện ("Đế Minh sinh ra Đế Nghi") rồi tới các đối tượng
ngữ cảnh — đúng với hướng mà gold path của bộ đánh giá viết.

### 9.3 Tên node của sự kiện

Node sự kiện được đặt tên theo **đúng công thức** dùng chung bởi cả KG và bộ
đánh giá (`hgrag.kg.labels.event_label`):

```
"<chủ thể> <động từ> <đối tượng>"     ví dụ: "Đế Minh sinh ra Đế Nghi"
```

Hai nơi dùng chung một bảng ánh xạ (`EVENT_TYPE_VN`, `ROLE_VN`) là điều kiện
để metric `path_hit_rate` đo được điều gì: nếu KG gọi sự kiện là
"Đế Minh sinh Đế Nghi" còn gold path ghi "Đế Minh sinh ra Đế Nghi", hệ thống
bị trừ điểm cho một sự kiện nó đã truy xuất hoàn hảo.

Các con số trên được tái lập bằng `bash scripts/run_all.sh` và kiểm chứng lại bằng
`bash scripts/08_smoke_test.sh` + `python scripts/09_live_check.py`.
