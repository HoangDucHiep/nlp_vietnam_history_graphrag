# Outline slide thuyết trình — Vietnamese History GraphRAG (905–1975)

> Thời lượng gợi ý: **12–15 phút** + 5 phút hỏi đáp.
> Mỗi slide ghi rõ: **Tiêu đề**, **Nội dung**, **Cách nói (script)**, **Bằng chứng cần chuẩn bị**.

---

## Slide 1 — Bìa (0:30)

* **Tiêu đề**: *Hệ thống GraphRAG hỏi đáp lịch sử Việt Nam (905–1975)*
* **Nội dung**: tên, lớp, môn học; tên GV; ngày.
* **Script**: "Đồ án của em xây một hệ thống hỏi đáp lịch sử Việt Nam, dựa trên Knowledge Graph và truy xuất lai, trả lời **có căn cứ**."
* **Bằng chứng**: logo trường (nếu có).

## Slide 2 — Vấn đề (1:30)

* **Nội dung**:
  * Câu hỏi lịch sử thường **nhiều bước**: "Vì sao nhà Trần gả Trần Khánh Dư cho Đinh Bộ Lĩnh?"
  * RAG thuần (embedding) khó nối 2–3 sự kiện qua quan hệ nhân quả/thời gian.
  * RAG thuần dễ **bịa** khi không có đủ context.
* **Script**: dẫn dắt bằng một ví dụ câu hỏi thật, hỏi khán giả "cần những bước suy luận nào?".
* **Bằng chứng**: ảnh chụp một câu trả lời sai của RAG thuần (nếu có).

## Slide 3 — Đóng góp (1:00)

* **Nội dung** (4 điểm):
  1. KG theo ontology VHE + provenance bắt buộc.
  2. Định danh thực thể QID-first 6 tầng.
  3. Truy xuất hybrid (vector + đồ thị) + RRF + chống nhiễu.
  4. Sinh trả lời có căn cứ + hệ thống đánh giá đầy đủ (baseline, ablation, CI, McNemar).
* **Script**: nói chắc, ngắn, đúng 4 điểm.

## Slide 4 — Kiến trúc tổng thể (1:30)

* **Nội dung**: sơ đồ mermaid từ `README.md` §2 (offline → online).
* **Script**: đi theo luồng: "ngoại tuyến dựng KG và index; truy vấn thì phân tích câu hỏi → entity linking → truy xuất song song vector & đồ thị → hợp nhất → sinh trả lời".
* **Bằng chứng**: sơ đồ vẽ tay trên giấy A3 (in sẵn).

## Slide 5 — Dữ liệu & ontology (1:00)

* **Nội dung**: 438 chunk Wikipedia + VHE + History Chat; 35 event type / 11 role; **provenance** `[:SUPPORTED_BY] → :Chunk`.
* **Script**: nhấn mạnh *"mọi cạnh đều truy vết được về đoạn văn bản nguồn — đây là nền cho câu trả lời có căn cứ"*.
* **Bằng chứng**: ảnh Neo4j Browser.

## Slide 6 — Bài toán nhập nhằng thực thể (1:30)

* **Nội dung**: "Trần Hưng Đạo" vs "Trần Quốc Tuấn" vs "Hưng Đạo Đại Vương"; bảng 6 tầng QID-first.
* **Script**: kể một ví dụ cụ thể em gặp khi chạy thật (ví dụ một mention không resolve được, sửa bằng tầng nào).
* **Bằng chứng**: log thực tế từ `artifacts/log_build_kg.txt` hoặc ảnh chụp console.

## Slide 7 — Thuật toán truy xuất (2:00)

* **Nội dung**:
  * Chấm điểm đường đi (công thức 4 yếu tố).
  * RRF + temporal boost + diversity cap + ngân sách token.
* **Script**: giải thích **vì sao** cần diversity cap ("5 đoạn về cùng một bài làm prompt nhiễu → giảm accuracy").

## Slide 8 — Sinh trả lời có căn cứ (1:00)

* **Nội dung**: prompt ràng buộc; ví dụ thật: câu trả lời + `[E#]` + "## Suy luận" + ảnh subgraph.
* **Script**: bấm demo trực tiếp nếu trình bày có màn hình; nếu không thì dùng ảnh chụp.
* **Bằng chứng**: ảnh chụp UI tab "Hỏi đáp" + tab "Truy vết KG".

## Slide 9 — Thiết lập thực nghiệm (1:00)

* **Nội dung**: 150 câu / 4 nhóm / 4 nguồn; 4 hệ thống (B0–B2, Ours); 7 ablation; Gemini judge; cùng LLM & prompt cho mọi hệ thống.
* **Script**: nhấn mạnh *"cùng model, cùng prompt, cùng corpus → khác biệt là do kiến trúc truy xuất"*.

## Slide 10 — Kết quả chính (2:00)

* **Nội dung**: bảng tổng hợp (điền số từ `reports/results.md`).
* **Script**: đọc 2–3 con số đáng chú ý nhất; nói rõ nhóm nào thắng, nhóm nào hoà.
* **Bằng chứng**: `reports/figures/accuracy_by_group.png`.

## Slide 11 — Thống kê & ablation (1:30)

* **Nội dung**: CI bootstrap 95% + McNemar; bảng ablation A1–A7 (heatmap).
* **Script**: giải thích ý nghĩa CI/McNemar ở mức đồ án (không đi sâu thống kê).
* **Bằng chứng**: `reports/figures/ablation_heatmap.png`.

## Slide 12 — Phân tích lỗi (1:30)

* **Nội dung**: bảng phân loại lỗi + 2–3 ví dụ cụ thể.
* **Script**: **đừng giấu lỗi** — nói rõ hệ thống thua ở đâu và vì sao. Đây là phần được đánh giá cao nhất.
* **Bằng chứng**: `reports/error_analysis.md`.

## Slide 13 — Demo trực tiếp (2:00)

* **Nội dung**: chạy `streamlit run src/hgrag/app/streamlit_app.py`; hỏi 1 câu multi-hop; click vào citation; xem subgraph; xem tab Đánh giá.
* **Script**: chuẩn bị sẵn 2 câu hỏi (1 multi-hop, 1 out-of-KB để khoe cơ chế từ chối).

## Slide 14 — Kết luận & hướng phát triển (1:00)

* **Nội dung**: 2 kết luận + 3 hướng (mở rộng ontology, fine-tune embedder, mở rộng corpus).
* **Script**: kết thúc bằng câu: *"điểm quan trọng nhất không chỉ là điểm số, mà là mỗi câu trả lời đều dẫn được về nguồn"*.

## Slide 15 — Cảm ơn & hỏi đáp

* **Nội dung**: link GitHub, lệnh tái lập (`bash scripts/run_all.sh`), danh sách tài liệu `docs/`.

---

## Phụ lục slide (dùng khi bị hỏi)

* **A1**: chi tiết 6 tầng entity resolution + điểm ngưỡng.
* **A2**: công thức chấm điểm đường đi + giải thích từng thành phần.
* **A3**: câu Cypher thực tế dùng trong `graph_search.py`.
* **A4**: rubric của LLM judge (file `src/hgrag/eval/judge.py`).
* **A5**: hạn chế của hệ thống & hướng cải thiện chi tiết.
