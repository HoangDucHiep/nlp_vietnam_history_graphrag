# Phân tích lỗi — Vietnamese History GraphRAG

Phân loại lỗi theo docs/04 §5.4. Mỗi lỗi được gán *dấu hiệu trong log* và *nguyên nhân gốc* để dẫn tới hướng khắc phục cụ thể.

## Hệ thống: ours

- Tổng câu: **150** · sai: **60** (40.0%)

| Loại lỗi | Số lỗi | Tỉ lệ | Nguyên nhân gốc |
|---|---|---|---|
| Entity linking sai | 5 | 8.3% | Alias thiếu / trùng tên → anchor node sai |
| Đường đi sai | 2 | 3.3% | Ontology thiếu quan hệ cần thiết / path scoring chưa tốt |
| Thiếu evidence | 4 | 6.7% | Chunk quá dài / embedding kém cho câu hỏi lịch sử |
| Sinh sai dù có evidence | 1 | 1.7% | Prompt chưa ràng buộc đủ / context quá dài |
| Từ chối vì truy xuất hụt | 36 | 60.0% | Đồ thị/vector chưa dựng được đường tới đáp án vàng |
| Từ chối dù đã có evidence | 7 | 11.7% | Đáp án vàng đã có trong CONTEXT nhưng mô hình vẫn từ chối |
| Trích dẫn không hợp lệ | 4 | 6.7% | Mô hình sinh [E#] không tồn tại / evidence thiếu text |
| Câu hỏi lỗi (ngoài KB / sai lệch) | 1 | 1.7% | Câu hỏi tự nó sai lệch lịch sử hoặc nằm ngoài phạm vi kho |

### Ví dụ

**Entity linking sai**

- `vhe_00010` (single_hop) — *Ai là người tấn công Nam Chiếu?*
  - Đáp án: Dựa trên ngữ cảnh được cung cấp, có các thông tin về việc tấn công Nam Chiếu như sau:

*   Cao Vương đi đánh Nam Chiếu [E4].
*   Biền được thêm quân, tiến đánh Nam Chiếu [E2].
*   Chiêu Nghĩa hầu tới Chiêm Chiêu muốn tấn công trại nhưng bị thua [E5].
  - Tham chiếu: Cao Vương
  - Judge: Đáp án liệt kê thêm Biền, trong khi đáp án tham chiếu chỉ có Cao Vương. Tuy nhiên, CONTEXT không có thông tin về Biền là người tấn công Nam Chiếu, CONTEXT [E2] là một đoạn văn bị lỗi/nhiễu-đáp án-tham-chiếu-sai-lỗi-logic-về-nguồn-dữ liệu.
- `vhe_00029` (single_hop) — *Ai là người qua đời Thứ sử Chu Ngung?*
  - Đáp án: Đồn binh người châu là người đã bắt và giết Thứ sử Chu Ngung [E1], [E2].
  - Tham chiếu: Đồn binh
  - Judge: Đáp án trả lời về việc Đồn binh giết chết Chu Ngung thay vì trả lời trực tiếp ai là người qua đời. Thông tin này không khớp với câu hỏi.
- `vhe_00033` (single_hop) — *Ai là người đảm nhiệm chức vụ Lữ Đại?*
  - Đáp án: Dựa trên các thông tin được cung cấp, câu hỏi "Ai là người đảm nhiệm chức vụ Lữ Đại?" có sự nhầm lẫn về mặt ngữ nghĩa trong cách đặt câu hỏi (Lữ Đại là tên người, không phải tên chức vụ). Tuy nhiên, theo nội dung các nguồn tin:

Vua Ngô đã cho Lữ Đại làm Thứ sử [E2], [E4], [E6], [E8], [E10], [E12].
  - Tham chiếu: Vua Ngô
  - Judge: Đáp án ứng viên nhận diện được sự nhầm lẫn trong câu hỏi và chỉ ra được thông tin đúng từ context. Tuy nhiên, câu hỏi yêu cầu tìm người đảm nhiệm chức vụ Lữ Đại, mà thực tế Lữ Đại là tên người. Do đó, không thể trả lời lời giải cho chức vụ đó.
- `vhe_00038` (single_hop) — *Ai là người được tha tội Khuông?*
  - Đáp án: Nhất, Vĩ và Khuông là những người được Ngô Vương tha tội [E1].
  - Tham chiếu: Ngô Vương
  - Judge: Đáp án ứng viên không trả lời đúng câu hỏi về người được tha tội Khuông. Câu hỏi hỏi về chủ thể của hành động tha tội, 


- `vhe_00043` (single_hop) — *Trong sự kiện do Tu Tắc thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Trong các sự kiện liên quan đến Tu Tắc, các đối tượng được nhắc đến bao gồm:

*   **Các tướng quân và nhân vật liên quan:** Mao Linh, Đổng Nguyên [E2], [E4], [E12]; Cố Dung [E6], [E8], [E10]; Dương Tắc [E2], [E4], [E12]; Lưu Tuấn [E2], [E4], [E12]; Lý Khắc Chính [E8]; Tiết độ sứ [Khúc] Thừa Mỹ [E8];
  - Tham chiếu: Giao Châu
  - Judge: Đáp án liệt kê quá nhiều đối tượng không trực tiếp liên quan đến sự kiện do Tu Tắc thực hiện theo yêu cầu câu hỏi. Nó không trả lời trực tiếp Giao Châu.

**Đường đi sai**

- `chat_00009` (single_hop) — *Di sản lịch sử của Võ Nguyên Giáp là gì?*
  - Đáp án: Đại tướng Võ Nguyên Giáp đã tặng đền thờ Ngô Quyền ở Đường Lâm một đĩa sứ vào mùa xuân Mậu Dần (1998) [E2], [E8]. Trên đĩa sứ có vẽ cảnh thủy chiến Bạch Đằng và dòng chữ: "Tổ quốc và nhân dân ta đời đời ghi công lớn của vị anh hùng dân tộc Ngô Quyền đã đánh tan quân xâm lược dựng nên nền độc lập của
  - Tham chiếu: Di sản của Võ Nguyên Giáp: tổng tư lệnh; Điện Biên Phủ 1954; chiến dịch Tết Mậu Thân. Hình tượng ông gắn với giá trị độc lập, cải cách và trí tuệ, được tôn vinh qua sử sách và ký ức tập thể.
  - Judge: Đáp án không trả lời trực tiếp câu hỏi về di sản lịch sử của Võ Nguyên Giáp. Nó chỉ liệt kê các hành động cụ thể trong quá khứ thay vì khái niệm khái quát quát về di sản. 
- `llm_00023` (multi_hop) — *Theo tài liệu về Lê Thái Tổ, Tháng có mối liên hệ gì với Lê Thái Tổ?*
  - Đáp án: Theo tài liệu về Lê Thái Tổ, các tháng có mối liên hệ với ông bao gồm:
- Ngày sinh của ông là ngày 6 tháng 8 năm Ất Sửu (tức ngày 10 tháng 9 năm 1385) [E2].
- Ngày giỗ của thân phụ ông là Lê Khoáng là ngày 17/10 âm lịch [E4].
- Ngày giỗ của thân mẫu ông là Trịnh Ngọc Thương là ngày 17/10 âm lịch [E4
  - Tham chiếu: Tháng 2 âm lịch năm 1429, Lê Thái Tổ lệnh cho tập trận trong 5 đạo, cả thủy binh và bộ binh.
  - Judge: Đáp án không chứa thông tin về việc Lê Thái Tổ lệnh cho tập trận vào tháng 2 âm lịch năm 1429 như đáp án tham chiếu. Tuy nhiên, lại có thông tin về ngày sinh và ngày giỗ của cha mẹ ông. Thông tin này được CONTEXT-grounded. grounded = true.

**Thiếu evidence**

- `vhe_00007` (single_hop) — *Trong sự kiện do Hùng Vương thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Trong sự kiện do Hùng Vương thực hiện (hành động xưng), đối tượng/tôn hiệu được nhắc đến là "Vương" [E2].
  - Tham chiếu: Văn Lang
  - Judge: Đáp án ứng viên trả lời là 'Vương' dựa trên ngữ cảnh [E2], tuy nhiên đáp án tham chiếu yêu cầu là 'Văn Lang'. Thông tin trong [E2] không đề cập đến Văn Lang nên đáp án chưa đầy đủ theo yêu cầu.
- `vhe_00018` (single_hop) — *Trong sự kiện do Vua Hán thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Trong các sự kiện liên quan đến Vua Hán, các đối tượng được nhắc đến bao gồm:

*   Trần Hữu Lượng: người thành lập Hán Vương, tiếm xưng đế, đặt quốc hiệu là Hán, đổi niên hiệu là Đại Nghĩa và đánh nhau với Minh Thái Tổ [E2].
*   Minh Thái Tổ: đối tượng mà Trần Hữu Lượng đánh nhau cùng [E2].
*   Giặc
  - Tham chiếu: Tể tướng Trần Bình
  - Judge: Đáp án ứng viên không trả lời trực tiếp câu hỏi về đối tượng được nhắc đến trong sự kiện do Vua Hán thực hiện. Đáp án liệt kê kê khai khai báo-đối tượng liên quan đến Hán Vương,</tbody>
- `vhe_00027` (single_hop) — *Trong sự kiện do Trưng Nữ Vương thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Trong sự kiện do Trưng Nữ Vương thực hiện, đối tượng được nhắc đến là quân nhà Hán [E2], [E5], [E7], [E9], [E11].
  - Tham chiếu: Tô Định
  - Judge: Đáp án không nhắc đến Tô Định như đáp án tham chiếu. Tuy nhiên, quân nhà Hán là đối tượng được nhắc đến trong context. Tuy nhiên, theo quy tắc, đáp án phải so với đáp án tham chiếu/gold facts. So với với tham chiếu là Tô Định, đáp án này sai.
- `vhe_00049` (multi_hop) — *Sự kiện qua đời Bố Linh và sự kiện tấn công Chiêm Thành cùng do nhân vật nào thực hiện?*
  - Đáp án: Sự kiện tấn công Chiêm Thành tại trại Bố Chính, thẳng đến núi Long Tỵ và sự kiện Bố Linh bị chém chết tại trận đều do Khai Thiên Vương [Phật Mã] và Đào Thạc Phụ thực hiện [E2].
  - Tham chiếu: Khai Thiên Vương [Phật Mã]
  - Judge: Đáp án đúng về nhân vật chính là Khai chia Thiên Vương [Phật Mã]. Tuy nhiên, đáp án ứng viên- được cho là do cả hai nhânthought}{

**Sinh sai dù có evidence**

- `vhe_00011` (single_hop) — *Trong sự kiện do Cao Vương thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Dựa trên các thông tin được cung cấp, các đối tượng được nhắc đến trong các sự kiện liên quan đến Cao Vương bao gồm:

*   **Vua:** Người đã gặp gỡ Cao Vương, có lời dặn về việc giao nhận quốc thư tại đầu địa giới [E10], [E12]. Vua và Tuyên Từ hoàng thái hậu đều bị hại [E10]. Trong một sự kiện khác,
  - Tham chiếu: Nam Chiếu
  - Judge: Đáp án liệt kê nhiều đối tượng khác nhau thay vì trả lời trực tiếp câu hỏi về đối tượng được nhắc đến trong sự kiện do Cao Vương thực hiện. Thông tin về Nam Chiếu không có trong context.

**Từ chối vì truy xuất hụt**

- `vhe_00015` (single_hop) — *Ai là người tấn công Nam Việt?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Lâm Lư hầu Chu Táo
  - Judge: Đáp án không trả lời được câu hỏi và không cung cấp thông tin dựa trên context.
- `chat_00001` (single_hop) — *Vì sao xảy ra sự kiện Đặt ách bảo hộ toàn cõi (1884)? Nêu bối cảnh và nguyên nhân chính.*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Bối cảnh trước 1884: Việt Nam vào Liên bang Đông Dương là phản ứng trước sức ép thời cuộc. Nguyên nhân chính gồm mâu thuẫn quyền lực, xâm lược/đe dọa bên ngoài và nhu cầu khẳng định chủ quyền. Kết cục
  - Judge: Đáp án ứng viên không trả lời được câu hỏi. Thông tin trong CONTEXT không chứa đựng thông tin về sự kiện 1884.
- `chat_00008` (single_hop) — *So sánh đặc điểm trị quốc của triều Bắc thuộc Minh (1407–1427) và Trần (1225–1400).*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Bắc thuộc Minh nhấn mạnh nhà Minh đô hộ, bóc lột; khởi nghĩa Lam Sơn bùng nổ; Trần nổi bật ba lần kháng chiến chống Mông–Nguyên (1258, 1285, 1288), phát triển kinh tế biển. Điểm chung: củng cố tập quy
  - Judge: Đáp án không trả lời câu hỏi và không cung cấp thông tin dựa trên CONTEXT.
- `vhe_00026` (single_hop) — *Ai là người tấn công Tô Định?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Trưng Nữ Vương
  - Judge: Đáp án ứng viên không trả lời được câu hỏi dựa trên thông tin có sẵn trong CONTEXT. Do đó, điểm 0.
- `vhe_00028` (single_hop) — *Sự kiện Mã Viện tấn công diễn ra ở đâu?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Lãng Bạc
  - Judge: Đáp án không trả lời được câu hỏi và không cung cấp thông tin.

**Từ chối dù đã có evidence**

- `vhe_00002` (single_hop) — *Trong sự kiện do Đế Minh thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Đế Nghi
  - Judge: Đáp án ứng viên không trả lời được câu hỏi dựa trên thông tin có thể tìm thấy trong CONTEXT.
- `vhe_00021` (single_hop) — *Sự kiện Vương Khôi tấn công Mân Việt diễn ra ở đâu?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Dự Chương
  - Judge: Đáp án không trả lời được câu hỏi dù thông tin có sẵn trong CONTEXT.
- `vhe_00037` (single_hop) — *Sự kiện Vua Ngô đảm nhiệm chức vụ Đái Lương diễn ra ở đâu?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Giao Châu
  - Judge: Đáp án ứng viên không trả lời được câu hỏi dù thông tin có sẵn trong CONTEXT.
- `vhe_00055` (multi_hop) — *Sự kiện tấn công Thăng Hoa và sự kiện tấn công Hoá Châu cùng do nhân vật nào thực hiện?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Chiêm Thành
  - Judge: Đáp án không trả lời được câu hỏi và không cung cấp thông tin có trong context.
- `vhe_00051` (multi_hop) — *Sự kiện đảm nhiệm chức vụ Thượng hoàng và sự kiện đảm nhiệm chức vụ Bảo Thánh hoàng hậu cùng do nhân vật nào thực hiện?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Hoàng thái tử Thuyên
  - Judge: Đáp án ứng viên không trả lời được câu hỏi, trong khi thông tin có sẵn trong CONTEXT.

**Trích dẫn không hợp lệ**

- `chat_00007` (single_hop) — *So sánh bối cảnh và tác động của Dời đô ra Thăng Long (1010) và Gia Long thống nhất (1802).*
  - Đáp án: Dựa trên các thông tin được cung cấp, dưới đây là sự so sánh về bối cảnh và tác động của hai sự kiện:

**1. Bối cảnh**
* **Dời đô ra Thăng Long (1010):** Vua Lý Thái Tổ cho rằng kinh đô Hoa Lư có "thành hẹp, đất thấp" [E3]. Triều đình đã khởi sự dời đô vào tháng 7 năm Canh Tuất (1010) sau khi các qu
  - Tham chiếu: Điểm giống: đều diễn ra trong thời khắc quyết định, tác động tới cục diện quyền lực. Khác: Dời đô ra Thăng Long (1010) chọn vị thế trung tâm, mở thời kỳ thịnh trị; trong khi Gia Long thống nhất (1802)
  - Judge: Đáp án ứng viên chứa nhiều thông tin không có trong CONTEXT, đặc biệt là về việc đổi tên Đại La thành Thăng Long và các loại chùa. 
- `chat_00003` (single_hop) — *Trình bày đặc điểm chính của triều Nguyễn (1802–1945).*
  - Đáp án: Dựa trên CONTEXT được cung cấp, các đặc điểm chính của triều Nguyễn bao gồm:

* **Về tính chất:** Đây là vương triều phong kiến cuối cùng trong lịch sử Việt Nam [E2].
* **Về lãnh thổ và quốc hiệu:** Triều đại này chính thức sử dụng quốc hiệu Việt Nam với cương thổ thống nhất rộng lớn nhất cho đến th
  - Tham chiếu: Triều Nguyễn (1802–1945) nổi bật với: Gia Long thống nhất; Huế là kinh đô; sau suy yếu trước xâm lược Pháp. Bộ máy trung ương dần hoàn thiện, luật lệ–lễ nhạc được quy củ, biên cương ổn định hơn.
  - Judge: Đáp án ứng viên không có thông tin trong CONTEXT. Các thông tin về lãnh thổ, giáo dục, kinh tế và kinh đô Phú Xuân được cung cấp trong CONTEXT không chính xác hoặc không có trong văn bản.
- `llm_00007` (single_hop) — *Theo tài liệu về Lê Đại Hành, Lê Đại Hành là ai và có vai trò gì?*
  - Đáp án: Lê Đại Hành (hay Lê Hoàn) là người làng Bảo Thái, huyện Thanh Liêm [E12]. Ông có tính tình phóng khoáng, có chí lớn, được sứ quân Đinh Bộ Lĩnh khen là người trí dũng, chắc thế nào cũng làm được việc và đã được giao cai quản 1 nghìn quân sĩ [E2], [E8]. Trong cuộc kháng chiến chống quân Tống, ông đã đ
  - Tham chiếu: Lê Hoàn chỉnh đốn quân lữ, đánh nhau với Đinh Điền, Nguyễn Bặc ở Tây Đô, Đinh Điền, Nguyễn Bặc bỏ chạy, đem quân thủy ra đánh.
  - Judge: Đáp án có thông tin về việc đóng cọc ngăn sông Chi Lăng không có trong CONTEXT. Do đó, đáp án án không được grounded.
- `vhe_00075` (temporal) — *Sự kiện Hán Vương đảm nhiệm chức vụ diễn ra vào mùa hay tháng nào?*
  - Đáp án: Dựa trên các thông tin được cung cấp, sự kiện Hán Vương qua đời diễn ra vào mùa hạ [E2], [E4], [E6], [E8], [E10], [E12]. Trong mùa hạ đó, các sự kiện liên quan đến việc đảm nhiệm chức vụ diễn ra vào các thời điểm sau:

*   Tháng 4: Nguyễn Đình Kế được tước Hoằng quận công [E4]; Đỗ Tử Bình được lấy l
  - Tham chiếu: Mùa xuân
  - Judge: Đáp án ứng viên không trả lời trực tiếp câu hỏi về thời điểm Hán Vương đảm nhiệm chức vụ. Nó tập trung vào các sự kiện diễn ra vào mùa hạ khi Hán Vương qua đời.

**Câu hỏi lỗi (ngoài KB / sai lệch)**

- `chat_00006` (single_hop) — *Phân tích chủ đề “Phong trào yêu nước đầu thế kỷ XX” trong bối cảnh triều Trần (1225–1400).*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Trong thời Trần, phong trào yêu nước đầu thế kỷ xx Đông Du, Duy Tân, Yên Thế… tạo nền tảng cho cách mạng sau này. Điều này góp phần định hình thể chế và bản sắc thời kỳ.
  - Judge: Câu hỏi gộp các sự kiện đầu thế kỷ XX vào bối cảnh triều Trần (1225-1400) là sai lệch lịch sử nghiêm trọng.

## Hệ thống: B1

- Tổng câu: **150** · sai: **118** (78.7%)

| Loại lỗi | Số lỗi | Tỉ lệ | Nguyên nhân gốc |
|---|---|---|---|
| Thiếu evidence | 1 | 0.8% | Chunk quá dài / embedding kém cho câu hỏi lịch sử |
| Sinh sai dù có evidence | 2 | 1.7% | Prompt chưa ràng buộc đủ / context quá dài |
| Từ chối vì truy xuất hụt | 114 | 96.6% | Đồ thị/vector chưa dựng được đường tới đáp án vàng |
| Trích dẫn không hợp lệ | 1 | 0.8% | Mô hình sinh [E#] không tồn tại / evidence thiếu text |

### Ví dụ

**Thiếu evidence**

- `llm_00005` (single_hop) — *Theo tài liệu về Hồ Chí Minh, Hồ Chí Minh là ai và có vai trò gì?*
  - Đáp án: Theo tài liệu, Hồ Chí Minh giữ các vai trò và chức vụ sau:
- Là nhà cách mạng, người cộng sản Việt Nam, nhà hoạt động độc lập, nhà dân tộc chủ nghĩa Việt Nam, nhà ngoại giao, chống chủ nghĩa đế quốc [E4].
- Là Bộ trưởng Bộ Ngoại giao Việt Nam, Bộ trưởng Việt Nam, Đại biểu Quốc hội Việt Nam khóa I, Ủ
  - Tham chiếu: Theo gia phả của dòng họ Nguyễn ở làng Sen, xã Kim Liên, tỉnh Nghệ An: :"Hoàng sơ tổ khảo là Thái bảo Nguyễn Bá Phụ, tổ đời thứ 2 là Nguyễn Bá Bạc, tổ đời thứ 3 là Nguyễn Bá Ban, tổ đời thứ 4 là Nguyễ
  - Judge: Đáp án chứa một số chức vụ không được context hỗ trợ trực tiếp. Do đó, điểm là 1 và grounded là false.

**Sinh sai dù có evidence**

- `llm_00006` (single_hop) — *Theo tài liệu về Đinh Tiên Hoàng, Đinh Tiên Hoàng là ai và có vai trò gì?*
  - Đáp án: Dựa trên các tài liệu được cung cấp, Đinh Tiên Hoàng (tên thật là Đinh Bộ Lĩnh) là người làng Đại Hoàng, động Hoa Lư, mồ côi cha từ bé và thuở nhỏ thường cùng bọn trẻ chăn trâu ngoài đồng [E3]. 

Về vai trò và sự nghiệp:
* Khi lớn lên, ông nương nhờ Trần Minh Công và làm chỉ huy của quân Trần, đánh
  - Tham chiếu: Theo Đại Việt Sử Ký Toàn Thư chép::"Vua mồ côi cha từ bé, mẹ họ Đàm đưa gia thuộc vào ở cạnh đền sơn thần trong động.
  - Judge: Ứng viên trả lời đúng phần thân thế từ context nhưng phần vai trò và sự nghiệp lại chứa thông tin không có trong context.
- `llm_00028` (temporal) — *Theo tài liệu về Lý Thái Tổ, điều gì đã xảy ra vào năm 1010?*
  - Đáp án: Vào năm 1010, các sự kiện sau đã diễn ra:
- Sau khi dời đô từ Hoa Lư về Thăng Long, vua xuất ra 2 vạn quan để làm tám sở chùa ở phủ Thiên Đức (tức Cổ Pháp) và dựng bia ghi công [E5].
- Tháng 12 năm Canh Tuất (năm 1010 dương lịch), vua sai sứ sang nước Tống để thỉnh kinh điển Phật giáo [E5].
- Sau kh
  - Tham chiếu: Tháng 7 năm Canh Tuất (1010), triều đình khởi sự dời đô.
  - Judge: Đáp án có chứa thông tin không có trong CONTEXT (về việc xây dựng chùa Thắng Nghiêm).

**Từ chối vì truy xuất hụt**

- `vhe_00002` (single_hop) — *Trong sự kiện do Đế Minh thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Đế Nghi
  - Judge: Đáp án không trả lời được câu hỏi và không cung cấp thông tin.
- `vhe_00005` (single_hop) — *Trong sự kiện do Thần Long thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Lạc Long Quân
  - Judge: Đáp án không trả lời được câu hỏi và không cung cấp thông tin.
- `vhe_00007` (single_hop) — *Trong sự kiện do Hùng Vương thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Văn Lang
  - Judge: Đáp án từ chối trả lời dù thông tin thực tế không có trong context.
- `vhe_00006` (single_hop) — *Ai là người thành lập Văn Lang?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Hùng Vương
  - Judge: Đáp án từ chối trả lời dù thông tin nằm ngoài context, nhưng thực tế Hùng Vương là đáp án chính xác.
- `vhe_00008` (single_hop) — *Ai là người cưới Sơn Tinh?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Mị Nương
  - Judge: Đáp án từ chối trả lời dù thông tin không có trong context.

**Trích dẫn không hợp lệ**

- `chat_00004` (single_hop) — *Phân tích ý nghĩa lịch sử của Toàn quốc kháng chiến (1946).*
  - Đáp án: Dựa trên ngữ cảnh được cung cấp, các thông tin liên quan đến cuộc kháng chiến có thể được phân tích như sau:

* Cuộc kháng chiến bùng nổ nhằm đáp trả các yêu cầu của quân Pháp có tính chất tước đoạt chủ quyền Việt Nam [E5].
* Cuộc chiến được ví như cuộc chiến giữa "hổ" và "voi", trong đó "hổ" sẽ nhả
  - Tham chiếu: Toàn quốc kháng chiến năm 1946 do Chính phủ Việt Nam Dân chủ Cộng hòa lãnh đạo: đối đầu Pháp. Ý nghĩa: mở đầu Chiến tranh Đông Dương lần thứ nhất.
  - Judge: Đáp án không trả lời trực tiếp câu hỏi về ý nghĩa lịch sử của Toàn quốc kháng chiến. Các thông tin cung cấp chỉ mô tả diễn biến và tính chất cuộc chiến.

## Hệ thống: B0

- Tổng câu: **150** · sai: **114** (76.0%)

| Loại lỗi | Số lỗi | Tỉ lệ | Nguyên nhân gốc |
|---|---|---|---|
| Thiếu evidence | 32 | 28.1% | Chunk quá dài / embedding kém cho câu hỏi lịch sử |
| Bịa ở câu ngoài KB | 8 | 7.0% | Thiếu cơ chế từ chối khi không có evidence |
| Từ chối vì truy xuất hụt | 74 | 64.9% | Đồ thị/vector chưa dựng được đường tới đáp án vàng |

### Ví dụ

**Thiếu evidence**

- `vhe_00004` (single_hop) — *Ai là người sinh ra Lạc Long Quân?*
  - Đáp án: Theo truyền thuyết, người sinh ra Lạc Long Quân là Thần Long Nữ.
  - Tham chiếu: Thần Long
  - Judge: Đáp án sai thông tin về tên gọi người sinh ra Lạc Long Quân.
- `vhe_00007` (single_hop) — *Trong sự kiện do Hùng Vương thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Trong các truyền thuyết về thời Hùng Vương, đối tượng thường được nhắc đến là các vị thần, yêu quái hoặc các thế lực thiên nhiên. Các nhân vật chính thường là các vua Hùng và trị vì qua nhiều đời. Các sự kiện tiêu trọng tâm thường xoạt quanh việc dựng nước và dựng lại nền văn văn hóa Đông Sơn.</div>
  - Tham chiếu: Văn Lang
  - Judge: Đáp án không trả lời trực tiếp vào đối tượng được nhắc đến trong đáp án tham chiếu là Văn Lang. Nội dung trả lời là kiến thức-ngoài-context-không-đúng-văng-văn.
- `vhe_00010` (single_hop) — *Ai là người tấn công Nam Chiếu?*
  - Đáp án: Trong lịch sử Việt Nam, không có sự kiện hay nhân vật lịch sử nổi tiếng nào trực sử tiếp tấn công vương quốc Nam Chiếu. Tuy nhiên, nếu xét về các cuộc tấn công vần quan hệ ngoại giao hoặc quân sự, các triều đại Việt Nam không có ghi chép rõ ràng về việc tấn công Nam Chiếu.
  - Tham chiếu: Cao Vương
  - Judge: Đáp án không cung cấp thông tin chính xác về người tấn công Nam Chiếu theo đáp án tham chiếu.
- `vhe_00006` (single_hop) — *Ai là người thành lập Văn Lang?*
  - Đáp án: Kinh Dương Vương là người được coi là thành lập nhà nước Văn Lang. Nhà nước này là nhà nước đầu tiên trong lịch sử Việt Nam. Thời kỳ này gắn liền với truyền thuyết Lạc Long Quân và Lạc Long Quân và Âu Cơ.
  - Tham chiếu: Hùng Vương
  - Judge: Đáp án sai thông tin về người thành lập Văn Lang. Kinh Dương Vương không phải là người thành lập Văn Lang theo đáp án tham chiếu.
- `vhe_00013` (single_hop) — *Trong sự kiện do Nhà Tần thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Trong sự kiện do Nhà Tần thực hiện, đối tượng được nhắc đến thường là các vùng đất hoặc các bộ tộc bị chinh phục. Điều này diễn quan ra trong quá trình bần hóa và own-ship của Tần Thủy Hoàng sau khi thống nhất Trung Hoa.
  - Tham chiếu: Nam Hải
  - Judge: Đáp án không trả lời trực tiếp vào nội dung cụ thể của đối tượng được nhắc đến. Thông tin về quá trình bần hóa và own-ship không có trong context.

**Bịa ở câu ngoài KB**

- `oob_00001` (out_of_kb) — *Thủ tướng Việt Nam hiện nay là ai?*
  - Đáp án: Thủ tướng Chính phủ nước Cộng hòa Xã hội Chủ nghĩa Việt Nam hiện nay là ông Phạm Minh Chính.
  - Tham chiếu: (không có)
  - Judge: out-of-KB: hệ thống cố trả lời câu hỏi ngoài phạm vi (hallucination)
- `oob_00002` (out_of_kb) — *Ai đoạt giải Nobel Văn học năm 2024?*
  - Đáp án: Người đoạt giải Nobel Văn học năm 2024 là nhà văn Han Kang. Bà là nhà văn Hàn Quốc đầu tiên giành được giải thưởng danh giá này.
  - Tham chiếu: (không có)
  - Judge: out-of-KB: hệ thống cố trả lời câu hỏi ngoài phạm vi (hallucination)
- `oob_00012` (out_of_kb) — *Tổng thống Mỹ hiện tại sinh năm nào?*
  - Đáp án: Tổng thống Mỹ hiện tại là Joe Biden, sinh năm 1942. Ông nhậm chức vào ngày 20 tháng 1 năm 2024. Tuy nhiên, thông tin về nhiệm kỳ-hiện tại có thể thay đổi tùy theo thời điểm truy vấn.
  - Tham chiếu: (không có)
  - Judge: out-of-KB: hệ thống cố trả lời câu hỏi ngoài phạm vi (hallucination)
- `oob_00020` (out_of_kb) — *Tốc độ giới hạn tối đa trên đường cao tốc tại Việt Nam là bao nhiêu km/h?*
  - Đáp án: Trên đường cao tốc tại Việt biển, tốc độ tối đa cho phép thường là 120 km/h. Tuy nhiên, tùy thuộc vào từng đoạn đường cụ thể và biển báo giao thông, tốc độ này có thể thay thể đổi thành 100 km/h hoặc thấp hơn. Người lái xe cần tuân thủ nghiêm ngặt các biển báo chỉ dẫn tại hiện trường.
  - Tham chiếu: (không có)
  - Judge: out-of-KB: hệ thống cố trả lời câu hỏi ngoài phạm vi (hallucination)
- `oob_00021` (out_of_kb) — *Trận chung kết World Cup 2022 diễn ra ở sân vận động nào?*
  - Đáp án: Trận chung kết World Cup 2022 diễn ra tại sân vận động Lusail ở Qatar. Đây là trận đấu giữa Argentina và Pháp. Trận đấu kết thúc với tỉ số hòa 3-3 sau hiệp phụ và Argentina thắng bằng loạt sút luân lưu.
  - Tham chiếu: (không có)
  - Judge: out-of-KB: hệ thống cố trả lời câu hỏi ngoài phạm vi (hallucination)

**Từ chối vì truy xuất hụt**

- `vhe_00005` (single_hop) — *Trong sự kiện do Thần Long thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Lạc Long Quân
  - Judge: Đáp án không trả lời được câu hỏi và không cung cấp thông tin hữu ích.
- `vhe_00011` (single_hop) — *Trong sự kiện do Cao Vương thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Nam Chiếu
  - Judge: Đáp án không trả lời được câu hỏi và không cung cấp thông tin hữu ích.
- `vhe_00002` (single_hop) — *Trong sự kiện do Đế Minh thực hiện, đối tượng được nhắc đến là gì?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Đế Nghi
  - Judge: Đáp án không cung cấp thông tin trả lời câu hỏi.
- `vhe_00012` (single_hop) — *Ai là người thành lập Nam Hải?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Nhà Tần
  - Judge: Đáp án không trả lời được câu hỏi và không cung cấp thông tin.
- `vhe_00017` (single_hop) — *Ai là người gặp gỡ Tể tướng Trần Bình?*
  - Đáp án: Tôi không tìm thấy đủ thông tin trong kho kiến thức để trả lời câu hỏi này.
  - Tham chiếu: Vua Hán
  - Judge: Đáp án không trả lời được câu hỏi và không cung cấp thông help tin.

