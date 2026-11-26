# Ôn bảo vệ luận văn GeoI-BQ

**Bắt đầu từ một chuyến đi.** Bạn đang tìm nhà thuốc trên đường về. Máy chủ có thể làm gì khi thấy một lượt hỏi? Sau mười phút, nó biết thêm gì? Còn bạn có tìm được nhà thuốc phù hợp để ghé không? Ba câu này dẫn tự nhiên từ bài toán của Chow và Mokbel sang chứng minh toàn phiên, rồi sang thí nghiệm tìm địa điểm của luận văn.

**Lần ôn đầu, khoảng 45 phút.** Dành 10 phút vẽ những gì điện thoại biết và máy chủ thấy. Dành 15 phút đọc hai bài của nhóm Mokbel về truy vấn theo vùng và truy vấn liên tục. Dành 10 phút giải thích vì sao năm người, năm điểm giả và năm địa điểm trả về là ba khái niệm khác nhau. Dành 10 phút tự nói đóng góp của mình và ghi lại câu đầu tiên mà bạn phải mở tài liệu mới trả lời được. [Tài liệu đọc theo lịch sử](01_lich_su_va_huong_nghien_cuu.md)

| Khi bạn muốn… | Đọc hoặc tập ở đây |
|---|---|
| Hiểu bài toán từ Mokbel đến các nhánh nghiên cứu và triển khai gần đây | [1 · Lịch sử và hướng nghiên cứu](01_lich_su_va_huong_nghien_cuu.md) |
| Đứng trước một paper và nói được nó giải bài toán gì, dùng thước đo nào, liên quan thế nào đến mình | [2 · Đối chiếu paper và câu hỏi theo nguồn](02_doc_paper_va_doi_chieu.md) |
| Tự dựng lại cơ chế và chứng minh riêng tư, chất lượng dịch vụ | [3 · Phương pháp và chứng minh](03_phuong_phap_va_chung_minh.md) |
| Đọc một dòng benchmark, giải thích mẫu số, đối chứng, khoảng tin cậy và giới hạn | [4 · Thực nghiệm và phản biện](04_thuc_nghiem_va_phan_bien.md) |
| Tích tiến độ hằng ngày hoặc tra nhanh bản luận văn | [Checklist 40 việc](../CHECKLIST_BAO_VE.md) · [Luận văn hiện hành](../../artifacts/reports/graduation_thesis.pdf) |

**Bộ 111 câu có đáp án** gồm 10 câu lịch sử, 24 câu về các paper, 41 câu về phương pháp và 36 câu về thực nghiệm. Nếu tối nay chỉ ôn được **sáu câu**, hãy lấy câu 2 và 4 ở chuyên đề lịch sử (năm người khác năm Q thế nào, Casper trả đúng kết quả ra sao), P02 và U10 ở chuyên đề chứng minh (vì sao cần Z rồi mới Q, dữ liệu nói gì về belief), câu 11 và 29 ở chuyên đề thực nghiệm (đọc 0,188% và phân biệt S5 với S6). Sáu câu này buộc bạn nối tài liệu gốc với công việc của chính mình.

## Nhận xét sau khi đọc lại bản 71 trang

**Theo đánh giá của tôi, bạn đã có một câu chuyện nghiên cứu đủ rõ để dành thời gian còn lại cho phần bảo vệ.** Mạnh nhất là ba mắt xích có thể kiểm tra riêng: xác định người quan sát muốn đoán gì, giới hạn thông tin mà cả phiên có thể tiết lộ, rồi đo dịch vụ người dùng thực nhận. Điểm yếu nhất là việc quy nguyên nhân cho thuật toán khi danh mục địa điểm nhỏ và khi một số tác vụ vốn đã khó ngay cả với GPS thật. Sáu điểm dưới đây đáng giải quyết khi rà bản nộp hoặc luyện nói.

| Ưu tiên | Chỗ hội đồng dễ hỏi | Cách trình bày hoặc chỉnh khi chốt bản nộp |
|---|---|---|
| Trước khi nộp | Chương 4 mô tả gộp theo mốc, chuyến rồi nhóm. Khi số mốc có đáp án khác nhau, người đọc có thể hiểu mỗi chuyến có cùng trọng số. | Với Recall, trong mỗi nhóm tuyến và từng mục đích, gộp các mốc hợp lệ qua các chuyến/lần lấy mẫu. Sau đó lấy trung bình đều theo mục đích và nhóm. Ví dụ khác biệt nằm ở mục đích tìm trong bán kính. Ghi rõ cách gộp này cạnh định nghĩa và khoảng tin cậy. [Quy trình](../chapters/experiments.tex) · [Đối chiếu chi tiết](04_thuc_nghiem_va_phan_bien.md) |
| Trước khi nộp | Câu “AnotherMe offline” ở Chương 4 có thể bị hiểu là mô tả bài gốc. Trạng thái PTPPM cũng đã đổi sau mốc tổng quan. | AnotherMe gốc sinh quỹ đạo **trực tuyến**. Bản thử lịch sử của nhóm dùng cả đoạn. IEEE ghi ngày công bố đầu tiên 11/09/2023, trước cửa sổ khảo sát dù số tạp chí mang năm 2024. PTPPM có preprint 2025 trong cửa sổ và bản TIFS ngày 05/11/2026 sau mốc chốt 28/10. Khi thuyết trình, ghi rõ phiên bản đã đọc. [AnotherMe](https://ieeexplore.ieee.org/document/10246991/) · [PTPPM trên arXiv](https://arxiv.org/abs/2511.21020) · [Bản tạp chí](https://doi.org/10.1109/TIFS.2026.3750736) |
| Mở đầu | Chương 2 chủ yếu so công trình gần đây. Người nghe có thể tưởng S1–S10 là phân loại của Mokbel, hoặc nhóm đề xuất ý tưởng xếp hạng cục bộ đầu tiên. | Dùng một hình vùng che phủ rồi nói cách quan sát nhiều lần làm thay đổi bài toán. Ghi công Chow–Mokbel cho khung snapshot/continuous và công bố dữ liệu. S1–S10 là cách **luận văn** cụ thể hóa các mục tiêu để chấm. EV querying đã có nhiều điểm truy vấn và chọn cục bộ. [Chow–Mokbel](https://www-users.cse.umn.edu/~mokbel/papers/SIGKDD2011.pdf) · [EV querying](https://wrap.warwick.ac.uk/id/eprint/183198/) |
| Khi trình bày định lý | “70,91%” dễ nghe như độ chính xác thực nghiệm hoặc xác suất người dùng an toàn. | Vẽ hai tuyến A/B có prior 50/50. Với các giả thiết của định lý, transcript của một phiên chỉ có thể nhân odds với một hệ số nhỏ hơn 2,437372 ở cấu hình minh họa. Đổi prior hoặc thêm kênh quan sát sẽ đổi cách đọc con số. [Phân tích riêng tư](../chapters/privacy.tex) |
| Khi trình bày utility | Recall 99,595% dễ khiến người nghe tin rằng belief nhất thiết giúp chọn Q tốt hơn. | Phương án chỉ dùng thông tin công khai đạt 99,711% trong phép so thành phần. Ngay gói đầu đã thu khoảng 91% của 418 địa điểm. Kết quả đo được là truy hồi tốt với cấu hình này. Lợi ích riêng của belief cần một phép thử phân biệt hai cách chọn Q tốt hơn. [Phân tích vai trò các thành phần](../chapters/discussion.tex) |
| Khi kết luận | “Bảo vệ cả quỹ đạo” dễ bị nghe thành “xóa danh tính và bảo vệ mọi chuyến ngoài thực tế”. | Nêu đối tượng là chuỗi quan sát trong **một phiên theo mô hình**. Người quan sát vẫn có thể đã biết địa chỉ nhà hoặc tài khoản. Phân rõ S4 theo nhóm tuyến, S5 có đối chứng yếu, S7 thay yêu cầu trên cùng chuyến và S8 cho thêm kênh của người đi cùng. [Kết quả theo kịch bản](../chapters/scenario_results.tex) |

## Lịch từ 26/11 đến ngày bảo vệ

Mốc làm việc theo lịch bạn đã dùng là **nộp ngày 14/12/2026**, hoàn tất phần cốt lõi trước 15/12 và phần đào sâu đến 20/12. Ngày bảo vệ chính thức chưa chốt. Các số 8/40 đến 40/40 là số **việc hoàn thành trên checklist**, không phải tỷ lệ kiến thức đã thuộc.

| Thời gian | Nội dung và đầu ra tự kiểm | Mốc |
|---|---|---|
| 26–29/11 | Kể lịch sử trong hai phút. Vẽ Casper cạnh GeoI-BQ. Mỗi ngày đọc sâu một trong ba đối chứng hoặc EV querying. Với mỗi bài ghi đúng hình/công thức, đầu ra, thước đo, điểm khác ở bản đã chạy. | A01–08 · 8/40 |
| 30/11–05/12 | 30/11 vẽ một chuyến; 01/12 tự chứng minh REM; 02/12 tính ngân sách giữ/đổi; 03/12 tính belief và xác suất chọn đoạn; 04/12 nối bốn mệnh đề; 05/12 tìm một phản ví dụ cho việc điểm lập kế hoạch cao kéo theo Recall cao. | B01–08 · 16/40 |
| 06–10/12 | Lần lượt học dữ liệu/chia tập, năm tác vụ định vị chính, bốn mục đích dịch vụ, S4–S8, cuối cùng là cách đo gốc và chi phí. Tự đọc một dòng bảng được chọn ngẫu nhiên, ghi mẫu số và một kết luận có thể rút ra. | C01–08 · 24/40 |
| 11–13/12 | Hôm đầu nói đủ bài trong 20 phút. Hôm sau nhờ người khác hỏi liên tiếp một bộ đề bên dưới. Hôm cuối đối chiếu ba câu yếu nhất với nguồn, ôn lại và rà các điểm trước khi nộp. | D01–07 · 31/40 |
| 14/12 | Chốt bản nộp, mở lại đúng PDF và lưu xác nhận nộp. | D08 · **32/40** |
| 15–17/12 | Tự suy ra cận cho prior lệch, hai kênh tương quan, ví dụ cận utility và phản ví dụ khi tập đầu ra thay đổi theo GPS. | E01–04 · 36/40 |
| 18–20/12 | Đọc nhánh chuyên sâu còn yếu nhất. Bốc thăm một paper, soạn thí nghiệm tiếp theo, diễn tập với câu hỏi bất ngờ và hình tĩnh dự phòng. | E05–08 · **40/40** |
| Từ 21/12 tới buổi bảo vệ | Mỗi lần ôn chọn một hình, một công thức và một bảng. Dành thời gian còn lại cho câu chưa giải thích được. | Duy trì |

**Nhịp một buổi 45–60 phút.** Đóng tài liệu và nói thử trong 5 phút. Đọc đúng phần bị vướng trong 15–20 phút. Dành 15 phút vẽ lại, tính tay hoặc giải thích một bảng. Cuối buổi thu âm hai câu, mỗi câu tối đa 90 giây. Hai ngày sau trả lời lại một trong hai câu mà không mở đáp án.

## Một bài nói 20 phút

| Phút | Thứ hội đồng cần nhìn | Ý cần nói thành tiếng |
|---|---|---|
| 0–3 | Một chuyến tìm nhà thuốc, hai nhánh đường và người quan sát | Các lần hỏi đều phục vụ một nhu cầu chính đáng. Khi ghép theo thời gian, chúng cũng cho biết nhánh nào người dùng có khả năng đã đi qua. |
| 3–5 | Vùng che phủ của Casper, một gói thật + giả và gói Q của mình | Đầu ra khác nhau dẫn tới cách chấm khác nhau. Chỉ vào tác vụ chung “đoán GPS thật” và đáp án dịch vụ chung “tìm đúng địa điểm”. |
| 5–8 | Một hình kiến trúc và một đoạn di chuyển 0–20–40–60 giây | Điện thoại giữ Z và belief. Máy chủ thấy các khung Q. Tại giây 60, nói riêng việc kiểm tra Z và chọn đoạn tiếp theo. |
| 8–12 | Bốn mệnh đề nối thành hai cận trên cùng quan sát | Nêu minh họa hai tuyến 50/50, vì sao quyết định giữ Z cũng tốn ngân sách và vì sao các khung của một đoạn đã chọn chỉ là hậu xử lý. |
| 12–14 | Một điểm dịch vụ xuất hiện trong phản hồi trước | Cache và xếp hạng tại GPS thật có thể lấy lại đáp án. Một địa điểm chưa từng được trả về sẽ không xuất hiện dù bộ xếp hạng rất tốt. |
| 14–18 | Bảng định vị, Recall và chi phí trên cùng cửa sổ | Dùng S1 và S3 của Fake-query* để cho thấy tác dụng của việc xem cả chuỗi. Đọc một hàng của GeoI-BQ với 20 nhóm kiểm tra, bao gồm cả chênh lệch utility bất lợi và 155 truy vấn tọa độ mỗi phiên. |
| 18–20 | Ba đóng góp và một thí nghiệm tiếp theo | Chốt bằng khung đánh giá, giới hạn suy luận có điều kiện và bằng chứng thực nghiệm. Thí nghiệm bạn ưu tiên có thể là cố định cơ chế rồi thay điều kiện quan sát và đo trên nhóm tuyến mới. |

## Ba đề vấn đáp thử

Mỗi đề làm trong 8 phút. Người hỏi chọn ngẫu nhiên thứ tự và hỏi tiếp “Dữ liệu hoặc giả thiết nào cho phép em nói vậy?”. Che cột đáp án trong các chuyên đề trước khi tập.

| Người hỏi | Chuỗi câu hỏi | Dấu hiệu trả lời đạt |
|---|---|---|
| Thầy cô về cơ sở dữ liệu | Casper giải truy vấn gần nhất khi máy chủ chỉ biết vùng bằng cách nào? Điểm nào còn giống ở GeoI-BQ? Nếu tải toàn danh mục thì vì sao cần truy vấn? | Nêu đúng bên tinh lọc kết quả trong từng thiết kế. Thừa nhận danh mục tĩnh đầy đủ giải được tác vụ cục bộ; nêu chi phí lấy và làm mới cần đo. |
| Thầy cô về xác suất | Nếu đã giấu Z, sao còn chứng minh REM? Nếu chọn đoạn Q xa GPS thật thì có thể bỏ phân tích phân phối không? Nếu prior đã là 90/10, 70,91% còn đúng không? | Vẽ hai con đường chặn cùng transcript. Tự đổi odds 9:1 thành cận khoảng 21,94:1, tương ứng khoảng 95,64%, với đúng điều kiện của ví dụ. |
| Thầy cô về thực nghiệm | Nếu bỏ belief thì kết quả có tệ hơn? Nếu dùng mô hình suy luận khác thì sao? Nếu người đi cùng cung cấp GPS thật thì sao? | Tra được hàng Γ=0, giải thích mô hình chọn trên SELECTION và tách thông tin phụ ra khỏi chuỗi do cơ chế bảo vệ. Nêu phép thử mới có thể làm mình đổi kết luận. |

**Tự chấm một câu.** 0 = chưa nói được. 1 = nhớ kết luận. 2 = giải thích được bằng ví dụ kèm giả thiết. 3 = đối chiếu được bài gốc, định lý hoặc bảng và trả lời được một thay đổi điều kiện. Với câu cốt lõi, đặt mục tiêu ít nhất 2 điểm ở hai buổi khác nhau. Câu đạt 1 điểm được đưa lên đầu buổi sau.
