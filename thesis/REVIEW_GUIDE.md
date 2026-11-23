# Hướng dẫn review bản luận văn hợp nhất

Bản chính: [PDF](../artifacts/reports/graduation_thesis.pdf), [LaTeX](main.tex).
Nội dung theo năm chương đã chốt; các báo cáo cũ còn nguyên.

## Tự biên tập theo từng chương

Có hai loại câu thường đứng cạnh nhau. Câu “vì sao tôi chọn phép so sánh này” thể hiện quyết định nghiên cứu của tác giả. Câu “phép so sánh chấm đại lượng nào, trên mẫu nào” là đặc tả để người khác có thể lặp lại. Hãy viết câu đầu bằng cách bạn giải thích với GVHD, rồi đối chiếu câu sau với công thức hoặc bảng nguồn.

| Vị trí trong bản thảo | Đoạn nên viết theo giọng của bạn | Phần cần giữ chính xác |
|---|---|---|
| [Tóm tắt](chapters/frontmatter.tex), hai đoạn đầu | Sửa sau cùng. Chọn một câu nêu vấn đề, một câu mô tả cách bạn giải quyết. | Phạm vi một phiên, năm tác vụ định vị, mẫu số, đơn vị, CI và cấu hình của kết quả trích dẫn. |
| [Chương 1](chapters/introduction.tex), “Bối cảnh và động lực” và “Hệ thống hóa mục tiêu bảo vệ” | Kể tình huống tìm một POI theo cách bạn vẫn giải thích. Đặt câu hỏi bạn từng hỏi về việc máy chủ chỉ thấy Q. Ví dụ giả định nên được giới thiệu là ví dụ. | Mỗi mã S đi cùng đúng mục tiêu và quyền quan sát. S2 đo tọa độ đại diện, S4 có nhãn nhóm tuyến. |
| [Chương 2](chapters/related_work.tex), đoạn ngay sau bảng tổng hợp | Chọn hai phương pháp và giải thích vì sao câu hỏi “điểm nào là thật?” có ý nghĩa khác nhau giữa chúng. | Tên phương pháp, trích dẫn, định nghĩa thước đo gốc, phạm vi tác vụ và thành phần thay thế trong bản đối chứng. |
| [Chương 3](chapters/method.tex), phần mở chương và lời dẫn hai sơ đồ | Giải thích vì sao thiết bị giữ Z và gửi Q. Bạn có thể đi từ góc nhìn của người đang cần tìm địa điểm rồi chuyển sang góc nhìn nhà cung cấp. | Thứ tự xử lý, dữ liệu nào nằm trên thiết bị, hai đồng hồ, đơn vị ngân sách và trạng thái được lưu qua lần khởi động lại. |
| [Phần riêng tư](chapters/privacy.tex) và [phần chất lượng dịch vụ](chapters/utility.tex) | Viết lời dẫn trước chứng minh và một câu sau mỗi kết quả để giải thích hệ quả. Chẳng hạn, vì sao giữ Z cũ cũng có thể mang thông tin? Vì sao thêm POI vào cache chỉ giúp tại cùng vị trí và yêu cầu? | Giả thiết, lượng từ, điều kiện hóa, miền hỗ trợ, chiều bất đẳng thức, chỉ số thời gian và đơn vị. Khi sửa phát biểu toán học, rà cả chứng minh và các hệ quả viện dẫn nó. |
| [Chương 4](chapters/experiments.tex), đoạn đầu mục xây dựng dữ liệu và đoạn sau bảng kết quả | Giải thích lựa chọn cùng cửa sổ thời gian. Sau bảng, chọn một đối chứng để kể điều bạn học được, như sự khác biệt giữa S1 và S3 của Fake queries*. | Dấu sao của bản điều chỉnh, chia nhóm 20/8/20, 120 lượt TEST mỗi phương án, cách chọn bộ suy luận và đơn vị bootstrap. Giữ dấu của chênh lệch Recall và CI ghép cặp. |
| [Thảo luận](chapters/discussion.tex) và [Chương 5](chapters/summary.tex) | Đây là nơi cần phán đoán của bạn nhất. Đánh đổi nào đáng chấp nhận cho ứng dụng dự kiến? Kết quả nào làm bạn muốn thử một danh mục POI lớn hơn? Viết nhận định sau con số làm căn cứ. | Phân biệt kết quả đo, hệ quả của định lý và giả thuyết cho nghiên cứu tiếp theo. Theo dõi các giới hạn về danh mục tĩnh, ngân sách chưa cạn và quyền quan sát của người đi cùng. |

**Ví dụ tự sửa một đoạn.** Trước bảng định vị, giữ nguyên định nghĩa “Hit100 là tỷ lệ ước lượng nằm trong bán kính 100 m”. Sau bảng, bạn có thể diễn giải bằng câu “Khi người quan sát được xem cả chuyến, các gói giả có còn gây nhầm lẫn như ở lần gửi đầu tiên không?”. Khi trả lời, đối chiếu đúng hai cột S1 và S3 của cùng một phương án. Nếu nêu nguyên nhân, cho biết đó là điều thiết kế thí nghiệm kiểm tra được hay mới là cách bạn diễn giải.

**Thứ tự đọc khi ít thời gian:** đọc mở đầu Chương 1, xem hai sơ đồ ở Chương 3, đọc định lý toàn phiên cùng hệ quả Bayes, rồi đọc hai bảng định vị và Recall ở Chương 4. Quay lại Chương 5 và kiểm tra xem mỗi câu kết luận có thể chỉ tới đúng một định lý hoặc một kết quả hay không.

## Những ý cần giữ khi tự viết lại

- **Quyền dữ liệu là giả thiết trọng tâm.** Client prototype nạp full public context;
  đánh giá dùng received-mask contract. Cần tách proxy thiết kế offline với metadata
  hiện hành chưa biết trước khi khẳng định online queries cần thiết.
- **Hai cận có vai trò khác nhau.** Với cấu hình hiện tại, envelope thô Geo-I/Q giao ở
  khoảng 3,874 m; guard Q cho cận nhỏ hơn ở khoảng cách lớn hơn. Không quy mọi bảo đảm cho Geo-I.
- Phương pháp cuối là **GeoI-BQ v1**. Thử nghiệm phát triển chỉ dùng làm ablation.
- Đối chứng chính là **ba paper adaptations**, kèm Raw/REM-Z. Dấu `*` phải đi cùng
  phần predictor/data thay thế; chưa phải official author reproduction.
- Recall **99,595%** cao trên workload này nhưng thấp hơn adaptations khoảng
  **0,4 điểm phần trăm**. Catalogue nhỏ/cache tĩnh giúp bão hòa. Cost **155 point
  queries/600 s** chưa gồm toàn metadata/network/energy.
- Main Hit100 **0–0,188%** là readout hữu hạn. Không-hit/CI `[0;0]` không chứng minh
  risk bằng 0. Hit và MAE có thể dùng hai selected estimators khác nhau.
- Public base `Γ=0` có utility cao hơn `Γ=1` ở cohort này. Contribution là bounded
  adaptation và phân tích được; chưa chứng minh adaptation luôn cần thiết.
- Audit toàn bộ 120 TEST draws: packet đầu nhận trung bình 380,2/418 POI; cuối phiên 415,875/418.
  Không trace nào chạm cap 23 (chi phí tối đa 21), nên utility sau cap exhaustion còn chưa đo.
- Float upstream, nguồn random thực và full restore chưa đủ chứng nhận deployment
  pure-DP. Định lý áp cho kernel và các giả thiết đã nêu.
- S4 chưa là danh tính người/xe; S5 thiếu positive control mạnh; S6 là hai đích biết
  trước; S7 là conditional payload; S8 là hai kênh tổng hợp sau căn clock.

## Nơi sửa

- `chapters/introduction.tex`, `related_work.tex`: động lực/literature.
- `chapters/method.tex`: đặc tả, hai kiến trúc và ví dụ.
- `chapters/component_roles.tex`: attribution của các thành phần và chính sách tham số.
- `chapters/privacy.tex`, `utility.tex`: phát biểu và chứng minh.
- `chapters/experiments.tex`, `scenario_results.tex`, `discussion.tex`: đánh giá.
- `chapters/summary.tex`, `appendices.tex`, `frontmatter.tex`: đầu/cuối tài liệu.
- `generated/*.tex`: bảng từ artifact. Không sửa tay score; sửa qua exporter/audit.

Đây là bản thảo để học viên review và diễn đạt lại. Biểu mẫu hành chính, xác nhận
và quy định nộp cuối cần theo hướng dẫn của trường/GVHD.
