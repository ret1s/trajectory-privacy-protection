# Hướng dẫn review bản luận văn hợp nhất

Bản chính: [PDF](../artifacts/reports/graduation_thesis.pdf), [LaTeX](main.tex).
Nội dung theo năm chương đã chốt; các báo cáo cũ còn nguyên.

## Thứ tự review

Revision 16/11 ưu tiên ba câu hỏi: thiết bị biết dữ liệu gì trước truy vấn; Geo-I/belief/Q
mỗi phần đóng góp gì; và định lý chuyển thành cận suy luận/utility dưới điều kiện nào.
Bản này bổ sung chứng minh và audit nguồn, không cải thiện benchmark bằng cách đổi cấu hình.

1. **Chương 1:** bài toán transcript LBS; ba RQs privacy, utility, trade-off.
   Kiểm tra scope và đóng góp có đúng ý định đề tài.
2. **Chương 2:** bốn nhóm subsection dẫn từ cơ chế/output paper tới khoảng trống.
   “Fully” chỉ tác vụ trực tiếp trong giả định, không là bảo mật tuyệt đối.
3. **Chương 3.1–3.4:** system boundary; GPS → Z → belief → đoạn Q → POI/cache
   → local ranking. Hai clock và hai ngân sách có vai trò khác nhau.
4. **Chương 3.5:** bốn mệnh đề trả lời: một Z an toàn thế nào; giữ/đổi tốn gì;
   một đoạn Q tiết lộ bao nhiêu; nhiều đoạn cộng dồn ra sao. Định lý cuối lấy minimum
   hai cận cho cùng transcript, không cộng đơn vị khác nhau.
   Sau đó xem hệ quả Hit theo prior và composition nhiều kênh: không gán prior bằng 1/K,
   không dùng cận một phiên cho S4/S8 joint. Ví dụ 70,91% và 85,59% là cận có điều kiện,
   không phải accuracy đo được.
5. **Chương 3.6:** phân biệt gain score, actual Recall và matched public base.
   Cache giúp tại cùng GPS/request; unknown proxy error không được đặt bằng 0.
   Ví dụ hữu tỷ cho cả gain dương, actual gain âm và floor 85% trên miền toy được khai báo;
   đó không phải floor 85% cho người dùng hoặc cohort thực.
6. **Chương 4:** main paper comparison trước, rồi S4–S8 và validity. Xem CI,
   selected attackers và failure trace ở phụ lục khi cần kiểm tra chi tiết.
7. **Chương 5:** mỗi RQ có câu trả lời; không có claim mạnh hơn định lý/dữ liệu.

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
