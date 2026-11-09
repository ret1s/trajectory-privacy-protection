# Hướng dẫn review bản luận văn hợp nhất

Bản chính: [PDF](../artifacts/reports/graduation_thesis.pdf), [LaTeX](main.tex).
Nội dung theo năm chương đã chốt; các báo cáo cũ còn nguyên.

## Thứ tự review

1. **Chương 1:** bài toán transcript LBS; ba RQs privacy, utility, trade-off.
   Kiểm tra scope và đóng góp có đúng ý định đề tài.
2. **Chương 2:** bốn nhóm subsection dẫn từ cơ chế/output paper tới khoảng trống.
   “Fully” chỉ tác vụ trực tiếp trong giả định, không là bảo mật tuyệt đối.
3. **Chương 3.1–3.4:** system boundary; GPS → Z → belief → đoạn Q → POI/cache
   → local ranking. Hai clock và hai ngân sách có vai trò khác nhau.
4. **Chương 3.5:** bốn mệnh đề trả lời: một Z an toàn thế nào; giữ/đổi tốn gì;
   một đoạn Q tiết lộ bao nhiêu; nhiều đoạn cộng dồn ra sao. Định lý cuối lấy minimum
   hai cận cho cùng transcript, không cộng đơn vị khác nhau.
5. **Chương 3.6:** phân biệt gain score, actual Recall và matched public base.
   Cache giúp tại cùng GPS/request; unknown proxy error không được đặt bằng 0.
6. **Chương 4:** main paper comparison trước, rồi S4–S8 và validity. Xem CI,
   selected attackers và failure trace ở phụ lục khi cần kiểm tra chi tiết.
7. **Chương 5:** mỗi RQ có câu trả lời; không có claim mạnh hơn định lý/dữ liệu.

## Những ý cần giữ khi tự viết lại

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
- Float upstream, nguồn random thực và full restore chưa đủ chứng nhận deployment
  pure-DP. Định lý áp cho kernel và các giả thiết đã nêu.
- S4 chưa là danh tính người/xe; S5 thiếu positive control mạnh; S6 là hai đích biết
  trước; S7 là conditional payload; S8 là hai kênh tổng hợp sau căn clock.

## Nơi sửa

- `chapters/introduction.tex`, `related_work.tex`: động lực/literature.
- `chapters/method.tex`: đặc tả, hai kiến trúc và ví dụ.
- `chapters/privacy.tex`, `utility.tex`: phát biểu và chứng minh.
- `chapters/experiments.tex`, `scenario_results.tex`, `discussion.tex`: đánh giá.
- `chapters/summary.tex`, `appendices.tex`, `frontmatter.tex`: đầu/cuối tài liệu.
- `generated/*.tex`: bảng từ artifact. Không sửa tay score; sửa qua exporter/audit.

Đây là bản thảo để học viên review và diễn đạt lại. Biểu mẫu hành chính, xác nhận
và quy định nộp cuối cần theo hướng dẫn của trường/GVHD.
