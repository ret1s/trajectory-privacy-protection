# GeoI-BQ: bản tiếng Việt để đọc và chỉnh sửa

Bản dịch riêng của `paper_hk261/` (bản HK261 tiếng Anh chốt ngày 19/11/2026).
Giữ bố cục A4 hai cột, năm phần nội dung, ba hình, phần giới thiệu S1–S10 và ba bảng so sánh. Không thay đổi
thuật toán, công thức, cấu hình, số liệu hay kết luận thực nghiệm. Giữ nguyên
tên tác giả, cơ quan bằng tên chính thức tiếng Anh, email và tài liệu tham khảo.
Hanh Ngo-Minh là tác giả liên hệ duy nhất. Bản tiếng Anh không bị ghi đè.

## Chỉnh sửa trên Overleaf

Upload `latex_sources.zip` tại
`artifacts/papers/geoi_bq_hk261_vi_20261119_v1/` bằng **New Project → Upload Project**.
Chọn **Main document: main.tex**, **Compiler: XeLaTeX**. Font TeX Gyre Termes/Heros
có trong TeX Live, không cần font riêng từ máy cá nhân.

Các file nội dung để review:

- `abstract.tex`, `introduction.tex`, `related_work.tex`: tóm tắt, động lực và nghiên cứu liên quan.
- `method.tex`: hệ thống, giả thiết, cơ chế và ví dụ nhiều bước.
- `analysis.tex`: bốn mệnh đề privacy, định lý toàn transcript và phân tích utility.
- `evaluation.tex`, `conclusion.tex`: thực nghiệm, giới hạn và kết luận.
- `figure_system.tex`, `figure_model.tex`: hai sơ đồ chỉnh sửa trực tiếp trong LaTeX.
- `table_*.tex`: bảng nghiên cứu liên quan và số liệu benchmark.

Các nhãn trong ví dụ bản đồ cũng được dịch; mẫu `cache-046_day7`, tọa độ,
mạng đường và các giá trị đều từ artifact tiếng Anh đã lưu, không chạy lại mô hình.

## Quy ước đọc

- **Transcript:** toàn bộ chuỗi dữ liệu bên quan sát nhìn thấy trên kênh truyền.
- **Belief:** phân phối xác suất vị trí do thiết bị ước lượng từ thông tin đã bảo vệ.
- **Prior/posterior:** xác suất trước/sau khi quan sát transcript.
- **Miền hỗ trợ chung:** các đầu ra có thể được chọn phải cùng khả dụng khi so sánh hai lịch sử.
- **Kernel:** quy tắc xác suất tạo đầu ra; **PMF:** hàm khối xác suất rời rạc.
- **Utility:** chất lượng đáp án dịch vụ; điểm lập kế hoạch chỉ là đại diện.
- **pp:** điểm phần trăm; **CI:** khoảng tin cậy; **N/A:** không áp dụng hoặc thiếu đầu vào phù hợp.

Giữ dấu chấm thập phân như artifact gốc để đối chiếu số liệu. Tên phương pháp,
ký hiệu, nhãn S1–S10 và các mục tài liệu tham khảo không dịch.

Các chỗ nhắc “tài liệu bổ sung” chỉ tới supplement của bộ paper tiếng Anh tại
`artifacts/papers/geoi_bq_20261116_v1/supplement.pdf`; gói này chỉ dịch bài chính.
Không thêm hai mục availability/AI declaration đã được bỏ khỏi bản review tiếng Anh.
Đây là bản làm việc để tác giả đọc, cân nhắc và viết lại bằng lời văn của mình.

## Dựng và kiểm tra trong repository

```bash
python experiments/export_vietnamese_walkthrough_20261119.py
python experiments/build_vietnamese_paper_20261119.py --tectonic /path/to/tectonic
python experiments/build_vietnamese_paper_20261119.py --check
```

Builder kiểm tra công thức, số liệu bảng, nhãn/trích dẫn, 18 tài liệu tham khảo,
nguồn HK261, bản tiếng Anh không đổi và biên dịch độc lập ZIP giao cho Overleaf.

Để sửa theo lời văn của mình, mở `scenarios.tex` (cách đặt bài toán, S1–S10) và `workload.tex` (lý do xây dựng dữ liệu so sánh).
