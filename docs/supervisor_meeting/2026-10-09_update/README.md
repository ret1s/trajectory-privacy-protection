# Bảo vệ riêng tư truy vấn và quỹ đạo bằng Geo-I/REM

Tài liệu chính là **report LaTeX 6 trang A4**, dùng trình bày khoảng **8–10 phút**. Font serif TeX Gyre Termes, nội dung 10 pt, bảng khoa học và sơ đồ vector.

- [report_explained.pdf](report_explained.pdf): bản đọc/trình bày.
- [report_explained.tex](report_explained.tex): nguồn chỉnh sửa được.
- [presentation_script.md](presentation_script.md): lời trình bày theo sáu trang.
- [speaker_notes.txt](speaker_notes.txt): ghi chú đồng bộ.

Thứ tự nội dung:

1. S7: request chung L30, nhu cầu giữ local; bốn tiêu chí; ngân sách Geo-I mỗi phiên.
2. **S4–S6 riêng:** target, cơ chế, likelihood/posterior odds và cận Bayes; phân biệt người/xe, prefix cạnh, lịch sử/đích; ví dụ lý thuyết và điều kiện.
3. Kiến trúc trước: bốn tầng, luồng đánh số, delay đầu/cuối tùy chọn.
4. Kiến trúc hiện tại: giữ Geo-I, L30/bốn nhu cầu local; không thêm một khối identity chưa được chọn.
5. Sample đã lưu: timeline, bản đồ, kích hoạt cơ chế và kết quả local; ghi đúng tham số mẫu.
6. Benchmark: L20–L30 theo bốn nhu cầu/CI/bytes; bảng tọa độ S1–S3 và đầu/cuối S9–S10 theo cấu hình riêng.

## Phạm vi kết luận

**S7:** năm Q × L30 cho tối đa 150 bản ghi **mỗi loại** trước bỏ trùng. Thiết bị nhận, gộp, bỏ trùng và lọc/sắp xếp theo GPS/ψ. Đổi nhu cầu không đổi request khi cùng trạng thái bảo vệ, lịch và ngữ cảnh công khai. Triển khai/benchmark vẫn dùng k5 và Recall@5; không tự nhận đã bỏ giới hạn trong mã.

**Cap phiên:** C_s=0,23 m⁻¹; H12/U23; u=0,01 m⁻¹; B danh nghĩa=0,24 m⁻¹. Khởi động lại cùng chuyến không nạp cap. H không là số GPS reads tối đa. Những phiên bị liên kết vẫn hợp thành tổng cap; chưa có readout mới cho kết hợp cap làm việc/L30. [Chính sách và đánh giá](../../research/2026-10-10_session_cap_identity.md); [cấu hình](../../research/2026-10-10_session_cap_identity.json).

**S4–S6:** phần chính chuyển sang phân tích cận của Geo-I hiện có: likelihood, mức thay đổi posterior odds, TV và Bayes theo prior. S4 phân biệt người/xe; S5 dùng prefix trước dự báo; S6 tính lịch sử đã liên kết và prior routine. Điều kiện all-pair trên hai support phải xác định; K5 không phải k-anonymity. Không chọn lớp identity mới. Diagnostic AUC/accuracy đã lưu vẫn ở tài liệu nghiên cứu, không phải bảng kết quả chính.

**S1–S3:** đưa lại bảng GeoI-Slack cap phiên/L10 từ `2026-09-26_brief/method_evidence.json`: Hit100 5,56/1,39/2,88%; MAE 1.119/874/904m; Recall 91,24/97,98/95,62%. Đây là 12 nhóm phát triển/macro theo các điều kiện; chưa phải confirmation L30. Nguồn được pin, không chạy lại hoặc gán số của pipeline khác cho Geo-I.

**Chứng minh:** nếu mọi cặp trace giữa hai support có D∞≤r và cùng ngữ cảnh công khai, Geo-I lý tưởng cho α=r∑C_s và cận Bayes cân bằng exp(α)/(1+exp(α)). Ví dụ chỉ khác một GPS 10m/100m dùng alpha≤2ur cho cận Bayes 54,98%/88,08%; không áp cho hai tuyến khác nhiều mẫu. Cận không tự hữu ích cho identity: C_s=0,23/m và r=100m đã cho α=23. Chưa chứng nhận sampler float; chưa bao phủ account/IP. Không thêm k-anonymity chỉ vì có nhiều hướng giải identity.

**Utility/đầu cuối:** L20 89,71→L30 92,69% trong study đã lưu dùng u=0,00125/m, cap mỗi chuyến 0,02875/m; không gán cho cấu hình làm việc. Gain 2,97 điểm %, CI [2,31;3,65], reply bytes +31,10%. Recall L10 95,44% lịch sử khác purpose/cohort. Nhánh Endpoint20 riêng tăng nhiễu, gửi ngay; bản cũ warmup/delay tùy chọn. Chưa có một cấu hình được xác nhận fully bảo vệ S1–S10; S8 còn mở.

**Sample:** cố định `freshqp-001`, TRAIN, draw1, u=0,00125/m, cap phiên 0,02875/m. Không nhân ngân sách rồi giữ Z/Q để gọi mẫu mới. 60s: 420 bản ghi→252 POI; trích k5 đã lưu. Tổng phí gồm các lần đọc không hiển thị. Nhánh giữ Z ở phiên6 tách timeline phiên1. Không đặt nhiễu Laplace chưa lưu. GPS SUMO; utility GPS oracle chưa là đo GNSS/năng lượng thật.

## Tái lập report

`build_report.py` kiểm tra nguồn pin, dựng sơ đồ TikZ/bảng, crop bản đồ vector và đồng bộ ghi chú. Nó đọc bảng lịch sử S1–S3 và bảng Endpoint20 đúng nguồn pin; tạo thêm bảng ví dụ Bayes từ công thức, không phải score thực nghiệm. Không chạy sampler, attacker hoặc evaluator.

```sh
python3 build_report.py
tectonic --keep-logs report_explained.tex
```

Chạy từ thư mục này, với Python/PyMuPDF và Tectonic hoặc XeLaTeX. Render/kiểm tra đủ sáu trang trước phát hành. `report_sources.json` pin nguồn nghiên cứu; `report_manifest.json` ghi hash và kiểm chứng.

## Các bản trước để tra cứu

- [Report trước](report_archive/epoch8/report_explained.pdf): snapshot nguồn/kịch bản/hình giữ nguyên trong thư mục lưu trữ.
- [slides.pdf](slides.pdf), [slides.html](slides.html): bản tám slide trước; [kịch bản](slides_presentation_script.md), [ghi chú](slides_speaker_notes.txt) riêng.
- `build_slides.py` không ghi đè ghi chú report; `manifest.json`, `source_pins.json` là bản phát hành slide độc lập.
- [detailed/](detailed/README.md): bản 18 trang lưu trữ.

Canonical thesis, mã core, các dataset/benchmark và các bản trước giữ nguyên. Replay bốn template L10 không được áp dụng: [logic/kết quả](../../research/2026-10-09_multi_purpose_retrieval.md).

© OpenStreetMap contributors — [nguồn và giấy phép](https://www.openstreetmap.org/copyright).
