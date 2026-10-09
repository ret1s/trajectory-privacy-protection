# Bảo vệ riêng tư truy vấn và quỹ đạo bằng Geo-I/REM

Tài liệu chính là **report LaTeX 6 trang A4**, dùng để trình bày khoảng **8–10 phút**. Chữ serif TeX Gyre Termes, cỡ nội dung 10 pt, bảng khoa học và sơ đồ vector. Không dùng bố cục slide hoặc metadata buổi gặp trong nội dung report.

- [report_explained.pdf](report_explained.pdf): bản đọc và trình bày.
- [report_explained.tex](report_explained.tex): nguồn LaTeX chỉnh sửa được.
- [presentation_script.md](presentation_script.md): lời trình bày theo sáu trang, cách chỉ hình và điểm cần nhớ khi trao đổi.
- [speaker_notes.txt](speaker_notes.txt): ghi chú đồng bộ theo sáu trang.

Thứ tự nội dung:

1. S7: nhu cầu giữ local, request chung L30; bốn tiêu chí lọc/sắp xếp; cơ chế hỗ trợ và khó khăn S4–S6, giới hạn S8.
2. Cap từng phiên: công thức tổng quát, đơn vị 0/1/2 và chặn một phiên; lớp S4 đề xuất cho người và xe vật lý.
3. Kiến trúc trước: một trang riêng, bốn tầng, khung mô hình tại thiết bị, máy chủ bên ngoài, đầu vào/đầu ra và delay tùy chọn.
4. Kiến trúc làm việc: cap phiên/L30, nhu cầu local và khối S4 nét đứt; phân biệt thiết kế được chọn với lớp identity chưa triển khai.
5. Ví dụ Epoch8 trước đã lưu: timeline 0/20/60/120/600 s, bản đồ 60 s, sáu bước kích hoạt và đáp án khác nhau trên cùng phản hồi.
6. Benchmark đã lưu: bảng Epoch8 L20–L30 theo bốn mục đích, CI và byte; bảng privacy bổ sung; phân biệt cấu hình/cohort và so bối cảnh báo cáo trước.

## Phạm vi kết luận

**S7:** nhu cầu thật ψ không gửi lên mạng. Năm Q × L30 lấy tối đa 150 bản ghi mỗi loại trước bỏ trùng; máy chủ trả về, thiết bị gộp, bỏ trùng và lọc/sắp xếp theo GPS cùng nhu cầu. Không mặc định 150 POI duy nhất toàn bộ phản hồi. Tính bất biến request xét cùng trạng thái bảo vệ, lịch và ngữ cảnh công khai. Triển khai/benchmark vẫn dùng k = 5 và Recall@5; không bỏ giới hạn trong mã hoặc dựng một danh sách mới cho mẫu.

**Chính sách được chọn 10/10:** giữ cap hiệu lực 0,23 m⁻¹ cho mỗi phiên, H12/U23, u = 0,01 m⁻¹ và B danh nghĩa 0,24 m⁻¹. Không chia cho tám phiên, không giới hạn phục vụ ở chuyến thứ tám. Nhiều chuyến liên kết vẫn cộng cap; cap từng phiên không được gọi là bảo đảm 0,23 cho cả lịch sử. H = 12 không là hard maximum số GPS reads. Khởi động lại cùng chuyến không được nạp cap.

**Danh tính S4:** hai target độc lập: cùng người và cùng phương tiện vật lý; thiết bị không được đồng nhất với cả hai. Lớp đề xuất loại ID bền và tách IP/request qua relay/gateway không thông đồng. Còn phải đánh giá linkage từ hình dạng, routine và timing. Lớp transport/identity này **chưa triển khai/chưa có benchmark mới**. Cặp một người đổi xe và hai người dùng chung xe là đối chứng bắt buộc. [Quyết định thiết kế và giao thức kiểm chứng](../../research/2026-10-10_session_cap_identity.md); [cấu hình máy đọc được](../../research/2026-10-10_session_cap_identity.json).

**Phiên bản đầu/cuối:** bản trước có warmup/delay 60 s tùy chọn, hủy Q đang chờ khi đóng phiên. Epoch8/L30 trong study trước gửi ngay. Nhánh Endpoint20 riêng đã đo tăng nhiễu trên mọi lần đọc được phép, không dùng delay; không nhập kết quả của nhánh L20 này thành xác nhận privacy của L30.

**Benchmark cũ:** phép so L20 89,71% → L30 92,69% giữ nguyên Q/Z/cap/lịch trên 24 nhóm mới của study Epoch8 trước, tăng 2,97 điểm phần trăm, CI [2,31; 3,65], reply JSON tăng 31,10%. Recall L10 95,44% lịch sử thuộc mục đích/cohort/cách gộp khác, chỉ dùng làm bối cảnh. S4 diagnostic, S5/S6 pilot và Endpoint20 là các thử nghiệm riêng. Chưa có readout mới cho cap phiên/L30/lớp S4; các số Epoch8 không được nhận là kết quả của thiết kế mới. Chưa có một cấu hình được xác nhận bảo vệ đầy đủ S1–S10; S8 vẫn giữ trong định nghĩa và giới hạn nghiên cứu.

**Ví dụ cũ:** dùng đầu ra cố định `freshqp-001`, TRAIN, draw 1, phiên đầu, u = 0,00125 thuộc Epoch8 trước. Không nhân ngân sách rồi giữ nguyên Z/Q để giả làm sample mới. Tại 60 s có 420 bản ghi → 252 POI duy nhất; trích phần đầu đáp án k = 5 đã lưu. Tổng ngân sách tính cả các lần đọc không hiển thị. Nhánh đọc rồi giữ Z thuộc phiên 6, không nối vào hành trình đầu. Không đặt giá trị nhiễu Laplace chưa lưu. GPS tổng hợp SUMO; utility dùng GPS local tại mỗi mốc như oracle, chưa đo chi phí GNSS/năng lượng thực.

## Tái lập report

`build_report.py` kiểm tra nguồn được pin, dựng hai sơ đồ TikZ, ba bảng từ số đo đã lưu, crop bản đồ vector của ví dụ và tạo ghi chú theo kịch bản. Không chạy sampler, attacker hoặc evaluator. Nguồn LaTeX phần diễn giải chỉnh trực tiếp trong `report_explained.tex`.

Cần Python với `PyMuPDF`, và Tectonic hoặc XeLaTeX với các gói LaTeX được khai báo trong nguồn:

```sh
python3 build_report.py
tectonic --keep-logs report_explained.tex
```

Chạy từ thư mục này. Khi sửa, render PDF và kiểm tra đủ sáu trang trước phát hành. `report_sources.json` pin nguồn nghiên cứu; `report_manifest.json` ghi hash tài liệu phát hành và kiểm chứng. Các benchmark, đầu ra mô hình và luận văn chính giữ nguyên.

## Các bản trước để tra cứu

- [Report Epoch8 trước](report_archive/epoch8/report_explained.pdf): bản sáu trang trước quyết định cap phiên; nguồn/kịch bản/hình giữ byte-identical trong `report_archive/epoch8/`. Canonical thesis vẫn là bản đã xác nhận trước quyết định này, không phải kết quả triển khai S4 mới.

- [slides.pdf](slides.pdf), [slides.html](slides.html): bản tám slide trước, giữ nguyên nội dung.
- [slides_presentation_script.md](slides_presentation_script.md), [slides_speaker_notes.txt](slides_speaker_notes.txt): kịch bản và ghi chú gốc của tám slide.
- `build_slides.py` dựng lại HTML và ghi chú slide riêng, không ghi đè ghi chú report.
- `manifest.json`, `source_pins.json`: bản phát hành tám slide, độc lập với manifest report.
- [detailed/](detailed/README.md): bản 18 trang lưu trữ trước đó.

Dữ liệu đối chiếu: `benchmark_tables.json`, `benchmark_evidence.json`, `multistep_sample.json`, `utility_sample.json`, `endpoint_focus.json`; model và chứng minh tại `thesis/`; bối cảnh L10 tại `2026-09-26_brief/`. S1–S3 và đối chứng paper đầy đủ nằm trong tài liệu chi tiết. Replay bốn template L10 chưa qua gate và không được áp dụng: [logic/kết quả](../../research/2026-10-09_multi_purpose_retrieval.md).

© OpenStreetMap contributors — [nguồn và giấy phép](https://www.openstreetmap.org/copyright).
