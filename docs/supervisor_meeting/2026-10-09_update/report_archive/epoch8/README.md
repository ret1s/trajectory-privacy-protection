# Bảo vệ riêng tư truy vấn và quỹ đạo bằng Geo-I/REM

Tài liệu chính là **report LaTeX 6 trang A4**, dùng để trình bày khoảng **8–10 phút**. Chữ serif TeX Gyre Termes, cỡ nội dung 10 pt, bảng khoa học và sơ đồ vector. Không dùng bố cục slide hoặc metadata buổi gặp trong nội dung report.

- [report_explained.pdf](report_explained.pdf): bản đọc và trình bày.
- [report_explained.tex](report_explained.tex): nguồn LaTeX chỉnh sửa được.
- [presentation_script.md](presentation_script.md): lời trình bày theo sáu trang, cách chỉ hình và điểm cần nhớ khi trao đổi.
- [speaker_notes.txt](speaker_notes.txt): ghi chú đồng bộ theo sáu trang.

Thứ tự nội dung:

1. S7: nhu cầu giữ local, request chung L30; bốn tiêu chí lọc/sắp xếp; cơ chế hỗ trợ và khó khăn S4–S6, giới hạn S8.
2. Cap tám phiên: cap từng phiên so với cap epoch, đơn vị 0/1/2, kiểm tra trước GPS và chặn hợp thành toán học.
3. Kiến trúc trước: một trang riêng, bốn tầng, khung mô hình tại thiết bị, máy chủ bên ngoài, đầu vào/đầu ra và delay tùy chọn.
4. Kiến trúc hiện tại: cùng bố cục trên một trang riêng; đánh dấu cập nhật cap, L30 và xử lý nhu cầu local.
5. Ví dụ nhiều bước: timeline 0/20/60/120/600 s, bản đồ 60 s, sáu bước kích hoạt và đáp án khác nhau trên cùng phản hồi.
6. Benchmark: bảng L20–L30 theo bốn mục đích, CI và byte; bảng privacy bổ sung; phân biệt cấu hình/cohort và so bối cảnh báo cáo trước.

## Phạm vi kết luận

**S7:** nhu cầu thật ψ không gửi lên mạng. Năm Q × L30 lấy tối đa 150 bản ghi mỗi loại trước bỏ trùng; máy chủ trả về, thiết bị gộp, bỏ trùng và lọc/sắp xếp theo GPS cùng nhu cầu. Không mặc định 150 POI duy nhất toàn bộ phản hồi. Tính bất biến request xét cùng trạng thái bảo vệ, lịch và ngữ cảnh công khai. Triển khai/benchmark vẫn dùng k = 5 và Recall@5; không bỏ giới hạn trong mã hoặc dựng một danh sách mới cho mẫu.

**Ngân sách:** cap chung 0,23/m cho tối đa tám phiên, mỗi phiên 0,02875/m, đơn vị 0,00125/m. Cap bổ sung chặn thông tin tọa độ tích lũy, không tự che account/IP hoặc bảo đảm không liên kết danh tính. H = 12 không đồng nghĩa chỉ được 12 lần đọc GPS. Theorem dùng kernel/ngữ cảnh lý tưởng và giả định nêu trong report; không chứng nhận sampler thực hoặc ngân sách cả đời.

**Phiên bản đầu/cuối:** bản trước có warmup/delay 60 s tùy chọn, hủy Q đang chờ khi đóng phiên. Epoch8/L30 hiện tại gửi ngay. Nhánh Endpoint20 riêng đã đo tăng nhiễu trên mọi lần đọc được phép, không dùng delay; không nhập kết quả của nhánh L20 này thành xác nhận privacy của L30.

**Benchmark:** phép so L20 89,71% → L30 92,69% giữ nguyên Q/Z/cap/lịch trên 24 nhóm mới, tăng 2,97 điểm phần trăm, CI [2,31; 3,65], reply JSON tăng 31,10%. Recall L10 95,44% lịch sử thuộc mục đích/cohort/cách gộp khác, chỉ dùng làm bối cảnh. S4 diagnostic, S5/S6 pilot và Endpoint20 là các thử nghiệm riêng. Chưa có một cấu hình được xác nhận bảo vệ đầy đủ S1–S10; S8 vẫn giữ trong định nghĩa và giới hạn nghiên cứu.

**Ví dụ:** dùng đầu ra cố định `freshqp-001`, TRAIN, draw 1, phiên đầu. Tại 60 s có 420 bản ghi → 252 POI duy nhất; trích phần đầu đáp án k = 5 đã lưu. Tổng ngân sách tính cả các lần đọc không hiển thị. Nhánh đọc rồi giữ Z thuộc phiên 6, không nối vào hành trình đầu. Không đặt giá trị nhiễu Laplace chưa lưu. GPS tổng hợp SUMO; utility dùng GPS local tại mỗi mốc như oracle, chưa đo chi phí GNSS/năng lượng thực.

## Tái lập report

`build_report.py` kiểm tra nguồn được pin, dựng hai sơ đồ TikZ, ba bảng từ số đo đã lưu, crop bản đồ vector của ví dụ và tạo ghi chú theo kịch bản. Không chạy sampler, attacker hoặc evaluator. Nguồn LaTeX phần diễn giải chỉnh trực tiếp trong `report_explained.tex`.

Cần Python với `PyMuPDF`, và Tectonic hoặc XeLaTeX với các gói LaTeX được khai báo trong nguồn:

```sh
python3 build_report.py
tectonic --keep-logs report_explained.tex
```

Chạy từ thư mục này. Khi sửa, render PDF và kiểm tra đủ sáu trang trước phát hành. `report_sources.json` pin nguồn nghiên cứu; `report_manifest.json` ghi hash tài liệu phát hành và kiểm chứng. Các benchmark, đầu ra mô hình và luận văn chính giữ nguyên.

## Bộ slide trước để tra cứu

- [slides.pdf](slides.pdf), [slides.html](slides.html): bản tám slide trước, giữ nguyên nội dung.
- [slides_presentation_script.md](slides_presentation_script.md), [slides_speaker_notes.txt](slides_speaker_notes.txt): kịch bản và ghi chú gốc của tám slide.
- `build_slides.py` dựng lại HTML và ghi chú slide riêng, không ghi đè ghi chú report.
- `manifest.json`, `source_pins.json`: bản phát hành tám slide, độc lập với manifest report.
- [detailed/](detailed/README.md): bản 18 trang lưu trữ trước đó.

Dữ liệu đối chiếu: `benchmark_tables.json`, `benchmark_evidence.json`, `multistep_sample.json`, `utility_sample.json`, `endpoint_focus.json`; model và chứng minh tại `thesis/`; bối cảnh L10 tại `2026-09-26_brief/`. S1–S3 và đối chứng paper đầy đủ nằm trong tài liệu chi tiết. Replay bốn template L10 chưa qua gate và không được áp dụng: [logic/kết quả](../../research/2026-10-09_multi_purpose_retrieval.md).

© OpenStreetMap contributors — [nguồn và giấy phép](https://www.openstreetmap.org/copyright).
