# Báo cáo chuẩn bị cho 26/09/2026

Cập nhật theo ghi chú 19/09 và yêu cầu bổ sung phương pháp/kết quả. Nguồn rà soát
đến 21/09/2026. Bản PDF hiện có **35 trang**.

- [PDF](report_explained.pdf), [LaTeX](report_explained.tex), [HTML](report_explained.html)
- Mục 8, trang 13–16: kiến trúc, neo/ngân sách, bộ lọc, chọn truy vấn và mẫu công bố.
- Mục 9, trang 17–20: kết quả lịch sử; tách paper-v2 khỏi fresh-switching.
- Mục 10, trang 21–23: đề xuất bảo vệ biên và phép thử cắt thêm cửa sổ.
- Mục 11, trang 24–33: in trực tiếp 30 mẫu A/B/C và hình cặp tuyến.

## Tệp bằng chứng

- `data_samples.json`: 30 bản ghi nguồn cùng các điểm FCD thật.
- `printed_samples.json`: điểm và nhãn rút gọn được in trong phụ lục.
- `sample_maps.html`: bản đồ tương tác offline cho 30 ca, zoom/pan và bật/tắt nhãn.
- `sample_maps.pdf`: tập 10 hình A/B/C trên mạng đường.
- `map_provenance.json`: nguồn hình học, checksum và giới hạn so khớp mạng.
- `scenario_inventory.csv`: số lượng và ID từng ca.
- `sources.json`: 13 nguồn khảo sát, URL và mức xác minh.
- `method_evidence.json`: SHA-256 dữ liệu/mã nguồn dùng cho phần phương pháp và kết quả.
- `score_example.json`: ví dụ tính điểm tổng, **không phải kết quả model**.
- `boundary_audit_protocol.md`: protocol thăm dò ghi trước khi tính thêm kết quả.
- `figures/`: 5 hình phương pháp và 10 hình bản đồ ở dạng vector PDF/SVG và PNG; hình cắt biên gốc vẫn có trong report.

## Sinh lại

Nội dung chung nằm trong `build_report.py` và `method_content.py`. Generator ghi
đè `.tex`/`.html`, vì vậy các sửa đổi trực tiếp vào tệp sinh ra cần được chuyển lại
vào nguồn nếu muốn tái sinh. Các hình dùng matplotlib; bước audit/build chỉ dùng
thư viện chuẩn Python. Không sửa dataset hoặc kết quả lịch sử.

Từ thư mục gốc repository:

```sh
python3 -m experiments.report_boundary_audit
python3 docs/supervisor_meeting/2026-09-26_brief/plot_method_figures.py
python3 docs/supervisor_meeting/2026-09-26_brief/build_report.py
python3 docs/supervisor_meeting/2026-09-26_brief/plot_sample_maps.py
tectonic --untrusted --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/report_explained.tex
python3 -m unittest discover -s tests -p test_report_boundary_audit.py -v
```

Có thể dùng XeLaTeX với font TeX Gyre Termes thay Tectonic. Hình được lưu kèm
để build báo cáo không cần chạy lại matplotlib nếu nội dung hình không đổi.

## Kiểm tra đã thực hiện

- Đối chiếu 30 mẫu và từng điểm FCD trích với dataset; 12×22 phiên, 393 bản ghi.
- Tính lại 36 dòng tóm tắt fresh-switching từ 636 hàng kết quả đã lưu, theo đúng
  lựa chọn đối thủ và mẫu số từng ca; đây không phải chạy lại defender.
- Kiểm tra lại 48 hàng S9/S10 nguồn bằng tọa độ dự đoán/phép chiếu gốc và ID POI.
- Chạy 144 hàng thăm dò cắt thêm cửa sổ; lưu dự đoán, sai số, mẫu số và hash nguồn.
- 5 unit tests: cắt đúng phía/không đổi nguồn; ngoại suy tuyến tính; mẫu số giữ
  truy vấn bị mất; dự đoán không dùng nhãn; từ chối grid chưa đăng ký.
- Kiểm tra liên kết HTML/hình, hash bằng chứng và ví dụ Q.
- PDF 35 trang: không tràn dòng hoặc thiếu glyph; render và kiểm tra trang/hình.

## Cách đọc kết quả

Lõi mới có số đo S1–S3; số đo S9/S10 trong paper-v2 thuộc cơ chế cũ và cửa sổ cũ.
Không ghép chúng thành một model mới đã xác nhận đủ năm scenario. Audit bổ sung
chỉ là hậu xử lý transcript đã lưu, dùng ba decoder cố định, không huấn luyện,
không chạy lại SUMO, và không tạo test độc lập. Cắt đuôi ngoại tuyến không chứng
minh khả năng biết trước điểm kết thúc khi chạy trực tuyến.

Máy hiện thiếu cache mạng đường/môi trường gốc để chạy lại đầy đủ pipeline.
Buffer, cache dịch vụ, padding và bảo vệ nhiều phiên vẫn là extension đề xuất.
Không lấy byte mã POI của vòng cũ để tính performance/điểm Q mới như thể đã có
lưu lượng và latency đầu-cuối.

Survey giữ 10 nguồn gần đây và 3 ngoại lệ đối chứng có giải thích; nguồn chưa đủ
định nghĩa được đánh dấu CX. AnotherMe vẫn là đối chứng thứ sáu trong kế hoạch,
chưa là một tái lập online hoàn chỉnh chỉ vì đã có tên trong bảng.

## Bản đồ sample

Nền có 9.138 polyline xuất từ mạng SUMO paper_benchmark seed 81. Checksum OSM
trùng với urban_fresh_v2; checksum file mạng khác nhau. Do thiếu file .net.xml
gốc của dataset, chưa thể xác minh trùng hình học hoặc kết nối làn. Không dùng
bản đồ nền này để tính lại metric. Các tọa độ, cửa sổ, nhãn đều trích đúng từ
30 bản ghi đã liệt kê. © OpenStreetMap contributors.
