# Bản trình bày và demo ngày 07/09/2026

## Đọc/mở ở đâu

- Luận văn chuẩn: `thesis/main.tex`, phần mới ở `thesis/report_demo_chapters.tex`.
- PDF chuẩn: `artifacts/reports/graduation_thesis.pdf`.
- Kết quả của bản này: `artifacts/benchmarks/report_demo/results.json` và `results.sha256`.
- Bảng số trong luận văn: `artifacts/benchmarks/report_demo/results_tables.tex`, sinh trực tiếp từ JSON đã kiểm chứng.
- Kiểm chứng và việc cần cải thiện: `docs/reviews/verification_report_demo_release_v1.md`.

Đây là bản có thể trình bày **khung nghiên cứu, hiện thực nhân quả và thực nghiệm thăm dò**, không phải bản chứng minh thắng SOTA hoặc hoàn thành cả mười tình huống. Bốn chương xác định bài toán được giữ; bổ sung ba chương về cơ chế, thực nghiệm và kết luận. Tài liệu lịch sử và snapshot không thay thế nguồn luận văn chuẩn.

## Cách tham khảo cấu trúc paper semantic correlation

[Liu, Peng và Zhou (04/06/2026)](https://link.springer.com/article/10.1007/s44443-026-00899-w) là bài **tạp chí** Journal of King Saud University Computer and Information Sciences, không phải hội nghị. Chưa xác nhận xếp hạng Q1 theo một năm/hệ thống cụ thể, nên không dùng nhãn đó làm căn cứ chất lượng.

Luận văn học cách tổ chức lập luận của bài này, không sao chép nội dung hay mặc nhiên nhận các bảo đảm của họ:

| Vai trò trong lập luận | Vị trí của luận văn |
|---|---|
| Động cơ: quan sát nhiều lần làm lộ tương quan | Tóm tắt; Chương 1, 3 và 5 |
| Kiến trúc, ký hiệu, giả thiết | Chương 1–2, phần đầu Chương 3 |
| Công trình liên quan và khoảng trống | Bảng kịch bản/công trình ở Chương 3, đối chứng ở Chương 4 |
| Cơ chế có thể thực thi | Chương 5: neo REM, hậu xử lý dummy, biến thể ràng buộc đường |
| Phân tích riêng tư và độ phức tạp | Chương 5: chứng minh cho phân phối lý tưởng, giới hạn hợp thành, lượng tử hoá, số thực máy |
| Thiết lập, kết quả, phân tích thành phần | Chương 6: SUMO, chia tập, đối thủ, POI, bảng thực đo |
| Kết luận và giới hạn | Cuối Chương 6 và Chương 7 |

Các phần metrics của paper vẫn được giữ tách khỏi metrics chung. Đặc biệt, không tự chuyển entropy/DER thành xác suất hậu nghiệm, và không lấy số ở dữ liệu khác để xếp hạng với SUMO của ta. DLS bổ sung một đối chứng không học sâu theo [Algorithm 1, INFOCOM 2014](https://doi.org/10.1109/INFOCOM.2014.6848002). Nền tảng Geo-I dẫn về [Andrés và cộng sự, CCS 2013](https://arxiv.org/abs/1212.1984).

## Nội dung có thể trình bày trong khoảng 10 phút

1. **Kiến trúc và phạm vi (1–2 phút).** Thiết bị có vị trí, lịch sử và truy vấn thật; lấy ngữ cảnh công khai từ bản đồ; cơ chế bảo vệ xử lý trước khi gửi LSP. LSP trả kết quả dịch vụ nhưng đồng thời có thể suy luận thông tin. Scope chuyển động hiện là xe chở khách đô thị, không phải mọi phương tiện.
2. **Mục tiêu → kịch bản (2 phút).** Bốn nhóm mục tiêu được tách thành mười tình huống có điều kiện quan sát/đáp án khác nhau. Core là S2 điểm dừng và S3 chuỗi đã đi; S1 là đối chứng đơn thời điểm. S4–S10 vẫn là đặc tả, không nói đã chống được chúng.
3. **Dữ liệu và đánh giá (2 phút).** SUMO thật sinh chuyển động và điểm dừng từ tuyến OSM. Dữ liệu tách xe huấn luyện, xe phụ trợ cho đối thủ, xe chọn đối thủ và xe kiểm thử. Đối thủ chỉ nhận dữ liệu công bố. Chấm hai câu hỏi: “đối thủ đoán gần sự thật đến đâu?” và “ứng dụng còn trả đúng POI không?”. Chi phí đo riêng.
4. **Ý tưởng cơ chế (1 phút).** Che vị trí bằng một neo ngẫu nhiên, rồi sinh dummy từ neo và ngữ cảnh đường. Bước thứ hai không đọc lại vị trí thật. Thêm ràng buộc đường có hướng để thử xem chuyển động hợp lý hơn có đổi chất lượng và mức bị suy luận thế nào.
5. **Demo và kết quả (2–3 phút).** Dùng các bước dưới. Nêu cả đánh đổi bất lợi của biến thể theo đường. Ba phương pháp từ paper hiện là bản thích nghi, không phải huấn luyện/tái lập đầy đủ mô hình gốc.
6. **Kết luận cần thầy phản hồi (1 phút).** Khung thực nghiệm đã chạy thông suốt. Ưu tiên tiếp theo: biểu diễn trạng thái cạnh tốt hơn, đối thủ toàn chuỗi mạnh hơn, ngân sách theo hành trình, so tại cùng chất lượng; sau đó mở rộng kịch bản. Chưa cần tuyên bố giải quyết mọi target.

## Demo không cần chạy lại SUMO

Từ thư mục gốc:

```bash
venv/bin/python -m web.benchmark_app --port 5050
```

Mở **http://127.0.0.1:5050/report-demo**. Nếu cổng đã có tiến trình demo thì dùng trang đang mở, hoặc chọn cổng khác. Server chỉ lắng nghe localhost. Đây là phát lại kết quả, không phải điều khiển SUMO trực tiếp hoặc benchmark chạy trong request HTTP.

Trình tự gợi ý:

1. Để S3, chọn “Neo Geo-I + dummy · thử nghiệm”, K=3. Ban đầu chỉ thấy điểm công bố màu xanh và đường OSM.
2. Bật “Hiện dữ liệu của người đánh giá”: đường thật màu đen và ước lượng đối thủ màu đỏ xuất hiện. Bấm Phát hoặc kéo thanh thời gian. Metrics bên trái là của **cả đoạn**, không phải riêng bước đang chọn.
3. Chuyển sang biến thể “ràng buộc đường”. Nêu rõ điều được đảm bảo là chuyển tiếp trên đồ thị; đánh đổi POI được xem ở bảng tổng hợp, không chọn một hình đẹp để kết luận.
4. Chọn S2 và DLS. Tập điểm tại mỗi lần gửi có entropy cao chưa ngăn được suy luận qua việc một vị trí đứng yên lặp lại.
5. Chọn các đối chứng TransProtect/Semantic để xem khác biệt giao diện. AnotherMe xem ở S3 như tham chiếu ngoại tuyến; S1/S2 sẽ hiện “Không áp dụng”.
6. Tắt chế độ người đánh giá để nhắc lại ranh giới dữ liệu. Nền đường được nhúng trong JSON, không cần tải tile OSM.

Các đoạn nối điểm chỉ minh hoạ thứ tự công bố, không khẳng định đó là tuyến đã định tuyến giữa hai điểm. Không dùng ứng dụng localhost này như server Internet hoặc cơ chế phân quyền sản phẩm. `--public-only` vô hiệu hoá hoàn toàn các endpoint trả đáp án thật; checkbox chỉ là chế độ trình bày khi endpoint đánh giá được cho phép.

## Tái chạy và kiểm chứng

Đã có Python 3.11, dependencies trong `requirements-dev.txt`, SUMO 1.27.1 theo `requirements-sumo.txt` và OSM ở `data/raw/Beijing.osm.gz`.

```bash
venv/bin/python -m experiments.run_report_demo
venv/bin/python -m experiments.verify_report_demo --raw
venv/bin/python -m experiments.export_report_demo
venv/bin/python -m pytest -q
```

`--raw` cần cache FCD/mạng và OSM cục bộ. Nếu chỉ nhận repository chưa sinh cache, vẫn chạy được `verify_report_demo` để kiểm tra hash, nguồn mã, số học và bản ghi đã commit; không gọi đó là kiểm chứng nguồn FCD trực tiếp.

Kiểm tra tính lặp lại (không ghi đè kết quả chuẩn):

```bash
venv/bin/python -m experiments.run_report_demo --output tmp/report_demo_replay
venv/bin/python -m experiments.verify_report_demo --raw --compare tmp/report_demo_replay/results.json
```

So sánh loại trừ số đo thời gian và timestamp trong phần đầu XML. Vẫn yêu cầu giống hệt dữ liệu được chọn, điểm công bố, kết quả đối thủ, POI, các metrics không phải thời gian và lựa chọn đối thủ trên validation. Cache mô phỏng được tạo lại khi chạy; không coi hash byte XML cũ là lỗi dữ liệu nếu chỉ phần thời điểm sinh đổi.

Build PDF:

```bash
cd thesis
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=../build/thesis main.tex
cd ..
cp build/thesis/main.pdf artifacts/reports/graduation_thesis.pdf
```

Kiểm tra browser là tùy chọn phát triển, không phải dependency chạy demo:

```bash
venv/bin/python -m pip install playwright
venv/bin/python -m playwright install chromium
venv/bin/python -m experiments.qa_report_demo_browser
```

Script mặc định kiểm tra localhost:5050. Không tải tile hoặc script bên ngoài trong phiên demo; ảnh QA nằm trong `tmp/`, không phải artifact thay thế web app.

## Quyết định trình bày và nguồn số

- Mặt trình bày chính giữ LaTeX/PDF theo định dạng luận văn, không đổi sang HTML report hoặc dịch vụ hosting.
- Khung báo cáo kỹ thuật: tóm tắt → phạm vi/định nghĩa → phương pháp → bằng chứng → giới hạn → bước tiếp theo. Dời kết quả sau định nghĩa để giữ kiến trúc và scope ở đầu theo yêu cầu của người viết.
- Dùng bảng tra cứu chính xác cho chín cấu hình, nhiều metrics và trường hợp không áp dụng. Không vẽ leaderboard vì các bản thích nghi/giao diện/độ trễ không có cơ sở xếp hạng chung. Không vẽ đường xu hướng từ chỉ hai giá trị K.
- Ba bảng S1–S3 là phép đo theo đoạn ở K=3; bảng thành phần S3 bổ sung K=5. Nguồn đầy đủ cả 54 hàng nằm trong JSON và web app. Ngữ cảnh, đơn vị, số đoạn và giới hạn nằm sát bảng.
- Phần sơ đồ kiến trúc dùng vector TikZ để giữ nét và có thể sửa trong LaTeX. Kết quả số chỉ được xuất sau khi script kiểm chứng vượt qua.
