# Verification — controlled SUMO report release v1

Ngày kiểm tra: **07/09/2026**. Đối tượng: dữ liệu và demo S1–S3, cơ chế sinh dummy, bản luận văn tích hợp. Kết luận: **Share with caveats — có thể demo và báo cáo thực nghiệm thăm dò; chưa đủ kết luận vượt SOTA hoặc hoàn thành luận văn thực nghiệm.**

## 1. Bằng chứng chính

Artifact chuẩn: `artifacts/benchmarks/report_demo/results.json`.

SHA-256: `be1ccb94ee439cb9c038e3065965b886ddce3122e8612b1e7301da8791cd611f`.

| Phép kiểm tra | Kết quả |
|---|---|
| Nguồn mã của pipeline khớp snapshot | 33 tệp |
| Kịch bản có mẫu thật | S1, S2, S3; 18 đoạn / 6 xe kiểm thử / 3 hạt giống |
| Tổ hợp đoạn–cấu hình | 324 = 18 đoạn × 9 cấu hình × 2 giá trị K |
| Trạng thái | 298 hợp lệ; 24 không áp dụng; 2 lỗi |
| Số học sai số, Hit, Recall và payload | Tính lại 1968 sự kiện, 54 hàng tổng hợp |
| Truy ngược đáp án về FCD thực | 117/117 điểm của 18 đoạn khớp đúng timestamp, lat/lon, tốc độ, cạnh/làn |
| Dịch vụ POI từ OSM và đường có hướng | Tính lại 1968 truy vấn; ID tham chiếu và ID sau hợp/lọc/xếp hạng khớp |
| Chia dữ liệu | Model-train / shadow / validation / test không giao xe trong từng seed |
| Kiểm tra tiền tố | 96/96 ở 8 cấu hình nhân quả; AnotherMe 0/6, đúng tính chất ngoại tuyến |
| Chạy lại độc lập pipeline | Trùng dữ liệu/đầu ra/kết quả không phải thời gian; chỉ loại timestamp sinh XML và đo wall-clock |
| Web browser | Không lỗi JavaScript; không gọi tài nguyên ngoài localhost; desktop và 390 px không tràn ngang |
| Pytest toàn repository | 141 passed (11.01 s ở lượt xác nhận cuối) |
| PDF cuối | 47 trang A4, build XeLaTeX thành công; không Overfull/undefined citation |

Scripts: `experiments/verify_report_demo.py`, `experiments/qa_report_demo_browser.py`. Kiểm thử hợp đồng: `tests/test_report_demo.py`, `tests/test_thesis_demo.py`, `tests/test_web_benchmark_app.py`. Các phép kiểm tra đơn vị khác của repository cũng đã được chạy cùng pytest.

PDF SHA-256: `aa60dc2248c957e7dc4e0305f4d84854160da935024553bfe5c7b27996decdda`.
Kiểm tra `git diff --check` không phát hiện lỗi khoảng trắng. Bản chuẩn đã được
chép từ đúng bản build được render/kiểm tra, không dùng PDF cũ trước khi bổ sung
Chương 5–7. Dùng pytest làm lệnh test chuẩn; runner lịch sử `tests.run_all` gọi
trực tiếp các hàm và không hỗ trợ fixture pytest như `tmp_path`.

## 2. Phạm vi và phương pháp kiểm tra

**Đơn vị phân tích:** đoạn quỹ đạo, không phải từng điểm độc lập. Mỗi hạt giống có 2 xe test, 4 xe shadow, 2 xe validation và 4 xe model-train. S1–S3 cùng xuất phát từ các xe có đủ điểm dừng và di chuyển; số hàng 324 không phải 324 người hay 324 hành trình độc lập.

**Sinh dữ liệu:** `randomTrips.py` + SUMO trên passenger-only OSM; lệnh đỗ 180 s được thêm trước mô phỏng. Nhãn S2 yêu cầu dừng ≥120 s, không trôi >5 m; S3 cần đoạn di chuyển liên tục ≥60 s. Parser kịch bản từ chối mất liên tục FCD 1 Hz và cửa sổ lấy mẫu quá ngắn. Các xe không đủ điều kiện bị ghi vào danh sách loại; không có fallback GeoLife.

**Mô hình:** tập đầu vào chung chiếu FCD tới nút đường. DLS theo Algorithm 1 của Niu et al. (2014), thay ô lưới bằng đỉnh đường và nêu rõ luật xử lý biên/tie. Thêm API streaming và ràng buộc đường cho cơ chế neo. Đối thủ được chọn trên validation cho từng phương pháp/case/K/seed từ sáu nhóm: centroid, running mean, continuity, road filter, shadow 3-NN và prior-only. Không lấy đáp án test để chọn đối thủ tốt nhất.

**Số học độc lập:** verifier không gọi lại hàm `attack_scores` hoặc `summary_rows` để xác nhận chính chúng. Nó tính khoảng cách từ lat/lon FCD sang hệ tọa độ phẳng, so với ước lượng đã lưu, tính Hit và giao tập POI, rồi kiểm tra mẫu số và tổng hợp. Với `--raw`, tra lại từng điểm FCD; xếp POI lại bằng `networkx` trên mạng SUMO, không dùng `PoiService.evaluate` làm oracle.

**Thời gian:** giây mô phỏng kể từ lúc bắt đầu, không phải ngày giờ di chuyển người thật. Thời gian chạy báo cáo là trung bình sinh đầu ra theo đoạn, tách khởi tạo; không coi là p95, SLA hay độ trễ end-to-end của điện thoại.

## 3. Vấn đề đã sửa trong vòng này

1. **Nhìn trước tương lai qua RNG:** nếu lấy hết neo trước rồi mới sinh dummy, độ dài tương lai có thể đổi kết quả tiền tố. Đã tách RNG neo/dummy và duyệt theo thời gian; batch gọi cùng `protect_step`. Có test streaming = batch và thay đổi phần đuôi.
2. **Ràng buộc đường bị thiếu:** khoảng cách thẳng không thể xác nhận đường một chiều/thời gian đi. Đã thêm biến thể giới hạn trạng thái tới được bằng Dijkstra có hướng và thêm đối thủ theo mạng đường. Không đánh đồng với đầy đủ luật giao thông.
3. **Mốc sai số mơ hồ:** kết quả chính giờ so với FCD thật. Sai số trên đầu vào chiếu nút lưu ở `represented_location_mae_m`; độ lượng tử hoá lưu riêng. Không gán “Không bảo vệ” sai số vật lý bằng 0 một cách giả tạo.
4. **Trộn không áp dụng với lỗi:** AnotherMe trên S1/S2 được ghi N/A, không nhận Recall=0. Hai lỗi phân loại xe đạp ở S3 giữ nguyên và giảm Recall giao được; không xoá khỏi mẫu số dịch vụ.
5. **Đối thủ dùng prior khác nhau:** prior chung lấy từ tập model-train, làm trơn công khai. Đưa prior-only vào lựa chọn validation để tránh quên baseline này. Bộ học shadow vẫn thích nghi với từng cơ chế như cần thiết.
6. **Chi phí payload tính cả thẻ phương pháp:** hiện chỉ đếm JSON sự kiện/loại truy vấn + ID phản hồi, loại metadata nghiên cứu. Vẫn là mô hình chi phí, không phải packet capture.
7. **UI biến null thành 0%:** sửa cách hiển thị giá trị không áp dụng, nút Phát ở lượt không có đầu ra, và trạng thái AnotherMe ngoại tuyến. Bản đồ đường nhúng sẵn, không cần tile.
8. **Test route nhận diện nhầm chữ `runs`:** test cũ cấm chuỗi `run` trong URL, khiến API đọc kết quả bị báo là chạy benchmark. Thay bằng kiểm tra mọi route chỉ có GET/HEAD/OPTIONS; POST vẫn bị từ chối.

## 4. Kết quả cần diễn giải thận trọng

- **S2, DLS, K=3:** MAE 447.96 m nhưng Hit≤100 m = 88.89%. Sai số trung bình cao không có nghĩa phần lớn lần gửi khó đoán. Điểm dừng thật lặp lại là một tín hiệu đối thủ có thể dùng. Đây là lý do không dùng entropy hay một MAE riêng để kết luận.
- **S3, K=3:** dummy hình học Recall@5 = 88.22%; bản theo đường = 70.64%. Bản sau có chuyển tiếp hợp lệ trên đồ thị 100%, nhưng mất chất lượng dịch vụ. Chưa so ngang tại cùng utility; không suy ra cải thiện riêng tư hữu ích chỉ từ MAE tăng.
- **Biểu diễn nút giao:** chuỗi không bảo vệ chỉ có 38.30% chuyển tiếp S3 vượt kiểm tra đường ở K=3. Đây là dấu hiệu mất tiến độ dọc cạnh khi chiếu nút, không phải SUMO sinh 61.70% bước vi phạm giao thông. Bản theo đường đạt 100% trong đúng không gian biểu diễn đã ép; không dùng tỷ lệ này như “độ thực tế” chung cho mọi giao diện.
- **SOTA:** TransProtect chưa có Transformer/GCN huấn luyện của bài gốc; Semantic chưa có LSTM/attention huấn luyện, dùng loại đường thay ngữ nghĩa đầy đủ. AnotherMe có phần VTGA nhưng không tương đương toàn bộ môi trường paper. Không xếp hạng paper từ các con số này.
- **Geo-I:** chứng minh chỉ cho kernel lý tưởng trên miền cố định và hậu xử lý đúng ranh giới. Còn giới hạn dấu phẩy động, lượng tử hoá, ngân sách cộng dồn; timestamp/loại truy vấn không được bảo vệ.

## 5. Những việc chưa làm — thứ tự ưu tiên cho vòng tiếp theo

### P1: Có thể làm thay đổi kết luận khoa học

1. **Dùng trạng thái cạnh + tiến độ dọc cạnh.** Cần tôn trọng một chiều/cấm rẽ và tránh artefact khi nhảy nút gần nhất. Test chuẩn: quỹ đạo SUMO không bảo vệ phải khớp một phép kiểm tra giao thông tương thích với biểu diễn, không chỉ geometry lookup.
2. **Đối thủ S3 toàn chuỗi và đối thủ S2 mạnh hơn.** Hiện S3 dùng bộ lọc nhân quả, chưa dùng hậu tố để suy luận quá khứ. Thêm smoothing/Viterbi có mô hình quan sát theo cơ chế, kiểm tra rò rỉ liên kết ứng viên và kiểm định hiệu chuẩn trước gọi là posterior.
3. **Ngân sách toàn hành trình.** Cần tổng epsilon cố định, chính sách tái sử dụng có phân tích riêng tư và lịch gửi có/không bảo vệ. Chưa thể coi dummy-only hậu xử lý là tự giải quyết S2.
4. **Cùng utility/chi phí.** Quét tham số trên validation, chọn cấu hình đạt Recall mục tiêu rồi giữ test cố định. Dùng nhiều seed/xe hơn, khoảng tin cậy theo chuyến/người. Nếu phân tích thành phần, ghép nguồn ngẫu nhiên phù hợp giữa các biến thể để giảm nhiễu.
5. **Tái lập đúng comparator.** Chốt paper-to-code map, dữ liệu huấn luyện/split, checkpoint/siêu tham số, ablation; chỉ nâng nhãn khi các thành phần đã có và được kiểm chứng. Không đổi tên `adaptation` thành `faithful` để đáp ứng hình thức.

### P2: Mở rộng độ bao phủ và tính ứng dụng

6. S4–S10 cần generator và kiểm tra đủ điều kiện riêng: nhiều phiên, persona/device mapping, lựa chọn nhánh, đích tương lai, truy vấn thật/giả, đồng hành, khoảng che đầu/cuối. Mỗi mục tiêu cần một đáp án/đối thủ/metric riêng trước khi ghi “covered”.
7. Dịch vụ POI cần đa dạng query hơn (hiện một loại theo seed), POI way/relation và chi phí truy cập từ đường tới POI; ràng buộc local ranking cần mô tả thông tin dịch vụ thiết bị có sẵn.
8. Đo p50/p95 theo sự kiện trên thiết bị phù hợp; tách client generation, LSP, mạng và local filter. Hiện chỉ có trung bình batch và byte JSON.
9. Bổ sung môi trường khác/thành phố khác khi khung nhỏ đã ổn. SUMO random demand không chứng minh đại diện cho hành vi dân cư.

## 6. QA tài liệu và demo

- Xác nhận semantic correlation là **bài tạp chí** (Springer, 04/06/2026). Chưa kiểm chứng Q1 theo hệ thống/năm; không đưa vào tuyên bố của luận văn.
- Giữ LaTeX làm mặt trình bày chuẩn; bảng số xuất từ JSON đã qua verifier, không điền số tay.
- Chọn bảng thay đồ thị xếp hạng vì có nhiều đơn vị/mẫu số/giao diện và chỉ 6 đoạn/case. Cột N/A không màu hoá thành “thất bại”. SD được ghi là độ lệch chuẩn, không phải CI.
- Kiểm tra PDF qua bản render: sơ đồ có tầng thiết bị/LSP, ngữ cảnh công khai, lõi bảo vệ; bảng kịch bản tiếp trang có tiêu đề; các bảng kết quả ở cạnh phần diễn giải. Không có lỗi Overfull hay trích dẫn chưa giải quyết ở lần build cuối.
- Chromium headless kiểm tra desktop 1440×1050 và mobile 390×844: đổi S2/S3, DLS/bản theo đường/AnotherMe, Phát/Dừng, bật tắt evaluator, 54 hàng tổng hợp, N/A không hiện 0%, không có request ngoài localhost.
- Flask mặc định `create_app` tắt evaluator; CLI dùng cho demo localhost bật evaluator trừ khi chỉ định `--public-only`. Đây không phải hệ xác thực/triển khai production.

## 7. Lệnh lặp lại

```bash
venv/bin/python -m pytest -q
venv/bin/python -m experiments.verify_report_demo --raw --compare tmp/report_demo_replay/results.json
venv/bin/python -m experiments.export_report_demo
venv/bin/python -m web.benchmark_app --port 5050
venv/bin/python -m experiments.qa_report_demo_browser
```

Muốn tái chạy so sánh độc lập, trước đó chạy `experiments.run_report_demo --output tmp/report_demo_replay`. Hướng dẫn PDF/demo đầy đủ ở `thesis/notes/report_demo_release_2026-09-07.md`. Các artifact cũ và review cũ giữ nguyên làm lịch sử, không dùng số của chúng lẫn với bản này.
