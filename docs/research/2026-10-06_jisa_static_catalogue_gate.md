# Static catalogue application gate — 06/10/2026

**Trong contract static hiện tại, tải toàn bộ 418 POI một lần rồi lọc local
giải quyết mọi reference không rỗng với chi phí payload thấp hơn rất nhiều.**
Đây là đối chứng cần giữ trước khi lập luận rằng ứng dụng cần gửi Q định kỳ.
Kết quả không bác bỏ Geo-I; nó chỉ ra rằng bài toán dịch vụ hiện tại quá dễ
cho một phương án không gửi vị trí truy vấn lên server.

Nguồn mới: [readout](../../artifacts/benchmarks/jisa_static_catalogue_control_20261006_v1/results.json),
[verification](../../artifacts/benchmarks/jisa_static_catalogue_control_20261006_v1/validation.json)
và [protocol](../../artifacts/benchmarks/jisa_static_catalogue_control_20261006_v1/protocol.json).
Không sửa nguồn, Q, budget, model hoặc điểm của pilot REM/Planar cũ.

## Đối chứng và cách chấm

Chạy trên toàn bộ 24 family / 192 session native đã từng được xem; sáu family
có nhãn `test` trong nguồn cũ vẫn là **development** trong phân tích này.
Không tạo private draw mới, không chọn lại defense hoặc attacker.

Full-catalogue control nhận toàn bộ `id/category/lat/lon` tại public epoch start
0, một lần cho mỗi family epoch [0,12000). Request chỉ gồm schema, public
catalogue version, public epoch ID và timestamp. Không chứa GPS, Q, category,
purpose, radius hay destination. Các câu trả lời sau đó thực hiện trên máy.
Catalogue version giữ nguyên và các POI đều static trong epoch.

Cùng một `QuerySpec`/reference được dùng cho bảy candidate pools: full catalogue;
Raw/REM/Planar chỉ phản hồi hiện tại; Raw/REM/Planar union các ID đã nhận từ đầu
epoch. Service pools lấy từ `wire_rows.json.gz` đã khóa, gồm L20 theo distance
ở mọi category. Cache chỉ thêm ID khi phản hồi tương ứng đã xuất hiện, không
đọc POI của các Q tương lai.

Bốn purpose dùng cùng directed road graph và lexical ID ties:

- **Nearest:** distance đường có hướng nhỏ nhất.
- **Fastest:** travel time từ length/speed công khai của graph.
- **Within radius:** distance ≤1000m rồi chọn top-5.
- **Minimum detour:** distance qua POI trừ direct distance tới destination.

Private destination ở đây là road state gần GPS cuối session, chỉ làm input
cho utility/local ranking. Không chuyển nó vào Q/planner/request. Đây là oracle
đánh giá có điều kiện, không mô phỏng cách một người dùng khai báo đích; study
sau nên ghi riêng private destination đã khai báo và lỗi local GPS.

Reference rỗng giữ `N/A`. Chấm Recall query trong mỗi session, average các
session có reference trong family, rồi average family; giữ đủ số query rỗng,
session/family undefined. Cách weighting mới này áp giống nhau cho mọi control,
nên số gần nhất có thể chênh nhẹ bảng distance-only của pilot cũ.

## Kết quả sáu family development mang nhãn test cũ

| Candidate pool | Nearest | Fastest | Radius 1000m | Detour |
|---|---:|---:|---:|---:|
| Full catalogue local | 100% | 100% | 100% | 100% |
| Raw current reply | 100% | 100% | 100% | 99,36% |
| REM current reply | 94,71% | 94,71% | 85,23% | 94,02% |
| Planar current reply | 96,97% | 96,96% | 90,85% | 96,13% |
| REM same epoch cache | 99,90% | 99,90% | 99,74% | 99,85% |
| Planar same epoch cache | 99,34% | 99,34% | 98,06% | 99,17% |

100% của full catalogue là **reference recovery khi reference tồn tại**,
không phải xác suất người dùng luôn tìm được POI. Radius có 6.174/9.060 query
reference rỗng; ba purpose còn lại đều có 648/9.060 query reference rỗng.
Các count này giữ nguyên cho mọi phương pháp. Full catalogue đạt 100% vì nó
có đúng cùng tập ứng viên với reference và cùng local ranking, không phải do
nhiễu hoặc một công thức metric mới.

Một lần bulk fetch có request **196B**, full-record response **36.132B**.
Sáu family tương ứng sáu fetch độc lập, không dùng cache chung giữa người dùng.

| Control, sáu family | Requests | Request bytes | Reply bytes | Total |
|---|---:|---:|---:|---:|
| Full catalogue / public epoch | 6 | 1.176 | 216.792 | 217.968B |
| Raw service stream | 1.510 | 203.489 | 7.658.607 | 7.862.096B |
| REM service stream | 7.550 | 1.140.292 | 41.235.772 | 42.376.064B |
| Planar service stream | 7.550 | 1.140.457 | 41.224.091 | 42.364.548B |

Đây là độ dài thật của compact UTF-8 JSON theo schema đã khai báo, gồm record
lặp trong các service replies. Chưa đo HTTP/TLS, radio, latency, CPU/RAM,
GNSS hoặc chi phí tải graph. Không giả lập timing. Road graph và POI access
rule là public input chung; planner cũ vốn đã dùng public catalogue/signatures.
Bulk access/version discovery/fixed public region là giả thiết, chưa được
xác nhận với một provider thật.

## Điều cần cải thiện để ứng dụng có ý nghĩa

1. Giữ local/bulk làm đối chứng và fallback thực dụng khi catalogue nhỏ, static
   và provider cho tải toàn bộ. Không dùng warm-cache gần 100% để tuyên bố Q
   planner cần thiết hoặc novel trong contract này.
2. Chốt API/provider contract trước khi đổi task: catalogue lớn/cross-region,
   bulk access hạn chế, hoặc trạng thái availability/price/travel time thực sự
   thay đổi và cần remote fetch. Đo riêng version/freshness và service success.
3. Nếu nghiên cứu dynamic service, planner chỉ được dùng public proxy/model
   hoặc replies đã nhận. Không cho nó biết toàn bộ provider state ở mọi Q,
   rồi gọi đó là postprocessing của protected history.
4. Với pilot planner mới, primary current-only và cùng cap/clock/K/L là phép
   so sánh rõ hơn để thấy contribution của việc chọn Q. Báo cùng cache sau đó;
   cache có lợi cho mọi comparator. Full static local control vẫn phải xuất hiện.

Không tính Hit/MAE coordinate attacker cho full catalogue bằng cách gán 0 hoặc
100%: control không phát coordinate query stream nên contract/attacker đó
không áp dụng trực tiếp. Fixed public region/epoch payload không phụ thuộc local
GPS/purpose, nhưng account/IP, region choice, activation và click vẫn cần threat
contract riêng. Đây không là claim ẩn danh hoặc giải quyết mọi scenario.

## Kiểm chứng và reusable API

[Helper](../../evaluation/static_catalogue_control.py) tách payload công khai
và `evaluate_local_purposes(...)` chỉ nhận local ranking/inputs/candidate IDs.
Root runner mới có thể dùng cùng evaluator mà không thay core Geo-I.

Tám tests kiểm tra: bốn mục đích thật sự khác nhau trên directed fixture;
full catalogue/reference equality; missing candidates; N/A; duplicate/invalid
IDs; coordinate-free request; exact payload bytes/version changes; và weighting
theo scope. Independent verifier không import helper/runner; nó dùng existing
`MultiPurposeRoadRanking.top` để kiểm tra toàn bộ **145.104 query / 1.015.728
ranked answers**, source hashes, causal pools, costs và summary arithmetic.

```bash
python -m pytest tests/test_jisa_static_catalogue_control.py -q
python -m experiments.verify_jisa_static_catalogue_control_20261006
```

Chạy mới vào thư mục mới, không ghi đè evidence:

```bash
python -m experiments.jisa_static_catalogue_control_20261006 \
  --out /private/tmp/jisa-static-catalogue-new
python -m experiments.verify_jisa_static_catalogue_control_20261006 \
  --out /private/tmp/jisa-static-catalogue-new
```

Không cần RNG master hoặc model pickle để recheck control này.
