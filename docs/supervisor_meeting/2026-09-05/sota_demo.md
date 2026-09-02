# Ranh giới thực nghiệm của benchmark sinh dữ liệu giả

Tài liệu này mô tả chính xác phần thực nghiệm đang chạy được cho luận văn. Đây
không phải bảng xếp hạng SOTA. Ba phương pháp từ bài báo hiện ở mức **bản thích
nghi có ánh xạ nguồn**; quy trình tự từ chối nhãn “tái hiện SOTA” nếu còn thiếu
thành phần cốt lõi.

## Chạy benchmark và web app

```bash
venv/bin/python -m pip install -r requirements-sumo.txt
venv/bin/python -m experiments.run_dummy_benchmark --quick
venv/bin/python -m web.benchmark_app
```

Mở `http://127.0.0.1:5000/`. Web app chỉ đọc artifact đã sinh, không có API chạy
thí nghiệm. Hai vùng dữ liệu được tách rõ:

- **attacker-visible**: đúng dữ liệu LSP/kẻ tấn công được phép quan sát;
- **evaluator-only**: quỹ đạo thật, nhãn thật và chỉ số; chỉ dùng để chấm điểm.

Các hình trực quan được kiểm tra SHA-256 trước khi web app phục vụ. Các route
evaluator mặc định chỉ phù hợp với localhost; phải tắt hoặc thêm xác thực trước
khi triển khai ra ngoài máy cá nhân.

## Artifact chuẩn

- `outputs/dummy_benchmark_results.json`: artifact v3, chứa provenance, thẻ
  phương pháp, public transcript, ground truth tách riêng và diagnostic metrics;
- `outputs/dummy_benchmark_map.html`: bản đồ tương tác evaluator-only, nhúng
  mạng đường OSM cục bộ nên vẫn đọc được khi tile trực tuyến không tải;
- `outputs/dummy_benchmark_preview.png`: ảnh tĩnh đối chiếu bốn cơ chế.

## Nguồn quỹ đạo

Mặc định, pipeline thực sự gọi Eclipse SUMO:

```text
Beijing.osm.gz
  -> netconvert: mạng passenger trong bbox đô thị
  -> randomTrips.py: sinh demand với seed cố định
  -> sumo: mô phỏng chuyển động và xuất FCD
  -> một trajectory được resample
  -> chạy qua mọi cơ chế trên cùng input
```

Đây là một controlled smoke scenario, chưa phải mô hình dân số Bắc Kinh đã hiệu
chỉnh. SUMO hiện chỉ chọn một quỹ đạo; chưa có scenario builder cho S1--S7,
population prior, POI labels hay nhiều người dùng. GeoLife chỉ được dùng khi chỉ
định rõ `--mobility-source geolife`; không có fallback im lặng.

SUMO chạy trên mạng passenger-only, còn candidate graph hiện là OSMnx graph đa
phương thức đã pin từ benchmark trước. Hai graph cùng nguồn/bbox nhưng chưa được
coi là đồng nhất. Benchmark kết luận cuối phải thống nhất graph hoặc kiểm chứng
mapping giữa hai graph.

## Các phương pháp hiện chạy

| Phương pháp | Giao diện | Đã cài | Còn thiếu để tái hiện paper |
|---|---|---|---|
| TransProtect adaptation | một quỹ đạo vị trí thay thế | candidate trên mạng đường, điểm utility/reachability/context, seed cố định | GCN/transformer và traffic pipeline, VehiTrack hoàn chỉnh, Rome/SF reproduction |
| AnotherMe adaptation | một quỹ đạo ảo thay thế | biến đổi nhất quán toàn quỹ đạo, route-aware snapping | virtual-user/history, POI/routing tương đương AMap, classifier/mobile reproduction |
| Semantic-correlation adaptation | thật + `K-1` dummy theo chuỗi | candidate ID ổn định, điểm temporal/reachability, semantic hook | LSTM/attention, grid/transition pipeline, POI dataset và ASR reproduction |
| Geo-I anchored dummy | `K` quỹ đạo giả, không công bố quỹ đạo thật | REM anchor và post-processing chỉ dùng anchor công khai | population/POI prior, tối ưu theo attacker, quản lý ngân sách toàn trajectory |

Chi tiết tới từng thành phần và phiên bản nguồn nằm trong
[`benchmark/README.md`](../../../benchmark/README.md) và `method_inventory` của
artifact JSON.

## Những gì kết quả hiện tại chứng minh

Kết quả v3 chỉ đủ để chứng minh:

1. cả bốn cơ chế chạy trên cùng quỹ đạo SUMO và cùng seed/provenance;
2. output contract được kiểm tra, ground truth không lọt vào attacker view;
3. bản đồ/preview/web app đọc được kết quả trên mạng đường; và
4. mỗi con số luôn gắn với mức độ cài đặt và các thành phần còn thiếu.

Kết quả **chưa** chứng minh cơ chế nào riêng tư hơn. Metrics hiện chủ yếu kiểm
tra hình học, tính liên tục và chi phí request. Trước khi đưa bảng so sánh vào
luận văn cần: POI utility chuẩn, context-aware continuous dummy-filtering
attacker, nhiều scenario/seed/trajectory, khoảng tin cậy, paper-specific
validation và một graph thống nhất.

LSPPM-SI vẫn là đối chứng phụ quan trọng nhưng chưa nằm trong executable set vì
thiếu mã nguồn, POI data và preprocessing. Bộ ba chạy hiện tại dùng
Semantic-correlation 2026 thay cho LSPPM-SI vì phù hợp trực tiếp hơn với attack
surface liên tục theo thời gian + ngữ nghĩa của luận văn.
