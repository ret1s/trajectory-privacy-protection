# Ranh giới thực nghiệm của benchmark sinh dữ liệu giả

Tài liệu này mô tả chính xác phần thực nghiệm đang chạy được cho luận văn. Đây
không phải bảng xếp hạng SOTA. Ba phương pháp từ bài báo đã có comparator local
chạy end-to-end và kiểm thử theo thành phần. Chúng vẫn ở mức **bản thích nghi
clean-room có ánh xạ nguồn**; quy trình tự từ chối nhãn “tái hiện SOTA” nếu chưa
có đủ artifact để đối chiếu kết quả paper.

## Chạy benchmark và web app

```bash
venv/bin/python -m pip install -r requirements-sumo.txt
venv/bin/python -m experiments.run_dummy_benchmark --quick
venv/bin/python -m web.benchmark_app
```

`--epsilon` là ngân sách theo mét của mô hình luận văn; TransProtect dùng cờ
riêng `--transprotect-epsilon-per-km` (mặc định 5 km$^{-1}$ theo paper) và được
đổi đơn vị sang m$^{-1}$ bên trong. Hai giá trị không bị coi là cùng nghĩa chỉ
vì đều mang tên epsilon.

Mở `http://127.0.0.1:5000/`. Web app chỉ đọc artifact đã sinh, không có API chạy
thí nghiệm. Hai vùng dữ liệu được tách rõ:

- **attacker-visible**: đúng dữ liệu LSP/kẻ tấn công được phép quan sát;
- **evaluator-only**: quỹ đạo thật, nhãn thật và chỉ số; chỉ dùng để chấm điểm.

Các hình trực quan được kiểm tra SHA-256 trước khi web app phục vụ. Cấu hình WSGI
mặc định tắt toàn bộ route evaluator; lệnh `python -m web.benchmark_app` chỉ bật
chúng khi bind vào localhost. Phải thêm xác thực trước khi triển khai ra ngoài
máy cá nhân.

## Artifact chuẩn

- `outputs/dummy_benchmark_results.json`: artifact v4, chứa provenance, thẻ
  phương pháp, public transcript, ground truth tách riêng và diagnostic metrics;
- `outputs/dummy_benchmark_map.html`: bản đồ tương tác evaluator-only, nhúng
  hình học mạng đường OSM và không bật raster tile theo mặc định; các thư viện
  giao diện Folium/Leaflet vẫn có thể cần CDN hoặc bộ nhớ đệm của trình duyệt;
- `outputs/dummy_benchmark_preview.png`: ảnh tĩnh đối chiếu bốn cơ chế và là
  phương án xem hoàn toàn ngoại tuyến.

## Nguồn quỹ đạo

Mặc định, pipeline thực sự gọi Eclipse SUMO:

```text
Beijing.osm.gz
  -> netconvert: mạng passenger trong bbox đô thị
  -> randomTrips.py: sinh demand với seed cố định
  -> sumo: mô phỏng chuyển động và xuất FCD
  -> giữ một trajectory làm test; các xe còn lại làm training context
     đã bỏ ID/lane/edge
  -> trajectory test được resample
  -> chạy qua mọi cơ chế trên cùng input
```

Đây là một controlled smoke scenario, chưa phải mô hình dân số Bắc Kinh đã hiệu
chỉnh. Một xe được giữ riêng để chấm điểm; TransProtect chỉ học proxy chuyển tiếp
từ các xe SUMO còn lại trong cùng lần mô phỏng. Đây là phép giữ riêng theo xe,
không phải một train/test simulation split độc lập. Chưa có scenario builder cho S1--S7, population prior
đã hiệu chỉnh hay POI labels thật. GeoLife chỉ được dùng khi chỉ định rõ
`--mobility-source geolife`; không có fallback im lặng.

SUMO và các cơ chế hiện dùng cùng một mạng passenger-only `.net.xml`; graph đó
được nạp lại thành miền candidate cùng node, cạnh, chiều đường và polyline. FCD
là điểm liên tục trên làn, còn đầu ra cơ chế nằm trên miền vertex/edge của chính
mạng này.

## Các phương pháp hiện chạy

| Phương pháp | Giao diện | Đã cài | Còn thiếu để tái hiện paper |
|---|---|---|---|
| TransProtect adaptation | một quỹ đạo vị trí thay thế | Eq.13 utility, `h + α/Δc`, top-K, Laplace/LP, causal model interface; Markov proxy học trên xe SUMO nền; utility `N×M` | weights/cấu hình GCN-transformer gốc, Rome/SF table parity và VehiTrack end-to-end parity |
| AnotherMe adaptation | một quỹ đạo ảo thay thế | public VTGA: speed/mode, lọc+dense route, Bezier, replay tốc độ, noise; local virtual endpoint + router | AMap/GCJ02/POI response parity, canonical mobile workflow và classifier/mobile parity |
| Semantic-correlation adaptation | thật + `K-1` dummy theo chuỗi | lưới/công thức/LSTM-attention inference/selector; local OSM semantics + empirical predictor | AMap semantic assets, trained weights, các hàm paper không công bố và ASR/DER parity |
| Geo-I anchored dummy | `K` quỹ đạo giả, không công bố quỹ đạo thật | REM anchor và post-processing chỉ dùng anchor công khai | population/POI prior, tối ưu theo attacker, quản lý ngân sách toàn trajectory |

Chi tiết tới từng thành phần và phiên bản nguồn nằm trong
[`benchmark/README.md`](../../../benchmark/README.md) và `method_inventory` của
artifact JSON.

## Những gì kết quả hiện tại chứng minh

Kết quả v4 chỉ đủ để chứng minh:

1. cả bốn cơ chế chạy trên cùng quỹ đạo SUMO và cùng seed/provenance; ba
   comparator không còn dùng các prototype heuristic ban đầu;
2. output contract được kiểm tra, ground truth không lọt vào attacker view;
3. xe test không nằm trong training context của proxy TransProtect;
4. bản đồ/preview/web app đọc được kết quả trên mạng đường; và
5. mỗi con số luôn gắn với mức độ cài đặt và các thành phần còn thiếu.

Kết quả **chưa** chứng minh cơ chế nào riêng tư hơn. Metrics hiện chủ yếu kiểm
tra hình học, tính liên tục và chi phí request. Trước khi đưa bảng so sánh vào
luận văn cần: POI utility chuẩn, context-aware continuous dummy-filtering
attacker, nhiều scenario/seed/trajectory, khoảng tin cậy, paper-specific
   validation và các scenario đe doạ hoàn chỉnh.

LSPPM-SI vẫn là đối chứng phụ quan trọng nhưng chưa nằm trong executable set vì
thiếu mã nguồn, POI data và preprocessing. Bộ ba chạy hiện tại dùng
Semantic-correlation 2026 thay cho LSPPM-SI vì phù hợp trực tiếp hơn với attack
surface liên tục theo thời gian + ngữ nghĩa của luận văn.
