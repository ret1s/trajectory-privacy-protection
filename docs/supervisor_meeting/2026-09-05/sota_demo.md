# Demo sơ bộ: ba hướng SOTA và mô hình đề xuất

**Trạng thái:** chạy được bằng một simulation SUMO thật để minh họa kiến trúc và
giao diện đầu ra; **không phải reproduction chính thức và chưa phải benchmark
dùng để kết luận mô hình nào tốt hơn**.

## Chạy demo

Từ thư mục gốc của repository:

```bash
venv/bin/python -m pip install -r requirements-sumo.txt
venv/bin/python -m experiments.run_sota_demo --quick
```

Kết quả được ghi vào:

- `outputs/sota_demo_map.html`: bản đồ tương tác của quỹ đạo do SUMO mô phỏng;
- `outputs/sota_demo_results.json`: public transcript, ground truth riêng cho
  evaluator, SUMO route/speed/lane truth, provenance và một số diagnostic
  metrics;
- terminal: ba bảng kết quả tách theo output contract.

Mặc định không có GeoLife hoặc synthetic fallback: thiếu SUMO/OSM thì chương
trình dừng và báo rõ dependency. GeoLife chỉ còn là phép kiểm tra tùy chọn trên
dữ liệu thực:

```bash
venv/bin/python -m experiments.run_sota_demo --quick --mobility-source geolife
```

## Dữ liệu demo SUMO được tạo như thế nào?

Pipeline mặc định thực sự gọi ba công cụ của Eclipse SUMO:

```text
Beijing.osm.gz
  -> netconvert: cắt bbox đô thị, chỉ giữ đường cho passenger vehicle
  -> randomTrips.py: sinh demand/route với seed cố định
  -> sumo: chạy movement physics và xuất FCD theo thời gian
  -> parse x=longitude, y=latitude
  -> cùng một trajectory đi qua cả bốn protection prototypes
```

Mạng và các file trung gian được tạo trong `cache/sumo_demo/` nên không được
commit. JSON kết quả lưu phiên bản SUMO, hai seed, lệnh chạy và SHA-256 của OSM,
network, routes và FCD. Theo tài liệu chính thức, FCD có thể xuất WGS84 bằng
`--fcd-output.geo`; `randomTrips.py` với seed cố định tạo demand có thể lặp lại
([FCD output](https://eclipse.dev/sumo/docs/Simulation/Output/FCDOutput.html),
[randomTrips](https://eclipse.dev/sumo/docs/Tools/Trip.html)).

Đây mới là **controlled smoke scenario**, không phải population model đã hiệu
chỉnh cho Bắc Kinh. Tài liệu SUMO cũng cảnh báo random trips không mặc nhiên là
mobility thực tế. Giai đoạn benchmark sẽ giữ SUMO làm movement engine nhưng thay
random demand bằng scenario builder riêng cho S1--S7. Dữ liệu bản đồ thuộc
© OpenStreetMap contributors, giấy phép ODbL.

## Bốn prototype hiện có

| Prototype | Ý tưởng đã minh họa | Dữ liệu LSP nhìn thấy | Phần quan trọng chưa cài |
|---|---|---|---|
| `TransProtectLite` | Chọn pseudolocation trên mạng đường bằng utility, reachability và context score | Một vị trí thay thế tại mỗi thời điểm | GCN/Transformer, traffic model, candidate pipeline và VehiTrack chính thức |
| `AnotherMeLite` | Dịch chuyển/biến đổi nhất quán cả quỹ đạo rồi snap lên mạng đường | Một quỹ đạo ảo thay thế | Virtual-user model, POI mapping, learned mobility pattern và mobile workflow |
| `SemanticDummyLite` | Sinh `K-1` dummy có temporal/reachability score; có hook cho semantic category | Vị trí thật + `K-1` dummy, với ID ổn định theo thời gian | LSTM/attention, semantic predictor, candidate filters và tham số đúng paper |
| `GeoIAnchoredDummyTrajectoriesLite` | REM tạo private anchor; chỉ hậu xử lý anchor công khai để sinh `K` dummy track hợp lý sơ bộ | `K` dummy track; quỹ đạo thật không phải thành viên public | Population/POI prior, group statistics, attacker-aware optimization, budget manager và proof ở mức toàn trajectory |

Ba prototype đầu chỉ là **paper-inspired clean-room sketches** để có demo sớm.
Tên paper trong source code dùng để chỉ hướng ý tưởng, không có nghĩa kết quả này
là kết quả của TransProtect, AnotherMe hay Liu--Peng--Zhou.

## Quy tắc đọc kết quả

Demo tách ba track vì câu hỏi đánh giá khác nhau:

1. **Replacement trajectory:** TransProtectLite và AnotherMeLite; xem
   displacement, QoS hình học, on-road và speed violation.
2. **Real + dummies:** SemanticDummyLite; xem kích thước tập, khoảng cách/spread,
   dummy survival và sau này là top-1/rank/MRR dưới attacker.
3. **Dummy-only:** mô hình đề xuất; xem độ phủ của batch, reconstruction error và
   tính hợp lý của từng track; không có “real index” trong output.

Không so trực tiếp các cột số giữa ba track và không gọi bảng hiện tại là
leaderboard. Các metric hiện tại chủ yếu là sanity check hình học; chưa có
context-aware continuous dummy-filtering attacker đã hiệu chỉnh.

## Boundary riêng tư đã cài

`core/demo_protocol.py` tách rõ:

- `PublicTranscript`: đúng dữ liệu LSP/attacker được phép đọc;
- `EvaluationTruth`: quỹ đạo thật và nhãn real candidate, chỉ dùng offline để
  chấm điểm.

JSON cố ý ghi hai object `attacker_view` và `evaluator_truth` riêng. Attack code
tiếp theo chỉ được nhận `attacker_view`. Với mô hình đề xuất, public transcript
chỉ chứa các dummy track; REM anchors cũng không được serialize cho attacker.

## Bước hoàn thiện tiếp theo

- thay các heuristic `*Lite` bằng reproduction/adaptation theo paper và pin rõ
  phiên bản code/dataset;
- thay random SUMO demand bằng controlled scenario builder cho từng threat S1--S7;
- bổ sung POI labels, population prior và scenario nhiều người dùng/OMoSim;
- cài attacker chung theo road, time, POI và population context;
- bổ sung metric đúng từng contract và chuẩn hóa request/communication cost;
- chỉ sau đó mới chạy benchmark nhiều seed và đưa ra kết luận thực nghiệm.

Một giới hạn kỹ thuật hiện tại: trajectory thật được SUMO chạy trên mạng
passenger-only, còn candidate graph của bốn prototype vẫn là OSMnx graph đa
phương thức đã pin từ benchmark cũ. Hai graph cùng OSM/bbox và pilot hiện khớp
hình học, nhưng bản benchmark tiếp theo phải xây candidate graph trực tiếp từ
SUMO network hoặc từ một OSMnx drive graph tương ứng.
