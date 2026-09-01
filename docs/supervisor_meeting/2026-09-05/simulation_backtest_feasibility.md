# Chốt hướng mô phỏng và khả năng đưa SOTA vào backtest

**Ngày kiểm tra:** 01/09/2026
**Phạm vi:** các scenario S1--S7 trong [kế hoạch nghiên cứu](../2026-09-05_research_plan.md), các mô hình dummy-generation đã chọn ở Mục 6, và hệ thống backtest hiện có trong repository.

## Kết luận ngắn

1. **Không có một simulator sẵn có nào tự tạo đầy đủ S1--S7.** Các công cụ hiện tại chủ yếu tạo chuyển động thật; chúng không tự tạo truy vấn LBS, nội dung truy vấn, output đã bảo vệ, transcript mà LSP nhìn thấy, attacker hay privacy metrics.
2. **Không cần tự viết lại toàn bộ mobility simulator.** Chọn **SUMO** làm engine chuyển động trên mạng đường OSM, dùng **OMoSim** khi cần lịch hoạt động nhiều ngày/population, rồi tự xây một lớp Python mỏng để điều khiển scenario, query, group, protection, attack và metrics.
3. **Backtest hiện tại chưa chạy trực tiếp được toàn bộ SOTA.** Nó đang cố định giao diện `một điểm thật -> một pseudolocation`. TransProtect gần giao diện này nhất; AnotherMe cần adapter ở mức cả quỹ đạo; LSPPM-SI và Liu semantic 2026 cần giao diện tập `thật + K-1 dummy`; Liu fake-query 2026 cần giao diện event/protocol.
4. **Một phần metrics hiện tại dùng lại được, nhưng không phải mọi metric đều có cùng ý nghĩa giữa các giao diện.** Geometry/realism có thể tái sử dụng có điều kiện. Privacy attack, POI semantics, route cost và overhead cần mở rộng. Không được lấy các cột attacker hiện tại để xếp hạng SOTA vì likelihood hiện chỉ chính xác cho REM.
5. **Quyết định triển khai:** giữ benchmark Internship 2 như regression test; xây `dummy benchmark v2` riêng với transcript tổng quát và tách ba track. Chưa viết simulator từ đầu; chỉ viết custom scenario/query layer và adapter.

---

## 1. Simulator nào phù hợp?

### 1.1. Lựa chọn chính: SUMO

[SUMO](https://sumo.dlr.de/docs/) phù hợp nhất với codebase Python hiện tại vì:

- nhập mạng đường từ OpenStreetMap bằng `netconvert`; có thể nhập thêm POI/polygon từ OSM bằng `polyconvert` ([OSM import](https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html));
- `person` có chuỗi `walk`, `ride`, `stop`; stop có thời lượng, thời điểm kết thúc và `actType`, đủ để biểu diễn ở nhà, đi làm hoặc mua sắm ([person plans](https://sumo.dlr.de/docs/Specification/Persons.html));
- TraCI/libsumo cho Python truy cập person, vehicle, route, POI và trạng thái mô phỏng theo từng timestep ([TraCI Python](https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html));
- FCD output cung cấp ID, thời gian, vị trí, tốc độ, edge/lane và có thể xuất WGS84, nên chuyển được thành GPS-like traces cho backtest ([FCD output](https://sumo.dlr.de/docs/Simulation/Output/FCDOutput.html));
- hỗ trợ seed cố định để tái lập thí nghiệm ([randomness](https://sumo.dlr.de/docs/Simulation/Randomness.html));
- vẫn đang được duy trì; trang tải chính thức liệt kê SUMO 1.27.1 phát hành ngày 25/06/2026 ([downloads](https://sumo.dlr.de/docs/Downloads.php)).

SUMO **không** tự tạo query như “bệnh viện gần nhất”, không biết query nào là nhạy cảm, không sinh dummy, không chạy attacker và không định nghĩa group-correlation threat. Vì vậy SUMO là **movement engine**, không phải toàn bộ experimental system.

### 1.2. Bổ sung khi cần: OMoSim

[OMoSim](https://github.com/L-Strobel/omosim) phù hợp để sinh demand trước khi đưa vào engine/backtest:

- tạo nhiều agent có ID và home location;
- tạo daily activity diary gồm `HOME`, `WORK`, `SCHOOL`, `SHOPPING`, `OTHER`;
- mỗi trip có start time, mode, duration, distance và tùy chọn path coordinates;
- chạy nhiều ngày, cố định seed, cấu hình population strata;
- xuất JSON, MATSim XML hoặc SQLite;
- có giấy phép MIT và đang được phát triển tích cực.

Giới hạn quan trọng: mô hình mặc định được hiệu chỉnh bằng khảo sát di chuyển của Đức. Chính tác giả cảnh báo độ chính xác ngoài Đức chưa chắc chắn. OMoSim cũng chỉ có activity category thô, không tạo query content và population strata không đồng nghĩa với nhóm người đi cùng nhau. Vì vậy:

- dùng OMoSim để tạo **controlled multi-day scenarios**;
- không gọi output mặc định của OMoSim là population đại diện cho Bắc Kinh;
- nếu dùng cho kết luận về realism, phải hiệu chỉnh/đối chiếu với GeoLife hoặc dữ liệu địa phương.

### 1.3. Các lựa chọn đã loại khỏi stack chính

| Công cụ | Điểm mạnh | Vì sao không chọn làm engine chính |
|---|---|---|
| [MATSim](https://www.matsim.org/) | Agent/activity/population, route và replanning quy mô lớn | Java-heavy và tích hợp Python không gọn bằng TraCI; chỉ cần nếu bài toán chuyển sang transport-demand quy mô lớn |
| [Eclipse MOSAIC](https://eclipse.dev/mosaic/) | Ghép SUMO với application/network simulators | Hợp lý khi cần cellular/V2X/communication simulation chi tiết; quá nặng cho mục tiêu privacy backtest hiện tại |
| [The ONE](https://akeranen.github.io/the-one/) | Working-day/group mobility và DTN messaging | Tập trung DTN, không phải LBS query; confidence về maintenance thấp hơn |
| [BonnMotion](https://bonnmotion.sys.cs.uos.de/) | Nhiều mobility models, group movement, seed/export | Thiếu activity/POI/query semantics; map-based stack dựa vào phiên bản OSRM cũ |
| MobiSim | MANET trace hoặc energy simulation tùy project | Không phải modern OSM urban trajectory engine phù hợp với bài toán này |

### 1.4. Coverage S1--S7

Ký hiệu: `Có` = tạo được mobility ground truth cần thiết; `Một phần` = có nền tảng nhưng cần operator riêng; `Không` = phải tự xây.

| Scenario | SUMO | OMoSim | Phần bắt buộc tự xây |
|---|---:|---:|---|
| S1 -- một lần công bố | Có | Có | Chọn query event và tạo observed transcript |
| S2 -- báo cáo lặp tại điểm dừng | Một phần | Một phần | Lịch phát query lặp bên trong dwell interval |
| S3 -- chuỗi liên tục có tương quan | Có | Có | Query cadence và attack window |
| S4 -- quay lại địa điểm quen thuộc | Một phần | Có | Ép số lần revisit, tách train/test theo user và nhãn significant location |
| S5 -- dự đoán destination/next route | Có | Có | Giữ prefix cho attacker; cất destination/future route làm evaluator-only label |
| S6 -- suy luận query/POI | Không | Một phần | POI taxonomy, query intent/content, fake-query policy và kết quả LBS |
| S7 -- tương quan nhóm người dùng | Một phần | Một phần | `group_id`, shared trips, co-location events và group/population prior |

Kết luận của ma trận: **SUMO + OMoSim vẫn không đủ nếu đứng riêng lẻ**, nhưng đủ làm nền để ta không phải tự viết movement physics/routing. Phần cần tự viết là logic nghiên cứu của luận văn, không phải một simulator đô thị hoàn chỉnh.

---

## 2. Kiến trúc benchmark được chốt

```text
Mobility source
  ├── SUMO FCD / TraCI                 controlled road-time movement
  ├── OMoSim JSON -> SUMO adapter      optional multi-day activity demand
  └── GeoLife / T-Drive replay         real-data validation
                    |
                    v
Canonical ground-truth trajectory/event schema
                    |
                    v
Custom Scenario + Query layer          S1--S7 operators
                    |
                    v
Protection adapters                    ours + SOTA
                    |
                    v
Attacker-visible transcript             never contains truth labels
                    |
                    v
Attack runner + privacy/utility metrics
```

### 2.1. Vai trò của dữ liệu

- **Benchmark mô phỏng có kiểm soát:** dữ liệu chính để chứng minh cơ chế xử lý từng threat scenario, vì có đầy đủ ground truth và có thể thay đổi một yếu tố tại một thời điểm.
- **GeoLife/T-Drive hoặc dữ liệu thực tương đương:** validation để kiểm tra kết luận không chỉ đúng trên scenario do ta thiết kế.
- Không dùng synthetic mobility để tuyên bố đại diện cho hành vi dân số Bắc Kinh nếu chưa hiệu chỉnh.

### 2.2. Canonical ground-truth record tối thiểu

```text
run_id, seed, split, user_id, group_id,
timestamp, lat, lon, edge_id, mode, speed,
activity_id, activity_type, dwell_id,
poi_id, poi_category,
query_id, query_type, query_content_class,
destination_id, ground_truth_route
```

Mọi generator/replay loader phải chuyển về schema này. Tất cả thông tin bí mật như real member, destination tương lai và real/fake-event flag chỉ nằm trong truth object của evaluator.

### 2.3. Khi nào mới cần viết trajectory generator riêng?

Chưa cần làm ngay. Thứ tự xử lý là:

1. tạo plan/scenario có kiểm soát;
2. để SUMO thực thi route, tốc độ, stop và timestamp;
3. thêm query/group operators bằng Python;
4. chỉ nếu SUMO/OMoSim không cho ta ép đúng scenario, viết **custom plan builder** chọn origin, destination, stop, revisit và group schedule; movement vẫn do SUMO chạy.

Chỉ viết cả trajectory engine từ đầu nếu plan builder + SUMO vẫn không tạo được ground truth cần thiết. Hiện chưa có bằng chứng cho thấy phải đi đến bước đó.

---

## 3. Backtest hiện tại đang hỗ trợ gì?

### 3.1. Output contract hiện tại

Backtest đang cố định:

```python
reset()
perturb(lat, lon, t=None) -> (reported_lat, reported_lon)
```

Evidence trong repository:

- contract một điểm vào/một điểm ra: `core/mechanisms.py:4-6`;
- danh sách cơ chế được hard-code: `experiments/run_benchmark.py:44-55`;
- mỗi điểm thật chỉ gọi `perturb` đúng một lần: `experiments/run_benchmark.py:132-137`;
- metrics giả định hai chuỗi đã căn hàng và thường cùng độ dài: `evaluation/metrics.py:33-99`;
- output JSON chỉ lưu scalar metrics, không lưu candidate set, transcript, posterior hay truth index: `experiments/run_benchmark.py:179-200`.

Do đó không thể chỉ thêm một class SOTA vào `build_mechanisms()` rồi gọi đó là reproduction công bằng.

### 3.2. Phần có thể tái sử dụng

- OSM graph wrapper, projection và graph provenance;
- GeoLife parser;
- semantic RNG và seed handling;
- displacement, Hausdorff, DTW, on-road rate và speed violation;
- bộ khung Bayesian/HMM để phát triển attacker mới;
- graph hash, run metadata và raw per-trajectory results.

### 3.3. Các giới hạn cần sửa trước khi benchmark SOTA

- POI hiện là 500 road vertices lấy mẫu ngẫu nhiên, không phải POI thật và không có category (`evaluation/metrics.py:102-128`). Nó chỉ là smoke test, chưa dùng được cho semantic attack.
- Bayesian/HMM likelihood hiện chính xác cho REM; với cơ chế khác chỉ là proxy (`evaluation/attacks.py:69-81`, `115-120`). Không dùng các cột này để xếp hạng privacy của SOTA.
- `runtime_s` hiện bao gồm generation, metrics và attacks; nó không phải generation latency của cơ chế (`experiments/run_benchmark.py:122-168`).
- graph Bắc Kinh hiện là multimodal, chưa phải vehicle-only/time-dependent graph cần cho TransProtect.
- môi trường hiện tại chưa có MATLAB/Octave, PyTorch hoặc package `haversine`; muốn chạy ML/MATLAB artifact phải có environment riêng được pin.

---

## 4. Có thể re-implement các SOTA trên cùng hệ thống không?

**Có thể, nhưng không phải plug-and-play.** Ta phải xây backtest v2 có nhiều output contracts. Mức khả thi cụ thể:

### 4.1. TransProtect / VehiTrack -- khả thi nhất

- **Giao diện:** top-K candidate chỉ là bước nội bộ; cuối cùng công bố một pseudolocation. Đây là mô hình gần contract hiện tại nhất.
- **Mã nguồn:** repository MATLAB công khai [VehiTrack](https://github.com/sourabhy1797/VehiTrack), snapshot đã kiểm tra `035684c6c666a9af7cbd9984d92300000eb65536`.
- **Không phải drop-in:** artifact gồm scripts, `.mat` files và dữ liệu Rome/San Francisco; đường dẫn/dataset bị hard-code và không có training implementation Python hoàn chỉnh như pipeline mô tả trong paper. Cần port độc lập sang Python hoặc chạy reproduction Rome/SF riêng trước.
- **Dữ liệu cần thêm:** directed vehicle graph, travel time/traffic prior, train/test data đủ lớn và node mapping.
- **Metrics dùng lại:** displacement, on-road, speed, Hausdorff/DTW và EIE về mặt khái niệm.
- **Metrics phải thêm:** expected travel-cost/route-time loss và VehiTrack mechanism-aware attacker. HMM hiện tại không thay cho VehiTrack.
- **So sánh tham số:** paper dùng khoảng `5, 7.5, 10 km^-1`, tương ứng `0.005, 0.0075, 0.01 m^-1`; chỉ `0.01 m^-1` giao với grid hiện tại. Không so cùng tên epsilon nếu unit/domain khác nhau.

**Đánh giá:** khả thi sau khi có `Point/TrajectoryAdapter`; ưu tiên tích hợp đầu tiên.

### 4.2. AnotherMe -- có code nhưng cần làm sạch và batch adapter

- **Giao diện:** nhận cả trajectory và sinh một virtual trajectory có thể khác số điểm/timestamp.
- **Mã nguồn:** repository [AnotherMe](https://github.com/fang-zhiyou/AnotherMe), snapshot đã kiểm tra `0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f`.
- **Không phải drop-in:** standalone Python gọi live AMap APIs, có import bị thiếu, đường dẫn Windows hard-code, giả định timestamp 3 giây và không có dependency/seed contract hoàn chỉnh.
- **Cách làm an toàn:** re-implement thuật toán theo paper/code structure; thay AMap bằng local OSM/SUMO routing; chuẩn hóa `(lat, lon, t)`; dùng history theo user và held-out trace.
- **Metrics dùng lại:** Hausdorff, DTW, on-road và speed nếu output timestamp hợp lệ; displacement/POI chỉ sau khi resample và định nghĩa utility phù hợp.
- **Metrics phải thêm:** trajectory recognition/detectability của một classifier chung, generation p50/p95; battery chỉ đo được nếu thực nghiệm trên mobile.
- **Privacy budget:** AnotherMe không có Geo-I epsilon tương đương. So ở cùng utility hoặc cùng attack rate, không ép so cùng epsilon.

**Đánh giá:** làm được một baseline VTGA độc lập; reproduction toàn bộ mobile system có effort và uncertainty cao hơn.

### 4.3. LSPPM-SI -- chỉ phù hợp track trajectory-set/offline

- **Giao diện:** trajectory thật + `k-1` dummy trajectories trên các stopovers.
- **Nguồn:** [paper Scientific Reports 2025](https://www.nature.com/articles/s41598-025-88553-9); không có public code được nêu, dữ liệu chỉ cung cấp theo yêu cầu.
- **Scope mismatch:** paper hiện xử lý trajectory publishing; chính phần kết luận đặt real-time LBS integration là future work. Vì vậy không gọi nó là online drop-in comparator.
- **Phải cài lại:** stopover extraction, POI hierarchy, Hilbert indexing, spatial influence, diverse semantic selection và Kuhn--Munkres matching.
- **Metrics gốc:** access entropy, anonymous-region information loss và semantic protection degree; hiện repo chưa có các metric này.
- **Common metrics cần thêm:** attacker top-1/MRR, candidate-path survival, real POI semantics và `K` lần communication/storage cost.

**Đánh giá:** high-effort clean-room implementation; giữ ở Track B hoặc secondary/offline comparison.

### 4.4. Liu semantic-correlation 2026 -- comparator online tốt cho set contract

- **Giao diện:** mỗi query gửi real location + `K-1` dummy; LSTM/attention dự đoán semantic type rồi lọc candidate bằng transition/time weight.
- **Nguồn:** [paper Springer 2026](https://link.springer.com/article/10.1007/s44443-026-00899-w); không có public code được nêu, dữ liệu cung cấp theo yêu cầu.
- **Phải cài lại:** PyTorch model, user-level historical sequences, time/weekday/POI embeddings, transition matrix, POI taxonomy và candidate-set evaluator.
- **Domain mismatch:** paper dùng grid 100 x 100 ở trung tâm Bắc Kinh; code hiện dùng OSM vertices. Phải giữ grid như paper hoặc công bố rõ adaptation grid-to-road.
- **Metrics phải thêm:** anonymity success rate, dummy effectiveness/survival, top-1/rank/MRR và generation delay.
- Epsilon trong paper là **ngưỡng transition probability**, không phải Geo-I epsilon.

**Đánh giá:** khả thi nhưng high effort; ứng viên chính của Track B nếu thesis chốt giao diện real + `K-1`.

### 4.5. Liu fake-query 2026 -- protocol wrapper

- **Giao diện:** chèn event chứa toàn dummy giữa hai real query; cần một base `real + K-1` generator trước.
- **Nguồn:** [paper Springer 2026](https://link.springer.com/article/10.1007/s44443-025-00438-z); không có public code được nêu, dữ liệu cung cấp theo yêu cầu.
- **Phải cài lại:** event stream có variable length/timestamps, evaluator-only real/fake labels, query content, request/response cost và reachability/path-correlation attacker.
- **Metrics phải thêm:** ASR, fake-query detection, path/dummy survival, traffic multiplier, bytes, latency, energy và service load.

**Đánh giá:** cài dưới dạng wrapper Track C sau khi Track B hoạt động; không phải class `perturb()` độc lập.

### 4.6. Kết luận về shortlist

Chưa có ba SOTA “apples-to-apples” chạy được ngay trong backtest hiện tại. Thứ tự hợp lý là:

1. **TransProtect** -- port độc lập, Track A, cùng VehiTrack attacker;
2. **AnotherMe** -- trajectory replacement adapter, Track A;
3. **Liu semantic 2026** -- clean-room set-based comparator, Track B;
4. **fake-query 2026** -- protocol wrapper, Track C;
5. **LSPPM-SI** -- secondary/offline trajectory-set comparator.

Nếu output cuối của thesis là dummy-only batch hậu xử lý từ một Geo-I anchor, không phương pháp `real + K-1` nào hoàn toàn cùng contract. Chúng vẫn có giá trị đối chứng nhưng phải tách track và chuẩn hóa utility/request/communication; không ghép thành một leaderboard duy nhất.

Hai repository AnotherMe và VehiTrack không có file giấy phép rõ ràng ở root tại các snapshot đã kiểm tra. Không vendor/copy code trực tiếp vào repository luận văn trước khi làm rõ quyền sử dụng; independent reimplementation theo paper an toàn hơn.

---

## 5. Metrics nào thực sự áp dụng được?

### 5.1. Matrix theo output contract

| Metric | Một pseudolocation / replacement | Real + `K-1` set | Fake-query protocol | Trạng thái hiện tại |
|---|---:|---:|---:|---|
| Mean/P95 displacement | Có | Không dùng làm utility chính nếu real nằm trong set | Chỉ real events | Có mean/max; cần P95 |
| Hausdorff/DTW | Có cho replacement trajectory | Có cho từng candidate/path realism | Có điều kiện | Đã có |
| On-road/speed/reachability | Có | Có cho từng dummy/path | Có | On-road/speed đã có; reachability cần thêm |
| POI Top-k preservation | Có | Query thật có thể đúng tuyệt đối; phải tính union/filter/overhead | Có điều kiện | Hiện dùng POI giả; phải thay bằng POI thật có category |
| EIE và `P(error <= r)` | Privacy metric chính | Có nếu attacker posterior được quy về estimate | Có | EIE có nhưng attacker chỉ exact cho REM; radius curve có ở averaging study |
| Top-1 attack success / MRR | Không cùng nghĩa nếu truth không nằm trong output | Privacy metric chính | Privacy metric chính | Chưa có |
| Posterior entropy / `k_eff` | Có trên hypothesis space | Có trên K candidates/paths | Có | Posterior đang bị discard; cần persist/expose |
| Dummy/path survival qua từng filter | Không áp dụng | Rất quan trọng | Rất quan trọng | Chưa có |
| Recognition/detectability | Có cho replacement trajectory | Có thể dùng cho real-vs-dummy paths | Có thể dùng cho real/fake event | Chưa có |
| Route/travel-time service loss | Có | Phải tách app utility và dummy plausibility | Có | Chưa có |
| Requests/bytes/latency/energy | Có | Bắt buộc vì cost xấp xỉ K lần | Bắt buộc | Chưa có; `runtime_s` hiện không hợp lệ |

### 5.2. Bộ metric chung được đề xuất

**Track A -- replacement/pseudolocation**

- EIE và xác suất attacker nằm trong bán kính `r`;
- mechanism-aware tracking/reconstruction error;
- POI Top-k preservation trên POI thật;
- route/travel-time loss;
- displacement P50/P95, on-road, speed/reachability;
- generation latency P50/P95 và bytes/request.

**Track B -- real + `K-1` dummy hoặc dummy trajectories**

- top-1 attacker success, true-member/path rank và MRR;
- posterior entropy và `k_eff`;
- dummy/path survival sau map, reachability, temporal, semantic và population filters;
- semantic/trajectory realism;
- request/response multiplier, bytes, latency và client filtering cost;
- paper-native entropy/semantic metrics chỉ để reproduce paper, không thay common attacker metrics.

**Track C -- fake-query protocol**

- các metric Track B;
- fake-query detection rate;
- ASR dưới path-correlation/reachability attack;
- số fake events, traffic multiplier, bytes, latency, energy và service load.

Nguyên tắc so sánh: giữ native metrics để kiểm tra reproduction, nhưng kết luận chính phải dùng cùng attacker và cùng common metrics trong từng track. Recognition rate, ASR, entropy và EIE không phải các đại lượng có thể đổi trực tiếp cho nhau.

---

## 6. Backtest v2 cần thay đổi tối thiểu gì?

Không sửa ép `experiments/run_benchmark.py`; giữ nó để regression cho Internship 2. Tạo pipeline mới với các contracts sau.

### 6.1. Protection interface

```python
protect_trajectory(record, context, rng) -> ProtectionTranscript
```

`PointwiseAdapter` sẽ wrap các cơ chế cũ dùng `reset/perturb`, nên không phải viết lại REM/T-REM/SM-REM.

### 6.2. Transcript mà attacker được nhìn thấy

```text
ProtectionTranscript
  events[]:
    event_id, timestamp, event_kind,
    candidates_or_tracks,
    query_type, query_content_class,
    generation_ns, request_count, serialized_bytes
```

Truth object tách riêng:

```text
EvaluationTruth
  real_event_mask, true_candidate_index,
  real_trajectory, destination, future_route, group_relation
```

`true_candidate_index` tuyệt đối không truyền cho attacker.

### 6.3. Ba track bắt buộc tách

- **Track A:** replacement/pseudolocation -- REM family, TransProtect, AnotherMe.
- **Track B:** real + `K-1` location/trajectory set -- Liu semantic 2026, LSPPM-SI khi phù hợp.
- **Track C:** fake-query protocol -- Liu fake-query wrapper trên một Track B mechanism.

### 6.4. Attacker và metric registry

- attacker trả estimate, scores, posterior và ranks thay vì chỉ một scalar error;
- thêm staged contextual filters: road -> reachability/time -> semantic/query -> population prior;
- metrics được dispatch theo contract, ghi rõ unit và denominator;
- persist transcript đã public, truth object riêng, posterior/rank, filter survival, isolated generation time và communication metadata.

### 6.5. Tests và provenance

Thêm test cho:

- `K=1` backward compatibility;
- variable-length trajectory và fake events;
- truth không rò vào attacker input;
- deterministic seed;
- train/test split không rò user;
- unit conversion `km^-1 <-> m^-1`;
- model checkpoint/source commit, OSM/POI snapshot và simulator config được ghi trong provenance.

---

## 7. Thứ tự thực hiện đề xuất

1. **Xác nhận với giáo viên:** chốt final output contract và target chính trước khi viết ba SOTA.
2. **SUMO pilot:** chọn một urban bounding box, pin OSM snapshot/seed, tạo một minimal instance cho từng S1--S7 và export canonical schema.
3. **Custom scenario/query layer:** cài seven operators; dùng OMoSim chỉ khi cần population/multi-day diary.
4. **Backtest v2:** transcript, truth separation, adapters, registry, persistence và contract tests.
5. **Comparator đầu tiên:** port TransProtect + VehiTrack trên vehicle-specific graph; reproduction Rome/SF riêng trước khi adaptation Bắc Kinh.
6. **Comparator thứ hai:** AnotherMe trajectory adapter với local routing và user-level split.
7. **Set/protocol track:** Liu semantic 2026; sau đó thêm fake-query wrapper. Chỉ cài LSPPM-SI nếu cần offline/semantic trajectory-set comparison.
8. **Final evaluation:** controlled simulator benchmark + real-data validation; multi-seed confidence intervals; common metrics theo track.

### Tiêu chí để coi simulator pilot là đủ

- sinh tái lập được ít nhất một positive và một control case cho mỗi S1--S7;
- có ground truth đầy đủ theo canonical schema;
- route/speed/stop hợp lệ trên mạng đường;
- query, POI, destination và group labels không bị suy ra ngược từ output bảo vệ;
- cùng trace chạy được qua cơ chế của ta và ít nhất một SOTA adapter;
- artifact lưu seed, map/config hash và scenario parameters.

Nếu các tiêu chí này đạt, **không xây trajectory simulator riêng**.

---

## 8. Provenance của lần kiểm tra này

Các public source snapshot được clone chỉ để audit vào thư mục tạm, không vendor vào repository:

- AnotherMe: `0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f`;
- VehiTrack/TransProtect: `035684c6c666a9af7cbd9984d92300000eb65536`;
- OMoSim: `a08ba056436c1d174f7fecc13bf1b1e3099dee52`.

Các kết luận về SOTA không chỉ dựa vào metric được kể lại trong paper; chúng đã được đối chiếu với output contract, data loader, graph, metrics, attacks và persisted result schema trong code hiện tại.
