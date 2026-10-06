# Audit đánh giá thực nghiệm để chuẩn bị bài JISA

Ngày kiểm tra: 06/10/2026. Giữ backbone Geo-I và toàn bộ dữ liệu, protocol, Q,
attacker checkpoint và kết quả đã lưu. Tài liệu này đề xuất thực nghiệm tiếp
theo; **không tạo số mới và không xác nhận khả năng được nhận bài**.

Kết luận: hệ thống đã có các module hoạt động, nguồn được pin và nhiều phép
kiểm tra độc lập. Khoảng trống lớn nhất hiện nay là **xác nhận ngoài các cohort
đã dùng để phát triển**, công bằng về ngân sách/dịch vụ, và attacker dùng đầy
đủ thông tin công khai. Nâng Recall của cache trên test đã xem không thay thế
được các bước này.

## 1. Bằng chứng hiện có dùng được đến đâu?

| Bằng chứng | Điểm đã xác lập | Giới hạn cần giữ trong bài |
|---|---|---|
| [Native S5/S6](2026-10-05_native_future.md) | Cùng mạng SUMO mới cho GPS, bảo vệ, POI và cạnh đích; decoder dựa vào hình học dự đoán được ID chưa có trong train; Raw đang rẽ đạt 100% | 24 family, chỉ sáu family test và 12 query; biết đúng hai cạnh/đích ứng viên; S5/S6 cùng phụ thuộc một quyết định rẽ. Không phải hai bài toán open-world độc lập |
| [Cap tám chuyến](../../artifacts/benchmarks/future_native_20261005_v1/protocol.json) | Cap prospectively reserved và ledger tồn tại qua restart | Reset có cap toàn epoch 1,84 m⁻¹, Epoch8-H12 có 0,23 m⁻¹. Không dùng so sánh này làm bằng chứng superiority ở cùng cap |
| [Static cache theo version](2026-10-06_native_static_utility.md) | Recall 99,34% từ những POI đã nhận, không đổi Q/GPS/request; cache causal, có giới hạn version/epoch | Chuyến đầu khoảng 94,73%, bảy chuyến sau khoảng 99,99%; hai đích query đã xuất hiện trong lịch sử. Reset dùng cùng cache đạt khoảng 99,82%. Lợi ích này là postprocessing dùng chung, chưa phải lợi thế riêng của Geo-I hoặc dịch vụ động |
| [Endpoint selector](2026-10-06_robust_endpoint_selection.md) | Cross-fit theo family giảm lỗi chọn attacker từ hai nhóm; giữ nguyên defense và Q | 28 family đã được xem trước; Hit không cải thiện đồng đều; fold dùng dữ liệu train chồng nhau. Không đổi thành một holdout mới bằng cách seal lại selector |
| [S7 snapshot/sequence](2026-10-05_query_purpose.md) | Request/reply giống nhau khi chỉ thay purpose local; explicit-content control có tín hiệu | Intent được gán cho cùng tuyến/clock, nên chứng minh việc loại một kênh payload. Chưa đo intent tương quan với tuyến, lịch sử, click hay tài khoản |
| [Metrics gốc](2026-10-05_native_metrics.md) | EIE có cùng estimator bằng MAE; status N/A đúng output contract; có chi phí đường được kiểm tra trên cohort mới | Δc/entropy/ASR/DER còn thiếu ở benchmark đối chứng gốc; cost Q-only mới là diagnostic, không thay metric singleton gốc |
| [Đối chứng paper](../../benchmark/README.md) | Contract và phần tái lập/thiếu input đã ghi rõ | TransProtect/AnotherMe/Semantic hiện là adaptations. Không gọi các hàng này là toàn bộ thuật toán chính thức hoặc “state of the art đã bị vượt” |

Native hiện có 66.189 state, 78.439 arc, 43.302 state trên đường rẽ nội bộ và
418 POI. Mạng này được biên dịch từ hình học công khai đã lưu; lane/turn chính
xác **cho mạng mới**, không xác lập fidelity với SUMO Beijing gốc. Tốc độ
8m/s, không congestion/lane change/traffic-light dynamics trong catalogue.
Các số nguồn nằm trong [accounting đã đóng băng](../../artifacts/benchmarks/future_native_20261005_v1/private_accounting.json.gz).

Số lần lặp, event hoặc cặp identity không phải số người độc lập. Hợp các lần
lặp trong family/subject trước, sau đó lấy macro và bootstrap ghép theo đúng
đơn vị độc lập. Mọi nhóm thiếu scenario, sinh thất bại, không tới được POI hoặc
không có nhãn phải có mẫu số và lý do; không chỉ giữ các hàng thành công.

## 2. Hai loại dịch vụ cần đánh giá riêng

**Static:** tọa độ/category theo public catalogue version, chi phí đường tĩnh.
Chấm current-only, cold cache ở chuyến đầu, và causal warm cache qua các phiên.
Áp dụng cùng cache/ranking cho tất cả phương pháp có contract tương thích;
báo kích thước catalogue, ID đã nhận, memory, reset/version invalidation và
đích chưa từng ghé. Cache gần như thu hết catalogue 418 POI phải được nhìn như
một điều kiện thực nghiệm, không mặc định là mô hình catalogue quy mô lớn.

**Dynamic:** availability, giá/queue hoặc travel time ở public epoch hiện tại.
Static ID có thể còn trong cache nhưng không chứng minh trạng thái còn đúng.
Client chỉ dùng status đã nhận ở epoch hợp lệ; không có status là **unknown**,
không suy ra unavailable hoặc available. Nếu chỉ có fixture mô phỏng thay đổi
trạng thái, ghi rõ là controlled dynamic workload, chưa phải dịch vụ thực địa.
Public refresh schedule phải giống nhau giữa defense; không gọi server theo
purpose riêng để sửa một query thiếu POI.

Giữ bốn mục đích đã có, nhưng thêm tốc độ đường khác nhau để nearest và fastest
không trùng bằng cấu trúc. Dynamic changes phải được sinh độc lập defense và
khóa trước scoring. Query radius/detour/category có phân bố workload khai báo;
nếu intent là tổng hợp, không viết như intent của người dùng thật.

**Control static quan trọng:** public context hiện chứa POI ID/category/tọa
độ và signature của catalogue chỉ 418 POI, khoảng 36KB metadata duy nhất.
Vì vậy phải có full-catalogue prefetch/offline local-ranking control: khi toàn
bộ metadata đã công khai và ổn định, client có thể lấy một lần hoặc dùng nguồn
public rồi trả lời local, không cần gửi tọa độ theo từng query. Đây là nhận xét
từ contract, **chưa là benchmark đã chạy**, và không phải PublicCover 30/67 cũ.
Không so 41MB phản hồi lặp với việc tải toàn catalogue một lần mà bỏ control
này. Nếu bài nhắm provider có nội dung/trạng thái riêng, tách public proxy POI
dùng cho planner khỏi catalogue/status thực của provider; chấm mismatch và
current unknown rõ ràng. Static-only gains cần giải thích vì sao server query
thực sự cần thiết cho dịch vụ đã chọn.

Utility cần một bộ trạng thái kiểm tra được, bên cạnh Recall@5:

- Reference tồn tại: Recall/precision, constraint violations, regret khoảng
  cách/thời gian hoặc detour; macro theo subject và purpose.
- Reference rỗng thật: correct-empty, false-positive answer và số event; không
  tự gán Recall=100%.
- Reference có kết quả nhưng client thiếu current status/candidate: unknown,
  partial hoặc retrieval miss; giữ trong service-success denominator.
- Trường hợp không tính được reference vì map/data: not-evaluable cùng lý do,
  coverage riêng; không trộn với empty thật.

Báo trung bình, median, p10, min subject/session, tỷ lệ dưới ngưỡng utility đã
khóa, và pha cuối theo thời gian công khai. Native Recall hiện chỉ xác định ở
1.402/1.510 event test; đoạn cuối 462/528. Không coi conditional Recall là tỷ
lệ phục vụ thành công trên toàn bộ event. Unknown không được làm biến mất khỏi
bảng bằng cách chỉ chấm các câu trả lời có trạng thái.

Utility hiện dùng evaluator GPS đúng tại mỗi event. Lượt supplier trong ledger
chỉ đếm các probe của cơ chế bảo vệ theo public pacing, **không phải tổng lượt
GNSS của điện thoại**. Cần đo riêng acquisition/freshness cho local ranking và
so với dùng last local fix nếu muốn nói về CPU/energy hoặc giảm GPS reads.
Không suy ra tiết kiệm pin từ số lượt chi ngân sách riêng tư.

## 3. So sánh công bằng và ablation cần có

Mỗi đường benchmark cần cùng traces, public map snapshot, public clock, query
workload, POI reference, observation surface và train/selection/test groups.
Pin riêng output contract: singleton replacement, true-plus-dummies hay
dummy-only set. Attacker suy ra GPS/endpoint từ **toàn output được quan sát**;
không chọn Q gần GPS thật nhất để biến set thành singleton.

Chạy hai đường so sánh khác nhau:

1. **Cap/cost matched:** cùng trajectory adjacency/metric, cap toàn subject
   epoch, K/L/clock và cache policy. Ghi epsilon đúng đơn vị m⁻¹; ε km⁻¹ không
   có cùng trị số. Báo cận bảo đảm, actual spent, số GPS read, request và bytes.
   Không cấp một bên .23 mỗi chuyến, bên kia .23 cả lịch sử rồi gọi bằng nhau.
2. **Utility/cost matched:** chọn trên selection các cấu hình đạt cùng utility
   floor và wire-cost ceiling; khóa trước test. Báo cả điểm không đạt và đường
   Pareto. Không chọn cặp “cùng utility” bằng cách dò attack score trên test.

Các paper không có bảo đảm Geo-I tương đương vẫn có thể so empirically với
utility/cost tương đương, nhưng không giả định epsilon danh nghĩa là cùng
privacy guarantee. Metric gốc chỉ chấm khi đủ input; entropy set chứa thật
không chuyển thành ASR của dummy-only set. Native metric và metric mở rộng
phải có tên/status/mẫu số riêng như API hiện có.

| Ablation giữ Geo-I | Thay đúng một yếu tố | Câu hỏi cần trả lời |
|---|---|---|
| PlanarPaced ↔ REM paced | Anchor kernel và **emission belief tương ứng**; giữ planner/cap/K/L/cache | REM có lợi ích ở cùng cap/cost hay chỉ khác độ nhiễu? Đây là primitive ablation, không full CCS/PETS reproduction |
| Private reuse ↔ always-refresh | Phân bổ public cap trước, hạch toán tất cả phép đọc/test | Reuse tiết kiệm cap/utility đến đâu; không dùng epsilon mỗi phép đo khác nhau mà bỏ qua composition |
| Belief/planner đầy đủ ↔ public prior/no temporal update | Cùng Z, Q clock và cap | Temporal prediction có giá trị và lỗi ở fork/cold start ra sao? |
| Utility-aware chọn Q ↔ public reachable sampling | Cùng protected anchors và K/L | Gains đến từ retrieval coverage hay primitive? |
| Current-only ↔ epoch60 ↔ versioned static | Cùng Q/replies/GPS, tất cả methods | Tách lợi ích generic cache khỏi protection; cold/novel-area bắt buộc |
| Full public catalogue/offline ranking | Chỉ public static metadata; không dùng GPS server | Làm rõ overhead của retrieval khi toàn catalogue đã public; dynamic/private provider giữ setting riêng |
| Per-trip allocation ↔ persistent epoch | **Cùng tổng cap**, allocation khai báo | Persistence/accounting khác chính sách cấp cap; control bằng nhau có thể trùng output và phải báo trùng |
| Fixed common payload ↔ explicit intent | Cùng Q/replies/service | S7 giảm kênh nội dung nào, chi phí fetch-all là bao nhiêu? |

Không yêu cầu mọi ablation thắng. Loại một component mà metric không đổi cũng
là kết quả giúp đơn giản hóa phương pháp. Chính sách H/slots/epoch không được
chọn bằng thời lượng, vị trí hoặc nhánh riêng tư của chuyến test.

### Extension planar có thể chạy ngay, nhưng chỉ là development

[PlanarPacedLaneDummy](../../benchmark/engines/planar_paced.py) và
[PlanarAnchorModel](../../benchmark/planar_anchor.py) đã có. Runner mới có thể
gọi `native_resources(data, new_work)` của
[future_sumo_eval](../../experiments/future_sumo_eval.py), bọc belief `.00125`
bằng `PlanarAnchorModel`, và dùng allocation hiện có của Epoch8-H12:
`slots=8, H=12, C=.23, u=.00125, Bsession=.03, K=5, L=20, theta=200,
slack=.03, read_interval=60`.

Phải giữ continuous planar emission/private-reuse atom; không đưa REM emission
vào planar, không snap bằng GPS riêng tư. Postprocessing mạng đường chỉ nhận
protected anchor/public inputs. Map/net/catalogue và POI source giống native
cũ; nguồn cho mọi input mới có hash. Dùng `fit_and_score(rows,use_history)`
riêng mỗi method/task/stage trên cùng bank/rule; không đổi globals/OUT của
runner cũ. Chấm static cold/warm cùng policy cho REM và planar, lưu source,
accounting, Q, mọi attacker prediction và lỗi. Không dùng kết quả extension
này làm confirmation trên native test đã xem.

## 4. Attacker phải quan sát đúng giao thức và có positive control

Đánh giá một bank hữu hạn cho biết sức kháng trước bank đó, không xác nhận
attacker tối ưu. Fit riêng từng mechanism để tránh emission/feature mismatch;
chọn trên nhiều subject/family selection, báo độ ổn định giữa fold. Lưu fitted
model, training array hash, selection objective và mọi prediction.

- **S1–S3, S9–S10:** mechanism-aware posterior/tracking trên public map, noisy
  reuse và cap/clock; cộng set statistics, kNN/ExtraTrees, geometric matching
  hoặc Hungarian track recovery, sequence/extrapolation. Nếu public order/ID
  bị xáo, attacker vẫn được suy track từ hình học. Chỉ gọi Bayes/HMM optimal
  trong mô hình giả định thực sự được giải tối ưu, không với REM-emission
  proxy cho mọi cơ chế.
- **S4:** same/different linkage của subject/source ID qua các phiên, tách
  person và vehicle khi thật sự có truth. Với subject mới, dùng gallery/query
  linkage hoặc pair task; classifier closed-set không thể dự đoán ID chưa có
  trong train. Thêm class-prior, permutation và account-known control; Geo-I
  tọa độ không che account/IP mà attacker đã được biết.
- **S5:** dự đoán cạnh tương lai từ causal prefix bằng public reachable-edge
  geometry/catalogue, không chỉ hai cạnh được chỉ trước. Báo label/candidate
  coverage, abstention và error; raw ở fork chung chance là ambiguity, không
  privacy gain. Có public clock sau turn để raw thực sự có signal.
- **S6:** history các ngày trước + current prefix; ngày 8 có cả public query
  ngày 7 khi giao thức cho phép. Thêm joint decoder biết ràng buộc cặp nếu
  workload đã công bố một routine/một rare. Đánh giá prior-only, query-only,
  history+query, routine/rare và novel destination. Không coi lựa chọn rare
  ngẫu nhiên riêng tư là thứ history bắt buộc dự đoán được.
- **S7:** ngoài đổi purpose trên cùng transcript, tạo workload intent tương
  quan activity/route/time được khai báo. So prior-only, location/history-only
  và full wire, explicit-payload positive control. Distinguish việc không gửi
  query content khỏi việc không thể suy intent từ mobility; clicks thuộc
  observation surface khác nếu chưa che.

Liên kết tám chuyến dùng cùng public association có sẵn, toàn bộ replies,
timestamp, order, request/reply sizes và các query prefix đã tới thời điểm
attack. Feature API không nhận evaluator ID, raw GPS, private seed, future
route hoặc destination truth. Raw/explicit positive controls phải có tín hiệu
khi bài toán có thể suy ra; kiểm tra permutation train và input-contract
leakage. Thêm history không làm attacker tối ưu yếu đi; nếu bank hữu hạn yếu
đi thì báo là selection/generalization issue, không protection gain.

## 5. Real GPS và cross-city: nguồn khả thi, chưa phải dữ liệu đã chạy

`data/raw/` không có trong workspace lúc audit. [Data README](../../data/README.md)
nói active experiments dùng SUMO, GeoLife chỉ legacy và Porto chưa tích hợp.
Không lấy khảo sát cũ “đã tải” làm bằng chứng dữ liệu hiện có hoặc đã rerun.

| Nguồn chính thức kiểm tra 06/10 | Vai trò phù hợp | Gate trước sử dụng |
|---|---|---|
| [Porto — UCI dataset 339](https://archive.ics.uci.edu/dataset/339/taxi%2B) | Real GPS/city thứ hai; 442 taxi, 1.710.671 trip, GPS 15s; nguồn UCI công bố CC BY 4.0 | Download/hash/attribution; giữ MISSING_DATA và lỗi DAYTYPE nguồn đã cảnh báo; TAXI_ID là **source linkage label**, chưa đủ xác nhận person/vehicle độc lập hoặc driver đổi xe |
| [GeoLife — Microsoft download](https://www.microsoft.com/en-us/download/details.aspx?id=52367) | Real subject GPS, lịch sử nhiều ngày; 182 users, 17.621 trajectories, sampling đa dạng | Pin archive/version; [guide chính thức 1.2](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/User20Guide-1.2.pdf) giới hạn non-commercial và không cho phân phối data/derivatives. Kiểm tra license kèm archive 1.3 trước khi phát hành artifact; ưu tiên acquisition recipe/hash, không vendor raw/derived traces |
| [T-Drive — Microsoft](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) | Backup vehicle/source-ID linkage và long-window spatial attacks; một tuần, 10.357 taxi | [Guide chính thức](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/User_guide_T-drive.pdf) có cùng hạn chế non-commercial/redistribution; sampling trung bình 177s, không tự nội suy thành GPS thật 20s để chấm exact next-edge |

Các guide GeoLife 1.2 và download hiện chỉ dẫn 1.3 có thống kê/version khác;
manifest phải chỉ rõ phiên bản thực tải. Không suy nhãn physical vehicle từ
transportation-mode labels. Porto ORIGIN_CALL chỉ có cho một loại call, không
được mặc nhiên coi như nhãn danh tính thật hoặc intent POI của mọi trip.

Đường/POI public dùng regional snapshot riêng mỗi city, pin bbox/projection,
mode profile, direction/turn policy, source time và attribution.
[OpenStreetMap công bố ODbL và yêu cầu ghi nguồn](https://www.openstreetmap.org/copyright).
Map mới khác thời điểm GPS cũ cần ghi temporal mismatch; dữ liệu đường tĩnh
không phải traffic/availability lịch sử. Không trộn WGS84 với tọa độ dịch vụ
khác hoặc chọn map support dựa vào vị trí riêng của holdout.

[Loader GeoLife legacy](../../data/geolife.py) lấy file theo thứ tự, dừng ở N
trajectory, cap số trip/user, truncate số điểm và lọc bbox trước gap splitting.
Đó là convenience loader, không đủ cho cohort confirmatory: có thể thiên về
user sớm, bỏ long tail hoặc nối các điểm qua đoạn ra ngoài bbox. Importer mới
cần chronological session boundaries, UTC, duplicate/nonmonotonic fixes,
native gaps và mode/map-matching confidence; không giữ một private route liên
tục bằng cách xóa các fix không đạt. Preflight ghi counts/rates loại mẫu theo
city/subject và từng rule **trước** bất kỳ defense score nào.

## 6. Protocol xác nhận mới: khóa trước khi xem score

Old artifacts giữ nguyên; toàn bộ cohort đã dùng để chọn L/cache/attacker/H
được gắn development. Một seed mới trên cùng map có thể là held-out generator
test, nhưng không đồng nghĩa real-user/cross-city generalization.

1. Chốt claim hẹp, output/service contracts, candidates, hyperparameter bank,
   accounting/cost model, metric/coverage rules và failure gates.
2. Import source nguyên bản read-only. Split theo subject/vehicle/route family,
   không theo point/event/rep. Kiểm tra near-duplicate route/window; trong mỗi
   subject giữ history trước query theo thời gian. Map/POI được build từ nguồn
   public, mobility/intent prior chỉ fit train.
3. Tách development train/selection khỏi **confirmation mới chưa xem**. Có
   test cùng city với subject mới và city khác. Bao gồm long-duration, cold
   start, novel-area, sparse-POI và rare destination theo rule nguồn đã khóa;
   không loại vì defense có score thấp.
4. Chọn toàn cấu hình và attacker bằng selection. Seal source/code/split/
   protocol/model hashes trước scoring confirmation. Public manifest dùng
   opaque keys; evaluator truth không xuất trong attacker view. Hash receipt
   chứng minh nội dung bất biến, không tự chứng minh người chạy chưa xem test.
5. Chấm confirmation một lần, giữ cả failures/N/A/all candidates. Báo paired
   effect và CI theo subject/family, số đơn vị độc lập, sensitivity đã khai
   báo, và multiple-comparison handling cho nhiều primary contrasts.
6. Nếu diagnostic dẫn đến thay model/metric/cache/selector, tạo version mới;
   confirmation vừa xem trở thành development cho version đó. Không seal lại
   cùng cohort rồi gọi fresh holdout.

Số subject test phải chọn bằng desired precision/power trên development,
không bằng số event dễ sinh. Có thể đặt gate lập kế hoạch **ít nhất 30 đơn vị
test độc lập mỗi city** rồi tính power/CI khả thi; 30 là đề xuất triển khai,
không bảo đảm đủ power. Predeclare co-primary contrasts ít và có hướng đọc rõ
(ví dụ cap-matched privacy, cold/dynamic service success); static warm Recall
là kết quả bổ sung. Giữ tất cả loss/tie, không đặt thành công là “thắng mọi
metric của mọi paper”.

## 7. Artifact đầu tiên có thể thực thi ngay

Ưu tiên một runner mới **write-once publication preflight + protocol manifest**,
ví dụ `experiments/jisa_preflight_20261006.py` xuất vào thư mục mới
`artifacts/publication/jisa_preflight_20261006_v1/`. Đây là đề xuất chưa được
implement trong audit; không phải một lệnh hiện đã có.

Runner không fit/chấm defense, không cần tải hết dataset và không sửa artifact
cũ. Nó thực hiện các kiểm tra sau và lưu status `ready/missing/blocked`:

- Inventory nguồn sẵn có: path/hash/version/license receipt; old results đều
  có status development hoặc historical first-pass đúng provenance.
- Baseline output contract, implementation fidelity, missing dependency/weight
  và source code pins; baseline chưa faithful không được tự chuyển thành SOTA.
- Same-map/bbox/projection/drive-vs-walk/internal-via invariants; catalogue/source
  hash; cached reply depth, same static/dynamic cache semantics.
- Budget units/epoch scopes/max composition, K/L/public clocks và actual
  request/reply schema; no private input trong public selection/schedule.
- Split key/history chronology/near-duplicate checks và evaluator-only fields;
  count nguồn đủ cho S4/S5/S6/intent labels, nếu thiếu ghi scenario unavailable.
- Metric status/coverage denominator, candidate/attacker selection rules,
  independent test-access policy, compute calibration trên development only.

Đầu ra tối thiểu: `source_manifest.json`, `protocol.json`, `preflight.json`,
SHA-256 cho mỗi file và receipt không ghi đè. Gate phải **từ chối confirmation
scoring** nếu chưa có raw source/license/split mới/selected model seal; không
điền số synthetic vào phần thiếu. Có thể seal design draft trước acquisition,
nhưng phải seal manifest cuối với source hashes/splits trước score mới.

Tiếp theo chạy planar primitive extension trên native **development** để kiểm
tra API/cost/attacker, sau đó acquisition/preflight real GPS, rồi mới seal và
chạy confirmation. Native generator lưu 473,96s cho 24×8×2=384 protected trip,
chưa gồm build resources. Một method planar trên cached map có thể mất vài
phút; 237s chỉ là phép chia lập kế hoạch, không measured runtime hoặc speed
claim vì hai primitive có chi phí khác nhau. City/map mới, posterior attacker
và large catalogue phải calibrate riêng; tránh chi phí all-pairs graph bằng
reverse shortest-path trên các POI đã khóa và cache source-bound.

Nếu clone thiếu SUMO/map cache gốc, status là missing và dùng native network
đã archive cho primitive development. Không chữa thiếu cache bằng bản đồ mới
rồi gọi là reproduction trên nguồn cũ. Protocol staged này vẫn cho phép tiếp
tục cải thiện Geo-I, đồng thời giữ ranh giới giữa engineering progress và
bằng chứng thực nghiệm mới có thể đưa vào bài.

## 8. Triển khai tiếp sau audit trong cùng phiên

[Kế hoạch publication](../publication/jisa_20261006/README.md) đã có thiết kế
draft và [runner preflight](../../experiments/jisa_publication_preflight.py)
thực tế. Tên runner ở mục 7 là ví dụ đề xuất; dùng entry point đã implement
trong kế hoạch thay vì gọi một tên giả định. Gate confirmation chưa được phép
qua khi chưa có source/split/candidate seal mới.

Một [runner ablation mới](../../experiments/jisa_native_anchor_ablation_20261006.py)
đã khóa protocol trước generation tại
[artifact riêng](../../artifacts/benchmarks/jisa_native_anchor_ablation_20261006_v1/protocol.json).
REM/Planar dùng draw mới, cùng cap/cấu hình nêu ở mục 3; native dataset và
old Q/readouts không đổi. [Verifier riêng](../../experiments/verify_jisa_native_anchor_ablation_20261006.py)
tính lại membership, cost, cap, selection và fitted checkpoints. Readout mới
vẫn là development. Kết quả pilot được ghi trong README của artifact sau khi
hoàn tất và kiểm chứng, không biến thành số confirmation của thiết kế draft.
