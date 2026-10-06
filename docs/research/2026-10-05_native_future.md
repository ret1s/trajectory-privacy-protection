# S5/S6 với chuyến đi SUMO mới: cạnh đường thật và lịch sử sáu chuyến

Ngày 05/10/2026. Đây là thí nghiệm tiếp nối, giữ nguyên Geo-I và giữ nguyên dữ liệu/kết quả trước đó. [Protocol đã chốt](../../artifacts/benchmarks/future_native_20261005_v1/protocol.json), [kết quả chính](../../artifacts/benchmarks/future_native_20261005_v1/results.json), [kiểm chứng](../../artifacts/benchmarks/future_native_20261005_v1/validation.json).

## Đã giải quyết được khoảng trống nào?

S5 trước đây chỉ có phép đo vị trí tiếp theo. Bộ phân loại tên cạnh đường không thể dự đoán những tên cạnh chưa xuất hiện trong tập train. Thí nghiệm mới dùng **hình dạng đường và hướng di chuyển** để chấm điểm hai cạnh ứng viên; tên cạnh chỉ dùng để xuất dự đoán. Cả 12 cạnh ứng viên trong test đều chưa xuất hiện trong train, nhưng vẫn có thể đánh giá bằng ID cạnh thật.

S6 trước đây gần với suy luận điểm cuối từ một đoạn đường ngắn. Thí nghiệm mới có **sáu chuyến lịch sử quan sát công khai**, sau đó dự đoán điểm đến từ prefix của chuyến thứ bảy hoặc thứ tám. Lịch sử có năm chuyến thường lệ và một chuyến ít gặp; hai chuyến truy vấn gồm một chuyến mỗi loại, thứ tự được gán riêng trong dữ liệu đánh giá. Vì vậy, không được lấy tỷ lệ 5/6 của lịch sử để thay cho xác suất mục tiêu trong test: đoán mãi điểm đến thường lệ chỉ đạt 50% trên hai chuyến truy vấn cân bằng.

Đây vẫn là bài toán **biết trước hai lựa chọn có thể xảy ra**. Attacker được biết hình dạng hai lối rẽ và hai vị trí đích từ dữ liệu công khai. Nó không được nhận lối rẽ đã chọn, tuyến đường tương lai, GPS tương lai hoặc nhãn thường lệ. S6 giả định attacker đã liên kết được sáu chuyến lịch sử với cùng một người; thí nghiệm này không chứng minh bảo vệ danh tính.

## Dữ liệu và đồng hồ quan sát

Mạng đường gốc có đầy đủ lane/turn không còn trong môi trường hiện tại. Ta biên dịch hình học đường đã lưu thành **một mạng SUMO mới**, có 4.849 cạnh ngoài và 21.179 cạnh rẽ nội bộ. Mạng dùng để sinh GPS, bảo vệ Geo-I, tìm POI và giải mã cạnh là cùng một mạng. ID/turn là chuẩn xác cho mạng mới này; không tuyên bố khôi phục chính xác mạng Beijing gốc.

Có 24 nhóm đường, mỗi nhóm tám chuyến SUMO thật: 12 nhóm train, sáu nhóm selection và sáu nhóm test, tổng cộng 192 chuyến. Chọn ngã rẽ và hai đồng hồ dựa trên mô phỏng công khai của **cả hai tuyến khả dĩ**, trước khi sinh Q hoặc tính attack score. Không chọn lại đường, đồng hồ hay mô hình bảo vệ sau khi đọc test.

- **Trước ngã rẽ:** hai prefix GPS giống hệt nhau. Attacker không có tín hiệu về lựa chọn riêng tư sắp tới; Raw đạt 50% là sự mơ hồ vốn có.
- **Đang rẽ:** dùng một đồng hồ công khai chung mà cả hai tuyến đều còn trên lane rẽ nội bộ khác nhau. Cạnh ngoài sẽ đi tiếp vẫn nằm ở tương lai, nhưng Raw đã có tín hiệu hướng rẽ. Đây là positive control cho S5/S6.

Bản [cohort v1](../../artifacts/datasets/future_controlled_20261005_v1/manifest.json) được giữ nguyên. Trước khi tính bất kỳ attack score nào, ta nhận ra thời lượng chuyến lịch sử có thể tiết lộ tuyến đã đi. [Cohort v2](../../artifacts/datasets/future_controlled_20261005_v2/manifest.json) chạy lại SUMO với một lần dừng thật tại lane đích đến 650 giây. Mọi chuyến đều có cửa sổ công khai 0–600 giây; không nội suy hoặc nối thêm GPS. Truy vấn hiện tại chỉ cung cấp những sự kiện trước đồng hồ dự báo. Lịch sử cung cấp đủ cửa sổ 600 giây.

## Giữ Geo-I, quản lý ngân sách qua nhiều chuyến

Hai cấu hình dùng cùng thuật toán: REM/noisy test, đọc GPS tối thiểu cách nhau 60 giây, ước lượng công khai trên mạng đường, chọn K=5 điểm Q, L=10 kết quả mỗi loại POI, ngưỡng 200m và slack=0,03. Cơ chế allocator dành ngân sách trước khi đọc GPS và giữ trạng thái qua SQLite; không đổi primitive Geo-I.

| Cấu hình | Số slot chuyến | H | ε mỗi đơn vị, m⁻¹ | B danh nghĩa mỗi chuyến | Cap hiệu dụng mỗi chuyến | Cap hiệu dụng toàn epoch |
|---|---:|---:|---:|---:|---:|---:|
| GeoI-SessionReset | 8 | 12 | 0,01000 | 0,240 | 0,23000 | 1,840 |
| **GeoI-Epoch8-H12** | **8** | **12** | **0,00125** | **0,030** | **0,02875** | **0,230** |

Trong protocol/mã, tên `geoi_epoch8` là **GeoI-Epoch8-H12**. Số 8 chỉ số slot chuyến, không phải H. Cấu hình S4 `GeoI-Epoch6-H8` ở thí nghiệm khác có sáu slot và H=8.

Với H=12, filter cho tối đa 2H−1=23 đơn vị mỗi chuyến. Do đó ε = 0,23/(8×23) = 0,00125 m⁻¹ và B = 2Hε = 0,03 m⁻¹. Tám cap 0,02875 cộng lại thành 0,23. Ngân sách này áp dụng cho transcript tọa độ trong epoch có đồng hồ công khai, dưới giả thiết kernel lý tưởng của Geo-I; không phải bảo đảm danh tính hoặc ngân sách vô hạn qua các epoch.

Hai cấu hình **khác tổng cap**, nên đây là so sánh cách phân bổ ngân sách và chi phí utility. Không suy ra superiority ở ngân sách bằng nhau. Trên sáu nhóm test, tổng chi thực tế của GeoI-Epoch8-H12 nằm trong 0,1900–0,20375 m⁻¹; GeoI-SessionReset nằm trong 1,25–1,49 m⁻¹. Tất cả đều nằm dưới cap đã cấp. Cả hai đọc GPS 88 lần cho tám chuyến của mỗi nhóm; allocator giảm ε mỗi phép đo, không giảm số lượt đọc trong cửa sổ này.

## Kết quả chính, chưa thay đổi L

Attacker gồm **ExtraTrees chấm điểm ứng viên**, hình học đường/hướng di chuyển, uniform; S6 thêm prior từ sáu lịch sử và prior kết hợp truy vấn. Fit dùng train; chọn attacker theo balanced accuracy rồi log-loss trên selection; test chỉ dùng báo cáo. Negative control hoán đổi nhãn train theo nhóm. Các dự đoán, từng nhóm và toàn bộ attacker bank được lưu trong artifact.

Tại đồng hồ đang rẽ, có 12 truy vấn test thuộc sáu nhóm:

| Output | S5 đoán đúng cạnh kế tiếp | S6 Hit100 đích | S6 MAE đích | Static Recall@5 |
|---|---:|---:|---:|---:|
| Raw | 100% | 100% | 5,0m | 100% |
| GeoI-SessionReset | 41,7% | 41,7% | 660,6m | 97,72% |
| **GeoI-Epoch8-H12** | **50,0%** | **50,0%** | **543,1m** | **89,95%** |

Accuracy/Hit100 thấp nghĩa là attacker khó suy ra mục tiêu hơn. MAE cao nghĩa là vị trí đích suy ra sai xa hơn. Đối với GeoI-Epoch8-H12, selection chọn **uniform**; chỉ có thể nói attacker đã chọn không tìm được lợi thế ổn định. Attacker tốt nhất nếu xem lại toàn bank trên test đạt 58,3%; đó là thống kê mô tả, không phải attacker đã chọn trước test. GeoI-SessionReset đã gần mức đoán ngẫu nhiên, nên kết quả mới **chưa chứng minh cap toàn epoch cải thiện attack score** so với reset.

Ở đồng hồ trước ngã rẽ, Raw đạt đúng 50% cho cả S5 và S6. Không dùng chênh lệch ở đây làm bằng chứng đóng góp bảo vệ. Raw ở đồng hồ đang rẽ đạt 100% và learned-control hoán nhãn train chỉ đạt khoảng 58,3%: bài toán có tín hiệu thật và đã thay được gate thất bại của bộ phân loại ID cũ. Các con số bảo vệ vẫn là kết quả của một bank hữu hạn trên sáu nhóm test.

Baseline S6 chỉ dùng lịch sử của Raw đoán đúng **100% chuyến thường lệ nhưng 0% chuyến ít gặp**, nên trung bình vẫn là 50%. Nó tìm được thói quen đã quan sát nhưng không dự đoán được một quyết định ít gặp riêng tư. Khi đang rẽ, tín hiệu của truy vấn hiện tại mới giúp Raw đạt 100% cho cả hai loại.

## Utility: đọc cả phần cuối và coverage

[Readout theo pha công khai](../../artifacts/benchmarks/future_native_20261005_v1/public_phase_readout.json) chia trước các cửa sổ 0–180, 200–380 và 400–600 giây. Không chọn pha theo vị trí, thời điểm hết ngân sách hoặc score.

| Static Recall@5 | GeoI-SessionReset | GeoI-Epoch8-H12 |
|---|---:|---:|
| Trung bình sáu nhóm, 0–600s | 97,72% | 89,95% |
| Nhóm thấp nhất, 0–600s | 94,64% | 84,95% |
| Trung bình phần cuối 400–600s | 99,66% | 90,29% |
| Nhóm thấp nhất, phần cuối | 99,32% | 81,36% |
| Chuyến thấp nhất, phần cuối | 95,76% | 66,36% |

Recall là tỷ lệ top-5 POI tham chiếu mà hợp các trả lời tại Q tìm lại được, tính macro theo các loại có POI tham chiếu. Mạng có những vị trí không có POI nào đi đến được: Recall **không xác định**, không đổi thành 0 hay 1. Trong test, 1.402/1.510 cửa sổ có tham chiếu (92,85%); phần cuối là 462/528 (87,5%). Sáu chuyến của nhóm native-21 không có tham chiếu ở toàn phần cuối. Các giá trị Recall trong bảng chỉ áp dụng cho phần có tham chiếu và giả định tất cả POI đều khả dụng; chưa đo availability hoặc mục đích truy vấn động.

## Extension utility: nhận thêm POI từ cùng những Q đã chốt

Sau khi đã đọc test chính, ta khai báo một [protocol riêng L10/20/40](../../artifacts/benchmarks/future_native_depth_20261005_v1/protocol.json). Chọn **L nhỏ nhất** đạt cả ba điều kiện trên sáu nhóm selection: Recall macro theo nhóm ≥90%, median Recall mỗi chuyến ≥90%, và byte trả lời trung bình ≤2 lần L10. Test chỉ dùng đọc kết quả sau lựa chọn. Đây là **development extension**, chưa phải xác nhận trên test mới.

GeoI-SessionReset đã đạt điều kiện nên giữ L10. GeoI-Epoch8-H12 chọn **L20**: trên selection, Recall=94,16%, median mỗi chuyến=97,31%, byte trả lời=1,5578×L10. L40 bị loại vì byte=2,5263×L10 dù utility cao hơn.

| GeoI-Epoch8-H12, cùng Q | Primary L10 | Extension L20 |
|---|---:|---:|
| Test Recall macro theo nhóm | 89,95% | **94,96%** |
| Nhóm test thấp nhất, 0–600s | 84,95% | **91,75%** |
| Median Recall mỗi chuyến test | 92,92% | **98,33%** |
| Trung bình phần cuối 400–600s | 90,29% | **94,39%** |
| Nhóm thấp nhất, phần cuối | 81,36% | **84,09%** |
| Chuyến thấp nhất, phần cuối | 66,36% | **70,30%** |
| Byte trả lời ước lượng mỗi event | 17,5KB | **27,3KB** |

L20 lấy nhiều POI hơn tại **cùng K=5 tọa độ Q**, sau đó vẫn lọc top-5 tại thiết bị. Không sinh lại Q, không đổi ε, reference5, belief/planner đã tính với L10, hoặc attacker đã chọn. Toàn bộ thứ tự và ID của prefix L10 trong catalogue L40 khớp catalogue L10 gốc; mọi utility L10 tính lại khớp kết quả primary. [Kết quả extension](../../artifacts/benchmarks/future_native_depth_20261005_v1/results.json) giữ hash của public Q và attack score chính; [verifier](../../artifacts/benchmarks/future_native_depth_20261005_v1/validation.json) tính lại 27.198 dòng depth/utility/byte từ bảng POI công khai đã lưu.

Byte là kích thước JSON UTF-8 của **phần trả lời POI** gồm `id/category/lat/lon`, cộng các trả lời cho từng Q, kể cả POI trùng. Không gồm request, header HTTP, hoặc độ trễ mạng. Coverage không thay đổi: thêm L không tạo ra POI tham chiếu ở nơi không có đường đến POI. Extension cải thiện utility nhưng phần cuối của một số chuyến vẫn yếu; chưa tuyên bố cải thiện privacy score, vượt GeoI-SessionReset, hoặc hoàn thiện availability.

## Giới hạn cần giữ khi trình bày

S6 dùng sáu lịch sử cộng prefix của **một** chuyến truy vấn. Khi đánh giá chuyến thứ tám, attacker chưa nhận chuyến thứ bảy; cũng chưa dùng ràng buộc chung biết rằng cặp truy vấn có một thường lệ/một ít gặp. Vì vậy, đây không phải đánh giá đầy đủ attacker đã liên kết toàn bộ tám chuyến. Hai đích ứng viên đã biết làm S5/S6 dùng chung quyết định nhánh; không thay cho dự báo đường/đích trong thế giới mở. Nhãn người/lựa chọn tuyến là dữ liệu mô phỏng phục vụ đánh giá, không phải danh tính người thật.

## Chạy lại kiểm chứng

```bash
PY=/private/tmp/trajectory-research-20261005-venv/bin/python
$PY -m pytest -q tests/test_candidate_future_attack.py
$PY -m experiments.verify_native_future
$PY -m experiments.verify_native_future_depth
# Ghi thêm bằng chứng kiểm chứng mới, không đè validation đã có:
$PY -m experiments.verify_native_future --validation-output /private/tmp/native_future_recheck.json
```

Verifier kiểm tra nguồn/hash, mạng có cùng lane rẽ, 192 GPS trace SUMO, đồng hồ cố định, nhãn cạnh thật nằm sau cutoff, ID test chưa nằm trong train, giới hạn feature công khai, lựa chọn attacker trên selection, tính lại metrics, cap từng chuyến/toàn epoch và utility coverage. [Validation recheck](../../artifacts/benchmarks/future_native_20261005_v1/validation_recheck.json) thêm kiểm tra pha/coverage và giữ nguyên bản validation đầu. Lần chạy mặc định giữ nguyên `validation.json` nếu đã có; vẫn chạy toàn bộ assertions. Tám unit tests kiểm tra hình học rẽ, hoán thứ tự ứng viên, loại bỏ ID/nhãn riêng tư, prior cân bằng và không lấy tọa độ tương lai.

Public transcript đã chốt có thể kiểm chứng lại hoàn toàn. Khóa RNG và SQLite chứa trạng thái bảo mật nằm ngoài repo ở thư mục riêng, không xuất vào artifact. Tái sinh Q từ đầu cần trạng thái bí mật đó hoặc cho ra một lần lấy mẫu mới; không coi public seed/session ID là khóa riêng tư.
