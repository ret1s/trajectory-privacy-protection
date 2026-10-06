# Utility POI cuối hành trình: giữ Geo-I, tái sử dụng metadata tĩnh

Chuẩn bị ngày 06/10/2026. Các điểm Q, GPS SUMO, Geo-I, belief, ngân sách và attacker trước đó được giữ nguyên. Ta cải thiện **khâu trả kết quả tại thiết bị** từ các POI đã nhận; không chọn lại sample hoặc làm đẹp attack score.

## Vấn đề được xác định

GeoI-Epoch8-H12 với L20 trước đó có Recall trung bình 94,96%, nhưng chuyến yếu nhất ở đoạn 400–600s chỉ đạt 70,30%. [Diagnosis](../../artifacts/benchmarks/native_static_cache_20261006_v1/weakest_tail_diagnostic.json) cho thấy chuyến native-21, slot 6 đã đứng yên tại đích, vẫn đọc GPS ở 600s, và mới dùng 21/23 đơn vị. **Đây không phải lỗi hết ngân sách.** Q gần GPS nhất vẫn cách trung bình khoảng 1,24km trong đoạn cuối; con số này chỉ giải thích khó khăn utility, không dùng làm privacy score.

Nếu chỉ lấy POI của event hiện tại, ta bỏ lại nhiều thông tin địa điểm đã tìm được trước đó. Dùng mọi trả lời *đã nhận* trong chính chuyến yếu này cho Recall 93,33%; hai POI tham chiếu vẫn chưa từng được nhận đến 600s. Đây là upper bound chẩn đoán cho chuyến đó, không phải một cấu hình được chọn từ test.

## Hai bước thử, có lưu riêng bằng chứng

Trước tiên, [protocol TTL](../../artifacts/benchmarks/native_static_cache_20261006_v1/protocol.json) so sánh current-only, cache epoch60 đang có và rolling 60/120/180 giây. Rolling60 giữ các trả lời có tuổi **nhỏ hơn** 60s, thường gồm event hiện tại và hai event 20s trước đó. Epoch60 xóa cache khi đi qua mốc 60s, nên hai cơ chế khác nhau.

Chọn TTL nhỏ nhất trên **sáu nhóm selection** có Recall trung bình ≥90%, nhóm thấp nhất ở phần cuối ≥90%, và không thêm request/byte. Rolling60 được chọn. Test cải thiện vừa phải; TTL120/180 cũng không xử lý hết chuyến yếu. Kết quả và tất cả candidate được giữ nguyên trong [artifact TTL](../../artifacts/benchmarks/native_static_cache_20261006_v1/results.json).

Sau diagnosis đó, ta khai báo **protocol mới** cho [cache theo phiên bản catalogue](../../artifacts/benchmarks/native_versioned_static_20261006_v1/protocol.json). `id/category/lat/lon` là metadata tĩnh của một phiên bản công khai, nên giữ chúng trong **cùng epoch tám chuyến [0,12000)** thay vì tự hết hạn sau 60s. Các phiên bắt đầu tại 0,1500,…,10500s. Cache nhận POI theo thứ tự thời gian; chuyến sau dùng được metadata đã nhận ở chuyến trước, không dùng trả lời tương lai. Cache tối đa 418 ID và bị vô hiệu khi phiên bản/epoch đổi hoặc hết hạn.

Candidate mới được chấp nhận trên selection với Recall trung bình 99,59%, nhóm thấp nhất ở phần cuối 99,43%, giới hạn 418 ID và đúng bằng request/byte cũ. Đây là **development sau khi đã xem test trước đó**, không phải xác nhận trên test mới.

## Kết quả test của GeoI-Epoch8-H12, cùng L20 và Q

| Cách dùng trả lời POI | Recall trung bình | Recall phần cuối | Nhóm thấp nhất, phần cuối | Chuyến thấp nhất, phần cuối |
|---|---:|---:|---:|---:|
| Chỉ event hiện tại | 94,96% | 94,39% | 84,09% | 70,30% |
| Epoch60 đang có | 95,24% | 94,74% | 85,00% | 72,12% |
| Rolling60 đã chọn | 95,64% | 95,22% | 85,45% | 72,42% |
| **Static theo version/epoch tám chuyến** | **99,34%** | **99,48%** | **98,33%** | **86,67%** |

Mỗi policy dùng đúng 7.550 request Q và 41.233.562 byte trả lời JSON trong test: **không tăng request, byte, hoặc lượt đọc GPS** so với L20 hiện có. Byte gồm metadata `id/category/lat/lon` đầy đủ của các trả lời, kể cả POI trùng; chỉ là ước lượng phần trả lời, chưa đo HTTP/request/latency.

Phần tăng lớn ở hàng cuối đến từ **reuse qua các phiên đã quan sát**, không phải TTL60. Trong cohort tĩnh có tuyến lặp lại này, chuyến đầu khi chưa có lịch sử đạt Recall trung bình khoảng **94,73%**; bảy chuyến sau khoảng **99,99%**. Chuyến test cuối yếu nhất chuyển sang chuyến đầu native-24, đạt 86,67%; chưa có bảo đảm mọi chuyến đều trên 90%. Hai đích truy vấn đều đã xuất hiện trong lịch sử nên các chuyến truy vấn được hưởng cache đã tích lũy; không suy ra kết quả tương tự cho vùng/đích chưa từng quan sát.

Cache test giữ 167–416 ID, trung bình khoảng 343 ID, dưới catalogue 418 ID. Metadata duy nhất được serialize thành JSON tối đa khoảng **35,9KB**, chưa gồm overhead container Python. Xem [validation độc lập](../../artifacts/benchmarks/native_versioned_static_20261006_v1/validation.json) để kiểm tra từng giá trị và dung lượng selection/test.

Recall vẫn **có điều kiện trên POI tham chiếu tồn tại**. Coverage không đổi: 1.402/1.510 event test có tham chiếu; phần cuối 462/528. Native-21 chỉ có hai trong tám chuyến có tham chiếu ở phần cuối; sáu chuyến còn lại là N/A. Cache không tạo ra đường đi hoặc POI ở nơi không có tham chiếu.

## Dùng trong client: bật rõ ràng chế độ static

`BudgetedGeoILbsClient` mặc định **vẫn giữ cache epoch60**. Không sửa wrapper hoặc dữ liệu đã pin. [VersionedStaticGeoILbsClient](../../benchmark/versioned_static_geoi_lbs.py) là adapter **tùy chọn** mới, bọc đúng client có sẵn:

```python
static_client = VersionedStaticGeoILbsClient(
    budgeted_client,
    catalogue_version=public_catalogue_version,
    public_epoch_id=public_epoch_id,
    start_s=public_start_s,
    end_s=public_end_s,
)
static_client.start_session(public_token, public_departure)
static_client.public_tick(public_time, lazy_gps_supplier, server)
# Chỉ mục tiêu dùng metadata/chi phí đường tĩnh:
static_client.answer_static(private_query, public_time, local_lat, local_lon)
# Mục tiêu cần trạng thái khả dụng hiện tại:
static_client.answer_live(private_query, public_time, local_lat, local_lon)
```

`public_tick` gọi đúng một lần pipeline Geo-I/retrieval có sẵn và cache chỉ những ID trả lời tại Q. `answer_static` đọc snapshot tại thiết bị, dùng GPS/mục đích riêng tư để xếp hạng; không đổi cache clock, không gọi supplier/server, không ảnh hưởng Q tiếp theo. Cache giữ trong bộ nhớ qua các phiên cùng subject/version/epoch; không tuyên bố lưu metadata bền vững qua restart. Khi phiên bản công khai hoặc epoch thay đổi, adapter/cache cũ hết hiệu lực; tạo cache cho scope mới, đồng thời vẫn giữ ledger ngân sách riêng tư của client theo chính sách đã khai báo.

Một adapter/cache gắn với **một subject đã được ứng dụng liên kết và ledger tương ứng**; không trộn cache của các subject khác nhau. Thí nghiệm tạo cache riêng cho từng nhóm và kiểm tra toàn bộ timestamp tuyệt đối, tránh dùng nhầm đồng hồ tương đối của chuyến mới.

`answer_live` kiểm tra **cả khoảng epoch đã khai báo của adapter và epoch trạng thái hiện tại**. Ví dụ, adapter hết hiệu lực ở 40s phải từ chối trả lời tại 50s dù phản hồi server còn thuộc epoch 0–60s. Kiểm tra này chỉ đọc trạng thái, không đổi cache clock hoặc các request sau đó. Metadata từ 20 phút trước không cho phép nói POI còn mở/khả dụng. `VersionedStaticPoiCache.dynamic_candidates` cũng yêu cầu known/available mask nhận trong epoch hiện tại; thiếu hoặc cũ trả unknown. POI không có trong cover hiện tại là **chưa biết**, không tự coi là unavailable. Thí nghiệm native này chỉ đo dữ liệu POI/đường tĩnh; chưa đo utility liveavailability hoặc traffic.

Phiên bản catalogue và epoch là **đầu vào công khai được khai báo**, không suy ra từ GPS hoặc mục đích truy vấn. Adapter không tự thêm request để refresh. Điều kiện sử dụng cache lâu là provider/cấu hình công khai xác nhận metadata ổn định cho phiên bản đó; không dùng chính sách này cho trạng thái động.

## Kiểm chứng

```bash
PY=/private/tmp/trajectory-research-20261005-venv/bin/python
$PY -m pytest -q tests/test_static_poi_cache.py tests/test_versioned_static_poi_cache.py tests/test_versioned_static_geoi_lbs.py
$PY -m experiments.verify_native_static_cache_20261006
$PY -m experiments.verify_native_versioned_static_20261006
```

14 tests kiểm tra hết hạn, đổi version/epoch, không cache trạng thái động, không coi thiếu phản hồi là unavailable, xếp hạng nhiều mục đích tại thiết bị, Geo-I/GPS supplier thật sau khi hết cap, và request/order/byte giữ nguyên dù số lần/thời gian trả lời tại thiết bị khác nhau. Regression về epoch hẹp kiểm tra cả trước mốc bắt đầu và tại/sau mốc kết thúc cho static/live, đồng thời chứng minh request tiếp theo giữ nguyên. Verifier TTL tính lại 30.220 dòng policy và diagnosis; verifier versioned tính lại 6.044 event/method theo đồng hồ tuyệt đối qua đủ tám phiên. Cả hai tính lại membership cache từ **Q và lịch sử trả lời công khai**, sau đó mới dùng GPS riêng tư để chấm Recall. Các source/hash, cost, coverage và selection đều được kiểm tra.

Các verifier chạy lại toàn bộ assertions và giữ validation cũ. Dùng `--validation-output /private/tmp/NEW-check.json` nếu cần thêm record mới; không ghi đè artifact đã có. [Validation recheck TTL](../../artifacts/benchmarks/native_static_cache_20261006_v1/validation_recheck.json) thêm kiểm tra diagnosis và giữ nguyên validation đầu.
