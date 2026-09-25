> **Phạm vi hiện tại:** S10 chỉ gồm A/C; xem [kết quả và lập luận đã cập nhật](active_scope_results.md). Nội dung dưới lưu kết quả/chẩn đoán của phiên bản trước; không dùng số tổng hợp A/B/C cũ thay cho phạm vi hiện tại.

# Truy vấn công khai theo loại POI: cải tiến cho S1/S2/S3/S9/S10

> Kiểm tra mở rộng 25/09: [S9/S10 trên 32 nhóm tuyến mới](endpoint_calendar_results.md), giữ nguyên kế hoạch/attacker đã khóa và thêm lịch gửi công khai trước/sau chuyến. Recall 100% của tập thật + dummy cần đọc cùng privacy và byte; các số lịch sử dưới được giữ riêng.

> Cập nhật 25/09: bảng so trực tiếp các bản triển khai DLS, RDG, TransProtect, Semantic, Fake-query và AnotherMe nằm tại [live_paper_comparison_results.md](live_paper_comparison_results.md). Nội dung dưới giữ giao thức và kết quả lịch sử; không trộn chi phí/seed của hai bộ đánh giá.

Ngày 24/09/2026, vòng 29–30. Đã hiện thực một **nhánh dịch vụ mới**, bên cạnh
Geo-I + planner đang có. Nhánh này chọn trước các truy vấn riêng cho từng loại
POI bằng bản đồ công khai; GPS chỉ dùng để xếp hạng kết quả trên thiết bị.
Nó không phải một cấu hình epsilon mới hay bản Geo-I thắng mọi tiêu chí.

**Kết quả chính:** ở availability 80%, bản 30 truy vấn loại–tọa độ đạt
**95,67% Recall, 15/15 ca** trên tập phát triển; S1.C tăng từ **80,18% lên
95,57%**. Sau khi khóa plan, kiểm tra thêm bốn nhóm tuyến mới đạt **94,61%,
15/15 ca**. Trong cùng bộ kiểm tra mới, availability 95% chỉ đạt **11/15 ca**;
bản phủ rộng 67 truy vấn đạt 100% trên tất cả các ca và mức availability đã thử,
với chi phí truyền lớn hơn. Giữ nguyên mọi kết quả, không chọn riêng mức thuận lợi.

## Cơ chế và đóng góp cụ thể

Trước đây, năm tọa độ cùng được dùng để hỏi sáu loại POI: 30 truy vấn
loại–tọa độ, nhưng chỉ năm vị trí khác nhau. Một tập vị trí thuận lợi cho cafe
không nhất thiết phủ tốt nhà hàng hoặc bệnh viện. Nhánh mới bỏ ràng buộc mọi
loại phải dùng chung năm tọa độ và phân bổ 30 truy vấn theo phần danh mục chưa
được phủ của từng loại.

1. Từ bản đồ có hướng và danh mục POI công khai, dựng top-10 tĩnh tại các tọa
   độ ứng viên. Tọa độ phải được snap qua đúng quy tắc server trước khi chấm.
2. Chọn tham lam truy vấn tăng nhiều nhất tỷ lệ POI chưa được phủ trong một loại.
   Không đọc tuyến của người dùng, nhãn scenario hay trạng thái availability.
3. Gửi cùng kế hoạch cho mọi hành trình trong vùng công khai đã xác định.
4. Hợp các POI khả dụng trong phản hồi; dùng GPS thật để chọn top-5 cục bộ.
5. Nếu server đảm bảo phản hồi có hiệu lực đến hết epoch, có thể chỉ tải lại
   ở sự kiện đầu epoch. Quy tắc này phụ thuộc đồng hồ công khai, không phụ thuộc
   GPS, cache hit của người dùng hoặc điểm đến. Đối chứng fixed K5/K12 được
   hưởng cùng tối ưu này.

Plan 30 phân bổ cafe/clinic/fuel/hospital/pharmacy/restaurant lần lượt
**11/1/1/1/2/14 truy vấn**, tại **19 tọa độ khác nhau**. Plan rộng phân bổ
19/1/1/1/2/43, tổng **67 truy vấn tại 52 tọa độ**. Các vị trí cố định có chuyển
động bằng không; đây là tập truy vấn công khai, không phải năm track giả di động.
Không gọi hai giao diện này là cùng K=5 hoặc cùng byte.

Điểm tiến bộ của implementation là phân bổ ngân sách truy vấn theo loại dịch
vụ, kết hợp hiệu lực phản hồi với một ranh giới dữ liệu không cần GPS cho network
policy. Tham lam/set cover, truy vấn giả và cache không được nhận là phát minh
mới. [Atmaca et al., OJVT 2024](https://wrap.warwick.ac.uk/id/eprint/183198/1/WRAP-privacy-preserving-querying-mechanism-high-utility-electric-vehicles-2024.pdf)
đã kết hợp AGeoI và dummy cho dịch vụ trạm sạc có trạng thái cập nhật; thiết kế
hiện tại khác ở việc plan công khai không đọc GPS và phân bổ truy vấn theo loại.
Khác biệt này chưa chứng minh novelty hoặc ưu thế so với paper đó. Chưa có bảng
so mới đầy đủ sáu phương pháp từ paper.

## Privacy: thêm được lập luận nào?

Gọi X là hành trình, C là vùng/danh mục/đồng hồ/plan công khai, W là trạng thái
server đã biết, T là các yêu cầu và phản hồi trên mạng. Với nhánh công khai:

**T = f(C, W), không có X làm đầu vào.**

Vì vậy, với C và W cố định, thay toàn bộ GPS, điểm dừng, origin hoặc destination
không thay transcript. Phần tọa độ không bổ sung thông tin về X so với các dữ
liệu phụ trợ đã điều kiện hóa; có thể diễn đạt bằng I(X; T | C, W) = 0.
Đây là lập luận về luồng thông tin, không phải suy ra từ một attacker yếu.

- **S1:** truy vấn không thay khi vị trí hiện tại thay.
- **S2:** truy vấn không bám theo vị trí hoặc thời gian dừng riêng tư; lịch
  yêu cầu đã được điều kiện hóa vẫn có thể tiết lộ việc dừng nếu ứng dụng tạo
  lịch dựa vào chuyển động.
- **S3:** chuỗi tọa độ truy vấn không tái hiện đường đi thật.
- **S9/S10:** tọa độ đầu/cuối truy vấn không thích nghi theo origin/destination.
  Liên kết nhiều phiên cùng plan không tạo thêm kênh tọa độ phụ thuộc GPS.

**Không diễn giải thành Hit=0 hoặc bảo vệ mọi metadata.** Đối thủ vẫn có thể
đoán từ prior, vùng dịch vụ, thời gian bắt đầu/kết thúc, account, IP hoặc các
lượt nhấp. Việc ứng dụng tự chọn vùng/loại/plan theo GPS cũng tạo kênh mới và
nằm ngoài mệnh đề. Client hiện hỏi các loại theo một plan công khai cố định.
Chưa chạy một bộ attacker mới khai thác đầy đủ timing/identity trên nhánh này;
không tái sử dụng MAE/Hit của bản Geo-I để gán cho nhánh công khai.

## Giới hạn của chứng nhận độ phủ

Nếu một POI nằm trong top-L tĩnh tại ít nhất một truy vấn, việc loại những POI
không khả dụng không thể đẩy POI đó ra khỏi top-L khi chính nó vẫn khả dụng.
Do đó, **nếu** hợp top-L tĩnh phủ mọi POI công khai có thể đạt tới, hợp các phản
hồi sống chứa mọi POI khả dụng cần thiết; xếp hạng cục bộ trả đúng top-5.
Điều kiện gồm cùng khoảng cách đường, quy tắc tie và tính hợp lệ của phản hồi.

Điều kiện phủ toàn danh mục **chưa đúng trên mạng này**. Plan 30 còn 160 POI
chưa được phủ tĩnh; plan 67 còn 8 POI. Ứng viên truy vấn nằm trong SCC lớn nhất,
còn tập đích bao gồm POI có thể đạt tới từ toàn bộ catalogue đường. Không bỏ
tám POI khó để làm chứng nhận thành công. **100% của plan 67 là kết quả thực
nghiệm trên các ca đã chấm, không phải bảo đảm 100% cho mọi nơi và mọi trạng thái.**

## Kết quả tại availability 80%

Hai cột đầu dùng cùng 12 nhóm phát triển, 102 chuyến nguồn và 173 record.
Cột kiểm tra mới dùng bốn nhóm được mô phỏng sau khi khóa plan: 88/88 chuyến
hoàn tất; năm scenario dùng 32 chuyến nguồn, 56 record. Vẫn là cùng thành phố
và cùng generator, không phải dữ liệu thực hoặc xác nhận ở thành phố mới.
Không dùng các con số này để so bản Geo-I trên tập cũ với nhánh mới trên tập mới.

| Ca | Geo-I + planner + cache, tập cũ | Plan 30, tập cũ | Plan 30, nhóm mới | Nhóm mới có tham chiếu / có record |
|---|---:|---:|---:|---:|
| S1.A | 97.64% | 96.48% | 95.11% | 4/4 |
| S1.B | 95.90% | 96.93% | 98.22% | 4/4 |
| S1.C | 80.18% | 95.57% | 96.11% | 4/4 |
| S2.A | 99.86% | 94.32% | 92.22% | 3/4 |
| S2.B | 99.49% | 94.71% | 91.29% | 3/4 |
| S2.C | 94.59% | 94.36% | 91.74% | 3/4 |
| S3.A | 95.63% | 96.20% | 96.21% | 4/4 |
| S3.B | 95.97% | 96.85% | 95.42% | 4/4 |
| S3.C | 95.26% | 96.21% | 96.33% | 4/4 |
| S9.A | 95.69% | 95.80% | 95.39% | 4/4 |
| S9.B | 95.30% | 95.52% | 95.11% | 2/2 |
| S9.C | 94.95% | 95.83% | 95.55% | 4/4 |
| S10.A | 97.63% | 95.25% | 93.77% | 4/4 |
| S10.B | 99.46% | 95.15% | 91.92% | 2/2 |
| S10.C | 96.42% | 95.81% | 94.69% | 4/4 |

S2 của family-904 không có POI tham chiếu truy cập được, nên giữ Recall null
và đếm riêng; không gán 1 và không xóa record. Quy tắc giống mọi đối chứng.
S9.B/S10.B chỉ có hai nhóm đủ đặc tả, nên bằng chứng còn ít.

Trên tập cũ, chênh lệch Recall trung bình so với Geo-I + planner chỉ
**+0,07 điểm phần trăm**, khoảng bootstrap ghép nhóm 95% **[-2,94; +2,70]**.
Không kết luận trung bình tốt hơn ổn định. Riêng S1.C tăng **15,40 điểm phần
trăm**, khoảng **[+9,19; +21,29]**. Đây là bootstrap thăm dò 10.000 lần trên
12 nhóm, chưa hiệu chỉnh đa kiểm định. S2 và một số ca S10 giảm utility dù vẫn
qua ngưỡng danh định; cải thiện không đồng đều giữa các mục tiêu.

## Độ nhạy và đối chứng trên nhóm mới

| Phương án | Availability 50% | Availability 80% | Availability 95% |
|---|---:|---:|---:|
| Plan 30 | 97.62%; 15/15 | 94.61%; 15/15 | 91.52%; **11/15** |
| Plan rộng 67 | 100%; 15/15 | 100%; 15/15 | 100%; 15/15 |
| Fixed K5 | 87.89%; 1/15 | 83.18%; 0/15 | 81.68%; 0/15 |
| Fixed K12 | 97.01%; 15/15 | 94.30%; 15/15 | 93.74%; 15/15 |
| Bulk trạng thái | 100%; 15/15 | 100%; 15/15 | 100%; 15/15 |

Mỗi ô là Recall trung bình theo ca và số ca ≥90%, gộp ba world seed mới.
Ở 95%, plan 30 thiếu S2.A/B/C và S10.B. Không chọn lại plan theo bốn nhóm này.
Plan 67 là một cấu hình đã khóa từ trước, không phải được tăng ngân sách sau
khi thấy các ca thất bại.

## Chi phí thật của thay đổi giao diện

Số dưới là thân JSON mô phỏng ở 80% trên nhóm mới; chưa gồm HTTP/TLS, RTT,
chi phí server hoặc benchmark điện thoại. Dùng đúng schema của từng phương án,
không che việc tọa độ dùng chung có thể được serialize gọn hơn.

| Phương án | Truy vấn loại–tọa độ / lần tải | Tọa độ khác nhau / lần tải | Request + response byte / lần tải¹ | Byte / sự kiện nếu chỉ refresh mỗi epoch |
|---|---:|---:|---:|---:|
| Plan 30 | 30 | 19 | 3,256.2 | 941.4 |
| Plan rộng 67 | 67 | 52 | 7,358.4 | 2,127.3 |
| Fixed K5 | 30 | 5 | 994.0 | 287.2 |
| Fixed K12 | 72 | 12 | 2,344.6 | 677.4 |
| Bulk trạng thái | 0 tọa độ | 0 | 90.4 | 26.1 |

¹ Trung bình khi gọi tại mọi sự kiện; số chữ số epoch làm byte header thay
đổi nhẹ theo lịch. Cột cuối đếm chính xác các lần refresh và toàn bộ sự kiện,
không nhân một tỷ lệ xấp xỉ. Có trung bình khoảng 0,289 lần refresh/sự kiện.
Artifact cũng có chi phí dưới schema canonical chung để kiểm tra ảnh hưởng
của encoding. Cần phân biệt số tọa độ, số truy vấn loại và số kết nối.

Plan 30 cải thiện utility so với fixed K5 với cùng số truy vấn loại nhưng
nhiều byte hơn. Plan 67 không phải lựa chọn rẻ hơn fixed K12 chỉ vì 67 < 72:
payload và số tọa độ khác nhau cao hơn. Nếu API bulk khả dụng thì bulk vẫn
là đối chứng tốt hơn trong workload này. Nhánh công khai chỉ có lý do sử dụng
khi API cung cấp truy vấn theo điểm và cho phép hỏi riêng từng loại.

## Kết luận phát triển

- Giữ Geo-I + planner khi bài toán thực sự yêu cầu tập năm tọa độ di động
  hoặc ngân sách truyền tương ứng. Chưa âm thầm thay giao diện benchmark đó.
- Giữ plan 30 như nhánh công khai tiết kiệm số truy vấn loại; nó xử lý được
  điểm yếu S1.C ở hai tập, nhưng chưa qua mọi ca trong stress availability 95%.
- Giữ plan 67 làm cấu hình ưu tiên độ phủ: đạt 100% trên các thử nghiệm đã
  khóa với chi phí cao hơn, chưa có chứng nhận toàn bản đồ.
- Cả hai bổ sung lập luận không đưa GPS vào network policy, áp dụng cho mục
  tiêu tọa độ của năm scenario. Điều kiện về timing/identity/vùng vẫn phải
  được nêu, và chưa thể kết luận thắng sáu paper hay sẵn sàng triển khai thực.

## Mã, bằng chứng và tái lập

- [Planner](../../evaluation/category_cover.py), [client online](../../benchmark/category_client.py).
- [Protocol vòng 29](../../artifacts/benchmarks/research_loop/iteration29_protocol.json),
  [plan khóa](../../artifacts/benchmarks/research_loop/iteration29_plans.json),
  [kết quả tập cũ](../../artifacts/benchmarks/research_loop/iteration29_category_cover.json).
- [Protocol nhóm mới](../../artifacts/benchmarks/research_loop/iteration30_protocol.json),
  [dataset mới](../../artifacts/datasets/research_loop_confirmation_v1/summary.json),
  [kết quả](../../artifacts/benchmarks/research_loop/iteration30_category_confirmation.json),
  [readout/bootstrap/chi phí](../../artifacts/benchmarks/research_loop/iteration30_readout.json).
- [Kiểm tra độc lập](../../artifacts/benchmarks/research_loop/iteration30_verification.json):
  dựng lại hai plan, 8.532 đối chiếu category bằng Dijkstra xuôi, 315 dòng
  tổng hợp, 14 kiểm tra transcript không đổi với cùng đồng hồ, 88/88 chuyến đến nơi.

```sh
python -m experiments.research_loop_category_cover prepare
python -m experiments.research_loop_category_cover evaluate
python -m experiments.build_research_loop_confirmation
python -m experiments.research_loop_category_confirmation
python -m experiments.research_loop_category_readout
python -m experiments.verify_research_loop_category
python -m pytest tests/test_category_cover.py tests/test_category_client.py tests/test_live_poi.py tests/test_public_cover.py
```

Các bước tạo dataset/plan/kết quả từ chối ghi đè bằng chứng đã hoàn tất. Các
bước readout và verifier có thể chạy lại. Môi trường dùng Python 3.11 và cache
OSM/SUMO cùng nguồn các vòng trước; hash đầu vào được lưu trong artifact.
