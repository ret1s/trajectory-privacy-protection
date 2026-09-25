> **Phạm vi hiện tại:** S10 chỉ gồm A/C; xem [kết quả và lập luận đã cập nhật](active_scope_results.md). Nội dung dưới lưu kết quả/chẩn đoán của phiên bản trước; không dùng số tổng hợp A/B/C cũ thay cho phạm vi hiện tại.

# Truy vấn POI khả dụng: cấu hình và kết quả

> Kiểm tra mở rộng 25/09: [S9/S10 trên 32 nhóm tuyến mới](endpoint_calendar_results.md), giữ nguyên kế hoạch/attacker đã khóa và thêm lịch gửi công khai trước/sau chuyến. Recall 100% của tập thật + dummy cần đọc cùng privacy và byte; các số lịch sử dưới được giữ riêng.

> Cập nhật 25/09: bảng so trực tiếp các bản triển khai DLS, RDG, TransProtect, Semantic, Fake-query và AnotherMe nằm tại [live_paper_comparison_results.md](live_paper_comparison_results.md). Nội dung dưới giữ giao thức và kết quả lịch sử; không trộn chi phí/seed của hai bộ đánh giá.

**Cập nhật vòng 29–30:** đã có [nhánh truy vấn công khai theo loại POI](category_cover_results.md).
Plan 30 đạt 15/15 ca ở availability 80% trên tập phát triển và bốn nhóm mới;
S1.C tập cũ tăng lên 95,57%. Nhánh này dùng 19 tọa độ, không dùng GPS để chọn
query và có chi phí khác bản K5 dưới đây. Stress 95% trên nhóm mới chỉ qua
11/15 ca; plan rộng 67 qua tất cả với chi phí cao hơn. Phần còn lại của tệp
giữ nguyên kết quả vòng 28 để không trộn hai giao diện/phương pháp.

**Cấu hình ưu tiên dịch vụ:** năm truy vấn thích nghi, slack 0,03 và bộ nhớ đệm
phản hồi còn hiệu lực. Recall trung bình theo 15 ca đạt **95,60%**, so với
**82,07%** của truy vấn cố định cùng K=5. Phần tăng riêng từ cache là **0,24 điểm
phần trăm**, không tăng số tọa độ gửi hoặc byte phản hồi.

## Bài toán và thiết lập

Thiết bị có mạng đường cùng danh mục 419 POI; máy chủ giữ trạng thái khả dụng.
Một yêu cầu trả tối đa 10 POI đang khả dụng cho mỗi loại tại từng tọa độ truy vấn.
Thiết bị hợp phản hồi rồi chọn năm POI gần vị trí thật nhất theo đường có hướng.
Vị trí thật dùng để xếp hạng cục bộ, không gửi kèm kết quả chọn.

Trạng thái mô phỏng giữ nguyên trong 60 giây; mức danh định là 80% POI khả dụng.
Kiểm tra thêm mức 50% và 95%, mỗi mức có ba chuỗi trạng thái đã cố định trước khi
chấm. Đồng hồ bắt đầu ở 0 cho mỗi replay; cùng một chuỗi trạng thái dùng cho mọi
phương pháp. Đây không phải dữ liệu khả dụng đo từ doanh nghiệp hoặc một mô phỏng
đồng thời toàn thành phố.

Dữ liệu phát triển gồm 12 nhóm tuyến, 102 chuyến nguồn dùng cho 173 record của
S1/S2/S3/S9/S10, đủ 15 A/B/C. Giữ nguyên tọa độ công bố, lịch truy vấn và cận
ngân sách 0,23 m⁻¹ của phương pháp gốc. Cấu hình slack 0,03 + cache được chọn từ
hai bản paced đã đánh giá, theo Recall trung bình với K=5. Ba chuỗi availability
và các lượt RNG không được tính thành những người dùng độc lập.

## Cải tiến ở thiết bị

Trong cùng một khoảng hiệu lực, thiết bị giữ hợp các phản hồi đã nhận. Khi
sang khoảng mới, cache được xóa. Không dùng phản hồi tương lai, không lấy toàn
bộ trạng thái từ evaluator, không đổi truy vấn theo GPS thật hoặc theo cache hit.

Ví dụ: phản hồi trước cho A, B, C; phản hồi mới cho C, D, E. Nếu vẫn cùng khoảng
hiệu lực, thiết bị có thể chọn trong A–E thay vì chỉ C–E. Một POI gần người dùng
đã nhận ở bước trước vì vậy không bị bỏ quên.

Gọi U_t là hợp phản hồi hiện tại và C_t là hợp phản hồi từ đầu khoảng hiệu lực.
Vì U_t ⊆ C_t và mọi POI trong C_t vẫn còn khả dụng, xếp hạng top-5 từ C_t không
thể có Recall thấp hơn xếp hạng từ U_t. Tính chất này áp dụng riêng cho hợp đồng
trạng thái cố định trong khoảng hiệu lực. Nếu trạng thái đổi tức thời, cần cơ chế
version/expiry tương ứng. Cache không cải thiện privacy: phía máy chủ thấy đúng
các tọa độ và thời điểm cũ.

## So sánh ở mức khả dụng 80%

Recall dưới đây là trung bình đồng đều của 15 ca, mỗi ca tổng hợp theo nhóm tuyến.
Cột byte là thân phản hồi mô phỏng; chưa gồm HTTP/TLS hoặc độ trễ mạng. Mọi phương
pháp đều có sẵn cùng bản đồ và danh mục POI.

| Cấu hình | K | Recall | Ca ≥90% | S1.C | Byte phản hồi / sự kiện |
|---|---:|---:|---:|---:|---:|
| Paced + slack | 5 | 95.36% | 14/15 | 79.64% | 776.6 |
| Paced + slack + cache | 5 | 95.60% | 14/15 | 80.18% | 776.6 |
| Paced + cache | 5 | 94.57% | 14/15 | 76.44% | 776.2 |
| Cố định K5 + cache | 5 | 82.07% | 0/15 | 82.88% | 767.0 |
| Cố định K12 + cache | 12 | 97.61% | 15/15 | 92.56% | 1832.6 |
| Truy vấn vị trí thật | 1 | 100.00% | 15/15 | 100.00% | 169.4 |
| Danh mục tĩnh cục bộ | 0 | 87.62% | 1/15 | 85.73% | 0.0 |
| Cache trạng thái ban đầu | 0 | 77.58% | 1/15 | 71.95% | 1.3 |
| Tải toàn bộ trạng thái mới | 0 | 100.00% | 15/15 | 100.00% | 17.1 |

Cùng K=5/L=10, phương pháp chọn truy vấn thích nghi tăng **13,53 điểm phần trăm**
so với truy vấn cố định; thân phản hồi trung bình cao hơn khoảng **1,26%** do số
POI trả về thực tế khác nhau. So với chính bản slack chưa cache, chi phí truyền
giống hệt và Recall tăng 0,24 điểm phần trăm. K12 tốt hơn về Recall nhưng cần
2,4 lần số tọa độ truy vấn. Riêng S1.C, cố định K5 đạt 82,88%, cao hơn cấu hình
được chọn đạt 80,18%; vì vậy lợi thế trung bình không phải thắng ở mọi ca.

Bootstrap ghép cặp theo 12 nhóm tuyến, 10.000 lần: chênh lệch với fixed K5 có
khoảng 95% **+9,36 đến +17,40 điểm phần trăm**; phần tăng riêng của cache là
**+0,13 đến +0,34 điểm phần trăm**. Đây là các khoảng thăm dò trên dữ liệu phát
triển, chưa hiệu chỉnh nhiều phép so sánh hoặc xác nhận trên địa điểm độc lập.

**Vai trò đối chứng:** fixed K5/K12 không dùng GPS để chọn tọa độ, nên không
được gọi mức tăng utility của ta là cải thiện đồng thời privacy. Nếu API cho tải
toàn bộ trạng thái, chỉ cần 53 byte bitmap cho 419 POI, cộng 8 byte epoch mỗi lần
refresh trong mô phỏng. Đối chứng này đạt Recall 100% mà không gửi tọa độ. Do đó
phạm vi ứng dụng của phương pháp là API truy vấn theo điểm; dữ liệu hiện tại
không chứng minh lợi thế so với API bulk.

## Độ nhạy theo tỷ lệ POI khả dụng

| Khả dụng | Slack, chỉ phản hồi hiện tại | Slack + cache | Cố định K5 | Cố định K12 |
|---|---:|---:|---:|---:|
| 50% | 96.20% | 96.36% | 86.76% | 98.72% |
| 80% | 95.36% | 95.60% | 82.07% | 97.61% |
| 95% | 94.99% | 95.28% | 81.01% | 96.84% |

## Kết quả đủ 15 ca ở mức khả dụng 80%

| Ca | Slack + cache | Slack không cache | Fixed K5 | Số nhóm / record |
|---|---:|---:|---:|---:|
| S1.A | 97.64% | 97.59% | 76.65% | 12 / 12 |
| S1.B | 95.90% | 95.48% | 79.44% | 12 / 12 |
| S1.C | 80.18% | 79.64% | 82.88% | 12 / 12 |
| S10.A | 97.63% | 97.47% | 83.10% | 12 / 12 |
| S10.B | 99.46% | 99.31% | 84.37% | 8 / 8 |
| S10.C | 96.42% | 96.16% | 81.93% | 12 / 12 |
| S2.A | 99.86% | 99.41% | 85.63% | 12 / 12 |
| S2.B | 99.49% | 99.27% | 85.01% | 12 / 12 |
| S2.C | 94.59% | 94.54% | 84.68% | 12 / 12 |
| S3.A | 95.63% | 95.37% | 80.94% | 12 / 12 |
| S3.B | 95.97% | 95.77% | 81.55% | 11 / 11 |
| S3.C | 95.26% | 95.21% | 81.00% | 12 / 12 |
| S9.A | 95.69% | 95.45% | 81.29% | 12 / 12 |
| S9.B | 95.30% | 95.07% | 81.39% | 10 / 10 |
| S9.C | 94.95% | 94.68% | 81.25% | 12 / 12 |

## Phạm vi lập luận và bằng chứng

Có thể lập luận: bộ chọn có nhận biết dịch vụ cải thiện Recall ở ngân sách K5
so với truy vấn cố định; cache tăng thêm utility với cùng transcript, ngân sách
và chi phí truyền. Cache là tối ưu triển khai thông thường, không phải primitive
privacy mới. Các kết quả privacy và counterexample endpoint của phương pháp
gốc vẫn áp dụng; xem [phép tấn công cùng site](repeated_site_density_attack.md).

Bảng này là so sánh dịch vụ với các control và ablation. Nó chưa phải bằng chứng
vượt sáu phương pháp từ paper. Muốn kết luận đó phải chạy các adapter tương ứng
trên cùng workload, đánh giá attacker phù hợp từng phương pháp và ghi rõ phạm vi
tái lập. Kết quả static trước vòng 28 được giữ riêng, không ghép để tính mức tăng.

Đã chấm 24.912 hàng record và 14.688 lượt đánh giá dịch vụ theo cấu hình; đây là
replay, không phải chuyến SUMO mới hoặc lần sinh nhiễu mới. Bốn test mới kiểm tra
dịch vụ/cache; 648 kiểm tra độc lập đối chiếu với Dijkstra xuôi. Nguồn, cấu hình,
tất cả mẫu số và hash nằm trong các tệp sau:

- [Thiết lập](../../artifacts/benchmarks/research_loop/iteration28_protocol.json).
- [Kết quả và chín shard](../../artifacts/benchmarks/research_loop/iteration28_live_service.json).
- [Bảng so sánh và bootstrap](../../artifacts/benchmarks/research_loop/iteration28_readout.json).
- [Kiểm tra](../../artifacts/benchmarks/research_loop/iteration28_verification.json).
- [Mã dịch vụ/cache](../../evaluation/live_poi.py).
