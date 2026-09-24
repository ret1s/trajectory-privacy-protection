# Kết hợp truy vấn cố định và truy vấn thích nghi

Vòng 23 kiểm tra một giả thuyết từ control vòng 22: khi belief dự đoán sai vùng
POI cần phục vụ, việc cho toàn bộ K truy vấn chạy theo belief có thể bỏ mất
coverage hữu ích. Giữ một phần coverage công khai có thể giảm sự phụ thuộc đó.
Đây là giả thuyết utility cần đo, không phải một bảo đảm mới.

## Thiết kế chốt trước phép chấm

Tổng K=5: hai tọa độ cố định chọn một lần từ prior công khai; ba tọa độ còn lại
di chuyển theo belief đã bảo vệ. Cùng L10, mục tiêu top-5, privacy cap .23 và
lịch đọc GPS 60 giây. Không học/chọn hai tọa độ từ nhãn các chuyến đánh giá.

Gọi B là tập cố định, A là tập truy vấn thích nghi, và F là objective hợp POI
có trọng số theo belief. Tối ưu phần bổ sung:

`F_B(A) = F(B ∪ A) − F(B)`.

Do B đã cố định, phép tính tương đương đặt trọng số bằng 0 cho các POI B đã trả
về. Ba truy vấn thích nghi dùng greedy/exchange và tiến về mục tiêu trên đường
có hướng như trước. Hai ablation là slack 0 và .03, không quét số truy vấn cố
định sau khi nhìn kết quả. Greedy cho objective phụ này là áp dụng công cụ tối
ưu đã biết, không phải định lý mới. Floor slack vẫn là objective gần đúng,
không phải Recall thật.

## Giới hạn lập luận privacy

Hai vị trí cố định là công khai; đối thủ có thể bỏ chúng và chỉ phân tích ba
track thích nghi. Với B cố định, quan sát `(B,A)` có lượng thông tin về GPS giống
như quan sát A khi biết B. Không được nhận số lượng năm candidate thành một
bảo đảm k-anonymity hoặc gọi các điểm cố định là “lớp mã hóa”.

Tất cả chọn/lập kế hoạch sau neo dùng tài nguyên công khai và lịch sử đã bảo vệ;
không có private read bổ sung. Theo lập luận hậu xử lý, cận lý tưởng có điều
kiện trên đồng hồ vẫn là .23 cho phiên và .46 cho cặp hai phiên. Không suy
privacy thực nghiệm tốt hơn từ cận không đổi. Cũng không mở rộng cận tọa độ sang
thời gian/identity. Các giới hạn số thực của kernel gốc vẫn giữ nguyên.

## Phép kiểm tra

Zero-backbone phải tái lập output, neo và ledger của bản cha; đã qua test cho
cả slack 0/.03. Kiểm tra thêm prefix causality, track đứng yên, mọi bước chuyển
khả thi và cận mất objective. Screen dùng đủ 15 ca core, 132 lượt mới và 198
control replay. Bộ tấn công hình học gồm cả bản bỏ hai track cố định; attacker
học phù hợp vẫn phải được huấn luyện lại trước khi quảng bá một ứng viên.

## Kết quả screen

Đã chạy đủ 132 lượt mới và replay đúng 198 control. Chỉ có hai nhóm validation
phát triển; chưa phải kiểm tra độc lập. Tổng K và các neo/ledger đều khớp.

| Case | Paced K5 | Cố định 2 + thích nghi 3 | Paced K5, slack .03 | Cố định 2 + thích nghi 3, slack .03 |
|---|---:|---:|---:|---:|
| S1.A | 94,17% | 86,67% | 94,17% | 86,67% |
| S1.B | 91,67% | 85,00% | 89,17% | 89,17% |
| S1.C | 74,17% | **90,00%** | 82,50% | **90,00%** |
| S2.A | 98,33% | 97,50% | 97,92% | 97,50% |
| S2.B | 100,00% | 99,91% | 99,26% | 97,78% |
| S2.C | 93,33% | 100,00% | 91,57% | 93,43% |
| S3.A | 91,17% | 87,75% | 94,10% | 91,33% |
| S3.B | 91,17% | 87,75% | 94,10% | 91,33% |
| S3.C | 91,88% | 88,12% | 93,75% | 92,50% |
| S9.A | 91,43% | 86,76% | 92,50% | 90,15% |
| S9.B | 93,03% | 92,30% | 94,25% | 93,18% |
| S9.C | 91,50% | 86,84% | 91,90% | 90,11% |
| S10.A | 96,01% | 96,14% | 97,94% | 92,59% |
| S10.B | 99,28% | 98,53% | 98,95% | 99,42% |
| S10.C | 93,32% | 88,93% | 93,60% | 91,64% |
| Số ca đạt 90% | 14/15 | 7/15 | 13/15 | 13/15 |

Recall toàn phiên trên 16 nguồn validation: paced 92,15% → 90,79%; paced/slack
93,12% → 91,96%. Hai tọa độ công khai lấy từ states 12875 và 74589, giống nhau
cho mọi phiên. Điểm yếu S1.C được cải thiện, nhưng loss ở S1.A/B và các ca khác
khiến chưa chọn bản mới. Không lấy riêng 90% S1.C làm bằng chứng giải quyết năm
scenario. Screen dừng ở core, chưa chạy mở rộng hoặc attacker học cho hai bản.

Kết quả gợi ý hạn chế của việc dành cứng số truy vấn cho coverage công khai.
Bước tiếp theo cần kiểm tra cách phân bổ coverage theo mức dịch vụ ở các vùng
khả dĩ, với tổng K giữ nguyên; chưa có bằng chứng rằng một cách chia khác tốt hơn.
