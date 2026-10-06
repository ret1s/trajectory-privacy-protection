# Chọn attacker bằng cross-fit theo family, giữ nguyên Geo-I

Vòng này sửa **cách đánh giá**, không tạo một defense mới. GeoI-Endpoint20,
GPS, năm Q, thứ tự công khai đã xáo, lịch gửi, phản hồi POI, Recall và byte
giữ đúng như [evidence 05/10](2026-10-05_endpoint_order.md). Lý do cần sửa:
bank mở rộng chọn trên hai family đã chọn learner có MAE tốt ở selection
nhưng rất kém ở test. Lấy MAE lớn đó làm thành tích privacy sẽ gây hiểu nhầm.

## Quy tắc đã khóa trước khi runner đọc lại test

Giữ bank gồm thống kê tập Q, ngoại suy, Viterbi, slot/track, Hungarian nối
điểm và **Extra Trees/kNN1/kNN5**. Chỉ dùng view xáo Q riêng tư. Không bỏ
candidate kém, sửa sample hay tìm lại cấu hình Geo-I.

Với tám family 701–708, lần lượt để một family ra ngoài: fit learner **và
mobility history** bằng bảy family còn lại, rồi dự đoán family bị giữ lại.
709–710 được dự đoán bởi bank fit đủ tám family. Như vậy có mười family
validation; mỗi family có cùng trọng số, không coi các session/rep là người
độc lập. Feature cache dùng history rỗng để tránh đưa support của family
đang validation vào learner. Bank cuối chỉ fit 701–708; không fit lại bằng
nhãn 709–710.

Chọn decoder theo **MAE trung bình family + một SE**; riêng Hit chọn **Hit
trung bình family − một SE**. `SE = sample SD / √10`, dùng `ddof=1`. So
objective ở 12 chữ số thập phân rồi hòa thì chọn tên theo thứ tự chữ; đây
là quy tắc số học đã khai báo trước. Không thử nhiều hệ số để chọn số đẹp.

SE ở đây là một **penalty khi chọn**, không phải khoảng tin cậy: các fold
cùng dùng phần lớn dữ liệu train, và số family fit là bảy hoặc tám. Hai
selector cũ, bank bất biến và bank mở rộng chọn từ 709–710, vẫn được báo cáo
làm control. Không đổi sang decoder thắng trên test làm fallback.

## Kết quả development trên 28 family đã xem

Runner khóa rule/model trước khi đọc lại 1205–1232. Nhưng các nhóm này đã
được xem ở nghiên cứu trước, nên kết quả **không phải holdout độc lập mới**.
Mỗi phương pháp có 112 session/rep, gộp đều 28 family. MAE cao và Hit thấp
có lợi cho defense trước đúng decoder đã chọn; Recall không thay đổi.

| Phương pháp, selector cross-fit | S9 Hit100 / MAE m | S10 Hit100 / MAE m | S9 / S10 Hit500 | Recall@5 |
|---|---:|---:|---:|---:|
| GPS thật | 100% / 0 | 100% / 0 | 100% / 100% | 100% |
| GeoI-Slack L10 | 3,57% / 620 | 0% / 673 | 43,75% / 14,29% | 96,81% |
| GeoI-Slack L20 | 1,79% / 790 | 0% / 758 | 24,11% / 12,50% | 98,94% |
| **GeoI-Endpoint20** | **0% / 1.407** | **0% / 1.337** | **7,14% / 4,46%** | **96,64%** |

Ở S10 của L20 thường, rule mới chọn `centroid_ols3_120s`: MAE validation
751 m, SE 79 m; MAE trên nhóm đã xem là 758 m. Selector mở rộng cũ chọn
`geometry_velocity_tree`, MAE test 1.691 m; control bất biến cũ là 901 m.
Rule mới giảm MAE attacker 933 m so bank mở rộng cũ, khoảng bootstrap ghép
family 95% [-1.190; -692], và giảm 144 m so control bất biến [-224; -59].
Đây là attacker ước lượng chính xác hơn trong diagnostic này; Q của defense
không hề thay đổi.

So cùng L20 bằng selector mới, Endpoint20 có MAE S9 cao hơn 617 m [474; 764]
và S10 cao hơn 579 m [468; 696]. Hit100 S10 cùng bằng 0, nên cột đó không
chứng minh có lợi thế. Hit500 S10 giảm 8,04 điểm %, khoảng [-16,96; 0], còn
chưa chắc chắn. Utility vẫn giảm 2,30 điểm % so L20 thường; toàn bộ 1.308
mốc gửi đúng thời điểm, không thêm delay hay ngân sách riêng tư.

## Mức độ ổn định và phần chưa giải quyết

**Vấn đề MAE đã giảm trong cohort này, chưa được xác nhận ngoài cohort.**
Gap validation→test của L20 thường chỉ +7 m, nhưng Endpoint20 vẫn +171 m
(1.166→1.337 m). Objective tốt nhất và thứ nhì cách nhau khoảng 31 m ở
L20 thường, chỉ 13 m ở Endpoint20. Không gọi đây là một lựa chọn hoàn toàn
ổn định hoặc một chứng minh attacker mạnh nhất.

Hit không đồng loạt tốt lên. Ví dụ S10 L10: selector mới có Hit500 14,29%,
thấp hơn control cũ 30,36% (−16,07 điểm %, khoảng [-25,89; -6,25]); Hit100
từ 3,57% thành 0%. Chọn riêng theo một ngưỡng bằng penalty vẫn có thể tổng
quát hóa kém. Vì vậy không chỉ giữ hàng MAE đẹp, không tuyên bố bank mới
mạnh hơn trên mọi metric, và không lấy minimum/max trên test để sửa bảng.

Thêm thông tin không làm attacker tối ưu yếu đi; các biến động này thuộc
bank/selector hữu hạn. Cần một cohort chưa xem và nhiều family hơn để xác
nhận rule, thay vì tiếp tục chọn rule trên 28 nhóm này.

## Kiểm tra và evidence

[Artifact](../../artifacts/benchmarks/endpoint_robust_selection_20261006/README.md)
giữ protocol, mọi candidate/fold prediction, tám fold model và bank cuối
theo từng phương pháp/scenario, control cũ, lỗi từng family, source snapshot
và CI. Đã tính lại 324.672 dự đoán, kiểm tra độc lập 32.040 giá trị số học
selector, 360 training arrays của learner, 5.232 bản tin và 120 metrics.
Mười hai test kiểm tra penalty, trọng số family, hòa, candidate thiếu,
rep trùng, feature/history và verifier.

Verifier đầu so sánh literal dictionary mobility history nên báo lỗi vì
`transition` thêm những cache key có count 0. Giữ nguyên checker và lỗi đó;
checker review chỉ bỏ key zero khi so sánh, xác nhận mọi count dương và prior
đúng với train của fold. Không sửa rule, model, protocol hoặc kết quả sau lỗi.

Bản đồ vẫn tái dựng từ đường công khai, chưa khôi phục đầy đủ làn/rẽ của
SUMO gốc. Dữ liệu tổng hợp, PRNG thí nghiệm và attacker hữu hạn không chứng
minh privacy thực địa hoặc vượt mọi paper. Nền tảng Geo-I và phạm vi bảo đảm
kernel lý tưởng vẫn giữ nguyên.
