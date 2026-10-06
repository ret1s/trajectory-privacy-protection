# Kiểm tra thứ tự Q và attacker nối điểm theo hình học

**Geo-I và toàn bộ điểm Q giữ nguyên.** Thay đổi đang được kiểm tra chỉ là
xáo trộn riêng thứ tự năm tọa độ trong mỗi bản tin và bỏ ID Q cố định. Thiết
bị vẫn giữ thứ tự nội bộ để cập nhật chuyển động. Máy chủ nhận cùng tập Q,
cùng thời gian; kết quả POI, Recall và byte của phép đo dịch vụ không đổi.
Đây là hậu xử lý công khai, không tiêu thêm ngân sách Geo-I.

Đây là **diagnostic phát triển sau khi đã xem test**, dùng lại toàn bộ 28
nhóm 1205–1232 của [vòng trước](2026-10-05_endpoint_generalization.md), không
phải một holdout độc lập mới. Không chạy lại defense, sửa GPS, chọn seed đẹp,
đổi cấu hình Endpoint20 hoặc bỏ nhóm. Chỉ fit attacker trên 701–708 và chọn
decoder trên 709–710; khóa trước khi runner này đọc lại nhãn test. Mỗi
phương pháp/view có bank riêng và decoder riêng theo MAE/Hit.

## Attacker được bổ sung

Ngoài bank cũ dùng tập tọa độ, bank mới đọc **từng vị trí trong danh sách Q**:
điểm đầu/cuối của mỗi slot, ngoại suy OLS từ 2/3/6 mốc, và **Extra Trees,
kNN1/kNN5** học từ chuỗi của cả năm track. Attacker thấy toàn bộ bản tin và
giờ đóng phiên, nhưng không nhận GPS thật, Z, seed hay nhãn điểm thật/giả.

Xáo thứ tự không làm các điểm biến mất. Hai bank hình học sắp điểm ban đầu
theo tọa độ, rồi dùng **Hungarian assignment** để nối điểm qua từng mốc:
một bank chọn tổng khoảng cách nhỏ nhất; một bank dùng dự đoán vận tốc trước
khi nối. Chúng cũng có OLS và các learner theo track. Hai bank này cho cùng
đặc trưng/ước lượng ở cả hai view, kể cả khi bản tin chứa tọa độ trùng nhau.
Vì vậy không gọi bản tin xáo thứ tự là “không thể theo dõi”.

Khóa xáo trộn của thí nghiệm tách khỏi RNG Geo-I, HMAC theo session/rep/event;
mỗi bản tin dùng substream riêng. Chỉ cùng session/rep/event giữa phương
pháp đối sánh được ghép RNG. Khóa tái lập của dữ liệu tổng hợp nằm riêng trong
evidence, không phải seed công khai cần dùng khi triển khai.

## Kết quả và vấn đề tổng quát hóa của attacker

MAE lớn là attacker sai xa hơn, nhưng chỉ diễn giải trước **decoder đã khóa**.
Mỗi ô dưới đây là MAE m / Hit100. Chấm tới GPS thật của chính chuyến, gộp đều
28 nhóm; 112 session/rep không được coi là 112 người độc lập.

| Phương pháp / bank | View | S9 MAE / Hit100 | S10 MAE / Hit100 |
|---|---|---:|---:|
| GPS thật, mọi bank | Cả hai | 0 / 100% | 0 / 100% |
| GeoI-Slack L10, bank cũ | Cả hai | 621 / 3,57% | 769 / 3,57% |
| GeoI-Slack L20, bank cũ | Cả hai | 891 / 1,79% | 901 / 0% |
| Endpoint20, bank cũ | Cả hai | 1.483 / 0% | 1.337 / 0% |
| GeoI-Slack L20, bank mở rộng | Thứ tự ban đầu | 1.376 / 0,89% | 1.677 / 0% |
| GeoI-Slack L20, bank mở rộng | Xáo từng bản tin | 1.376 / 0% | 1.691 / 0% |
| Endpoint20, bank mở rộng | Thứ tự ban đầu | 2.200 / 0% | 1.337 / 0% |
| Endpoint20, bank mở rộng | Xáo từng bản tin | 2.200 / 0% | 1.337 / 0% |

**Endpoint20 không có lợi ích MAE đo được từ xáo thứ tự** ở bank mở rộng:
S9/S10 chênh lệch bằng 0. Với L20 thường, MAE S10 tăng 14 m, khoảng bootstrap
ghép theo nhóm 95% [-19; 48], chưa cho thấy lợi ích. Các decoder Hit được chọn
riêng có thể khác; giữ tất cả Hit50/100/200/500 và CI trong evidence.

Bank lớn hơn không bảo đảm decoder chọn trên hai nhóm sẽ tốt hơn trên test.
Ví dụ S10 của GeoI-Slack L20:

| View / decoder đã chọn | MAE selection m | MAE test đã xem m |
|---|---:|---:|
| Bank cũ: `median_ols6_120s` | 672 | 901 |
| Mở rộng, có thứ tự: `observed_slots_tree` | 470 | 1.677 |
| Mở rộng, xáo thứ tự: `geometry_velocity_tree` | 559 | 1.691 |

Hai learner mới thắng trên selection nhưng sai hơn bank cũ ở test: +776 m
[505; 1.059] và +789 m [516; 1.071]. Đây là lỗi tổng quát hóa của lựa chọn
attacker, không phải Q/GPS của defense đã thay đổi. `readout.json` giữ lỗi
từng nhóm của mọi decoder được chọn. Không thay attacker bằng một decoder
thắng trên test, không lấy minimum theo test để sửa số.

Do đó, so cùng L20 ở S10, bank cũ vẫn cho Endpoint20 MAE cao hơn 436 m
[313; 563], nhưng bank mở rộng đã chọn lại cho chênh **-340 m** [-624; -54]
ở view có thứ tự và **-354 m** [-636; -66] ở view xáo. Giữ cả hai kết quả;
không thể từ nghiên cứu này nói Endpoint20 vượt trội bền vững trước mọi
attacker. Cũng không lấy việc bank mở rộng sai hơn bank cũ để tuyên bố privacy
được cải thiện: attacker tối ưu có thêm thông tin vẫn có thể dùng bank cũ.

## Evidence và phạm vi

[Artifact và lệnh tái kiểm tra](../../artifacts/benchmarks/endpoint_order_20261005/README.md)
giữ protocol, mọi error của bank, selection đã khóa, model, source snapshot,
CI theo nhóm và kiểm tra độc lập. Đã tính lại **478.464 ước lượng**, **320
chỉ số**, kiểm tra **10.464 view bản tin** và **177.408 đẳng thức** của các
channel bất biến/hình học. Bảy test kiểm tra xáo trộn, giữ tọa độ trùng,
nối track qua giao cắt, bất biến theo permutation và loại dữ liệu riêng tư.

Bản đồ vẫn là bản đồ công khai tái dựng của cùng bộ SUMO; quy tắc làn/rẽ gốc
chưa được khôi phục chính xác. Diagnostic đã xem test không tạo xác nhận
mới. Xáo Q đóng một kênh nhãn slot thừa; nó không che thời gian, account, IP,
định danh phương tiện/người hay đảm bảo không liên kết được track hình học.
