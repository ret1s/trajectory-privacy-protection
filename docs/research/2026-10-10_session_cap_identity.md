# Cap từng phiên và đánh giá S4–S6 bằng cơ chế Geo-I hiện có

Quyết định thiết kế 10/10/2026: giữ Geo-I/REM và cap hiệu lực 0,23 m⁻¹ mỗi
phiên. Đánh giá bảo vệ danh tính **người** và **phương tiện vật lý** riêng,
trước khi quyết định cần thêm cơ chế nào. Chưa chọn lớp identity, relay hay
k-anonymity làm thành phần của mô hình. Bản này sửa đề xuất thêm lớp S4 quá
sớm trong bản trước; không thay đổi mã cơ chế hoặc số đo đã lưu.

## 1. Ngân sách tọa độ từng phiên

Một phiên là một chuyến/lượt sử dụng theo giao thức công khai, có nhiều lần
đọc GPS và gửi Q. Khởi động lại để tiếp tục cùng chuyến không tự nạp ngân sách.
Với cap C_s và tham số phân bổ H công khai, U = 2H − 1:

- u = C_s/U; B_nominal = 2H u.
- C_s = 0,23 m⁻¹; H = 12; U = 23; u = 0,01 m⁻¹; B_nominal = 0,24 m⁻¹.
- Lịch đọc tối thiểu 60 s; dự toán trước GPS: lần đầu 1 đơn vị, lần sau 2.
- Chi phí thực: đầu tiên 1; thử rồi giữ Z 1; thử rồi tạo mới 2; không đọc 0.
- H không phải giới hạn cứng 12 GPS reads. Hết cap đọc thì dự đoán từ lịch sử
  bảo vệ. C_s là hệ số riêng tư, không phải bán kính nhiễu.

Cơ chế per-session đã có trong GeoI-Slack/session-reset. Kết hợp chính sách này
với L30 và bốn nhu cầu local chưa có readout mới. Không đổi cấu hình các study
frozen để gán lại nhãn cho kết quả. Chặn lý tưởng một phiên là exp(C_s D∞);
những phiên bị liên kết vẫn hợp thành tổng cap. Công thức phân bổ không chứng
minh 0,23 tối ưu phổ quát; còn cần khảo sát độ nhạy và lịch đọc.

## 2. S4: danh tính người và xe vật lý

S4-person hỏi hai phiên có cùng người không; S4-vehicle hỏi có cùng phương
tiện vật lý không. ID thiết bị/tài khoản là kênh phụ trợ, không thay cho hai
nhãn. Evaluator đã có `person_id` và `physical_vehicle_id` riêng.

| Cặp phiên | Cùng người | Cùng xe |
|---|---:|---:|
| Một người dùng lại cùng xe | 1 | 1 |
| Một người đổi xe | 1 | 0 |
| Hai người dùng chung xe | 0 | 1 |
| Hai người, hai xe | 0 | 0 |

**Cơ chế hiện có:** REM làm nhiễu tham chiếu; phép thử tái sử dụng có nhiễu
bảo vệ tín hiệu giữ/đổi Z; lịch đọc và cap giới hạn quan sát GPS mới. b/Q chỉ
xử lý từ lịch sử bảo vệ và dữ liệu công khai. Chúng không thêm GPS thô vào
transcript, nhưng không có bảo đảm chống linkage riêng mạnh hơn Geo-I.

**Observer:** diagnostic nhìn Q và timing, không nhận nhãn người/xe/GPS thật.
Việc không có account/IP/ID trong simulator là giả định phạm vi, không phải
một cơ chế transport đã được triển khai hoặc một gain riêng tư mới.

**Bằng chứng:** cap phiên 0,23/m, AUC người raw 0,778 → Geo-I 0,532; AUC xe
0,718 → 0,448. Ba nhóm test, 45 cặp phụ thuộc nhau. AUC người theo nhóm còn
tới 0,861; AUC dưới 0,5 có thể đảo điểm. Nhãn danh tính và các cặp dùng chung/
đổi xe là tổng hợp. Kết luận: bằng chứng một phần cho linkage từ hình học,
chưa chứng minh ẩn danh hoặc bảo vệ trước account/IP. Xem
[diagnostic đầy đủ](2026-10-05_identity_future.md).

## 3. S5/S6: cơ chế hiện có và pilot cap phiên

**S5 – cạnh đường tương lai:** attacker chỉ nhận prefix Q; REM làm mờ hướng
rẽ, lịch đọc thưa/tái dùng Z giảm việc bám từng chuyển động GPS. b/Q dùng
chuyển tiếp công khai, không biết cạnh tương lai đã chọn. Tính nhân quả tự nó
không đủ để chứng minh riêng tư trước tương quan hình học.

**S6 – đích chưa tới:** bảo vệ tọa độ trong lịch sử lẫn prefix hiện tại;
đích riêng của nhu cầu ít đi vòng chỉ dùng local. Nhiều chuyến vẫn có thể
lộ routine; S6 pilot giả định attacker đã nối được lịch sử cùng người.

[Protocol](../../artifacts/benchmarks/future_native_20261005_v1/protocol.json),
[kết quả](../../artifacts/benchmarks/future_native_20261005_v1/results.json),
[validation](../../artifacts/benchmarks/future_native_20261005_v1/validation.json):
nhánh **geoi_session_reset**, L10, u=0,01/m, cap phiên 0,23/m, K5. Tại mốc
đang rẽ, Candidate Trees chọn trên selection cho S5 accuracy/S6 Hit100
100% → 41,67%; S6 MAE 4,99 → 660,62m; Static Recall@5 97,72%.

Sáu nhóm test, 12 query, hai cạnh/đích ứng viên đã biết. Query thường lệ/ít gặp
cân bằng; sáu chuyến lịch sử có năm thường lệ/một ít gặp không thay cho prior
query. S5/S6 dùng chung quyết định nhánh, không là hai xác nhận độc lập.
Trước ngã rẽ raw cũng đạt 50% vì prefix giống nhau: không tính đó là gain.
41,67% không chứng minh tốt hơn đoán ngẫu nhiên hoặc cận cho mọi attacker.
Utility chỉ là POI tĩnh phần có tham chiếu, không phải macro bốn nhu cầu L30.

Những số này là pilot đã lưu, không phải chạy mới. Pilot REM/Planar L20 khác
vẫn giữ nguyên số 50%/50%; không coi chúng là phép so ngang cùng cấu hình.

## 4. Khi nào Geo-I đủ để chứng minh hạn chế suy identity?

Với hai giả thuyết có cùng ngữ cảnh công khai, nếu **mọi cặp trace giữa hai
support** có D∞ ≤ r trong mỗi phiên, đặt α = r ∑ C_s. Chặn Geo-I lý tưởng cho
hai phân phối quan sát P,Q: P(A) ≤ exp(α)Q(A) và ngược lại. Vì vậy:

- TV(P,Q) ≤ tanh(α/2).
- Prior cân bằng: Bayes success ≤ exp(α)/(1+exp(α)).
- Prior p: success ≤ max(p,1−p,exp(α)/(1+exp(α))).

Đây là hệ quả đã có trong [chứng minh suy luận](../../thesis/current_formal_inference.tex),
không phải một primitive mới. Không suy điều kiện all-pair từ việc hai người
đang đứng gần nhau; hai nhóm identity có thể khác cả routine/hành trình.
Một phiên C_s=0,23/m, r=100m đã cho α=23, cận gần 100%: cận đúng nhưng yếu,
không chứng minh bảo vệ identity mạnh ở thang đó. Cần điều kiện kernel lý tưởng,
lịch/ngữ cảnh công khai chung và thiết bị tin cậy; sampler float chưa chứng nhận.

[Geo-I gốc, Andrés et al. (2013), §2–3](https://arxiv.org/html/1212.1984v3)
phân biệt bảo vệ vị trí với k-anonymity. K=5 Q không phải k=5 người dùng.
Không cần mặc định dùng k-anonymity: chỉ bổ sung khi kiểm chứng mô hình hiện
có cho thấy thiếu bảo vệ và một cơ chế phù hợp với observer/utility đã định.

## 5. Hướng kiểm chứng trước khi mở rộng mô hình

1. Chốt per-session/L30, target riêng người và xe, attacker bank/ngưỡng trước
   readout mới. Không điều chỉnh sample/test để làm đẹp điểm.
2. Dùng cặp một người đổi xe/hai người dùng chung xe và split theo thực thể/
   route family. Báo AUC có xét đảo điểm, balanced accuracy và độ bất định theo nhóm.
3. Mở rộng S5/S6 nhiều ứng viên/bản đồ mới, prior lệch và lịch sử dài; báo
   utility, bytes, latency cùng privacy. Phân biệt S6 giả định linked history với S4.
4. Nếu không đạt tiêu chí đã chốt, đánh giá các hướng k-anonymity, mix-zone,
   tách metadata/transport theo threat model. Chưa chọn hướng nào hoặc thêm vào
   sơ đồ như một component đã có. Account/IP cần threat/triển khai riêng nếu đưa vào phạm vi.

Report dùng sample và utility đã lưu ở cấu hình nhiễu khác; nêu u/cap của
chúng tại chỗ, không nhân ngân sách rồi giữ nguyên Z/Q. Benchmark, mã core,
canonical thesis và các bản lưu giữ nguyên. S8 còn là giới hạn mở.

## 6. Trọng tâm trình bày: tọa độ thực nghiệm, identity/future phân tích

Report giữ benchmark S1–S3, S9–S10 cùng utility, theo đúng cấu hình đã đo.
S1–S3 dùng `2026-09-26_brief/method_evidence.json` GeoI-Slack per-session/L10;
S9–S10 dùng Endpoint20 đã lưu. Không thay score hoặc gán chúng cho L30 mới.
Cận tọa độ là cơ sở toán học; không biến nó thành bảo đảm Hit100/MAE hoặc
thành tuyên bố các scenario đã được giải quyết hoàn toàn.

S4–S6 tập trung công thức likelihood/posterior odds, TV/Bayes có điều kiện.
Không đưa diagnostic AUC/accuracy nhỏ vào bảng chính; các diagnostic ở trên
và nguồn gốc vẫn giữ nguyên. Với S5 cùng prefix và ngữ cảnh, luật Q giống nhau
nên không thêm thông tin về lựa chọn tương lai; prior có thể đã mạnh. S6 dùng
lịch sử + prefix cần hợp thành mọi cap và giữ prior routine. S4 cần chặn all-pair
trên luật trace của giả thuyết cùng/khác người hoặc xe, không tự có từ K5.

Ví dụ lý thuyết u=0,01/m: chỉ khác một GPS 10m cho α≤0,2, Bayes cân bằng
≤54,98%; chỉ khác một GPS 100m cho α≤2, ≤88,08%. Toàn trace với D∞=100m và
cap phiên 0,23/m cho α=23, cận gần 100% và yếu. Đây không phải benchmark hoặc
cận identity hữu ích đã được xác nhận. Không suy cận một GPS cho hai tuyến.

Metric chọn theo secret/observer: tọa độ Hit/MAE; linkage AUC/BA; cạnh đúng
edge; đích accuracy/Hit/MAE; dịch vụ Recall/cost. Không buộc mọi scenario có
một metric khác hoàn toàn, cũng không ép một score vào mọi model/task.
