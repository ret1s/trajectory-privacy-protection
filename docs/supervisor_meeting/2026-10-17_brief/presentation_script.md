# Kịch bản trình bày GeoI-BQ với GVHD

Buổi 17/10/2026; khoảng 10–12 phút. Đi theo sáu trang của `report_explained.pdf`.
Các câu dưới đây là lời nói; slide/report chỉ giữ nội dung nghiên cứu.

## Trang 1 — Phương pháp hiện tại

“Thưa thầy/cô, tuần này em tập trung chốt một phiên bản cân bằng: chất lượng
dịch vụ tốt, nhưng trọng tâm là có lập luận riêng tư cho toàn bộ quá trình
truy vấn, thay vì chỉ một lần thêm nhiễu.

Nền tảng Geo-I vẫn giữ nguyên. GPS đi qua REM và phép thử giữ hay tạo lại
điểm tham chiếu Z. Từ thông tin đã bảo vệ, thiết bị duy trì belief về vị trí.
Z và belief đều giữ nội bộ.

Phần mới là chọn cả một đoạn gồm ba khung Q, với xác suất được giới hạn mức
phụ thuộc vào belief. Cứ 60 giây chọn đoạn mới; mỗi 20 giây gửi một khung
của đoạn đó. Thư viện chỉ dựa vào bản đồ, những Q đã công bố và POI đã nhận.

Máy chủ nhận năm Q, truy hồi mọi loại POI với L60. Thiết bị hợp phản hồi,
bỏ trùng, rồi dùng GPS và nhu cầu thật để trả danh sách phù hợp đã sắp xếp.
Không phải chỉ trả năm địa điểm; top-5 chỉ dùng để đánh giá.”

**Chỉ hình:** GPS → Geo-I → chọn đoạn Q → request chung → cache/local answer.
Chỉ đường GPS local đi thẳng xuống ranking, không đi vào support Q.
Chỉ khung xanh và máy chủ bên ngoài để xác định ranh giới mô hình.

## Trang 2 — Vì sao có cận riêng tư cả phiên?

“Em tách hai loại ngân sách vì chúng đo hai kênh khác nhau.

Ngân sách Geo-I có đơn vị trên mét, kiểm soát mức phân biệt giữa các vị trí.
Cap 0,23 là ngân sách cả phiên, tương ứng 23 đơn vị, không phải bán kính
nhiễu và không phải epsilon của từng lần đọc. Giữ Z sau một noisy test
vẫn tốn một đơn vị; không đọc GPS mới thì không tốn.

Ngân sách Q không có đơn vị mét. Em phân bổ cho lần chọn đoạn thứ j theo
1 chia j nhân j cộng một. Tổng của lịch này không vượt một, dù chưa biết
chuyến kéo dài bao lâu.

Điểm quan trọng trong phân phối chọn Q là mọi đoạn đều còn xác suất dương.
Belief có thể nghiêng phân phối, nhưng không loại đoạn theo vị trí riêng.
Vì vậy hai đầu vào riêng tư bất kỳ chỉ làm xác suất cùng một đoạn thay đổi
trong một tỷ số có giới hạn.

Nhân các conditional probabilities qua cả phiên cho cận transcript.
Tại Gamma bằng một, tỷ số không vượt khoảng 2,437372. Cận này bao gồm thứ
tự năm Q, phản hồi và retry logic; nó không dựa vào attacker phải suy ra Z trước.

Geo-I còn cho cận theo khoảng cách trong mô hình lý tưởng. Khi cả hai cận
đúng, ta lấy cận nhỏ hơn. Không cộng hai ngân sách có đơn vị khác nhau.”

**Nếu hỏi về float:** “PMF Q dùng integer weights với bit đều giả định.
Cận Geo-I theo khoảng cách vẫn là ideal-kernel guarantee. Em đã tách rõ,
không gọi toàn stack float là một pure-DP implementation đã được chứng nhận.”

## Trang 3 — Các mục tiêu suy luận và utility

“Từ likelihood ratio, em dùng Bayes để chặn mức posterior của attacker
tăng so với prior. Đây là giới hạn thông tin thêm, không phải lời khẳng định
attacker không thể đoán đúng.

Ngoài posterior cho từng transcript, em có cận trung bình qua cả phiên.
Mọi attacker với reward được chặn chỉ tăng expected success tối đa 5,53
điểm phần trăm so với chiến lược tốt nhất từ prior. Con số 55,53% chỉ là
ví dụ khi có hai lựa chọn với prior đều; nó không phải accuracy benchmark.

S1 đến S3 và S9, S10 có cả phân tích và benchmark tọa độ. S4 đến S6 nhận
các hệ quả suy luận có điều kiện cho danh tính người/xe, cạnh sắp đi và
đích đã dự định. S7 giữ request thật tại thiết bị. S8 cần một mô hình
joint transcript riêng nên chưa đưa vào claim single-user.

Ở utility, khi metadata còn hợp lệ và ranking dùng cùng GPS hiện tại,
cache chỉ tích lũy nên không làm mất những POI đã có. Em cũng chặn chênh
lệch utility của whole policy so với cùng public-base planner mà không cần
belief phải chính xác.

Qua audit, em làm rõ các điều kiện chuyển từ public-reference floor sang
utility thật. Không thể lấy một mean trên bảng công khai rồi nói mọi GPS
và mọi nhu cầu đều đạt 99%.”

**Nếu hỏi restart:** “Q ledger được persist. Phục hồi Geo-I còn cần caller
restore đầy đủ upstream; cùng timestamp chưa đủ chứng nhận điều đó. Đây
là phần triển khai còn mở, em đã giữ phản ví dụ trong audit.”

## Trang 4 — Ví dụ nhiều bước

“Mẫu này là trip TRAIN đầu tiên theo thứ tự cố định, đọc từ log của đúng
cấu hình hiện tại. Em dùng để minh họa cơ chế, không chọn vì số bảo vệ đẹp.

Ở giây không, GPS tạo Z đầu tiên và chi một đơn vị Geo-I. Kernel chọn một
đoạn, reserve một phần hai ngân sách Q, và khung đầu nhận 381 POI unique.

Ở giây 20 và 40, không có private GPS read mới. Thiết bị gửi khung đã chọn;
không lấy mẫu Q lại và không chi thêm ngân sách Q. GPS hiện tại vẫn có thể
được dùng tại local để xếp hạng đáp án.

Ở giây 60, đủ lịch và dự toán, noisy test dẫn đến tạo Z mới. Tổng Geo-I
thành ba đơn vị. Chọn đoạn thứ hai dùng một phần sáu, nên Q reservation
thành hai phần ba. Khung này có 377 POI nhưng union với cache chỉ thành
385 vì nhiều POI trùng nhau.

Ở giây 300, mô hình đọc GPS rồi giữ Z: vẫn chi một đơn vị test. Đây khác
với không đọc GPS. Đồng thời Q vẫn được chọn theo lịch đoạn mới.

Ở hình, GPS và Z bên trái là thông tin nội bộ. Bên phải là năm tuyến Q
máy chủ thấy. Chúng tuân theo đường và không phải GPS được gửi dưới tên khác.”

## Trang 5 — Kết quả benchmark

“Các kết quả chính dùng cùng Geo-I history, L60 và cache/cadence cho các
K5 arms. Attacker học trên TRAIN, chọn trên SELECTION, rồi mới đánh giá TEST.

GeoI-BQ đạt 99,595% Recall. So với baseline K5, giảm khoảng 0,405 điểm
phần trăm. Cận dưới khoảng tin cậy vẫn nằm trong mức chấp nhận hai điểm
phần trăm, cả macro và từng purpose, nên đạt tiêu chí đã đặt trước.

Về privacy thực nghiệm, gửi trực tiếp Z có S10 Hit100 là 35% và MAE khoảng
247 mét; GeoI-BQ có 0% và khoảng 2.124 mét. Đây là evidence đối với attacker
bank đã kiểm tra, không biến 0% trên mẫu hữu hạn thành xác suất tấn công bằng không.

Phương pháp trả thêm traffic: K5 có 155 point requests mỗi phiên so với 31
của K1; reply khoảng 82,6 KB so với 16,9 KB. Em ghi riêng chi phí, vì không
thể kết luận hơn mọi aspect khi output contracts khác nhau.”

## Trang 6 — Diễn giải và đóng góp

“Chất lượng dịch vụ tốt đến từ truy hồi đủ ứng viên và cache. L60 được chọn
từ thiết kế công khai trước private defense draws. Ở cuối phiên cache gần
đầy catalogue 418 POI, nên đây là một điều kiện quan trọng của kết quả.

Control Gamma không có Recall hơi cao hơn bản chính. Do đó em không nói
private belief đã làm utility tăng trong catalogue này. Vai trò Geo-I là
nền tảng spatial privacy và protected belief; Q bổ sung cận toàn phiên.

Đóng góp em muốn nhấn mạnh là đặc tả đầy đủ bài toán, cơ chế continuous
queries có cận xác suất, chuỗi chứng minh privacy và utility, cùng benchmark
thể hiện trade-off có kiểm soát. Em giữ cấu hình này để hoàn thiện luận văn,
ưu tiên độ chặt chẽ thay vì tiếp tục tối ưu một vài điểm benchmark.

Nếu mở rộng sang implementation được chứng nhận, hai bước kỹ thuật tiếp
theo là verified Geo-I sampling và phục hồi upstream đầy đủ có xác thực.”

## Những câu cần tránh

- “Có năm Q nên đã k-anonymity” hoặc “không thể suy ra Z/GPS”.
- “0% Hit100 nghĩa không attacker nào tấn công được”.
- “99,595% là theorem cho mọi người dùng” hoặc “Gamma1 thắng mọi control”.
- Gọi kết quả mới là untouched-geometry test: GPS/geometry đã được dùng;
  đây là stochastic replication với defense draws mới.

Nguồn số và sample: `report_sources.json`, `sample_walkthrough.json`;
chứng minh/audit: các tài liệu nghiên cứu và review14/10 được ghim trong nguồn report.
