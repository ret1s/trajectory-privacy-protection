# S9/S10: bảo vệ tọa độ đầu/cuối, không trì hoãn truy vấn

Cập nhật theo buổi gặp 03/10/2026. Đây là **vòng phát triển mới**, không thay số
benchmark đã công bố. Kết quả đủ để chọn một cấu hình thử tiếp; chưa xác lập
vượt các paper nguyên bản hoặc mọi attacker.

## Thay đổi đã chạy

Giữ Geo-I trên mạng đường, kiểm tra tái dùng Z có nhiễu, lịch đọc GPS 60 giây,
ước lượng lịch sử đã bảo vệ và chọn năm Q theo đường đi/độ phủ POI. Bỏ warmup và
delay: **gửi ngay cả ở mốc đầu và mốc cuối**. Máy chủ trả POI; thiết bị hợp, bỏ
trùng và dùng GPS hiện tại chọn top-5.

Không biết trước lúc chuyến kết thúc, nên không dùng GPS tương lai để nhận diện
“60 giây cuối”. Cấu hình được chọn tăng nhiễu **trên mọi lần dùng GPS cho cơ chế
bảo vệ**. Giờ bắt đầu/kết thúc, account/IP và thời điểm gửi vẫn có thể lộ. Đây là
bảo vệ tọa độ, không phải che thời điểm hoạt động.

## Hai vòng có lựa chọn và kiểm tra tách biệt

Fit attacker trên nhóm 701–708; chọn cấu hình/attacker trên 709–710. Mỗi nhóm
lấy hai session đầu theo thứ tự ID; hai seed cơ chế cố định. Không loại session
theo privacy/utility. Các session cùng nhóm và các seed không được coi là những
người dùng độc lập. Truy vấn ở mỗi 60 giây và thêm mốc cuối, toàn bộ hành trình
được giữ nguyên.

Hai seed được dùng như common random numbers giữa các session/phương pháp
trong thí nghiệm. Khoảng bất định vì vậy còn điều kiện trên hai lựa chọn ngẫu
nhiên này; chưa bao quát mọi seed cơ chế. Khi triển khai phải có nguồn ngẫu
nhiên riêng tư và không tái khởi tạo cùng seed cho các chuyến liên kết. Cận
0,0575/m là **mỗi phiên**; nhiều phiên của cùng người phải cộng ngân sách,
không tự nhận một cận chung 0,0575/m cho cả lịch sử.

- **Vòng 1:** thử hệ số epsilon 1, 1/2, 1/4, 1/8 với L=10. Các bản tăng nhiễu
  đạt Recall trung bình tập chọn 89,33%, 87,17%, 84,07%, đều trượt ngưỡng 90%.
  Lựa chọn bị khóa là bản thường; kiểm tra một lần trên nhóm 711–712. Giữ toàn
  bộ kết quả thất bại, không chỉ lưu bản tốt nhất.
- **Vòng 2:** sau chẩn đoán trên, khóa lưới hệ số 1, 1/2, 1/4 × L=10/20/40,
  thêm cấu hình pha đầu quarter-epsilon và đối chứng delay. Chọn trên 709–710;
  khóa trước khi đọc tập kiểm tra mới gồm nhóm **1201–1204**. Không dùng lại
  711–712 để chọn rồi gọi đó là holdout. Đây vẫn là các dataset SUMO có sẵn,
  không phải xác nhận độc lập bằng dữ liệu thực địa mới.

L tăng là thông số dịch vụ công khai, không phụ thuộc GPS thật. Planner được
cập nhật theo đúng độ sâu phản hồi L. Có đối chứng GeoI-Slack thường **cùng L**
để tách hiệu ứng tăng nhiễu khỏi hiệu ứng tăng độ phủ POI/chi phí mạng.

Quy tắc chọn vòng 2: Recall ≥90%; giảm trung bình Hit100 S9/S10; nếu hòa thì
ưu tiên MAE cao hơn, rồi byte thấp hơn. Attacker được chọn riêng theo MAE và
từng ngưỡng Hit trên tập chọn; các cột Hit không nhất thiết tạo một CDF chung.

## Attacker và positive control

Mỗi phương pháp được fit bằng shadow transcript của chính phương pháp đó.
Bank gồm centroid/median, Viterbi theo lịch sử và chuyển động, ngoại suy OLS,
OLS dùng giờ mở/đóng phiên công khai, **kNN và Extra Trees** trên thống kê tập
điểm, và **kNN/Extra Trees** bổ sung đặc trưng cả chuỗi. Không giả định ID Q
ổn định; đặc trưng không đổi khi hoán vị thứ tự Q trong mỗi sự kiện.

Attacker chỉ nhận tọa độ/thời điểm công bố và thời điểm đóng phiên được phép
quan sát. Không nhận Z, GPS thật, branch giữ/tạo Z, ngân sách đã dùng hoặc nhãn
đích của tập kiểm tra. Nhãn thật của tập fit chỉ dùng để huấn luyện. Positive
control gửi GPS thật đạt Hit100=100%, MAE=0, Recall=100%: kiểm tra được đường
đánh giá đầy đủ endpoint hoạt động. Điều đó **không chứng minh bank là đối thủ
mạnh nhất có thể**.

Target của vòng này là vị trí đầu/cuối của hành trình mô phỏng đầy đủ. Chưa
đánh giá toàn bộ các biến thể chỉ thấy prefix hoặc liên kết nhiều chuyến của
benchmark endpoint trước; không cộng kết quả này vào bảng cũ.

## Cấu hình được chọn: GeoI-Endpoint20

Tên dễ nhớ **GeoI-Endpoint20** tương ứng ID `scale025_L20` trong artifact:

| Tham số | Giá trị |
|---|---:|
| Hệ số epsilon so với GeoI-Slack thường | 1/4 |
| ε kiểm tra / ε tạo Z | 0,0025/m / 0,0025/m |
| B thực tế / H | 0,06/m / 12 |
| Cận phiên hiệu lực | 23 × 0,0025 = **0,0575/m** |
| Lịch dùng GPS cho cơ chế bảo vệ | 60 giây, khi còn đủ ngân sách |
| K / L / top-k tại thiết bị | 5 / **20** / 5 |
| Ngưỡng tái dùng / slack | 200 m / 0,03 |
| Warmup / delay | **0 / 0 giây** |

B=0,06 là ngân sách thực tế sau khi nhân 0,24 với 1/4, không phải vẫn tiêu
0,24/m. Lần tạo đầu tốn 0,0025; lần kiểm tra giữ Z tốn 0,0025; kiểm tra rồi tạo
mới tốn 0,005/m. Dự toán đủ bước tệ nhất trước khi đọc GPS; hết ngân sách thì
tiếp tục dự đoán/hậu xử lý công khai, không đọc GPS mới cho Q. GPS cục bộ dùng
xếp hạng POI không được đưa vào truy vấn.

## Kết quả kiểm tra vòng 2

Bốn nhóm, tám session, hai seed: **16 lần chạy cho mỗi phương pháp**. Tổng
170 mốc dịch vụ mỗi phương pháp. Gộp đều trong nhóm rồi đều giữa các nhóm.
Hit thấp và MAE cao thuận lợi cho privacy trước bank đã chọn; Recall cao tốt
cho dịch vụ. Đơn vị byte là JSON yêu cầu + ID POI phản hồi mỗi mốc đầu vào,
chưa gồm HTTP/TLS.

| Phương pháp | S9 Hit100 / MAE m | S10 Hit100 / MAE m | Recall@5 | Gửi / đầu vào | Delay | Byte/mốc |
|---|---:|---:|---:|---:|---:|---:|
| GPS thật | 100% / 0 | 100% / 0 | 100% | 170/170 | 0s | 939 |
| GeoI-Slack, L10 | 0% / 653 | 0% / 668 | 97,32% | 170/170 | 0s | 4.593 |
| GeoI-Slack, L20 | 0% / 970 | 0% / 1.844 | **99,59%** | 170/170 | 0s | 6.953 |
| GeoI-Slack + warmup/delay60, L10 | 0% / 1.179 | 0% / 1.180 | 78,08% | 122/170 | 60s | 3.247 |
| **GeoI-Endpoint20** | 0% / **1.852** | 0% / **2.297** | **96,11%** | **170/170** | **0s** | **6.951** |

Không dùng Hit100=0 ở mọi bản để nói bản mới vượt trội. Ở **Hit500 S9**, L20
thường đạt 37,5%, Endpoint20 đạt 0%; S10 cả hai đạt 0%. Đối chứng delay đạt
0% ở cả hai ngưỡng này. Vì vậy khác biệt còn được đọc qua MAE và utility.

So cùng L20: S9 tăng MAE **882 m**, bootstrap ghép theo nhóm 95% [416; 1.216].
S10 tăng trung bình **453 m**, nhưng khoảng [-382; 1.236] qua 0: **chưa chắc
chắn có lợi thế S10**. S9 ΔHit500=-37,5 điểm %, khoảng [-75; 0], cũng chưa
đủ để tuyên bố giảm chắc chắn ngoài mẫu. Chỉ bốn nhóm; không hiệu chỉnh nhiều
so sánh. Bootstrap Hit100 hòa 0 là suy biến, không phải bảo đảm rủi ro bằng 0.

Nếu giả sử bốn nhóm là Bernoulli độc lập từ bộ sinh này, không nhóm nào có
Hit100 vẫn cho cận trên một phía 95% là **52,7%** cho xác suất một nhóm mới
có ít nhất một hit: `1 − 0,05^(1/4)`. Đây là minh họa bất định của mẫu nhỏ,
không phải cận với mọi attacker hoặc người dùng thực tế.

Endpoint20 dùng khoảng **1,51 lần byte của L10**, gần bằng byte đối chứng
cùng L20. Recall thấp hơn L20 thường **3,48 điểm %**. Recall trung bình 96,11%
không phải bảo đảm mỗi chuyến: lần chạy thấp nhất đạt **85,92%**. So với delay,
nó duy trì dịch vụ ở tất cả mốc và tăng Recall 18,03 điểm %, với lưu lượng
lớn hơn khoảng 2,14 lần tính mỗi mốc đầu vào.

Kết luận dùng được: **thử tăng nhiễu kèm phản hồi sâu hơn có thể giữ utility
khá cao mà không trì hoãn**, và có tín hiệu S9 tốt hơn đối chứng cùng L trên
mẫu phát triển này. S10 cần thêm nhóm và attacker; chưa kết luận vượt mọi
phương pháp hoặc bảo vệ hoàn toàn điểm đầu/cuối.

## Nguồn bản đồ và giới hạn

Cache SUMO/OSM gốc không còn. Mạng mới được dựng từ toàn bộ **9.138 polyline
công khai lưu trong repo**, không dùng GPS để chọn vùng. Clustering đầu cạnh
35 m; loại cạnh tự vòng bị collapse theo quy tắc hình học công khai; giữ 4.849
cạnh. Catalogue 22.106 state cách tối đa 40 m, 34.356 cung; tốc độ công khai
8 m/s, 418 POI, 907 ô ước lượng. Native `netconvert` đọc thành công. Hướng
chuyển tại giao lộ được tái dựng, **không xác minh luật rẽ/làn của mạng gốc**.

GPS nguồn không thay đổi. Sai lệch snap lớn nhất trên tập kiểm tra mới là
34,75 m. Endpoint attacker chấm tới **GPS nguồn**, không đổi đáp án thành
đỉnh gần nhất để làm số dễ hơn. Dịch vụ tính khoảng cách trên mạng mới nên
không so trực tiếp Recall/MAE với artifact mạng gốc hoặc paper.

## Evidence và tái kiểm tra

[Thư mục evidence](../../artifacts/benchmarks/endpoint_noise_20261005/README.md)
có hai protocol, toàn bộ cấu hình phát triển, selection đã khóa, transcript,
kết quả, nguồn code lossless và model attacker gzip có SHA-256.

Kiểm tra độc lập đã tính lại **104 lần chạy, 80 metric endpoint, 1.060 mốc
dịch vụ và 12.480 dự đoán attacker**; kiểm tra clock, schema công khai, cận
ngân sách, tách nhóm và byte. Tests cơ chế/attacker và wrapper không delay
được bổ sung. Không sửa dataset, artifact benchmark hoặc báo cáo trước.
