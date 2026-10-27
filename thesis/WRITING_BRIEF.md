# Brief hướng viết luận văn: từ bài toán đến bảo đảm toàn transcript

Ngày soạn: 27/10/2026. **Đề xuất để duyệt trước khi viết chi tiết.**

Mục tiêu là thống nhất câu chuyện khoa học, nội dung và thứ tự trình bày. Brief này chưa thay cấu trúc hoặc kết quả trong `main.tex`; chưa chạy thêm thực nghiệm. Sau khi chốt, tiếp tục dùng `main.tex` làm nguồn luận văn duy nhất.

## 1. Luận điểm xuyên suốt

**Bảo vệ một vị trí riêng lẻ chưa đủ để đánh giá privacy của một chuỗi truy vấn. Luận văn giữ Geo-I làm nền tảng, kiểm soát việc tạo/cập nhật thông tin vị trí đã bảo vệ và bổ sung cơ chế chọn đoạn Q có cận tiết lộ toàn phiên. Chất lượng dịch vụ được hỗ trợ bằng lựa chọn Q, thu hồi POI và xử lý nhu cầu thật tại thiết bị.**

Luồng lập luận:

> Bài toán và quyền quan sát của attacker → khoảng trống của bảo vệ từng điểm → yêu cầu thiết kế → thuật toán → giả thiết và chứng minh toàn transcript → đánh giá privacy–utility–cost → kết luận đúng phạm vi.

Không tổ chức luận văn theo lịch sử các tuần phát triển. Dùng phiên bản tham chiếu **GeoI-BQ v1** làm phương pháp chính; những bản cũ và thử nghiệm chưa được chọn chỉ phục vụ đối chiếu hoặc giải thích quyết định thiết kế.

Ba câu hỏi nghiên cứu dự kiến:

- **RQ1 — Privacy:** với attacker quan sát cả transcript, giới hạn nào cho khả năng phân biệt các quỹ đạo có thể được chứng minh?
- **RQ2 — Utility:** việc chọn Q, hợp/cache và xếp hạng local đem lại những bảo đảm nào; khi nào chúng áp dụng cho vị trí thật?
- **RQ3 — Thực nghiệm:** trong protocol đã khai báo, phương pháp đánh đổi privacy, chất lượng dịch vụ và chi phí ra sao so với các đối chứng?

## 2. Thứ tự đọc: cấu trúc bảy chương đề xuất

Đây là cấu trúc đích, chưa phải mục lục đang được import trong bản LaTeX cũ.

| Chương | Nội dung chính | Điều người đọc cần hiểu trước khi chuyển chương |
|---|---|---|
| **1. Giới thiệu bài toán và mục tiêu nghiên cứu** | Bối cảnh LBS; vì sao cả chuỗi truy vấn có thể tiết lộ thông tin; yêu cầu giữ chất lượng dịch vụ; ba RQ; đóng góp và phạm vi. | Cần kiểm soát thông tin của cả transcript, không chỉ che GPS hoặc tạo các điểm giả. |
| **2. Cơ sở lý thuyết và nghiên cứu liên quan** | Geo-I là thuộc tính, REM là cơ chế; hợp thành, hậu xử lý và Bayes; các hướng bảo vệ quỹ đạo; hợp đồng đầu ra và metrics của nghiên cứu liên quan. Phân biệt nền tảng kế thừa với phần thiết kế/phân tích của luận văn. | Vì sao chọn Geo-I và vì sao cần thêm một đặc tả cho quá trình chọn Q nhiều bước. |
| **3. Đặc tả bài toán, đối thủ và mục tiêu đánh giá** | Dữ liệu riêng/công khai; phiên và transcript; attacker biết gì; S1–S10; yêu cầu privacy, utility và cost; giả thiết về clock, provider, trạng thái và randomness. Định nghĩa metrics theo đúng tác vụ và dạng đầu ra. | Ta đang bảo vệ thông tin nào, trước ai, trong điều kiện nào và đánh giá bằng gì. |
| **4. Phương pháp GeoI-BQ** | Kiến trúc theo lớp và ranh giới client/server; REM/noisy reuse; ngân sách; cập nhật belief; thư viện đoạn khả thi; điểm utility và phân phối chọn đoạn; gửi khung, nhận POI, cache và xếp hạng local. Có giả mã và một ví dụ nhiều bước. | Thuật toán chạy được như thế nào, vì sao cần từng thành phần và thông tin nào được phép đi qua mỗi bước. |
| **5. Phân tích và chứng minh toán học** | Phát biểu giả thiết trước định lý; bốn mệnh đề privacy và bước nối thành bảo đảm toàn transcript; diễn giải suy luận; các kết quả utility riêng; bảng định lý → giả thiết → cơ chế trong mã → kiểm tra. | Bảo đảm cụ thể là gì, vì sao đúng và điều gì chưa được kết luận từ định lý. |
| **6. Thiết kế thực nghiệm và kết quả** | Nguồn dữ liệu, chia family, cấu hình đóng băng; baseline và ablation; lựa chọn attacker; S1–S3/S9–S10; utility theo purpose; CI, chi phí và độ nhạy. Thảo luận kết quả không thuận lợi và giới hạn tổng quát hóa. | Bằng chứng thực nghiệm hỗ trợ những kết luận nào; có phù hợp với phạm vi lý thuyết và đánh đổi đã nêu không. |
| **7. Kết luận và hướng phát triển** | Trả lời từng RQ bằng kết quả tương ứng; tổng kết đóng góp kế thừa/mới; giới hạn và các hướng tiếp tục có ưu tiên. | Luận văn đã hoàn thành điều gì và còn mở điều gì. |

Các kiến thức giao thông đô thị/SUMO cần thiết được đặt ở chương 2 và phần dữ liệu chương 6; không tách thành một tuyến nội dung làm gián đoạn lập luận privacy.

## 3. Trình bày phương pháp: cơ chế trước, công thức sau

Chương 4 đi theo một luồng duy nhất:

> GPS → kiểm tra giữ/đổi và REM → lịch sử Z đã bảo vệ → belief b → chấm điểm các đoạn Q khả thi → lấy mẫu một đoạn → gửi lần lượt các khung → hợp POI/cache → lọc và sắp xếp local bằng GPS và nhu cầu thật.

Mỗi thành phần trình bày theo cùng bốn câu: **đầu vào là gì → thực hiện gì → đầu ra là gì → giải quyết vấn đề nào**. Sau đó mới đưa công thức và giả mã.

Các phân biệt phải nhất quán:

- **Z** là tọa độ tham chiếu đã bảo vệ, không phải điểm truy vấn gửi mặc định lên server.
- **b** là phân phối ước lượng trên các trạng thái/ô đại diện; không mặc định là posterior thật đã hiệu chuẩn.
- **Một khung Q** gồm năm query slots tại một thời điểm. **Một đoạn Q** gồm ba khung được lấy mẫu chung, rồi phát ở các mốc 0/20/40 giây; đoạn sau bắt đầu sau 60 giây.
- **p** là phân phối chọn đoạn; khác phân phối belief b. Thư viện và xác suất nền dùng thông tin công khai, belief chỉ điều chỉnh xác suất trong giới hạn.
- Nhịp gửi là 20 giây; nhịp dùng GPS mới cho lớp bảo vệ tối thiểu 60 giây nếu đủ ngân sách. GPS dùng xếp hạng local có vai trò riêng, không suy ra tổng số lần đọc GNSS hoặc tiết kiệm pin.
- Cấu hình chính dùng K=5, L=60 **mỗi loại POI cho mỗi Q**, ngân sách Q Γ=1 và cap Geo-I từng phiên. Không đưa cap tám phiên hoặc L30 của bản cũ vào mô tả phương pháp chính.
- Ứng dụng trả danh sách phù hợp đã sắp xếp; Recall@5 là metric đánh giá, không phải giới hạn đáp án người dùng.

Ví dụ nhiều bước phải cho thấy: lúc nào đọc GPS bảo vệ; giữ hay đổi Z và chi phí; belief cập nhật hay chỉ dự đoán; lúc nào chọn đoạn mới; lúc nào chỉ phát khung đã chọn; cache và đáp án local thay đổi thế nào. Dùng một ví dụ minh họa, không chọn ví dụ để thay thế kết quả benchmark.

## 4. Trình bày chứng minh: cần kết luận gì ở từng bước?

### 4.1. Privacy

Giữ bốn mệnh đề theo cách giải thích trong report; khi tích hợp phải đối chiếu với cách nhóm khác trong tài liệu toán đầy đủ. Dùng nhãn nội dung ổn định, không sao chép số mệnh đề giữa tài liệu một cách máy móc.

| Mệnh đề | Khẳng định cần chứng minh | Cách nối vào toàn mô hình |
|---|---|---|
| **1 — REM** | Xác suất chuẩn hóa; với cùng đầu ra, tỷ số xác suất dưới hai GPS bị chặn theo khoảng cách. | Có một cơ chế tạo Z trên miền đường công khai thỏa Geo-I lý tưởng. |
| **2 — Noisy reuse và ngân sách** | Phép kiểm tra giữ/đổi thỏa Geo-I; tính cả chi phí test và REM; bộ lọc bảo đảm tổng chi phí không vượt cap. | Hợp thành có điều kiện theo lịch sử để đi từ thao tác riêng lẻ đến lịch sử đã bảo vệ của cả phiên. |
| **3 — Chọn một đoạn Q** | Phân phối centered hợp lệ; support chung; tỷ số xác suất chọn cùng đoạn bị chặn dù belief/điểm riêng khác nhau. | Có cận tiết lộ riêng cho một lần chọn đoạn, không cần belief được hiệu chuẩn. |
| **4 — Nhiều đoạn Q** | Lịch phân bổ có tổng không vượt Γ; chain rule hợp thành các lần chọn thích nghi; phát khung/retry không tạo lần lấy mẫu chưa tính. | Có cận cho toàn transcript, không cần biết trước số đoạn và không giả sử các bước độc lập. |

**Bước nối bắt buộc sau bốn mệnh đề:** hậu xử lý truyền cận Geo-I từ lịch sử Z xuống transcript; cận Q áp dụng riêng cho cùng transcript. Khi cả hai cùng có hiệu lực, dùng cận chặt hơn. Không cộng ngân sách Geo-I theo m⁻¹ với Γ không có đơn vị.

Diễn giải bằng Bayes: transcript chỉ được làm thay đổi posterior odds so với prior odds trong giới hạn đã chứng minh. Không kết luận “không thể suy ra GPS/Z” hoặc “attacker chỉ có xác suất đoán đúng bằng một con số cố định” với mọi prior.

Về ngân sách, tách **tính đúng của accounting** khỏi **mức bảo vệ có ý nghĩa**. Giải thích Cs=0,23 m⁻¹ đến từ cấu hình B/H; không coi đó là hằng số Geo-I, bán kính nhiễu hoặc mức an toàn tối ưu đã được chứng minh. Trình bày đánh đổi giữa cap, chi phí mỗi thao tác và khả năng thích nghi.

### 4.2. Utility

Trình bày riêng ba mức, không dùng điểm proxy để thay cho chất lượng thực tế:

1. **Điểm lập kế hoạch:** chọn Q theo xác suất ưu tiên điểm tốt; có thể chứng minh gain kỳ vọng của điểm chuẩn hóa so với phân phối nền. Đây chưa phải gain Recall thật.
2. **POI thực nhận và xử lý local:** union/cache giữ metadata còn hợp lệ; ở cùng GPS, nhu cầu và quy tắc xếp hạng, thêm ứng viên không làm giảm Recall. Cache đơn điệu không có nghĩa Recall tăng theo thời gian khi người dùng di chuyển.
3. **Chuyển sang utility thực:** floor trên miền tham chiếu cần điều kiện phản hồi; chuyển từ belief/proxy sang GPS thật cần cận sai số. Cận biến dạng utility cả phiên so với public-base chỉ áp cho cùng chính sách nền phù hợp, không cho baseline bất kỳ.

Kết thúc chương bằng bảng **định lý → giả thiết → mã bảo đảm → phép kiểm tra → phần chưa chứng nhận**. Tách kernel lý tưởng khỏi sampler Geo-I số thực; tách ledger Q khỏi giả thiết khôi phục đầy đủ upstream qua restart. Không gọi các kiểm thử hữu hạn là chứng minh tổng quát.

## 5. Phân vai scenario và bằng chứng thực nghiệm

| Nhóm | Vai trò trong luận văn |
|---|---|
| **S1–S3, S9–S10** | Trọng tâm đánh giá suy luận vị trí/quỹ đạo/đầu cuối, với Hit100, MAE và protocol đúng tác vụ. Nêu rõ S2 trong benchmark mới là proxy tọa độ, không đồng nhất với phát hiện nơi dừng ngữ nghĩa. |
| **S4–S6** | Phân tích giới hạn suy luận có điều kiện từ transcript: liên kết người/phương tiện, cạnh tiếp theo và đích tương lai. Không biến thành một đề tài xây dựng bộ dự đoán identity/route tối ưu; diagnostic cũ chỉ là bằng chứng phụ đúng phạm vi. |
| **S7** | Nhu cầu thật ở local và request chung không thay đổi theo nhu cầu khi giữ cùng bối cảnh/đầu vào. Phân biệt bảo vệ nội dung request trực tiếp với suy luận ý định qua tuyến đường. |
| **S8** | Mục tiêu mở rộng; tóm tắt giới hạn hoặc đưa diagnostic lịch sử vào phụ lục. Không dùng để tuyên bố mô hình chính đã giải quyết bảo vệ nhóm. |

Thứ tự kết quả chương 6: **protocol trước → bảng privacy chính → utility theo bốn purpose → ablation và chi phí → độ ổn định/giới hạn**. Mỗi bảng ghi rõ phiên bản phương pháp, cohort, loại cache và metric.

Giữ các kiểm soát quan trọng: bare Geo-I Z, đối chứng K5 phù hợp, public-base Γ0, initial-only và các ablation liên quan. Giải thích lựa chọn attacker trên TRAIN/SELECTION trước TEST; CI tính theo family, không coi tick hoặc draw là người dùng độc lập. Metrics gốc của paper chỉ tính khi có thể tái lập đúng định nghĩa và output contract; phần không tương thích phải giải thích, không ép về một score chung.

Không gộp các vòng L30/L60 hoặc cohort khác nhau thành một bảng cải thiện trước–sau thiếu kiểm soát. Giữ kết quả âm và Γ0 có Recall tốt hơn cấu hình chính. Phân biệt randomness mới trên GPS đã xem với kiểm thử trên hình học/cohort chưa xem; nêu vai trò của catalogue tĩnh nhỏ và chi phí dữ liệu. Bản continual là thí nghiệm riêng, chưa thay cấu hình chính.

## 6. Thứ tự viết thực tế

Thứ tự viết khác thứ tự đọc, để tránh viết phần mở đầu rồi phải sửa theo kết luận chưa chốt.

| Lượt | Phần cần viết | Điều kiện hoàn thành trước khi chuyển lượt |
|---|---|---|
| **1** | Chốt chương 3 và bảng ký hiệu/giả thiết | Input, output, phiên, attacker, scenario và phạm vi bảo đảm không mâu thuẫn nhau. |
| **2** | Chương 4: phương pháp | Sơ đồ, giả mã, công thức và ví dụ mô tả cùng một cấu hình; giải thích được luồng b → điểm → p → đoạn Q. |
| **3** | Chương 5: chứng minh | Mỗi kết luận có giả thiết và bước nối; tách cận Geo-I/cận Q, privacy/utility và lý tưởng/triển khai. |
| **4** | Chương 6: thực nghiệm | Bảng lấy từ nguồn ghim; so sánh đúng protocol; giải thích CI, control, chi phí và kết quả chưa thuận lợi. |
| **5** | Chương 2: cơ sở và related works | Mỗi nền tảng được dẫn nguồn; chỉ ra khoảng trống thực sự và phần kế thừa. |
| **6** | Chương 1, chương 7 và tóm tắt | RQ, đóng góp, kết luận và các bảng thực nghiệm nhất quán; không hứa nhiều hơn định lý/bằng chứng. |
| **7** | Rà soát toàn bản LaTeX | Không còn cấu hình cũ được gọi là hiện tại; ký hiệu, nguồn, tham chiếu và PDF đã kiểm tra. |

Sau khi duyệt brief, ưu tiên thảo luận sâu **chương 4–5**, rồi đọc benchmark theo protocol ở **chương 6**. Chưa cần mở thêm hướng thuật toán để có thể bắt đầu viết.

## 7. Đóng góp dự kiến và nguồn dùng khi viết

Đóng góp được phát biểu ở mức **đặc tả và thiết kế hệ thống có phân tích toàn transcript, cơ chế chọn đoạn Q có cận riêng, cùng đánh giá tái lập về privacy–utility–cost**. Geo-I, REM, private reuse, hợp thành và hậu xử lý là nền tảng kế thừa; mức độ mới so với công trình gần nhất phải được đối chiếu nguồn, không mặc định là định lý mới chỉ vì kết hợp chúng.

Nguồn chính để tích hợp:

- [Đặc tả GeoI-BQ v1](../docs/research/2026-10-14_main_model.md) và [manifest cấu hình](../configs/geo_i_bq_v1.json): phương pháp chính và ranh giới các thử nghiệm.
- [Cơ sở toán học thống nhất](../docs/research/2026-10-14_geoi_bq_mathematical_foundation.md): định lý và giả thiết; ưu tiên bản này khi tài liệu phát triển cũ khác cách nhóm/ký hiệu.
- [Rà soát chứng minh](../docs/reviews/2026-10-14_mathematical_adversarial_audit.md): điều kiện triển khai còn mở và các phản ví dụ cần giữ.
- [Report privacy rút gọn](../docs/supervisor_meeting/2026-10-17_brief/report_explained.tex) và [kịch bản](../docs/supervisor_meeting/2026-10-17_brief/presentation_script.md): mạch giải thích bốn mệnh đề cho GVHD.
- [Kết quả cấu hình chính](../artifacts/benchmarks/main_depth_20261014_v1/results.json), [protocol](../artifacts/benchmarks/main_depth_20261014_v1/protocol.json) và [kiểm chứng độc lập](../artifacts/benchmarks/main_depth_20261014_v1/independent_validation.json): nguồn bảng thực nghiệm; không chép số từ kết luận của vòng khác.

**Hướng cần duyệt:** giữ bảy chương trên, đặt chứng minh trước benchmark, lấy GeoI-BQ v1 làm trục chính, phân vai scenario như mục 5 và giữ chi tiết phụ ở phụ lục. Đây là kế hoạch viết và trình bày, không phải tuyên bố luận văn đã đạt mọi yêu cầu nghiệm thu hoặc xuất bản.
