# Kịch bản trình bày: Bảo vệ riêng tư truy vấn và quỹ đạo

Theo 6 trang của `report_explained.pdf`, khoảng 8–10 phút. Mỗi trang có lời nói chính, cách chỉ hình/bảng và điểm cần nhớ khi trao đổi. Hai kiến trúc nằm trên hai trang liên tiếp; không cần đọc mọi dòng trong report.

## Trang 1: Cơ chế bảo vệ nội dung và các scenario còn lại

**Lời trình bày**

Thưa thầy/cô, nền tảng của phương pháp vẫn là Geo-I trên mạng đường. Vị trí GPS tạo quan sát được bảo vệ Z; từ lịch sử bảo vệ, mô hình ước lượng phân bố vị trí b và chọn năm tọa độ truy vấn Q. Máy chủ nhận Q, không nhận GPS thật hay Z.

Với nội dung truy vấn, em giữ nhu cầu thật tại thiết bị. Mỗi Q lấy L30 cho mọi loại POI công khai. Sau khi nhận phản hồi, thiết bị gộp, bỏ trùng rồi dùng GPS cùng nhu cầu riêng để lọc và sắp xếp. Năm Q cho tối đa 150 bản ghi mỗi loại trước bỏ trùng. Cùng trạng thái bảo vệ và lịch công khai, đổi nhu cầu không làm đổi request mạng.

Độ gần giúp thu ứng viên cho dịch vụ lân cận; người dùng vẫn có thể chọn theo thời gian, bán kính hoặc độ đi vòng. Câu trả lời phụ thuộc độ phủ ứng viên.

S4 có hai target: danh tính người và phương tiện vật lý. Em tách nó thành lớp bảo vệ chuyên biệt đang thiết kế. S5 và S6 được hỗ trợ bằng lịch sử bảo vệ và không dùng tương lai thật để tạo Q. Các thành phần này chưa đủ để khẳng định bảo vệ danh tính hoặc chống mọi dự đoán tương lai. S8 vẫn là hướng mở.

**Cách chỉ nội dung**

Chỉ luồng 5 Q × L30 → nhận → bỏ trùng → lọc/sắp xếp local; sau đó chỉ bốn tiêu chí và ba hàng S4–S6. Nhấn mạnh yêu cầu chung không chứa nhu cầu ψ.

**Khi trao đổi**

- ψ đọc là “psi”, biểu diễn nhu cầu riêng, gồm loại POI và tiêu chí/bán kính/đích liên quan.
- 150 là trần bản ghi mỗi loại, không phải 150 POI duy nhất toàn bộ phản hồi. Có sáu loại nên trần tổng là 900.
- S7 không cần điều kiện top-5; triển khai và benchmark hiện tại vẫn dùng k = 5. Không tuyên bố mã đã trả danh sách vô hạn.
- Hình dạng tuyến đường, click, account/IP hoặc việc đổi hành trình có thể giúp suy luận ý định; tính bất biến chỉ xét cùng trạng thái bảo vệ, lịch và ngữ cảnh công khai.
- S5 là dự đoán cạnh đường tiếp theo, S6 là dự đoán đích chưa tới. Chỉ dùng prefix tránh lấy trực tiếp tương lai làm đầu vào; không tự loại được tương quan hành trình.

## Trang 2: Cap từng phiên; bảo vệ danh tính bằng lớp riêng

**Lời trình bày**

Em giữ cap tọa độ theo từng phiên, thay vì chia cho tám chuyến. Một phiên là một chuyến/lượt sử dụng, có nhiều lần đọc GPS và gửi Q. Cap hiệu lực là 0,23 trên mét; đây là hệ số riêng tư, không phải bán kính nhiễu.

Mỗi phiên có 23 đơn vị, mỗi đơn vị bằng 0,01. Đọc lần đầu tốn một đơn vị. Đọc rồi giữ Z tốn một đơn vị phép thử; thử rồi tạo mới tốn hai. Kiểm tra lịch và dự toán trước GPS. Hết cap đọc thì dự đoán từ lịch sử bảo vệ; không từ chối chuyến thứ chín vì một giới hạn tám phiên.

Về toán học, REM và phép thử tái sử dụng cho chặn của từng phiên. Nếu attacker nối nhiều chuyến, ngân sách vẫn cộng lại, không được gọi cả lịch sử có cùng cap 0,23.

S4 cần lớp riêng: không gửi định danh bền của người, xe hoặc thiết bị; tách nguồn IP khỏi nội dung truy vấn bằng relay độc lập. Tuy vậy, hình dạng tuyến đường còn có thể nối chuyến. Em sẽ kiểm chứng hai target riêng, gồm một người đổi xe và hai người dùng chung xe. Đây là thiết kế, chưa có kết quả triển khai relay hoặc chống liên kết mới.

**Cách chỉ nội dung**

Chỉ bảng chi phí và công thức cap phiên. Chỉ bảng bốn cặp người/xe để giải thích tại sao hai nhãn không được đồng nhất.

**Khi trao đổi**

- Với cap C_s và H công khai: U = 2H − 1; u = C_s/U; B danh nghĩa = 2H u. Cấu hình H12 cho u = 0,01, B = 0,24, cap hiệu lực 0,23 m⁻¹. H không là hard maximum 12 GPS reads.
- Tắt/mở lại để tiếp tục cùng chuyến không được tự nạp cap. Chuyến mới có cap mới, nhưng tuyên bố cho lịch sử phải hợp thành tổng các cap.
- Cap tọa độ không tự chống nhận diện. Đổi mã/shuffle cũng không đủ chống home/work, timing hoặc routine.
- OHTTP cần relay/gateway độc lập, HTTPS và không thông đồng; payload không chứa account/cookie/VIN/API key cá nhân. Đây là thành phần đề xuất chưa triển khai.
- Nhãn người và phương tiện vật lý khác ID thiết bị. Evaluator đã dùng `person_id` và `physical_vehicle_id`; phải kiểm tra hai target trên cặp đổi xe/chung xe, không chỉ AUC gộp.
- Chặn lý tưởng giả định kernel/ngữ cảnh công khai và thiết bị tin cậy. Không chứng nhận sampler float hay anonymous transport.

## Trang 3: Kiến trúc trước: cap từng phiên và tìm POI gần nhất

**Lời trình bày**

Em trình bày kiến trúc trước để làm mốc so sánh. Khung xanh là phần chạy tại thiết bị; máy chủ POI ở bên ngoài. Luồng chính đi từ trên xuống qua bốn tầng.

Tầng đầu kiểm tra lịch và cap từng phiên, rồi đọc GPS khi được phép. Phép thử có nhiễu quyết định giữ Z hay tạo Z mới bằng Geo-I/REM. Tầng hai cập nhật phân bố b từ quan sát được bảo vệ, rồi chọn năm Q theo mạng đường và độ phủ POI. Khi chưa đến lúc đọc GPS, mô hình đi theo nhánh dự đoán b.

Tầng ba gửi truy vấn chung với L10. Tầng cuối nhận phản hồi và xếp hạng gần nhất bằng GPS local. GPS và Z không gửi lên máy chủ.

Bản trước còn có bảo vệ đầu/cuối tùy chọn: chờ đầu chuyến, delay công bố và hủy Q đang chờ khi đóng phiên. Cách đó che bớt các mốc đầu/cuối nhưng làm chậm phản hồi. Đây là nhánh lịch sử, không phải cách gửi của mô hình hiện tại.

**Cách chỉ hình**

Chỉ khung mô hình, đầu vào trên cùng và đầu ra dưới cùng. Đi theo bước 1–6; sau đó chỉ đường rẽ “không đọc GPS mới” từ bước 1 tới bước 3 và phản hồi từ máy chủ về bước 6.

**Khi trao đổi**

- Geo-I/REM là cơ chế tạo quan sát bảo vệ; b là ước lượng phục vụ chọn Q, không phải posterior attacker đã hiệu chuẩn.
- Q là tọa độ truy vấn được chọn từ lịch sử bảo vệ, không phải năm mẫu nhiễu REM độc lập quanh Z. P là POI trong danh mục.
- Delay 60 s là tùy chọn lịch sử. Ví dụ Q tại 60 s công bố ở 120 s; khi đóng tại 384 s, hủy Q còn chờ ở 340/360/380/384 s. Sơ đồ không áp delay cho mọi cấu hình cũ.

## Trang 4: Kiến trúc làm việc: cap phiên và lớp S4 riêng

**Lời trình bày**

Em giữ Geo-I/REM, tái sử dụng Z, ước lượng b và chọn Q theo mạng đường. Thay đổi chính sách ngân sách là quay về cap từng chuyến; không còn giới hạn cấp tám chuyến trong thiết kế được chọn.

L30 và xử lý nhu cầu local vẫn được giữ. Máy chủ nhận tọa độ Q và yêu cầu chung; GPS thật, Z và nhu cầu riêng không gửi ra. L_plan bằng 10 là bảng POI cho bộ chọn Q, khác L30 của phản hồi máy chủ.

Khối nét đứt là lớp S4 đề xuất cho cả người và xe. Nó xử lý định danh và nguồn mạng, còn Geo-I xử lý tọa độ. Lớp này chưa có triển khai OHTTP và chưa có benchmark chống nối quỹ đạo. Nếu attacker vẫn nối được người hoặc xe từ hình dạng, ta chưa thể gọi target đó đã được bảo vệ.

Các kết quả Epoch8/L30 trước đây không chuyển nguyên sang cấu hình cap từng phiên. Đổi u làm đổi phân bố Z và Q. Em giữ số liệu cũ để đối chiếu và cần chạy cấu hình mới, cùng các ablation S4, trên dữ liệu chưa dùng để chọn thiết kế.

**Cách chỉ hình**

Chỉ cap ở bước 1, L30 ở bước 5 và nhu cầu local ở bước 6. Chỉ khối S4 nét đứt, phân biệt đề xuất với các bước nền đã chạy.

**Khi trao đổi**

- Cấu hình chọn: cap hiệu lực 0,23 m⁻¹/phiên, u = 0,01, K5, L30, L_plan10, slack0,03. Không có benchmark mới cho toàn bộ tổ hợp này.
- Lớp S4 phải loại person/vehicle/device ID, định danh tài khoản và nguồn mạng; không gửi hai bí danh người/xe riêng mà vô tình cung cấp thêm khóa liên kết.
- Relay không tự che nội dung tự nhận diện hoặc hình dạng/timing. Chỉ đổi pseudonym trên tuyến liên tục không bảo đảm unlinkability.
- Endpoint20 là nhánh L20 riêng. Chưa có xác nhận đầu/cuối cho tổ hợp mới và chưa nhận các số đo của nhánh đó làm kết quả mới.

## Trang 5: Ví dụ cơ chế đã lưu: từ GPS đến danh sách POI

**Lời trình bày**

Em dùng một hành trình Epoch8 cũ đã lưu để minh họa cơ chế, không xem là kết quả của cap phiên mới. Ở giây 0, mô hình tạo Z đầu tiên, cập nhật b và chi một đơn vị. Ở giây 20, chưa đủ lịch đọc GPS; giữ Z, chỉ dự đoán b và không chi thêm. Q vẫn có thể di chuyển theo lịch sử bảo vệ và mạng đường.

Ở giây 60, đủ lịch và ngân sách để đọc. Khoảng cách GPS tới Z cũ khoảng ba kilômét; phép thử có nhiễu không đạt nên tạo Z mới. Lần này chi hai đơn vị, tổng ba trên hai mươi ba. Từ quan sát mới, mô hình cập nhật b, chọn năm Q và truy hồi L30.

Máy chủ trả 420 bản ghi; sau bỏ trùng còn 252 POI. Với cùng phản hồi, tìm gần nhất, trong bán kính hoặc ít đi vòng cho thứ tự khác nhau. Không gửi nhu cầu thật lên máy chủ.

Tới giây 600 đã chi 21 đơn vị, gồm cả các lần đọc không hiện trên bảng. Mẫu chưa hết cap. Nhánh đọc rồi giữ Z được minh họa riêng ở phiên sáu, khác với không đọc GPS ở giây 20.

**Cách chỉ hình/bảng**

Đi qua timeline 0 → 20 → 60 → 120 → 600 s. Chỉ dấu GPS, Z và năm Q trên bản đồ tại 60 s, rồi sáu bước xử lý và bảng ba nhu cầu quán cà phê.

**Khi trao đổi**

- Threshold 200 m; scale Laplace 1/u = 800 m. Log lưu kết quả thử, không lưu nhiễu cụ thể; không lấy 3.049,59 > 200 làm điều kiện tất định.
- Tại 60 s: chi thêm 0,0025/m; tổng 0,00375/m; còn 0,025/m trong phiên. Đã dự toán đủ hai đơn vị trước GPS.
- Các POI trong bảng là phần đầu kết quả k = 5 đã lưu, không dựng một danh sách đầy đủ mới. Màu cam minh họa trọng số b, không khẳng định độ hiệu chuẩn.
- GPS là tổng hợp SUMO. Utility dùng vị trí thật local tại mốc 20 s như oracle; số lần đọc bảo vệ theo lịch 60 s không tương đương chi phí GNSS hoặc năng lượng thực.
- Đầu/cuối hiện gửi ngay; giờ mở/đóng còn có thể lộ. Không có component dùng tương lai thật để biết trước phần cuối.

## Trang 6: Benchmark: cải thiện đã đo và phạm vi kết luận

**Lời trình bày**

Kết quả đã xác nhận trong study Epoch8 trước là phép so L20–L30 giữ nguyên Q, Z, lịch và cap trên 24 nhóm mới. Recall trung bình đều bốn mục đích tăng từ 89,71 lên 92,69 phần trăm: tăng 2,97 điểm phần trăm, khoảng tin cậy 2,31 đến 3,65. Cả ba lượt nhiễu đều tăng. Đánh đổi là byte phản hồi tăng 31,10 phần trăm, số request giữ nguyên.

Recall 95,44 phần trăm trong báo cáo trước là tìm gần nhất với cấu hình và cohort khác. Không lấy chênh số đó với macro bốn nhu cầu để kết luận tốt hơn hoặc kém hơn.

Về privacy, S4 cho thấy giảm khả năng liên kết trong diagnostic lịch sử, nhưng vẫn còn nhóm bị liên kết tốt. Pilot S5/S6 giảm dự đoán đúng từ 100 xuống 50 phần trăm, song mới là bài toán hai ứng viên. Endpoint20 tăng sai số suy vị trí đầu/cuối với một ít giảm utility; kết quả chưa thắng mọi metric.

Em kết luận đã xác nhận lợi ích utility của L30 và bổ sung bằng chứng privacy theo từng cấu hình. Bước tiếp theo là chạy per-session L30 và kiểm chứng lớp S4 mới với hai target người/xe, cùng chi phí triển khai.

**Cách chỉ bảng**

Chỉ hàng macro, CI và byte ở bảng utility. Ở bảng privacy, đọc từng metric cùng cột cấu hình; tránh trình bày mọi hàng như kết quả của một mô hình thống nhất.

**Khi trao đổi**

- Recall@5 càng cao càng tốt cho dịch vụ; Hit100 càng thấp càng tốt cho privacy; MAE càng cao càng tốt cho privacy. AUC dưới 0,5 có thể đảo điểm, không tự chứng minh ẩn danh.
- Attacker chọn theo task/metric trên tập chọn. S4 dùng shape/kNN hoặc trees; S5/S6 có Candidate Trees, Motion, Curve Mean. Không phải tất cả cùng Shadow kNN.
- CI bootstrap theo nhóm, không theo từng tick. Chỉ đo kích thước reply JSON, chưa tính HTTP/TLS, latency hoặc năng lượng.
- S4: ba nhóm/45 cặp, AUC người từng nhóm còn tới 0,861. S5/S6: sáu nhóm/12 query, Planar cũng đạt 50% và có Recall cao hơn REM. Không kết luận REM vượt trội trên mọi đối chứng.
- Endpoint20: 28 nhóm đã khảo sát, 112 quan sát/task, u = 0,0025/m so với 0,01/m lịch sử. Recall 98,94 → 96,64%; S10 Hit100 cùng 0%, CI chênh Hit500 chạm 0. Không gán nhánh L20 này cho cap phiên/L30 và lớp S4 mới.
- S1–S3 và đối chứng paper ở tài liệu chi tiết; S8 còn mở. Không tuyên bố fully giải quyết S1–S10.

## Nguồn đối chiếu

- `report_explained.tex`, `report_tables.tex`: nội dung và bảng của report.
- `benchmark_tables.json`, `benchmark_evidence.json`: kết quả, giao thức và cấu hình attacker.
- `multistep_sample.json`, `utility_sample.json`: timeline, GPS và phản hồi POI đã lưu.
- `endpoint_focus.json`: delay lịch sử và ranh giới phiên.
- `thesis/current_method.tex`, `thesis/current_formal_privacy.tex`, `thesis/current_formal_service.tex`: mô hình và lập luận toán học.
- `docs/supervisor_meeting/2026-09-26_brief/method_evidence.json`: kết quả bối cảnh lịch sử.
- `docs/research/2026-10-09_multi_purpose_retrieval.md`: replay bốn template L10 chưa được áp dụng.
- `docs/research/2026-10-10_session_cap_identity.md` và `.json`: chính sách được chọn, lớp identity đề xuất và điều kiện đánh giá.
- `report_sources.json`: hash nguồn của bản report; các nguồn này không được đánh giá lại khi biên soạn.

Bản kịch bản của 8 slide trước được giữ riêng tại `slides_presentation_script.md`; ghi chú nguồn tương ứng là `slides_speaker_notes.txt`.

Bản report Epoch8 trước được giữ tại `report_archive/epoch8/`. Các số đo và sample Epoch8 vẫn giữ nguyên cấu hình gốc; không có kết quả mới cho cap phiên/L30 hoặc relay.
