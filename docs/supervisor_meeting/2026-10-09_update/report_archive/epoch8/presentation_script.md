# Kịch bản trình bày: Bảo vệ riêng tư truy vấn và quỹ đạo

Theo 6 trang của `report_explained.pdf`, khoảng 8–10 phút. Mỗi trang có lời nói chính, cách chỉ hình/bảng và điểm cần nhớ khi trao đổi. Hai kiến trúc nằm trên hai trang liên tiếp; không cần đọc mọi dòng trong report.

## Trang 1: Cơ chế bảo vệ nội dung và các scenario còn lại

**Lời trình bày**

Thưa thầy/cô, nền tảng của phương pháp vẫn là Geo-I trên mạng đường. Vị trí GPS tạo quan sát được bảo vệ Z; từ lịch sử bảo vệ, mô hình ước lượng phân bố vị trí b và chọn năm tọa độ truy vấn Q. Máy chủ nhận Q, không nhận GPS thật hay Z.

Với nội dung truy vấn, em giữ nhu cầu thật tại thiết bị. Mỗi Q lấy L30 cho mọi loại POI công khai. Sau khi nhận phản hồi, thiết bị gộp, bỏ trùng rồi dùng GPS cùng nhu cầu riêng để lọc và sắp xếp. Năm Q cho tối đa 150 bản ghi mỗi loại trước bỏ trùng. Cùng trạng thái bảo vệ và lịch công khai, đổi nhu cầu không làm đổi request mạng.

Độ gần giúp thu ứng viên cho dịch vụ lân cận; người dùng vẫn có thể chọn theo thời gian, bán kính hoặc độ đi vòng. Câu trả lời phụ thuộc độ phủ ứng viên.

S4 được hỗ trợ bằng giới hạn thông tin tích lũy. S5 và S6 được hỗ trợ bằng lịch sử bảo vệ và không dùng tương lai thật để tạo Q. Các thành phần này chưa đủ để khẳng định bảo vệ danh tính hoặc chống mọi dự đoán tương lai. S8 vẫn là hướng mở.

**Cách chỉ nội dung**

Chỉ luồng 5 Q × L30 → nhận → bỏ trùng → lọc/sắp xếp local; sau đó chỉ bốn tiêu chí và ba hàng S4–S6. Nhấn mạnh yêu cầu chung không chứa nhu cầu ψ.

**Khi trao đổi**

- ψ đọc là “psi”, biểu diễn nhu cầu riêng, gồm loại POI và tiêu chí/bán kính/đích liên quan.
- 150 là trần bản ghi mỗi loại, không phải 150 POI duy nhất toàn bộ phản hồi. Có sáu loại nên trần tổng là 900.
- S7 không cần điều kiện top-5; triển khai và benchmark hiện tại vẫn dùng k = 5. Không tuyên bố mã đã trả danh sách vô hạn.
- Hình dạng tuyến đường, click, account/IP hoặc việc đổi hành trình có thể giúp suy luận ý định; tính bất biến chỉ xét cùng trạng thái bảo vệ, lịch và ngữ cảnh công khai.
- S5 là dự đoán cạnh đường tiếp theo, S6 là dự đoán đích chưa tới. Chỉ dùng prefix tránh lấy trực tiếp tương lai làm đầu vào; không tự loại được tương quan hành trình.

## Trang 2: Cap tám phiên và lập luận riêng tư

**Lời trình bày**

Geo-I bảo vệ từng quan sát, nhưng quan sát nhiều lần vẫn tích lũy thông tin. Trước đây, mỗi phiên có cap 0,23 trên mét. Nếu cho tám phiên độc lập cùng mức đó, mức hợp thành có thể lên 1,84. Em bổ sung cap chung 0,23 cho một epoch hữu hạn gồm tối đa tám phiên, nên mỗi phiên được cấp 0,02875.

Mỗi phiên dùng 23 đơn vị, mỗi đơn vị bằng 0,00125. Lần tạo Z đầu tiên tốn một đơn vị; lần đọc rồi giữ Z tốn một đơn vị cho phép thử; thử không đạt rồi tạo Z mới tốn hai đơn vị. Không đọc GPS mới thì không chi thêm. Lịch và dự toán được kiểm tra trước khi đọc GPS.

Về toán học, REM tạo phân bố trên miền đường cố định. Kiểm tra giữ Z dùng nhiễu Laplace. Chi phí của từng nhánh được cộng lại và bị chặn bởi ngân sách epoch. Các bước tạo Q từ thông tin đã bảo vệ là xử lý tiếp, nên không cộng thêm chi phí vị trí theo lý thuyết này.

Đây là chặn tích lũy thông tin tọa độ trong một epoch, chưa phải bảo đảm ẩn danh hoặc ngân sách vô hạn cả đời.

**Cách chỉ nội dung**

Đi từ bảng cap cũ/mới tới phép chia 0,23/8, rồi bảng chi phí 0/1/2 đơn vị. Chỉ bất đẳng thức cuối để giải thích vì sao tổng chi phí không vượt cap.

**Khi trao đổi**

- Phiên là một chuyến; epoch là nhóm tối đa tám phiên có cấp ngân sách theo lịch công khai. H = 12 dẫn tới U = 2H − 1 = 23 đơn vị, không có nghĩa chỉ được 12 lần đọc GPS trong mọi nhánh.
- C_epoch là ngân sách cả nhóm phiên; C_phiên là phần được cấp cho một phiên. Mức thực cấp 0,02875 khác cap danh nghĩa 0,03.
- Ledger lưu phần đã dành/chi qua các phiên, đặt dự toán trước hành động riêng và không hoàn lại theo nhánh bí mật. Phiên đang được nhận nhưng hết ngân sách còn có thể dự đoán và gửi Q; quá tám slot thì từ chối phiên mới trước GPS và Q.
- Theorem xét kernel và randomness lý tưởng, miền đường/ngữ cảnh công khai cố định, thiết bị tin cậy. GPS/nhu cầu local không được làm phát sinh request riêng. Chưa có chứng nhận tương đương cho sampler dấu phẩy động.
- Cap không tự che IP/account, giờ mở/đóng hay giải quyết reset ngân sách cả đời. Thêm epoch khác cần cộng ngân sách, không mặc định bắt đầu lại miễn phí.

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

## Trang 4: Kiến trúc hiện tại: cap liên phiên và nhu cầu local

**Lời trình bày**

Ở kiến trúc hiện tại, em giữ cùng bốn tầng và cùng nền Geo-I/REM. Ba phần viền xanh là những thay đổi chính.

Thứ nhất, trước GPS có kiểm tra cap chung tám phiên, để hạn chế thông tin tích lũy qua nhiều chuyến. Thứ hai, độ sâu phản hồi tăng từ L10 lịch sử lên L30 để thu tập ứng viên rộng hơn. Máy chủ vẫn chỉ nhận Q và yêu cầu chung; mô hình hiện tại gửi ngay.

Thứ ba, xử lý local mở rộng từ chọn gần nhất sang bốn nhu cầu. Nhu cầu thật ψ và GPS local chỉ đi vào bước lọc, sắp xếp sau khi nhận POI. Đổi nhu cầu dùng lại cùng tập phản hồi, không làm thay đổi Q hoặc tạo query riêng.

L_plan = 10 ở bước chọn Q và L = 30 ở bước truy hồi là hai tham số khác nhau. Slack 0,03 là dung sai trên điểm độ phủ của bộ chọn Q; không phải bảo đảm chỉ mất ba phần trăm Recall. Tăng ứng viên và kiểm soát ngân sách giúp hai mặt utility và privacy, nhưng mỗi mặt vẫn cần kiểm chứng riêng.

**Cách chỉ hình**

Đối chiếu bước 1, 5, 6 với trang trước; chỉ rõ mũi tên ψ/GPS chỉ vào bước 6. Không cần đọc lại những bước Geo-I, b và chọn Q đã giữ nguyên.

**Khi trao đổi**

- Không đổi nền tảng sang phương pháp khác. Quan sát Z nội bộ, dự đoán b và planner Q theo mạng đường vẫn giữ.
- L_plan là bảng POI công khai để chấm điểm khi chọn Q; L30 là số POI mỗi loại máy chủ trả tại mỗi Q.
- S9/S10 có bằng chứng từ nhánh Endpoint20 riêng: tăng nhiễu, không delay. Chưa nhập kết quả đó thành privacy đã xác nhận cho Epoch8/L30.
- “Danh sách đã lọc/sắp xếp” mô tả cơ chế S7; cấu hình triển khai/đánh giá hiện tại vẫn cắt k = 5.

## Trang 5: Ví dụ đầy đủ: từ GPS đến danh sách POI

**Lời trình bày**

Em dùng một hành trình đã lưu để minh họa, không chọn nó làm bằng chứng benchmark. Ở giây 0, mô hình tạo Z đầu tiên, cập nhật b và chi một đơn vị. Ở giây 20, chưa đủ lịch đọc GPS; giữ Z, chỉ dự đoán b và không chi thêm. Q vẫn có thể di chuyển theo lịch sử bảo vệ và mạng đường.

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

Kết quả rõ nhất hiện tại là phép so L20–L30 giữ nguyên Q, Z, lịch và cap trên 24 nhóm mới. Recall trung bình đều bốn mục đích tăng từ 89,71 lên 92,69 phần trăm: tăng 2,97 điểm phần trăm, khoảng tin cậy 2,31 đến 3,65. Cả ba lượt nhiễu đều tăng. Đánh đổi là byte phản hồi tăng 31,10 phần trăm, số request giữ nguyên.

Recall 95,44 phần trăm trong báo cáo trước là tìm gần nhất với cấu hình và cohort khác. Không lấy chênh số đó với macro bốn nhu cầu để kết luận tốt hơn hoặc kém hơn.

Về privacy, S4 cho thấy giảm khả năng liên kết trong diagnostic lịch sử, nhưng vẫn còn nhóm bị liên kết tốt. Pilot S5/S6 giảm dự đoán đúng từ 100 xuống 50 phần trăm, song mới là bài toán hai ứng viên. Endpoint20 tăng sai số suy vị trí đầu/cuối với một ít giảm utility; kết quả chưa thắng mọi metric.

Em kết luận đã xác nhận lợi ích utility của L30 và bổ sung bằng chứng privacy theo từng cấu hình. Bước tiếp theo là xác nhận privacy trên cùng Epoch8/L30, mở rộng attacker và đo chi phí triển khai.

**Cách chỉ bảng**

Chỉ hàng macro, CI và byte ở bảng utility. Ở bảng privacy, đọc từng metric cùng cột cấu hình; tránh trình bày mọi hàng như kết quả của một mô hình thống nhất.

**Khi trao đổi**

- Recall@5 càng cao càng tốt cho dịch vụ; Hit100 càng thấp càng tốt cho privacy; MAE càng cao càng tốt cho privacy. AUC dưới 0,5 có thể đảo điểm, không tự chứng minh ẩn danh.
- Attacker chọn theo task/metric trên tập chọn. S4 dùng shape/kNN hoặc trees; S5/S6 có Candidate Trees, Motion, Curve Mean. Không phải tất cả cùng Shadow kNN.
- CI bootstrap theo nhóm, không theo từng tick. Chỉ đo kích thước reply JSON, chưa tính HTTP/TLS, latency hoặc năng lượng.
- S4: ba nhóm/45 cặp, AUC người từng nhóm còn tới 0,861. S5/S6: sáu nhóm/12 query, Planar cũng đạt 50% và có Recall cao hơn REM. Không kết luận REM vượt trội trên mọi đối chứng.
- Endpoint20: 28 nhóm đã khảo sát, 112 quan sát/task, u = 0,0025/m so với 0,01/m lịch sử. Recall 98,94 → 96,64%; S10 Hit100 cùng 0%, CI chênh Hit500 chạm 0. Không gọi u này là một phần tư u = 0,00125/m hiện tại.
- S1–S3 và đối chứng paper ở tài liệu chi tiết; S8 còn mở. Không tuyên bố fully giải quyết S1–S10.

## Nguồn đối chiếu

- `report_explained.tex`, `report_tables.tex`: nội dung và bảng của report.
- `benchmark_tables.json`, `benchmark_evidence.json`: kết quả, giao thức và cấu hình attacker.
- `multistep_sample.json`, `utility_sample.json`: timeline, GPS và phản hồi POI đã lưu.
- `endpoint_focus.json`: delay lịch sử và ranh giới phiên.
- `thesis/current_method.tex`, `thesis/current_formal_privacy.tex`, `thesis/current_formal_service.tex`: mô hình và lập luận toán học.
- `docs/supervisor_meeting/2026-09-26_brief/method_evidence.json`: kết quả bối cảnh lịch sử.
- `docs/research/2026-10-09_multi_purpose_retrieval.md`: replay bốn template L10 chưa được áp dụng.
- `report_sources.json`: hash nguồn của bản report; các nguồn này không được đánh giá lại khi biên soạn.

Bản kịch bản của 8 slide trước được giữ riêng tại `slides_presentation_script.md`; ghi chú nguồn tương ứng là `slides_speaker_notes.txt`.
