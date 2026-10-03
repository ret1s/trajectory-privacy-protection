# Hai tài liệu cho buổi trình bày

- [Report](report_explained.pdf): **6 trang**, gồm **5 trang chính** và 1 trang nguồn.
- [Preparation guide](preparation_guide.pdf): **5 trang**. Tạm bỏ phụ lục mẫu; trình bày ở mức scenario.

| Nội dung | Report | Guide |
|---|---|---|
| Related works: bảng chung scenario–metrics; lý do dùng bộ đo chung | 1 | 1 |
| Sơ đồ luồng kiến trúc theo số thứ tự và layer | 2 | 2 |
| Vai trò component, cấu hình GeoI-Paced/GeoI-Slack và đóng góp | 3 | 3 |
| Attacker theo scenario; Geo-I v2 so với adapter | 4 | 4 |
| Geo-I hiện tại: kết quả gộp theo scenario | 5 | 5 |
| Nguồn và truy nguyên | 6 | Đọc cùng report |

**Cách đọc ví dụ S1:** Hit100 = 33,33% là tỷ lệ attacker đoán cách vị trí thật không quá 100 m; MAE = 299 m là sai số suy luận trung bình. Thấp hơn tốt hơn với Hit100; cao hơn tốt hơn với MAE. Bảng 3.1 ghi Hit100 (%) / MAE (m) / Recall@5 (%); ví dụ 33,33% / 299 / 95,83%. Dấu / tách ba chỉ số, không phải phép chia. Bảng bản hiện tại gộp theo scenario và giữ cột riêng.

Mạch nói: **paper nào bao phủ nhiệm vụ nào → vì sao cần bộ đo chung → mô hình Geo-I hoạt động thế nào → attacker được thấy gì và suy gì → kết quả benchmark**.

**“v2” là phiên bản benchmark, không phải tên attacker.** Cùng tập loại attacker đánh giá các phương pháp; **Shadow kNN** được học riêng từ output từng phương pháp. Chọn bộ suy luận trên tập chọn, giữ cố định khi test. Tên các bộ suy luận đã in đậm ở trang 4–5; phép thử bản hiện tại dùng thêm **kNN**, **ExtraTrees** và các bộ hình học/đường.

Có thể trình bày phần kết quả như sau: “Khung nghiên cứu có S1–S10; hiện đo S1, S2, S3, S9, S10. Ở benchmark v2, BR trên nền Geo-I giảm Hit100 S2/S3 so với ba adapter, nhưng utility còn đánh đổi. Bản hiện tại đạt Recall 95,44%, cả năm scenario đạt 90% ở mức trung bình. S1 vẫn có mức thấp nhất 80,18%. Cache có lợi ích đo được; slack còn cần xác nhận. Hai phiên bản khác protocol nên báo riêng.”

Hai cấu hình hiện tại cùng ε/ngân sách: GeoI-Paced là bản so sánh nội bộ; GeoI-Slack cho phép nới điểm đánh giá độ phủ POI tối đa 0,03 để chọn điểm giúp tiếp tục di chuyển. Đây không phải mức giảm Recall 3%. Geo-I tạo vị trí tham chiếu đã làm nhiễu, chỉ dùng trong thiết bị. Ước lượng vùng vị trí, kiểm tra đường đi và chọn điểm truy vấn dùng lịch sử đã bảo vệ cùng dữ liệu công khai. GPS thật còn dùng để xếp hạng kết quả cuối tại thiết bị. Việc hợp phản hồi không sửa truy vấn. Che đầu/cuối chuyến là mở rộng S9/S10: đã kiểm tra tích hợp, chưa benchmark kết hợp với dịch vụ hiện tại.

**Cách nói theo sơ đồ:** bước 1 kiểm tra lịch/ngân sách; bước 2 giữ hoặc tạo vị trí đã bảo vệ; 3a ước lượng vùng vị trí và 3b kiểm tra điểm xe có thể tới; bước 4 chọn 5 điểm truy vấn; máy chủ trả POI; bước 5 hợp phản hồi và xếp hạng trên thiết bị. Không đọc GPS mới thì bỏ bước 2, dựa vào lịch sử đã bảo vệ. S9 nằm trước các bước tạo truy vấn; S10 nằm trước lúc gửi máy chủ.

Phần 1 gộp coverage và metrics vào một bảng. Bốn lý do cần bộ đo chung: khác đầu ra; khác nhiệm vụ/quyền quan sát; utility khác tác vụ; chi phí khác phạm vi. Ta chấm cùng đáp án từ toàn bộ output attacker được phép thấy bằng Hit100/MAE, cùng dịch vụ POI bằng Recall@5 và request + response bằng byte/sự kiện.

Mỗi dòng của benchmark hiện tại là trung bình đều các điều kiện trong scenario. Recall tổng trung bình đều năm scenario. Không coi đạt ngưỡng trung bình là mọi điều kiện đều đạt; số đo chi tiết vẫn giữ trong nguồn để truy nguyên.

Bảng coverage, định lượng và benchmark được dùng chung để hai bản không lệch số. Sửa lời dẫn guide trong `preparation_guide.tex`; sửa nội dung chung trong `geoi_content.py`; dựng report trước guide. Lệnh và provenance ở [README](README.md).

## Kịch bản giải thích thuật toán với GVHD

Khoảng 4 phút; chỉ từng ô trên sơ đồ trang 2. Khi nói “vị trí đã bảo vệ”, chỉ ô 2; khi nói “điểm truy vấn”, chỉ ô 4 và đầu ra công khai.

“Mục tiêu của mô hình là giúp người dùng tìm địa điểm quan tâm, chẳng hạn trạm sạc, nhưng không gửi GPS thật lên máy chủ. Đầu vào gồm GPS trên thiết bị, bản đồ đường và danh sách địa điểm công khai. Đầu ra gửi máy chủ là năm điểm truy vấn giả; kết quả cuối cho người dùng là năm địa điểm mỗi loại, được chọn ngay trên thiết bị.

Ở bước 1, mô hình kiểm tra đã đến lịch đọc GPS và còn ngân sách riêng tư hay chưa. Hai lần đọc để tạo truy vấn cách nhau ít nhất 60 giây. Nếu chưa được đọc, mô hình tiếp tục dùng lịch sử đã bảo vệ. Bước này giúp hạn chế thông tin tích lũy khi người dùng bị quan sát nhiều lần.

Ở bước 2, Geo-I làm nhiễu GPS để tạo một vị trí tham chiếu đã bảo vệ. Một phép kiểm tra có nhiễu quyết định giữ vị trí tham chiếu cũ hay tạo vị trí mới. Việc tái dùng giúp giảm số mẫu nhiễu mới khi người dùng ít di chuyển. Tuy nhiên, phép kiểm tra vẫn tiêu ngân sách. Vị trí tham chiếu này chỉ dùng trong thiết bị, chưa phải điểm gửi lên máy chủ.

Tiếp theo là hai phần cùng hỗ trợ bước chọn truy vấn. Bước 3a dùng lịch sử đã bảo vệ để ước lượng vùng người dùng có thể đang ở, vì một tọa độ đã làm nhiễu có thể lệch khỏi vị trí thật. Bước 3b dùng điểm giả trước đó, thời gian đã trôi qua, hướng làn và luật rẽ để kiểm tra xe có thể tới đâu. Như vậy, điểm truy vấn được chọn vừa xét sự không chắc chắn về vị trí, vừa xét tính hợp lý của đường đi.

Ở bước 4, mô hình chọn năm điểm giả sao cho các điểm có thể mang về những địa điểm bổ sung nhau. Nếu nhiều điểm cùng trả về một danh sách giống nhau thì lợi ích tìm kiếm sẽ thấp. GeoI-Paced ưu tiên độ phủ hiện tại. GeoI-Slack cho phép nới điểm đánh giá độ phủ tối đa 0,03 để chọn điểm giúp tiếp tục di chuyển tới vùng hữu ích hơn. Hai bản dùng cùng ngân sách; 0,03 không có nghĩa là giảm Recall 3%.

Máy chủ trả tối đa mười địa điểm mỗi loại cho từng điểm truy vấn. Bước 5 hợp các phản hồi còn hiệu lực trong khoảng 60 giây, rồi dùng GPS thật ngay trên thiết bị để chọn năm địa điểm mỗi loại. GPS thật không được gửi trong các truy vấn này.

Với S1, Geo-I là nền bảo vệ vị trí. Với S2, tái dùng, lịch đọc và ngân sách xử lý quan sát lặp. Với S3, kiểm tra đường đi giúp chuỗi điểm giả phù hợp chuyển động; tác động riêng tư vẫn phải đo bằng attacker. Nếu bật S9, GPS của đoạn đầu không đi vào các bước tạo truy vấn. Nếu bật S10, kết quả được giữ tạm trước khi gửi và phần chưa gửi bị hủy khi chuyến kết thúc. Hai phần mở rộng này chưa được benchmark kết hợp với dịch vụ hiện tại.

Vì vậy, đóng góp của mô hình nằm ở cách phối hợp Geo-I với quan sát lặp, đường đi và chất lượng tìm địa điểm. Em dùng benchmark để kiểm tra sự phối hợp đó có cải thiện cân bằng riêng tư và tiện ích hay không.”

## Ví dụ để chiếu và phân tích thuật toán

[Bốn slide PDF](walkthrough/walkthrough.pdf) hoặc [bản tương tác theo thời gian](walkthrough/walkthrough.html). Đây là lần chạy minh họa riêng: giữ nguyên GPS chuyến SUMO `u701_00`, nhưng dùng mạng công khai tái dựng vì không còn cache mạng benchmark. Không dùng các số này thay benchmark hoặc kết luận về privacy. Bật/tắt S9/S10 cùng seed không cô lập riêng tác động truyền tin: S9 còn làm thời điểm GPS đầu tiên được bảo vệ thay đổi.

| Slide | Lời dẫn khi chiếu |
|---|---|
| 1 · Toàn chuyến | “Chuyến dài 384 giây. S9 bỏ các mốc 0/20/40. Từ giây 60 mới tạo truy vấn; S10 giữ thêm ít nhất 60 giây nên lần gửi đầu ở giây 120.” |
| 2 · Bảo vệ và chọn truy vấn | “Giây 60, GPS là 39,992052 / 116,293807. Geo-I tạo Z là 39,983384 / 116,294048, lệch khoảng 964 m; chi phí 0,01/m. Ước lượng 177 ô chỉ dùng thông tin đã bảo vệ; chọn 5 điểm truy vấn trên đường. Giây 80 không đọc GPS mới; slack giảm điểm độ phủ 0,000557, trong giới hạn 0,03.” |
| 3 · Nhận POI | “Giây 120 gửi bộ tạo từ giây 60. Riêng nhà hàng, 5 điểm nhận 50 lượt POI, còn 28 ID sau bỏ trùng. GPS hiện tại chỉ dùng trên thiết bị để chọn 5 nhà hàng. Cả 5 đều nằm trong top-5 chuẩn ở mốc này.” |
| 4 · Ngân sách và đoạn biên | “Giây 360, khoảng cách 361,5 m cộng nhiễu −468,2 thành giá trị kiểm tra −106,7, nhỏ hơn 200: giữ Z, vẫn tốn 0,01/m. Số âm là giá trị kiểm tra có nhiễu, không phải khoảng cách vật lý. Khi đóng phiên, 4 bộ đang chờ bị hủy. Tổng đã dùng 0,10/m; hủy không hoàn lại ngân sách.” |

Điểm để thảo luận: tham chiếu Z khác 5 điểm gửi máy chủ; θ = 200 m là ngưỡng tái dùng, không phải bán kính nhiễu. Recall tại các mốc có gửi là 100%, nhưng tính cả 21 mốc đầu vào là 71,43% do giai đoạn chờ. Đây là một mẫu trên mạng nhỏ, chưa chạy attacker. Mạng demo kiểm tra được đường đi theo chính các nối tái dựng, chưa xác minh luật rẽ gốc.

Bản HTML có nút nhảy đến 0/60/80/120/360/384 giây, chọn loại POI, đổi trạng thái S9/S10 và xem bảng ngân sách. GPS thật, Z và trọng số ước lượng trên hình chỉ là góc nhìn phân tích. [Transcript công khai riêng](walkthrough/public_transcript.json) chỉ chứa điểm truy vấn và thời điểm công bố.
