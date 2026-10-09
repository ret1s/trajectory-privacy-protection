# Kịch bản trình bày: Bảo vệ riêng tư vị trí bằng Geo-I/REM

Kịch bản đi theo đúng **14 slide** trong [slides.pdf](slides.pdf). Thời lượng gợi ý khoảng **12-15 phút**, chưa tính trao đổi.

Đọc phần **Lời trình bày**; dùng **Chỉ vào hình** để dẫn người nghe theo sơ đồ hoặc biểu đồ. **Chuyển ý** nối sang trang tiếp theo. Phần **Nếu được hỏi** là nội dung chuẩn bị, không cần đọc trong lượt trình bày chính. Slide chỉ chứa nội dung trình chiếu; mở script ở cửa sổ riêng.

Trong lời nói, dùng **điểm tham chiếu Z**, **tọa độ truy vấn Q** và **địa điểm POI** để phân biệt ba đối tượng. Dùng **điểm phần trăm** khi trừ hai tỷ lệ; **MAE** là sai số trung bình theo mét, còn **Hit100/Hit500** là tỷ lệ attacker đoán trong phạm vi 100/500 mét.

## Slide 1: Bảo vệ riêng tư vị trí bằng Geo-I/REM

**Lời trình bày**

Thưa thầy/cô, em xin trình bày phương pháp bảo vệ riêng tư vị trí bằng Geo-I/REM, kết hợp mạng đường và truy hồi POI theo nhiều mục đích. Mục tiêu là hạn chế thông tin vị trí gửi ra ngoài và duy trì chất lượng tìm địa điểm. REM tạo điểm tham chiếu đã bảo vệ; từ đó mô hình dùng mạng đường và dữ liệu công khai để chọn các tọa độ truy vấn.

Em sẽ trình bày kiến trúc, minh họa từng bước trên các hành trình đã chạy, rồi đánh giá kết quả cùng chi phí và phạm vi bảo vệ. Em sẽ phân biệt GPS, điểm tham chiếu Z, tọa độ Q gửi máy chủ và POI trả về. Các ví dụ và benchmark hiện dùng dữ liệu mô phỏng; chưa khẳng định bảo vệ đầy đủ cả mười kịch bản hoặc đã kiểm chứng trên GPS thực.

**Chỉ vào hình**

Chỉ vào tên Geo-I/REM và dòng mô tả truy hồi POI trên mạng đường theo nhiều mục đích.

**Chuyển ý**

Trước hết, em trình bày luồng xử lý của mô hình.

## Slide 2: Kiến trúc phương pháp

**Lời trình bày**

Em chia mô hình thành bốn tầng theo thứ tự xử lý. Tầng thứ nhất kiểm tra lịch và ngân sách trước khi đọc GPS, rồi giữ hoặc tạo điểm tham chiếu đã bảo vệ Z. Tầng thứ hai dùng Z và thông tin công khai để chọn năm tọa độ truy vấn Q trên mạng đường. Tầng thứ ba gửi Q lên server và nhận các địa điểm POI. Tầng cuối gộp danh sách, rồi chọn kết quả ngay trên thiết bị.

Khung này là các bước của mô hình chạy trên thiết bị; server nằm bên ngoài. Z không được gửi đi. Vị trí cục bộ và nhu cầu thật chỉ vào bước xếp hạng cuối, không quay lại để sửa Z hoặc Q. Vì vậy cần phân biệt GPS đưa vào cơ chế bảo vệ với vị trí dùng để chọn POI cho người dùng.

**Chỉ vào hình**

Đi theo mũi tên qua bốn tầng, rồi chỉ riêng nhánh vị trí cục bộ vào tầng cuối.

**Chuyển ý**

Em giải thích cụ thể lúc nào mô hình được đọc GPS và lúc nào giữ Z.

## Slide 3: Bảo vệ GPS và tái sử dụng điểm tham chiếu

**Lời trình bày**

Không phải cứ 60 giây là mô hình chắc chắn đọc GPS. Trước khi đọc, nó phải đủ khoảng cách thời gian và đủ ngân sách dự toán. Lần đầu, REM lấy mẫu Z trên toàn bộ miền đường công khai cố định; điểm gần GPS thường có xác suất cao hơn. Những lần đọc sau dùng phép kiểm tra có thêm nhiễu để quyết định giữ Z hay tạo Z mới.

Tạo Z đầu tiên chi một đơn vị. Đọc rồi giữ Z cũng chi một đơn vị, vì quyết định giữ đã dùng thông tin riêng. Kiểm tra rồi tạo Z mới chi hai đơn vị. Nếu không được đọc GPS, mô hình giữ lịch sử đã bảo vệ và không chi thêm. Phiên đã nhận vẫn có thể gửi Q từ lịch sử đó; phiên bị từ chối thì không đọc GPS và không gửi Q.

**Chỉ vào hình**

Đi từ bước 1 đến bước 4 ở cột trái, dừng ở phép thử có nhiễu để phân biệt giữ và tạo Z. Sau đó chỉ công thức REM chuẩn hóa trên toàn miền đường công khai ở cột phải, rồi đối chiếu các mức chi phí u, u và 2u.

**Chuyển ý**

Khi đã có Z, mô hình cần chọn Q sao cho phản hồi vẫn hữu ích.

**Nếu được hỏi**

Cấu hình hiện tại dành ngân sách hiệu lực 0,23/m cho tám phiên trong một epoch, tức một khoảng thời gian áp dụng ngân sách chung. Mỗi phiên có tối đa 23 đơn vị, mỗi đơn vị 0,00125/m; khi đã có Z phải dự toán đủ hai đơn vị trước lần đọc tiếp. Sổ ngân sách phải đáng tin cậy; các epoch khác vẫn phải cộng dồn. Cận Geo-I xét tọa độ của cơ chế lý tưởng với ngữ cảnh/lịch công khai cố định; mã dùng số học hữu hạn chưa có chứng nhận tương ứng.

## Slide 4: Ước lượng vị trí và lựa chọn truy vấn

**Lời trình bày**

Z có nhiễu, nên chỉ truy vấn quanh Z có thể bỏ sót POI gần người dùng. Mô hình dùng phân bố b để gán trọng số cho các vùng có thể đang ở. b được cập nhật từ lịch sử đã bảo vệ và chuyển động công khai, không đọc thêm GPS thật. Khi không có lần đọc mới, b chỉ dự đoán chuyển động; nó vẫn là một ước lượng xấp xỉ.

Từ b, mô hình chọn năm Q để phủ POI có khả năng hữu ích. Sau lần gửi đầu, mỗi Q phải đi được từ Q trước theo đường và thời gian đã qua. Năm Q là kết quả lựa chọn này, không phải năm lần lấy nhiễu độc lập quanh Z. Phần này phục hồi chất lượng dịch vụ sau Geo-I; Q hợp lệ trên đường chưa tự chứng minh chống tái dựng hành trình.

**Chỉ vào hình**

Chỉ luồng Z → b → năm Q. Với hai công thức, phân biệt lúc có lần đọc GPS bảo vệ với lúc chỉ dự đoán giữa hai lần đọc; các đường bên dưới minh họa ràng buộc chuyển động.

**Chuyển ý**

Sau khi nhận phản hồi từ Q, thiết bị xử lý nhu cầu thật ở đâu?

**Nếu được hỏi**

Trong công thức, T là mô hình chuyển động công khai trong khoảng thời gian Δt; ℒ là likelihood, tức mức phù hợp của điểm tham chiếu quan sát được với từng vị trí giả thuyết. Khi không có lần đọc mới, chỉ dùng T; không coi Z cũ là một quan sát mới.

Bộ chọn Q hiện vẫn dùng danh sách POI công khai ở độ sâu L10 để tính điểm độ phủ dự đoán. L30 là độ sâu máy chủ trả về sau đó. Thử nghiệm tăng L giữ nguyên Q để tách riêng lợi ích truy hồi thêm POI; chưa khẳng định bộ chọn Q tối ưu cho L30 hoặc b đã khớp xác suất di chuyển thực tế.

## Slide 5: Truy hồi POI theo nhiều mục đích

**Lời trình bày**

Ở đây, K bằng năm là số tọa độ Q gửi đi. L bằng 30 là số POI tối đa server trả cho mỗi loại địa điểm tại mỗi Q. Server vẫn trả các POI gần Q; bốn mục đích là cách xử lý tại thiết bị. Thiết bị gộp và bỏ trùng ID, rồi chọn tối đa năm kết quả: gần nhất, nhanh nhất theo tốc độ công khai, trong bán kính riêng hoặc ít đi vòng tới đích.

Vị trí cục bộ, loại địa điểm thật, bán kính và đích chỉ dùng ở bước cuối. Khi giữ cùng trạng thái đã bảo vệ và lịch công khai, đổi mục đích không làm đổi Q hay yêu cầu mạng. Điều này loại bỏ kênh gửi trực tiếp nhu cầu; nhu cầu tương quan với vị trí hoặc tuyến đường vẫn có thể bị suy luận.

**Chỉ vào hình**

Chỉ một luồng phản hồi đi vào cả bốn cách xếp hạng, cùng nhánh nhu cầu riêng ở thiết bị.

**Chuyển ý**

Tiếp theo em dùng các mốc của một hành trình đã chạy để minh họa toàn bộ quá trình.

**Nếu được hỏi**

Mục đích ít đi vòng giả định thiết bị đã biết đích; benchmark dùng đích thật làm đối chứng lý tưởng. Nếu thiếu đáp án rồi gửi thêm yêu cầu theo nhu cầu riêng, điều kiện bảo vệ kênh nội dung S7 ở đây không còn được giữ.

## Slide 6: Ví dụ 1: bảo vệ GPS và tạo truy vấn

**Lời trình bày**

Ở mẫu đầu tiên, em minh họa một chuyến mô phỏng của cấu hình hiện tại. Tại giây 0, thiết bị đọc GPS và tạo Z, chi một đơn vị ngân sách. Đến giây 20, chưa đủ khoảng cách 60 giây nên không đọc GPS bảo vệ; Z giữ nguyên, chi phí bằng 0. Tại giây 60, thiết bị được phép đọc lại; phép thử có nhiễu dẫn tới tạo Z mới, chi thêm hai đơn vị. Tổng đến đây là ba đơn vị. Giữa các lần đọc, thiết bị vẫn có thể cập nhật Q từ lịch sử đã bảo vệ.

Trên bản đồ, GPS và Z là thông tin nội bộ; chỉ năm Q được gửi lên máy chủ. Phân bố b giúp ưu tiên vùng tìm POI, không phải đoán chính xác một GPS để gửi đi. Ví dụ bên dưới thuộc chuyến 6, là một trường hợp khác: thiết bị đọc GPS nhưng vẫn giữ Z, dù khoảng cách khoảng 1.204 mét. Điều này xảy ra vì phép kiểm tra có nhiễu, không phải ngưỡng cứng 200 mét. Nhánh giữ vẫn chi một đơn vị.

**Chỉ vào hình**

Đi qua ba dòng 0–20–60 giây trong bảng, rồi chỉ GPS, Z và năm Q trên bản đồ; cuối cùng chỉ ví dụ giữ Z của chuyến 6.

**Chuyển ý**

Sau khi có năm Q, bước tiếp theo là xem phản hồi từ máy chủ được chuyển thành địa điểm phù hợp với nhu cầu như thế nào.

**Nếu được hỏi**

Một đơn vị bằng 0,00125/m; ba đơn vị là 0,00375/m, dưới ngân sách tối đa của phiên 0,02875/m. Trước lần đọc sau phải dự toán hai đơn vị, dù nhánh giữ chỉ chi một. Mười hai vùng màu cam chỉ chiếm 9,88% khối lượng b, không phải vùng tin cậy. Giá trị nhiễu Laplace cụ thể không được lưu nên em chỉ nêu điều kiện suy ra từ nhánh đã ghi nhận.

## Slide 7: Ví dụ 2: xếp hạng POI tại thiết bị

**Lời trình bày**

Em giữ nguyên năm Q ở giây 60 của mẫu trước. Mỗi Q yêu cầu mọi loại POI công khai, với tối đa 30 địa điểm cho mỗi loại. Trong mẫu này, mỗi Q nhận 84 bản ghi; năm Q nhận tổng cộng 420 bản ghi. Gộp theo ID và bỏ trùng còn 252 POI. Vì vậy, năm Q là năm tọa độ truy vấn, không phải năm địa điểm trả cho người dùng. Việc chọn địa điểm diễn ra sau đó tại thiết bị.

Với nhu cầu tìm café gần nhất, thiết bị trả năm POI trong bảng theo khoảng cách đường. Nếu đổi sang tìm trong 1.000 mét, chỉ P167 phù hợp, nên trả một POI thay vì ép đủ năm. Nếu muốn ít đi vòng tới đích, danh sách lại khác. Cả bốn mục đích dùng cùng Q và cùng tập ứng viên, không gửi thêm truy vấn theo nhu cầu. Riêng ví dụ đi vòng giả định đích đã biết ở local; thí nghiệm lấy đích thật từ dữ liệu làm tham chiếu lý tưởng.

**Chỉ vào hình**

Chỉ phép gộp 420 → 252, rồi đối chiếu danh sách gần nhất với P167 trong bán kính và danh sách ít đi vòng.

**Chuyển ý**

Hai ví dụ vừa rồi giải thích luồng xử lý chung. Tiếp theo, em dùng một cấu hình lịch sử riêng để minh họa cách gửi truy vấn ở đầu và cuối hành trình.

**Nếu được hỏi**

Nhanh nhất có cùng thứ tự với gần nhất trong mẫu này, nhưng được tính bằng thời gian đi theo tốc độ công khai. Khi triển khai, đích phải do ứng dụng hoặc người dùng cung cấp trước; không được dùng tương lai chưa biết. Kết quả đúng ở một mẫu không thay cho Recall trung bình của benchmark, cũng không chứng minh ý định không thể suy từ tuyến đường.

## Slide 8: Ví dụ 3: bảo vệ điểm đầu và điểm cuối

**Lời trình bày**

Mẫu này thuộc Endpoint20 lịch sử, dùng L20, và được tách khỏi cấu hình L30 hiện tại. Màu xám là Q của GeoI-Slack, màu xanh là Q của Endpoint20; dấu đỏ là GPS đầu hoặc cuối để phân tích, không gửi cho máy chủ. Z không được lưu trong transcript này nên em không suy Z từ Q. Hai cấu hình giữ cùng tám mốc gửi: truy vấn đầu tạo và gửi ở giây 0; truy vấn cuối tạo và gửi ở giây 384. Không bỏ phần đầu, giữ chờ hoặc hủy phần cuối.

Điểm cần chú ý là giây 384 chỉ cách mốc 360 khoảng 24 giây. Vì chưa đủ khoảng cách 60 giây, lúc này không đọc GPS bảo vệ mới; Q vẫn được chọn từ lịch sử đã bảo vệ. Thuật toán không cần biết trước đâu là điểm GPS cuối. Endpoint20 dùng hệ số epsilon bằng 25% của đối chứng ở mọi lần được phép đọc, để tăng nhiễu. Hình chỉ minh họa cách hoạt động; mức bảo vệ phải đánh giá qua attacker, và giờ mở, đóng phiên vẫn quan sát được.

**Chỉ vào hình**

Chỉ cặp “tạo Q → gửi Q” dưới hai bản đồ, rồi chỉ dòng giải thích khoảng cách 24 giây giữa mốc 360 và 384.

**Chuyển ý**

Từ ba ví dụ này, em tổng hợp lại mỗi thành phần hỗ trợ scenario nào và bằng chứng hiện có đến đâu.

**Nếu được hỏi**

Cap lý tưởng mỗi phiên của Endpoint20 là 0,0575/m, so với 0,23/m của đối chứng lịch sử; đây không phải ngân sách tối đa của phiên của Epoch8/L30. Khoảng cách một Q tới GPS không phải MAE attacker, và giảm epsilon xuống 25% không có nghĩa mọi sai số REM tăng đúng bốn lần.

## Slide 9: Phạm vi bảo vệ theo kịch bản

**Lời trình bày**

Trong bảng này, em phân biệt thành phần hỗ trợ bảo vệ với việc đã giải quyết đầy đủ một scenario. Với S1–S3, Geo-I/REM và ngân sách có cận lý tưởng cho đầu vào tọa độ, cùng benchmark lịch sử. Với S4, ngân sách chung chặn việc tự cấp lại ngân sách qua các chuyến trong cùng epoch, nhưng không ẩn tài khoản hay địa chỉ IP; bài toán liên kết người và xe mới được thử trên dữ liệu tổng hợp. S5–S6 hiện kiểm tra dự đoán cạnh kế tiếp và đích trong bài toán hai lựa chọn.

S7 có kết quả rõ ở mức request: khi giữ cùng trạng thái đã bảo vệ và lịch công khai, đổi nhu cầu cục bộ không đổi truy vấn gửi lên máy chủ. Điều này chưa loại bỏ suy luận ý định từ tuyến đường. S8 mới có phép thử người đồng hành trên dữ liệu lịch sử. S9–S10 có bằng chứng của Endpoint20 riêng, chưa chuyển thành xác nhận cho L30. Vì vậy, đóng góp hiện tại là cơ chế và bằng chứng theo từng bài toán; em chưa ghép chúng thành tuyên bố một mô hình đã bảo vệ đầy đủ cả mười scenario.

**Chỉ vào hình**

Đi từ cột “Thành phần / cơ chế” sang “Phạm vi bằng chứng”, nhấn dòng S7 và dòng S9–S10.

**Chuyển ý**

Với phạm vi đó, em chuyển sang kết quả chính: tăng độ sâu phản hồi từ L20 lên L30 cải thiện utility bao nhiêu và phải trả thêm chi phí gì.

**Nếu được hỏi**

Các cận Geo-I là kết quả dưới giả định kernel lý tưởng và giao thức đã khai báo; chưa phải chứng chỉ cho bộ lấy mẫu số thực của simulator. Nhiều epoch hoặc nhiều thiết bị liên kết vẫn phải hợp thành ngân sách. Với S7, nếu ứng dụng gửi thêm request do thiếu đáp án hoặc theo click riêng, điều kiện request không đổi sẽ không còn đúng.

## Slide 10: Độ sâu phản hồi và chất lượng dịch vụ

**Lời trình bày**

Đây là kết quả chính sau khi em khóa cấu hình rồi đánh giá trên 24 nhóm tuyến mới. Recall đo mức giữ lại những POI thuộc top-5 tham chiếu cho nhu cầu đang xét. Khi tăng độ sâu phản hồi từ L20 lên L30, Recall trung bình tăng từ 89,71% lên 92,69%, tức 2,97 điểm phần trăm. Khoảng tin cậy 95% của mức tăng là từ 2,31 đến 3,65 điểm, nằm hoàn toàn trên 0. Cả bốn mục đích đều tăng, trong đó tìm trong bán kính tăng nhiều nhất.

Đánh đổi là byte phản hồi tăng 31,10%. Em vẫn giữ nguyên năm Q, điểm tham chiếu Z, lịch gửi và ngân sách riêng tư; số request cũng không tăng. Vì vậy kết quả này cho thấy lấy thêm ứng viên giúp phục hồi chất lượng trả lời trên cùng luồng đã bảo vệ. Phạm vi xác nhận là dữ liệu mô phỏng mới trên cùng bản đồ. So sánh với các phương pháp khác ở cùng chi phí vẫn cần được đánh giá riêng.

**Chỉ vào hình**

Chỉ mức tăng 2,97 điểm, hàng “Trong bán kính”, rồi chuyển sang hai thanh byte phản hồi.

**Chuyển ý**

Ngoài việc so L20 với L30, em cũng kiểm tra nền tảng REM với một cơ chế Geo-I đối chứng.

**Nếu được hỏi**

2,97 điểm phần trăm là lấy 92,69% trừ 89,71%, khác với phần trăm tăng tương đối. Recall được gộp đều theo mục đích và nhóm tuyến; tham chiếu rỗng giữ N/A. CI lấy mẫu lại theo 24 nhóm, giữ ba lượt nhiễu và tám chuyến bên trong mỗi nhóm. L là số POI tối đa mỗi category tại mỗi Q.

## Slide 11: So sánh REM và Planar

**Lời trình bày**

Ở phép thử này, em so REM với Planar Laplace trên cùng ngân sách, mạng đường, K=5 và L20. Bước ước lượng cũng dùng mô hình quan sát khớp với từng loại nhiễu. Planar đạt Recall 97%, cao hơn mức 94,69% của REM. Hiện chưa có căn cứ để nói REM tốt hơn Planar về chất lượng phản hồi hiện tại.

Về attacker, cả hai phương pháp bảo vệ đều có accuracy cạnh kế tiếp và Hit100 của đích là 50%, so với 100% khi GPS chưa được bảo vệ. Tuy nhiên, tác vụ ở đây chỉ có hai lựa chọn công khai, và S5, S6 dùng cùng một quyết định. MAE đích của REM lớn hơn, nhưng riêng con số đó chưa chứng minh ưu thế riêng tư. Đây là pilot sáu nhóm đã xem, chỉ chấm tìm gần nhất theo category; em tách nó khỏi kết quả bốn mục đích của L30 ở slide trước.

**Chỉ vào hình**

So hai hàng REM và Planar ở cột Recall, rồi chỉ hai cột attacker cùng bằng 50%.

**Chuyển ý**

Sau đối chứng cơ chế, em kiểm tra một giả định ứng dụng: nếu vị trí cục bộ không được cập nhật liên tục thì kết quả thay đổi ra sao.

**Nếu được hỏi**

Hit100 là tỷ lệ đoán đích trong sai số 100m. Attacker được chọn trên tập validation: **Motion** cho REM, **CurveMean(0.02)** cho Planar và **CandidateTrees** cho GPS chưa bảo vệ. Pilot có 12 phiên truy vấn thuộc sáu nhóm test; GPS dùng một tọa độ request, còn hai phương pháp bảo vệ dùng năm Q.

## Slide 12: Ảnh hưởng của tần suất và sai số GPS

**Lời trình bày**

Kết quả chính trước đó dùng vị trí chuẩn tại mỗi event để xếp hạng local. Ở đây, em thử thay bằng GPS cập nhật mỗi 60 giây, kèm nhiễu cảm biến giả lập ở ba mức. Q và dữ liệu đầu vào Geo-I đã đóng băng được giữ nguyên; chỉ vị trí dùng để xếp hạng POI thay đổi. Khi giữ lần đo GPS gần nhất và chưa thêm nhiễu cảm biến, L30 đạt 89,35%, thấp hơn mức 92,69% với vị trí chuẩn mỗi event. L30 vẫn tăng Recall so với L20 ở các biến thể đã thử.

Kết quả đáng chú ý là ngoại suy từ hai lần đo GPS giảm sai số vị trí trung bình, nhưng Recall lại thấp hơn cách giữ lần đo GPS gần nhất ở cả ba mức nhiễu. Vì vậy em chưa chọn ngoại suy vào pipeline chính. Vị trí cũ còn có thể làm bộ lọc bán kính trả POI ngoài miền thực, nên phải đọc thêm tính hợp lệ của đáp án. Phép thử này giúp thấy giới hạn của local ranking; nó chưa phải phép đo GPS thật hay năng lượng trên thiết bị.

**Chỉ vào hình**

Chỉ khoảng cách giữa đường L20 và L30, sau đó chỉ đường ngoại suy luôn thấp hơn đường giữ lần đo GPS gần nhất của L30.

**Chuyển ý**

Vị trí người dùng có thể cũ; thông tin POI từ server cũng có thể hết hạn. Slide tiếp theo kiểm tra trường hợp thứ hai.

**Nếu được hỏi**

σ là độ lệch chuẩn Gaussian trên mỗi trục, lần lượt 0, 5 và 15m; đây là nhiễu kiểm soát, chưa được hiệu chỉnh theo thiết bị thật. Clock GPS local được tách khỏi clock supplier bảo vệ. Phép thử dùng lại 72 luồng đã được phân tích; không tạo Q mới. Đích của mục đích detour vẫn là đầu vào local oracle, và kết quả không gồm detour cũng cho thấy ngoại suy kém hơn giữ lần đo GPS gần nhất.

## Slide 13: Truy hồi POI có trạng thái thay đổi

**Lời trình bày**

Trong phép thử này, POI thay đổi trạng thái khả dụng theo từng khoảng công khai 60 giây. Thiết bị chỉ dùng trạng thái hiện tại đã nhận được. Biết ID của một POI chưa đủ để kết luận nó đang khả dụng; thông tin trạng thái hết hạn không được dùng để xác định POI còn khả dụng. Với L30, dùng phản hồi hiện tại đạt Recall 91,44%. Gộp các phản hồi còn hiệu lực trong cùng khoảng đưa Recall lên 91,78%, mà không tăng Q hoặc byte truyền.

Em cũng giữ đối chứng tải toàn catalogue với trạng thái hiện tại. Đối chứng đó đạt 100% và dùng khoảng 126,89MB JSON, thấp hơn mức 1.023,95MB của L30. Nó có API mạnh hơn truy hồi top-L, nhưng là lựa chọn hợp lệ nếu ứng dụng cho phép tải toàn bộ. Vì catalogue của workload này còn nhỏ, kết quả này chưa chứng minh cần truy vấn từ xa theo vị trí. Điều em kiểm tra được là cách dùng cache còn hiệu lực trong một thế giới trạng thái mô phỏng.

**Chỉ vào hình**

Chỉ hai điểm L30 để thấy lợi ích nhỏ của cache, rồi chỉ đối chứng toàn catalogue ở mức 100%.

**Chuyển ý**

Cuối cùng, em trình bày riêng kết quả bảo vệ điểm đầu và cuối chuyến của cấu hình Endpoint20.

**Nếu được hỏi**

Provider lấy top-L theo vị trí tĩnh rồi gắn bit trạng thái hiện tại, không chọn POI khả dụng trước. Cache hết hạn theo clock công khai; thiếu đáp án không dẫn đến request riêng theo nhu cầu. Đây là diagnostic trên các luồng đã xem và một thế giới trạng thái, không phải xác nhận độc lập. Chi phí là compact JSON ứng dụng, chưa đo HTTP/TLS, latency hay pin.

## Slide 14: Đánh giá bảo vệ điểm đầu và điểm cuối

**Lời trình bày**

Endpoint20 là cấu hình thử nghiệm lịch sử riêng ở L20. Cấu hình này dùng epsilon và ngân sách bằng 25% của đối chứng lịch sử, nên nhiễu mạnh hơn tại mọi lần đọc được phép. Khi đang chạy, hệ thống không biết trước lần đọc nào là cuối chuyến; cấu hình này vẫn gửi ngay, không dùng delay. MAE là sai số trung bình của vị trí attacker đoán. Ở S9, MAE tăng từ khoảng 790 lên 1.407m; ở S10, từ 758 lên 1.337m. Mức tăng S10 khoảng 579m, với CI mô tả từ 468 đến 696m.

Hit100 ở S10 đều bằng 0 cho cả hai cấu hình, nên em đọc thêm Hit500: tỷ lệ đoán trong 500m giảm từ 12,50% xuống 4,46%. Tuy nhiên CI của chênh lệch này chạm 0, nên chưa kết luận mức giảm Hit500 chắc chắn. Đây là phép thử trên 28 nhóm đã xem và một bank attacker hữu hạn. Giờ mở, đóng phiên vẫn quan sát được; các điểm số này cũng không được gán cho cấu hình Epoch8/L30 hiện tại.

**Chỉ vào hình**

Chỉ hai cặp thanh MAE, rồi chỉ dòng Hit100 bằng 0 và cảnh báo CI của Hit500 chạm 0.

**Chuyển ý**

Em xin dừng phần kết quả ở đây và trao đổi với thầy/cô về ưu tiên tiếp theo, nhất là workload dịch vụ thực và xác nhận attacker cho cấu hình hiện tại.

**Nếu được hỏi**

MAE S9 dùng **public_boundary_centroid_ols6**, S10 dùng **centroid_ols3_120s**; Hit500 có decoder được chọn riêng trên validation. CI95% của chênh lệch Hit500 là [-16,96; 0] điểm phần trăm. Nhiễu mạnh hơn này áp cho mọi protected read, không dựa vào phát hiện một “đoạn cuối” bí mật.

## Tài liệu đối chiếu khi chuẩn bị

Các nguồn này dùng để kiểm tra số liệu, không cần đọc đường dẫn trong buổi trình bày.

- [Nội dung và số liệu theo slide](slide_content.json).
- [Ví dụ GPS, Z, b và Q](sample_walkthrough.json), [phản hồi và kết quả POI local](utility_sample.json), [ví dụ Endpoint20](endpoint_sample.json).
- [Phạm vi thuật toán và scenario](algorithm_scope.json), [nguồn benchmark và giới hạn của từng phép thử](benchmark_evidence.json).
