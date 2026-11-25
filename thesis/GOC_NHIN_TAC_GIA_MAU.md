# Câu trả lời mẫu để đối chiếu với sổ viết của tác giả

Mở [phiếu để điền](GOC_NHIN_TAC_GIA.md) cạnh file này. Hai bên có cùng 25 mục. Mỗi đoạn dưới đây là **một cách trả lời phù hợp với bản thảo hiện tại**. Những lựa chọn như dùng ví dụ trạm sạc, xếp đóng góp nào đầu tiên hay ưu tiên thí nghiệm nào là đề xuất để bạn cân nhắc. Ví dụ trạm sạc là giả định; chuyến mô phỏng có số liệu thật của benchmark xuất hiện ở mục 3.2.

Khi review, bạn có thể ghi ngay vào phiếu: **Giữ** câu mình đồng ý, **Đổi** chỗ có cách hiểu khác, **Hỏi lại** chỗ chưa đủ rõ để tự giải thích. Với các mục kỹ thuật, hãy sửa cách diễn đạt trước và đánh dấu nếu muốn thay cả phạm vi kết luận.

## 1. Mở đầu

### 1.1. Một người dùng đang làm gì?

Giả sử tôi đang lái xe tới một cuộc hẹn và muốn ghé sạc trên đường. Gợi ý đầu tiên trên bản đồ là một trạm ngay bên kia dải phân cách. Trạm thứ hai ở xa hơn, nhưng nằm trên đường tôi đang đi. Nếu ứng dụng cho biết tôi phải lái thêm bao lâu để ghé từng trạm rồi tiếp tục hành trình, tôi có thể chọn trạm thứ hai. Đây là nhu cầu tôi muốn bắt đầu từ đó: người dùng sẵn sàng hỏi dịch vụ vì câu trả lời giúp họ đưa ra một quyết định cụ thể. Vấn đề riêng tư xuất hiện khi dịch vụ có thể ghi nhận những lần hỏi liên tiếp trong lúc xe di chuyển. [Bối cảnh trong Chương 1](chapters/introduction.tex)

### 1.2. Khoảnh khắc một truy vấn trở thành bài toán quỹ đạo

Ở lần hỏi đầu, giả sử các vị trí gửi đi đều nằm quanh một ngã rẽ. Người quan sát có thể hình dung xe đang ở nhánh phía đông hoặc phía tây. Hai mươi giây sau, gói tiếp theo xuất hiện ở một vùng mà từ nhánh đông có thể tới theo chiều đường, còn từ nhánh tây phải đi vòng. Thêm một lần hỏi nữa, chuỗi quan sát tiếp tục phù hợp với nhánh đông. Dù chưa nhìn thấy GPS thật lần nào, người quan sát đã có lý do để đặt niềm tin lớn hơn vào một hành trình. Khi dùng từ *suy luận*, tôi muốn nói đến mức thay đổi niềm tin này. Nó xảy ra trước cả khi người quan sát đủ tự tin để đoán một tọa độ cụ thể.

### 1.3. Bạn đặt ranh giới cho bài toán ở đâu?

Tôi xét người dùng đang hỏi một nhà cung cấp dịch vụ trong một phiên di chuyển. Thiết bị giữ GPS thật, lịch sử tham chiếu đã bảo vệ và nhu cầu tìm kiếm. Nhà cung cấp nhận các vị trí dùng để truy vấn, biết thứ tự, lịch gửi và các phản hồi dịch vụ. Tôi chọn một phiên vì các lượt hỏi trong khoảng đó dùng chung trạng thái và ngân sách; ta có thể lần theo mọi quyết định dẫn đến chuỗi công bố. Nếu nhà cung cấp đã biết tài khoản của người dùng hoặc tuyến đường họ thường đi, thông tin đó thuộc về tri thức có trước. Câu hỏi còn có ý nghĩa là **riêng phiên trao đổi này có thể bổ sung bao nhiêu bằng chứng về hành trình**. Khi liên kết nhiều phiên, phải tính thêm thông tin từ những phiên ấy. [Phạm vi và giả thiết](chapters/method.tex)

### 1.4. Chất lượng dịch vụ có ý nghĩa gì đối với người dùng này?

Trở lại hai trạm sạc: nếu tôi muốn trạm gần vị trí hiện tại nhất, trạm bên kia dải phân cách có thể đứng đầu. Nếu tôi muốn ít đi vòng trên đường tới cuộc hẹn, trạm thứ hai có thể đứng đầu. Vậy chất lượng phải được xét ở danh sách mà điện thoại hiển thị cho đúng nhu cầu. Một tọa độ truy vấn cách xa GPS vẫn có thể giúp tìm đúng trạm nếu phản hồi chứa đủ ứng viên. Ngược lại, một tọa độ gần GPS vẫn có thể bỏ sót trạm thuận đường. Vì thế, khi chọn các vị trí sẽ gửi, tôi phải quan tâm cả hai đầu: máy chủ có thể suy ra gì từ chúng và thiết bị có thể thu về những địa điểm nào. [Định nghĩa đáp án dịch vụ](chapters/utility.tex)

### 1.5. Vì sao cần đặt lại cách đánh giá và tự xây dựng dữ liệu?

Khi đặt một phương pháp gửi GPS cùng các điểm giả cạnh một phương pháp gửi một tọa độ thay thế, tôi gặp ngay vấn đề về cách chấm. Câu “phần tử nào trong gói là GPS thật?” có đáp án trong trường hợp đầu. Ở trường hợp sau, người quan sát vẫn có thể suy luận rất tốt dù GPS thật chưa bao giờ xuất hiện trong gói. Tôi muốn cho người quan sát ở cả hai trường hợp cùng quyền đoán một tọa độ bất kỳ, rồi đo sai số ở cùng thời điểm. Tôi cũng muốn biết danh sách POI đúng cho người dùng lúc ấy. Vì vậy, dữ liệu đánh giá phải ghép được GPS tham chiếu, thời gian, mạng đường, danh mục POI và nhu cầu tìm kiếm trên cùng một chuyến. Mô phỏng cho phép kiểm soát và đồng bộ các thành phần đó. Kết quả thu được mô tả điều xảy ra trên nhóm tuyến đã xây dựng. [Thiết kế dữ liệu](chapters/experiments.tex)

### 1.6. Vì sao lại có mười kịch bản?

Tôi bắt đầu từ câu hỏi mà bên quan sát muốn trả lời. Với định vị, S1 yêu cầu đoán vị trí từ một sự kiện. S3 có thể chấm chính vị trí đó sau khi đã xem cả chuỗi. S9 và S10 dành riêng cho điểm đi và điểm đến. S2 hướng tới nơi dừng; phép thử hiện tại dùng tọa độ trung bình ở phút cuối làm đại diện. Năm trường hợp này có nhãn tọa độ để đối chiếu trên cùng tuyến nên được đưa vào bảng chính. S4 hỏi hai phiên có liên hệ hay không, S5 và S6 hỏi về di chuyển tương lai, S7 hỏi nội dung yêu cầu lộ ra đâu, S8 xét thêm thông tin từ người đi cùng. Việc tách chúng giúp tôi nói chính xác lúc nào cần nhãn danh tính, một tập đích ứng viên hay một kênh quan sát thứ hai. [Định nghĩa S1–S10](chapters/introduction.tex)

### 1.7. Câu hỏi nghiên cứu nào dẫn đến lựa chọn Geo-I?

| Câu hỏi tôi sẽ nêu | Lý do cần câu trả lời |
|---|---|
| **RQ1.** Sau một phiên, cùng một chuỗi công bố có thể khiến một trong hai hành trình trở nên đáng tin hơn bao nhiêu? | Một tỷ lệ đoán đúng phụ thuộc bộ suy luận được thử. Tôi cần thêm phát biểu về chính phân phối mà cơ chế phát ra. |
| **RQ2.** Khi dùng những tọa độ đã bảo vệ để tìm POI, điều kiện nào giúp điện thoại vẫn trả được địa điểm phù hợp ở GPS thật? | Điểm mà thuật toán dùng để chọn truy vấn chỉ là đại diện cho lợi ích dịch vụ. Người dùng trải nghiệm danh sách cuối. |
| **RQ3.** Trên cùng những chuyến đi, mức giảm khả năng định vị phải đánh đổi bằng bao nhiêu chất lượng và bao nhiêu lượt truy vấn? | Đây là căn cứ để đánh giá ý nghĩa thực tế của thiết kế cùng các phương án đối chứng. |

**Câu chuyển:** Geo-I cho tôi một điểm xuất phát cụ thể. Tôi có thể giữ nguyên sự kiện quan sát rồi hỏi xác suất của nó đổi bao nhiêu khi đầu vào dịch chuyển. Việc tiếp theo là đưa cách lập luận ấy qua những quyết định có trạng thái trong một phiên. [Ba RQ trong bản thảo](chapters/introduction.tex)

### 1.8. Bạn nhận phần đóng góp nào là của mình?

| Thứ tự đề xuất | Đóng góp tôi muốn nhận | Nền tảng kế thừa và chỗ kiểm tra |
|---|---|---|
| 1 | Đặc tả mô hình truy vấn theo đoạn và nối quyết định tạo/giữ tham chiếu, chọn đoạn, phát khung thành cận cho toàn bộ những gì nhà cung cấp quan sát trong phiên. | Kế thừa Geo-I, dự đoán riêng tư, hợp thành và hậu xử lý. [Định lý toàn phiên](chapters/privacy.tex) |
| 2 | Hệ thống hóa từng mục tiêu suy luận theo dữ liệu cần bảo vệ, cửa sổ quan sát và đáp án tham chiếu; từ đó xây được phép so sánh giữa nhiều dạng đầu ra. | Dùng các phép đo đã biết như Hit100, MAE và Recall. [Kịch bản](chapters/introduction.tex) và [quy trình đánh giá](chapters/experiments.tex) |
| 3 | Đo đồng thời riêng tư, chất lượng POI và chi phí cho sáu phương án, rồi tách riêng phép thử xem việc thích nghi theo belief đóng góp bao nhiêu. | Ba đối chứng được điều chỉnh từ nghiên cứu gốc. [So sánh chính](generated/comparison.tex) và [phân tích thành phần](chapters/discussion.tex) |

Nếu chỉ đọc thêm một chương, tôi muốn người đọc mở Chương 3 với câu hỏi “mỗi lần thiết bị dùng GPS hoặc chọn một gói để gửi, kết quả ấy được tính vào lập luận toàn phiên ở đâu?”. **Điểm bạn nên tự chốt:** nếu với bạn việc đặt lại bài toán quan trọng hơn thiết kế cơ chế, hãy đưa dòng 2 lên đầu và sửa tương ứng câu kết.

## 2. Nghiên cứu liên quan

### 2.1. Hai công trình nào ảnh hưởng nhiều nhất đến cách bạn nhìn bài toán?

**Một cặp có thể chọn là Predictive Mechanism và EV querying.** Bài về cơ chế dự đoán cho thấy tương quan di chuyển có thể được dùng ngay trên thiết bị. Khi quan sát trước đã đủ hữu ích, thiết bị có cơ hội dùng lại nó. Điều khiến bài này quan trọng đối với lập luận của tôi là phép thử để đưa ra quyết định giữ hay đổi cũng đọc thông tin riêng. Tôi kế thừa nguyên tắc hạch toán đó và phải giải quyết thêm bước chọn cả đoạn truy vấn mà máy chủ sẽ nhận.

Bài về truy vấn trạm sạc giúp đặt lại tiêu chí chất lượng. Người lái quan tâm đến trạm có thể chọn và chi phí đi tới đó. Gửi từ nhiều tọa độ và lựa chọn ở phía thiết bị đã có tiền lệ. Phần tôi cần giải thích thêm là điều gì xảy ra khi nhà cung cấp nhìn thấy liên tiếp nhiều khung truy vấn, và làm sao chấm chất lượng theo nhu cầu thật xuyên suốt phiên. Đây là lựa chọn hai bài để kể một mạch kỹ thuật; bạn có thể thay một bài bằng công trình thực sự ảnh hưởng nhiều hơn đến mình. [Đối chiếu trong Chương 2](chapters/related_work.tex)

### 2.2. Đoạn tổng hợp sau bảng nên dẫn sang phương pháp thế nào?

TransProtect phát một tọa độ thay thế ở mỗi mốc. Nếu có bảng chi phí đi lại và các đích tham chiếu phù hợp, ta có thể tính mức sai lệch chi phí đi lại theo định nghĩa của bài. Semantic correlation phát một gói chứa GPS thật cùng các dummy, nên tác giả có thể đánh giá khả năng nhận ra thành viên thật. Khi chuyển sang GeoI-BQ, cơ chế có thể gửi năm vị trí truy hồi đều khác GPS. Tôi vẫn muốn hỏi đối thủ có đoán được vị trí người dùng hay không, nên cho phép họ đưa ra một tọa độ bất kỳ rồi chấm sai số. Hai lớp phép đo phục vụ hai việc: giữ ý nghĩa của kết quả gốc ở nơi có đủ đầu vào và tạo một câu hỏi chung để so các phương án. Bước kế tiếp là thiết kế cơ chế sao cho ta xác định được chính xác đầu ra nào được phép quan sát và đáp án nào thực sự đến tay người dùng. [Bảng metrics gốc và trường hợp N/A](generated/native.tex)

## 3. Phương pháp

### 3.1. Lời dẫn cho hai sơ đồ

**Sơ đồ hệ thống.** Tôi muốn người xem nhìn đường biên giữa điện thoại và máy chủ trước. Điện thoại biết người dùng đang ở đâu và họ cần tìm gì. Máy chủ nhận các tọa độ truy vấn Q theo lịch, rồi trả các POI cho những tọa độ đó. Điện thoại hợp nhất kết quả, lọc và xếp hạng theo GPS và nhu cầu thật. Hai bên vì thế có hai góc nhìn khác nhau về cùng một lượt phục vụ.

**Sơ đồ thuật toán.** Z là tham chiếu đã bảo vệ trên mạng đường mà thiết bị giữ để cập nhật hiểu biết về vị trí. Sau cùng một Z, hai nhánh quanh ngã rẽ vẫn có thể có xác suất đáng kể. Belief biểu diễn sự phân bố đó và giúp đánh giá những đoạn truy vấn nào có khả năng thu về POI hữu ích. Thiết bị chọn ngẫu nhiên một đoạn theo quy tắc bị chặn rồi mới gửi từng khung Q. Phân biệt Z với Q cho phép tôi thiết kế cách đọc GPS riêng và cách truy hồi địa điểm riêng, sau đó chứng minh ảnh hưởng của cả hai lên chuỗi công bố. [Hai sơ đồ và quy trình](chapters/method.tex)

### 3.2. Quyết định thiết kế nào cần giải thích bằng một chuyến đi?

Tôi chọn giây 180 của chuyến mô phỏng *cache-029*. Ở giây 120, bộ nhớ có 402 POI. Đến giây 180, phép thử giữ Z cũ, chi phí GPS tích lũy đi từ 5 lên 6 đơn vị, trong khi tọa độ Q đầu tiên trong khung đã đổi và bộ nhớ có 407 POI. Về phía người dùng, điều hữu ích là có ứng viên để xếp hạng cho yêu cầu tại vị trí hiện tại. Dữ liệu mẫu ghi số POI, chưa cho biết người này thực sự muốn ghé trạm sạc. Về phía máy chủ, có một khung tọa độ mới đến đúng lịch. Cờ “giữ Z” và sổ chi phí chỉ có trong dữ liệu kiểm tra nội bộ. Tôi chọn ví dụ này vì nó trả lời được một thắc mắc rất cụ thể: giữ nguyên tham chiếu vẫn cho phép thay đổi hành động truy vấn. Việc bộ nhớ tăng năm POI trong ví dụ cũng chưa đủ để kết luận Recall tăng ở mọi lần gửi. [Chuyến đã lưu](generated/sample_trace.tex)

### 3.3. Trực giác nào nối bốn mệnh đề riêng tư?

| Mệnh đề | Cách tôi giải thích tại bảng |
|---|---|
| 1. Tạo Z | Đặt hai GPS gần nhau trên cùng bản đồ và xét khả năng sinh **cùng một Z**. Khi GPS dịch một đoạn ngắn, bất đẳng thức tam giác chặn thay đổi của khoảng cách tới từng ứng viên. Áp dụng cho cả trọng số lẫn mẫu số chuẩn hóa cho cận Geo-I của phân phối REM. |
| 2. Giữ hoặc đổi Z | Quyết định làm mới tham chiếu chứa thông tin về việc người dùng đã đi xa đến đâu. Dù cờ giữ/đổi ở trong máy, nó vẫn ảnh hưởng đến các lần gửi sau. Phép thử có nhiễu và chi phí mỗi nhánh cho phép cộng dồn ảnh hưởng của các lần đọc GPS trong ngân sách phiên. |
| 3. Chọn một đoạn Q | Với cùng lịch sử công khai, hai belief có thể ưu tiên các đoạn khác nhau. Nếu một đoạn chỉ khả dụng dưới một belief, thấy đoạn đó là đủ để loại hẳn giả thuyết kia. Một thư viện chung và xác suất chọn bị chặn giúp so sánh cùng một đoạn ở cả hai phía. |
| 4. Công bố nhiều đoạn | Xét xác suất của lựa chọn tiếp theo sau khi đã quan sát cùng một tiền sử, rồi nhân các tỷ số qua từng đoạn. Lịch phân bổ có tổng hữu hạn giới hạn tích đó dù chưa biết chuyến đi dài bao lâu. Các khung đã chọn trước được phát theo lịch công khai. |

**Kết luận tôi sẽ nói:** lịch sử tham chiếu đã bảo vệ cho một cận theo khoảng cách giữa hai quỹ đạo. Kênh chọn Q cho một cận nữa lên cùng sự kiện công bố. Khi thỏa cả hai nhóm giả thiết, ta dùng được cận nhỏ hơn. Với Γ = 1, hệ số của kênh Q nhỏ hơn 2,437372. Nếu *chỉ có hai hành trình giả thuyết ngang khả năng*, odds ban đầu là 1:1; sau khi thấy transcript, xác suất hậu nghiệm của hành trình được ủng hộ nhiều nhất bị chặn ở khoảng 70,91%. Đây là câu trả lời cho “phiên này có thể làm lệch cán cân bao nhiêu?”. Muốn suy ra tỷ lệ đoán trúng tọa độ trong 100 m còn cần biết phân bố ban đầu của vị trí. Nếu nhà cung cấp vốn đã biết người này thường đi một tuyến, ta phải bắt đầu từ odds đã lệch ấy. [Định lý và hệ quả Bayes](chapters/privacy.tex)

### 3.4. Chứng minh chất lượng dịch vụ giúp trả lời điều gì?

Giả sử đoạn Q được ưu tiên vì thường lấy được nhiều trạm quanh nhánh đông. Xe thực tế lại đang ở nhánh tây, nơi chỉ có một trạm thuận đường. Nếu máy chủ không trả trạm ấy và cache chưa có, thuật toán xếp hạng tốt đến đâu cũng thiếu ứng viên quan trọng nhất. Ngược lại, nếu cache còn bản ghi hợp lệ của trạm, thiết bị có thể dùng nó khi áp dụng cùng tiêu chí lọc và thứ tự xếp hạng với đáp án tham chiếu. Khi xe đi tiếp hoặc yêu cầu đổi, đáp án đúng cũng có thể đổi. Đó là lý do tôi đọc Recall@5 như kết quả đo trên từng yêu cầu tại GPS thật. Việc ưu tiên được đoạn có điểm lập kế hoạch cao hơn tự nó chỉ nói về điểm mà bộ lập kế hoạch sử dụng. [Chứng minh chất lượng](chapters/utility.tex)

## 4. Thực nghiệm và cách đọc kết quả

### 4.1. Một phép so sánh công bằng theo bạn phải giữ cái gì giống nhau?

Tôi cố định **bên quan sát được nhìn bao lâu và phải đoán cái gì**. Nếu chỉ cho xem năm gói mỗi phương pháp, bên chèn thêm gói giả có thể gửi đủ năm gói khi mới đi được một đoạn rất ngắn. Khi đó hai bên được nhìn hai khoảng hành trình khác nhau. Tôi giữ cùng tuyến, cửa sổ thời gian, danh mục và đáp án tham chiếu; từng phương pháp vẫn có lịch gửi tự nhiên và chịu chi phí của mọi gói phát sinh. Raw GPS giúp kiểm tra tác vụ có tín hiệu để đoán hay không. REM-Z giúp xem chuyện gì xảy ra nếu gửi thẳng tham chiếu. Ba bản điều chỉnh từ paper trả lời câu hỏi so sánh với các cách bảo vệ khác.

Mô phỏng cho phép biết GPS và đáp án dịch vụ ở đúng mốc cần chấm. Tôi chia 48 nhóm tuyến thành 20/8/20 cho huấn luyện, chọn bộ suy luận và kiểm thử. Các chuyến cùng nguồn vì thế ở cùng một tập. Ở TEST, 20 nhóm × 2 chuyến × 3 lần lấy mẫu cho 120 lượt mỗi phương án. Dù có nhiều lượt, bằng chứng vẫn gắn với cohort mô phỏng đã xem; tính khái quát sang địa bàn hoặc người dùng mới còn cần kiểm tra độc lập. [Quy trình](chapters/experiments.tex)

### 4.2. Kết quả định vị nào thuyết phục bạn nhất?

Với Fake-query insertion*, tỷ lệ đoán trúng trong 100 m tăng từ 53,387% ở S1 lên 98,790% ở S3. Điều đáng chú ý là đối thủ được phép dùng thêm lịch sử để suy lại những vị trí đã qua. Với GeoI-BQ, hai ô tương ứng cùng là 0,188%. Tôi đọc hàng đầu như bằng chứng rằng cửa sổ quan sát có ý nghĩa đối với bản đối chứng này. Tôi đọc hàng sau thận trọng hơn: trong phép thử đã làm, bộ suy luận được chọn không cải thiện Hit100 khi được xem cả phiên. Đối với GeoI-BQ, cả hai tác vụ đều chọn centroid cho Hit100. Vì vậy, sự bằng nhau của hai ô chưa cho thấy lịch sử hoàn toàn vô ích.

**Tôi sẽ tự hỏi tiếp:** nếu cho cùng một họ bộ suy luận khai thác lịch sử phong phú hơn và kiểm tra trên nhóm tuyến độc lập, chênh lệch S1–S3 sẽ còn như cũ không? Dấu sao trên tên đối chứng chỉ kết quả của phiên bản điều chỉnh trong luận văn. [Hai hàng số liệu](generated/privacy.tex) · [Bộ suy luận đã chọn](generated/main_selected.tex)

### 4.3. Bạn có chấp nhận đánh đổi chất lượng và chi phí này không?

Tôi xem đây là đánh đổi đáng nghiên cứu cho tác vụ tìm POI. Trên cohort hiện tại, GeoI-BQ đạt Recall@5 trung bình 99,595%, thấp hơn ba bản điều chỉnh khoảng 0,4 điểm phần trăm; riêng mục đích “trong bán kính” là 99,388%. Đó là tỷ lệ thu hồi các đáp án tham chiếu. Với người lái xe sắp hết pin, bỏ sót đúng trạm cuối cùng có thể quan trọng hơn nhiều so với điều thể hiện qua giá trị trung bình. Ứng dụng trả danh sách đã xếp hạng; số 5 ở đây là độ sâu để đánh giá.

Về vận hành, trong 600 giây có 31 gói chứa tổng cộng 155 lượt truy vấn tọa độ. Raw GPS dùng 31 lượt, còn Semantic correlation* cũng dùng 155. Vì vậy, nhận định về chi phí phải nói rõ đang so với ai. Để quyết định triển khai, tôi còn muốn biết độ trễ người dùng thực sự chờ, năng lượng trên thiết bị và hậu quả của các trường hợp bỏ sót quan trọng. [Chất lượng](generated/utility.tex) · [Chênh lệch ghép cặp](generated/comparison.tex) · [Chi phí](generated/cost.tex)

### 4.4. Kết quả bất lợi nào làm bạn thay đổi cách viết về mô hình?

Một kỳ vọng hợp lý của thiết kế là belief giúp chọn các đoạn có POI phù hợp hơn. Nhưng trên danh mục tĩnh đang xét, ngay gói đầu đã thu về trung bình khoảng 91% danh mục, và phương án nền công khai Γ=0 có Recall nhỉnh hơn bản thích nghi. Tôi có cơ sở để nói các truy vấn Q thu về đủ ứng viên cho chất lượng cao trên workload này. Tôi chưa có bằng chứng rằng thích nghi theo belief làm Recall tăng ở đây. Hơn nữa, nếu thiết bị đã có đầy đủ danh mục đúng như phía máy chủ, nó có thể tự xếp hạng cục bộ. Tôi muốn thử lại với dữ liệu dịch vụ mới chỉ có trên máy chủ, giữ cùng độ sâu phản hồi và so từng cặp chính sách ở cả hai chế độ có/không dùng cache. Khi đó có thể thấy lợi ích đến từ lần lấy dữ liệu mới, từ cache, hay từ chọn đoạn. [Phân tích thành phần](chapters/discussion.tex)

### 4.5. Mỗi kịch bản bổ sung cho phép bạn phát biểu đến đâu?

| Kịch bản | Câu tôi sẵn sàng nói và điều cần kiểm tra thêm |
|---|---|
| S4 | Trong bài toán nối hai chuyến theo nhóm tuyến, GeoI-BQ có balanced accuracy 47,50%, CI 95% [40,00%; 54,17%]. Muốn đánh giá danh tính thật cần nhãn người và nhãn xe riêng, vì một người có thể đi nhiều xe và một xe có thể có nhiều người lái. |
| S5 | Trước ngã rẽ, ngay đối chứng GPS thật cũng chỉ đạt 50% khi đoán một trong hai nhánh. Tác vụ hiện tại ít khả năng phân biệt hiệu quả giữa các cơ chế. Cần một thiết kế trong đó dữ liệu thô cung cấp tín hiệu dự báo đủ mạnh. |
| S6 | Sau khi đã có dấu hiệu rẽ, Raw GPS đạt 100%, GeoI-BQ đạt 50,83% khi chọn giữa hai đích đã biết. Kết quả cho thấy tín hiệu đích khai thác được ở đối chứng suy giảm trong phép thử này. Dự đoán một đích bất kỳ trong thành phố là một phạm vi rộng hơn. |
| S7 | Khi giữ nguyên tuyến và thay nhu cầu trên thiết bị, thiết kế truy vấn chung cô lập được phần tiết lộ trực tiếp do nội dung yêu cầu. Chính tuyến di chuyển vẫn có thể gợi ý ý định. Cần thêm dữ liệu liên hệ tuyến và nhu cầu để đo phần suy luận ấy. |
| S8 | Trên phép thử hai kênh đã căn thời gian, Hit100 của GeoI-BQ đi từ 0% khi chỉ nhìn kênh mục tiêu lên 100% khi thêm kênh GPS thật tương quan. Đây là bằng chứng rõ về giới hạn của bảo đảm một kênh. Để bàn về đồng hành thực tế còn cần dữ liệu hai người thật và mức đồng bộ phù hợp. |

Trong bài báo ngắn, tôi sẽ giữ S4 để làm rõ giới hạn về danh tính và S8 để minh họa kênh phụ. Phần diễn giải chi tiết S5–S7 có thể đặt trong luận văn vì cần giải thích riêng tín hiệu đối chứng và cách dựng từng tác vụ. Đây là một lựa chọn biên tập để bạn cân nhắc. [Năm phép thử bổ sung](chapters/scenario_results.tex)

### 4.6. Chứng minh và benchmark hỗ trợ nhau ở đâu?

“Nếu thầy hỏi liệu điểm định vị thấp có thể do dữ liệu mô phỏng hoặc attacker chưa đủ mạnh, em đồng ý rằng cả hai đều ảnh hưởng đến số đo. Em đã tách nhóm tuyến để huấn luyện, chọn và kiểm thử; đối chứng Raw giúp phát hiện những tác vụ mà ngay cả GPS thật cũng ít tín hiệu. Khi có bộ suy luận mạnh hơn, em sẽ đánh giá lại Hit100 và MAE trên cùng cửa sổ. Phần chứng minh trả lời một câu khác: dưới hai hành trình đầu vào, xác suất xuất hiện cùng một chuỗi quan sát có thể thay đổi tới đâu. Nếu bộ suy luận mới chỉ dùng đúng chuỗi ấy thì định lý vẫn áp dụng theo giả thiết đã nêu. Nếu nó còn có dữ liệu từ người đi cùng, biết nhãn riêng của thiết bị hoặc tận dụng hành vi gửi lại ngoài mô hình, em phải xét lại kênh quan sát. Một kết quả thực nghiệm xấu hơn là lý do xem kỹ cả giao thức lẫn giả thiết, chứ riêng con số ấy chưa quyết định tính đúng của định lý.” [Phạm vi chứng minh](chapters/privacy.tex) · [Độ tin cậy thực nghiệm](chapters/discussion.tex)

## 5. Kết luận

### 5.1. Sau tất cả, bạn có câu trả lời nào cho người đọc?

| RQ | Câu trả lời tôi chọn | Bằng chứng và điều kiện |
|---|---|---|
| Riêng tư cả phiên | Có thể giới hạn mức mà chuỗi trao đổi làm thay đổi khả năng phân biệt hai quỹ đạo. | Bốn mệnh đề và định lý hậu xử lý. Cần cùng bối cảnh công khai, miền chọn đoạn chung, trạng thái ngân sách đúng và phản hồi theo mô hình đã nêu. |
| Chất lượng dịch vụ | Thiết bị có thể nhận diện các POI thuộc đáp án chuẩn trong tập đã truy hồi khi dùng dữ liệu hợp lệ và cùng quy tắc xếp hạng. Việc tăng điểm lập kế hoạch chuyển thành tăng chất lượng thật còn tùy vào mức phù hợp của điểm này. | Phân tích chọn đoạn, cache và xếp hạng. Sai số của belief/proxy trên người dùng thật vẫn cần hiệu chuẩn. |
| Đánh đổi thực nghiệm | Trong phép thử hiện tại, các bộ suy luận định vị kém chính xác hơn trên GeoI-BQ, còn chất lượng truy hồi gần các đối chứng với chi phí 155 lượt truy vấn tọa độ mỗi phiên 600 giây. | Năm tác vụ định vị, bốn nhu cầu tìm POI, nhóm tuyến TEST và ba adaptation đã khai báo. |

Với người đang tìm trạm, kết quả cho thấy trong các chuyến đã thử, thiết bị có thể giữ nhu cầu thật ở phía mình và vẫn thu được phần lớn các địa điểm phù hợp. Với người làm nghiên cứu tiếp theo, phần đáng dùng lại là cách xác định toàn bộ thông tin nhà cung cấp quan sát, rồi nối một phát biểu toán học với phép đo trên đúng đầu ra đó. [Kết luận trong bản thảo](chapters/summary.tex)

### 5.2. Nếu tiếp tục đề tài, bạn sẽ kiểm tra điều gì đầu tiên?

Tôi sẽ ưu tiên danh mục động vì đó là nơi có thể kiểm tra đồng thời hai điều còn bỏ ngỏ: khi nào client cần liên hệ máy chủ và khi ấy belief có giúp chọn truy vấn hữu ích hơn không. Tôi sẽ cho thiết bị bản đồ công khai cùng dữ liệu từ lần cập nhật gần nhất, còn máy chủ giữ trạng thái POI hiện hành. Trên các tuyến chưa dùng để chọn cấu hình, tôi sẽ so xử lý hoàn toàn cục bộ, chính sách nền công khai và chính sách thích nghi. Hai phương án online cần cùng độ sâu phản hồi và cùng chính sách cache. Tôi sẽ đo Recall theo nhu cầu, độ trễ và chi phí truyền. Nếu bản thích nghi vẫn không có lợi thế ở cùng chi phí, tôi sẽ thu hẹp lập luận về belief xuống vai trò *cho phép điều chỉnh có cận*, và ưu tiên chính sách đơn giản hơn khi triển khai. Đây là đề xuất thực nghiệm, chưa phải kết quả của luận văn. [Hướng phát triển trong Chương 5](chapters/summary.tex)

## 6. Tóm tắt, giọng văn và phần cá nhân

### 6.1. Bản tóm tắt bạn muốn người đọc nhớ

Khi một người liên tiếp tìm địa điểm trên đường đi, nhà cung cấp có thể kết hợp nhiều lượt truy vấn để suy luận hành trình. GeoI-BQ giữ một tham chiếu được bảo vệ trên mạng đường, dùng lịch sử đã bảo vệ để chọn các đoạn truy vấn và xếp hạng địa điểm theo nhu cầu thật trên thiết bị. Dưới các giả thiết về lịch gửi, trạng thái phiên và cơ chế chọn đoạn, luận văn suy ra cận cho mức mà toàn bộ chuỗi trao đổi có thể làm thay đổi suy luận. Trên 20 nhóm tuyến kiểm thử, tỷ lệ định vị trúng trong 100 m là 0–0,188% ở năm tác vụ chính với các bộ suy luận đã thử. Recall@5 toàn phiên đạt 99,595%, thấp hơn ba đối chứng điều chỉnh từ paper khoảng 0,4 điểm phần trăm, với 155 lượt truy vấn tọa độ mỗi 600 giây. Kết quả cho thấy một đánh đổi có thể định lượng; lợi ích riêng của thích nghi theo belief còn cần kiểm tra trên dịch vụ có dữ liệu thay đổi. [Tóm tắt của luận văn](chapters/frontmatter.tex)

### 6.2. Điều gì cần được giữ khi cô đọng thành bài báo?

- **Một câu có thể giữ:** “Câu hỏi của luận văn là sau cả chuyến đi, chuỗi trao đổi đã cho nhà cung cấp thêm bao nhiêu bằng chứng về hành trình của người dùng?”
- **Ý giữ trong cả hai ngôn ngữ:** cùng một phiên phải được nhìn từ phía máy chủ quan sát chuỗi và phía người dùng nhận danh sách POI. Khi nêu số riêng tư, đi kèm chất lượng và chi phí của cùng thí nghiệm.
- **Chi tiết dành cho luận văn:** ví dụ đầy đủ từ giây 0 đến 180, diễn giải từng nhánh ngân sách, và cách đọc từng phép thử bổ sung. Paper có thể giữ một mốc di chuyển tiêu biểu.
- **Thuật ngữ dùng nhất quán:** “điểm tham chiếu đã bảo vệ” cho Z (*protected anchor*), “đoạn truy vấn” cho một lựa chọn gồm nhiều khung (*query segment*), “toàn bộ chuỗi trao đổi trong phiên” cho *session transcript*. Bạn có thể đổi cách gọi nếu thấy một cụm khác dễ nói hơn.

### 6.3. Lời cảm ơn và lời cam đoan

**Mẫu lời cảm ơn để bạn xác nhận:** “Tôi cảm ơn thầy Phan Trọng Nhân đã hướng dẫn tôi trong quá trình thực hiện luận văn. Những trao đổi với thầy về cách so sánh các phương pháp giúp tôi xem lại cách đọc một bảng kết quả: trước khi kết luận phương pháp nào tốt hơn, tôi cần xác định mỗi bên được quan sát gì, đang bảo vệ đại lượng nào và người dùng còn nhận được chất lượng dịch vụ ra sao.” Chỉ giữ chi tiết này nếu đúng với trải nghiệm của bạn; tên người khác và đóng góp của họ nên do bạn bổ sung.

**Mẫu ghi chú để tự rà lời cam đoan:** “Trước khi ký, tôi sẽ đối chiếu phần mô tả đóng góp với công việc thực tế, kiểm tra việc dẫn nguồn dữ liệu và các phương pháp đối chứng, đồng thời rà cách mô tả hỗ trợ biên tập theo yêu cầu của trường.” Lời cam đoan cuối cùng cần chính bạn đọc, chỉnh và xác nhận. [Phần đầu luận văn hiện tại](chapters/frontmatter.tex)
