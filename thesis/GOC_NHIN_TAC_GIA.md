# Sổ viết của tác giả · Luận văn và bài báo GeoI-BQ

Hãy dùng file này như bản nháp bạn gửi GVHD. Bạn có thể viết thành đoạn hoàn chỉnh, ghi lại một cuộc trao đổi có thật, phác một ví dụ, hoặc để hai cách diễn đạt cạnh nhau khi còn phân vân. Viết bằng tiếng Việt, dài tùy ý. Những câu bạn muốn giữ nguyên có thể đặt trong ngoặc kép. Nếu một ý mới là dự đoán, ghi thêm “tôi nghĩ vậy vì…” và điều gì sẽ giúp kiểm tra nó.

Mở [25 câu trả lời mẫu theo bản thảo hiện tại](GOC_NHIN_TAC_GIA_MAU.md) cạnh phiếu này để đối chiếu theo cùng số mục.

**Bản suy diễn để tác giả duyệt.** Bốn ý đã chốt là rủi ro từ lịch sử đi lại, ưu tiên chứng minh theo xác suất, đọc benchmark như bằng chứng thực nghiệm có phạm vi, và xem khung kịch bản là đóng góp chính. Những ví dụ, lựa chọn paper và thiết kế thí nghiệm dưới đây là cách biên tập từ bốn ý ấy; phần xác nhận cá nhân ở 6.3 dành cho tác giả. Các câu hỏi phía trên gợi ý những quyết định mà người khác khó viết thay bạn. Phần công thức, chứng minh, trích dẫn, định nghĩa phép đo, số liệu và khoảng tin cậy đã có trong bản thảo. Bạn có thể dẫn tên bảng hoặc tên mệnh đề ngay giữa đoạn đang viết, tôi sẽ nối lại đúng nguồn khi biên tập.

| Phần để điền | Nơi dùng trong luận văn | Nơi dùng trong paper |
|---|---|---|
| [1. Mở đầu](#mo-dau) | [Chương 1](chapters/introduction.tex) | [Introduction](../paper_hk261/introduction.tex) |
| [2. Đối thoại với nghiên cứu trước](#nghien-cuu) | [Chương 2](chapters/related_work.tex) | [Related work](../paper_hk261/related_work.tex) |
| [3. Lý do thiết kế và ý nghĩa chứng minh](#phuong-phap) | [Phương pháp](chapters/method.tex), [riêng tư](chapters/privacy.tex), [chất lượng](chapters/utility.tex) | [Method](../paper_hk261/method.tex), [Analysis](../paper_hk261/analysis.tex) |
| [4. Lựa chọn thực nghiệm và cách đọc kết quả](#thuc-nghiem) | [Thiết kế và kết quả chính](chapters/experiments.tex), [kịch bản bổ sung](chapters/scenario_results.tex), [thảo luận](chapters/discussion.tex) | [Workload](../paper_hk261/workload.tex), [Evaluation](../paper_hk261/evaluation.tex) |
| [5. Kết luận và công việc tiếp theo](#ket-luan) | [Chương 5](chapters/summary.tex) | [Conclusion](../paper_hk261/conclusion.tex) |
| [6. Tóm tắt, giọng văn và lời cảm ơn](#bien-tap) | [Các trang đầu](chapters/frontmatter.tex) | [Abstract tiếng Việt](../paper_hk261_vi/abstract.tex) và [tiếng Anh](../paper_hk261/abstract.tex) |

<a id="mo-dau"></a>

## 1. Mở đầu — viết thành một lập luận mà bạn muốn đứng tên

Bạn có thể viết liền từ 1.1 đến 1.8 rồi đọc lại như một chương. Mục tiêu là để người đọc hiểu *vì sao câu hỏi tiếp theo nảy sinh từ câu hỏi trước*. Phần này nên đọc được ngay cả với người chưa biết Z, Q hay Geo-I.

### 1.1. Một người dùng đang làm gì?

Bản hiện tại mở bằng ví dụ tìm trạm sạc. Bạn có muốn dùng ví dụ này xuyên suốt hay có một tình huống tìm POI gần với cách bạn hình dung đề tài hơn? Hãy kể từ lúc người dùng cần một thông tin đến lúc họ quyết định đi đâu. Họ đang chạy trên đường nào, cần biết điều gì, và một gợi ý sai hoặc thiếu có hệ quả gì? Chỉ giữ các chi tiết giúp người đọc hiểu nhu cầu. Nếu là tình huống giả định, hãy dẫn vào như một ví dụ.

**Bản viết của tôi — cảnh mở đầu**

> Tôi dùng bản đồ để chọn đường hoặc tìm địa điểm khi đang di chuyển. Một yêu cầu chỉ đường giúp ích ngay lúc đó, nhưng lịch sử các yêu cầu có thể kể một câu chuyện khác. Giả sử một người nhiều lần tới khu vực bệnh viện. Người quan sát có thể phỏng đoán về sức khỏe, dù người dùng cũng có thể làm việc gần đó hoặc tới thăm người thân.

### 1.2. Khoảnh khắc một truy vấn trở thành bài toán quỹ đạo

Tiếp tục chính câu chuyện trên thêm hai lần gửi. Sau lần đầu, nhà cung cấp còn thấy hai hướng di chuyển nào hợp lý? Sau lần kế tiếp, thứ tự, thời gian hoặc đường một chiều có thể khiến một hướng trở nên dễ đoán hơn ra sao? Bạn muốn dùng từ “suy ra” theo nghĩa đoán chính xác một điểm, thu hẹp tập khả năng, hay tăng độ tin cậy vào một hành trình? Chọn cách nói bạn thấy chính xác nhất.

**Bản viết của tôi — điều thay đổi khi quan sát nhiều lần**

> Một gói gần ngã rẽ có thể phù hợp với cả hai nhánh. Gói kế tiếp xuất hiện ở vùng chỉ một nhánh dễ tới trong khoảng thời gian ấy. Điều thay đổi trước tiên là trọng lượng của từng giả thuyết trong mắt người quan sát. Đó là lý do tôi muốn bảo vệ cả quá trình di chuyển.

### 1.3. Bạn đặt ranh giới cho bài toán ở đâu?

Hãy phát biểu bài toán như khi thầy hỏi “em bảo vệ ai, trước ai, trong khoảng thời gian nào?”. Trong ví dụ của bạn, dữ liệu nào người dùng sẵn sàng gửi để có dịch vụ? Dữ liệu nào thiết bị cần giữ? Vì sao bạn lấy *một phiên* làm đơn vị phân tích? Nếu nhà cung cấp đã có tài khoản định danh người dùng hoặc biết trước thói quen di chuyển của họ, phần nào của câu hỏi nghiên cứu vẫn còn ý nghĩa?

**Bản viết của tôi — phát biểu bài toán và giả định quan trọng nhất**

> Tôi xét nhà cung cấp nhìn thấy tọa độ truy vấn theo thứ tự, thời gian và phản hồi trong một phiên. GPS thật, tham chiếu và nhu cầu thật nằm trên thiết bị. Nếu bên quan sát đã biết thói quen của người dùng, câu hỏi vẫn còn là phiên này bổ sung bao nhiêu bằng chứng. Các chuyến lặp lại là động lực nghiên cứu; cận cho nhiều phiên cần hợp thành riêng.

### 1.4. Chất lượng dịch vụ có ý nghĩa gì đối với người dùng này?

Thử mô tả hai POI mà thứ tự “gần nhất” và “ít vòng đường nhất” khác nhau. Người dùng sẽ chọn điểm nào, vì sao? Nếu một cơ chế làm tọa độ gửi đi lệch xa nhưng điện thoại vẫn tìm đúng địa điểm mong muốn, bạn muốn đánh giá nó thế nào? Từ tình huống đó, viết câu nối hai mục tiêu riêng tư và chất lượng vào *cùng một quyết định thiết kế*.

**Bản viết của tôi — tiêu chuẩn của một câu trả lời hữu ích**

> Người đang lái xe có thể ưu tiên một nhà thuốc nằm ngay chiều đi dù một nhà thuốc bên kia dải phân cách ở gần hơn. Tôi chấm các địa điểm thu được theo yêu cầu thật tại GPS thật. Làm các gói truy vấn khó dùng để định vị chỉ có ích cho ứng dụng nếu người dùng vẫn có ứng viên phù hợp.

### 1.5. Vì sao cần đặt lại cách đánh giá và tự xây dựng dữ liệu?

Hãy kể một lần bạn thử đọc hai kết quả nghiên cứu cạnh nhau và nhận ra chúng trả lời hai câu hỏi khác nhau. Nếu bạn không nhớ một tình huống cụ thể, có thể lấy ví dụ “chọn phần tử thật trong tập dummy” và giải thích chuyện gì xảy ra với phép đo ấy khi cơ chế phát các tọa độ truy vấn mà không buộc GPS thật có mặt. Trên cùng một chuyến đi, bạn cần biết thêm những gì ngoài tọa độ để xác định người dùng đáng lẽ phải nhận được POI nào?

**Bản viết của tôi — vướng mắc khi đối chiếu, dữ liệu cần có, lý do chọn mô phỏng**

> Nếu gói chứa một GPS thật cùng dummy thì có thể chấm việc chọn đúng thành viên. Nếu tất cả tọa độ đều dùng để truy hồi, bên quan sát có thể đoán một điểm nằm ngoài gói. Khung chung vì thế phải xuất phát từ điều cần suy ra. Mô phỏng cho tôi cả GPS chuẩn để chấm định vị lẫn mạng đường, thời gian và danh mục để dựng đáp án dịch vụ trên cùng chuyến.

Ở Chương 1, chừng nào người đọc đã hiểu vì sao cần dữ liệu có kiểm soát thì bạn có thể dừng. Tên bộ sinh, cách chia nhóm tuyến và cách chọn bên suy luận dành cho mục 4 của phiếu. Nếu nhận xét về việc thiếu quy trình chung, hãy cho biết bạn rút ra nhận xét ấy từ nhóm công trình nào.

### 1.6. Vì sao lại có mười kịch bản?

Hãy thử giới thiệu các kịch bản qua câu hỏi người quan sát muốn trả lời. Chẳng hạn “người này đã ở đâu?”, “hai chuyến có liên quan gì?”, “nhu cầu tìm kiếm có bị lộ qua nội dung gói không?”. Chọn hai kịch bản dễ bị đánh đồng và giải thích sự khác nhau bằng *điều bên quan sát biết tại thời điểm đoán*. Ví dụ, S1 và S3 có thể chấm cùng một vị trí với hai cửa sổ quan sát khác nhau. Sau đó giải thích vì sao năm tác vụ định vị S1–S3, S9–S10 được chọn cho bảng chính, và bạn muốn người đọc dùng các kịch bản bổ sung để hiểu điều gì.

**Bản viết của tôi — giới thiệu phạm vi và lý do chia nhóm**

> Tôi dùng các kịch bản để xác định quyền quan sát và câu hỏi suy luận trước khi chọn metric. S1 và S3 có thể hỏi về cùng vị trí nhưng đối thủ có một gói hoặc cả phiên. S1–S3/S9–S10 có mục tiêu tọa độ để đối chiếu; các kịch bản còn lại đòi hỏi nhãn liên kết, đích tương lai, nội dung hoặc kênh quan sát phụ.

### 1.7. Câu hỏi nghiên cứu nào dẫn đến lựa chọn Geo-I?

Bản hiện tại có ba câu hỏi. Hãy viết lại từng câu theo cách bạn thực sự muốn được hỏi khi bảo vệ, kèm lý do câu trả lời có ích. Nếu có câu quá rộng, bạn có thể đề xuất cách thu hẹp. Sau đó viết một câu giải thích vì sao một ràng buộc xác suất như Geo-I là điểm khởi đầu hợp lý cho các câu hỏi ấy.

| Ý cần giữ | Cách tôi đặt câu hỏi | Vì sao tôi cần biết câu trả lời |
|---|---|---|
| RQ1 · Mức phân biệt giữa hai hành trình khi quan sát cả phiên | Transcript giúp phân biệt hai hành trình tới đâu? | Độc lập với bộ suy luận cụ thể trong phạm vi định lý. |
| RQ2 · Điều kiện để lựa chọn truy vấn mang lại đáp án tốt ở GPS thật | Khi nào chọn Q đem lại đủ POI ở GPS thật? | Điểm đại diện và chất lượng thật có thể khác nhau. |
| RQ3 · Đánh đổi riêng tư, chất lượng và chi phí so với các đối chứng | Phải đánh đổi gì trên cùng tác vụ? | Đọc riêng tư cùng chất lượng và chi phí. |

**Câu chuyển của tôi — từ câu hỏi sang hướng giải quyết**

> Geo-I cho phép so xác suất của cùng một quan sát giữa hai GPS. Tôi dùng nguyên lý này để xây cận cho những gì máy chủ thấy suốt một phiên, đồng thời kiểm tra xem đáp án POI còn tốt đến đâu.

### 1.8. Bạn nhận phần đóng góp nào là của mình?

Nếu một độc giả đã biết Geo-I và các phép đo như MAE, họ còn học được gì từ công trình này? Bạn có thể nói về cách xác lập tác vụ để các phương pháp khác đầu ra vẫn so sánh được, cách nối các quyết định cục bộ thành một cận toàn phiên, hoặc bằng chứng thực nghiệm về cái giá của việc truy vấn từ nhiều tọa độ. Hãy tự xếp thứ tự. Với mỗi đóng góp, nêu rõ phần kế thừa, việc bạn thực hiện và nơi độc giả có thể kiểm tra.

| Thứ tự tôi chọn | Việc cụ thể tôi muốn nhận là đóng góp | Nền tảng kế thừa | Định lý, sơ đồ, bảng hoặc dữ liệu làm căn cứ |
|---|---|---|---|
| 1 | Khung chung về mục tiêu, quyền quan sát và dữ liệu chấm. | Các mục tiêu riêng tư và phép đo đã có. | Chương 1, 2 và bảng theo scenario. |
| 2 | Thiết kế và phân tích cả transcript. | Geo-I, phép thử riêng tư, hợp thành và hậu xử lý. | Chương 3. |
| 3, nếu cần | Đối chiếu có thể lặp lại. | Ba adaptation, Raw và REM-Z. | Chương 4. |

**Đoạn kết phần mở đầu của tôi.** Nếu độc giả chỉ đọc thêm một chương sau Introduction, bạn muốn họ đọc chương nào trước và với câu hỏi gì trong đầu?

> Tôi muốn người đọc xem khung kịch bản trước, rồi đọc Chương 3 với câu hỏi: các quyết định riêng lẻ của thiết bị dẫn đến bảo đảm cho đúng chuỗi mà đối thủ được thấy như thế nào?

<a id="nghien-cuu"></a>

## 2. Nghiên cứu liên quan — cho thấy quá trình bạn đi tới khoảng trống

### 2.1. Hai công trình nào ảnh hưởng nhiều nhất đến cách bạn nhìn bài toán?

Chọn hai bài bạn thực sự muốn trao đổi, có thể vì một bài cho bạn ý tưởng kỹ thuật còn bài kia khiến bạn đổi cách đánh giá. Với từng bài, hãy viết công bằng về điều tác giả làm tốt, đầu ra của họ và câu hỏi họ giải đáp được. Sau đó chỉ ra **một** giả định hoặc khác biệt khiến bạn cần cách tiếp cận riêng. Có thể xem lại [bảng tổng hợp nghiên cứu](generated/related.tex).

**Bài thứ nhất — tên, điều tôi học được, điểm tôi đi tiếp**

> Gợi ý từ bản thảo: Geo-I của Andrés và cộng sự cho một tiêu chuẩn xác suất để giải thích điều quan sát viên có thể phân biệt. Phần cần làm thêm là đưa tiêu chuẩn ấy qua cơ chế trên đường, những lần đọc GPS và chuỗi truy vấn có trạng thái.

**Bài thứ hai — tên, điều tôi học được, điểm tôi đi tiếp**

> Gợi ý thứ hai: Fake-query insertion giúp thấy sự khác biệt giữa một tập tọa độ và lịch sử nhiều gói có thời điểm gửi. Vì vậy phép so sánh của tôi phải cho đối thủ xem cùng cửa sổ thời gian và đếm đủ gói giả.

### 2.2. Đoạn tổng hợp sau bảng nên dẫn sang phương pháp thế nào?

Giả sử có người đề nghị dùng nguyên bộ đo của một bài đối chứng. Bạn đồng ý dùng phép đo nào khi có đủ đầu vào? Chỗ nào cần đổi câu hỏi thành “bên quan sát ước lượng vị trí sai bao xa” hoặc “người dùng nhận đủ POI đến đâu”? Vì sao giữ cả phép đo chung lẫn phép đo gốc giúp lập luận thuyết phục hơn? Để tránh lặp phần mở đầu, ở đây hãy viện dẫn đích danh một phương pháp và hợp đồng đầu ra của nó.

**Bản viết của tôi — đoạn kết Chương 2**

> Tôi giữ phép đo gốc ở nơi đầu ra và dữ liệu cho phép, ví dụ sai lệch chi phí di chuyển của TransProtect*. Với các dạng gói khác nhau, tôi bổ sung một tác vụ chung: cho đối thủ đoán tọa độ bất kỳ và so với GPS ở cùng mốc. Cách này cho phép bổ sung phương pháp mới mà vẫn biết đang so cái gì.

<a id="phuong-phap"></a>

## 3. Phương pháp — giải thích vì sao mô hình có hình dạng hiện tại

### 3.1. Lời dẫn cho hai sơ đồ

Ở sơ đồ hệ thống, hãy theo một yêu cầu từ điện thoại qua nhà cung cấp rồi trở về màn hình. Bạn muốn độc giả nhìn ngay ra ranh giới nào? Ở sơ đồ thuật toán, hãy trả lời câu hỏi “đã có điểm tham chiếu Z rồi, vì sao máy chủ còn nhận các khung Q?”. Dùng một quyết định cụ thể để giải thích vai trò của belief, chẳng hạn vì sao thiết bị vẫn cân nhắc nhiều hướng di chuyển sau cùng một Z.

**Khi chỉ vào sơ đồ hệ thống, tôi sẽ nói**

> Thiết bị nhận GPS và yêu cầu người dùng. Máy chủ chỉ cần các tọa độ để tìm POI rồi trả về ứng viên. Việc lọc và xếp hạng diễn ra tại thiết bị; chính yêu cầu riêng có thể đổi mà dạng truy vấn gửi đi vẫn giống nhau.

**Khi chuyển sang sơ đồ thuật toán, tôi sẽ nói**

> Z giúp thiết bị quản lý điều đã biết về vị trí và chi phí mỗi lần cập nhật. Belief biểu diễn các trạng thái còn khả dĩ từ lịch sử đã bảo vệ. Từ đó ta lấy mẫu một đoạn Q có ích cho truy hồi và có cận riêng cho việc công bố. Một khung có năm tọa độ truy vấn, với GPS thật không bắt buộc là thành viên.

### 3.2. Quyết định thiết kế nào cần giải thích bằng một chuyến đi?

Ở chuyến đã lưu, các mốc 20 và 40 giây phát tiếp đoạn đã chọn. Đến 180 giây, thiết bị giữ Z mà vẫn có thể chọn đoạn Q mới. Chọn một trong các thời điểm đó và kể hai phiên bản của cùng sự kiện: người dùng đang cần gì và máy chủ thực sự thấy gì? Nếu bạn từng cân nhắc gửi thẳng Z, chọn từng khung Q độc lập, hoặc gửi nhiều loại yêu cầu công khai, vì sao bạn chọn cách hiện tại? Hãy ghi rõ đâu là lý do thiết kế và đâu là lợi ích đã được đo. [Ví dụ nhiều bước](generated/sample_trace.tex)

**Bản viết của tôi — mốc tôi chọn, phương án tôi cân nhắc, lý do quyết định**

> Ở giây 180 của mẫu cache-029, phép thử giữ Z, chi phí tích lũy lên 6 đơn vị và khung Q mới vẫn được gửi. Máy chủ thấy khung và giờ gửi; thiết bị mới biết nó vừa giữ tham chiếu. Trường hợp này cho thấy vì sao nên tách quyết định cập nhật vị trí khỏi quyết định chọn nơi truy hồi.

### 3.3. Trực giác nào nối bốn mệnh đề riêng tư?

Đây là lời dẫn cho chứng minh. Mỗi hàng chỉ cần giải thích vì sao bỏ qua bước đó sẽ khiến kết luận về cả phiên thiếu căn cứ. Phần phát biểu chính thức và phép biến đổi đã có trong [chứng minh](chapters/privacy.tex).

| Bước trong lập luận | Câu hỏi để bạn diễn giải | Cách tôi nói |
|---|---|---|
| Tạo Z trên mạng đường | Điều gì bảo đảm hai điểm thật gần nhau có thể sinh cùng một Z với xác suất đủ gần nhau? Vì sao chọn Z nằm trên đường vẫn cần một lập luận riêng? | Hai GPS gần nhau làm trọng số và mẫu số của REM thay đổi có giới hạn trên cùng miền đường. |
| Giữ hoặc đổi Z | Nếu quyết định ấy đọc GPS, điều gì về chuyển động có thể ảnh hưởng đến những lần công bố sau? Vì sao tính cả quyết định này vào ngân sách phiên? | Kiểm tra giữ hay đổi dùng GPS nên phải tính cả chi phí phép thử, ngay cả khi giữ. |
| Chọn một đoạn Q | Cùng một lịch sử công khai, hai belief có thể ưu tiên các đoạn khác nhau. Việc giữ cùng tập đoạn khả dĩ và giới hạn thay đổi xác suất có ý nghĩa gì? | Một thư viện chung và mức điều chỉnh có cận giúp cùng đoạn ấy luôn khả dĩ dưới các belief được so. |
| Công bố nhiều đoạn | Nếu chuyến đi dài hơn dự kiến, vì sao việc mỗi lần chọn chỉ tiết lộ ít vẫn chưa đủ để kết luận cho cả chuyến? | Nhân các tỷ số có điều kiện qua từng đoạn; lịch phân bổ có tổng hữu hạn cho cận toàn phiên. |

**Đoạn diễn giải sau định lý.** Giả sử nhà cung cấp cân nhắc hai hành trình có xác suất ban đầu bằng nhau. Trong cấu hình đang xét, dưới giả thiết của định lý, cận hậu nghiệm cho một giả thuyết sau khi xem chuỗi trao đổi của một phiên là khoảng 70,91%. Bạn sẽ giải thích con số này thế nào nếu hội đồng hỏi “đó có phải tỷ lệ đoán đúng vị trí của mọi người dùng không?”. Điều gì thay đổi nếu nhà cung cấp biết trước người này thường đi một tuyến nhất định?

> Hai cận cho cùng transcript đến từ lịch sử GPS đã bảo vệ và từ kênh chọn Q. Với hai giả thuyết ngang nhau, dưới các giả thiết của phiên, cận posterior khoảng 70,91% nói về mức độ phiên trao đổi có thể làm nghiêng lựa chọn giữa hai hành trình. Với prior đã lệch hoặc có kênh phụ, phải xét lại odds ban đầu hoặc mô hình quan sát.

### 3.4. Chứng minh chất lượng dịch vụ giúp trả lời điều gì?

Có ba thời điểm để kể câu chuyện: thiết bị *dự kiến* một đoạn Q sẽ hữu ích, máy chủ *trả về* một tập ứng viên, rồi người dùng *nhận được* danh sách xếp hạng ở GPS thật. Hãy chọn một ví dụ trong đó bước đầu có vẻ tốt nhưng danh sách cuối vẫn thiếu POI người dùng cần. Giả sử cache đã có đúng POI ấy từ lần trước, cần điều kiện gì để nó còn hữu ích? Bạn muốn độc giả mang trực giác nào từ đây sang bảng Recall@5? [Phân tích chất lượng](chapters/utility.tex)

**Bản viết của tôi — ví dụ và câu chuyển sang đánh giá**

> Một đoạn Q có thể có điểm dự kiến cao nhưng bỏ sót địa điểm ở nhánh mà xe thực sự đi. Cache chỉ giúp nếu có bản ghi hợp lệ, áp dụng cùng bộ lọc và thứ tự xếp hạng với đáp án tham chiếu. Lập luận toán học cho biết cần điều kiện gì; Recall@5 kiểm tra đáp án ở GPS thật trong dữ liệu.

<a id="thuc-nghiem"></a>

## 4. Thực nghiệm — giải thích lựa chọn và tự đọc bằng chứng

### 4.1. Một phép so sánh công bằng theo bạn phải giữ cái gì giống nhau?

Quay lại khó khăn đã nêu ở mục 1.5. Bây giờ hãy nói cụ thể tại sao cần cùng tuyến, cùng danh mục, đáp án tham chiếu tại GPS thật và cùng cửa sổ thời gian. Một phương pháp chèn thêm gói sẽ được lợi hay chịu bất lợi gì nếu ta ép mọi bên gửi cùng số gói? Vì sao chia theo *nhóm tuyến* quan trọng hơn việc chỉ báo số lần chạy? Phần đối chiếu dùng 48 nhóm tuyến chia 20/8/20; ba bản điều chỉnh từ bài báo có vai trò khác với hai đối chứng GPS thật và REM-Z. [Quy trình](chapters/experiments.tex)

**Bản viết của tôi — quyết định quan trọng nhất khi thiết kế phép thử và lý do**

> Tôi giữ cùng mục tiêu và cùng khoảng thời gian quan sát. Năm gói của một phương pháp chèn thêm gói giả có thể chỉ tương ứng một phần ngắn của chuyến. Ép cùng số gói sẽ vô tình thay đổi độ dài lịch sử.

**Điều mô phỏng cho phép tôi kiểm soát; khía cạnh thực tế còn thiếu**

> Mô phỏng đồng bộ GPS tham chiếu, transcript và POI. Chia 48 nhóm tuyến theo 20/8/20 giúp tách nguồn chuyến giữa các bước huấn luyện, chọn và kiểm thử. Kết quả trên nhóm tuyến này còn cần được đánh giá lại trên dữ liệu độc lập.

### 4.2. Kết quả định vị nào thuyết phục bạn nhất?

Với Fake-query insertion*, tỷ lệ đoán trúng trong 100 m là 53,387% ở S1 và 98,790% ở S3. Với GeoI-BQ, hai giá trị tương ứng đều là 0,188%. Theo bạn, việc xem thêm lịch sử làm thay đổi cách đọc kết quả của **từng phương pháp** thế nào? Hoặc hãy chọn cặp S9/S10 nếu điểm đầu và cuối chuyến là điều bạn quan tâm hơn. Bắt đầu bằng quan sát trong bảng, rồi cho biết cách bạn giải thích nó và điều cần đo thêm để kiểm tra lời giải thích. [Hit100 và MAE](generated/privacy.tex) · [Khoảng tin cậy](generated/privacy_ci.tex)

**Tôi đọc hai cột/hai hàng này như sau**

> Khi gửi trực tiếp Z, Hit100 ở S1 là 18,522%; khi dùng các khung Q, con số còn 0,188%. Ở S10, hai con số lần lượt là 35% và 0%. Còn với Fake-query insertion*, từ một gói sang cả cửa sổ, Hit100 tăng từ 53,387% lên 98,790%. Quyền xem lịch sử có thể thay đổi đáng kể kết quả của cùng một phương pháp.

**Nếu được phản biện kết luận của chính mình, tôi sẽ hỏi**

> Tôi muốn biết kết quả thay đổi ra sao nếu bộ suy luận khai thác lịch sử tốt hơn. Đối chứng REM-Z và GeoI-BQ khác cả số tọa độ lẫn cơ chế công bố. Những con số này ủng hộ thiết kế chung, còn đóng góp riêng của belief cần so với một cơ chế cũng gửi Q.

### 4.3. Bạn có chấp nhận đánh đổi chất lượng và chi phí này không?

GeoI-BQ đạt Recall@5 toàn phiên 99,595%, thấp hơn ba bản điều chỉnh từ bài báo khoảng 0,4 điểm phần trăm. Mỗi phiên 600 giây có 31 gói, chứa tổng cộng 155 lượt truy vấn tọa độ. Ở mục đích “trong bán kính”, Recall@5 là 99,388%. Quay về người dùng ở mục 1.1: một POI bị bỏ sót có thể quan trọng trong hoàn cảnh nào? Người duyệt chi phí vận hành dịch vụ còn muốn biết gì ngoài số lượt truy vấn? Bạn có thể kết luận “chưa đủ dữ liệu để quyết định” và giải thích cần thêm đại lượng nào. [Kết quả theo mục đích](generated/utility.tex) · [Chi phí](generated/cost.tex)

**Bản viết của tôi — đánh đổi dưới góc nhìn người dùng và người vận hành**

> Recall@5 của GeoI-BQ là 99,595%, với 31 gói chứa 155 lượt truy vấn tọa độ trong 600 giây. So với REM-Z, Recall giảm 0,105 điểm phần trăm; khoảng tin cậy ghép cặp bao gồm 0. Tôi xem đây là đánh đổi khả quan trên workload đã thử. Với ứng dụng thật cần biết thêm độ trễ và hậu quả của việc bỏ sót một POI quan trọng.

### 4.4. Kết quả bất lợi nào làm bạn thay đổi cách viết về mô hình?

Ngay gói đầu, thiết bị đã thu được trung bình khoảng 91% danh mục tĩnh. Phương án nền công khai có Recall nhỉnh hơn phương án thích nghi theo belief. Với dữ liệu hiện có, bạn sẵn sàng nhận định điều gì về **khả năng truy hồi POI** và điều gì về **lợi ích riêng của việc thích nghi**? Nếu thiết bị đã có đầy đủ danh mục tĩnh, trường hợp sử dụng nào còn cần truy vấn trực tuyến? Bạn sẽ thiết kế lần kiểm tra tiếp theo thế nào để phân biệt ảnh hưởng của cache, danh mục và chính sách chọn đoạn? [Phân tích thành phần](chapters/discussion.tex)

**Điều tôi từng kỳ vọng, điều tôi quan sát được, và kết luận tôi chọn**

> Nền công khai có Recall 99,711%, nhỉnh hơn bản thích nghi 99,595% trên cohort này. Gói đầu đã thu gần 91% danh mục tĩnh. Tôi tin vào khả năng giới hạn tiết lộ khi thích nghi; số liệu hiện có chưa cho thấy thích nghi giúp tăng Recall. Nếu thử lợi ích online, nên dùng danh mục động với thông tin mới chỉ phía máy chủ biết.

### 4.5. Mỗi kịch bản bổ sung cho phép bạn phát biểu đến đâu?

Điền câu bạn sẵn sàng nói trước hội đồng cho từng hàng. Nếu muốn lược một hàng khỏi paper sáu trang, ghi lý do bên cạnh. Các kết quả đầy đủ có trong [phân tích kịch bản](chapters/scenario_results.tex).

| Kịch bản | Điểm cần cân nhắc | Kết luận của tôi và bằng chứng còn thiếu |
|---|---|---|
| S4 · Liên kết phiên | Nhãn hiện có cho biết hai chuyến có cùng nhóm tuyến. Muốn phân biệt danh tính người với danh tính xe cần quan sát thêm gì? | Kết quả đo liên kết nhóm tuyến; cần nhãn riêng cho người và phương tiện. |
| S5 · Nhánh đường sắp đi | Trước điểm rẽ, ngay cả GPS thật cũng chỉ đạt 50% trong phép chọn hai nhánh. Phép thử này cho biết gì về độ khó của dự đoán? | Đối chứng yếu trước ngã rẽ nên điểm thấp khó quy cho bảo vệ. |
| S6 · Đích tương lai | Sau điểm rẽ, GPS thật đạt 100%, GeoI-BQ đạt 50,83% khi chọn giữa hai đích đã biết. Vì sao kết quả có ý nghĩa khác S5? | Sau rẽ, chọn giữa hai đích đã biết: 100% với Raw và 50,83% với GeoI-BQ. |
| S7 · Nội dung yêu cầu | Cố định hành trình rồi đổi nhu cầu tại thiết bị giúp kiểm tra kiểu tiết lộ nào? Ý định nào vẫn có thể tương quan với hành trình? | Nhu cầu thật ở thiết bị; ý định tương quan với tuyến vẫn có thể được suy luận. |
| S8 · Người đi cùng | Khi thêm kênh GPS thật đã căn thời gian trong mô phỏng, kết quả thay đổi ra sao? Bạn cần dữ liệu gì để bàn về hai người đồng hành thực tế? | Kênh người đi cùng phát GPS thật có thể chi phối kết quả; cần dữ liệu đồng hành thực. |

### 4.6. Chứng minh và benchmark hỗ trợ nhau ở đâu?

Giả sử một thành viên hội đồng nói: “Kết quả định vị thấp có thể do dữ liệu tự sinh và bộ suy luận còn yếu.” Bạn đồng ý phần nào? Nếu ngày mai xuất hiện một bộ suy luận mạnh hơn, phần kết quả nào cần đo lại? Để biết định lý còn áp dụng hay không, bạn phải kiểm tra giả thiết nào về thông tin mà bên suy luận quan sát được? Đây là đoạn thảo luận bạn có thể viết trực tiếp bằng giọng vấn đáp, sau đó mới chuyển thành văn phong học thuật.

**Câu trả lời đầy đủ tôi muốn đưa ra**

> Em đồng ý dữ liệu và bộ suy luận ảnh hưởng đến kết quả benchmark. Em dựa vào chứng minh để nói về phân phối transcript dưới các giả thiết đã nêu, và dựa vào thí nghiệm để nói về sai số định vị hay chất lượng POI quan sát được. Nếu có bộ suy luận tốt hơn em sẽ đo lại; nếu nó có thêm thông tin, em còn phải kiểm tra cận nào áp dụng cho kênh mới.

<a id="ket-luan"></a>

## 5. Kết luận — trở lại câu hỏi đầu bài

### 5.1. Sau tất cả, bạn có câu trả lời nào cho người đọc?

Đọc lại ba câu hỏi bạn đã viết ở 1.7. Lần này, trả lời bằng một phát biểu mà bạn có thể chỉ ra bằng chứng. Ô cuối dành cho điều kiện khiến câu trả lời phải thay đổi.

| Câu hỏi | Câu trả lời theo lời tôi | Bằng chứng tôi chọn | Phạm vi áp dụng |
|---|---|---|---|
| RQ1 · Riêng tư cả phiên | Có cận cho mức transcript làm thay đổi khả năng phân biệt hai hành trình. | Định lý toàn phiên. | Cùng ngữ cảnh, miền hỗ trợ, trạng thái và kênh công khai hợp lệ. |
| RQ2 · Chất lượng dịch vụ | Chất lượng tại GPS thật phụ thuộc độ phủ và sự phù hợp của điểm lập kế hoạch. | Phân tích cache/xếp hạng và bảng Recall. | Dữ liệu POI hợp lệ và cùng quy tắc xếp hạng. |
| RQ3 · Đánh đổi thực nghiệm | Các bộ suy luận đã thử khó định vị từ Q hơn; chất lượng cao có chi phí truy vấn. | Bảng Hit100/MAE, Recall và chi phí. | Cohort mô phỏng và bộ suy luận được đánh giá. |

**Đoạn kết tôi muốn viết.** Quay lại người dùng ở 1.1. Họ có thể kỳ vọng điều gì từ phương pháp trong điều kiện đã thử? Theo bạn, ý nghĩa của công trình đối với người tiếp tục nghiên cứu bài toán này là gì?

> Tôi muốn để lại một khung mà người làm tiếp có thể bổ sung kịch bản hoặc bộ suy luận rồi xác định ngay điều gì được quan sát, cái gì phải đoán và dùng dữ liệu nào để chấm. GeoI-BQ là một phương pháp cụ thể có thể phân tích và kiểm tra trong khung ấy.

### 5.2. Nếu tiếp tục đề tài, bạn sẽ kiểm tra điều gì đầu tiên?

Chọn giới hạn ảnh hưởng lớn nhất tới kết luận *theo đánh giá của bạn*. Có thể là danh mục tĩnh, chất lượng khi ngân sách GPS cạn, độ bền khi khôi phục trạng thái, hoặc khả năng khái quát sang thành phố khác. Hãy mô tả dữ liệu cần thu, phương án nào so với phương án nào, và kết quả nào sẽ khiến bạn thu hẹp hoặc sửa một kết luận của luận văn.

**Giới hạn tôi ưu tiên, phép kiểm tra tôi đề xuất, kết quả có thể làm tôi đổi ý**

> Tôi ưu tiên nghiên cứu thêm các mô hình suy luận. Có thể cố định dữ liệu công bố, mục tiêu và ngân sách, sau đó thay lượng lịch sử hoặc thông tin phụ mà bên quan sát được phép dùng. Tôi sẽ bắt đầu bằng nhóm tuyến độc lập và đối chứng GPS thật. Nếu một đối thủ hợp lệ định vị tốt hơn, tôi sẽ cập nhật kết luận thực nghiệm và kiểm tra lại phạm vi giả thiết.

<a id="bien-tap"></a>

## 6. Tóm tắt, giọng văn và phần dành riêng cho luận văn

### 6.1. Bản tóm tắt bạn muốn người đọc nhớ

Sau khi điền các chương, hãy thử viết một đoạn ngắn theo mạch **vấn đề → ý tưởng → bằng chứng → ý nghĩa và điều kiện**. Bạn chọn thành phần nào để giới thiệu phương pháp bằng một câu? Nếu chỉ trích một kết quả thực nghiệm, con số nào phải đi cùng nó để người đọc thấy được đánh đổi? Bạn có thể viết câu đầu và câu cuối trước.

**Tóm tắt theo lời tôi**

> Dịch vụ bản đồ hữu ích nhưng lịch sử các yêu cầu có thể tiết lộ thói quen và nơi thường lui tới. Luận văn tổ chức những rủi ro đó thành các kịch bản có mục tiêu, cửa sổ quan sát và đáp án tham chiếu rõ ràng. Trong khung này, GeoI-BQ dùng tham chiếu đã bảo vệ và các đoạn truy vấn có giới hạn tiết lộ để truy hồi POI. Định lý xét cả phiên; trên năm tác vụ định vị đã thử, Hit100 là 0–0,188%, Recall@5 toàn phiên 99,595% với 155 lượt truy vấn tọa độ mỗi 600 giây.

### 6.2. Điều gì cần được giữ khi cô đọng thành bài báo?

Luận văn có thể kể trọn quá trình suy nghĩ. Trong paper sáu trang, chỗ dành cho ví dụ và thảo luận ít hơn. Hãy đánh dấu đoạn nào ở trên bạn thấy đặc biệt đúng giọng mình; luận điểm nào nhất thiết xuất hiện trong cả hai bản; chi tiết nào chỉ cần ở luận văn. Nếu muốn giữ nguyên một cụm tiếng Việt khi trao đổi với GVHD, ghi nó ở đây cùng ý nghĩa bạn muốn giữ khi dịch.

- Câu hoặc đoạn tôi muốn giữ nguyên trong luận văn — Mỗi kịch bản bắt đầu từ điều người quan sát muốn suy ra.
- Ý nhất thiết giữ khi biên tập bài báo Việt và Anh — Khung kịch bản, bảo đảm toán học có điều kiện, bằng chứng thực nghiệm và chi phí.
- Ví dụ hoặc thảo luận dành riêng cho luận văn — Ví dụ giây 180 và chi tiết diễn giải từng kịch bản bổ sung.
- Thuật ngữ hoặc lối diễn đạt tôi muốn dùng nhất quán — Tham chiếu đã bảo vệ (Z), đoạn truy vấn (Q), toàn bộ chuỗi trao đổi trong phiên (transcript).

### 6.3. Lời cảm ơn và lời cam đoan

Ở lời cảm ơn, bạn nhớ góp ý hay sự giúp đỡ cụ thể nào đã làm công trình thay đổi? Có ai khác bạn thực sự muốn cảm ơn, và vì điều gì? Lời cam đoan là phần bạn tự đọc và xác nhận theo yêu cầu của trường. [Bản hiện tại](chapters/frontmatter.tex)

**Lời cảm ơn theo lời tôi**

> Chờ tác giả xác nhận trải nghiệm và lời cảm ơn muốn giữ.

**Điều tôi muốn chỉnh hoặc tự xác nhận trong lời cam đoan**

> Chờ tác giả tự rà và xác nhận trước khi nộp.

Khi gửi lại, bạn chỉ cần chỉ ra những mục đã viết xong. Tôi sẽ dùng chính lời của bạn để biên tập luận văn và rút phần tương ứng cho paper Việt/Anh. Chỗ nào trong bản nháp còn là giả thuyết hoặc vượt quá phạm vi bằng chứng, tôi sẽ giữ dấu hỏi để bạn quyết định.
