# Ôn nghiên cứu liên quan và bối cảnh triển khai · GeoI-BQ

**Câu hỏi xuyên suốt:** Một người vừa rời nhà, liên tục tìm trạm sạc khi lái xe. Ai quan sát được gì qua các lần tìm? Phương pháp bảo vệ đại lượng nào? Người dùng có còn chọn được trạm phù hợp? Dùng cùng tình huống này để kể lại mỗi paper. Khi đổi sang bài về công bố dữ liệu, hãy đổi vai người nhận thành người tải cả bộ quỹ đạo.

Tài liệu ôn ngày 25/11/2026 bao quát 17 nguồn trong [thư mục tài liệu của luận văn](generated/references.tex). Các nguồn đọc mở rộng và ví dụ sản phẩm dưới đây dùng cho vấn đáp. Chương 2 khảo sát có chọn lọc; bảng thực nghiệm chính có ba bản điều chỉnh từ paper. [Chương 2](chapters/related_work.tex) · [Bảng cơ chế và phép đo](generated/related.tex) · [Quy trình đối chứng](chapters/experiments.tex).

## 1. Bài toán phát triển theo những câu hỏi nào?

| Mốc | Tình huống bộc lộ vấn đề | Điều cần hiểu khi chuyển sang câu hỏi tiếp theo |
|---|---|---|
| 2003 · Ẩn một cá nhân trong vùng có nhiều người | Một truy vấn chính xác có thể tiết lộ ai đã hỏi. Spatial/temporal cloaking giảm độ chính xác không gian hoặc thời gian đến khi vùng công bố chứa đủ người dùng theo yêu cầu. | Bảo đảm đếm người phụ thuộc mật độ và giả định về bên trung gian. Năm **người** trong một vùng, năm **dummy** trong gói và năm **vị trí truy hồi** Q là ba đại lượng khác nhau. [Gruteser & Grunwald, MobiSys](https://www.usenix.org/legacy/event/mobisys03/tech/full_papers/gruteser/gruteser_html/index.html) |
| 2013 · Nhận ra sức mạnh của lịch sử | de Montjoye và cộng sự cho thấy bốn quan sát không gian–thời gian đủ làm dấu vết của 95% cá nhân trở nên duy nhất trong bộ dữ liệu và độ phân giải họ xét. | Tính duy nhất đo khả năng phân biệt dấu vết trong tập đã biết. Muốn gán tên thật cần thêm thông tin. Với S4 của luận văn, nhãn cùng nhóm tuyến chỉ cho phép đánh giá liên kết theo nhóm. [Unique in the Crowd](https://www.nature.com/articles/srep01376) |
| 2013 · Phát biểu bảo đảm theo khoảng cách | Geo-I hỏi cùng một đầu ra có thể xuất hiện dưới hai GPS với xác suất chênh nhau bao nhiêu. Người nhận vẫn có thể biết đại khái khu vực để phục vụ truy vấn. | Có thể kiểm tra một cơ chế bằng tỷ số xác suất với mọi sự kiện đầu ra. Khi người dùng di chuyển, phải xác định các lần đọc mới và thông tin tích lũy. [Andrés và cộng sự, CCS](https://arxiv.org/abs/1212.1984) |
| 2014 · Tận dụng tính liên tục của di chuyển | Dấu vết ở các mốc gần nhau có thể dự đoán từ các kết quả đã công bố. Phép thử riêng tư quyết định khi nào cần quan sát mới. | Quyết định giữ hay đổi cũng phụ thuộc GPS. Đây là cầu nối từ Geo-I tới ngân sách cả phiên trong luận văn. [Chatzikokolakis và cộng sự, PETS](https://petsymposium.org/2014/papers/Chatzikokolakis.pdf) |
| 2024–2026 · Hỏi thêm về đường, nghĩa của địa điểm, thời gian và dịch vụ | TransProtect xét tính hợp lý trên mạng đường; hai bài của Liu xét chuỗi dummy và lịch hỏi; EV querying đánh giá quyết định chọn trạm. | Mỗi bài làm rõ một giả định khác. Muốn đối chiếu, phải cố định đối tượng suy luận, cửa sổ quan sát và đáp án dịch vụ. Xem hồ sơ từng bài ở mục 2. |
| Các hướng song song | VANET quan tâm tới nối lại bí danh xe. Phát hành dữ liệu quan tâm tới việc người khác phân tích cả tập hành trình. Truy hồi riêng tư bằng mật mã thay đổi chính giao thức máy khách–máy chủ. | Cùng nói về quyền riêng tư vị trí nhưng người nhận, đầu ra và thước đo khác nhau. Các hướng này vẫn tồn tại đồng thời. Xem mục 3–4. |

## 2. Sáu bài cần trình bày được trên bảng trắng trước 14/12

### Geo-I · Andrés và cộng sự, 2013

**Đọc:** định nghĩa Geo-I, cơ chế Laplace phẳng và ví dụ truy hồi dịch vụ. Với hai GPS x và x′, hãy giữ nguyên một sự kiện đầu ra A và viết P(A|x) ≤ exp(εd(x,x′)) P(A|x′). Khi hai GPS gần nhau, một quan sát chỉ có thể làm cán cân bằng chứng nghiêng đi có giới hạn. ε nhỏ hơn siết cận ở cùng khoảng cách; phân phối vẫn có thể lấy mẫu gần GPS. [Bài gốc](https://arxiv.org/abs/1212.1984)

**Nối vào luận văn:** GeoI-BQ lấy mẫu trực tiếp trên tập trạng thái đường chung bằng REM. Dùng bất đẳng thức tam giác để chặn thay đổi của cả trọng số lẫn tổng chuẩn hóa. Từ một lần tạo Z đến cả chuỗi Q còn cần các mệnh đề về tái sử dụng, chọn đoạn và hợp thành. **Tự hỏi:** nếu hội đồng thay Laplace phẳng bằng một kernel khác, mình phải kiểm tra tính chất nào? [Chứng minh REM](chapters/privacy.tex)

### Predictive mechanism · Chatzikokolakis và cộng sự, 2014

**Đọc:** sơ đồ dự đoán, phép thử chất lượng dự đoán và chiến lược tiêu ngân sách. Giả sử hai quan sát đã bảo vệ gợi ý xe đang đứng chờ đèn. Cơ chế dự đoán vị trí từ lịch sử, kiểm tra dự đoán bằng phép thử riêng tư rồi mới quyết định phát dự đoán hay tạo quan sát có nhiễu mới. Tác giả so với thêm nhiễu độc lập; lợi ích phụ thuộc độ dự đoán được của hành trình và chi phí phép thử. [Bài gốc](https://petsymposium.org/2014/papers/Chatzikokolakis.pdf)

**Nối vào luận văn:** ta dùng tham chiếu cũ làm phương án giữ, và lấy mẫu REM khi cần đổi. Hãy tính trên giấy “tạo lần đầu → kiểm tra rồi giữ → kiểm tra rồi đổi” thành 1 + 1 + 2 đơn vị. **Tự hỏi:** nếu client âm thầm đọc GPS để quyết định *có chạy phép thử hay không*, bước nào trong lập luận ngân sách cần xét lại? [Đặc tả](chapters/method.tex)

### TransProtect · Yadav và cộng sự, 2024

**Đọc:** mô hình suy luận VehiTrack trước, cơ chế TransProtect sau, rồi đến bảng kết quả. VehiTrack sử dụng bối cảnh giao thông và mạng đường để loại những vị trí khó xảy ra. TransProtect học mẫu di chuyển để thu hẹp miền sinh tọa độ thay thế về các vị trí hợp lý. Đọc kết quả sai số suy luận cùng loại nhiễu, mô hình quan sát và phương pháp so sánh. Một prior mạnh có thể giúp định vị tốt ngay cả khi cơ chế có cận tỷ số xác suất. [Bài gốc](https://arxiv.org/abs/2409.09495)

**Nối vào luận văn:** mỗi mốc của TransProtect* phát một tọa độ; ta dùng predictor Markov thay predictor neural. Bảng metrics gốc báo sai lệch chi phí di chuyển Δc khi đủ dữ liệu. MAE S3 dùng bộ suy luận của luận văn; gán số ấy cho VehiTrack sẽ sai xuất xứ. **Tự hỏi:** cùng một điểm công bố, vì sao độ lệch chi phí đi lại tới đích và sai số suy ra GPS có thể khác nhau? [Phương án đã chạy](chapters/experiments.tex) · [Metrics](generated/native.tex)

### Semantic correlation · Liu, Peng và Zhou, 2026

**Đọc:** Hình 1, cơ chế học tương quan trên chuỗi vị trí và mục 6.3 về ASR. Một dummy có thể hợp lý ở thời điểm hiện tại nhưng trở nên dễ loại khi ghép với loại địa điểm và thời gian của các lần trước. Phương pháp học đặc trưng ngữ nghĩa–không gian–thời gian và dùng xác suất chuyển để chọn dummy. Mỗi gói có GPS thật và K−1 dummy. ASR đếm tỷ lệ truy vấn mà theo tiêu chí của bài, nhà cung cấp không nhận ra vị trí thật với xác suất cao hơn 1/K. [Bài gốc](https://link.springer.com/article/10.1007/s44443-026-00899-w)

**Nối vào luận văn:** bản có dấu sao dùng POI công khai và predictor thực nghiệm thay nguồn AMap/LSTM. Đối với GeoI-BQ, GPS thật không bắt buộc nằm trong tập Q; yêu cầu chọn đúng một phần tử trong tập sẽ chấm nhầm bài toán. **Tự hỏi:** tại sao vẫn dùng được Hit100 nếu bộ suy luận trả về một tọa độ nằm ngoài cả năm điểm? [Bảng khảo sát](generated/related.tex)

### Fake-query insertion · Liu, Hu và Zhou, 2026

**Đọc:** ví dụ các truy vấn liên tiếp, quy tắc chèn gói và phép đánh giá tương quan đường đi. Khi lần hỏi tiếp theo diễn ra, các gói cũ đã được gửi. Tác giả chèn vào giữa các lượt hỏi thật một số gói hoàn toàn giả, tạo thêm khả năng giải thích chuỗi quan sát. Vì vậy, câu hỏi “gói nào tương ứng với nhu cầu thật?” trở thành một phần của suy luận. [Bài gốc](https://link.springer.com/article/10.1007/s44443-025-00438-z)

**Nối vào luận văn:** phiên bản road/history giữ các sự kiện giả trong transcript và chi phí. Hit100 của adaptation tăng từ 53,387% ở S1 lên 98,790% ở S3 trên cohort luận văn. Hai con số minh họa giá trị của cửa sổ quan sát đối với bộ suy luận đã thử; hiệu quả nguyên bản phải đọc trong bài gốc. **Tự hỏi:** nếu cho hai phương pháp cùng số gói thay vì cùng thời lượng, bạn đã vô tình thay đổi thông tin nào? [Kết quả theo kịch bản](generated/privacy.tex)

### EV querying · Atmaca và cộng sự, 2024

**Đọc:** lý do giới hạn truy vấn trong vùng người lái quan tâm, định nghĩa approximate Geo-I và Cost of Privacy của bài toán chọn trạm. Nhóm tác giả kết hợp nhiễu với dummy và đánh giá chất lượng quyết định phục vụ xe điện. Cụm “privacy-for-free” của tác giả gắn với điều kiện và chi phí dịch vụ họ đo; hãy xác định chính xác loại chi phí trước khi diễn giải. [Bài gốc và bản PDF tại Warwick](https://wrap.warwick.ac.uk/id/eprint/183198/)

**Nối vào luận văn:** đây là tiền lệ bạn cần chủ động nhắc khi được hỏi về tính mới của “Geo-I + nhiều tọa độ + chọn ở thiết bị”. Hãy chuyển sang điều GeoI-BQ thực sự phân tích: xác suất chọn *đoạn* trên miền chung, tổng mức tiết lộ cả phiên và chất lượng danh sách POI tại GPS thật. **Tự hỏi:** có đủ căn cứ để so Cost of Privacy bằng 0 của bài xe điện với Recall@5 bằng 100% hay không? [Phân tích chất lượng](chapters/utility.tex)

## 3. Những nguồn còn lại trong luận văn thuộc mắt xích nào?

Bốn công trình dưới đây hoàn tất tám bài khảo sát gần đây ở Chương 2. Trước ngày nộp, với *từng bài*, hãy tìm trong nguồn gốc một hình, bảng hay phát biểu và giải thích luận văn dẫn nó ở đâu.

| Nguồn | Cơ chế, bằng chứng và giới hạn khi liên hệ |
|---|---|
| CPCROK, 2025 | Xe thay bí danh và dùng beacon giả trong bối cảnh VANET mật độ thấp. Đọc hình vùng thay bí danh và chỉ số nối lại quỹ đạo. Liên hệ S4 qua khả năng liên kết các quan sát; muốn nói tới danh tính người lái còn cần nhãn và mô hình định danh tương ứng. [Bài gốc](https://www.mdpi.com/1999-5903/17/4/165) |
| DP-FETC, 2025 | Sinh tập hành trình từ đặc trưng và tương quan, có xử lý phân bố điểm đi–đến. Xem privacy nhắm đến dữ liệu nguồn và utility đo mức bảo toàn đặc tính phân bố. Hai tập có thể giống phân bố OD nhưng khác chất lượng POI tại một vị trí cụ thể. [Bài gốc](https://www.aimspress.com/article/doi/10.3934/era.2025293) |
| PTPPM trên mạng đường, phiên bản 2025 được luận văn khảo sát | Ngân sách cá nhân hóa, tương quan thời gian–không gian và Permute-and-Flip cho vị trí đã làm nhiễu. Đọc cách họ đánh đổi khoảng cách dịch chuyển và bảo vệ. Luận văn chưa chạy comparator của work này. [Bản tác giả đã khảo sát](https://arxiv.org/abs/2511.21020) |
| SoK về trajectory generation, 2024 | Đọc cách các tác giả đặt câu hỏi một bộ sinh bảo vệ cái gì và giữ utility cho tác vụ nào. Dùng để phân biệt bảo vệ một người trong dữ liệu huấn luyện, một quỹ đạo công bố và một phiên hỏi đáp đang chạy. SoK cung cấp khung phân tích. [Bài gốc](https://petsymposium.org/popets/2024/popets-2024-0068.pdf) |

Bảy nguồn dưới đây hoàn tất 17 tài liệu của bản thảo. Khi ôn, quan trọng nhất là giải thích vì sao nguồn được dẫn ở đúng chỗ đó.

| Nguồn | Câu phải tự trả lời |
|---|---|
| de Montjoye và cộng sự | Tính duy nhất có điều kiện trên dữ liệu nào? Tại sao kết quả kinh điển này là động lực để xét chuỗi, còn phép thử danh tính người/xe cần nhãn riêng? [Bài gốc](https://www.nature.com/articles/srep01376) |
| Dwork & Roth | Hậu xử lý áp dụng khi bước sau chỉ dùng thông tin đã bảo vệ và đầu vào công khai nào? Hợp thành cần tính phần chi phí nào khi đọc dữ liệu riêng nhiều lần? [Sách của tác giả](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) |
| Mironov | Vì sao các giá trị có thể biểu diễn trong máy tính làm việc hiện thực cơ chế nhiễu trở thành một giả thiết riêng? Liên hệ giới hạn chứng nhận sampler của GeoI-BQ. [Bài gốc](https://www.microsoft.com/en-us/research/publication/on-significance-of-the-least-significant-bits-for-differential-privacy/) |
| Alvarez Lopez và cộng sự, SUMO | Thành phần nào do mô phỏng giao thông sinh, thành phần nào do ta định nghĩa để tạo workload và target? [Bài SUMO](https://doi.org/10.1109/ITSC.2018.8569938) |
| OpenStreetMap contributors | Phần nào là nguồn dữ liệu địa lý, phần nào là mạng và danh mục mà prototype dùng? Đây là nguồn dữ liệu trong danh mục trích dẫn. [Nguồn OSM](https://www.openstreetmap.org/copyright/) |
| Efron | Bootstrap xây dựng độ bất định bằng lấy mẫu lại. Vì sao đơn vị lấy lại trong so sánh chính là nhóm tuyến ghép cặp? Việc chọn đơn vị này thuộc protocol của luận văn. [Bài gốc](https://doi.org/10.1214/aos/1176344552) |
| Geurts, Ernst & Wehenkel | ExtraTrees cung cấp họ mô hình nào cho bộ suy luận? Lựa chọn feature, TRAIN/SELECTION và estimator thực sự thắng cho mỗi loss là quyết định thực nghiệm của ta. [Bài gốc](https://doi.org/10.1007/s10994-006-6226-1) |

**AnotherMe trong quá trình phát triển:** tác giả mô tả một hệ thống sinh quỹ đạo ảo *online*. Bản thử lịch sử trong repo xử lý cả đoạn, có bước căn thời gian dùng thông tin tương lai và có trường hợp lỗi; đó là lý do không đặt nó trong bảng chính cùng sáu phương án. Khi được hỏi về định lý của AnotherMe, phạm vi đã xác minh trong hồ sơ hiện tại chỉ đủ để trình bày ý tưởng và phân biệt bản gốc với adapter. [Bài IEEE](https://ieeexplore.ieee.org/document/10246991/) · [Hồ sơ đối chiếu](../docs/research/2026-10-07_anotherme_theory_comparison.md)

## 4. Đến cuối 2026, học thuật và sản phẩm đã làm được gì?

**Trong học thuật**, đã có cơ chế với cận xác suất theo khoảng cách, cơ chế dự đoán có phép thử riêng tư, phương pháp sinh tọa độ hợp lý trên đường, phương pháp che bằng nhiều gói, và bộ sinh dữ liệu ngoại tuyến. Một ví dụ năm 2026 là Yu và cộng sự: công trình tách truy vấn điểm và truy vấn quỹ đạo, đánh giá trên GeoLife bằng độ chính xác truy vấn và các đại lượng về cấu trúc không gian–thời gian. Khi đối chiếu với luận văn, cần xác định một quỹ đạo được bảo vệ có tương đương chuỗi POI request/reply mà server quan sát hay không. [Yu và cộng sự, Sensors, 2026](https://www.mdpi.com/1424-8220/26/17/5456)

**Một hướng khác là thay giao thức bằng mật mã.** VeLoPIR dùng mã hóa đồng cấu cho truy hồi riêng tư dựa trên vị trí; tác giả nêu các chế độ kiểm tra vùng, tọa độ và định danh, kèm chứng minh theo giả thiết mật mã và đánh giá thực nghiệm. Khi hội đồng hỏi “sao không dùng PIR?”, hãy so bên phải thay phần mềm, các phép truy vấn hỗ trợ, chi phí tính toán và metadata có thể quan sát qua nhiều lần gọi. GeoI-BQ nhắm tới giao diện máy chủ nhận tọa độ truy vấn thông thường. Benchmark luận văn chưa so trực tiếp với PIR. [VeLoPIR, bản tác giả](https://arxiv.org/abs/2506.12761)

**Trong thực tế**, cần hỏi *đang bảo vệ ở ranh giới nào*. Ba ví dụ có tài liệu sản phẩm công khai:

| Tính năng đã triển khai | Giá trị và câu hỏi còn lại cho luận văn |
|---|---|
| Android cho phép người dùng cấp vị trí gần đúng từ Android 12, kể cả khi ứng dụng xin vị trí chính xác. | Kiểm soát độ chính xác ở ranh giới hệ điều hành–ứng dụng. Để phát biểu mức phân biệt hai hành trình từ nhiều request còn cần cơ chế, lịch gửi và giả thiết về server. [Android Developers](https://developer.android.com/develop/sensors-and-location/location/permissions/runtime) |
| Apple Maps mô tả việc gửi thông tin tuyến khi xin chỉ đường và tạo một định danh ngẫu nhiên tồn tại trong phiên điều hướng đó. | Giảm gắn kết trực tiếp với danh tính là một mục tiêu thiết kế; các lượt trong một phiên vẫn có thể cùng phục vụ điều hướng. Tài liệu công khai mô tả dữ liệu và định danh, còn một cận posterior kiểu GeoI-BQ cần đặc tả xác suất riêng. [Apple Maps & Privacy](https://www.apple.com/legal/privacy/data/en/apple-maps/) |
| Google Maps cho tải khu vực và chỉ đường lái xe offline trong vùng đã tải. Khi offline, thông tin giao thông và tuyến thay thế bị giới hạn. | Nếu dữ liệu đã đủ và còn hiệu lực, xử lý cục bộ là đối chứng có ý nghĩa. Lợi ích của truy vấn riêng tư online đáng kiểm tra khi POI hoặc trạng thái dịch vụ thay đổi. Prototype luận văn hiện có danh mục tĩnh gần bão hòa. [Google Maps Help](https://support.google.com/maps/answer/6291838?hl=en) |

**Nhận định để bảo vệ:** có các kết quả mạnh cho từng bài toán cụ thể. Một hệ thống triển khai phải đồng thời xác định thông tin server thấy, quyền truy cập dữ liệu, trạng thái sau gián đoạn, chi phí và chất lượng trên người dùng thực. Các tài liệu sản phẩm trên xác nhận tính năng nhà cung cấp công bố. Chúng không cung cấp bằng chứng rằng nhà cung cấp đang dùng GeoI-BQ hay đáp ứng định lý của luận văn. Phần đã đạt của đề tài là một thiết kế có cận có điều kiện cho transcript và phép đối chiếu có kiểm soát; lợi ích của thích nghi theo belief trên catalogue động vẫn là câu hỏi mở. [Định lý](chapters/privacy.tex) · [Giới hạn thực nghiệm](chapters/discussion.tex)

## 5. Cách ôn và tự vấn đáp

| Khi nào | Việc làm và chuẩn hoàn thành |
|---|---|
| 25–29/11 · A04–A07 | Đọc nguồn của sáu bài ở mục 2. Tự vẽ mỗi phương pháp bằng một hình, lấy đúng một hình/phương trình/thước đo từ bản gốc để giải thích, rồi phân biệt cái ta kế thừa với cái ta thay. |
| 30/11–10/12 · B03–B08 và C02–C07 | Đối chiếu từng mắt xích với định lý hoặc phép đo của mình. Ôn mục 3 theo vai trò trong luận văn. Lấy một dòng kết quả của adaptation, truy ngược về đầu ra được phép quan sát. |
| 11–14/12 · D05–D07 | Nhờ người khác chọn bất kỳ một trong 17 nguồn. Trong 90 giây, nói mục tiêu, ý chính, nơi nguồn được dùng và một giới hạn. Với ba đối chứng, thêm phần triển khai khác bản gốc. Tập cả câu hỏi sản phẩm bằng tài liệu chính thức. |
| 15–20/12 · E05 | Đào sâu chứng minh hoặc phân tích độ nhạy của bài khó nhất. Đọc AnotherMe, VeLoPIR hoặc công trình 2026 để trả lời hướng tiếp theo. Đây là phần mở rộng trong lịch 32/40 trước 15/12, 8/40 còn lại đến 20/12. |

**Phiếu để điền mỗi khi đọc.** Sao chép dòng dưới cho từng bài. Tự tra trang/hình theo đúng bản PDF sử dụng.

| Bài và phiên bản | Người quan sát được thấy gì? | Cơ chế + bằng chứng tác giả đưa ra (trang/hình) | Ta kế thừa/thay gì? | Câu hỏi còn vướng |
|---|---|---|---|---|
| … | … | … | … | … |

**Năm câu buộc phải nối nhiều nguồn:**

1. *VehiTrack suy ra vị trí khá tốt có bác bỏ Geo-I không?* Nêu sự khác biệt giữa một cận thay đổi odds và sai số tuyệt đối của một estimator có prior và bối cảnh mạnh.
2. *Tại sao EV querying là paper phải hiểu dù không có hàng benchmark?* Chỉ ra tiền lệ gần nhất cho nhiều truy vấn và chọn cục bộ, rồi nói phần chọn đoạn và chứng minh toàn phiên của mình.
3. *Có thể so ASR của Semantic correlation với 1 − Hit100 của ta không?* Nêu mẫu số, sự kiện thành công và điều kiện GPS thật có mặt trong gói của từng phép đo.
4. *Nếu Google Maps đã chạy offline, tại sao còn làm đề tài?* Thừa nhận ưu thế của offline khi danh mục đã có đủ. Đặt lại câu hỏi cho dịch vụ cần dữ liệu mới và xác định thí nghiệm còn thiếu để chứng minh giá trị.
5. *Nếu làm tiếp một năm, nên cải thiện dummy, thay bằng PIR, hay giảm số lượt truy vấn?* Chọn theo người cần được bảo vệ, loại máy chủ được phép thay đổi, thông tin còn lộ và phép đo dịch vụ. Câu trả lời tốt nêu được một giả định mà nếu đổi, bạn sẽ chọn hướng khác.
