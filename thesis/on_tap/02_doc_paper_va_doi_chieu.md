# Đọc paper để trả lời hội đồng

**Bài tập có ích nhất:** vẽ đầu ra của ba phương pháp đối chứng lên cùng trục thời gian. TransProtect phát một tọa độ thay thế. Semantic correlation phát GPS thật cùng các điểm giả. Fake-query insertion còn chèn cả gói toàn điểm giả. Sau đó thêm hàng thứ tư cho GeoI-BQ: các điểm Q dùng để tìm địa điểm, GPS thật không bắt buộc là một thành viên. Chỉ từ hình này bạn đã giải thích được vì sao bốn phương pháp cùng tính được sai số định vị, còn phép đo nhận diện thành viên thật chỉ áp dụng trong những điều kiện nhất định. [Bảng đối chiếu của luận văn](../generated/related.tex)

## Những bài cần đọc sâu

Mỗi bài dành một thẻ giấy. Mặt trước ghi câu hỏi ở cột cuối. Mặt sau ghi ai quan sát, thấy gì, định nghĩa thành công, và luận văn dùng lại hoặc thay đổi thành phần nào.

| Nguồn và điểm dừng khi đọc | Trả lời được câu gì sau khi đọc? |
|---|---|
| Chow–Mokbel. Xem phần phân biệt LBS liên tục và phát hành quỹ đạo. Đọc cùng Casper và bài truy vấn liên tục trong chuyên đề 1. [Tổng quan 2011](https://www-users.cse.umn.edu/~mokbel/papers/SIGKDD2011.pdf) | Vì sao cùng một tuyến thật lại cần hai cách bảo vệ khi một bên nhận lần lượt từng yêu cầu, còn bên kia nhận bộ dữ liệu đã công bố? |
| Andrés và cộng sự. Xem định nghĩa Geo-I trước khi đọc cách lấy mẫu Laplace phẳng. Sau đó đọc bước kiểm tra dự đoán của Chatzikokolakis và cộng sự. [Geo-I](https://arxiv.org/abs/1212.1984) · [Cơ chế dự đoán](https://petsymposium.org/2014/papers/Chatzikokolakis.pdf) | Cái gì là đặc tính phải chứng minh, cái gì chỉ là một cách xây cơ chế có đặc tính đó? Khi nào quyết định dùng lại thông tin cũ cũng tiêu ngân sách? |
| TransProtect. Xem Hình 7 về dự đoán vị trí, phương trình (13) về chi phí và phần đánh giá với VehiTrack. [Bản tác giả](https://arxiv.org/html/2409.09495v1) | Nêu đúng vai trò của VehiTrack và TransProtect. Nếu thay bộ dự đoán bằng Markov, thành phần nào còn so được và con số nào chỉ thuộc thí nghiệm gốc? |
| Semantic correlation. Đọc §6.3 về ASR, §6.4 về DER. [Bài của Liu, Peng và Zhou](https://link.springer.com/article/10.1007/s44443-026-00899-w) | Một điểm giả quá xa có thể dễ bị loại vì lý do gì? Vì sao một điểm giả đạt tiêu chí ngữ nghĩa vẫn có thể trả sai danh sách nhà thuốc cho người dùng? |
| Fake-query insertion. Xem §6.5 và Hình 10 về các kiểu suy luận, rồi §6.6 về chi phí. [Bài của Liu, Hu và Zhou](https://link.springer.com/article/10.1007/s44443-025-00438-z) | Khi đồng hồ và lịch hỏi thực sự là thông tin công khai, giả định nào của phép thử thay đổi? Ai chịu chi phí của những gói toàn giả? |
| EV querying. Đọc giả thiết vùng người lái quan tâm, approximate Geo-I, chỉ số Cost of Privacy và thao tác chọn trạm sạc. [Atmaca và cộng sự](https://wrap.warwick.ac.uk/id/eprint/183198/) | Nếu ý tưởng “tạo nhiều truy vấn rồi chọn ở thiết bị” đã có, em đứng ở đâu so với bài này? Hãy chỉ vào một giả thiết hoặc định lý riêng của GeoI-BQ. |

## Câu hỏi theo paper

### Ba đối chứng chính

| Câu hội đồng có thể hỏi | Hạt nhân câu trả lời |
|---|---|
| **TL01. VehiTrack và TransProtect, cái nào là phương pháp bảo vệ?** | VehiTrack suy ra vị trí xe từ quan sát và bối cảnh di chuyển. TransProtect chọn miền tọa độ thay thế hợp lý hơn trên đường, rồi thực hiện làm mờ. Mỗi cái đứng ở một phía của phép đánh giá. |
| **TL02. Tại sao mạng đường vừa giúp bảo vệ vừa giúp suy luận?** | Cùng một bản đồ giúp bên quan sát loại vị trí xe không thể tới kịp, và giúp bên bảo vệ tránh đưa ra các lựa chọn vô lý. Trong GeoI-BQ, còn phải kiểm tra việc tạo miền khả dụng cho Q có phụ thuộc bí mật hay không. |
| **TL03. Có tái lập số liệu của TransProtect gốc không?** | Hàng có dấu sao dùng dự đoán Markov trên dữ liệu huấn luyện và môi trường so sánh của luận văn. Sai số tái dựng S3 thuộc bộ suy luận của luận văn; phép sai lệch chi phí đi lại dựa vào (13) khi có một tọa độ thay thế và các chi phí, trọng số đích tương ứng. |
| **TL04. Hai phương pháp cùng ghi K=5 có cùng mức bảo vệ?** | Trong TransProtect, phải đọc K theo bước chọn ứng viên của bài. Với phương pháp dummy, K có thể là số điểm trong gói chứa điểm thật. Với GeoI-BQ, K là số tọa độ truy hồi công bố trong một khung. Số người thật, kích thước miền chọn và số tọa độ được gửi là ba đại lượng khác nhau. |
| **TL05. Hai bài của Liu khác nhau ở quyết định nào?** | Semantic correlation cải thiện *thành viên nào* được chọn làm giả sao cho phù hợp lịch sử và ngữ nghĩa. Fake-query insertion còn thay đổi *khi nào* máy chủ nhận một gói và liệu gói đó có chứa vị trí thật hay không. |
| **TL06. Chính xác DER đang đo gì?** | Bài Semantic correlation nêu trung bình độ tương tự của các điểm giả với điểm thật; phần đánh giá diễn giải tỷ lệ điểm giả đạt tiêu chí tương quan. Muốn tái lập con số phải chốt hàm tương tự, ngưỡng và mẫu số. Một danh sách địa điểm hữu ích cho người dùng là đại lượng khác. [§6.4](https://link.springer.com/article/10.1007/s44443-026-00899-w) |
| **TL07. Có lấy 1 − Hit100 làm ASR của paper được không?** | Hit100 hỏi tọa độ ước lượng có cách GPS thật tối đa 100 m không. Người quan sát có thể chọn sai thành viên trong gói nhưng vẫn cách GPS 20 m. Một lần thử đó thất bại theo tiêu chí nhận diện, song thành công theo Hit100. |
| **TL08. Tại sao giữ mọi gói giả khi chấm Fake-query*?** | Trong 600 giây máy chủ nhận được cả các gói chèn thêm. Bỏ chúng sẽ đồng thời thay bằng chứng người quan sát có và giảm chi phí được ghi nhận. |
| **TL09. Nếu biết trước gói nào là thật thì đánh giá dễ hơn cho ai?** | Bên quan sát có thể bỏ qua nhiều nhánh khả dĩ. Khi thiết kế phép so sánh, cấp đồng hồ công khai đúng như giả thiết rồi kiểm tra xem lịch tự nó tiết lộ được gì. Nhãn thật/giả do bộ sinh biết chỉ dùng để chấm. |
| **TL10. Vì sao kết quả của ba bài gốc không được bê nguyên vào bảng chính?** | Cần cùng mục tiêu, cửa sổ quan sát, dữ liệu tham chiếu và cách tính chi phí để quy chênh lệch cho phương pháp. Các số gốc vẫn có giá trị để hiểu tác giả chứng minh điều gì trong điều kiện riêng của họ. |

Để đối chiếu TL01–TL10, xem các đoạn về cấu hình thay thế, bảng các phép đo có thể áp dụng và ba bài gốc. [Thiết lập luận văn](../chapters/experiments.tex) · [Bảng phép đo gốc](../generated/native.tex) · [TransProtect](https://arxiv.org/abs/2409.09495) · [Semantic correlation](https://link.springer.com/article/10.1007/s44443-026-00899-w) · [Fake-query insertion](https://link.springer.com/article/10.1007/s44443-025-00438-z)

### Các hướng gần và các câu hỏi về tính mới

| Câu hội đồng có thể hỏi | Hạt nhân câu trả lời và chỗ đọc |
|---|---|
| **TL11. EV querying đã giải bài toán của em chưa?** | Bài toán trạm sạc dùng approximate Geo-I, dummy và lựa chọn cục bộ. Luận văn của mình phải chỉ ra phần cụ thể bổ sung: chọn một đoạn truy vấn nhiều khung theo phân phối có cận, nối với ngân sách của cả phiên và đo bốn nhu cầu POI trên cùng giao thức. [EV querying](https://wrap.warwick.ac.uk/id/eprint/183198/) |
| **TL12. Với PTPPM, vì sao hai vị trí có thể nhận ngân sách khác nhau?** | Cơ chế xét mức nhạy cảm của địa điểm và tương quan theo thời gian, kết hợp Geo-I với yêu cầu về sai số suy luận. GeoI-BQ đặt giới hạn theo phiên và theo lần chọn đoạn. Nếu chính ngân sách phụ thuộc dữ liệu nhạy cảm, cần đưa mọi hệ quả quan sát được của lựa chọn đó vào mô hình. [PTPPM](https://arxiv.org/abs/2511.21020) |
| **TL13. CPCROK bảo vệ xe. Vì sao S4 của mình chỉ dám nói “liên kết phiên”?** | Trong VANET, đối tượng là các thông điệp phát bằng bí danh và khả năng nối chúng sau khi đổi bí danh. Trong dữ liệu của mình, nhãn chỉ xác nhận hai chuyến thuộc cùng nhóm tuyến. Hai người có thể đi cùng một tuyến. [CPCROK](https://www.mdpi.com/1999-5903/17/4/165) |
| **TL14. DP-FETC cũng sửa đầu/cuối tuyến. Có được so với S9/S10 không?** | DP-FETC tổng hợp tập quỹ đạo, xử lý các điểm đi/đến trên quỹ đạo tương quan. S9/S10 của mình chấm sai số khi đoán đầu/cuối chuyến từ thông điệp trực tuyến. Muốn so phải định lại ai nhận đối tượng nào và lúc nào. Đọc sơ đồ SynFE/PreOD ở Hình 1. [DP-FETC](https://www.aimspress.com/article/doi/10.3934/era.2025293) |
| **TL15. Một đường tổng hợp trông rất giống thật chứng minh được gì?** | Nó có thể phục vụ tốt một tác vụ mô phỏng. Để nói tới riêng tư còn cần xác định che giấu việc một người, một chuyến hay một sự kiện từng có trong dữ liệu gốc; sau đó đánh giá theo đơn vị đó. Đây là câu hỏi trung tâm của SoK. [Buchholz và cộng sự](https://petsymposium.org/popets/2024/popets-2024-0068.php) |
| **TL16. AnotherMe sinh quỹ đạo trực tuyến, vì sao thiếu hàng benchmark?** | Hệ thống gốc làm việc trực tuyến trên thiết bị. Bản thử của nhóm dùng bước căn thời gian cần nhìn tương lai. Vai trò phù hợp là công trình để thảo luận về quỹ đạo ảo; một so sánh trực tuyến đòi phiên bản chỉ dùng thông tin có tại đúng thời điểm. [Bài IEEE](https://ieeexplore.ieee.org/document/10246991/) · [Đối chiếu của nhóm](../../docs/research/2026-10-07_anotherme_theory_comparison.md) |
| **TL17. Nếu làm lại S8 với hai người, DP trên từng người có tự động đủ?** | Người thứ hai có thể cung cấp thông tin về người thứ nhất. Cần nêu rõ thông tin phụ và xác suất có điều kiện sau khi thấy kênh kia. Một kênh cho GPS thật khi hai người cùng vị trí có thể tiết lộ trực tiếp vị trí cần đoán. [Hệ quả hai kênh](../chapters/privacy.tex) |
| **TL18. Đóng góp nào vẫn có giá trị nếu ngày mai có thuật toán Recall tốt hơn?** | Khung quy một mục tiêu suy luận về dữ liệu được quan sát và nhãn chấm, cùng cách nối cận từng cơ chế thành cận cho một phiên. Một phương pháp mới có thể được đặt vào khung đó, với giả thiết và thước đo tương ứng. [Ba câu hỏi nghiên cứu](../chapters/introduction.tex) |

### Những tài liệu hỗ trợ cũng phải biết dùng đúng chỗ

| Câu kiểm tra | Câu trả lời tối thiểu |
|---|---|
| **TL19. Bài *Unique in the Crowd* có phải là mô hình tấn công mình đã chạy?** | Đây là bằng chứng thực nghiệm về tính độc đáo của dấu vết di chuyển khi có ít thông tin phụ. Dùng nó để giải thích động lực nghiên cứu. Một tỷ lệ trên tập dữ liệu của tác giả không phải xác suất nhận diện của phương pháp mình. [Bài gốc](https://doi.org/10.1038/srep01376) |
| **TL20. Từ Dwork–Roth, em dùng hai tính chất nào?** | Hậu xử lý và hợp thành. Việc lọc theo GPS ở thiết bị không làm đổi chứng minh về máy chủ nếu kết quả lọc ở lại thiết bị. Nếu lần nhấp tiếp theo được gửi lên, cần phân tích kênh đó. [Sách của tác giả](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) |
| **TL21. Mironov liên quan gì khi công thức đã chứng minh đúng?** | Tỷ số xác suất có thể hữu hạn trên số thực lý tưởng nhưng gặp đầu ra chỉ một bên sinh được trong triển khai hữu hạn. Chứng minh cần được nối với thuật toán lấy mẫu và nguồn ngẫu nhiên thực. [Bài gốc](https://doi.org/10.1145/2382196.2382264) |
| **TL22. Bản đồ OSM và SUMO có vai trò khác nhau thế nào?** | OSM là nguồn thông tin địa lý; SUMO mô phỏng chuyển động trên mạng đã dựng. Cả hai đều không cung cấp danh tính người thật hay xác nhận ý định họ ghé một địa điểm. [OpenStreetMap](https://www.openstreetmap.org/copyright/) · [Bài giới thiệu SUMO](https://doi.org/10.1109/ITSC.2018.8569938) |
| **TL23. Efron biện minh cho việc lấy mẫu lại từng tọa độ sao?** | Bootstrap là phương pháp đánh giá biến thiên bằng lấy mẫu lại. Người thiết kế thí nghiệm vẫn phải chọn đơn vị phản ánh sự phụ thuộc của dữ liệu. Ở đây các lần chạy chung nhóm tuyến được giữ cùng nhau. [Bài bootstrap](https://doi.org/10.1214/aos/1176344552) |
| **TL24. Geurts và cộng sự có đề xuất một mô hình suy luận vị trí chuyên dụng?** | Họ đề xuất họ cây quyết định ngẫu nhiên hóa Extra Trees. Nhóm đưa phương pháp học này vào bộ ước lượng, chọn đặc trưng và mục tiêu cho thí nghiệm. Cần phân biệt nguồn của thuật toán học và nguồn của giao thức đánh giá. [Bài Extra Trees](https://doi.org/10.1007/s10994-006-6226-1) |

**Lần tự kiểm cuối:** nhờ người khác đọc tên ngẫu nhiên một trong 17 tài liệu tham khảo. Trong 40 giây, hãy nói bài ấy cung cấp *định nghĩa*, *cơ chế đối chứng*, *bằng chứng động lực* hay *công cụ đánh giá*. Nếu bảo rằng paper là đối chứng, nói thêm có thật sự chạy nó trong bảng chính hay chỉ khảo sát. [Danh mục 17 nguồn](../generated/references.tex)
