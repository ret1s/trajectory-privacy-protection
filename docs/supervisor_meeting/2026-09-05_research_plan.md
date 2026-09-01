# Kế hoạch nghiên cứu và phạm vi báo cáo ngày 05/09/2026

- **Thời hạn báo cáo:** 15:00, ngày 05/09/2026.
- **Phạm vi của buổi báo cáo này:** trình bày từ Mục 1 đến hết Mục 6.
- **Mục tiêu chung của luận văn:** cải thiện Geo-Indistinguishability (Geo-I) cho bài toán bảo vệ quỹ đạo trong đô thị, sau đó chứng minh bằng cả phân tích lý thuyết và thực nghiệm rằng cơ chế đạt được sự cân bằng giữa riêng tư và độ hữu ích.
- **Lớp cơ chế được chốt:** cơ chế của luận văn thuộc nhóm **sinh vị trí/quỹ đạo giả (dummy generation)**. Vì vậy, các mô hình SOTA dùng làm đối chứng chính cũng phải thuộc nhóm này; các cơ chế chỉ thêm nhiễu tọa độ, chỉ sinh bộ dữ liệu tổng hợp ngoại tuyến, hoặc chỉ xây dựng mô hình tấn công không được tính là ba đối chứng SOTA chính.

## Phần I — Nội dung trình bày trong buổi báo cáo 05/09

## 1. Vẽ lại kiến trúc bài toán

Kiến trúc cần mô tả rõ các thành phần và luồng dữ liệu sau:

```text
Quỹ đạo thật trên thiết bị người dùng
                 |
                 v
     Cơ chế bảo vệ dựa trên Geo-I
                 |
                 v
  Vị trí/quỹ đạo đã được bảo vệ
                 |
                 v
      Ứng dụng hoặc máy chủ LBS
                 |
                 v
   Kết quả truy vấn POI/route/dịch vụ
```

Kẻ tấn công chỉ quan sát phía sau cơ chế bảo vệ, nhưng có thể kết hợp nhiều nguồn thông tin:

- Vị trí hoặc quỹ đạo đã công bố.
- Mốc thời gian và tần suất báo cáo.
- Các truy vấn được gửi tới ứng dụng.
- Mạng đường, POI và bản đồ đô thị công khai.
- Dữ liệu lịch sử hoặc dữ liệu phụ trợ của cùng người dùng.
- Dữ liệu của những người dùng khác hoặc mobility pattern của một nhóm người dùng.
- Kiến thức về cơ chế bảo vệ và các tham số của nó.

Cơ chế bảo vệ được đặt ở phía người dùng, trước khi vị trí hoặc truy vấn được gửi tới LBS. Kiến trúc phải chỉ rõ dữ liệu nào là bí mật, dữ liệu nào được công bố và thông tin nào kẻ tấn công được giả định là đã biết.

## 2. Thu hẹp phạm vi vào khu vực đô thị

Luận văn không giải quyết mọi loại quỹ đạo trong mọi môi trường. Phạm vi được giới hạn vào một khu vực đô thị có:

- Ranh giới địa lý cố định.
- Mạng đường công khai.
- Các POI như nhà hàng, bệnh viện, trường học, cửa hàng và trạm giao thông.
- Quỹ đạo có tọa độ và mốc thời gian.
- Người dùng có thể di chuyển, dừng lại hoặc quay lại cùng địa điểm.
- Ứng dụng LBS chỉ cần một mức độ chính xác đủ để trả lời truy vấn, không nhất thiết cần tọa độ tuyệt đối chính xác.

Việc giới hạn vào đô thị cho phép ta định nghĩa tập output hợp lệ trên mạng đường, kiểm tra khả năng di chuyển theo thời gian, đo utility bằng truy vấn POI và xây dựng các threat scenario có ground truth rõ ràng.

## 3. Định nghĩa các trường hợp quỹ đạo bị đe dọa

Đây **không phải là phân loại các loại quỹ đạo**. Mục tiêu là định nghĩa các tình huống mà cùng một quỹ đạo có thể bị kẻ tấn công khai thác theo những cách khác nhau.

### 3.1. Công bố một vị trí đơn lẻ

Người dùng gửi một vị trí đã được làm nhiễu để sử dụng LBS. Kẻ tấn công dùng prior, cơ chế đã biết, mạng đường hoặc POI xung quanh để suy luận vị trí thật.

### 3.2. Nhiều báo cáo tại cùng một điểm dừng

Người dùng liên tục báo cáo khi đang ở nhà, nơi làm việc, bệnh viện hoặc một địa điểm nhạy cảm. Nếu mỗi lần tạo nhiễu mới và độc lập, kẻ tấn công có thể kết hợp các báo cáo bằng averaging, Bayesian inference hoặc maximum-likelihood estimation để tiến gần vị trí thật.

### 3.3. Chuỗi di chuyển có tương quan thời gian

Kẻ tấn công quan sát toàn bộ chuỗi thay vì từng điểm độc lập. Mốc thời gian, vận tốc, mạng đường và khả năng di chuyển giữa hai điểm giúp loại các vị trí phi lý và khôi phục quỹ đạo bằng tracking, HMM hoặc map matching.

### 3.4. Quay lại các địa điểm quen thuộc

Các lần quay lại nhà, nơi làm việc hoặc một POI nhạy cảm tạo thành pattern. Kẻ tấn công có thể liên kết các lần ghé thăm, phát hiện significant locations và dùng chúng để suy luận danh tính hoặc thói quen.

### 3.5. Dự đoán điểm đến hoặc tuyến tiếp theo

Từ phần quỹ đạo đã quan sát, kẻ tấn công sử dụng lịch sử di chuyển, road connectivity và mobility pattern để dự đoán next location, destination hoặc next route.

### 3.6. Suy luận từ nội dung truy vấn

Ngay cả khi tọa độ đã được bảo vệ, loại POI hoặc nội dung truy vấn vẫn có thể tiết lộ ý định. Ví dụ, truy vấn bệnh viện, cơ sở tôn giáo hoặc một dịch vụ nhạy cảm có thể cho biết hoạt động hoặc mối quan tâm của người dùng.

### 3.7. Tương quan giữa nhiều người dùng

Kẻ tấn công có thể học prior từ một nhóm người dùng hoặc khai thác việc nhiều người di chuyển cùng nhau. Scenario này cần phân biệt rõ ba khả năng: dùng nhóm để xây mobility prior, dùng nhóm làm anonymity context, hoặc yêu cầu privacy guarantee cho cả nhóm. Ý nghĩa chính xác của “kết hợp group of users” cần được chốt trước khi xây thuật toán.

## 4. Nghiên cứu attack threat models

Với mỗi scenario ở Mục 3, threat model phải trả lời đầy đủ:

1. Kẻ tấn công quan sát được dữ liệu gì?
2. Kẻ tấn công có kiến thức phụ trợ nào?
3. Các báo cáo có thể được liên kết theo người dùng, thời gian hoặc địa điểm hay không?
4. Kẻ tấn công biết cơ chế và tham số bảo vệ đến mức nào?
5. Thuật toán tấn công khai thác thuộc tính nào của quỹ đạo?
6. Điều kiện nào được tính là tấn công thành công?

Các nhóm attack model cần khảo sát gồm:

- Bayesian localization đối với một vị trí.
- Repeated-observation, averaging và MLE đối với điểm dừng.
- HMM/temporal tracking đối với chuỗi vị trí.
- Map matching và road-constrained reconstruction.
- Next-location, destination hoặc route prediction.
- Re-identification từ significant locations và dữ liệu phụ trợ.
- Query-content hoặc semantic inference.
- Group/correlation attack sử dụng dữ liệu của nhiều người dùng.

Kết quả của phần survey phải là một bảng ánh xạ:

```text
Scenario bị đe dọa
    -> dữ liệu attacker nhìn thấy
    -> thuộc tính attacker khai thác
    -> target bị lộ
    -> tiêu chí thành công
    -> defense requirement
```

## 5. Xác định các target cần bảo vệ

### 5.1. Identity

Ngăn kẻ tấn công gắn quỹ đạo hoặc mobility pattern với một người cụ thể. Threat điển hình là re-identification từ cặp nhà–nơi làm việc, các POI thường ghé hoặc dữ liệu phụ trợ.

### 5.2. Location

Ngăn kẻ tấn công khôi phục vị trí hiện tại, điểm dừng hoặc địa điểm nhạy cảm. Đây là target gần nhất với guarantee gốc của Geo-I và repeated-observation threat hiện tại.

### 5.3. Next route

Ngăn hoặc làm giảm khả năng dự đoán điểm tiếp theo, điểm đến hoặc tuyến đường sắp đi từ phần quỹ đạo đã được công bố.

### 5.4. Query content

Ngăn việc suy luận ý định hoặc thuộc tính nhạy cảm từ loại truy vấn và POI mà người dùng đang tìm kiếm.

Geo-I nguyên bản chủ yếu hạn chế khả năng phân biệt các vị trí gần nhau. Nó không tự động bảo vệ đầy đủ identity, toàn bộ route hoặc nội dung truy vấn. Vì vậy report phải phân biệt rõ target nào sẽ được thuật toán trực tiếp giải quyết và target nào được dùng để mô tả giới hạn hoặc hướng mở rộng.

## 6. Tìm các mô hình SOTA để làm đối chứng

Cần chọn ít nhất ba phương pháp gần với bài toán, ưu tiên các công trình trong khoảng ba năm gần đây. **Điều kiện bắt buộc là phương pháp phải sinh vị trí giả, tập vị trí giả hoặc quỹ đạo giả để bảo vệ người dùng trong LBS/quỹ đạo.**

Trong nhóm dummy generation cần phân biệt rõ giao diện đầu ra, vì chúng không tự động so sánh công bằng với nhau:

- Thay vị trí thật bằng một vị trí giả duy nhất.
- Gửi một tập gồm vị trí thật và \(k-1\) vị trí giả.
- Sinh một hoặc nhiều quỹ đạo giả để che quỹ đạo thật.
- Chèn các truy vấn giả hoặc chuỗi truy vấn chỉ chứa dummy.

Các mô hình đối chứng phải đại diện cho những hướng khác nhau, chẳng hạn:

1. Dummy generation có xét mạng đường hoặc khả năng di chuyển.
2. Dummy generation có xét tương quan thời gian giữa các truy vấn.
3. Dummy generation có xét POI/ngữ nghĩa để dummy khó bị lọc.
4. Nếu khả thi, một mô hình sinh quỹ đạo giả dựa trên học máy.

Tiêu chí chọn SOTA:

- Threat model có giao với các scenario ở Mục 3.
- Cùng thuộc lớp dummy generation và phải ghi rõ giao diện đầu ra.
- Có paper và mô tả thuật toán đủ rõ.
- Ưu tiên có source code hoặc implementation có thể chạy lại.
- Có thể dùng cùng urban map, cùng dữ liệu benchmark và cùng bộ metrics.
- Privacy guarantee và privacy budget có thể căn chỉnh để so sánh công bằng.

Planar Laplace và các baseline không sinh dummy vẫn có thể được giữ để kiểm tra nền tảng hoặc làm ablation, nhưng không thay thế yêu cầu so sánh với ít nhất ba mô hình SOTA thuộc lớp dummy generation. Nếu các phương pháp dùng giao diện khác nhau (một dummy so với \(k-1\) dummy), phải tách nhóm thí nghiệm hoặc chuẩn hóa thêm chi phí truyền thông, số truy vấn và utility; không gộp trực tiếp vào một bảng rồi kết luận hơn/kém.

> **Prototype demo ngày 01/09/2026:** đã có bản chạy sơ bộ cho
> TransProtectLite, AnotherMeLite, SemanticDummyLite và kiến trúc
> Geo-I-anchor + dummy-only của luận văn. Đây chỉ là paper-inspired sketches để
> minh họa ba output contract, chưa phải reproduction hoặc kết quả SOTA. Lệnh
> chạy, boundary public/truth và giới hạn được ghi tại
> [sota_demo.md](2026-09-05/sota_demo.md).

---

**Điểm dừng của report ngày 05/09/2026:** trình bày đến hết Mục 6. Phần bên dưới là kế hoạch nghiên cứu sau khi kiến trúc, scenario, target và comparator đã được xác nhận.

## Phần II — Công việc sau buổi report

## 7. Cải thiện Geo-I và kết hợp thông tin nhóm người dùng

Sau khi chốt threat model, ta mới xác định cơ chế cần cải thiện Geo-I theo hướng nào. Các khả năng đang xem xét gồm:

- Output hợp lệ trên mạng đường đô thị.
- Sử dụng tương quan thời gian và vùng có thể tiếp cận.
- Hạn chế việc báo cáo lặp tạo ra vô hạn mẫu nhiễu độc lập.
- Quản lý privacy budget qua nhiều lần công bố.
- Bảo vệ các điểm dừng và significant locations.
- Kết hợp thông tin của một nhóm người dùng theo một định nghĩa rõ ràng và không làm rò thêm dữ liệu cá nhân.

Đóng góp cuối phải được phát biểu hẹp: giải quyết threat nào, bảo vệ target nào và không bảo vệ target nào.

## 8. Chứng minh lý thuyết và đánh giá thực nghiệm

### 8.1. Chứng minh lý thuyết

Cần chứng minh probability bound dựa trên Geo-I, ví dụ:

\[
\Pr[M(x)=z]
\leq
e^{\varepsilon d(x,x')}
\Pr[M(x')=z].
\]

Nếu cơ chế có trạng thái, reuse, composition theo thời gian hoặc group of users, phải bổ sung định nghĩa adjacency, transcript và privacy budget tương ứng. Thực nghiệm không thay thế formal proof.

### 8.2. Đánh giá thực nghiệm

Thực nghiệm cần cho thấy tại cùng privacy budget hoặc cùng utility:

- Kẻ tấn công có sai số cao hơn hoặc xác suất thành công thấp hơn.
- Ứng dụng vẫn giữ được kết quả POI cần thiết.
- Vị trí công bố hợp lý trên mạng đường.
- Chuỗi vị trí không tạo ra chuyển động phi thực tế.
- Cơ chế được so sánh công bằng với ít nhất ba SOTA comparator.

## 9. Tìm trajectory-simulation tool và xây dữ liệu benchmark

> **Kết quả khảo sát ngày 01/09/2026:** chọn **SUMO** làm movement engine trên OSM; dùng **OMoSim** như demand generator tùy chọn cho multi-day activity/population; tự xây lớp Python cho scenario, query, group, protection, attack và metrics. Không tool nào tự cover toàn bộ S1--S7, nhưng hiện chưa cần viết cả trajectory simulator từ đầu. Ma trận coverage, kiến trúc dữ liệu và tiêu chí dừng được ghi tại [simulation_backtest_feasibility.md](2026-09-05/simulation_backtest_feasibility.md).

Dữ liệu benchmark có kiểm soát sẽ được xây dựa trên một trajectory-simulation tool phù hợp với bài toán. Quy trình dự kiến:

1. Khảo sát các công cụ mô phỏng quỹ đạo hoặc urban mobility hiện có.
2. Kiểm tra công cụ có sinh được các scenario đã định nghĩa ở Mục 3 hay không.
3. Chọn công cụ và cố định cấu hình đô thị, mạng đường, population, seed và thời gian mô phỏng.
4. Sinh dữ liệu benchmark có đầy đủ ground truth cho vị trí, danh tính mô phỏng, route, query và quan hệ nhóm nếu cần.
5. Dùng cùng dữ liệu đó để chạy cơ chế đề xuất và các SOTA comparator.

Một công cụ chỉ được xem là đủ nếu hỗ trợ hoặc cho phép cấu hình:

- Mạng đường đô thị thực hoặc tương thích OSM.
- Nhiều người dùng/agent và, nếu cần, các nhóm có tương quan.
- Tọa độ, timestamps, route và mode di chuyển.
- Điểm dừng, quay lại địa điểm cũ và báo cáo lặp.
- Điểm đến, next route và truy vấn POI theo scenario.
- Ground truth đầy đủ để chấm attack success.
- Seed cố định và khả năng tái lập thí nghiệm.
- Xuất dữ liệu ở định dạng có thể đưa vào evaluation pipeline.

Nếu không có tool nào tạo đúng các threat scenario, ta sẽ:

- Mở rộng một simulator hiện có bằng scenario generator riêng; hoặc
- Tự xây trajectory generator trên urban road graph.

Generator tự xây phải tạo được từng scenario theo cấu hình rõ ràng, thay vì chỉ sinh các tuyến ngẫu nhiên. Ví dụ, nó phải điều khiển được điểm dừng nhạy cảm, số lần báo cáo lặp, lịch sử quay lại, nhóm người dùng, destination và query content.

Dữ liệu mô phỏng là benchmark chính để kiểm tra từng threat scenario vì có ground truth và có thể kiểm soát điều kiện. Dataset thực như GeoLife có thể được dùng thêm để kiểm tra tính thực tế và khả năng tổng quát hóa, nhưng không thay thế benchmark mô phỏng nếu nó thiếu nhãn scenario cần thiết.

## 10. Metrics privacy và utility

> **Kết quả audit backtest ngày 01/09/2026:** pipeline hiện tại chỉ hỗ trợ một điểm vào/một pseudolocation ra. TransProtect là comparator gần contract này nhất; AnotherMe cần trajectory adapter; các cơ chế `real + K-1` và fake-query cần backtest v2 với candidate/event transcript. Geometry metrics có thể tái sử dụng có điều kiện, nhưng top-1/MRR, dummy survival, semantic metrics, route cost và communication overhead phải bổ sung. Không dùng attacker proxy hiện tại để xếp hạng SOTA. Chi tiết tại [simulation_backtest_feasibility.md](2026-09-05/simulation_backtest_feasibility.md).

### Privacy

- Expected inference error.
- Attack success probability theo bán kính.
- Reconstruction error theo số báo cáo lặp.
- HMM/tracking error.
- Re-identification accuracy nếu identity nằm trong scope.
- Next-location/route prediction accuracy nếu next route nằm trong scope.
- Query inference accuracy nếu query content nằm trong scope.

### Utility và realism

- POI Top-k preservation.
- Mean/P95 displacement.
- On-road rate.
- Speed/reachability violation rate.
- Hausdorff hoặc DTW nếu cần đánh giá hình dạng toàn quỹ đạo.

Metrics phải được chọn theo target và scenario; không dùng một QoS radius tự định nghĩa làm kết luận duy nhất. Kết quả cuối phải trình bày trade-off privacy–utility tại cùng privacy budget và, khi cần, tại cùng mức utility.

## Các điểm cần xác nhận với giáo viên hướng dẫn

1. Trong bốn target identity, location, next route và query content, target nào là đóng góp thuật toán chính?
2. “Kết hợp group of users” nghĩa là population prior, anonymity context hay group-privacy guarantee?
3. Yêu cầu “SOTA trong ba năm” được tính theo năm xuất bản nào và có bắt buộc source code hay không?
4. Dữ liệu mô phỏng sẽ là benchmark chính, còn dataset thực chỉ dùng validation, có đúng với kỳ vọng của thầy không?
5. Simulator cần bao phủ tất cả scenario hay chỉ các scenario thuộc target được chọn làm trọng tâm?
