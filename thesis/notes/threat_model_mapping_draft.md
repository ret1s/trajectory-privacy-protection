# Bảng ánh xạ threat model - bản draft chuyên sâu

## Đáp án và cách đánh giá — lưu từ bảng hợp nhất

Phần này được tạm rút khỏi bảng chính để ưu tiên diễn giải và ví dụ; giữ lại làm tham chiếu khi hoàn thiện đánh giá.

- **S1**: Giữ vị trí thật và thời điểm. Đo sai số vị trí suy luận; kiểm tra đối thủ không nhận các lần gửi khác.
- **S2**: Giữ tâm, khoảng dừng và số lần gửi; kiểm tra ngưỡng ρ,D_min. Đo sai số suy luận điểm dừng khi tăng số quan sát.
- **S3**: Giữ vị trí theo thời gian và tuyến thật; kiểm tra kết nối, hướng, vận tốc, thời gian đi. Đo sai số tái dựng quỹ đạo.
- **S4**: Giữ mã người, cặp phiên cùng/khác người và điểm thường ghé. Đo tỷ lệ liên kết hoặc nhận diện đúng; tách hồ sơ được cấp khỏi đáp án kiểm thử.
- **S5**: Giữ đoạn tương lai, đích và hướng rẽ. Đo mức dự đoán đúng so với chỉ biết thông tin ban đầu; kiểm tra không lộ tương lai.
- **S6**: Giữ truy vấn thật, ý định, POI liên quan và kết quả dịch vụ tham chiếu. Đo tỷ lệ suy luận đúng ý định hoặc địa điểm; FCD đơn thuần chưa đủ.
- **S7**: Giữ thành viên, khoảng đồng hành; xác nhận quan hệ trong chuyển động thực tế. Đo mức tăng khả năng suy luận khi có thêm thông tin nhóm.


Bản tham chiếu cho bảng ánh xạ S1--S7 (nhãn LaTeX `tab:threatmap`) trong `thesis/main.tex`. Bảng chính chỉ giữ diễn giải đơn giản; các mô tả chuyên sâu dưới đây được lưu để hoàn thiện bộ dữ liệu và thiết kế đánh giá, không phải kết quả tấn công đã được xác nhận.

## S1

- **Thông tin đối thủ có:** Một lần gửi vị trí đã bảo vệ, cùng bản đồ và các địa điểm.
  - Chi tiết: Một $z_t$ hoặc $C_t$; bản đồ, POI, phân bố tiên nghiệm (prior).
- **Cách khai thác:** Loại điểm vô lý, ưu tiên nơi người dùng có khả năng xuất hiện.
  - Chi tiết: Bayesian ranking, homogeneity, map filtering.
- **Muốn suy ra:** Người dùng đang ở đâu?
  - Chi tiết: Vị trí hiện tại (location).
- **Cách đánh giá:** Vị trí đoán gần vị trí thật đến đâu?
  - Chi tiết: Xác suất sai số dưới $r$; thứ hạng điểm thật nếu điểm thật thuộc tập gửi.

## S2

- **Thông tin đối thủ có:** Nhiều lần gửi trong lúc người dùng vẫn đứng ở một nơi.
  - Chi tiết: Nhiều output cùng điểm dừng.
- **Cách khai thác:** Gộp các lần quan sát để thu hẹp nơi người dùng đang dừng.
  - Chi tiết: Averaging, MLE, intersection, reuse pattern.
- **Muốn suy ra:** Người dùng đang dừng ở địa điểm nào?
  - Chi tiết: Điểm dừng hoặc POI nhạy cảm.
- **Cách đánh giá:** Điểm dừng đoán được gần điểm dừng thật đến đâu?
  - Chi tiết: Sai số ước lượng tâm điểm dừng; tỷ lệ xác định đúng POI.

## S3

- **Thông tin đối thủ có:** Các lần gửi liên tiếp, thời gian gửi và mạng đường.
  - Chi tiết: Chuỗi $O_{1:T}$; mạng đường, thời gian.
- **Cách khai thác:** Nối các điểm thành đường đi hợp lý; loại bước nhảy quá xa hoặc sai chiều.
  - Chi tiết: Reachability, HMM/Viterbi, map matching.
- **Muốn suy ra:** Người dùng đã đi qua những đâu?
  - Chi tiết: Vị trí và quỹ đạo (location, trajectory).
- **Cách đánh giá:** Đường đi đoán được giống đường thật đến đâu?
  - Chi tiết: EIE/DTW càng thấp hoặc tỷ lệ điểm đúng càng cao thì suy luận càng tốt.

## S4

- **Thông tin đối thủ có:** Lịch sử nhiều phiên và hồ sơ phụ trợ nếu được cấp.
  - Chi tiết: Significant locations; lịch sử liên phiên; hồ sơ phụ.
- **Cách khai thác:** Nhận ra nơi thường xuất phát, nơi thường đến và lịch đi lại lặp lại.
  - Chi tiết: Cụm nhà--nơi làm việc, lịch ghé, tính duy nhất của mẫu di chuyển.
- **Muốn suy ra:** Nhà, nơi làm việc ở đâu; người này là ai?
  - Chi tiết: Danh tính và vị trí quan trọng (identity, location).
- **Cách đánh giá:** Có nối đúng các phiên hoặc nhận ra đúng người không?
  - Chi tiết: Tỷ lệ liên kết phiên, tái định danh và suy ra đúng địa điểm quan trọng.

## S5

- **Thông tin đối thủ có:** Đoạn đường đã đi, thời điểm và thói quen di chuyển đã biết.
  - Chi tiết: Prefix, giờ/ngày, history/prior.
- **Cách khai thác:** Dựa vào hướng đi và thói quen để đoán chỗ rẽ hoặc đích đến.
  - Chi tiết: Transition, destination và route model.
- **Muốn suy ra:** Người dùng sẽ đi đâu tiếp?
  - Chi tiết: Vị trí, đích đến hoặc tuyến tiếp theo (next route).
- **Cách đánh giá:** Dự đoán đúng nhiều hơn khi chỉ biết thói quen ban đầu không?
  - Chi tiết: Accuracy/MRR so với đối chứng chỉ dùng prior; xác định rõ đối tượng dự đoán.

## S6

- **Thông tin đối thủ có:** Truy vấn đã công bố, vị trí đi kèm và kết quả dịch vụ.
  - Chi tiết: Nội dung công bố $\widetilde Q_t$, POI, kết quả đi cùng dummy.
- **Cách khai thác:** Đối chiếu nội dung tìm kiếm với địa điểm và hành vi để nhận ra ý định thật.
  - Chi tiết: Semantic consistency, intent classifier.
- **Muốn suy ra:** Người dùng thực sự muốn tìm gì, ở đâu?
  - Chi tiết: Nội dung/ý định truy vấn; vị trí liên quan.
- **Cách đánh giá:** Có đoán đúng nhu cầu tìm kiếm hoặc địa điểm liên quan không?
  - Chi tiết: Độ chính xác phân loại intent; sai số vị trí hoặc tỷ lệ đúng POI.

## S7

- **Thông tin đối thủ có:** Dữ liệu nhiều người hoặc thống kê di chuyển của cộng đồng.
  - Chi tiết: Nhiều user, co-mobility, population statistics.
- **Cách khai thác:** Dùng việc đi cùng nhau hoặc thói quen chung để suy ra thông tin một người.
  - Chi tiết: Population prior, co-location, group correlation.
- **Muốn suy ra:** Vị trí, danh tính hoặc đường đi của một người.
  - Chi tiết: Nhiều target, tuỳ cấu hình tấn công.
- **Cách đánh giá:** Biết thêm dữ liệu nhóm có giúp đoán tốt hơn không?
  - Chi tiết: So sánh cùng tấn công khi có/không có thông tin nhóm; giữ các điều kiện khác cố định.

## Điều kiện sử dụng khi thiết kế dataset

- Chỉ cấp cho đối thủ kiến thức phụ trợ đã khai báo trong cấu hình; giữ dữ liệu thật và nhãn đánh giá tách khỏi dữ liệu công bố.
- Với đầu ra chứa điểm thật, có thể đánh giá khả năng chọn hoặc xếp hạng đúng điểm đó. Với replacement hoặc dummy-only, điểm thật không nhất thiết được gửi; dùng sai số tái dựng thay vì yêu cầu chọn điểm thật trong tập công bố.
- Mỗi kịch bản cần đặc tả dữ liệu mô phỏng, thông tin quan sát và nhãn đáp án thật tương ứng. S4 cần lịch sử liên phiên; S5 cần tách đoạn tương lai khỏi đầu vào đối thủ; S6 cần truy vấn và nhãn ý định; S7 cần quan hệ nhóm hoặc thống kê cộng đồng theo cấu hình.
- SUMO cung cấp chuyển động; các nhãn danh tính, ý định truy vấn và quan hệ nhóm cần được lớp sinh kịch bản quản lý, không mặc nhiên có sẵn từ quỹ đạo.
- Các phương pháp tấn công và chỉ số trong bản draft là lựa chọn thiết kế cần kiểm chứng, không hàm ý tất cả đã được triển khai.
