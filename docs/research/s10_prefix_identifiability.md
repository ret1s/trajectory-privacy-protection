# S10.B: khó đoán đích chưa chắc là tác dụng của bảo vệ

S10.B hiện cho đối thủ xem đoạn trước khi các tuyến tách ra, còn đích nằm trong
phần bị giấu. Tập mở rộng có tám nhóm đủ điều kiện. Với GPS thật, attacker học
được chọn trên auxiliary có MAE khoảng **1.367 m**, Hit100/Hit200 bằng **0%** và
Hit500 **12,5%**. Với bản objective top-10 + pacing, các số tương ứng là 1.593 m,
0%, 0% và 9,375%. Đây chưa phải phép thử đủ nhạy để chứng nhận bảo vệ đích.

Nguyên nhân cần phân biệt với một attacker triển khai yếu: hai hành trình có
thể giống nhau tới ngã rẽ nhưng đi tới hai đích khác nhau. Biết chính xác đoạn
đầu không đồng nghĩa biết chắc lựa chọn tương lai. Tập auxiliary hiện cũng
không cung cấp lịch sử cá nhân hay ý định đến đích của người dùng.

## Mệnh đề có thể phát biểu chính xác

Giả sử hai thế giới có **cùng GPS và đồng hồ quan sát tới thời điểm τ**, chỉ
khác phần hành trình sau τ. Với một defender nhân quả, phân phối transcript
tới τ giống nhau trong hai thế giới. Nếu hai đích cách nhau hơn 2r và có xác suất
tiên nghiệm bằng nhau, một dự đoán điểm chỉ có thể nằm trong bán kính r của tối
đa một đích. Thành công trung bình do đó không quá 1/2 nếu không có thông tin
phân biệt khác. Điều này đúng cả với GPS thật; nó không phải contribution
privacy riêng của phương pháp đề xuất.

Mệnh đề trên là một lập luận về cặp thế giới lý tưởng, **không phải cận đã chứng
minh cho các trace SUMO đang đo**. Các trace thực có thể khác tốc độ, số mẫu hoặc
thời điểm cuối prefix. Không được tự thay chúng bằng hai prefix hoàn toàn giống
nhau rồi công bố đó là kết quả thực nghiệm gốc.

## Cách giữ S10.B trong báo cáo

- Giữ đầy đủ metric, raw control và giới hạn suy luận; không xóa case khó.
- Việc defender không dùng suffix hay đích tương lai là một thuộc tính đúng đắn
  cần kiểm tra, không chứng minh riêng rằng nó tăng privacy ở S10.B.
- Muốn kiểm tra rủi ro dự đoán đích từ thói quen, cần đặc tả nguồn lịch sử/đích
  ứng viên mà đối thủ được biết, huấn luyện trên nguồn đó và kiểm tra raw trước.
  Đó là một phép thử có thông tin phụ trợ rõ ràng, không lén cung cấp nhãn test.
- S10.A/C hiện có raw Hit500 lần lượt 100% và 91,67%; các phép thử này nhạy hơn
  ở thang 500 m. Vẫn phải báo Hit100/200, MAE và mọi giới hạn attacker.

Nguồn: `iteration18_expanded_attacks.json`; đặc tả record trong
`artifacts/datasets/research_loop_expanded_v1/dataset.json`. Mọi learner chỉ fit
trên 64 nhóm auxiliary và chọn trên 16 nhóm khác. Chưa có xác nhận trên site/map
độc lập và chưa mô hình hóa likelihood theo đường/ngữ nghĩa POI đầy đủ.
