# Vì sao chọn tốt ở bước hiện tại vẫn thiếu dịch vụ?

Chẩn đoán sau vòng 15, trên đúng các phiên đã công bố kết quả. Mỗi replay khớp
toàn bộ prefix output đã lưu trước khi lấy trạng thái để kiểm tra. Vị trí thật
chỉ được dùng ở evaluator; các nghiệm oracle không đi vào defender.

Với K track và các trạng thái trước đó q_j, gọi R_j(Δt) là miền có thể đi tới
trên đường có hướng trong thời gian giữa hai truy vấn. Ta giải bài toán phủ POI
với trọng số từ **top-5 thật**, mỗi track chọn đúng một trạng thái trong R_j.
MILP lưu cả nghiệm khả thi và cận trên; 24/24 bài nhỏ đã giải tới tối ưu.

Kết quả S1.C, trung bình bốn lần chạy ở hai nhóm validation phát triển:

| Bộ chọn | Recall L10 thật | Oracle trong miền đi tới được | Mục tiêu toàn mạng từ belief bảo vệ |
|---|---:|---:|---:|
| Pacing, objective cũ | 75,00% | **75,00%** | 95,00% |
| Objective top-10, chưa pacing | 75,00% | **75,00%** | 73,33% |
| Objective top-10 + pacing | 74,17% | **76,67%** | 98,33% |

Cột oracle cho biết: với lịch sử query track đã tạo và khoảng thời gian ở bước
cuối, chỉ thay bộ chọn cuối cùng không đủ đạt 90%. Đây là cận **có điều kiện trên
lịch sử đó**, không phải chứng minh mọi phương pháp đều không thể đạt 90%.
11/12 mẫu validation trong bảng đã đạt đúng oracle tại bước cuối.

Cột cuối không áp dụng giới hạn đi tới được; nó chỉ chấm Recall thật của bộ mục
tiêu mà planner tự tính từ belief bảo vệ. Với pacing, các mục tiêu này thường
có ích nhưng đang ở quá xa. Do đó cần đánh giá cách di chuyển sớm hơn và khả năng
giữ các lựa chọn tốt cho những bước sau. Không được dùng nghiệm “nhảy tới mục
tiêu” này làm kết quả của thuật toán online có ràng buộc vận tốc.

Belief cũng có sai lệch: tại nhiều mẫu, khối lượng trong 500 m quanh vị trí thật
rất nhỏ, dù objective dự đoán utility cao. Tối ưu expected Recall theo belief
gần đúng không đồng nghĩa bảo đảm Recall của người dùng thật.

Vòng 17 thử mức giảm objective tối đa .03 để tiến về mục tiêu. S1.C tăng tới
82,5%, S3.A tăng tới 94,10%, nhưng S1.B giảm còn 89,17%; chỉ 13/15 ca qua ngưỡng.
Đây là đánh đổi giữa các ca, chưa phải người thắng. Mức .03 giới hạn objective
theo belief tại từng bước, không giới hạn mức giảm Recall thật của toàn chuyến.

Nguồn: `iteration16_s1_oracle_diagnosis.json` và `iteration17_paced_slack_cases.json`
trong `artifacts/benchmarks/research_loop/`. Bốn test oracle đối chiếu với vét cạn
trên bài nhỏ và kiểm tra không tạo ra POI không thể truy hồi. Các nhóm hiện đã
được xem nhiều lần; vòng tiếp theo mở rộng dữ liệu phát triển trước khi chỉnh
thêm tham số. Dữ liệu này vẫn chưa phải confirmation cuối cùng.

Vòng 18 trên 12 nhóm mới xác nhận S1.C vẫn chỉ đạt 72–79% với bốn ứng viên.
Vòng 19 giữ nguyên tổng epsilon nhưng giãn đọc theo ngân sách còn lại. Trên
toàn bộ 12 nguồn S1.C, lần đọc gần nhất mới hơn rõ rệt (tuổi 200 → 98 giây),
nhưng bản không slack giảm Recall và bản có slack chỉ tăng 78,75% → 80,83%,
gain chưa ổn định. Không thể quy toàn bộ thất bại cho ngân sách hết sớm.
Miền đi tới được và độ phù hợp của belief vẫn phải được giải quyết; không tăng
epsilon hoặc bỏ ràng buộc chuyển động rồi gọi đó là cải tiến cùng điều kiện.
