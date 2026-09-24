# Tiền lệ của lập kế hoạch trước trong bảo vệ vị trí

Rà soát ngày 24/09/2026 bổ sung một dòng nghiên cứu gần với hướng planner đang
thử. **Không thể nhận “dùng dự đoán/MPC để bảo vệ vị trí” là ý tưởng mới.** Đây là
related work cần đối chiếu, chưa phải comparator đã tái lập trong benchmark.

| Công trình | Nội dung xác minh được | Hệ quả cho luận điểm của ta |
|---|---|---|
| Molina et al., IFAC 2023, *Optimal privacy protection of mobility data: a predictive approach* | Tối ưu vị trí công bố cho hiện tại và các bước tới, với giới hạn mất utility; xét thời gian lấy mẫu không đều | Look-ahead và receding-horizon trong location privacy đã có tiền lệ |
| Molina et al., IEEE Control Systems Letters 2023, *React to the Worst* | Học offline từ bài toán MPC, dùng dự đoán trường hợp xấu và lời giải online nhẹ | Không được tự nhận cơ chế dự báo nhẹ là hoàn toàn mới |
| Molina et al., Control Engineering Practice 156 (2025), 106223, *Application of a predictive method to protect privacy of mobility data* | Abstract mô tả ba predictor: biết tương lai làm tham chiếu, dự đoán trường hợp xấu và tuyến tính để chạy online; đánh giá Privamov/Cabspotting | Đây là nguồn gần đây, cần có trong related work của hướng phát triển |

Nguồn chính: [IFAC 2023, nhà xuất bản](https://www.sciencedirect.com/science/article/pii/S2405896323011783),
[LCSS 2023, toàn văn tác giả](https://www.gipsa-lab.grenoble-inp.fr/~mirko.fiacchini/files/23_MolinaLCSS.pdf),
[CEP 2025, nhà xuất bản](https://www.sciencedirect.com/science/article/pii/S0967066124003824),
[trang tác giả xác nhận bản 2025](https://ox217.github.io/publication/molina-2025-apli-react/).
Đã đọc toàn văn LCSS; với CEP 2025 hiện dựa trên abstract và metadata công khai,
chưa xác minh toàn bộ phương trình hay tái lập implementation của bản này.

## Metric gốc và giới hạn so sánh

LCSS định nghĩa privacy bằng độ phân tán bình phương của các vị trí công bố
quanh centroid trong cửa sổ; utility loss là khoảng cách giữa vị trí thật và
vị trí thay thế (mục II, phương trình 1–3). Dịch vụ nhận một vị trí thay thế.

Suy luận của nghiên cứu này: độ phân tán tự nó không chứng minh attacker đoán
khó. Chẳng hạn công bố GPS thật của một xe đang chạy cũng có thể tạo phương sai
lớn. Do đó cần kiểm tra suy luận thực tế và raw control, không chuyển trực tiếp
“privacy tăng” theo metric đó thành “ASR giảm” hoặc bảo vệ endpoint của ta.
Khoảng cách tọa độ cũng không tương đương Recall của hợp các phản hồi POI.

## Phần khác biệt cần kiểm chứng của prototype vòng 21

Prototype chỉ dự đoán từ belief đã bảo vệ và dữ liệu công khai; không đọc GPS
tương lai. Nó chọn **K cặp vị trí hiện tại/tương lai** từ các đường đi có hướng,
tối ưu hợp kết quả dịch vụ ở hai thời điểm, rồi chỉ công bố K vị trí hiện tại.
Privacy primitive và cap được giữ ở lớp neo; planner là hậu xử lý. Đầu ra và
objective khác một vị trí thay thế được tối ưu theo độ phân tán.

Các khác biệt này mới xác định phạm vi so sánh, chưa chứng minh tính mới ưu tiên
hay hiệu quả vượt paper. Phải đo gain so với planner hiện tại, chi phí tính toán,
đánh đổi attacker và các ca không đạt. Không dùng tương lai thật trong một ablation
online, không gọi dữ liệu forecast là ground truth.
