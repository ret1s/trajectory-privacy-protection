# Tách mục tiêu top-k và phản hồi top-L

Đây là sửa sự lệch giữa objective và hợp đồng dịch vụ, không phải tăng lượng
phản hồi để đạt điểm cao. Vòng thử giữ k=5 cho câu trả lời của người dùng, L=10
cho mỗi phản hồi máy chủ và K=5 vị trí truy vấn, giống đối chứng ở L10.

Gọi R_c(s) là top-k POI tham chiếu của category c tại vị trí s; A(s) là số
category có tham chiếu. Gọi T_c,L(q) là danh sách tối đa L POI máy chủ trả cho
truy vấn q. Belief b(s) chỉ được suy ra từ neo đã bảo vệ và bản đồ công khai.

Với mỗi POI p thuộc category c, giữ trọng số tham chiếu:

```
w(p) = Σ_s b(s) · 1[p ∈ R_c(s)] / (A(s) · |R_c(s)|).
F_L(Q) = Σ_p w(p) · 1[p ∈ ∪_{q∈Q} T_c,L(q)].
```

Đổi thứ tự phép cộng cho thấy F_L(Q) chính là expected macro Recall@k của hợp
phản hồi top-L, theo belief gần đúng đó. Trạng thái không có category hợp lệ
đóng góp 0. Nếu điều kiện trên các trạng thái có tham chiếu, chỉ cần chia cho
tổng khối lượng hợp lệ, một hằng số không phụ thuộc Q; nghiệm chọn không đổi.

Điều cần giữ đúng:

- Chỉ thay chữ ký **phản hồi** thành top-L; không tính lại w(p) từ top-L, vì như
  vậy đã đổi mục tiêu từ phục hồi top-k sang một bài toán khác.
- Cùng thứ tự POI, danh mục, category và phép ánh xạ tọa độ sang trạng thái.
- Giữ nguyên emission, prior, chuyển tiếp belief và private accountant.
- Các bước sau vẫn chọn trong miền đi tới được. Không nhảy query track sang vị
  trí xa chỉ vì vị trí đó có objective tốt hơn.

`ResponseAwareAnchorModel` là một view chỉ đọc của cấu hình công khai. Test nhỏ
đối chiếu công thức với việc cộng Recall trực tiếp trên tất cả trạng thái latent;
test L=k giữ nguyên transcript; test L>k kiểm tra tiền tố nhân quả, ngân sách và
tính khả thi trên đường. Đây không phải chứng minh belief đúng với người dùng,
planner tối ưu toàn cục, hay Recall thật luôn tăng.

Vì transcript thay đổi khi L>k, cùng epsilon không có nghĩa empirical privacy
giữ nguyên. Phải huấn luyện lại attacker theo cơ chế mới. Số POI thực nhận cũng
có thể khác dù cùng giới hạn K/L; cần đo chi phí thực, không tuyên bố byte bằng
nhau chỉ từ hai tham số đó. Mọi kết quả L5 của cấu hình này chỉ là ablation khi
phản hồi thực ít hơn mức planner dự kiến.
