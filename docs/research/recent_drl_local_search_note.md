# Related work bổ sung: DRL cho local search (2026)

Abbasi, Welscher và Scholz dùng PPO chọn độ lệch chuẩn Gaussian từ tọa độ thật,
trả một vị trí nhiễu. Utility là Jaccard giữa tập nhà hàng trong bán kính 1 km;
thử lại tới khi thỏa điều kiện dịch vụ. Metrics gồm số lần thử và khoảng dịch
chuyển. Thí nghiệm là snapshot tại Salzburg, không phải quỹ đạo trên làn có
hướng. Đọc bản đầy đủ, §§3.2–4.5; công bố 29/07/2026.
[Bài gốc](https://onlinelibrary.wiley.com/doi/full/10.1111/tgis.70363).

**Nhận định khi đối chiếu:** dịch chuyển lớn chưa chứng minh attacker suy luận
sai. Policy đọc tọa độ thật không thể cắm thẳng vào phần hậu xử lý của ta rồi
giữ nguyên cận ngân sách. Muốn so thực nghiệm phải khớp query/utility, tính chi
phí thử lại và đánh giá attacker; hiện chỉ là related work, không phải baseline
đã tái lập. Link Figshare do bài dẫn trả HTTP 202 rỗng khi kiểm tra; chưa lấy
được code/model. Không tự thêm vào sáu phương pháp đối chứng đã định.
