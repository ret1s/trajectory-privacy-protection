# Truy vấn công khai bổ sung: dịch vụ và chi phí

Giữ nguyên năm truy vấn thích nghi rồi thêm 1–3 vị trí cố định công khai.
Xóa phần bổ sung khôi phục đúng transcript ban đầu: không phát sinh lợi ích privacy.
Các phản hồi tăng tập POI có thể sử dụng, với chi phí truy vấn được tính đầy đủ.

| Phương pháp | Tổng K | Recall toàn phiên | S1.C | Ca đạt 90% |
|---|---:|---:|---:|---:|
| response_paced+public1 | 6 | 95.30% | 76.39% | 14/15 |
| response_paced+public2 | 7 | 95.54% | 79.31% | 14/15 |
| response_paced+public3 | 8 | 96.25% | 79.58% | 14/15 |
| response_paced_slack03+public1 | 6 | 95.33% | 79.17% | 14/15 |
| response_paced_slack03+public2 | 7 | 95.54% | 81.81% | 14/15 |
| response_paced_slack03+public3 | 8 | 96.24% | 82.08% | 14/15 |

Đánh giá trên 12 nhóm tuyến phát triển, 173 record. 1.224 lượt đánh giá dịch vụ
là replay xác định, không phải 1.224 chuyến hay lượt sinh nhiễu mới. Đối chứng
cố định cùng K6/7/8 đều chưa đạt ngưỡng trung bình theo ca; riêng S1.C lần lượt
đạt 84,44%, 86,39%, 86,11%, cao hơn các bản thích nghi trong bảng.

Kết luận: tăng số truy vấn giúp utility nhưng chưa giải quyết S1.C; không chọn
một bản mới từ thí nghiệm này. Các counterexample S9.C của phương pháp gốc
vẫn giữ nguyên. Không ghép số POI tĩnh này với workload availability vòng 28.

Nguồn: [artifact](../../artifacts/benchmarks/research_loop/iteration27_public_supplement.json),
[wrapper](../../benchmark/engines/public_supplement.py),
[kiểm tra](../../experiments/verify_research_loop_supplement.py).
