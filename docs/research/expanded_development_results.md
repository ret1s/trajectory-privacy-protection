# Kiểm tra trên 12 nhóm phát triển bổ sung

**Chưa có ứng viên đạt mọi yêu cầu.** Cả bốn bản bảo vệ qua 14/15 ngưỡng Recall; S1.C vẫn thiếu utility. Các thay đổi planner có đánh đổi privacy. Đây là dữ liệu phát triển, không phải confirmation cuối cùng.

Có 264/264 chuyến SUMO hoàn tất, 415 record trên 30 loại A/B/C; 173 record thuộc S1/S2/S3/S9/S10. Năm cấu hình gồm raw được khóa trước khi chấm, chạy hai RNG trên 102 phiên nguồn liên quan: **1.020 lượt đầy đủ và 1.730 hàng case**. Các record trong một nhóm có thể dùng chung chuyến, không coi là mẫu độc lập. S3.B có 11 nhóm, S9.B có 10, S10.B có 8; các ca còn lại có 12.

Giữ cùng K=5, phản hồi top-10 cho mỗi category, mục tiêu top-5 thật, trần riêng tư 0,23 m⁻¹ mỗi phiên và ràng buộc chuyển động có hướng. Mỗi truy vấn đều có phản hồi, không thêm trì hoãn hay đặt lại ngân sách tại cửa sổ case. Ba bản pacing có cùng chuỗi neo khi ghép RNG; khác nhau ở planner. Hai phiên liên kết có cận cộng 0,46 m⁻¹.

## Utility: Recall (%)

| Case | Nhóm | Pacing cũ | Top-10 | Top-10 + pacing | Top-10 + pacing + slack |
|---|---:|---:|---:|---:|---:|
| S1.A | 12 | 92.92 | 93.61 | 94.03 | 97.36 |
| S1.B | 12 | 91.53 | 92.50 | 93.75 | 94.58 |
| S1.C | 12 | 75.69 | 71.81 | 76.25 | 78.75 |
| S2.A | 12 | 98.75 | 100.00 | 99.86 | 99.24 |
| S2.B | 12 | 98.86 | 99.86 | 99.40 | 99.32 |
| S2.C | 12 | 98.54 | 100.00 | 97.29 | 94.57 |
| S3.A | 12 | 92.29 | 92.91 | 93.13 | 94.78 |
| S3.B | 11 | 93.27 | 93.51 | 94.07 | 95.26 |
| S3.C | 12 | 92.31 | 92.72 | 93.09 | 94.83 |
| S9.A | 12 | 92.35 | 92.81 | 93.05 | 94.75 |
| S9.B | 10 | 93.01 | 92.78 | 93.77 | 94.38 |
| S9.C | 12 | 92.74 | 92.93 | 93.22 | 93.84 |
| S10.A | 12 | 97.03 | 96.09 | 97.20 | 97.06 |
| S10.B | 8 | 99.07 | 99.48 | 99.49 | 99.19 |
| S10.C | 12 | 94.58 | 94.94 | 95.03 | 95.41 |

Ngưỡng 90% giữ nguyên và áp dụng cho trung bình của từng ca; 14/15 ca đạt không có nghĩa mọi chuyến đều đạt. Top-10 ở đây là objective phù hợp độ sâu phản hồi, không phải tăng độ sâu riêng cho ứng viên: cả bốn cột đều nhận top-10. Slack .03 giới hạn giảm objective theo belief tại một bước; không bảo đảm Recall thật chỉ giảm .03.

## Privacy: trường hợp đại diện

Các attacker học dùng 64 nhóm auxiliary để fit và 16 nhóm khác để chọn; không dùng nhãn của 12 nhóm mới để fit/chọn. Mỗi metric giữ attacker riêng đã chọn trước. MAE cao hơn và Hit thấp hơn có lợi cho defender. Bộ này gồm kNN, ExtraTrees, các quyết định theo mất mát và chiếu lên đường; chưa phải likelihood đầy đủ theo đường hoặc ngữ nghĩa.

| Case | Raw: MAE / Hit500 | Top-10 + pacing: MAE / Hit500 | Thêm slack: MAE / Hit500 |
|---|---:|---:|---:|
| S1.A | 0 m / 100.00% | 978 m / 25.00% | 852 m / 12.50% |
| S1.C | 0 m / 100.00% | 2318 m / 0.00% | 1674 m / 0.00% |
| S2.A | 0 m / 100.00% | 834 m / 16.67% | 910 m / 37.50% |
| S3.A | 0 m / 100.00% | 1101 m / 19.80% | 906 m / 20.52% |
| S9.A | 194 m / 100.00% | 675 m / 50.00% | 607 m / 58.33% |
| S9.B | 77 m / 100.00% | 741 m / 40.00% | 701 m / 50.00% |
| S9.C | 195 m / 100.00% | 590 m / 54.17% | 514 m / 58.33% |
| S10.A | 191 m / 100.00% | 1462 m / 20.83% | 1345 m / 8.33% |
| S10.B | 1367 m / 12.50% | 1593 m / 9.38% | 1318 m / 9.38% |
| S10.C | 190 m / 91.67% | 1306 m / 8.33% | 1200 m / 12.50% |

Hit100/200 và mọi estimator đều được giữ trong JSON, không chỉ các số thuận lợi ở bảng này. Ví dụ S1.A: thêm slack tăng Recall 94,03% → 97,36%, nhưng Hit100 của attacker tăng 0% → 8,33%. S3.A: Recall tăng 93,13% → 94,78%, nhưng MAE suy luận giảm 1.101 → 906 m. Đó là đánh đổi, không phải cải thiện privacy đồng thời.

S10.B raw còn yếu; xem [phân tích tính suy luận được của prefix](s10_prefix_identifiability.md). Không dùng Hit100 bằng 0 của mọi bản làm chứng cứ đã giải quyết case này.

## Mức chắc chắn và đóng góp

Phân tích ghép cặp resample 10.000 lần theo nhóm, giữ hai RNG và các record cùng nhóm đi cùng nhau. Đây là khoảng bootstrap thăm dò trên cùng bản đồ, chưa hiệu chỉnh nhiều phép so sánh, chưa suy rộng sang dân số hay site độc lập. So với bản top-10 chưa pacing, pacing tăng Recall S1.C **4,44 điểm phần trăm**, khoảng 95% **[2,50; 6,39]**, đồng thời làm MAE suy luận giảm khoảng **367 m**. Thêm slack chỉ tăng Recall S1.C trung bình 2,50 điểm, khoảng **[−1,11; 6,11]**; chưa có bằng chứng tăng ổn định.

Đóng góp có chứng cứ vững nhất hiện là phép gộp ứng viên theo chữ ký dịch vụ giữ nguyên output, với hiệu quả tính toán đã đo ở vòng 1. Objective phản hồi, pacing và tiến tới mục tiêu là các thành phần có ablation, nhưng chưa chứng minh ưu thế toàn diện hay thắng sáu paper. Chẩn đoán oracle cho thấy cần lập kế hoạch từ sớm, không chỉ chọn tốt tại bước cuối.

Vòng 19 đã kiểm tra lịch đọc giãn dần theo ngân sách còn lại, có và không có slack. Giữ epsilon, cap và lịch trả lời; không biết trước độ dài chuyến. Sau 132 lượt core và 48 lượt trên tất cả nguồn S1.C mở rộng, chưa chọn thay thế bản cũ. Ở tập mở rộng, bản có slack tăng Recall S1.C 78,75% → 80,83%, khoảng gain ghép cặp 95% [−1,94; 7,22] điểm phần trăm; toàn phiên trên các nguồn này giảm 87,53% → 87,31%. Bản không slack giảm S1.C 76,25% → 74,44%. Chưa có matched learned-privacy cho hai output mới, không chuyển số cũ sang chúng.

## Nguồn và giới hạn

- `iteration18_expanded_screening.json`: cấu hình, chữ ký nguồn và utility từng nhóm/category.
- `iteration18_expanded_attacks.json`: lỗi mọi attacker, lựa chọn cố định ngoài tập mới và raw control.
- `iteration18_paired_comparisons.json`: mọi so sánh ghép cặp và khoảng bootstrap.
- `verification.json`: kiểm tra chỉ số mẫu, ngân sách, output control, lựa chọn attacker và phép tổng hợp.

Các file trên ở `artifacts/benchmarks/research_loop/`. Runtime lần chạy song song chỉ mô tả, chưa dùng để kết luận tốc độ. Mạng OSM phục hồi khác nguồn benchmark lịch sử. Local-cache control vẫn trả POI tĩnh chính xác mà không gửi vị trí; cần chốt thông tin nào bắt buộc lấy từ máy chủ trước khi lập luận giá trị triển khai của dịch vụ.
