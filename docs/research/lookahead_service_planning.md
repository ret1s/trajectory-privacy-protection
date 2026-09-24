# Lập kế hoạch truy vấn cho hiện tại và tương lai

Vòng 21 kiểm tra liệu tối ưu một bước có làm dummy khó theo kịp vùng POI sắp
cần phục vụ. Planner mới chấm hai tập phản hồi: ngay lúc này và sau 120 giây.
Vị trí tương lai được dự đoán từ belief đã bảo vệ, bằng mô hình auxiliary của
vòng 20. Không dùng GPS tương lai, đích chuyến đi hoặc nhãn scenario.

## Cơ chế được kiểm tra

1. Tính các mục tiêu phủ POI tốt ở hiện tại và ở belief dự báo.
2. Với mỗi dummy, tạo một số cặp vị trí hiện tại–tương lai trên đường có hướng
   tới các mục tiêu; thêm phương án hiện tại của planner cũ và đứng yên.
3. Chọn một cặp cho mỗi dummy, tối đa hóa trung bình coverage hai thời điểm,
   bằng greedy rồi tối đa ba lần thay một track.
4. Chỉ nhận phương án nếu objective hiện tại không thấp hơn planner cũ quá
   δ và objective hai thời điểm không thấp hơn phương án cũ rồi đứng yên.
5. Gửi các vị trí hiện tại; tại truy vấn sau lập kế hoạch lại.

Giữ K5, phản hồi L10, cap .23 và toàn bộ neo/ledger giống bản cha. Đường đi phải
khả thi theo mạng làn cũ. Đây là tập đường ứng viên hữu hạn, không phải tìm trên
mọi đường. Floor áp dụng cho objective theo belief, **không bảo đảm Recall thật**.
Không suy cận greedy của bài toán không có floor thành cận tối ưu cho toàn bộ
thuật toán có bước nhận/từ chối. Dự báo được tính lại liên tục cũng chưa chứng
minh planner tránh được việc trì hoãn di chuyển.

## Pilot đã chạy

Tất cả bốn nguồn S1.C core, mỗi nguồn hai RNG, toàn phiên với đồng hồ cũ:
24 lượt mới và 16 control replay. Hai nhóm validation đã được xem nhiều lần;
đây là chẩn đoán, chưa phải confirmation hoặc kết quả đủ 15 ca.

| Phương pháp | Recall S1.C validation | Recall toàn phiên trên cùng nguồn |
|---|---:|---:|
| Chuyển động học, paced | 77,50% | 82,54% |
| Như trên + slack .03 | 78,33% | 84,94% |
| Hai thời điểm, δ=0 | 76,67% | 82,86% |
| Hai thời điểm, δ=.03 | **80,83%** | 83,20% |
| Hai thời điểm, giữ nguyên belief tương lai, δ=.03 | 75,83% | 82,28% |

Có tín hiệu dự báo giúp bản có slack tại mẫu S1.C, nhưng vẫn dưới ngưỡng 90% và
utility toàn phiên thấp hơn control slack. Chưa chọn làm phương pháp cuối cùng.
Chưa có attacker học lại cho các output này, nên chưa kết luận privacy tốt hơn.
Thời gian đo khoảng 120–127 ms/bước của các bản mới chỉ mang tính mô tả; không
so tốc độ trực tiếp với các phép đo lịch sử chạy khác điều kiện.

## Quan hệ với nghiên cứu trước

MPC và dự đoán cho location privacy đã có trong Molina và cộng sự, IFAC 2023,
LCSS 2023 và Control Engineering Practice 2025. Không nhận “nhìn trước” là phát
minh. Khác biệt đang kiểm tra ở đây là tối ưu hợp phản hồi của K truy vấn trên
mạng đường như hậu xử lý của lịch sử đã được tính ngân sách. Cần chứng minh lợi
ích riêng so với các đối chứng phù hợp; hiện chưa đủ kết quả.
Xem [đối chiếu nguồn và giới hạn truy cập](predictive_planning_prior_art.md).

Code: `benchmark/engines/lookahead_cover.py`; bằng chứng:
`artifacts/benchmarks/research_loop/iteration21_lookahead_pilot.json`.
