# Tập truy vấn công khai cố định: contribution phải vượt qua đối chứng nào?

Vòng 22 cho thấy việc thích nghi theo lịch sử đã bảo vệ có giá trị utility ở
cùng K5 trên 14 ca, nhưng chưa giải quyết S1.C. Đồng thời, một control rất đơn
giản dùng K12 truy vấn cố định đã vượt 90% ở cả 15 ca. Vì vậy, Recall cao riêng
lẻ không đủ làm contribution: phải chứng minh đánh đổi tốt ở cùng chi phí và
nêu rõ dịch vụ nào thực sự cần máy chủ.

## Đối chứng hoạt động như thế nào?

Chọn trước K tọa độ trên mạng công khai để phủ nhiều POI tham chiếu nhất theo
prior công khai. Mọi người, mọi phiên gửi đúng tập tọa độ đó. Không đọc GPS để
chọn hoặc cập nhật tập truy vấn. Client nhận hợp các POI rồi dùng vị trí thật
cục bộ để xếp hạng. Các vị trí cố định có vận tốc bằng không, hợp lệ dưới ràng
buộc output hiện tại. Không thêm ràng buộc “dummy phải luôn chạy” chỉ để loại
đối chứng này.

Selector giữ prior, mục tiêu top-5, phản hồi top-10, largest SCC và tie rule của
planner; greedy rồi tối đa ba lượt exchange. Fit chỉ dùng tài nguyên công khai.
Năm K được chốt trước phép chấm đầy đủ; riêng K5/S1.C đã xem ở thăm dò, nên đây
là phát triển, không phải confirmation. Có 173 record thuộc 15 ca trên 12 nhóm,
102 nguồn phiên và 4.910 event theo đúng đồng hồ đã dùng ở vòng 18. Không nhân
đôi cỡ mẫu bằng RNG cho một control tất định.

Với cùng vùng, POI và đồng hồ công khai, tọa độ gửi ra giống nhau cho mọi quỹ
đạo GPS: phần tọa độ không bổ sung thông tin về GPS. Điều này không làm xác suất
đoán đúng từ prior bằng không, cũng không che thời điểm query, mở/đóng phiên,
identity hoặc vùng đã tải. Không gọi đây là bảo vệ toàn bộ metadata.

## Kết quả và chi phí

| K tọa độ mỗi event | POI slots trả về mỗi event¹ | Recall toàn phiên² | Ca đạt Recall ≥90% | S1.C |
|---:|---:|---:|---:|---:|
| 1 | 40 | 71,04% | 0/15 | 65,56% |
| 3 | 120 | 78,99% | 0/15 | 71,11% |
| 5 | 200 | 81,77% | 0/15 | 81,94% |
| 8 | 320 | 88,43% | 0/15 | 86,11% |
| 12 | 480 | 96,54% | 15/15 | 91,39% |

¹ Cộng các phản hồi category, giữ cả ID trùng giữa truy vấn. Có sáu category
được gọi; không phải category nào cũng có POI truy cập được. Đây là số slots,
không phải byte mạng hoặc payload thuộc tính thực tế. ² Trung bình Recall của
102 phiên, chỉ các truy vấn có tham chiếu; không gán điểm 1 cho tham chiếu rỗng.
Không đưa 510 phép đánh giá service tất định vào số lượt chạy cơ chế ngẫu nhiên.

K12 tốn 2,4 lần số tọa độ và slots của K5 ở control này. 15/15 là ngưỡng **trung
bình theo ca trên tập đã xem**, không bảo đảm mọi chuyến hay mọi vị trí đạt 90%.
Coverage kỳ vọng theo prior công khai của K12 chỉ khoảng 86,10%; phân phối
chuyến phát triển thuận lợi hơn prior, nên không suy rộng 96,54% ra toàn bản đồ.

| Ca | Công khai cố định K5 | Thích nghi K5, paced + slack .03 (vòng 18) |
|---|---:|---:|
| S1.A | 75,28% | 97,36% |
| S1.B | 78,61% | 94,58% |
| S1.C | **81,94%** | 78,75% |
| S2.A | 84,17% | 99,24% |
| S2.B | 84,17% | 99,32% |
| S2.C | 84,17% | 94,57% |
| S3.A | 79,94% | 94,78% |
| S3.B | 80,58% | 95,26% |
| S3.C | 80,01% | 94,83% |
| S9.A | 80,46% | 94,75% |
| S9.B | 80,62% | 94,38% |
| S9.C | 80,42% | 93,84% |
| S10.A | 81,96% | 97,06% |
| S10.B | 84,11% | 99,19% |
| S10.C | 80,85% | 95,41% |

Chênh lệch được ghép theo nhóm, sau khi trung bình RNG của phương pháp thích
nghi. Khoảng bootstrap 95% với 10.000 mẫu lại nằm trên 0 ở 14 ca; riêng S1.C là
−3,19 điểm phần trăm, khoảng [−9,58; 3,61]. Đây là 30 so sánh thăm dò cho hai
bản thích nghi, không hiệu chỉnh đa kiểm định và không phải xác nhận độc lập.
Không suy từ utility tăng thành privacy tăng: control cố định không dùng GPS,
còn bản thích nghi sử dụng neo riêng tư và vẫn có leakage thực nghiệm.

## Hệ quả cho hướng phát triển

- Có bằng chứng ban đầu cho **utility cao hơn với số truy vấn giới hạn**, chứ
  chưa có ưu thế privacy–utility tổng thể hoặc kết luận hơn paper gốc.
- S1.C cần cách xử lý uncertainty/coverage tốt hơn; thêm predictor chưa đủ.
- Cần giữ control fixed-query và local cache khi so sánh. Với POI tĩnh công
  khai hiện tại, cache cục bộ vẫn trả chính xác mà không cần truy vấn vị trí.
- Chưa nên tiếp tục tối ưu để đạt số đẹp trước khi chốt phần thông tin chỉ
  server có, chi phí/freshness của cache và workload hợp lệ.

## Kiểm tra và tái lập

`evaluation/public_cover.py`, `experiments/research_loop_public_cover.py` và
`experiments/research_loop_public_cover_readout.py` tạo các artifact `iteration22_*`.
Verifier dựng lại các plan từ tài nguyên công khai, truy vấn trực tiếp dịch vụ
tại tọa độ gửi ra và tính lại toàn bộ Recall/cửa sổ.

Bản chấm đầu đã dùng ID trạng thái nội bộ thay vì snap lại tọa độ server nhận.
Tại các làn trùng tọa độ, hai cách có thể khác. Đã giữ nguyên bản đầu và source
trong `iteration22_public_cover_state_access_v1.json` / `sources/`, sửa về cùng
giao diện của các runner thích nghi rồi kiểm tra lại. Các bảng trên chỉ dùng
`iteration22_public_cover.json` đã sửa; tập tọa độ không đổi.
