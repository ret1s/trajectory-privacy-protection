# Phạm vi hiện tại: S1/S2/S3/S9 và S10.A/C

**S10 chỉ gồm A và C.** Ca B đã loại khỏi danh mục mẫu và kết quả chính; hướng đoán đích từ tiền tố thuộc S6.A. Giữ mã A/C để truy nguyên. Kho dữ liệu và benchmark cũ được bảo toàn, không nhận việc thu hẹp phạm vi là cải tiến thuật toán hay một phép xác nhận mới.

Quyết định phạm vi được thực hiện sau [chẩn đoán B](s10b_diagnostic.md). Không chọn lại model, attacker hoặc seed. Các số dưới được tính lại từ kết quả đã khóa.

## Phạm vi và cách tổng hợp

- 29 ca toàn bộ S1–S10; 14 ca trong năm scenario ưu tiên.
- Bộ minh họa: 389/393 record gốc, 12 nhóm, 264 chuyến; 29 mẫu trên map.
- Đối chứng phát triển: 165 record từ 94 chuyến; kiểm tra bốn nhóm: 54 record từ 30 chuyến.
- Cohort 32 nhóm: giữ 1.093/1.118 record; utility dùng 435 record từ 246 chuyến, privacy S9/S10 dùng 150 record.
- Gộp seed/world trong record rồi nhóm; trung bình đều ca trong mỗi scenario và đều năm scenario. S10 có A/C nên chia hai, các scenario khác chia ba. Không lấy trung bình phẳng 14 ca để làm giảm trọng số S10.
- Chi phí chỉ trên chuyến còn thuộc phạm vi; giữ toàn bộ clock, mẫu số sự kiện và traffic gốc của chuyến. Client theo lịch bị tính đủ cả giờ, kể cả trước/sau chuyến. Không tái sinh transcript hoặc xóa traffic để làm chi phí giảm.
- AnotherMe có lỗi và thiếu ca: số privacy chỉ mô tả tập con thành công; không đồng hạng với năm adapter trực tuyến.

## Bằng chứng cho cả năm scenario

| Scenario | Hit100 năm adapter trực tuyến | Đề xuất | Phạm vi bằng chứng |
|---|---:|---:|---|
| S1 | 50.00%–83.33% | 0% | 4 nhóm, kế hoạch mỗi sự kiện |
| S2 | 33.33%–100.00% | 0% | 4 nhóm, kế hoạch mỗi sự kiện |
| S3 | 74.59%–96.91% | 0% | 4 nhóm, kế hoạch mỗi sự kiện |
| S9 | 19.76%–40.81% | 0% | 32 nhóm, kế hoạch theo lịch |
| S10.A/C | 10.75%–26.11% | 0% | 32 nhóm, kế hoạch theo lịch |

**Có cơ sở trình bày đóng góp thiết kế và thực nghiệm trên cả năm scenario trong phạm vi đã thử.** Giải pháp phối hợp kế hoạch truy vấn theo độ phủ POI, xếp hạng bằng GPS tại thiết bị và lịch công khai. GPS và giờ đọc cache không điều khiển payload/lịch gửi khi vùng, kế hoạch, trạng thái server và khoảng đăng ký cố định. Điều này hỗ trợ lập luận giảm lộ vị trí hiện tại, nơi dừng, đường đi và endpoint trong dịch vụ trạng thái động.

Ba lớp bằng chứng cần đọc cùng nhau: (1) lập luận phụ thuộc thông tin có điều kiện; (2) kết quả chống bộ đối thủ đã thử; (3) utility và chi phí đo trên cùng API. Chọn truy vấn công khai/cover traffic tự thân chưa được chứng minh là ý tưởng mới. Số bốn nhóm S1–S3 không được gán thành kết quả của client theo lịch trên 32 nhóm.

## S9 và S10.A/C: kết quả cùng cohort

Mỗi ô privacy: Hit100 ↓ / MAE m ↑. Recall tại availability 80%; byte là request + response JSON mỗi sự kiện dịch vụ, chưa gồm HTTP/TLS.

| Phương pháp | S9 A/B/C | S10 A/C | Recall@5 | Byte/sự kiện |
|---|---:|---:|---:|---:|
| Vị trí thật | 42.48% / 164.6 | 17.39% / 211.6 | 100.00% | 249.9 |
| DLS | 34.33% / 462.3 | 11.16% / 221.6 | 100.00% | 952.2 |
| RDG | 19.76% / 792.9 | 10.75% / 224.5 | 100.00% | 944.4 |
| TransProtect (Markov) | 33.61% / 243.4 | 13.31% / 259.3 | 99.70% | 249.9 |
| Semantic (thực nghiệm) | 40.81% / 183.6 | 11.82% / 292.3 | 100.00% | 975.8 |
| Fake-query adapter | 31.36% / 522.4 | 26.11% / 213.1 | 100.00% | 1295.6 |
| AnotherMe offline† | 0.00% / 1687.2 | 0.00% / 1231.8 | 16.85% | 239.1 |
| Đề xuất 30 + lịch | 0.00% / 2232.5 | 0.00% / 1916.3 | 95.00% | 3071.0 |
| Đề xuất 67 + lịch | 0.00% / 2311.1 | 0.00% / 1916.3 | 100.00% | 6878.6 |

† AnotherMe offline dùng tập con privacy thành công; utility giữ lỗi. TransProtect dùng Markov thay Transformer, Semantic dùng predictor thực nghiệm thay LSTM. Đây là so sánh adapter, chưa xác lập vượt sáu paper nguyên bản.

### Chênh lệch S10.A/C đã tính lại

Bootstrap ghép theo nhóm, 3.000 lần, khoảng 95% chưa hiệu chỉnh nhiều so sánh. Phân tích theo phạm vi sửa sau chẩn đoán, không phải cohort mới.

| 67 + lịch so với | ΔHit100 (điểm %) | Khoảng 95% |
|---|---:|---:|
| DLS | -11.16 | [-18.02, -4.81] |
| RDG | -10.75 | [-18.27, -4.30] |
| TransProtect (Markov) | -13.31 | [-19.58, -7.08] |
| Semantic (thực nghiệm) | -11.82 | [-19.53, -4.91] |
| Fake-query adapter | -26.11 | [-37.31, -15.62] |

Raw S10.A có Hit100 12,90%, S10.C 21,88%; sau bảo vệ theo lịch đều 0% trong bank đã thử. S10.A có 31 nhóm, C có 32; không cộng thành 63 nhóm độc lập. Các khoảng A/C ở đây được tính lại từ metric từng nhóm, không tái sử dụng khoảng A/B/C.

### Utility và chi phí

- Bản 30: Recall 95.00%, 14/14 ca ≥90% ở p=0,8. Stress p=0,95: 93.01%, 13/14 ca đạt ngưỡng.
- Bản 67: Recall 100.00%; 14/14 ca đạt ngưỡng, vẫn 100.00% trong stress p=0,95.
- Trên chuyến thuộc phạm vi mới, lịch công khai tốn trung bình theo nhóm 4.48 lần byte so với cùng kế hoạch chỉ refresh khi hoạt động; giữ 86,712 phép so utility bằng nhau.
- DLS/RDG/Semantic/Fake đạt trần Recall nhờ chứa truy vấn tại vị trí thật trong giao diện đang dùng. Đây không phải bằng chứng privacy. Bản 67 còn tám POI ngoài độ phủ tĩnh, nên Recall 100% trên mẫu chưa là chứng nhận toàn bản đồ.

## Các kết luận còn cần kiểm chứng

- Chạy lại privacy S1/S2/S3 trên 32 nhóm bằng client theo lịch và lựa chọn attacker phù hợp, giữ riêng với kết quả bốn nhóm đã có.
- So sánh ở cùng ngân sách byte/latency; chi phí cao hơn ngăn kết luận thống trị toàn diện.
- Kiểm tra một vùng/bộ sinh khác và củng cố đối chứng gốc trước khi kết luận tổng quát hoặc độ mới phương pháp.
- Bảo đảm hiện tại có điều kiện vùng/lịch công khai cố định; IP, account, click, lỗi mạng và kích hoạt dịch vụ theo chuyến nằm ngoài phạm vi.

## Nguồn và tái lập

[Readout hiện tại](../../artifacts/benchmarks/active_scope_ac_v2/readout.json) · [Verification](../../artifacts/benchmarks/active_scope_ac_v2/verification.json). Đã đối chiếu 981 giá trị/lựa chọn mẫu số với dữ liệu đo gốc; kiểm tra phân tầng và chi phí có tests riêng.

`python -m experiments.reaggregate_active_scope` → `python -m experiments.verify_active_scope` → `python -m experiments.summarize_active_scope`. Các lệnh chỉ tạo readout mới; không ghi đè kết quả gốc hoặc train lại model.
