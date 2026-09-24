# Vòng 25–26: tách objective dịch vụ và primitive tạo nhiễu

**Chưa chọn phiên bản mới.** Cả bốn biến thể vẫn thiếu ngưỡng S1.C; đổi sang planar Laplace còn làm probe đầu phiên dễ suy luận hơn. Không kết luận phương pháp đã bảo vệ đủ S1/S2/S3/S9/S10 hoặc đạt mức luận văn từ các số dưới đây.

Đã chạy thêm 264 lượt bảo vệ toàn phiên, giữ nguyên 198 control đã lưu; 33 chuyến nguồn, hai RNG, bốn biến thể. Mọi bản có K5/L10, top-5 tham chiếu và cùng cận lý tưởng .23 m⁻¹. Slack là .03. Core có hai nhóm validation đã dùng trong phát triển; S9.B và S10.A chỉ có một nhóm đủ điều kiện.

## Kết quả utility

| Biến thể | Recall trung bình toàn phiên | Số ca đạt Recall ≥90% |
|---|---:|---:|
| REM | 92.15% | 14/15 |
| REM + slack | 93.12% | 13/15 |
| Cap .9 | 92.70% | 12/15 |
| Cap .9 + slack | 93.08% | 13/15 |
| Planar | 93.67% | 14/15 |
| Planar + slack | 94.44% | 13/15 |

Số ca đạt được tính bằng giá trị chưa làm tròn. Trung bình toàn phiên không thay thế ngưỡng từng ca.

| Ca | Nhóm / record validation | REM | REM + slack | Cap .9 | Cap + slack | Planar | Planar + slack |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1.A | 2 / 2 | 94.167 | 94.167 | 92.500 | 90.000 | 97.500 | 97.500 |
| S1.B | 2 / 2 | 91.667 | **89.167** | 93.333 | 91.667 | 90.833 | 99.167 |
| S1.C | 2 / 2 | **74.167** | **82.500** | **73.333** | **75.833** | **73.333** | **77.500** |
| S2.A | 2 / 2 | 98.333 | 97.917 | 97.500 | 100.000 | 100.000 | 100.000 |
| S2.B | 2 / 2 | 100.000 | 99.259 | 95.093 | 97.685 | 100.000 | 98.333 |
| S2.C | 2 / 2 | 93.333 | 91.574 | 96.667 | **89.954** | 94.167 | **87.407** |
| S3.A | 2 / 2 | 91.167 | 94.097 | **89.958** | 92.458 | 94.000 | 94.917 |
| S3.B | 2 / 2 | 91.167 | 94.097 | **89.958** | 92.458 | 94.000 | 94.917 |
| S3.C | 2 / 2 | 91.875 | 93.750 | 90.417 | 92.292 | 94.375 | 94.375 |
| S9.A | 2 / 2 | 91.432 | 92.503 | 92.765 | 93.575 | 93.962 | 95.509 |
| S9.B | 1 / 1 | 93.026 | 94.254 | 96.820 | 97.215 | 95.022 | 96.404 |
| S9.C | 2 / 2 | 91.499 | 91.898 | 93.566 | 93.072 | 92.960 | 95.286 |
| S10.A | 1 / 1 | 96.009 | 97.939 | 96.316 | 96.798 | 97.851 | 95.833 |
| S10.B | 2 / 2 | 99.280 | 98.951 | 97.001 | 95.264 | 100.000 | 99.199 |
| S10.C | 2 / 2 | 93.317 | 93.597 | 95.250 | 94.156 | 94.728 | 96.323 |

Đơn vị: %. **In đậm là chưa đạt**. Mỗi record được chạy với hai RNG; đây không phải hai nhóm/người dùng mới. Các điểm trong một chuỗi cũng không là các mẫu độc lập.

## Vòng 25: giảm thiếu hụt dịch vụ theo belief

Hàm mục tiêu mới lấy kỳ vọng của Recall bị chặn ở .9 **tại từng vị trí khả dĩ**. Nó khuyến khích phục vụ đủ nhiều vùng thay vì tiếp tục nâng một vùng đã đủ. Tuy nhiên, trên dữ liệu thực nghiệm, S1.C chỉ đạt 73,33% / 75,83%, thấp hơn các bản REM tương ứng 74,17% / 82,50%. Bản không slack còn làm S3.A/B tụt dưới 90%; bản slack làm S2.C tụt dưới 90%. Vì vậy chưa tiếp tục ứng viên sang confirmation.

Phép gộp các hàng nhu cầu POI giống nhau giữ nguyên hàm mục tiêu trong số học thực, giảm 2.073 hàng còn 458. Profile auxiliary 12 thời điểm giảm từ 688,80 xuống 196,63 ms/bước, với output giống nhau trên đoạn thử. Đây là cải tiến tính toán cho objective này; không phải bằng chứng cap cải thiện privacy hoặc utility. Runtime của các job toàn phiên chạy trùng thời gian nên không dùng để tuyên bố tăng tốc.

Xem [công thức, tiền lệ và giới hạn](capped_service_argument.md).

## Vòng 26: kernel chuẩn tăng utility nhưng có giá về privacy

Giữ planner, thay REM bằng planar Laplace chuẩn và cập nhật likelihood phù hợp. Recall toàn phiên tăng, nhưng S1.C vẫn không đạt. Bản planar + slack có S10.A bị đoán đúng trong 50 m ở 1/2 lượt RNG của **một** nhóm validation; REM + slack là 0/2 dưới bộ đối thủ hình học được chọn trên core train. Đây là counterexample nhỏ, không phải ASR tổng thể.

Attacker đầu phiên được học lại từ 2.000 vị trí công khai, chọn trên 16 nhóm auxiliary và chấm 16 chuyến × hai RNG thuộc hai nhóm core:

| Đầu ra đầu phiên | MAE attacker | Hit100 | Hit500 |
|---|---:|---:|---:|
| REM | 606.64 m | 0/32 | 12/32 |
| Planar (cả hai bản) | 404.55 m | 2/32 | 26/32 |
| Raw | 0 m | 32/32 | 32/32 |

MAE giảm và Hit tăng đều bất lợi privacy. Probe này cho phép nhìn truy vấn đầu tiên, **không thay thế điểm của S9.A/B/C có cửa sổ bị giới hạn**. Chưa có attacker chuỗi được học lại cho các bản mới. Các metric bán kính có bộ chọn riêng; không xem cả cột Hit như một CDF của cùng attacker.

Xem [kernel, tái lập và phạm vi](planar_anchor_ablation.md).

## Ý nghĩa đối với contribution

1. Giữ cải tiến tính toán có chứng cứ; phân biệt nó với hiệu quả bảo vệ.
2. Không nhận saturation, planar Laplace hay private reuse là ý tưởng mới. Lợi ích của tích hợp cần được quy về đúng thành phần bằng ablation.
3. Không chọn theo một con số trung bình thuận lợi: cả bốn bản đều có ca giảm utility; planar có counterexample privacy rõ hơn.
4. Chưa chọn phương pháp cuối hoặc dùng tập xác nhận mới. Cần làm rõ workload chỉ máy chủ cung cấp, vì local cache POI tĩnh đang trả đúng hoàn toàn mà không gửi vị trí.

## Bằng chứng và tái lập

- [Readout có hash nguồn và chênh lệch theo nhóm](../../artifacts/benchmarks/research_loop/iteration25_26_readout.json).
- [Kết quả cap](../../artifacts/benchmarks/research_loop/iteration25_capped_service_cases.json), [kết quả planar](../../artifacts/benchmarks/research_loop/iteration26_planar_anchor_cases.json).
- [Probe học riêng](../../artifacts/benchmarks/research_loop/iteration26_first_query_attack.json).
- Kiểm tra bằng `python -m experiments.verify_research_loop`: nguồn, control, neo/ledger, đường đi, utility từ tọa độ server, cửa sổ, features và lựa chọn attacker. Đây là kiểm tra tính đúng; không phải chứng minh hiệu quả hoặc confirmation.
- Có 75 test liên quan đã qua (69 test nhóm cũ/cap và 6 test planar). Tổng lượt toàn phiên đến vòng 26: 4.162; 2.032 lượt mới chỉ sinh truy vấn đầu được đếm riêng, không phải chuyến SUMO mới.
