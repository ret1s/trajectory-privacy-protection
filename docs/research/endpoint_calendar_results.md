> **Phạm vi hiện tại:** S10 chỉ gồm A/C; xem [kết quả và lập luận đã cập nhật](active_scope_results.md). Nội dung dưới lưu kết quả/chẩn đoán của phiên bản trước; không dùng số tổng hợp A/B/C cũ thay cho phạm vi hiện tại.

# S9/S10: lịch truy vấn công khai và kiểm tra trên nhiều nhóm tuyến

Cập nhật 25/09/2026. Kết quả dưới dùng **cohort mới, tách khỏi pilot**. Kế hoạch 30/67 query và cách chọn attacker được giữ nguyên trước khi xem cohort này.

## Quy mô và cách đọc mẫu

Đã thử đủ 32 seed 1201–1232: 32 nhóm dựng được, 704 chuyến hoàn tất, 1118 bản ghi toàn bộ S1–S10. Năm scenario ưu tiên dùng 460 record từ 271 chuyến; riêng S9/S10 có 175 record. Vòng này chấm utility trên 15 ca của năm scenario, privacy trên sáu ca S9/S10; không nhận là đã chạy lại privacy S1/S2/S3 trên cohort mới.

Nhóm tuyến là đơn vị gộp/ước lượng bất định. 22 chuyến cùng nhóm, record A/B/C, hai lần ngẫu nhiên cơ chế và ba trạng thái thế giới không được tính thành những người dùng độc lập. Đây vẫn là dữ liệu SUMO cùng thành phố và bộ sinh.

Một số nhóm không đủ điều kiện dựng mọi record: S9.B có 23 nhóm, S10.A có 31, S10.B có 25; các ca endpoint còn lại có 32. Các điều kiện tạo ca kiểm tra dữ liệu chuyển động, trước khi chạy model; không loại record theo kết quả privacy/utility.

Lỗi dựng nhóm: []. Giữ lỗi, không đổi seed để bù. Chọn đủ 32 seed trước khi xem kết quả; không dừng khi số đẹp.

Attacker fit trên 10 nhóm 501–502/701–708, chọn trên 10 nhóm 601–602/709–712/901–904. Pilot có bảy nhóm (1104 lỗi dựng); mở rộng này dùng nguyên selection đã khóa của pilot, không fit/chọn lại. Không gộp pilot vào kết quả chính.

Trùng nguyên chuỗi cạnh với dữ liệu phát triển: 0; với pilot: 0. Kiểm tra này không chứng minh các đường hoàn toàn không giao nhau.

## Vì sao nhiều đối chứng đạt Recall 100%?

DLS/RDG/Semantic/Fake giữ vị trí thật trong tập truy vấn. Trong giao diện đang thử, server trả top-10 POI khả dụng theo cùng khoảng cách đường; client hợp phản hồi rồi chọn top-5 thật. Phản hồi tại vị trí thật đã chứa top-5 cần dùng. Vì vậy Recall đạt trần là hệ quả của giao diện, không chỉ vì ít sample. Tăng số chuyến không làm mất tính chất này.

Recall đo chất lượng dịch vụ, **không đo privacy**. Phân biệt các phương pháp bằng khả năng suy luận của attacker và chi phí để giữ Recall đó. Không làm yếu API của đối chứng để tạo chênh lệch. Cấu hình 67 vẫn còn tám POI chưa phủ tĩnh, nên Recall 100% trên mẫu chưa bảo đảm 100% toàn bản đồ.

## Thành phần bảo vệ S9/S10

Client gửi kế hoạch theo loại POI ở mọi tick 60 giây trong khoảng công khai [0, 3600), kể cả trước/sau chuyến và khi không có truy vấn riêng tư. Thiết bị chỉ đọc cache hợp lệ và xếp hạng bằng GPS tại chỗ. Không dùng GPS để kích hoạt/dừng lịch.

Đây là mở rộng từ độc lập tọa độ sang độc lập thời điểm hoạt động **trong khoảng đã đăng ký trước**. Với vùng, kế hoạch, khoảng đăng ký và trạng thái server cố định, đổi hành trình hoặc giờ đọc cache không đổi bản tin yêu cầu/phản hồi. Sáu kiểm tra luồng dữ liệu đã so lịch đọc sớm, muộn và không đọc. Bảo đảm không bao gồm IP/account, click, lỗi mạng, đổi vùng theo GPS hoặc đăng ký/hủy dịch vụ theo giờ đi lại.

Giữ bank Viterbi/thống kê/ngoại suy/kNN/Extra Trees; bổ sung suy luận endpoint theo khoảng cách đường có hướng, vận tốc từ phần được quan sát và nhiều horizon. S9 suy ngược, S10 suy xuôi; C kết hợp các chuyến liên kết được phép. Đối thủ không nhận endpoint thật, đoạn bị giấu, thời gian cắt thật hoặc tuyến tương lai. Chọn decoder riêng cho MAE và từng ngưỡng Hit trên tập chọn, không bằng đáp án holdout. Vì có thể khác decoder, các cột Hit50/100/200 không phải một CDF chung. Log-gain, entropy và vùng credible trong artifact chỉ là chẩn đoán phụ; phải đọc cùng coverage thực tế của posterior, không coi entropy cao là bảo đảm privacy.

Phần đóng góp hiện có là thiết kế và kiểm chứng phối hợp kế hoạch phủ POI, lịch công khai và xếp hạng cục bộ trong dịch vụ cụ thể. Số này chưa xác lập rằng cover traffic hoặc truy vấn công khai là ý tưởng mới so với toàn bộ nghiên cứu trước.

## Kết quả cùng cohort và cùng dịch vụ

Mỗi scenario gộp đều A/B/C sau khi gộp theo record và nhóm. Hit100 thấp và MAE cao tốt hơn cho privacy. Recall gộp 15 ca ở p=0,8. Byte là yêu cầu + phản hồi JSON trên mỗi sự kiện dịch vụ, chưa gồm HTTP/TLS. Tính toàn bộ giờ cover traffic cho lịch công khai.

| Phương pháp | S9 Hit100 / MAE m | S10 Hit100 / MAE m | Recall@5 | Byte/sự kiện |
|---|---:|---:|---:|---:|
| Vị trí thật | 42.48% / 164.6 | 11.59% / 547.8 | 100.00% | 249.8 |
| DLS | 34.33% / 462.3 | 7.44% / 550.5 | 100.00% | 951.5 |
| RDG | 19.76% / 792.9 | 7.17% / 561.3 | 100.00% | 943.7 |
| TransProtect (Markov) | 33.61% / 243.4 | 8.87% / 581.2 | 99.69% | 249.8 |
| Semantic (thực nghiệm) | 40.81% / 183.6 | 7.88% / 586.0 | 100.00% | 976.0 |
| Fake-query adapter | 31.36% / 522.4 | 17.41% / 546.7 | 100.00% | 1300.3 |
| AnotherMe offline | 0.00% / 1687.2 | 0.00% / 1469.8 | 17.57% | 237.3 |
| 30 / mỗi sự kiện | 0.00% / 2202.9 | 0.67% / 2035.1 | 95.06% | 2395.3 |
| 67 / mỗi sự kiện | 0.00% / 2311.1 | 0.67% / 2035.1 | 100.00% | 5361.3 |
| 30 / lịch công khai | 0.00% / 2232.5 | 0.00% / 1982.9 | 95.06% | 3321.9 |
| 67 / lịch công khai | 0.00% / 2311.1 | 0.00% / 1982.9 | 100.00% | 7440.5 |

### Độ nhạy theo tỷ lệ POI khả dụng

| Cấu hình lịch | p=0,5 | p=0,8 | p=0,95 |
|---|---:|---:|---:|
| 30 / lịch công khai | 97.79%; 15/15 ca ≥90% | 95.06%; 15/15 ca ≥90% | 93.07%; 14/15 ca ≥90% |
| 67 / lịch công khai | 100.00%; 15/15 ca ≥90% | 100.00%; 15/15 ca ≥90% | 100.00%; 15/15 ca ≥90% |

AnotherMe là tham chiếu VTGA offline và có lỗi sinh đầu ra; không so privacy của tập con thành công như cùng mẫu số. TransProtect dùng Markov thay Transformer; Semantic dùng dự báo thực nghiệm thay LSTM. Đây là **adapter công khai, không phải tuyên bố vượt sáu paper nguyên bản**.

### Đọc từng A/B/C

| Ca | Nhóm | Raw Hit100 / Hit200 | 67 lịch Hit100 / Hit200 | Raw MAE m | 67 lịch MAE m |
|---|---:|---:|---:|---:|---:|
| S9.A | 32 | 43.75% / 59.38% | 0.00% / 0.00% | 166.8 | 2225.5 |
| S9.B | 23 | 58.70% / 76.09% | 0.00% / 0.00% | 163.9 | 2482.1 |
| S9.C | 32 | 25.00% / 62.50% | 0.00% / 0.00% | 163.0 | 2225.5 |
| S10.A | 31 | 12.90% / 35.48% | 0.00% / 0.00% | 204.1 | 1913.7 |
| S10.B | 25 | 0.00% / 0.00% | 0.00% / 0.00% | 1220.3 | 2116.2 |
| S10.C | 32 | 21.88% / 46.88% | 0.00% / 0.00% | 219.0 | 1918.9 |

S10.B chỉ cho thấy phần trước khi phân nhánh: ngay control vị trí thật có thể không xác định được đích. Phải đọc năng lực raw trước khi nhận Hit=0 là bằng chứng bảo vệ. Không gộp mọi endpoint thành một kết luận chắc chắn.

### Chênh lệch với năm adapter trực tuyến

Bootstrap ghép theo nhóm, 3.000 lần, khoảng 95% chưa hiệu chỉnh nhiều phép so sánh. Âm tốt cho ΔHit100; dương tốt cho ΔMAE. Đây là khoảng bất định theo nhóm trong một bộ sinh, không đại diện mọi thành phố.

| Ca | 67 lịch so với | ΔHit100 điểm % [CI95] | ΔMAE m [CI95] |
|---|---|---:|---:|
| S9 | DLS | -34.33 [-43.23, -26.34] | 1848.78 [1502.16, 2200.24] |
| S9 | RDG | -19.76 [-27.20, -12.93] | 1518.17 [1129.35, 1918.22] |
| S9 | TransProtect (Markov) | -33.61 [-42.98, -24.53] | 2067.65 [1776.91, 2359.44] |
| S9 | Semantic (thực nghiệm) | -40.81 [-50.44, -32.21] | 2127.48 [1848.78, 2409.53] |
| S9 | Fake-query adapter | -31.36 [-40.69, -23.02] | 1788.70 [1426.38, 2144.96] |
| S10 | DLS | -7.44 [-11.98, -3.29] | 1432.41 [1147.25, 1727.04] |
| S10 | RDG | -7.17 [-11.98, -2.86] | 1421.57 [1138.11, 1718.51] |
| S10 | TransProtect (Markov) | -8.87 [-13.54, -4.69] | 1401.69 [1115.68, 1690.76] |
| S10 | Semantic (thực nghiệm) | -7.88 [-12.63, -3.16] | 1396.92 [1113.48, 1694.00] |
| S10 | Fake-query adapter | -17.41 [-25.35, -10.14] | 1436.17 [1151.59, 1731.85] |

### Lập luận từ kết quả

- **S9:** Hit100 của năm adapter nằm trong 19.76%–40.81%, bản 67 theo lịch là 0.00%. 5/5 khoảng 95% của chênh lệch Hit100 nằm hoàn toàn dưới 0.
- **S10:** Hit100 của năm adapter nằm trong 7.17%–17.41%, bản 67 theo lịch là 0.00%. 5/5 khoảng 95% của chênh lệch Hit100 nằm hoàn toàn dưới 0.

Đây là lợi thế privacy trong bộ đối thủ và dịch vụ đã thử, với chi phí truyền cao hơn. Số 0% không chứng minh đối thủ bất kỳ đều thất bại; lập luận độc lập payload có điều kiện được kiểm tra riêng bằng luồng dữ liệu. S10.B control yếu vẫn là giới hạn, dù trung bình toàn scenario có chênh lệch.


## Giá của việc che giờ bắt đầu/kết thúc

So với chính kế hoạch đó chỉ refresh ở epoch có hoạt động, lịch công khai dùng trung bình theo nhóm 4.72 lần byte (30 query) và 4.72 lần (67 query). Đã đối chiếu 90618 lượt dùng dịch vụ có utility bằng nhau. Chi phí tăng là cover traffic ngoài thời gian hoạt động, không phải utility tăng.

Cả bản mỗi sự kiện và bản theo lịch đã dùng cùng tọa độ công khai. Vì vậy lợi ích mới của lịch là xóa phụ thuộc vào giờ hoạt động trong subscription, không nên gán toàn bộ chênh lệch MAE với DLS cho riêng bộ scheduler. Chưa có so sánh cùng byte/latency hoặc bằng chứng hơn bulk trên danh mục nhỏ 419 POI.

## Kiểm tra và tái chạy

Đã tính lại 1485 chỉ số, 33 aggregate chi phí, kiểm tra 4878 transcript. Generation failures: `{'anotherme_offline': 450}`. Lỗi phiên nhận Recall=0 ở các yêu cầu có đáp án; record nhiều phiên có thể vẫn có Recall từ phiên khác chạy thành công. Privacy thiếu đầu ra giữ NA, không gán bảo vệ hoàn hảo.

Dữ liệu và số đầy đủ: [protocol](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/protocol.json), [readout](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/readout.json), [66 dòng ca](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/case_results.csv), [verification](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/verification.json), [ablation](../../artifacts/benchmarks/endpoint_calendar_expanded_v1/calendar_ablation.json).

Runner: `python -m experiments.extend_endpoint_cohort STAGE` với `build → generate → attacks/service/flow → ablation → readout → verify`. Protocol/selection đã khóa; không dùng `prepare` để ghi đè. Bảng trình bày: `python -m experiments.summarize_endpoint_evidence`.

Dataset được lưu lossless bằng `dataset.json.gz`; phục hồi bằng `python -m experiments.endpoint_dataset_archive unpack` trước khi đọc hoặc dựng lại report. `archive.json` giữ SHA-256 của cả JSON gốc lẫn gzip; không thay nội dung đầu vào đã khóa.
