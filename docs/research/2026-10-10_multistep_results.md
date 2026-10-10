# Kết quả kiểm chứng Geo-I + lựa chọn đoạn Q

**Đã hoàn thiện prototype và chứng minh cho cả phiên; chưa đưa thành engine
chính vì utility chưa đạt tiêu chí.** Cận privacy mới áp dụng cho kernel lý
tưởng, không phải chứng nhận sampler NumPy/PCG64.

## 1. Cơ chế thêm vào và bằng chứng kiểm tra

Geo-I giữ Cs=0,23/m mỗi phiên, u=0,01/m, H=12, nhịp đọc tối thiểu 60s.
Baseline là Geo-I Slack hiện có, được dùng với L30 cho so sánh này. Mỗi trip
chạy một lịch sử Geo-I duy nhất; tất cả phương án chọn Q nhận cùng belief.

Prototype chọn chung ba khung Q ở 0/20/40s rồi chọn đoạn tiếp theo ở 60s.
Support chỉ dùng mạng đường, goal công khai và Q đã gửi. Mỗi khung vẫn K5.
Mỗi vòng có allowance Γ/[j(j+1)], tổng mọi prefix không vượt Γ. Ledger lưu
lựa chọn và bộ đếm trước khi gửi, khởi động lại không được cấp lại credit.

Cohort native mới có 60 nhóm: 24 train, 12 selection, 24 test; 480 chuyến
SUMO 1Hz được sinh trước khi đo chất lượng bảo vệ. Kiểm tra lại FCD gốc xác
nhận 312.480 GPS fixes và 240 chuyến calibration đối chứng. Benchmark dùng
hai chuyến ngày7/8 của mỗi nhóm: **120 trips, trong đó 48 test trips**. Một
private RNG draw mỗi trip; chưa là kiểm chứng ổn định qua nhiều private draws.

Kiểm tra độc lập xác nhận:

- 126.000 chuyển tiếp track đi được theo mạng đường có hướng, không resnap
  nhầm lane có tọa độ trùng nhau.
- 15.120 lần lựa chọn thuộc thư viện công khai, calibration không vượt
  allowance, mọi prefix giữ cap; tests còn kiểm tra phiên300frames và restart.
- Cùng protected belief/randomness tạo cùng request dù GPS/request dùng
  local khác nhau. Client trả danh sách sorted đầy đủ, Recall@5 chỉ là metric.
- **Cả15.120 quyết định đều không đạt floor công khai0,75.** Thuật toán giữ
  common support và báo degraded; không tuyên bố có hard QoS floor0,75.

## 2. Utility trên 24 nhóm test

Current-only Recall@5, trung bình đều theo family và purpose. N/A được giữ
riêng, không đổi thành điểm0. L30 là độ sâu mỗi category mỗi Q.

| Phương án | Nearest | Fastest | Radius1000m | Detour đến đích local thật | Macro | Utility gate |
|---|---:|---:|---:|---:|---:|---|
| Geo-I Slack baseline | 99,83% | 99,83% | 99,67% | 99,84% | 99,79% | Reference |
| PQB từng bước, Γ0,5 | 90,30% | 90,33% | 79,89% | 90,17% | 87,67% | Không đạt |
| PQB từng bước, Γ1 | 89,57% | 89,57% | 78,29% | 89,29% | 86,68% | Không đạt |
| PQB từng bước, Γ2 | 88,67% | 88,70% | 75,53% | 88,47% | 85,34% | Không đạt |
| PQB theo đoạn, Γ0,5 | 89,09% | 89,09% | 76,56% | 88,90% | 85,91% | Không đạt |
| **PQB theo đoạn, Γ1** | **89,74%** | **89,75%** | **78,06%** | **89,45%** | **86,75%** | **Không đạt** |
| PQB theo đoạn, Γ2 | 89,24% | 89,24% | 76,26% | 88,81% | 85,89% | Không đạt |

Arm chính Γ1 đã chọn trước test. Chênh lệch macro với baseline là
**−13,04 điểm phần trăm**, CI95 paired-family **[−18,58;−7,81]**. Gate yêu
cầu cận dưới không thấp hơn −2pp cả macro và từng purpose, nên không đạt.
Within-radius giảm21,61pp, là mục đích có mức giảm lớn nhất.

Cache trong epoch60s chỉ nâng macro Γ1 lên86,86%; chưa bù được khoảng cách.
Các giá trị frontier không đơn điệu theo Γ do một Q draw, support chuyển
động và geometry; không dùng test để chọn Γ0,5 thay arm Γ1 đã đóng băng.

## 3. Tái dựng vị trí: baseline và arm chính Γ1

Attacker được fit trên TRAIN, chọn theo loss trên SELECTION rồi lưu model
và lựa chọn trước khi dự đoán TEST. S9/S10 dùng cả observed-slot và geometry
tracking bank; không giả sử attacker chỉ quan sát centroid. MAE và Hit100
có thể chọn hai attacker khác nhau theo protocol.

| Mục tiêu đo | Baseline Hit100 ↓ | Γ1 Hit100 ↓ | Baseline MAE ↑ | Γ1 MAE ↑ |
|---|---:|---:|---:|---:|
| S1: định vị từ một event | 0,87% | 0,13% | 900,3m | 2.338,9m |
| S2: tọa độ cửa sổ dừng540–600s | 0,00% | 0,00% | 709,0m | 2.647,5m |
| S3: tái dựng toàn cửa sổ0–600s | 0,87% | 0,07% | 763,6m | 2.338,9m |
| S9: tọa độ đầu tại0s | 0,00% | 0,00% | 1.047,4m | 2.525,5m |
| S10: tọa độ cuối tại600s | 6,25% | 0,00% | 948,8m | 2.486,5m |

Hit100 là tỷ lệ dự đoán cách GPS thật≤100m; thấp hơn tốt hơn cho privacy.
MAE là sai số trung bình; lớn hơn nghĩa attacker định vị kém hơn. S2 ở đây
là **parked-window coordinate proxy**, không phải accuracy phân loại một địa
điểm nhạy cảm có nhãn ngữ nghĩa. S1/S3 dùng tất cả event, S9/S10 có48 mục
tiêu endpoint; 0% quan sát được không chứng minh xác suất tấn công bằng0.

Chất lượng tái dựng giảm trong bank này, nhưng không kết luận thắng mọi
attacker, mọi dataset hoặc mọi paper. S4–S6 vẫn dùng hệ quả posterior odds
có điều kiện, không chuyển bảng trên thành chứng minh anonymity.

## 4. Chi phí và điều học được từ chứng minh

Mỗi phương án đều có155 point requests/trip, K5/L30, cùng cadence và cache.
Baseline trung bình50.505 reply JSON bytes/trip; Γ1 theo đoạn50.414 bytes.
Đây là payload simulator trả POI indices, không phải HTTP/TLS, latency hay
năng lượng. Prototype không cần tăng số truy vấn để đạt cận privacy mới.

Điểm tiến bộ là biến điều kiện của định lý thành cơ chế thực thi: common
support theo lịch sử công khai, segment joint sampling, prospective budget
và phạm vi utility certificate rõ ràng. Điểm chưa đạt nằm ở **pool POI của
thư viện công khai**: floor đa mục đích không khả thi, nhất là radius; cận
Γ chặt không tự sinh thêm POI cần thiết. Hướng tiếp theo hợp lý là mở rộng
và tối ưu cover công khai, kiểm tra per-purpose feasibility và cache/version,
rồi đóng băng trước cohort mới. Không sửa mẫu, tăng cap riêng tư hoặc chọn
winner từ test để che khoảng cách hiện tại.

Trong dịch vụ tĩnh cho phép bulk catalogue download, control tải toàn bộ
catalogue và lọc local có thể tránh location queries. Prototype này đánh giá
hợp đồng point-query L hữu hạn, không chứng minh tối ưu mọi hợp đồng LBS.

## 5. Tái lập và nguồn

[Chứng minh và bảng định lý→mã](2026-10-10_multistep_privacy_utility.md).
[Bằng chứng đầy đủ](../../artifacts/benchmarks/query_segments_20261010_v2/README.md),
[JSON kết quả](../../artifacts/benchmarks/query_segments_20261010_v2/results.json),
[kiểm tra độc lập](../../artifacts/benchmarks/query_segments_20261010_v2/independent_validation.json).

```sh
python -m experiments.replay_query_segment_evidence
python -m experiments.verify_qplanner_fresh_native_20261006 \
  --output artifacts/datasets/query_segment_fresh_native_20261010_v2
python -m pytest tests
```

Dùng môi trường Python3.11 có requirements-query-segments.txt. Replay dựng
source snapshot đã đóng băng trong thư mục tạm, không cần private RNG keys,
không refit hoặc thay draws. Sau khi benchmark/validation hoàn tất, helper
calibration snapshot cũ được sửa thêm về exact-zero; kernel đoạn trong study
đã dùng cận riêng chặt chẽ từ đầu và không đổi. Metadata maintenance ghi rõ
thay đổi, source snapshot và mọi số liệu cũ giữ nguyên.
