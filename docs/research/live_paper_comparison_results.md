> **Phạm vi hiện tại:** S10 chỉ gồm A/C; xem [kết quả và lập luận đã cập nhật](active_scope_results.md). Nội dung dưới lưu kết quả/chẩn đoán của phiên bản trước; không dùng số tổng hợp A/B/C cũ thay cho phạm vi hiện tại.

# Đối chứng trực tiếp trên cùng dịch vụ POI sống

Ngày 25/09/2026. Kết quả dùng trong báo cáo 26/09. Nguồn: [`live_paper_comparison_v1`](../../artifacts/benchmarks/live_paper_comparison_v1).

**Đã chạy cùng benchmark cho năm adapter trực tuyến và AnotherMe VTGA offline**, cùng hai cấu hình đề xuất đã khóa. Không đổi tên fixed K5/K12 thành phương pháp từ paper. Đây là các bản thích nghi được nêu rõ; chưa phải tái lập đầy đủ sáu hệ thống gốc.

**Kết quả chính trên bốn nhóm 901–904:** cấu hình 30 đạt Recall@5 **94,53%**, cấu hình 67 đạt **100%**. Cả hai qua ngưỡng 90% ở 15/15 ca khi availability=80%. Hit100 của đề xuất bằng 0 trong bộ attacker hữu hạn này; bằng chứng phân biệt tốt nhất ở S1–S3. Chi phí truyền cao hơn đối chứng. Không kết luận thắng mọi mặt hoặc đã giải hoàn toàn S9/S10.

## Bản triển khai và giới hạn

| Phương pháp | Lõi được chạy | Thay thế / phạm vi |
|---|---|---|
| DLS | Chọn tập thật + dummy theo entropy | K=5; trạng thái làn SUMO và prior background thay ô lưới nguyên bản; pool/tie/seed được kiểm tra tương đương DLS hiện có. |
| RDG | Thuật toán 6: greedy theo entropy trọng số max-product; posterior forward sum-product | K=5, pool 20 dummy từ DLS; bám bản thảo arXiv 2018, chưa xác minh parity với bản TMC cuối. [Hồ sơ](../reproduction/rdg.md). |
| TransProtect | Lựa chọn ứng viên theo xác suất + utility, nhiễu trên tập ứng viên | Predictor Markov, **không phải Transformer được huấn luyện của tác giả**; K ứng viên=10, 8 target, alpha=10000, epsilon=5/km. [Hồ sơ](../reproduction/transprotect.md). |
| Semantic correlation | Bộ chọn dummy theo ngữ nghĩa/chuyển tiếp | K=5, pool=128; nearest-POI category và predictor thực nghiệm thay AMap/LSTM. [Hồ sơ](../reproduction/semantic_correlation.md). |
| Fake-query insertion | Reference/companion continuation, chèn bản tin toàn giả, fallback DLS | K=5, tối đa 2 fake, chờ 5–15 giây, N=24, sigma=0.75; đường có hướng và lịch sử quần thể. Thất bại sinh fake được đếm. [Hồ sơ](../reproduction/fake_queries.md). |
| AnotherMe | VTGA theo pipeline được ánh xạ từ mã tác giả | **Full-trip offline**; routing/mapping nội bộ, tối thiểu 20 điểm đầu ra thô. Không xếp hạng online hoặc suy thất bại adapter thành chất lượng paper gốc. [Hồ sơ](../reproduction/anotherme.md). |
| Đề xuất | Truy vấn theo loại từ kế hoạch công khai | 30 query tại 19 tọa độ hoặc 67 query tại 52 tọa độ; GPS chỉ để xếp hạng cục bộ. |

## Giao thức chung

- Cùng OSM/SUMO, 419 POI thuộc sáu loại; top-10 khả dụng ở server, top-5 theo khoảng cách đường ở client. Availability giữ trong epoch 60 giây, p=0.5/0.8/0.95, world seed 3101/3102/3103.
- Fit thống kê và attacker bằng nhóm 501–502, chọn decoder bằng 601–602. Đánh giá 173 record / 102 chuyến / 12 nhóm 701–712 và 56 record / 32 chuyến / 4 nhóm 901–904; không có giao nhau nhóm. Các tập đã được xem trước đây: **đánh giá phát triển, không phải xác nhận độc lập mới**.
- Hai seed cơ chế; seed/world/record dùng chung chuyến không được coi là người dùng độc lập. Gộp seed/world trong record, record trong nhóm, nhóm trong ca; trung bình đều 15 ca chỉ dùng cho tác vụ có cùng nghĩa.
- Mọi phương pháp nhận cùng tọa độ trạng thái làn. Audit 6.719 sự kiện xác nhận đổi tọa độ → snap lại không đổi state; utility reference giữ state gần FCD gốc, privacy target là FCD gốc.
- Gửi tại mọi sự kiện dịch vụ; cho mọi phương pháp hợp phản hồi cùng epoch. Đếm thêm toàn bộ bản tin fake. Đây **không phải** giao thức chỉ refresh một lần/epoch trong kết quả nội bộ cũ.
- Byte là JSON compact UTF-8 request + response thực của từng giao diện, có thời gian/epoch; chưa gồm HTTP/TLS. Mỗi chi phí chia số yêu cầu thật trong phiên, rồi gộp phiên/nhóm. Thời gian sinh tọa độ không gồm setup dùng chung, server, RTT hoặc tính POI trên thiết bị.
- Giao thức bảng lịch sử này: nếu record thiếu đầu ra của bất kỳ phiên nào, mọi yêu cầu có tham chiếu trong record nhận utility=0, yêu cầu không có tham chiếu giữ null; privacy=NA. Cohort endpoint mới tính lỗi theo từng yêu cầu/phiên và được báo riêng. Không bỏ lỗi để nâng Recall và không xem im lặng do lỗi là bảo vệ hoàn hảo.

## Utility và tài nguyên trên cùng dữ liệu

Bảng dưới ở availability=80%. Chi phí AnotherMe chỉ có trên phiên thành công; không so trực tiếp chi phí này với nhánh online.

### 12 nhóm phát triển

| Phương pháp | Recall@5 ↑ | Ca ≥90% | Byte / yêu cầu thật ↓ | Sinh tọa độ ms / yêu cầu ↓ | Lượt record thành công |
| --- | --- | --- | --- | --- | --- |
| Vị trí thật | 100.00% | 15/15 | 245.3 | 0.000091 | 346/346 |
| DLS (road) | 100.00% | 15/15 | 931.1 | 0.440994 | 346/346 |
| RDG (road) | 100.00% | 15/15 | 933.1 | 1.478778 | 346/346 |
| TransProtect (Markov) | 99.87% | 15/15 | 244.9 | 5.532284 | 346/346 |
| Semantic (POI/thực nghiệm) | 100.00% | 15/15 | 981.7 | 1.037237 | 346/346 |
| Fake-query (road) | 100.00% | 15/15 | 1263.2 | 0.771456 | 346/346 |
| AnotherMe (VTGA offline) | 19.23% | 0/15 | 240.0 | 1.991764 | 86/346 |
| Đề xuất 30 | 95.61% | 15/15 | 2389.1 | 0.000166 | 346/346 |
| Đề xuất 67 | 100.00% | 15/15 | 5353.1 | 0.000314 | 346/346 |

### Bốn nhóm 901–904

| Phương pháp | Recall@5 ↑ | Ca ≥90% | Byte / yêu cầu thật ↓ | Sinh tọa độ ms / yêu cầu ↓ | Lượt record thành công |
| --- | --- | --- | --- | --- | --- |
| Vị trí thật | 100.00% | 15/15 | 242.7 | 0.000112 | 112/112 |
| DLS (road) | 100.00% | 15/15 | 961.1 | 0.423551 | 112/112 |
| RDG (road) | 100.00% | 15/15 | 948.8 | 1.402690 | 112/112 |
| TransProtect (Markov) | 99.70% | 15/15 | 242.6 | 5.552305 | 112/112 |
| Semantic (POI/thực nghiệm) | 100.00% | 15/15 | 966.0 | 1.056873 | 112/112 |
| Fake-query (road) | 100.00% | 15/15 | 1313.5 | 0.763810 | 112/112 |
| AnotherMe (VTGA offline) | 16.56% | 0/15 | 241.7 | 1.857739 | 30/112 |
| Đề xuất 30 | 94.53% | 15/15 | 2389.6 | 0.000169 | 112/112 |
| Đề xuất 67 | 100.00% | 15/15 | 5353.8 | 0.000270 | 112/112 |

Lượt record là record × seed; không phải số chuyến độc lập. DLS/RDG/Semantic/Fake chứa vị trí thật trong tập gửi, nên việc đạt utility 100% là phù hợp với cơ chế hợp phản hồi/lọc cục bộ. Đề xuất 30 đổi một phần utility và nhiều byte hơn lấy việc không dùng GPS trong kế hoạch gửi.

## Privacy theo từng scenario

Mỗi ô: **Hit100 ↓ / MAE (m) ↑**, trung bình đều A/B/C sau gộp nhóm. MAE và Hit chọn decoder riêng trên nhóm 601–602. Median/p90 được tính trong từng record rồi gộp, không phải percentile của toàn bộ điểm gộp. AnotherMe chỉ có một phần ca: không coi các ô của nó là cùng mẫu số với phương pháp khác.

### 12 nhóm phát triển

| Phương pháp | S1 | S2 | S3 | S9 | S10 |
| --- | --- | --- | --- | --- | --- |
| Vị trí thật | 100.00% / 4 | 100.00% / 2 | 100.00% / 4 | 25.00% / 217 | 2.78% / 576 |
| DLS (road) | 13.89% / 1729 | 75.00% / 1802 | 96.03% / 84 | 20.83% / 656 | 2.78% / 540 |
| RDG (road) | 25.00% / 2093 | 88.89% / 1774 | 94.62% / 127 | 20.83% / 1245 | 1.39% / 1122 |
| TransProtect (Markov) | 75.00% / 65 | 63.89% / 87 | 74.02% / 77 | 25.42% / 236 | 2.78% / 575 |
| Semantic (POI/thực nghiệm) | 86.11% / 58 | 100.00% / 12 | 80.54% / 58 | 32.78% / 343 | 5.56% / 578 |
| Fake-query (road) | 27.78% / 1851 | 48.61% / 1381 | 97.68% / 60 | 14.17% / 877 | 4.17% / 559 |
| AnotherMe (VTGA offline) | 0.00% / 2127 | NA | 0.00% / 1519 | 0.00% / 1757 | 0.00% / 1640 (2/3 ca) |
| Đề xuất 30 | 0.00% / 1958 | 0.00% / 2035 | 0.00% / 1914 | 0.00% / 2555 | 0.00% / 2197 |
| Đề xuất 67 | 0.00% / 1958 | 0.00% / 2132 | 0.00% / 1914 | 0.00% / 2541 | 0.00% / 2172 |

### Bốn nhóm 901–904

| Phương pháp | S1 | S2 | S3 | S9 | S10 |
| --- | --- | --- | --- | --- | --- |
| Vị trí thật | 100.00% / 5 | 100.00% / 6 | 100.00% / 4 | 16.67% / 207 | 8.33% / 701 |
| DLS (road) | 50.00% / 1400 | 41.67% / 1613 | 95.76% / 93 | 8.33% / 1042 | 8.33% / 690 |
| RDG (road) | 58.33% / 1771 | 62.50% / 755 | 95.83% / 59 | 4.17% / 1484 | 8.33% / 1813 |
| TransProtect (Markov) | 66.67% / 101 | 79.17% / 50 | 74.59% / 83 | 18.75% / 219 | 8.33% / 666 |
| Semantic (POI/thực nghiệm) | 83.33% / 59 | 100.00% / 13 | 84.39% / 42 | 33.33% / 312 | 0.00% / 767 |
| Fake-query (road) | 50.00% / 1536 | 33.33% / 1489 | 96.91% / 47 | 12.50% / 483 | 4.17% / 689 |
| AnotherMe (VTGA offline) | 0.00% / 1697 (2/3 ca) | NA | 0.00% / 1877 | 0.00% / 1565 (2/3 ca) | 0.00% / 1914 (1/3 ca) |
| Đề xuất 30 | 0.00% / 2104 | 0.00% / 2055 | 0.00% / 2068 | 0.00% / 2618 | 0.00% / 2207 |
| Đề xuất 67 | 0.00% / 2104 | 0.00% / 1859 | 0.00% / 2068 | 0.00% / 2488 | 0.00% / 2262 |

Bộ attacker có centroid, median, prior theo truy vấn, Viterbi lịch sử/chuyển động, ngoại suy OLS (2/3/6 quan sát, 30/60/120 giây), kNN, Extra Trees, prior học từ background, road projection và joint estimator cho S9.C/S10.C. Mỗi adapter được fit/chọn riêng bằng cùng luật. Đầu vào chỉ có tập tọa độ, thời gian tương đối trong cửa sổ được phép; evaluator giữ kín nhãn thật/giả, server state, index nguồn và đáp án. Chưa phải attacker trên đầy đủ clock tuyệt đối/account/IP/click hoặc mọi tín hiệu phản hồi.

**Kiểm tra mức đoán đúng của dữ liệu thật trước khi kết luận:** trên bốn nhóm, raw Hit100=100% ở S1–S3; S9.A/C=0%, S9.B=50%; S10.A/B=0%, S10.C=25%. Vì vậy bằng chứng endpoint chủ yếu ở S9.B/S10.C; chưa thể dùng Hit100 để khẳng định bảo vệ S9.A/C và S10.A/B. Ca B chỉ có hai nhóm. Hit=0 của đề xuất là kết quả trên bộ hữu hạn, không phải riêng tư tuyệt đối.

Bộ endpoint được tăng cường **sau khi xem kết quả control của bank đầu**, áp dụng cho tất cả phương pháp, không đổi tham số phòng vệ/dịch vụ và vẫn chỉ chọn decoder trên 601–602. Diagnostic ban đầu được giữ tại `diagnostics/initial_attack_bank.json.gz`; `attacker_protocol.json` ghi rõ phạm vi thăm dò này.

## Độ nhạy và bất định

| Cấu hình | Availability 50% | Availability 80% | Availability 95% |
| --- | --- | --- | --- |
| Đề xuất 30 | 97.85%; 15/15 | 94.53%; 15/15 | 91.82%; 11/15 |
| Đề xuất 67 | 100.00%; 15/15 | 100.00%; 15/15 | 100.00%; 15/15 |

| Đề xuất 30 so với | Δ Recall (điểm %) | Bootstrap 95% |
| --- | --- | --- |
| DLS (road) | -5.47 | [-7.24; -3.51] |
| RDG (road) | -5.47 | [-7.24; -3.51] |
| TransProtect (Markov) | -5.17 | [-6.69; -3.38] |
| Semantic (POI/thực nghiệm) | -5.47 | [-7.24; -3.51] |
| Fake-query (road) | -5.47 | [-7.24; -3.51] |

Bootstrap ghép theo nhóm 5.000 lần, thăm dò và chưa hiệu chỉnh đa so sánh. Bốn nhóm là quy mô nhỏ; không dùng khoảng này để tuyên bố tổng quát hóa cho thành phố/quần thể mới. Cấu hình 67 giữ utility tốt hơn nhưng còn tám POI ngoài độ phủ tĩnh, nên không có chứng nhận toàn bản đồ.

## Lỗi, kiểm tra và khả năng chạy lại

- AnotherMe: 164/204 lượt phiên lỗi ở 12 nhóm; 53/64 ở bốn nhóm. 216/217 lỗi do đầu ra thô ít hơn 20 mẫu; một lỗi không có tuyến có hướng. Giữ cả lỗi và mẫu số, không kéo dài quỹ đạo suy biến để làm hợp lệ.
- Năm adapter online vượt 15 kiểm tra tiền tố/tương lai trên ba phiên phụ trợ: sửa tương lai không đổi đầu ra đã công bố. AnotherMe full-trip được loại khỏi tuyên bố này.
- Verification tính lại 2.700 chỉ số, đối chiếu 2.412 transcript đánh giá và 114.289 bản tin với whitelist. Điều này xác nhận accounting và ranh giới đầu vào; không chứng nhận tái lập paper hoặc privacy phổ quát.
- 55 tests liên quan đã qua: 7 tests comparator/attacker mới và 48 tests core/service/category hiện có.

Các artifact: [`case_results.csv`](../../artifacts/benchmarks/live_paper_comparison_v1/case_results.csv) (270 dòng), [`readout.json`](../../artifacts/benchmarks/live_paper_comparison_v1/readout.json), [`protocol.json`](../../artifacts/benchmarks/live_paper_comparison_v1/protocol.json), [`attacker_protocol.json`](../../artifacts/benchmarks/live_paper_comparison_v1/attacker_protocol.json), [`verification.json`](../../artifacts/benchmarks/live_paper_comparison_v1/verification.json). Full service observations, attack errors/selection and seed-locked transcripts are retained beside them.

Lệnh kiểm tra bằng môi trường dự án (NumPy/SciPy/scikit-learn/NetworkX và tài nguyên SUMO đã dựng):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.verify_live_paper_comparison
python -m pytest tests/test_paper_comparators.py tests/test_anotherme_faithful.py tests/test_semantic_correlation_faithful.py tests/test_transprotect_faithful.py tests/test_live_poi.py tests/test_category_cover.py tests/test_category_client.py
```

Runner: `experiments.run_live_paper_comparison` gồm `prepare`, `generate --split auxiliary/development/new_groups`, `evaluate --split development/new_groups`; `experiments.read_live_paper_comparison` gồm `attacks` và `readout`. Mỗi split gọi riêng. Stage tạo protocol/đánh giá không ghi đè artifact đã có; giữ một snapshot đầu ra riêng trước khi chạy lại. Cần giữ cả attacker protocol và phiên bản bank tương ứng, không trộn v1/v2.

Điểm Q chưa được xếp hạng mới vì thiếu latency đầu-cuối. Fixed K5/K12 và bulk là kiểm tra thiết kế bổ sung ở hồ sơ cũ. Với API bulk và danh mục nhỏ hiện tại, phương pháp chưa chứng minh lợi thế tài nguyên trước tải toàn bộ trạng thái.

Readout SHA-256: `b64ec0f5c1754ed2271069e958eae1348c7e5b908d3fb48720bba96629f5fa19`.

Kiểm tra tiếp theo: [S9/S10 và cohort 32 nhóm](endpoint_calendar_results.md)
bổ sung attacker theo đường và client giữ lịch công khai trước/sau chuyến.
Số mới được trình bày riêng, không thay hoặc gộp vào các bảng lịch sử ở đây.
Recall 100% của tập thật + dummy có nguyên nhân từ giao diện top-10/top-5;
tăng sample nhằm kiểm tra privacy/độ phủ và độ bất định, không ép Recall giảm.
