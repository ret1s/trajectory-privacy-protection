# Bổ sung phép đo gốc của các phương pháp đối chứng — 05/10/2026

**Quyết định:** giữ Hit/MAE và chất lượng dịch vụ chung, đồng thời bổ sung metric gốc khi có đủ đầu vào và cùng ý nghĩa. Không biến một tập Q thành một tọa độ bằng cách chọn điểm gần GPS thật nhất. Không coi metric thiếu dữ liệu là 0 hoặc là bảo vệ hoàn hảo. Việc dùng metric gốc giúp kiểm tra công bằng; chưa bảo đảm phương pháp đề xuất thắng mọi metric.

## Định nghĩa và phạm vi có thể chấm

| Đối chứng đang chạy | Metric gốc và hướng đọc | Đầu vào cần có | Đối chiếu với Geo-I dummy-only |
|---|---|---|---|
| **TransProtect** | EIE ↑: sai số vị trí mà attacker suy ra; Δc ↓: chênh chi phí đi tới các đích, lấy kỳ vọng theo prior | Ước lượng attacker/đáp án; bảng chi phí từ vị trí thật và vị trí công bố tới cùng tập đích | EIE áp dụng cho mọi output qua attacker. Δc gốc dùng một vị trí công bố; tập 5 Q cần báo phần mở rộng riêng |
| **DLS** | Cell entropy ↑, dựa vào trọng số lịch sử của các thành viên trong tập | Prior lịch sử, tập thật + dummy | Q-only không có một thành viên thật để giải thích entropy này như xác suất tìm ra người dùng |
| **RDG** | Cell/transition entropy ↑; tỷ lệ vị trí được bảo vệ trước Viterbi ↑ | Lịch sử query, transition, posterior/path weights và vị trí thật trong tập | Không thay trọng số max-product bằng posterior forward; không ép các giả định của tập chứa thật lên Q-only |
| **Semantic correlation** | ASR ↑: tỷ lệ query có P(thành viên thật) ≤ 1/K; DER ↑: phần dummy hợp lý về ngữ nghĩa | Posterior của attacker, nhãn thành viên thật; quyết định hiệu quả của K−1 dummy | ASR/DER gốc là N/A trên Q-only; không gán ASR=100% chỉ vì không có GPS thật trong tập |
| **Fake-query insertion** | Số đường khó phân biệt giữa hai query ↑; ASR theo từng kiểu attack ↑; thời gian/RSS ↓ | Kiểm tra tương đồng thời gian tới nơi và hướng di chuyển; attacker và phép đo tài nguyên tương ứng | Có thể đo số đường như chẩn đoán nếu cùng phép kiểm tra; chỉ kiểm tra đi tới được chưa đủ |
| **AnotherMe** | Thử nghiệm nhận diện thật/ảo và mobile workflow cần đúng bộ đánh giá nguồn | Dữ liệu/nhãn/recognition model, phân chia và thiết bị/dịch vụ phù hợp | Chưa xác minh đủ protocol toàn hệ thống để tái lập; không tùy ý gọi một classifier mới là metric gốc của paper |

Nguồn trực tiếp: [TransProtect §4.4, Eq. 13 và §5.1.4](https://arxiv.org/html/2409.09495v1), [DLS — bản tác giả](https://mcn.cse.psu.edu/paper/other/infocom-ben-niu14.pdf), [RDG — bản tác giả 2018, §IV–VII](https://arxiv.org/pdf/1805.06104), [Semantic correlation §6.3–6.5](https://link.springer.com/article/10.1007/s44443-026-00899-w), [Fake queries §6.3–6.6](https://link.springer.com/article/10.1007/s44443-025-00438-z), [AnotherMe — mã nguồn tác giả](https://github.com/fang-zhiyou/AnotherMe). Bản DLS trên server tác giả không mở được trực tiếp trong lần kiểm tra này; định nghĩa cell entropy cũng được xác minh ở §VII.B.1 của bản RDG. DOI AnotherMe không cung cấp toàn văn qua công cụ trong lần này; không suy ra công thức metric từ tên paper.

**Không đồng nhất entropy với privacy thực nghiệm.** Entropy của các trọng số trong tập có thể cao trong khi cả chuỗi vẫn bị suy luận. DER là chẩn đoán độ hợp lý; khoảng cách giữa GPS và tọa độ công bố là độ méo đầu ra. Chúng không thay được sai số/tỷ lệ thành công của một attacker có quyền quan sát được quy định rõ.

## Công thức được hiện thực

- **EIE của point estimator:** `mean_t d(x_t, xhat_t)`. Khi dùng cùng ước lượng, cùng metric khoảng cách và cùng phép gộp, đây chính là MAE đang có. Đổi tên sang EIE không tạo ra bằng chứng mới. `posterior_expected_error_m` tính riêng `mean_t sum_j p_t(j) d(x_t,j)`; hai phép đo này không được trộn.
- **Chi phí:** `mean_t sum_l q_l abs(c(x_t,l) - c(z_t,l))`. Phải lấy trị tuyệt đối **trước** kỳ vọng. Các đích, prior, đơn vị, hướng đường và cách xử lý không tới được phải giống nhau giữa phương pháp.
- **Entropy:** `-sum_i p_i log2(p_i)`, với `p_i` được chuẩn hóa từ các trọng số lịch sử đã cung cấp. Một phân bố đều tự tạo không phải posterior của attacker.
- **ASR:** `100 × mean[P(real) <= 1/K]`; **DER:** trung bình theo query của tỷ lệ dummy được đánh giá hiệu quả. Paper Semantic không cung cấp một hàm `Sim`/posterior thực thi đủ để tự tái lập các giá trị gốc.

Mã: [`evaluation/native_comparator_metrics.py`](../../evaluation/native_comparator_metrics.py). API chính `comparator_native_metrics(output_kind, ...)` trả từng metric kèm `value`, `unit`, `direction`, `category`, `denominator`, `status`, `reason`. Thiếu đầu vào là `not_available`; khác hợp đồng output là `not_applicable`; cả hai dùng JSON `null`.

Các helper dùng lại được:

```python
inference_error_m(errors_m)
posterior_expected_error_m(posterior_rows, distance_rows_m)
posterior_entropy_bits(probabilities)
weight_entropy_bits(weights)
expected_travel_cost_distortion(real_cost_rows, released_cost_rows, target_prior)
anonymity_success_rate(real_posteriors, k_values)
dummy_effectiveness_rate(effective_boolean_flags)
```

Với Q-only, có thể gọi Eq. 13 **cho từng Q**, rồi báo phân bố/mean theo query như `per_query_cost_distortion`, một **chẩn đoán mở rộng**, không phải chi phí của kết quả cuối cùng. Đánh giá dịch vụ cuối vẫn dựa vào POI đã nhận và cách chọn trên thiết bị. Không lấy min theo GPS thật để cho mô hình một ưu thế mà server không có.

## Đã tính lại từ dữ liệu đóng băng

Runner [`experiments/native_metric_readout.py`](../../experiments/native_metric_readout.py) đã tạo [readout mới](../../artifacts/benchmarks/native_metrics_20261005/readout.json), [bảng CSV](../../artifacts/benchmarks/native_metrics_20261005/summary.csv) và [biên bản kiểm tra](../../artifacts/benchmarks/native_metrics_20261005/verification.json). **5.336 dòng nguồn, 135 bảng gộp theo cohort/phương pháp/scenario.** Không fit lại, không đổi mẫu và không sửa nguồn benchmark.

Đã kiểm tra độc lập **431 dòng thành công của paper-v2 K=5**: chiếu đáp án GPS/điểm biên bằng đúng projection lưu trong manifest, tính lại khoảng cách tới ước lượng attacker đã khóa. EIE khớp MAE lưu sẵn. Các lỗi/không áp dụng vẫn giữ trong mẫu số coverage.

Ví dụ EIE (m) trên **cùng paper-v2**, cao hơn nghĩa attacker đang sai xa hơn:

| Phương pháp | S1 | S2 | S3 | S9 | S10 |
|---|---:|---:|---:|---:|---:|
| DLS road adaptation | 1.318 | 39 | 1.688 | 1.582 | 2.029 |
| TransProtect Markov adaptation | 84 | 88 | 99 | 184 | 680 |
| BR tái dùng Geo-I | 299 | 247 | 776 | 316 | 2.315 |

BR có EIE lớn hơn TransProtect ở năm scenario, nhưng chưa lớn hơn DLS ở S1/S3/S9. Phải đọc cùng Hit100, Recall và chi phí; không kết luận thắng mọi mặt. AnotherMe chỉ thành công 11/12 chuyến S3, không được thay N/A của các scenario còn lại bằng điểm tốt.

**GeoI-Slack hiện tại có protocol khác**, nên báo riêng: EIE S1/S2/S3/S9/S10 lần lượt **1.119/874/904/607/1.273 m**. Đây là dữ liệu phát triển cũ được đọc lại, không phải một phép so mới với các đối chứng trên cùng attacker.

Giới hạn dữ liệu gốc:

- Các archive so sánh không giữ bảng chi phí tới đích và cache đường gốc, nên **chưa tính lại Δc** cho cùng cohort. Không thay bằng khoảng cách đường thẳng hoặc bản đồ demo tái dựng. Smoke lịch sử 8 sự kiện giữ scalar Δc=62,584 m cho TransProtect; readout ghi `archived_reported_not_recomputed`, không đưa vào bảng so trực tiếp.
- Không có posterior ứng viên đã hiệu chuẩn, đầy đủ prior/transition và nhãn `Sim`, nên entropy/ASR/DER/path similarity còn N/A. Không tạo posterior bằng cách đo khoảng cách tới GPS thật.
- Chưa tái lập VehiTrack, Google Maps arrival-time hay AnotherMe mobile measurement. Các lớp hiện có vẫn là local adaptations, không phải điểm số của paper gốc.
- Không xếp hạng các protocol riêng thành một bảng chung; bỏ public-cover 30/67 khỏi readout mới vì chúng không phải mô hình Geo-I của ta.

## Cách dùng cho vòng thực nghiệm mới

Khóa đồng thời các mô hình, attacker, prior đích và mục đích query. Lưu các bảng chi phí thực dùng, posterior của attacker khi có, quyết định semantic/path similarity và toàn bộ lỗi. Chấm từng phương pháp bằng **metric chung + metric gốc có hợp đồng phù hợp + chẩn đoán mở rộng được đặt tên riêng**. Chọn cấu hình trên tập chọn, sau đó mới chấm tập giữ lại. Báo kết quả thua/tie và đánh đổi, không chọn metric/mẫu chỉ để khẳng định vượt trội.

```sh
python -m pytest -q tests/test_native_comparator_metrics.py
python -m experiments.native_metric_readout --output /private/tmp/native-metric-review-new
```

Runner từ chối ghi đè thư mục kết quả đã tồn tại. Đã qua **16 tests**, gồm Eq. 13 đối chiếu core hiện có, không triệt tiêu sai số bằng kỳ vọng có dấu, kiểm tra EIE từ tọa độ, mẫu số lỗi và chặn ASR giả 100% trên Q-only.

## Bổ sung metric-family thực đo trên mạng mới: chi phí tới các đích công khai

Đã chấm **GeoI-Endpoint20** (`scale025_L20`) bằng phép đo thuộc nhóm Eq. 13, từ public Q đã đóng băng của vòng endpoint mới. Đây là **chẩn đoán hậu nghiệm** sau khi cấu hình đã được chọn theo privacy/Recall; không dùng chi phí này để chọn lại mô hình. Readout cũ ở trên giữ nguyên.

Prior đích là **đều trên toàn bộ 418 POI OSM công khai được chấp nhận**; không học prior từ GPS hay lượt ghé. Tính khoảng cách đường **có hướng** từ GPS đã snap và từng Q tới cùng các đích, trên đúng mạng tái dựng của source transcript. Không dùng khoảng cách đường thẳng, không thêm penalty cho đường không tới được, không chọn lại tập đích theo kết quả. Nếu một Q không tới được một đích có prior dương thì toàn bộ metric ở mốc đó giữ N/A; không lấy `inf-inf=0` hay bỏ Q đó.

| Phép đo trong cohort mới | Kết quả | Đọc thế nào? |
|---|---:|---|
| Raw singleton: Δc theo Eq. 13 | **0 m** | Vị trí truy vấn trùng vị trí thật sau cùng phép snap |
| GeoI-Endpoint20: trung bình Δc của **cả 5 Q** | **1.386,74 m** | Chẩn đoán mở rộng Q-only; không chọn Q gần GPS nhất |
| Phân bố Δc của từng Q | Median **1.379,69 m**; p95 **2.340,46 m** | Phân bố trên 850 Q đủ đầu vào |
| Coverage | **170/170 mốc**, 4 family, 8 chuyến × 2 seed | Cả 5 Q đều tới được đủ 418 đích trong mẫu này |

**Số lớn hơn ở đây là độ méo chi phí lớn hơn, không phải privacy tốt hơn.** Không có nghĩa người dùng phải đi thêm 1.387 m: điện thoại vẫn hợp POI rồi xếp hạng bằng GPS thật. Metric này đo sự khác nhau giữa chi phí ước tính từ tọa độ Q và từ vị trí thật; cần đọc cùng chất lượng POI cuối cùng. Không dùng raw-vs-Q để tuyên bố thắng TransProtect: singleton TransProtect chưa được chạy trên cohort này, và hai hợp đồng dịch vụ khác nhau.

Đã giữ [bảng chi phí chính xác](../../artifacts/benchmarks/native_cost_diagnostic_20261005/target_cost_tables.npz) gồm **400 trạng thái đã dùng × 418 đích**, prior/ID đích, từng mốc và N/A. Bảng giữ cả `+inf` nếu có, không sửa cho số đẹp. Có **3.200** kiểm tra khoảng cách độc lập bằng NetworkX trên tám đích chọn theo thứ tự công khai; kiểm tra lại toàn bộ Δc từ bảng đã lưu. Nguồn benchmark/network/transcript không đổi. [Readout](../../artifacts/benchmarks/native_cost_diagnostic_20261005/readout.json) và [protocol](../../artifacts/benchmarks/native_cost_diagnostic_20261005/protocol.json) nói rõ các luật rẽ tái dựng chưa tương đương cache SUMO gốc. Đã qua **7 tests** riêng về hướng đường, averaging cả Q, prior và N/A.

```sh
python -m experiments.native_cost_diagnostic --out /private/tmp/new-native-cost-diagnostic
python -m experiments.verify_native_cost_diagnostic
python -m pytest -q tests/test_native_cost_diagnostic.py
```

**Δc trong benchmark gốc vẫn N/A.** Kết quả mới này không lấp chỗ trống đó bằng cách đổi mạng/cohort; nó bổ sung một phép đo thực ngoài MAE, có hợp đồng và dữ liệu kiểm tra được.
