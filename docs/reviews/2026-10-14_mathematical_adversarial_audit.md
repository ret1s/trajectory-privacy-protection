# Rà soát đối kháng chứng minh GeoI-BQ

Đối tượng rà soát là [bản cơ sở toán học](../research/2026-10-14_geoi_bq_mathematical_foundation.md)
và việc các giả thiết của nó được bảo đảm trong mã của GeoI-BQ v1.
Đây là lần kiểm tra bổ sung trên cùng cấu hình; không lấy mẫu lại GPS/Q,
chọn lại attacker hoặc thay benchmark.

**Kết luận:** chưa tìm thấy phản ví dụ cho normalization, common-support
likelihood cap, adaptive χ²/TV bound hoặc Bayes bounds **dưới các giả thiết
đã ghi rõ**. Có hai điều kiện triển khai chưa được toàn chương trình chứng
nhận: kernel Geo-I dùng float và khôi phục đầy đủ upstream qua restart.
Một số bước trước đây viết quá gọn đã được bổ sung điều kiện và diễn giải.
Pass finite oracles không thay thế chứng minh tổng quát hoặc review độc lập.

## Các phát hiện và cách xử lý

| Phát hiện | Vì sao có thể làm lập luận sai? | Trạng thái sau rà soát |
|---|---|---|
| **A1. Cận Geo-I lý tưởng và cận Q bị gộp phạm vi** | REM float chưa có chứng chỉ cùng PMF lý tưởng; không tự suy `Cs·D` cho toàn executable | Đã tách: cận Q `R_n` không cần upstream Geo-I đúng; chỉ phần minimum với `G` cần ideal Geo-I. Gap sampler còn mở |
| **A2. Restore upstream chưa được xác thực đầy đủ** | Matching timestamp/config không chứng minh đã giữ history, ledger và RNG; một engine reset có thể được coi là restored | Đã tái hiện bằng actual `PacedSlackProgressLaneDummy`. Q ledger giữ nguyên; giả thiết trusted upstream restore vẫn cần. Chưa sửa frozen core hoặc gọi đây là restore an toàn |
| **A3. Score có thể ngẫu nhiên tại cùng public prefix** | Covariance của marginal mean scores không bằng average covariance của local-state scores | Đã điều kiện công thức gain theo local state; dùng cùng state law cho counterfactual. Phản ví dụ exact có gain dương trong khi covariance marginal bằng0 |
| **A4. Floor predicted pool không tự là floor received cache** | Generic server có thể trả thiếu POI dù cùng response law và metadata ổn định | Đã thêm response completeness/signature matching. Actual client nhận empty replies hợp lệ và cache floor0 dù predicted floor dương. Monotone cache vẫn đúng |
| **A5. Input index, endpoint và causal scope** | Đánh số chỉ các GPS đã đọc làm hai trace khác miền; public termination riêng tư hoặc future target do Q tạo ra không nằm trong open-loop proof | Đã cố định lịch potential inputs, phép chiếu mét, xử lý skipped events; endpoint event-bound chỉ áp trực tiếp khi là scheduled input. Interactive/future-induced targets vẫn cần model riêng |
| **A6. Augmented history khác wire coordinates** | Lane aliases làm support ở bước sau không xác định duy nhất từ wire. Không thể dùng nguyên centered kernel ở coordinate prefix | Đã viết rõ chứng minh trên augmented history và conditional-expectation/Jensen khi chiếu xuống wire; oracle nhiều bước cập nhật hidden-state posterior thật |
| **A7. Public stopping khác output conditioning** | Giữ lại subset rồi chuẩn hóa có thể tăng TV; cận average/info không tự đúng cho subgroup đã lọc | Đã phân biệt hai phép toán. Phản ví dụ TV toàn law0,004 nhưng sau conditioning0,4 |
| **A8. Điểm có xác suất0 và sequence vô hạn** | `P(O=o)` có thể bằng0 cho replies liên tục; “largest distances” không nhất thiết tồn tại trên sequence vô hạn | Đã dùng likelihood densities/a.e. posterior; `D∞` và TopSum là supremum trên sequence, finite trace vẫn dùng max/top sums |

A1–A2 là điều kiện chưa được chứng nhận của implementation, không phải
lỗi trong phép nhân PMF Q. A3–A8 đã được đóng ở cách phát biểu/chứng minh
hoặc điều kiện chuyển định lý sang hệ thống cụ thể. Chứng minh không được
dùng để biến các điều kiện còn mở thành thuộc tính của mọi deployment.

## Oracle và phản ví dụ đã chạy

[Oracle mới](../../tests/test_geoi_bq_adversarial_proof_audit.py) gồm:

1. Duyệt joint private hidden state → adaptive Q/support → randomized
   provider replies → wire lane aliases trên ba selections. Posterior của
   hidden state thay theo từng sampled history; không giữ prior cũ để tính
   conditional kernel. Kiểm tra normalization, `χ²_wire ≤ χ²_augmented`,
   conditional-expectation identity, full likelihood ratio và optimal Bayes
   gain với cả prior đều và không đều.
2. Upstream có support hoàn toàn khác giữa hai inputs, thậm chí tiết lộ X
   hoàn hảo nếu công bố: Q-only wire vẫn giữ global bound. Điều này kiểm
   tra đúng việc cận Q độc lập với chất lượng upstream Geo-I.
3. Covariance trước/sau averaging private state; output-conditioning bias;
   private support/private stop làm support disjoint dù scores bằng0.
4. Predicted floor so với received floor, cả toy service references và
   `BoundedQuerySession` nhận empty common replies.
5. Actual engine reset nhưng timestamp khớp được nhận bởi pipeline:
   upstream chi phí1 thay vì ít nhất3 ở history cũ; Q reservation vẫn3/4.
   Đây là một **phản ví dụ cho việc timestamp là chứng nhận restore**,
   không là phép thử PASS rằng rollback đã được bảo vệ.

Các tests sampler-support có sẵn cũng đã được chạy: chúng ghi nhận defect
float, không khẳng định NumPy Gumbel có cùng PMF REM lý tưởng. Không thay
mã đã đóng băng hoặc bỏ phản ví dụ để có kết quả PASS.

## Cách đọc kết quả chính sau audit

- **Cận Q:** toàn logical transcript một phiên có likelihood ratio bị
  chặn, cùng số hữu tỷ như trước; không cần belief calibrated hoặc Geo-I
  float có PMF hoàn hảo. Vẫn cần common support, trusted Q ledger, fresh
  uniform bits và không có private-dependent packet/error/clock channel.
- **Cận Geo-I:** `G` và `Cs·D∞` là ideal-upstream guarantee, với cap và
  trusted full restore. Công thức minimum đúng khi cả hai cận cùng đúng.
- **Identity/future:** giới hạn additional inference với prior/context/
  causal target hợp lệ, không phải anonymity của tài khoản đã định danh hay
  bảo đảm cho một future action được tạo từ chính Q.
- **Utility:** cache monotonicity và whole-path distortion đúng theo contract.
  Public mean floor cần correct replies; actual-user floor vẫn cần các
  history-conditional TV/proxy bounds chưa được mặc định bằng0. Mean benchmark
  99,595% không biến thành worst-case theorem.

Priorities nếu chuyển từ research reference sang implementation được
chứng nhận là exact/verified upstream sampling và authenticated full
upstream checkpoint/recovery. Hai việc này không đòi thay Geo-I backbone,
nhưng cần một release/protocol riêng để không sửa nguồn benchmark đã khóa.

## Tái lập

**57 tests đạt**, gồm8 kiểm tra đối kháng mới. Một test PASS có thể xác
nhận rằng phản ví dụ còn tái hiện được; nó không có nghĩa điều kiện mở
đã được sửa. Số hữu tỷ privacy/utility/inference không đổi;108 main-reference
files vẫn được xác thực.

Chạy các finite oracles mới và các kiểm tra kernel/utility/inference/ledger:

```sh
/private/tmp/trajectory-query-segment-venv/bin/python -m pytest -o addopts='' -q \
  tests/test_geoi_bq_adversarial_proof_audit.py \
  tests/test_reference_utility_transfer.py tests/test_geoi_bq_proof_chain.py \
  tests/test_finite_query_kernel.py tests/test_finite_session_bounds.py \
  tests/test_conditional_gain.py tests/test_whole_path_chi_square.py \
  tests/test_session_utility_bounds.py tests/test_bounded_inference.py \
  tests/test_geo_i_event_bound.py tests/test_exact_public_reference.py \
  tests/test_bounded_query_session.py tests/test_sampler_support.py
/private/tmp/trajectory-query-segment-venv/bin/python -m experiments.verify_geoi_bq_mathematics_20261014 --check
```

Kết quả, sources và số hữu tỷ được ghim trong
[supplement receipt](2026-10-14_mathematical_completion.json).
Receipt xác thực108 files của main reference và hash nguồn bổ sung;
`--check` tái tính các cận số, còn finite oracles chạy bằng pytest ở trên.
Các conditional transfer assumptions vẫn có
`assumptions_verified_by_this_function=False`.
