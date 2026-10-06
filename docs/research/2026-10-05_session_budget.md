# Ngân sách Geo-I qua nhiều chuyến: một tài khoản, một thời hạn cố định

**Thay đổi là cách chia và giữ ngân sách; Geo-I, REM và phép thử tái sử dụng điểm vẫn giữ nguyên.** Trước đây mỗi chuyến của cấu hình Endpoint20 có giới hạn `0,0575 m⁻¹`. Sáu chuyến liên kết có thể cộng thành `0,345 m⁻¹`; không thể gọi cả sáu chuyến là được bảo vệ với cùng giới hạn `0,0575`.

Geo-I giới hạn mức thay đổi xác suất đầu ra khi thay đổi vị trí. Việc gửi nhiều thông tin được bảo vệ vẫn cần cộng chi phí; phép thử tái sử dụng cũng có chi phí. Cách quản lý ngân sách dự đoán và thử riêng tư có nền tảng từ [Geo-Indistinguishability, CCS 2013](https://arxiv.org/abs/1212.1984) và [A Predictive Differentially-Private Mechanism for Mobility Traces, PETS 2014](https://arxiv.org/abs/1311.4008). Cơ chế của dự án và phạm vi lập luận được ghi trong [predictive_filter_argument.md](predictive_filter_argument.md).

## Chính sách cố định trước khi xem kết quả

Ấn định một ngày và tối đa **6 phiên**. Chia đều giới hạn hiệu dụng `C=0,23 m⁻¹`; mỗi phiên nhận `C/6≈0,038333 m⁻¹`. Khi bắt đầu phiên, giữ toàn bộ phần ngân sách đó trong sổ cục bộ. Phiên ngắn không trả lại ngân sách dựa trên đường đi hay số lần GPS đã dùng. Tắt và mở lại mô hình không cấp thêm phần ngân sách. Phiên thứ 7 không đọc GPS và không gửi truy vấn.

Với `H` cố định, bộ lọc hiện có cho tối đa `2H−1` đơn vị. Vì vậy:

\[
u=\frac{C}{6(2H-1)},\qquad B_{\text{phiên}}=2Hu.
\]

`u` là ε của **mỗi lần tạo điểm tham chiếu** và **mỗi phép thử tái sử dụng**. Điểm đầu tốn một đơn vị; lần sau tốn một đơn vị nếu tái sử dụng, hai nếu thử rồi tạo mới. Trước khi lấy GPS, bộ lọc phải còn đủ cho trường hợp xấu nhất. Hết đơn vị thì chỉ dự đoán và chọn Q từ thông tin đã bảo vệ; không hỏi GPS mới.

| Cấu hình của cùng pipeline | H | ε thử = ε tạo điểm /m | B danh nghĩa /phiên | Giới hạn hiệu dụng /phiên | Tổng hiệu dụng /6 phiên |
|---|---:|---:|---:|---:|---:|
| Reset từng phiên, tham chiếu hiện tại | 12 | 0,0025 | 0,06 | 0,0575 | 0,345 |
| Một giới hạn chung, H12 | 12 | 0,001667 | 0,04 | 0,038333 | **0,23** |
| Một giới hạn chung, H8 | 8 | 0,002556 | 0,040889 | 0,038333 | **0,23** |
| Đối chứng cùng tổng ngân sách với reset | 12 | 0,0025 | 0,06 | 0,0575 | 0,345 |

H8 chia cùng giới hạn cho ít lần thử/tạo hơn, nên mỗi lần có ε lớn hơn và thường ít nhiễu hơn. Đổi lại, chuyến dài có thể hết lượt đọc GPS sớm hơn. **H được chọn công khai trước; không chọn H theo đường đi hoặc thời lượng riêng tư của người dùng.** Hai hàng H8/H12 mới là so sánh cùng tổng giới hạn. So sánh với reset là so sánh đánh đổi giữa hai mức ngân sách.

H không phải số GPS chắc chắn được đọc: nếu lần nào cũng tạo điểm mới, H8 đủ cho 8 lần; nếu tái sử dụng thì mỗi lần chỉ tốn một đơn vị và có thể đọc thêm, nhưng luôn dừng trong 15 đơn vị của phiên.

## Điều gì được bảo đảm, điều gì cần đo

Trong lập luận kernel lý tưởng, với thời điểm và số phiên công khai cố định, đặt `D∞` là khoảng cách lớn nhất giữa các GPS tương ứng trong toàn bộ thời hạn. Mỗi phiên có tỷ lệ xác suất tối đa `exp(C/6 × D∞)`; nhân các phiên cho tối đa **`exp(C × D∞)`**. Chọn Q, tìm POI và lọc POI từ thông tin đã bảo vệ không cộng thêm ε. Đây là lập luận về **tọa độ**, không phải định lý ẩn danh tính.

Người tấn công vẫn có thể biết tài khoản, IP, thời điểm bắt đầu/kết thúc hoặc nhận ra kiểu đường đi. AUC liên kết người/xe vẫn phải được đo bằng attacker mạnh; không suy ra rằng AUC sẽ giảm chỉ vì sổ ngân sách đúng. Đối chứng cùng tổng ngân sách phải cho **Q, trạng thái và lịch thử giống hệt reset** khi dùng cùng các luồng ngẫu nhiên. Nếu nó khác, đó là lỗi triển khai hoặc cấu hình; sổ ngân sách tự nó không tạo thêm nhiễu.

Các kernel hiện có dùng số thực và PRNG hữu hạn. Luồng riêng tư tách bằng HMAC tránh dùng seed công khai và tránh vô tình dùng chung RNG giữa các phiên; nó không biến sampler hiện tại thành một triển khai pure-DP an toàn ở độ chính xác hữu hạn.

## Tích hợp trong client

Mã mới nằm ở [core/session_budget.py](../../core/session_budget.py). Dùng **một SQLite ledger chung cho cùng chủ thể/xe và thời hạn**, giữ ở bộ nhớ cục bộ riêng tư; mở lại đúng file để giữ số phiên đã nhận. Một giao dịch giữ phần ngân sách trước khi khởi tạo mô hình hay lấy GPS. Nếu crash, phần đã giữ vẫn mất. File/key được giữ ngoài artifact công khai; wrapper không gửi token phiên, seed, nhánh thử hay số dư cho attacker.

- `FixedEpochPolicy(...)`: ngày, số phiên, H, tổng giới hạn hiệu dụng và khoảng đọc GPS công khai.
- `PersistentEpochBudget(policy, path)`: lưu cap đã cấp và khóa OS ngẫu nhiên riêng tư; file có quyền `0600`.
- `FixedEpochProtectedSessions(ledger, engine_factory)`: `start_session(token, t)`, `protect_step(t, gps_supplier)`, `close_session(t)`. `gps_supplier` chỉ được gọi khi đồng hồ công khai và bộ lọc cho phép đọc.
- Factory dùng `allocation.nominal_budget_per_m`, `allocation.horizon`, `allocation.read_interval_s`. **Belief phải khớp `allocation.unit_epsilon_per_m` cho cả emission thử và emission tạo điểm.** Giữ nguyên prior/mạng đường công khai; chỉ tính lại emission cho ε của phần ngân sách.

`SessionAllocation` và các hàm RNG là API nội bộ của client đáng tin cậy. Việc xóa/rollback ledger, dùng ledger riêng cho hai thiết bị liên kết, hoặc tự dựng lại mô hình ngoài wrapper phá điều kiện triển khai. Module không có cơ chế chống client độc hại hoặc tự khám phá người/xe nào cùng danh tính. Ngày kế tiếp là một ngân sách mới **cộng với** ngày trước; không xóa chi phí lịch sử trong một tuyên bố dài hơn.

## Thực nghiệm S4 được lưu riêng

Giao thức ở [round3/protocol.json](../../artifacts/benchmarks/session_budget_20261005/round3/protocol.json), kết quả ở [round3/readout.json](../../artifacts/benchmarks/session_budget_20261005/round3/readout.json). Có 72 phiên SUMO gốc × 2 lần ngẫu nhiên độc lập; giữ nguyên toàn bộ 72 phiên và kết quả trước đó. Mỗi nhóm nghiên cứu gồm sáu vai trò được cấp **một** giới hạn chung, nên tất cả tập con cùng người hoặc cùng xe đều nằm trong giới hạn đó. Đây là cách gom bảo thủ của thực nghiệm, không phải suy luận danh tính từ GPS.

Attacker nhận Q và thời gian, dùng đặc trưng tổng hợp và ghép hình dạng K đường bằng Hungarian, rồi **ExtraTrees / kNN1 / kNN5 / kNN15**. Chọn attacker và ngưỡng trên các family selection; báo cả lựa chọn tốt nhất theo balanced accuracy và ROC-AUC, cả kết quả từng family để không che family có AUC cao. Toàn bộ phiên và seed của một family ở cùng split. Các family test này đã được xem trong vòng phát triển trước, nên kết quả mới vẫn là **development**, chưa phải xác nhận độc lập.

Utility đo Recall@5 khi hợp kết quả top20/category/Q rồi lọc tại GPS cục bộ, trên cùng trạng thái POI khả dụng. Báo riêng tám tick đầu và phần công khai `t≥480s`; 23/72 chuyến thật có phần sau 480s. Đây là purpose tìm gần nhất, không thay thế thực nghiệm S7 nhiều purpose. Bytes chỉ gồm JSON truy vấn và ID POI, chưa gồm HTTP/TLS.

| Cấu hình | Tổng cap /6 phiên | AUC liên kết người ↓ | AUC liên kết xe ↓ | Recall@5 ↑ | Bytes/tick ↓ |
|---|---:|---:|---:|---:|---:|
| GPS gốc | — | 0,773 | 0,716 | 100% | 1.415 |
| Reset mỗi phiên, H12 | 0,345 | 0,632 | 0,544 | 97,03% | 6.966 |
| Cap chung, H12 | **0,23** | 0,640 | 0,577 | 95,53% | 6.967 |
| Cap chung, H8 | **0,23** | 0,526 | 0,504 | 97,23% | 6.966 |

Đây là test development trên 3 family, 36 phiên tính cả hai lần ngẫu nhiên, 90 cặp/target. AUC trong bảng dùng attacker chọn **trên selection theo ROC-AUC**, không chọn attacker theo điểm test; 0,5 là mức ngẫu nhiên. Lựa chọn riêng theo balanced accuracy của H8 vẫn cho AUC người `0,556`, nên phải đọc cả hai.

**H8 là cấu hình đáng kiểm tra tiếp, chưa phải kết luận giải quyết S4.** H12 có cap tốt hơn reset nhưng AUC không tốt hơn trong lần chạy này. Trong toàn bộ bank attacker cố định, AUC cao nhất của một family vẫn là `0,694` cho người và `0,681` cho xe ở H8; H12 lần lượt `0,889` và `0,743`. Đây là kiểm tra rủi ro mô tả, không dùng để chọn cấu hình hoặc attacker sau khi xem test.

Chỉ có **7 tick test sau khi H8 hết ngân sách đọc GPS**: Recall 100% ở đoạn đó chưa đủ chứng minh chuyến dài được phục vụ tốt. Trên train có 56 tick như vậy và Recall trung bình cân bằng family là **87,51%**. Sổ ngân sách đúng không tự giảm bytes: sau khi hết lượt, mô hình vẫn gửi Q được dự đoán từ lịch sử đã bảo vệ. Đối chứng cùng tổng `0,345` khớp reset chính xác ở cả **144 phiên** về Q, trạng thái và nhánh thử, đúng với kỳ vọng.

Các file cơ chế `core/mechanisms.py` và toàn bộ bộ lọc/pacing được pin SHA-256, kiểm tra giữ nguyên; kết quả cũ cũng được pin. Hai lỗi chuẩn bị (prior sai kích thước, so sánh float bằng dấu bằng tuyệt đối) được giữ cùng folder và đều xảy ra trước khi có điểm attacker. Chúng không dẫn tới đổi ε, H, dữ liệu hay quy tắc chọn mô hình theo kết quả.

Đọc lại độc lập: `python -m experiments.verify_session_budget`. Tái chạy vào folder mới: `python -m experiments.session_budget_linkage --out /private/tmp/new-budget-readout`; khóa và ledger thực luôn nằm riêng ở `/private/tmp`.
