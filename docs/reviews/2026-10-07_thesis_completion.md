# Luận văn Geo-I: bổ sung và kiểm tra ngày 07/10/2026

[PDF canonical](../../artifacts/reports/graduation_thesis.pdf) có **74 trang**,
dựng từ [main.tex](../../thesis/main.tex). Bản này bổ sung bằng chứng và cách
giải thích cho GVHD phản biện; giữ nền tảng Geo-I/REM và kết quả xác nhận L30.

## Bằng chứng mới

| Nội dung | Kết quả | Phạm vi |
|---|---|---|
| GPS local mỗi 60 s | L30 tăng Recall4 **2,24–2,61 pp** so L20 trong sáu biến thể giữ fix/ngoại suy, nhiễu mỗi trục 0/5/15 m. Giữ fix không nhiễu đạt **89,35%**, so oracle L30 **92,69%**. | Đủ 24 family × 3 draw, 18.114 event; Q, neo, ngân sách và traffic theo từng độ sâu giữ nguyên. Cohort đã xem, chỉ thay vị trí ở local ranking. |
| Kết quả âm của ngoại suy | Hai fix giảm sai số vị trí trung bình nhưng Recall4 và Recall3 đều thấp hơn giữ fix ở mọi mức nhiễu và cả hai độ sâu. | Chưa chọn ngoại suy cho mô hình chính. Sai số tọa độ thấp hơn chưa bảo đảm đáp án POI tốt hơn. |
| S8 đồng hành lịch sử | Target-only và joint protected cùng MAE **664,27 m**, Hit100 **3,70%**. Raw đồng hành: **594,33 m / 1,85%**; không tốt hơn ở mọi metric. | Ba test family, năm cặp, 41 pair-event nhưng 24 event mục tiêu duy nhất. RNG lịch sử có seed tái tạo; bank chưa khai thác inversion. Không xác nhận S8 hiện tại hoặc group privacy. |
| Tài nguyên | NPZ L60 nén **1,71 MB**; signature mở **95,31 MB**, access **0,26 MB**. L10 materialized **15,89 MB**; cache mặc định 256 vector nếu đầy **135,56 MB**. | Payload/capacity suy ra từ header và cấu trúc dữ liệu, chưa đo peak RAM. Planner L10 riêng; response prefix L20/L30 giữ allocation L60 trong replay offline. |

Hai đồng hồ GPS được tách rõ: supplier của lớp bảo vệ có pacing/cap riêng;
local sensor diagnostic có 11 fix ảo mỗi phiên. Không cộng chúng thành số
lần GNSS hoặc tiết kiệm pin. Nhiễu Gaussian có kiểm soát chưa được hiệu chuẩn
bằng GPS đo thực; GPS đầu vào Geo-I không bị thay trong phép thử này.
Đích detour vẫn là oracle local đã biết; bảng Recall3 bỏ purpose đó.

Mẫu số utility dùng reference từ GPS thật tại event: rỗng giữ N/A, đáp án
thiết bị rỗng khi reference tồn tại chấm 0. Bảng bán kính giữ cả POI ngoài
miền thật và category-event trả rỗng; counts có lặp qua draw, không là số
người độc lập. Ở L30/σ15, giữ fix trả 16.430/106.488 item ngoài miền thật;
ngoại suy trả 17.359/107.622.

Kết quả primary trước đó vẫn là **89,714% → 92,688%**, gain 2,974 pp,
CI95% [2,308; 3,646] pp và reply JSON tăng 31,102%, trên tuyến mới cùng bản
đồ. Các bổ sung 07/10 là diagnostics trên nguồn đã xem, không biến thành
một lần xác nhận holdout mới hoặc tăng mức bảo đảm riêng tư.

## Nội dung và kiểm tra luận văn

Chương 5 thêm trình tự bảy bước, từ admission/ngân sách trước GPS đến bảo vệ,
ước lượng, gửi Q và trả lời local. Một phép thử reuse mới vẫn cập nhật belief
khi giữ Z; event không đọc chỉ dự đoán. Chương 6 thêm ba bảng GPS local, một
bảng S8 và bảng tài nguyên; hình primary được đặt trước section GPS để luồng
đọc liên tục. Abstract, cohort registry, ma trận S1–S10 và kết luận đồng bộ.

- Kiểm thử hoàn tất exit 0: **988 passed, 9 skipped**, trên 997 test đã
  collect. Default `-q` kép không in terminal summary; [run record](../../artifacts/reports/thesis_review_20261007/tests_20261007.log)
  và [collection log](../../artifacts/reports/thesis_review_20261007/tests_collection_20261007.log)
  ghi cách đối chiếu. Không lưu elapsed time nên không nhận thời gian chạy.
- Local-GPS [checker độc lập](../../artifacts/benchmarks/local_gps_robustness_20261007_v1/validation.json)
  PASS đủ 72 block, 126.798 estimate và 253.596 utility rows; khôi phục sáu
  metric/mẫu số baseline trước score mới. Checker/protocol khóa trước scoring.
- S8 [checker độc lập](../../artifacts/benchmarks/s8_companion_inference_20261007_v1/validation.json)
  PASS ancestry, causal prefix, features, fit/selection và family arithmetic;
  [dedup check](../../artifacts/benchmarks/s8_companion_inference_20261007_v1/descriptive_counts_validation.json)
  kiểm tra 41/24 cho mọi view. Chia train/selection/test theo family;
  không dùng test chọn lại attacker.
- Exporter cũ PASS **29 nguồn / 8 TeX**; exporter mới PASS **10 nguồn / 3 TeX**.
  Các script chỉ định dạng số đã lưu, không fit, rescore hoặc bootstrap mới.
- Tectonic 0.17.0 build exit 0, không có overfull, undefined reference/citation
  hoặc lỗi. Còn underfull ở vài paragraph/path; render không cắt hay đè chữ.
  Đã render 74 trang, xem chín contact sheet và các trang chi tiết; 23 font
  đều nhúng, 307 link không có đích nội bộ ngoài PDF.
- [Review thuật toán/tài nguyên](2026-10-07_thesis_method_resource_review.md),
  [review PDF phương pháp](2026-10-07_thesis_pdf_method_review.md) và
  [QA sensor](2026-10-07_thesis_sensor_review.md) giữ đúng hash/phạm vi.
- [Receipt ranh giới artifact](2026-10-07_thesis_public_boundary.json) và
  [scope](2026-10-07_thesis_public_boundary.md) ghi inventory và các encoding
  được scan. Phép scan Git-new không bao gồm tracked modifications/ignored
  files và không là chứng nhận phát hiện mọi dạng bí mật.

[Manifest bản dựng](../../artifacts/reports/thesis_review_20261007/manifest.json)
giữ 22 nguồn TeX đang import, snapshot, certificates, exporter và log.
SHA256 PDF đã xem:
`8b36dae14084ce0a762488f762e2a2417bfd7bbb2b24b1ea0e03271311ff54b7`.
PDF/source đúng byte của review 06/10 được giữ tại
[snapshot_location.json](../../artifacts/reports/thesis_review_20261006/snapshot_location.json);
review cũ không bị sửa để khớp kết quả mới. Xem [hướng dựng lại](../../thesis/README.md).
Font/TeX phụ thuộc môi trường; không hứa PDF bit-identical trên mọi máy.
Workflow PDF dùng Tectonic/PyMuPDF; Node cho operation-marker helper không
có trong môi trường, như đã ghi ở review trước.

## Phạm vi tiếp tục

Phương pháp và evidence hiện là nghiên cứu SUMO cùng một bản đồ. Chưa có
GPS người dùng thực, deployment hoặc xác nhận cấu hình hiện tại bảo vệ đủ
S1–S10. Đối chứng tải catalogue đầy đủ vẫn mạnh hơn trong API cho phép bulk;
đích thật và provider world tổng hợp còn là các giả định quan trọng. Bước
tiếp theo cần dữ liệu sensor thực/hiệu chuẩn, attacker biết cơ chế, cap chặt
hơn và xác nhận trên cohort chưa xem cho bất kỳ thay đổi model được chọn.
Các gate novelty/fidelity/dữ liệu thực cho bài JISA vẫn theo kế hoạch riêng.
