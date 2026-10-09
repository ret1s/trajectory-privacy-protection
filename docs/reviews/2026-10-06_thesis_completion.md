# Luận văn Geo-I: bản tích hợp và kiểm tra ngày 06/10/2026

Bản hiện tại là một luận văn bảy chương, **67 trang**, có thể gửi GVHD để đọc
và phản biện. [PDF canonical](../../artifacts/reports/graduation_thesis.pdf)
được dựng từ [main.tex](../../thesis/main.tex), không có một bản thesis cuối
khác song song. Việc đánh giá đủ yêu cầu tốt nghiệp vẫn thuộc GVHD/hội đồng;
bản này không nhận đã đạt yêu cầu công bố JISA.

## Nội dung đã hoàn thiện

Chương 1 đặt ba câu hỏi nghiên cứu về ranh giới bảo vệ, utility/cost và độ ổn
định. Chương 2 mô tả chuyển động và xử lý nhân quả; Chương 3 nối S1–S10 với
đúng cohort và nguồn SUMO. Chương 4 tách output contract, metric gốc và metric
chung để không chấm một tập Q giả như tập có thành viên thật.

Chương 5 trình bày Geo-I/REM, phép thử tái sử dụng có nhiễu, cap nhiều phiên,
ước lượng từ lịch sử đã bảo vệ, Q khả thi theo đường và bốn mục đích xếp hạng
cục bộ. Hình 5.1 tách GPS thật, Z nội bộ, Q công khai, server và đầu ra POI;
ba đầu vào nằm ngoài khung phương pháp. H=12, K=5, L=30 và slack có định
nghĩa riêng; L30 không phải mã cấu hình 30/67 trước đây. Phân tích kernel lý
tưởng được tách khỏi float/PRNG của mã chạy.

Chương 6 đưa từng kết quả vào đúng protocol: S1–S3 lịch sử; pilot native
S5/S6; S4/S7; endpoint development; hai planner không qua gate; xác nhận L30;
static bulk và chẩn đoán trạng thái POI thay đổi. Chương 7 trả lời các câu hỏi
nghiên cứu và giới hạn kết luận. Các module đánh giá cũ, số liệu đã khóa và
thử nghiệm thất bại còn nguyên, nhưng không còn là chapter đang được import.

## Kết quả được giữ và cách diễn giải

| Bằng chứng | Kết quả | Phạm vi |
|---|---|---|
| Xác nhận L30 đã chọn trước | Recall@5 current-only 89,714% → 92,688%; +2,974 pp, CI95% [2,308; 3,646]; reply JSON +31,102% | 24 nhóm tuyến mới cùng bản đồ × 3 draw/nhóm × 8 phiên; giữ Q, neo, lịch và ngân sách. Đây là đánh đổi utility/cost có xác nhận. |
| Trạng thái POI tổng hợp | L20 current 87,85%; L30 current 91,44%; L30 cache còn hiệu lực 91,78% | 72 stream đã xem, một thế giới khả dụng, 88 epoch tuyệt đối. Cache không thêm request/byte; không phải xác nhận mới hoặc kết quả riêng tư mới. |
| Control đầy đủ | Static bulk và full-current bulk đạt 100% trên reference không rỗng; dynamic bulk ít byte hơn L30 | Nếu API cho phép tải đầy đủ thì control vẫn tốt hơn. Trạng thái động chưa giải quyết application gap của catalogue nhỏ. |
| Phạm vi scenario | Các tác vụ và giới hạn từng S được ghi riêng; S8 chưa có đánh giá đầy đủ | Không cộng kết quả nhiều cấu hình/cohort thành một model đã bảo vệ đủ S1–S10. |

Metric Recall chỉ được tính khi reference tồn tại; tham chiếu rỗng giữ N/A
và mẫu số riêng. CI ghép theo family giữ các draw bên trong, không coi tick,
POI hoặc bốn purpose là các subject độc lập. Byte là compact JSON của ứng
dụng, chưa đo HTTP/TLS, latency hoặc năng lượng thiết bị.

Chẩn đoán động phân biệt POI chưa truy hồi, trạng thái hiện tại chưa biết và
đã biết không khả dụng. Các arm vận hành không trả POI unavailable/unknown.
Control cố dùng bit hết hạn được đánh dấu không hợp lệ. Seed/fixture công
khai cho tái lập có thể giúp client tính toàn bộ trạng thái nếu được cấp;
thí nghiệm chỉ kiểm tra freshness/dataflow dưới API giả định.

## Kiểm tra đã hoàn thành

- **Kiểm thử:** `python -m tests.run_all -o addopts='' -q` với giới hạn luồng
  BLAS/OMP: **953 passed, 9 skipped**, 17,63 s. [Log](../../artifacts/reports/thesis_review_20261006/tests_20261006.log).
- **Bảng số:** exporter `--check` PASS **29 nguồn JSON / 8 fragment TeX**;
  không fit, scoring hoặc bootstrap mới. [Manifest nguồn](../../thesis/current_evaluation_generated/sources.json).
- **Audit động độc lập:** PASS đủ 72 block, 18.114 event, 163.026 dòng arm/event;
  tái dựng availability, đường có hướng, cache, mẫu số và cost. [Certificate](../../artifacts/benchmarks/dynamic_provider_status_20261006_v1/validation.json).
  V1 dừng do thứ tự serialization; giữ failure/source/declaration. V2 sửa
  ghép khóa event/arm sau scoring và khai báo đúng thời điểm; runner/world/
  readout không đổi. Không trình bày sửa checker này như preregistration mới.
- **PDF:** Tectonic 0.17.0 dựng thành công; không có overfull, citation/reference
  undefined hoặc lỗi. Còn một số underfull do đường dẫn/bibliography nhưng
  không cắt chữ. Đã render toàn bộ 67 trang, xem tám contact sheet và kiểm tra
  riêng sơ đồ, biểu đồ, bảng động, tóm tắt và kết luận. Mọi font dùng đều nhúng;
  290 link PDF không có đích nội bộ ngoài trang.
- **Review phương pháp độc lập:** PASS toàn bộ trang PDF 39–48, với sơ đồ ở
  trang PDF 40. [Ghi chú kiểm tra](2026-10-06_thesis_method_review.md).
- **Ranh giới artifact mới:** scanner bất biến PASS 155 file Git mới và 73
  gzip, không phát hiện key/digest/state. [Receipt](2026-10-06_thesis_public_boundary.json).
  Phạm vi là inventory tại lúc scan, exact raw/hex/base64/SHA256 và gzip;
  không bao gồm tracked modifications, ignored files hoặc mọi dạng bí mật.

[Manifest bản dựng](../../artifacts/reports/thesis_review_20261006/manifest.json)
giữ hash PDF, 15 nguồn LaTeX đang hoạt động, export/figure, certificate và log.
SHA256 PDF đã xem:
`2c739564dcebc6d04999e51153c654a5cd035700a0e44a2c2b361fd2415bac60`.
[Hướng dựng lại](../../thesis/README.md) giữ một đầu ra canonical. Font Times
New Roman và TeX bundle phụ thuộc môi trường; không hứa dựng lại bit-identical
trên mọi máy. PDF operation-marker helper đã được gọi một lần nhưng Node
không có trong môi trường; PDF được dựng và kiểm tra hình bằng Tectonic/PyMuPDF.

## Phần còn cần bằng chứng trước khi nâng thành bài báo

Dữ liệu hiện là SUMO trên một bản đồ; chưa có GPS người dùng thực, dịch vụ
live hoặc nhiều vùng. Utility local dùng GPS/đích thật tại từng event như
oracle; supplier pacing 60 s của lớp bảo vệ không là phép đo tiết kiệm GNSS.
Cap epoch 0,23/m cho cận exp(23) ở 100 m còn rất lỏng. Sampler số thực chưa
có chứng chỉ pure Geo-I; S4 metadata/identity thật, intent tương quan tuyến
và S8 còn ngoài bằng chứng hiện tại.

Đóng góp thesis có thể bảo vệ là thiết kế/hiện thực/tái lập hệ thống và kết
quả utility–cost ổn định trong phạm vi đã định. Để phát triển bài JISA cần
thêm đối chiếu toàn văn prior art gần nhất, fidelity của comparator, dữ liệu
thực và lý do cần retrieval từ xa dưới hợp đồng provider cụ thể. Một thay đổi
model sau khi đọc các cohort hiện tại phải được phát triển trên dữ liệu này
rồi xác nhận bằng protocol/cohort mới.
