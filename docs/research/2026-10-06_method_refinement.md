# Tiếp tục hoàn thiện Geo-I và chuẩn bị buổi GVHD tiếp theo

**Giữ nguyên backbone Geo-I.** Vòng 06/10 thay chính sách cache POI cố định và
cách chọn attacker endpoint; không sửa GPS, nhãn, Q hoặc kết quả lịch sử để
làm mô hình thắng. [Report mới](../supervisor_meeting/2026-10-06_brief/report_explained.pdf)
và [script trình bày](../supervisor_meeting/2026-10-06_brief/preparation_guide.pdf)
được chuẩn bị 06/10, chưa gán ngày cho buổi gặp.

## Utility: phân biệt metadata cố định với trạng thái hiện tại

Cache 60s tăng Recall ít và vẫn bỏ lỡ POI ở những chuyến yếu. Tọa độ/category
của POI trong danh mục cố định không cần hết hiệu lực giống availability.
Candidate mới giữ metadata **đã nhận từ các request trước đó**, trong cùng
epoch công khai và phiên bản danh mục; không đọc metadata tương lai hoặc gửi
thêm request theo purpose thật. Hết epoch/đổi phiên bản vô hiệu cache.

Trên cohort native phát triển, GeoI-Epoch8-H12 L20: Recall trung bình
**94,96% → 99,34%**, đoạn 400–600s **94,39% → 99,48%**. Chuyến thấp nhất ở
đoạn cuối vẫn **86,67%**; min toàn cửa sổ **77,74%**. Lợi ích lớn đến từ các
chuyến đã có lịch sử trong danh mục cố định, không phải bảo đảm mọi chuyến.
Các phương án L20 giữ nguyên 7.550 Q requests và 41.233.562 reply JSON bytes.
Availability realtime chưa đánh giá; thiếu trạng thái mới nghĩa chưa biết.

[Module, giao thức, selection, kiểm tra causal và giới hạn](2026-10-06_native_static_utility.md).
Không chuyển những điểm utility này thành điểm privacy S5/S6 mới.

## Endpoint: sửa cách đo trước khi kết luận lợi thế

Selector cũ chỉ dùng hai nhóm chọn nên dễ chọn một attacker có test MAE tệ.
Rule mới chọn qua tám fold giữ ngoài từng nhóm train, cộng hai nhóm selection,
dùng mean và standard error đã khóa trước khi chấm lại dữ liệu test đã xem.
Nó giữ nguyên Q/GPS/traffic/phương pháp bảo vệ.

MAE S10 của attacker trên GeoI-Slack20 giảm **901m → 758m** so với bank cố
định; Endpoint20 giữ **1.337m**. Chênh cùng L20 là **579m**, bootstrap theo
28 nhóm: **[468; 696]m**. Nhưng Hit chưa cải thiện đều: Slack L10 Hit500 S10
giảm 30,36% xuống 14,29%. Endpoint20 Recall 96,64%; 6/112 runs dưới 90%.
Không chọn lại theo test hay tuyên bố bank mới mạnh hơn ở mọi metric.

[Evidence và kiểm tra độc lập](2026-10-06_robust_endpoint_selection.md) giữ
cả verifier ban đầu gặp lỗi biểu diễn cache và verifier sửa cách kiểm tra;
không đổi dự đoán để vượt kiểm tra. Đây là diagnostic/development,
không là holdout mới.

## Report và ưu tiên nghiên cứu

Related works trong ba năm có một bảng output, coverage Fully/Partially và
metrics gốc; coverage là đối chiếu theo giả định từng paper. Bảng đối chứng
paper-v2 giữ nguyên. EIE point estimate trùng MAE; Δc/ASR/entropy/DER thiếu
input giữ N/A. Native road-cost diagnostic thuộc cohort riêng và đo méo
utility, không phải bảo vệ tốt hơn. [Audit nguồn/metrics](2026-10-06_report_evidence.md).

Report vẽ bốn layer có vùng model, input/output và mũi tên Z→Q→POI→P;
giới thiệu attacker trước kết quả. Không dùng nhãn sample hoặc phụ lục.
S7 vẫn có bốn purpose local; S4–S6 có cap qua nhiều phiên và attacker prefix,
chưa che account/IP hoặc mọi liên kết lịch sử. S8 chưa có đánh giá mới.

Tiếp theo cần khóa candidate/selector rồi xác nhận trên nhóm/thành phố chưa
dùng để chỉnh, ưu tiên chuyến đầu/chuyến dài; mở rộng attacker liên kết lịch
sử và bổ sung metrics gốc đúng output contract. Không đổi core Geo-I.

## Kiểm chứng

Hai entry point `python -m tests.run_all -o addopts='' -q` và
`python -m pytest -q tests -o addopts=''` đều **521 passed, 9 skipped**;
chín integration test cần cache SUMO gốc chưa có. `pip check`, compile và
whitespace đều qua. Hai verifier utility cache và verifier endpoint tính
lại các readout trong phạm vi ghi tại từng artifact.

Review độc lập phát hiện `answer_live` chưa kiểm tra epoch ngắn của adapter;
đã sửa bằng phép kiểm tra read-only dùng chung với static answer. Regression
chặn trước start/đúng end/sau end, giữ nguyên GPS, cap, cache clock và request
tương lai. Không sửa client mặc định hoặc Q đã đóng băng.

Report giữ source pins và có [review bản xuất](../supervisor_meeting/2026-10-06_brief/review.json)
riêng. PDF/SVG được render để xem bố cục; môi trường không bật browser CUA,
nên HTML chỉ kiểm tra cấu trúc, link và nội dung cùng nguồn với PDF.
