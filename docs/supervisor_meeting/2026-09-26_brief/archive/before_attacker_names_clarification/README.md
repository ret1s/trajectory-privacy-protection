# Báo cáo trình bày: Geo-I là phương pháp chính

Hai tài liệu đi theo mạch: related works trong ba năm và bộ metrics → kiến trúc BR-Dummy trên nền Geo-I → benchmark và lập luận đóng góp → phụ lục 14 sample.

**Bản rút gọn:** report 11 trang (5 trang chính + 5 trang sample + 1 trang nguồn), guide 10 trang. Trước bảng 3.1 có giới thiệu attacker theo scenario. Mỗi ô so sánh ghi Hit100 (%) / MAE (m) / Recall@5 (%), kèm chú giải và ví dụ. Bảng 14 ca giữ cột riêng. Kết quả/model/dataset không thay đổi. Bản trước lần thêm attacker lưu ở archive/before_attacker_introduction/.


- [Report PDF](report_explained.pdf), [HTML](report_explained.html), [LaTeX](report_explained.tex).
- [Preparation guide PDF](preparation_guide.pdf), [LaTeX](preparation_guide.tex).
- [Cách dùng hai tài liệu](PREPARATION.md); [bản đồ 29 ca](sample_maps.html).

Sơ đồ chia input, các layer, output công khai/riêng và đóng khung mô hình. Neo Geo-I, tái dùng/ngân sách/pacing, belief, mạng làn và bộ chọn POI tạo lõi. Cache phản hồi và xếp hạng GPS tại thiết bị phục vụ top-5 POI. BR-Boundary là mở rộng S9/S10 đã kiểm tra tích hợp, chưa có benchmark kết hợp toàn bộ dịch vụ.

**GeoI-Paced** và **GeoI-Slack** là tên trình bày cho hai cấu hình Geo-I hiện tại. Cả hai K=5, L=10, cận phiên 0,23 /m, đọc GPS cách ít nhất 60 giây và cache trong epoch 60 giây. Bản Slack thêm slack 0,03 trong objective chọn dummy, không phải đổi ε hoặc một cận giảm Recall.

## Phạm vi bằng chứng

| Phép thử | Phạm vi | Kết luận được hỗ trợ |
|---|---|---|
| Geo-I / BR v2 so trực tiếp với adapter | 12 chuyến test, 3 seed, K=5; 60 cửa sổ có tương quan | BR có Hit100 S2 22,22%, S3 7,44%, thấp hơn DLS/TransProtect/Semantic thích nghi; có đánh đổi utility |
| Geo-I hiện tại | 12 nhóm phát triển, 165 record từ 94 chuyến; 14 ca | GeoI-Slack có Recall@5 95,44%, 13/14 ca ≥90%, khoảng 994,1 byte/sự kiện ở p=0,8 |
| Bỏ/thêm component | Bootstrap ghép cặp 12 nhóm, 10.000 lần | Cache tăng 0,24 điểm % với CI không chứa 0; slack tăng 1,06 điểm % nhưng CI chứa 0 |

Hai phiên bản không được gộp thành một bảng xếp hạng. Adapter v2 không phải tái lập paper nguyên bản, cũng không phải adapter mới dùng ở phép thử khác. Chưa có so trực tiếp Geo-I với RDG/Fake-query cùng protocol; AnotherMe v2 chỉ hoàn thành 11/12 ca S3 nên có mẫu số riêng. BR v2 chưa đạt Recall 90% ở S3/S10; bản mới chưa đạt ở S1.C. S9.C có Hit100=0 cả với GPS thật, và Geo-I còn Hit500=58,33%.

Geo-I, tái dùng neo và dummy là nền kế thừa. Lập luận đóng góp tập trung vào phối hợp lịch sử đã bảo vệ, ngân sách hữu hạn, miền đi được trên mạng làn và mục tiêu phủ POI, kèm kiểm tra privacy–utility–chi phí. Chưa chứng minh vượt sáu paper nguyên bản hoặc tốt hơn ở cùng ngân sách.

Bảng related works dùng Fully/Partially theo nhiệm vụ trong giả định paper; CX khi thiếu bằng chứng nguồn. Đây không phải chứng nhận vượt mọi ca A/B/C. Cửa sổ khảo sát giữ 21/09/2023–21/09/2026.

Phụ lục dùng bộ minh họa 12 nhóm/264 chuyến, khác bộ số liệu phát triển. S10 chỉ có A: một chuyến; B: nhiều chuyến cùng đích. B ánh xạ mã nguồn cũ S10.C. Không có C độc lập trong phạm vi hiện tại.

## Nguồn và dựng lại

Không sửa dataset/model/transcript/benchmark gốc. Số tổng và CI bản mới được gộp lại từ kết quả gốc: đều ca trong scenario, đều năm scenario. Chi phí tính toàn bộ traffic của 94 chuyến được giữ, theo sự kiện, request + response JSON; chưa gồm HTTP/TLS/latency.

`geoi_evidence.py` kiểm tra hash nguồn và tính tổng/CI; `geoi_content.py` viết cấu hình, benchmark và đóng góp; `concise_presentation_content.py` viết mạch report. Metrics và kiến trúc dùng chung qua `metrics_explained.tex` và `model_architecture.tex`. Bảng/số dùng chung qua `geoi_configuration.tex`, `geoi_benchmark.tex`, `related_work_coverage.tex` và `scenario_appendix.tex`. Không chỉnh các tệp sinh tự động.

```sh
python docs/supervisor_meeting/2026-09-26_brief/plot_model_architecture.py
python docs/supervisor_meeting/2026-09-26_brief/build_report.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/report_explained.tex
python docs/supervisor_meeting/2026-09-26_brief/prepare_guide_evidence.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/preparation_guide.tex
python docs/supervisor_meeting/2026-09-26_brief/validate_documents.py
```

Python cần NumPy, matplotlib và PyMuPDF cho tổng hợp, hình và kiểm tra PDF. Thông tin kiểm tra nằm ở `method_evidence.json`, `preparation_evidence.json`, `report_validation.json` và `preparation_validation.json`. Có thể truyền `--log-dir` cho validator nếu build PDF ở thư mục khác. Font fallback của môi trường được lưu trong validation.

Bản trình bày trước khi sửa trọng tâm và các bộ dựng cũ được lưu ở `archive/before_geoi_main_restore/` để truy nguyên; chúng không được dùng trong hai tài liệu hiện tại.
