# Báo cáo trình bày: Geo-I là phương pháp chính

Hai tài liệu đi theo mạch: related works trong ba năm và bộ metrics → kiến trúc BR-Dummy trên nền Geo-I → attacker, benchmark và lập luận đóng góp. Tạm bỏ phụ lục mẫu khỏi bản trình bày.

**Bản rút gọn:** report 6 trang (5 trang chính + 1 trang nguồn), guide 5 trang. Chỉ trình bày mức scenario S1–S10. Benchmark hiện có S1, S2, S3, S9, S10; S4–S8 chưa được đánh giá. Bảng hiện tại gộp đều các điều kiện trong từng scenario; trung bình tổng vẫn gộp đều năm scenario. Trang 2 là sơ đồ luồng, trang 3 là component/cấu hình, trang 4–5 là benchmark. Số đo gốc, model và dataset giữ nguyên. Bản đầy đủ trước khi rút gọn được lưu ở archive/before_scenario_only_presentation/.


- [Report PDF](report_explained.pdf), [HTML](report_explained.html), [LaTeX](report_explained.tex).
- [Preparation guide PDF](preparation_guide.pdf), [LaTeX](preparation_guide.tex).
- [Cách dùng hai tài liệu](PREPARATION.md).

Sơ đồ chia input, các layer, output công khai/riêng và đóng khung mô hình. Neo Geo-I, tái dùng/ngân sách/pacing, belief, mạng làn và bộ chọn POI tạo lõi. Cache phản hồi và xếp hạng GPS tại thiết bị phục vụ top-5 POI. BR-Boundary là mở rộng S9/S10 đã kiểm tra tích hợp, chưa có benchmark kết hợp toàn bộ dịch vụ.

Đọc sơ đồ theo **1 → 2 → 3a/3b → 4 → server → 5**: ngân sách/pacing kiểm tra trước neo; không đọc mới thì bỏ bước 2; belief và miền đường cùng vào bộ chọn; phản hồi server vào cache. Boundary S9 đứng trước lõi, S10 đứng trước công bố. Bản dùng trong report: [PDF](figures/architecture_report_flow.pdf), [SVG](figures/architecture_report_flow.svg).

**GeoI-Paced** và **GeoI-Slack** là tên trình bày cho hai cấu hình Geo-I hiện tại. Cả hai K=5, L=10, cận phiên 0,23 /m, đọc GPS cách ít nhất 60 giây và cache trong epoch 60 giây. Bản Slack thêm slack 0,03 trong objective chọn dummy, không phải đổi ε hoặc một cận giảm Recall.

## Phạm vi bằng chứng

| Phép thử | Phạm vi | Kết luận được hỗ trợ |
|---|---|---|
| Geo-I / BR v2 so trực tiếp với adapter | 12 chuyến test, 3 seed, K=5; 60 cửa sổ có tương quan | BR có Hit100 S2 22,22%, S3 7,44%, thấp hơn DLS/TransProtect/Semantic thích nghi; có đánh đổi utility |
| Geo-I hiện tại | 12 nhóm phát triển, 165 record từ 94 chuyến; gộp theo năm scenario | GeoI-Slack có Recall@5 95,44%, 5/5 scenario có Recall trung bình ≥90%, khoảng 994,1 byte/sự kiện ở p=0,8 |
| Bỏ/thêm component | Bootstrap ghép cặp 12 nhóm, 10.000 lần | Cache tăng 0,24 điểm % với CI không chứa 0; slack tăng 1,06 điểm % nhưng CI chứa 0 |

Hai phiên bản không được gộp thành một bảng xếp hạng. Adapter v2 không phải tái lập paper nguyên bản. Chưa có so trực tiếp Geo-I với RDG/Fake-query cùng protocol; AnotherMe v2 chỉ hoàn thành 11/12 chuyến S3. BR v2 chưa đạt Recall 90% ở S3/S10. Bản hiện tại có Recall S1 trung bình 91,24%, nhưng mức thấp nhất vẫn là 80,18%; đạt ngưỡng trung bình không bảo đảm mọi điều kiện đều đạt. S9 có Hit100 trung bình 0,83%, nhưng Hit500 còn 55,56%.

Geo-I, tái dùng neo và dummy là nền kế thừa. Lập luận đóng góp tập trung vào phối hợp lịch sử đã bảo vệ, ngân sách hữu hạn, miền đi được trên mạng làn và mục tiêu phủ POI, kèm kiểm tra privacy–utility–chi phí. Chưa chứng minh vượt sáu paper nguyên bản hoặc tốt hơn ở cùng ngân sách.

Bảng related works dùng Fully/Partially theo nhiệm vụ trong giả định paper; CX khi thiếu bằng chứng nguồn. Đây không phải chứng nhận bảo vệ hoàn toàn. Cửa sổ khảo sát giữ 21/09/2023–21/09/2026. Phụ lục và dữ liệu mẫu được giữ trong bản lưu để có thể khôi phục sau.

## Nguồn và dựng lại

Không sửa dataset/model/transcript/benchmark gốc. Số tổng và CI bản mới được gộp lại từ kết quả gốc: đều ca trong scenario, đều năm scenario. Chi phí tính toàn bộ traffic của 94 chuyến được giữ, theo sự kiện, request + response JSON; chưa gồm HTTP/TLS/latency.

`geoi_evidence.py` kiểm tra hash nguồn, gộp theo scenario và tính tổng/CI; `geoi_content.py` viết cấu hình, benchmark và đóng góp; `concise_presentation_content.py` viết mạch report. Metrics và kiến trúc dùng chung qua `metrics_explained.tex` và `model_architecture.tex`. Bảng/số dùng chung qua `geoi_configuration.tex`, `geoi_benchmark.tex` và `related_work_coverage.tex`. Không chỉnh các tệp sinh tự động.

```sh
python docs/supervisor_meeting/2026-09-26_brief/plot_report_architecture.py
python docs/supervisor_meeting/2026-09-26_brief/build_report.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/report_explained.tex
python docs/supervisor_meeting/2026-09-26_brief/prepare_guide_evidence.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/preparation_guide.tex
python docs/supervisor_meeting/2026-09-26_brief/validate_documents.py
```

Python cần NumPy, matplotlib và PyMuPDF cho tổng hợp, hình và kiểm tra PDF. Thông tin kiểm tra nằm ở `method_evidence.json`, `preparation_evidence.json`, `report_validation.json` và `preparation_validation.json`. Có thể truyền `--log-dir` cho validator nếu build PDF ở thư mục khác. Font fallback của môi trường được lưu trong validation.

Bản trình bày trước khi sửa trọng tâm và các bộ dựng cũ được lưu ở `archive/before_geoi_main_restore/` để truy nguyên; chúng không được dùng trong hai tài liệu hiện tại.
