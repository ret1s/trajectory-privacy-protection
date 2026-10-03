# Báo cáo ngày 26/09/2026

**S10 gồm A: một chuyến; B: nhiều chuyến lặp cùng đích.**
Có 29 ca trong khung S1–S10 và 14 ca trong năm scenario ưu tiên.
Nhãn được dùng thống nhất trong report, bảng số liệu và bản đồ.

- [Report PDF](report_explained.pdf), [LaTeX](report_explained.tex), [HTML](report_explained.html).
- [Bản đồ tương tác: 29 ca](sample_maps.html), [atlas PDF](sample_maps.pdf).
- [Bản chuẩn bị trình bày](preparation_guide.pdf) và [hướng dẫn hai tài liệu](PREPARATION.md).
- [Kết quả và lập luận hiện tại](../../research/active_scope_results.md).

**Sắp xếp lại ngày 02/10/2026 cho buổi 03/10.** Số liệu, dataset, model, attacker và transcript giữ
nguyên bản 30/09; chỉ thay thứ tự và phần dẫn nhập. Báo cáo 31 trang: 1–2 tổng quan; 3–8 Phần I
kiến trúc và kết quả (mục 2–4); 9–14 Phần II lập luận và bộ đo chung (mục 5–7); 15–25 Phần III
dataset và 29 ca (mục 8–9); 26–31 phụ lục khảo sát, đối chứng, metrics gốc, nguồn (mục 10–13).
Hai mục mới là tổng quan (mục 1) và bốn lớp bằng chứng (mục 5); các mục còn lại giữ nguyên nội dung,
chỉ đánh số lại tham chiếu chéo. Bản 30/09 lưu ở `archive/before_2026-10-03_restructure/`.

## Phạm vi số liệu

| Bộ | Record trong phạm vi / kho gốc | Năm scenario ưu tiên |
|---|---:|---|
| Minh họa, 12 nhóm/264 chuyến | 389 / 393 | 29 mẫu cho toàn bộ khung |
| Phát triển, 12 nhóm/264 chuyến | 407 / 415 | 165 record từ 94 chuyến |
| Bốn nhóm, 88 chuyến | 129 / 131 | 54 record từ 30 chuyến |
| Endpoint, 32 nhóm/704 chuyến | 1.093 / 1.118 | Utility 435 record từ 246 chuyến; endpoint privacy 150 record |

Dataset, model, attacker selection và transcript gốc không bị sửa. Readout mới
ở `artifacts/benchmarks/active_scope_ac_v2/`. Gộp đều ca trong scenario rồi đều
scenario: S10 chia hai, các scenario khác chia ba. Chi phí chỉ giữ các chuyến
thuộc phạm vi, nhưng tính toàn bộ clock và traffic gốc của từng chuyến.

## Kết luận hiện tại

- Đã có lợi thế privacy thực nghiệm ở S1/S2/S3/S9 và S10.A/B so với năm adapter
  trực tuyến. S1–S3 dùng bốn nhóm và kế hoạch mỗi sự kiện; S9/S10 dùng 32 nhóm
  và kế hoạch theo lịch. Không gán số bốn nhóm cho client theo lịch trên 32 nhóm.
- Hit100 S9 của đối chứng: 19,76–40,81%; S10.A/B: 10,75–26,11%; bản theo lịch
  đạt 0% trong bank đã thử. CI 95% của chênh lệch được tính lại đúng phạm vi,
  dưới 0 cho từng đối chứng; chưa hiệu chỉnh nhiều so sánh.
- Bản 30 theo lịch: Recall 95,00% ở p=0,8, 14/14 ca ≥90%; stress p=0,95:
  93,01%, 13/14 ca. Bản 67 đạt 100%, 14/14 ca ở các mức đã thử.
- Byte/sự kiện khoảng 3.071 (30) và 6.879 (67), chưa gồm HTTP/TLS. Lịch công
  khai dùng khoảng 4,48 lần byte so với cùng kế hoạch chỉ refresh khi hoạt động.
- Đóng góp được hỗ trợ là thiết kế phối hợp phủ POI, xếp hạng GPS tại thiết bị
  và lịch công khai, cùng kiểm chứng privacy–utility–chi phí. Chưa xác lập độ
  mới toàn diện, vượt sáu paper nguyên bản hoặc ưu thế ở cùng ngân sách.

TransProtect dùng Markov thay Transformer; Semantic dùng predictor thực nghiệm
thay LSTM. AnotherMe là tham chiếu VTGA offline, có lỗi/thiếu ca và mẫu số khác.
Bảo đảm phụ thuộc vùng/lịch đăng ký trước; không bao gồm IP/account/click, lỗi
mạng, đổi vùng theo GPS hoặc bật/tắt dịch vụ theo chuyến. Recall 100% trên mẫu
không là chứng nhận toàn bản đồ: bản 67 còn tám POI ngoài độ phủ tĩnh.

## Dựng lại

```sh
python -m experiments.endpoint_dataset_archive unpack
python -m experiments.reaggregate_active_scope
python -m experiments.verify_active_scope
python -m experiments.summarize_active_scope
python docs/supervisor_meeting/2026-09-26_brief/plot_sample_maps.py
python docs/supervisor_meeting/2026-09-26_brief/plot_case_panels.py
python docs/supervisor_meeting/2026-09-26_brief/build_report.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/report_explained.tex
```

`build_report.py` dựng các mục rồi sắp xếp lại theo thứ tự trình bày và đánh số lại tham chiếu "mục N";
phần nội dung nằm trong các module `*_content.py`. `report_case_labels.py` ánh xạ nhãn trình bày sang mã nguồn thực nghiệm;
`evaluation/report_scope.py` giữ phạm vi tính toán đã khóa. `scenario_guide.json`,
`case_readings.json`, `printed_samples.json`, `data_samples.json` và
`scenario_inventory.csv` chỉ chứa mẫu hiện tại. Mã bản ghi và số liệu gốc được bảo toàn;
`method_evidence.json` lưu ánh xạ nhãn để truy nguyên.
Không sửa trực tiếp LaTeX/HTML được sinh tự động.

Đã đối chiếu 981 phép tính/mẫu số với dữ liệu đo gốc; bốn tests cho trọng số,
thiếu dữ liệu, chi phí và bootstrap đã qua. `report_validation.json` ghi QA bản
PDF hiện tại; `method_evidence.json` ghi hash nguồn. Chưa chạy lại toàn bộ model
hoặc tạo cohort mới trong lần cập nhật nhãn này và lần sắp xếp lại ngày 02/10.
