# Báo cáo ngày 26/09/2026

**Phạm vi hiện tại: S10 chỉ gồm A/C.** Bài toán đoán đích từ tiền tố được xử lý
ở S6.A. Có 29 ca trong khung S1–S10 và 14 ca trong năm scenario ưu tiên.
Quyết định này được đưa ra sau chẩn đoán; giữ mã A/C để truy nguyên dữ liệu.

- [Report PDF](report_explained.pdf), [LaTeX](report_explained.tex), [HTML](report_explained.html).
- [Bản đồ tương tác: 29 ca](sample_maps.html), [atlas PDF](sample_maps.pdf).
- [Kết quả và lập luận hiện tại](../../research/active_scope_results.md).

Báo cáo 27 trang: 1–11 dữ liệu và mẫu; 12–19 khảo sát/metrics; 20–21 kiến trúc;
22–24 đối chứng và mức bằng chứng cho năm scenario; 25 kết quả S9/S10.A/C;
26–27 nguồn và tài liệu tham khảo.

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

- Đã có lợi thế privacy thực nghiệm ở S1/S2/S3/S9 và S10.A/C so với năm adapter
  trực tuyến. S1–S3 dùng bốn nhóm và kế hoạch mỗi sự kiện; S9/S10 dùng 32 nhóm
  và kế hoạch theo lịch. Không gán số bốn nhóm cho client theo lịch trên 32 nhóm.
- Hit100 S9 của đối chứng: 19,76–40,81%; S10.A/C: 10,75–26,11%; bản theo lịch
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
python docs/supervisor_meeting/2026-09-26_brief/build_report.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/report_explained.tex
```

`evaluation/report_scope.py` là danh mục phạm vi dùng chung. `scenario_guide.json`,
`case_readings.json`, `printed_samples.json`, `data_samples.json` và
`scenario_inventory.csv` chỉ chứa mẫu hiện tại. Giữ bản trước thu hẹp trong
`archive/scope_before_ac_v2/`; số gốc và chẩn đoán còn trong các artifact cũ.
Không sửa trực tiếp LaTeX/HTML được sinh tự động.

Đã đối chiếu 981 phép tính/mẫu số với dữ liệu đo gốc; bốn tests cho trọng số,
thiếu dữ liệu, chi phí và bootstrap đã qua. `report_validation.json` ghi QA bản
PDF hiện tại; `method_evidence.json` ghi hash nguồn. Chưa chạy lại toàn bộ model
hoặc tạo cohort mới trong lần cập nhật phạm vi này.
