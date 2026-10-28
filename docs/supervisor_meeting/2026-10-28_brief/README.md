# Bản sơ bộ dạng bài báo — 28/10/2026

[PDF](report_explained.pdf) · [LaTeX](report_explained.tex) ·
[Nguồn paper/coverage](literature_evidence.json) ·
[Source hashes](report_sources.json) · [QA receipt](report_manifest.json)

Bản thảo tiếng Việt theo năm phần **Introduction → Related Work → Proposed
Method → Experiments and Analysis → Summary**, rồi References và bốn phụ lục.
Đây là bản để review/làm cơ sở tự viết lại; chưa thay canonical thesis.

## Thứ tự đọc

- Trang i: mục lục có liên kết tới các phần, tiểu mục, References và phụ lục.
  PDF có tổng cộng 19 trang; nội dung vẫn đánh số 1–18 như dưới đây.
- Trang 1: motivation, vấn đề và đóng góp, scenario overview.
- Trang 2: tám paper/context trong cửa sổ 28/10/2023–28/10/2026;
  output, scenario, metrics và fidelity. Geo-I/predictive là nguồn nền tảng riêng.
- Trang 3–5: hai sơ đồ độc lập (system placement, internal model), công thức,
  giả mã, mock belief và chuyến TEST cố định `cache-029` qua nhiều bước.
- Trang 6–8: bốn mệnh đề privacy, theorem toàn transcript/Bayes và utility.
- Trang 9–13: metrics, dữ liệu, setup và **đối chiếu paper adaptations là chính**.
- Trang 14: Summary và References.
- Trang 15–18: CI đầy đủ, internal ablation lịch sử, reproduction và theorem/code map.

## Kết quả chính và phạm vi

Trial mới: [`paper_comparison_20261028_v1`](../../../artifacts/benchmarks/paper_comparison_20261028_v1/).
Sáu hàng chính: Raw GPS, REM-Z K1, TransProtect **Markov adaptation**, Semantic
correlation **OSM/empirical adaptation**, Fake-query insertion **road/history
adaptation**, và **GeoI-BQ v1**. Dấu `*` trong PDF luôn chỉ adaptation.

Giữ đúng 48 family (20 TRAIN / 8 SELECTION / 20 TEST), hai chuyến/family,
ba draws tại TEST, 120 TEST trip-draw mỗi arm. Attacker fit/chọn trên split
riêng theo loss; cùng cửa sổ thời gian và tính mọi fake packet. CI ghép family
10.000 lần. Đây là controlled comparison trên cohort đã xem, không phải
generalization trên tuyến hoặc thành phố mới.

GeoI-BQ đạt **99,595% session Recall@5**, macro CI95% **[99,224; 99,903]%**.
Utility thấp hơn ba adaptation khoảng 0,4 pp; Hit100 thấp hơn trong bank này,
đổi lại nhiều truy vấn hơn singleton methods. Không kết luận thắng mọi metric,
hoặc đánh bại implementation chính thức chưa tái lập. Metrics gốc không đúng
contract/thiếu đầu vào được giữ N/A. Không đổi core hay lấy lại draws proposed.

Hai lỗi triển khai chưa đóng vẫn nêu rõ: continuous Geo-I float sampler và
full upstream restore chưa được chứng nhận. Fresh uniform bits/trusted ledger
là giả thiết của Q bound; Bayesian bound không đồng nghĩa anonymity trước ID.

## Tái lập

Từ repo root, dùng môi trường Python 3.12 với:

```sh
python -m pip install -r docs/supervisor_meeting/2026-10-28_brief/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.check_paper_comparison_20261028
python -m experiments.verify_geoi_bq_mathematics_20261014 --check
```

Checker mới replay frozen verifier ở chế độ **read-only**, so receipt đã lưu;
không generate defense, fit attacker hoặc ghi đè artifact. Verifier write-once
gốc chỉ tạo receipt lần đầu, không dùng để ghi đè lần hai.

Trong thư mục report:

```sh
python build_report.py --check
# Chỉ khi muốn xuất lại bảng/hình từ đúng nguồn:
python build_report.py
tectonic --keep-logs report_explained.tex
python validate_report.py
```

Render PDF bằng Poppler và kiểm tra từng trang trước khi chạy
`python validate_report.py --visual-reviewed`. Tables/figure được xuất từ JSON,
không ghi số bằng tay; `--check` không score hay sample. Có 14 bảng, ba hình
(hai TikZ architecture và một PDF scientific chart). Render QA là scratch trong
`build/paper_report_20261028/`, không là một phiên bản PDF cạnh tranh.

Metadata cũ `response_l=30` không khớp callback top60; replay mới chuẩn hóa
request thành L60 cho mọi arm. Q/time, POI masks, utility và byte lengths giữ
nguyên; discrepancy được ghi trong protocol/PDF. Trial cũ giữ nguyên.
