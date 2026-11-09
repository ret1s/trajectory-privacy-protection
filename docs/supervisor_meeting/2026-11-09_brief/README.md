# Bổ sung benchmark S4–S8 — 09/11/2026

[PDF](report_explained.pdf) · [LaTeX](report_explained.tex) ·
[Nguồn số liệu](report_sources.json) · [QA receipt](report_manifest.json)

Phụ bản cho [báo cáo 28/10](../2026-10-28_brief/README.md); bản cũ, core GeoI-BQ
v1 và mọi GPS/Q/REM/comparator outputs được giữ nguyên. PDF gồm mục lục và tám
trang nội dung/phụ lục, tổng cộng chín trang. Sáu arm được giữ ở mọi phép đo.

## Kết quả và phạm vi

| Scenario | Đo gì | GeoI-BQ | Kết luận có thể dùng |
|---|---|---|---|
| S4 | Cùng route family qua hai phiên khác nhau | BA47,50%, CI95%[40,00;54,17]% | Association proxy; không phải person/vehicle re-ID |
| S5 | Cạnh tương lai trong hai ứng viên | Turn accuracy50,83% | Raw50% vì cadence bỏ lỡ turn: chưa chứng minh đóng góp bảo vệ |
| S6 | Đích tương lai, hai lựa chọn công khai, sau rẽ | Accuracy/Hit10050,83%, CI[44,17;57,50]% | Raw100%; có tín hiệu task, nhưng không là unrestricted destination prediction |
| S7 | Đổi purpose/category trong khi giữ cùng wire context | Equal-wire100%; purpose25%, category16,67%, joint4,17% | Conditional payload identifiability, cùng wrapper ở cả sáu arm |
| S8 | Hai phiên có shared prefix, căn clock tương đối | Hit1000% ở target/protected partner; raw partner100% | Diagnostic correlated-channel, không có native simultaneous companionship |

S4 identity người/xe thật, S6 lịch sử sáu chuyến đã bảo vệ, S7 natural intent và
S8 real companionship vẫn N/A với lý do cụ thể. Không đổi tên route-family ID
thành person/vehicle identity. Điểm dưới chance hoặc CI[0;0] không chứng minh
anonymity hay xác suất tấn công bằng0.

## Đọc report

- Trang i: mục lục có liên kết.
- Trang1: split/protocol, S4 và lý do giới hạn identity.
- Trang2: S5/S6, các cutoff và positive controls.
- Trang3: S7 payload audit, S8 joint observation.
- Trang4: vì sao bổ sung attacker fusion sau V1, V1→V2 và model được chọn.
- Trang5–8: mọi phase/CI/Brier/log loss, S8 paired deltas, số tọa độ quan sát và reproduction.

## Hai vòng bằng chứng được giữ riêng

1. [`remaining_scenarios_20261109_v1`](../../../artifacts/benchmarks/remaining_scenarios_20261109_v1/):
   66 task, source/config chốt trước scoring, fit TRAIN/chọn SELECTION/đánh giá TEST.
2. [`s8_fusion_20261109_v2`](../../../artifacts/benchmarks/s8_fusion_20261109_v2/):
   24 S8 tasks với attacker bổ sung sau khi thấy V1 thiếu phép fusion đơn giản.
   Đây là **post-V1 diagnostic trên cohort đã xem**. Không tạo defense outputs mới.

Fusion làm Semantic* Hit100 tăng45→75%, Fake-query*16,67→68,33% trong view hai
kênh protected. Không bỏ V1 hoặc chọn attacker theo TEST. V2 vẫn lưu mọi
SELECTION/TEST estimate để replay selector. Selected Hit và MAE có thể từ hai
estimators khác nhau; strong bank không bảo đảm mọi TEST metric tăng.

Verifier V2 ban đầu có một false gate: yêu cầu selected Raw-partner estimator
phải đạt100% trên TEST. Một fusion estimator tied ở SELECTION đạt98,33% với
TransProtect* trên TEST, trong khi estimator `partner_centroid` có sẵn vẫn
đạt100%. Failure được giữ trong `verification_failures/initial_gate.json`;
checker độc lập xác nhận available control và selected score riêng. Không sửa
predictions hoặc điểm số để qua gate.

## Tái lập từ repo root

```sh
python -m pip install -r docs/supervisor_meeting/2026-11-09_brief/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.verify_remaining_scenarios_20261109
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.check_s8_fusion_20261109
python -m pytest -q tests/test_remaining_scenarios_20261109.py tests/test_correlated_channel_fusion.py
```

Không chạy lại `freeze/run` vào path đã tồn tại hoặc frozen V2 verifier với gate
sai. Verifier dùng `--write` chỉ khi tạo receipt lần đầu; các lệnh trên chỉ đọc.

Từ thư mục report:

```sh
python build_report.py --check
tectonic --keep-logs report_explained.tex
python validate_report.py
```

Render/xem PDF trước `python validate_report.py --visual-reviewed`.
Protocol chi tiết: [S4–S8](../../research/2026-11-09_remaining_scenarios_protocol.md)
và [S8 fusion](../../research/2026-11-09_s8_fusion_protocol.md).
