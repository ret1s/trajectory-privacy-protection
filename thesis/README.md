# Luận văn hợp nhất — revision lập luận 16/11/2026

[main.tex](main.tex) là nguồn LaTeX chính; PDF review nằm tại
[graduation_thesis.pdf](../artifacts/reports/graduation_thesis.pdf).
Đây là bản thảo đầy đủ để học viên/GVHD review và chỉnh sửa, chưa phải bản được
nhà trường phê duyệt hoặc chứng nhận deployment.

## Cấu trúc đã chốt

1. **Introduction:** motivation, transcript/utility/đối chiếu, S1–S10, RQs và đóng góp.
2. **Related Work:** continuous queries; Geo-I/road-aware utility; identity/offline;
   bảng cơ chế, output, coverage, metrics và fidelity.
3. **Proposed Method:** hai kiến trúc system/model, GeoI-BQ v1 cuối, ví dụ nhiều bước,
   bốn mệnh đề privacy, định lý transcript và phân tích utility.
4. **Experiments and Analysis:** bộ đo, SUMO/POI, adaptations/attackers, main comparison,
   supplement S4–S8, utility/cost/native metrics, supporting analysis và validity.
5. **Summary:** trả lời RQs, đóng góp, limitations và hướng tiếp theo.

References có 17 nguồn trích dẫn. Phụ lục giữ CIs, selected attackers, các pha bổ sung,
ablation và lệnh tái lập. Xem [REVIEW_GUIDE.md](REVIEW_GUIDE.md) để review lập luận.

## Bổ sung sau review

- Đặc tả quyền truy cập dữ liệu: offline designer/evaluator, client prototype và provider;
  catalogue đầy đủ hiện đã có trong prototype. Control offline là hệ quả thông tin,
  không phải benchmark mới hoặc tuyên bố zero setup cost.
- Viết cap bằng tham số tổng quát \(C_s,u,M\), rồi mới thay cấu hình 0,23.
  Hình profile phân biệt Geo-I theo khoảng cách và guard Q; giữ backbone hiện tại.
- Bổ sung hệ quả Hit theo prior cho estimator bất kỳ, joint-channel composition,
  public-base independence và ví dụ hữu tỷ cho utility có cận dương/counterexample.
- Audit read-only đủ 120 TEST draws: saturation, cap và utility aggregation; giữ nguyên
  model, outputs, attacker, main scores và source benchmark.
- Đổi nhãn MAE proxy trong bảng metrics; không gọi là đã tái lập VehiTrack EIE.

Giới hạn còn nguyên: chưa có utility sau cap exhaustion, chưa chứng nhận actual prior/
calibration hay deployment sampler/restore, chưa chứng minh online querying cần thiết
khi client đã có đủ catalogue tĩnh. Γ0 có Recall tốt hơn Γ1 trên cohort hiện tại.

## Nguồn bằng chứng

- `artifacts/benchmarks/paper_comparison_20261028_v1/`: six-arm controlled comparison
  trên cohort đã xem. Ba paper adaptations là đối chứng chính, kèm Raw/REM-Z.
- `artifacts/benchmarks/remaining_scenarios_20261109_v1/`: 66 tác vụ bổ sung.
- `artifacts/benchmarks/s8_fusion_20261109_v2/`: 24 tác vụ joint-channel với fusion;
  giữ V1, initial verifier failure và checker sửa gate quá chặt.
- `configs/geo_i_bq_v1.json`: 108 frozen pins; không đổi core, sinh Q/GPS hoặc fit attacker.
- `docs/research/2026-10-14_geoi_bq_mathematical_foundation.md` và audit/receipt:
  chuỗi chứng minh có điều kiện.
- `artifacts/analysis/thesis_arguments_20261116_v1/`: audit masks/ledger/saturation và
  control thông tin có điều kiện; `evaluation/argument_bounds_20261116.py`: oracle hữu tỷ.

Kernel lý tưởng được phân biệt với float sampler, randomness và full upstream restore
chưa chứng nhận deployment. S4 là route-family proxy; S5 thiếu positive control mạnh;
S7 là conditional wrapper audit; S8 là aligned synthetic correlation. Danh tính
người/xe thật và native co-travel chưa có nhãn đánh giá tương ứng.

## Dựng và kiểm tra

Từ repo root, dùng `thesis/requirements.txt`:

```sh
python -m experiments.export_thesis_consolidated_20261109
python -m experiments.export_thesis_consolidated_20261109 --check
python -m experiments.audit_thesis_arguments_20261116 --check
python -m evaluation.argument_bounds_20261116
python -m unittest tests.test_argument_bounds_20261116 tests.test_thesis_argument_audit_20261116
```

Từ `thesis/`:

```sh
tectonic --keep-logs --outdir ../build/thesis main.tex
```

Sau khi kiểm tra, đặt `build/thesis/main.pdf` vào đường dẫn canonical và chạy từ root:

```sh
python -m experiments.validate_thesis_consolidated_20261109
```

Render mọi trang vào `build/thesis/render/page-01.png`…; chỉ dùng `--visual-reviewed`
sau khi xem bản cuối. Receipt hiện tại lưu tại `thesis/validation_20261116.json`; receipt 09/11 giữ nguyên như lịch sử.

Checker benchmark read-only:

```sh
python -m experiments.check_paper_comparison_20261028
python -m experiments.verify_remaining_scenarios_20261109
python -m experiments.check_s8_fusion_20261109
python -m experiments.verify_geoi_bq_mathematics_20261014 --check
```

Không gọi runner hoặc write-once verifier để dựng tài liệu.

## Bản trước

Nguồn và README trước consolidation ở `notes/snapshots/*before_consolidation_20261109*`;
PDF ở `artifacts/reports/archive/`. Các module `current_*.tex`, brief bảy chương 27/10
và report lịch sử còn để tra cứu nhưng không được import vào bản chính năm chương.
Kết quả khác protocol không được ghép vào main comparison.
