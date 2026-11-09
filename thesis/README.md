# Luận văn hợp nhất — 09/11/2026

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

## Nguồn bằng chứng

- `artifacts/benchmarks/paper_comparison_20261028_v1/`: six-arm controlled comparison
  trên cohort đã xem. Ba paper adaptations là đối chứng chính, kèm Raw/REM-Z.
- `artifacts/benchmarks/remaining_scenarios_20261109_v1/`: 66 tác vụ bổ sung.
- `artifacts/benchmarks/s8_fusion_20261109_v2/`: 24 tác vụ joint-channel với fusion;
  giữ V1, initial verifier failure và checker sửa gate quá chặt.
- `configs/geo_i_bq_v1.json`: 108 frozen pins; không đổi core, sinh Q/GPS hoặc fit attacker.
- `docs/research/2026-10-14_geoi_bq_mathematical_foundation.md` và audit/receipt:
  chuỗi chứng minh có điều kiện.

Kernel lý tưởng được phân biệt với float sampler, randomness và full upstream restore
chưa chứng nhận deployment. S4 là route-family proxy; S5 thiếu positive control mạnh;
S7 là conditional wrapper audit; S8 là aligned synthetic correlation. Danh tính
người/xe thật và native co-travel chưa có nhãn đánh giá tương ứng.

## Dựng và kiểm tra

Từ repo root, dùng `thesis/requirements.txt`:

```sh
python -m experiments.export_thesis_consolidated_20261109
python -m experiments.export_thesis_consolidated_20261109 --check
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
sau khi xem bản cuối. Receipt lưu tại `thesis/validation_20261109.json`.

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
