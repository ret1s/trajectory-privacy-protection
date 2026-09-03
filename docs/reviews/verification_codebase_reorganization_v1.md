# Verification — Codebase reorganization v1

Ngày kiểm tra: 2026-09-03
Base commit: `c564f1868be8cdb1f6c57b44050305e0e4940867`

Trạng thái: **PASS — active thesis pipeline, web app và benchmark SUMO vẫn chạy sau khi tái cấu trúc**

## 1. Mục tiêu và invariant

Đợt này chỉ thay đổi ranh giới và khả năng điều hướng của repository; không nâng
claim khoa học hay đổi output contract. Các invariant phải giữ là:

1. `thesis/main.tex` là một nguồn luận văn tự chứa và
   `output/pdf/graduation_thesis.pdf` là bản phát hành duy nhất.
2. Active code không import code Internship 2 hoặc prototype đã lưu trữ.
3. CLI cũ `python -m experiments.run_dummy_benchmark` và các patch point mà test
   đang dùng vẫn tương thích.
4. Dashboard chỉ đọc artifact công khai khi import; không kéo SUMO/data loader
   hoặc runner nặng vào process.
5. Bộ ba JSON/map/preview hiện hành vẫn ở `outputs/`; artifact cũ không bị xoá
   khỏi lịch sử mà chuyển sang `archive/`.
6. Ba comparator vẫn là clean-room `paper_adaptation`; việc đổi cấu trúc không
   được diễn giải thành faithful reproduction.

## 2. Cấu trúc mới và migration map

| Trước | Sau | Vai trò |
|---|---|---|
| `docs/supervisor_meeting/2026-09-05/report.tex` | `thesis/main.tex` | Nguồn LaTeX canonical |
| ba bản sao PDF luận văn | `output/pdf/graduation_thesis.pdf` | Một release PDF duy nhất |
| `Notes.md`, các note nghiên cứu lẫn trong `docs/` | `docs/research/` | Định nghĩa bài toán và backlog đang hoạt động |
| `docs/supervisor_meeting/` | `docs/meetings/<date>/` | Snapshot theo buổi gặp |
| Internship 2 code/doc nằm trong `core/`, `web/`, `legacy/`, `thesis/` | `archive/internship_2/` | Provenance, không thuộc runtime hiện hành |
| `outputs/sota_demo_*`, các map timestamp cũ | `archive/prototypes/` và `archive/internship_2/demo_outputs/` | Artifact đã thay thế |
| runner benchmark 1.658 dòng | façade 171 dòng + `experiments/dummy_benchmark/` | Tách config/source/execution/result/render/orchestration |

Các index mới: `README.md`, `docs/README.md`, `docs/reviews/README.md`,
`archive/README.md` và README cục bộ trong `core/`, `experiments/`, `web/`,
`outputs/`, `output/pdf/`, `thesis/`.

`benchmark/engines/` và `benchmark/methods/` vẫn tách riêng có chủ đích: engine
chứa thuật toán, method chứa adapter và evidence/limitation card. Đây không phải
hai implementation trùng nhau.

## 3. Ranh giới dependency

- `core/__init__.py` không còn eager-import GIS stack cũ. Import
  `core.demo_protocol` không load `folium`, `geopandas` hoặc `sklearn`.
- `web/benchmark_app.py` lấy schema từ
  `experiments/dummy_benchmark/constants.py`; import dashboard không load
  `experiments.run_dummy_benchmark` hay `data.sumo_demo`.
- Runtime dependencies hiện hành nằm trong `requirements.txt`; dependency chỉ
  dành cho ứng dụng Internship 2 nằm trong `requirements-legacy.txt`.
- `.idea/`, cache Python/pytest, LaTeX build files và scratch render không còn
  là nội dung được version-control.
- Test cấu trúc mới từ chối active imports từ `archive/`, thesis source/PDF
  trùng lặp và artifact cũ quay lại `outputs/`.

## 4. Kiểm tra tự động

Đã chạy trong `venv` của dự án:

```bash
MPLCONFIGDIR=/private/tmp/msc-mpl-pytest-venv \
  venv/bin/python -m pytest -q tests
MPLCONFIGDIR=/private/tmp/msc-mpl-runall-final \
  venv/bin/python -m tests.run_all
venv/bin/python -m compileall -q \
  benchmark core data evaluation experiments web tests archive
venv/bin/python -m pip check
```

Kết quả:

- `pytest`: exit code `0`.
- canonical runner: **124 passed, 0 failed**; tăng 5 structural regression test
  so với baseline 119 test.
- `compileall`: exit code `0`.
- `pip check`: không có dependency hỏng.
- 14 README/index hiện hành: không có relative link bị thiếu.
- Web factory smoke test: `GET /` trả `200`, `text/html`, 14.860 byte.

## 5. Benchmark SUMO sau khi tách runner

Lệnh smoke được ghi vào `/private/tmp`, không ghi đè canonical artifact:

```bash
venv/bin/python -m experiments.run_dummy_benchmark \
  --quick \
  --sumo-workdir /private/tmp/msc-sumo-reorg \
  --output /private/tmp/msc-dummy-reorg.json \
  --map-output /private/tmp/msc-dummy-reorg.html \
  --preview-output /private/tmp/msc-dummy-reorg.png
```

Kết quả: exit code `0`; nạp 4.892 đỉnh/9.138 cạnh đường hành khách, chọn một
record tám điểm và chạy đủ bốn pipeline. JSON evaluator, map HTML và preview PNG
đều được sinh. Output tiếp tục tách thành replacement, real-plus-dummies và
dummy-only; không tạo leaderboard chéo contract.

SHA-256 của smoke artifact tạm:

- JSON: `39e7165ec919a032613a1cfa148e09ebd141b05559e75df66effb66f347875b2`
- HTML: `a5e8aef201125d33a6b95ce5a2d7e01fa188d4c411c493a5e5883bbb48f6f950`
- PNG: `7615b6f64886cba68955064c5cf5e467ea66e217be975ea96f057efa31c826ce`

Các file tạm này chỉ chứng minh luồng tích hợp; timestamp, runtime và platform
làm chúng không byte-reproducible và chúng không phải kết quả SOTA.

## 6. PDF release QA

Hai source được build bằng XeLaTeX ra thư mục tạm, sau đó render toàn bộ bằng
Poppler ở 110 DPI:

- `output/pdf/graduation_thesis.pdf`: 50 trang A4, SHA-256
  `3b220d5d7ef40be2cf9e3d290f62c0880555293dfebd5addf79bf5e5074ff4eb`.
- `output/pdf/location_trajectory_privacy_foundations.pdf`: 43 trang A4,
  SHA-256
  `8ac273b7ca4c502b6e8ca6a2d3213eb66744566f3bce05fd676510f2ed861018`.

Không có overfull box, citation/reference chưa resolve hoặc yêu cầu rerun.
Luận văn chỉ khác bản đã duyệt ở physical pages 39 và 45: con số regression
suite được cập nhật 119 → 124 và mô tả thêm structural checks. Hai trang đã xem
ở kích thước gốc; không clipping/overlap. Foundations guide chỉ khác physical
page 39 do cập nhật đường dẫn archive; xem verifier v2 riêng.

## 7. Những điều cố ý chưa làm

- Không chuyển active packages vào `src/`: nhiều command, artifact và notebook
  hiện phụ thuộc chạy từ repository root; đây sẽ là migration API riêng nếu cần.
- Không đổi schema, default output path, RNG order hoặc metric implementation
  trong lúc tách runner.
- Không sửa path/line number trong verifier lịch sử. Đọc các file đó tại commit
  được ghi trong chính review.
- Không xoá dữ liệu raw, SUMO/OSM cache hoặc virtualenv cục bộ.
- Không nâng ba comparator thành `faithful_reimplementation`.

## 8. Gợi ý vòng tiếp theo

1. Chỉ tiếp tục tách `renderers.py` nếu có thay đổi tính năng thật; hiện tại tách
   thêm sẽ tăng churn nhưng chưa giảm coupling ở public API.
2. Khi bỏ compatibility façade, tạo một release có migration note và sửa các
   notebook/script ngoài repository trước.
3. CI nên cài `requirements-dev.txt`, chạy cả `pytest` và build thesis, đồng thời
   cache dependency thay vì commit cache sinh ra.
4. Reorganization không đóng các gap khoa học: POI/query workload, calibrated
   attacker, nhiều user/seed, uncertainty và parity với paper vẫn là ưu tiên.
5. `web/simulator.py` vẫn nạp graph và GeoLife ngay khi import (khoảng 13 giây
   trên máy kiểm tra). Đây là hành vi có trước đợt này; nên chuyển sang lazy
   application state ở vòng tối ưu web riêng, có test lifecycle tương ứng.
