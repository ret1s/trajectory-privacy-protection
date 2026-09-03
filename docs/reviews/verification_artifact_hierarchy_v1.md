# Verification — Unified artifact hierarchy v1

Ngày kiểm tra: 2026-09-03

Base commit: `253001ecfa17e1f1118fcf04a786a18146dd6a8e`

Trạng thái: **PASS — chỉ còn một cây `artifacts/`; benchmark, web app và PDF
release tiếp tục dùng đúng artifact canonical**

## 1. Phạm vi và kết quả tái cấu trúc

Hai thư mục cấp cao dễ nhầm lẫn `output/` và `outputs/` đã được thay bằng một
namespace duy nhất:

```text
artifacts/
├── benchmarks/   # JSON, bản đồ HTML và ảnh preview do thực nghiệm sinh ra
└── reports/      # các PDF phát hành để đọc/review
```

Migration map:

| Đường dẫn cũ | Đường dẫn canonical mới |
|---|---|
| `outputs/*` | `artifacts/benchmarks/*` |
| `output/pdf/*` | `artifacts/reports/*` |

`benchmark/` vẫn là **mã nguồn** của các phương pháp đối chứng; còn
`artifacts/benchmarks/` là **kết quả sinh ra**. Hai vai trò này được ghi rõ trong
`artifacts/README.md` và `artifacts/benchmarks/README.md`.

Các đường dẫn mặc định của runtime Python được gom tại
`experiments/artifact_paths.py`; CLI, web app và provenance filter import cùng
contract này. Test và thesis khẳng định/ghi lại cùng hierarchy ở ranh giới phát
hành. Structural test từ chối việc tạo lại `output/` hoặc `outputs/` ở cấp
repository.

Các verifier và snapshot lịch sử không bị sửa đường dẫn cũ. Chúng là bằng chứng
append-only và phải được đọc tại commit được ghi trong từng file.

## 2. Canonical benchmark sau migration

Canonical quick run được chạy từ source tree sạch tại commit
`253001ecfa17e1f1118fcf04a786a18146dd6a8e`:

```bash
MPLCONFIGDIR=/private/tmp/msc-artifacts-canonical-mpl \
  venv/bin/python -m experiments.run_dummy_benchmark --quick
```

Kết quả provenance trong JSON:

- `source_dirty_before_run = false`;
- 4 pipeline đều chạy: TransProtect, AnotherMe, semantic-correlation và
  Geo-I anchored;
- evaluator giữ ba output contract riêng: replacement trajectory,
  real-plus-dummies và dummy-only;
- đường dẫn visual manifest đều là đường dẫn tương đối từ repository root dưới
  `artifacts/benchmarks/`;
- semantic graph hash ổn định:
  `41c40b6e2ad63b57ccecfa8ebb89b97475cd12df54e9a340a11dd8d94b21f960`.

SHA-256 của artifact canonical:

- JSON: `08464467aea15a0f22346d4256c203ba7590e3638e0a00adcc58333ea893e880`;
- map HTML: `f0f2dc168737560dbaa55e3bc5d37a24c93f284e0c80e033b7c499ef85524594`;
- preview PNG:
  `7615b6f64886cba68955064c5cf5e467ea66e217be975ea96f057efa31c826ce`.

Timestamp, runtime, metadata SUMO và serialization của map có thể thay đổi giữa
các lần chạy; semantic graph hash, schema, cấu hình và output phương pháp mới là
các trường dùng để đối chiếu logic. Đây vẫn là paper adaptation, không phải tuyên
bố faithful reproduction.

## 3. Kiểm tra tự động và tích hợp

Các kiểm tra cuối được chạy từ repository root trong `venv`:

```bash
venv/bin/python -m pytest -q tests
venv/bin/python -m tests.run_all
venv/bin/python -m compileall -q \
  benchmark core data evaluation experiments web tests archive
venv/bin/python -m pip check
```

Kết quả:

- `pytest`: exit code `0`;
- canonical suite: **124 passed, 0 failed**;
- `compileall`: exit code `0`;
- `pip check`: không có dependency hỏng;
- link scan của tài liệu hiện hành: không có relative link bị thiếu;
- web app mặc định trả `200` cho `/`, `/api/benchmark`, map evaluator và
  preview evaluator;
- visual manifest resolve đúng file và đúng SHA-256;
- active source/doc không còn tham chiếu đến hai đường dẫn cấp cao đã bỏ.

## 4. PDF release QA

Luận văn được build lại bằng XeLaTeX sau khi đổi đường dẫn, rồi render đủ 50
trang bằng Poppler ở 110 DPI. Physical pages 37, 39, 46 và 47 thay đổi so với
release trước do cập nhật provenance, runtime và manifest; cả bốn trang đã được
xem ở kích thước gốc và không có clipping, overlap hoặc trang trắng bất thường.
Log cuối không có citation/reference chưa resolve;
toàn bộ font đều embedded. Text extraction không còn `output/pdf/` hoặc
`outputs/`.

SHA-256 của các PDF phát hành:

- graduation thesis:
  `ad1f9ae66cd478fc54d17a98130bb1965f64ba724845517e3e5cd9d19868e422`;
- foundations guide:
  `8ac273b7ca4c502b6e8ca6a2d3213eb66744566f3bce05fd676510f2ed861018`;
- research improvements report:
  `68558375031514479c8369f5e6e977fd562899129598f73dc93c406469b08562`.

Hai PDF foundations/research-improvements chỉ được di chuyển byte-for-byte.
Graduation thesis được build lại vì source và appendix manifest đã đổi đường
dẫn/hash.

## 5. Giới hạn của kết quả xác minh

Đợt này chuẩn hoá layout và tính tái lập, không thay đổi thuật toán hoặc nâng
claim khoa học. Các gap về attacker calibration, workload POI/query, nhiều
user/seed và parity với paper vẫn giữ nguyên như verifier benchmark/thesis hiện
hành. Thư mục build tạm, SUMO cache và dữ liệu raw cục bộ không được đưa vào Git.
