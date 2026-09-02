# Verification v1 — final-thesis source, dummy benchmark và Flask dashboard

**Ngày kiểm tra:** 2026-09-02

**Nhánh:** `verifier`

**Commit nguồn dùng để chạy artifact:** `bccf7fd64b8127db109ead71323ac7c125cad10e`
**Phạm vi:** nguồn LaTeX luận văn, cấu trúc benchmark, ba phương pháp đối chứng,
phương pháp luận văn, SUMO smoke run, protocol public/evaluator và web app.

## Kết luận ngắn

| Hạng mục | Kết luận |
|---|---|
| Nguồn LaTeX luận văn chính thức | **PASS** — tiêu đề đúng, không còn nhãn báo cáo tiến độ/bản nháp, build được 29 trang A4 |
| Tổ chức code benchmark | **PASS** — package/registry/runner chuẩn, compatibility shim tách riêng |
| Tính đúng của output contract | **PASS** — kiểm tra 1:1 số event/timestamp và tách ground truth |
| Flask dashboard | **PASS cho demo localhost** — read-only, artifact v3, integrity check, attacker/evaluator endpoints riêng |
| SUMO integration artifact | **PASS ở mức smoke/integration** — chạy từ commit sạch, provenance và SHA hợp lệ |
| Ba SOTA reproduction hoàn chỉnh | **NOT YET / fail-closed đúng thiết kế** — hiện là paper adaptations, không được gọi là reproduced SOTA |
| Kết quả đủ để xếp hạng mô hình trong luận văn | **NO** — chưa có attacker cuối, POI utility, scenario/multi-seed và paper-specific validation |

Nói cách khác, phần hạ tầng luận văn + benchmark + web application đã đủ ổn để
phát triển tiếp. Phần khoa học so sánh SOTA chưa hoàn tất; repository đã được
thiết kế để không thể vô tình trình bày các con số smoke như kết quả SOTA.

## 1. Luận văn LaTeX

Nguồn nội dung hiện tại:

- `docs/supervisor_meeting/2026-09-05/report.tex` — nguồn tự chứa đang được phát
  triển dần;
- `thesis/main.tex` — build entry point ổn định, nhập đúng nguồn trên;
- `thesis/chapters/*` và `thesis/refs.bib` — nguồn Internship 2 cũ, được ghi rõ
  là không còn nằm trong build hiện tại.

Ba PDF đã được build lại từ cùng nội dung:

- `output/pdf/graduation_thesis.pdf`;
- `docs/supervisor_meeting/2026-09-05/report.pdf`;
- `thesis/main.pdf`.

Ba đường dẫn trên đã được đồng bộ từ cùng canonical binary ở
`output/pdf/graduation_thesis.pdf`, tránh lệch hash do timestamp của những lần
biên dịch riêng.

Kết quả kiểm tra:

- title metadata: `Bảo vệ tính riêng tư về quỹ đạo cho người dùng dịch vụ dựa
  trên vị trí`;
- 29 trang, A4, không mã hoá, không JavaScript;
- không còn `BÁO CÁO TIẾN ĐỘ LUẬN VĂN`, phụ đề cũ hoặc `BẢN NHÁP` trong text;
- không có undefined citation/reference và không có overfull box ở pass cuối;
- còn ba cảnh báo `Underfull \hbox` trong danh mục tài liệu tham khảo, không gây
  cắt/chồng nội dung;
- đã render toàn bộ 29 trang thành PNG để kiểm tra bố cục; kiểm tra lại riêng
  trang bìa và các trang bảng SOTA sau lần chỉnh cuối, không thấy lỗi hiển thị.
- SHA-256 chung của cả ba bản PDF:
  `b03c2c5fd023dcbbc1ecccdaaf39cd5ebf13cda2df18d011330f6c574ae5f702`.

Nội dung Chương 6 đã được đồng bộ với executable set: ba đối chứng chính là
TransProtect, AnotherMe và Semantic-correlation 2026. LSPPM-SI được ghi là đối
chứng phụ chưa cài vì thiếu code/POI/preprocessing. Phát biểu “code TransProtect
đã đủ cho attack và defense” đã được thay bằng mô tả có mức độ.

## 2. Cấu trúc benchmark

Đường chạy chuẩn:

```bash
venv/bin/python -m experiments.run_dummy_benchmark --quick
```

`experiments.run_sota_demo` chỉ còn là compatibility entry point. Các phần
chính được tách như sau:

- `benchmark/contracts.py`: mức độ cài đặt, source mapping, validation evidence
  và reproduced-SOTA gate;
- `benchmark/engines/`: algorithm engines dependency-light;
- `benchmark/methods/`: adapters với ID ổn định và machine-readable method card;
- `benchmark/registry.py`: inventory duy nhất cho runner/web;
- `core/sota_demo.py`, `core/thesis_demo.py`: alias tương thích cho tên `*Lite`
  cũ, không còn được canonical runner import.

Mỗi `ProtectedRun` hiện kiểm tra số event và timestamp khớp 1:1 với trajectory
thật cho cả ba contract: replacement, real-plus-dummies và dummy-only. Với
real-plus-dummies, real candidate ID chỉ tồn tại trong evaluator truth.

Gate `reportable_as_reproduced_sota` chỉ thành công khi đồng thời:

1. level là `official` hoặc `faithful_reimplementation`;
2. không còn component `MISSING`;
3. source evidence đủ và được pin phù hợp với level; và
4. có validation evidence không rỗng.

Lệnh dưới đây thoát mã 1 như mong đợi trước khi load graph/mobility:

```bash
venv/bin/python -m experiments.run_dummy_benchmark \
  --quick --no-map --require-faithful-sota
```

Thông báo đầu tiên liệt kê đúng các blocker của TransProtect, gồm learned
GCN/transformer, VehiTrack evaluation, paper dataset reproduction và validation
evidence.

## 3. Trạng thái ba đối chứng

| Method | Level hiện tại | Thành phần chạy được | Blocker chính |
|---|---|---|---|
| TransProtect | `paper_adaptation` | candidate pool trên road graph; utility/reachability/context score; replacement contract | GCN/transformer + traffic pipeline, VehiTrack calibration, Rome/SF reproduction |
| AnotherMe | `paper_adaptation` | whole-trajectory relocation, road snapping và continuity | virtual-user/history, POI/routing tương đương AMap, classifier/mobile reproduction |
| Semantic-correlation | `paper_adaptation` | real + `K-1`, stable opaque IDs, temporal/reachability score, semantic hook | LSTM/attention, 100x100 grid/transition pipeline, POI/train split và ASR reproduction |

Audited upstream revisions được pin trong method cards:

- TransProtect/VehiTrack:
  `035684c6c666a9af7cbd9984d92300000eb65536`;
- AnotherMe: `0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f`;
- Semantic-correlation: paper DOI được pin, chưa có public repository.

Không source nào từ upstream được copy vào repository vì chưa xác nhận license
cho việc vendoring. Đây là lựa chọn đúng về provenance, nhưng cũng có nghĩa
không được đổi nhãn adaptations thành official implementations.

## 4. Kết quả SUMO smoke run v3

Artifact được sinh sau khi commit nguồn và khi worktree sạch:

- schema: `msc-dummy-benchmark-v3`;
- status: `REFERENCE_ADAPTATIONS`;
- source commit: `bccf7fd64b8127db109ead71323ac7c125cad10e`;
- `source_dirty_before_run: false`;
- lệnh replay được ghi đúng dạng
  `/Users/geohanz/Project/msc/venv/bin/python -m experiments.run_dummy_benchmark --quick`;
- graph manifest: PASS, 13,813 road vertices;
- mobility: một SUMO controlled passenger record, 8 samples;
- không còn chuỗi `_lite`, `demo_only` hoặc `DEMO_ONLY` trong JSON;
- cả bốn method card đều có `reportable_as_reproduced_sota: false`;
- attacker views không chứa `evaluator_truth`, `real_trajectory`,
  `real_candidate_ids` hoặc `real_index`.

SHA-256 được kiểm tra lại từ file:

| Artifact | SHA-256 |
|---|---|
| `outputs/dummy_benchmark_map.html` | `197f16a79d5e01c1390bf2f3b974df5ff91ebf456afa8b7e94f40eb337a6a706` |
| `outputs/dummy_benchmark_preview.png` | `6cf5dd4936a753b941cac922af494d45fb16237fdf4ad20addc0edc4815b81ff` |

Các số smoke dưới đây chỉ là sanity diagnostics theo từng contract, không phải
leaderboard:

| Track | Method | Một số output đã quan sát |
|---|---|---|
| replacement | TransProtect adaptation | mean displacement 76.9 m; QoS-rate 1.000; on-road 1.000 |
| replacement | AnotherMe adaptation | mean displacement 2934.9 m; QoS-rate 0.000; on-road 1.000 |
| real + dummies | Semantic adaptation | K=4; real-in-set 1.000; set spread 712.0 m |
| dummy-only | thesis candidate | K=4; nearest-public 200.3 m; on-road 1.000 |

Preview đã được kiểm tra trực quan: bốn panel dùng cùng phạm vi, có mạng đường
OSM nền, trajectory thật evaluator-only và nhãn adaptation rõ ràng.

HTML nhúng hình học mạng đường và tắt raster tile trực tuyến theo mặc định,
nhưng Leaflet/CSS/JavaScript vẫn được tham chiếu từ CDN. Vì vậy HTML chưa phải
artifact air-gapped; PNG là bản xem hoàn toàn ngoại tuyến. JSON ghi rõ
`runtime_assets_may_require_network_or_browser_cache: true` để không overclaim.

## 5. Flask web application

Chạy bằng:

```bash
venv/bin/python -m web.benchmark_app
```

Kết quả test-client trên artifact thật:

- `/`: 200;
- `/api/benchmark`: 200, đúng schema v3, 4 mechanisms;
- attacker endpoint: 200, header `X-Benchmark-Visibility: attacker-visible`;
- evaluator endpoint: 200, header `X-Benchmark-Visibility: evaluator-only`;
- map/preview endpoints: 200 và
  `X-Artifact-Integrity: sha256-verified`;
- `POST /`: 405; không có route chạy benchmark;
- schema v2 cũ bị từ chối 503, không bị hiển thị như kết quả mới;
- evaluator routes có thể tắt bằng
  `BENCHMARK_ENABLE_EVALUATOR_VIEW=False`.

UI hiển thị implementation level, reproduced status, missing components,
upstream revision và output contract. Evaluator JSON chỉ được fetch khi người
dùng mở tab evaluator; nhiều record có selector riêng. Map được đặt trong iframe
sandbox và chỉ serve nếu SHA-256 khớp manifest.

Giới hạn kiểm tra: browser runtime của môi trường trả về danh sách rỗng, nên chưa
thực hiện được browser click-through/E2E screenshot. Không dùng backend browser
khác để thay thế. JavaScript đã qua `node --check`; route, JSON, headers và HTML
initial state đã được kiểm bằng Flask test client.

## 6. Test suite

```text
venv/bin/python -m tests.run_all
75 passed, 0 failed

venv/bin/python -m compileall -q benchmark core experiments web tests
PASS

node --check web/static/benchmark/dashboard.js
PASS

git diff --check
PASS
```

`pytest` không được cài trong project venv. System Python có pytest nhưng không
có runtime dependencies (`networkx`, `osmnx`, `Flask`), nên collection bằng
system Python không phải phép thử hợp lệ cho project này. Canonical dependency-
light runner trong venv là phép kiểm tra được dùng để kết luận.

Hai regression test mới xác nhận metadata luôn ghi lệnh replay hợp lệ: runner
nội bộ được đổi từ đường dẫn `.py` sang dotted module sau `-m`; script ngoài
repository dùng direct-file execution thay vì ghép một lệnh `-m` sai cú pháp.

## 7. Finding còn mở và thứ tự xử lý đề xuất

### V1-SOTA-001 — P0 — OPEN

Ba comparator chưa phải faithful SOTA implementations. Không được dùng số hiện
tại để nói phương pháp luận văn tốt hơn paper. Bước tiếp theo phải hoàn thiện
từng missing component hoặc chạy official upstream qua adapter đã pin, sau đó
bổ sung paper-equivalent validation evidence để gate mở.

### V1-EVAL-002 — P0 — OPEN

Chưa có context-aware continuous dummy-filtering attacker cuối: road/time/POI/
population prior, top-1/rank/MRR, EIE và reconstruction probability. Đây là
blocker trực tiếp trước mọi privacy conclusion.

### V1-DATA-003 — P0 — OPEN

SUMO mới là một randomTrips smoke record. Cần scenario builder cho S1--S7, nhiều
trajectory/user/seed, population calibration, train/test separation và khoảng
tin cậy. POI labels/semantics chưa đi vào runner.

### V1-GRAPH-004 — P1 — OPEN

SUMO movement graph là passenger-only, candidate graph là OSMnx multimodal.
Phải dùng chung graph hoặc xây mapping có test trước benchmark kết luận.

### V1-METRIC-005 — P1 — OPEN

Metrics hiện thiên về geometry/sanity. Cần POI utility hoặc route-service loss,
request/byte/latency overhead, cùng attack metrics đúng từng output contract.

### V1-LSPPM-006 — P1 — ACCEPTED LIMITATION

LSPPM-SI chưa executable. Luận văn đã ghi nó là đối chứng phụ; nếu quay lại chọn
làm một trong ba comparator chính thì phải cài Hilbert/spatial-influence/
bipartite-matching pipeline cùng dữ liệu POI.

### V1-WEB-007 — P1 — LOCAL-ONLY

Evaluator endpoints có ground truth và không có authentication. Cấu hình tắt đã
có; không deploy public với evaluator view bật. Trước production cần auth, CSP
và một WSGI server. Đây không chặn demo với giáo viên trên localhost.

### V1-WEB-008 — P2 — OPEN

Cần chạy browser E2E khi browser runtime khả dụng: chọn từng contract/method,
đổi record, mở evaluator tab, preview/map, test màn hình mobile và chụp evidence.

### V1-PROV-010 — P1 — RESOLVED

Lỗi cũ ghi `python -m /absolute/path/to/runner.py` đã được sửa. Artifact mới pin
đúng module `experiments.run_dummy_benchmark`, source commit sạch và có
regression tests. Lệnh trong JSON hiện copy/paste được từ repository root.

### V1-MAP-011 — P1 — RESOLVED BY SCOPE

Các mô tả “standalone/offline HTML” đã được thu hẹp. Mạng đường được nhúng và
raster tile không bật mặc định, nhưng runtime HTML còn dùng CDN. PNG được ghi là
offline fallback chính thức. Muốn HTML chạy air-gapped trong tương lai phải
vendor Leaflet/CSS/icon assets có kiểm tra license và thêm browser test khi ngắt
mạng.

### V1-THESIS-009 — P1 — OPEN

Nguồn hiện là khung luận văn đúng hướng, chưa phải bản nộp cuối. Còn phải bổ sung
chương thuật toán hoàn chỉnh, thiết kế thực nghiệm, kết quả, thảo luận, validity
threats và kết luận sau khi V1-SOTA/EVAL/DATA/METRIC được xử lý.

## 8. Hướng giao việc cho agent tiếp theo

Ưu tiên theo dependency:

1. xây `ScenarioSpec` + SUMO scenario builder cho S1--S7 và POI/population
   context snapshot;
2. cài attacker chung chỉ nhận `PublicTranscript` và side information công khai;
3. hoàn thiện một comparator mỗi lần, kèm upstream/paper conformance fixtures;
4. khóa metrics theo output contract và request budget;
5. phát triển thesis candidate theo objective của attacker;
6. chạy multi-seed benchmark, bootstrap confidence intervals và ablation;
7. chỉ khi faithful gate + scientific checks pass mới viết bảng so sánh vào
   chương kết quả.

Mọi agent phải đọc `benchmark/README.md`, method cards và file verifier này trước
khi thay đổi nhãn fidelity hoặc diễn giải kết quả.
