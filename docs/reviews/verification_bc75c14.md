# Verification vòng 5 — audit package C4/O4/R4 tại `bc75c14`

## Hồ sơ review

| Trường | Giá trị |
|---|---|
| Branch được review | `verifier` |
| Candidate source (`C4`) | `4aea94b376510d2a66cb18eb19869237c2801be6` |
| Candidate outputs (`O4`) | `7f3b8d2bfb06af703f23cf7196c380ad66a9c000` |
| Candidate response (`R4`) | `bc75c142e2f838181cb394766d40333b5e2abc77` |
| Verifier target | toàn bộ chuỗi `C4 -> O4 -> R4` |
| Ngày review | 2026-08-24 (Asia/Ho_Chi_Minh) |
| Phạm vi | Formal statements, production mechanisms, API/state, tests/RNG, graph/data provenance, attackers, experimental design/statistics, thesis/docs/web/PDF consistency, novelty positioning |
| Kết luận | **NEEDS REVISION — có tiến bộ tái lập được, nhưng result package hiện tại chưa final-verified** |

Review này độc lập và commit-scoped. Mọi line reference trỏ tới snapshot
`bc75c14` (code/output gốc lần lượt ở `C4`/`O4`), không phải một working tree đã
sửa sau review. Commit `R4` chỉ là response; việc một dòng trong response ghi
`RESOLVED` không được xem là evidence tự thân.

### Quy ước trạng thái

- **PASS/RESOLVED**: root cause, implementation/evidence và manuscript cùng đạt
  acceptance criteria.
- **NARROWED**: capability mạnh hơn chưa có, nhưng claim hiện tại đã được hạ đúng
  phạm vi evidence.
- **PARTIAL**: có tiến bộ thực chất nhưng còn ít nhất một acceptance criterion fail.
- **FAIL/REGRESSED**: có phản ví dụ trực tiếp hoặc artifact hiện tại không support
  claim đang công bố.
- **PLANNED**: chỉ là future work, không được gọi là contribution đã verify.

Audit tiếp tục tách bốn lớp, vì pass ở một lớp không kéo theo pass ở lớp khác:

1. **Ideal theorem** — phát biểu và proof kernel real-arithmetic có literal-correct?
2. **Executable mechanism** — float/RNG/state/API hiện thực đúng kernel và scope?
3. **Evaluation evidence** — sampling, attacker, metric và statistics support kết luận?
4. **Artifact provenance** — source/data/environment/command có pin đủ để tá lập?

## 1. Kết luận điều hành

### 1.1 Những gì đã verify được

Package vòng này có nhiều tiến bộ thật, không chỉ là sửa câu chữ:

- workflow ba commit đúng thứ tự: source sạch `C4`, output/PDF `O4`, response-only
  `R4`; cả hai result JSON pin `source_commit=4aea94b` và
  `source_dirty_before_run=false`;
- `venv/bin/python -m tests.run_all` chạy **19 passed, 0 failed**;
- semantic SHA-256 seed derivation trong `experiments/rng_util.py` loại bỏ ordinal
  mechanism/epsilon; spot-check production xác nhận đảo/chèn mechanism không đổi
  stream REM khi semantic key không đổi;
- full regeneration đường utility/QoS của 18 benchmark rows khớp output tuyệt đối
  (`max absolute difference = 0`);
- raw-to-aggregate reconciliation pass: benchmark sai khác tối đa
  `1.821e-5` do raw row làm tròn 4 chữ số; toàn bộ median, bootstrap CI endpoints và
  success probability của averaging tái tạo đúng từ raw rows;
- local graph artifact khớp manifest: SHA-256, `13,813` nodes, `41,040` edges,
  `1` weak component, bbox và highway distribution đều pass;
- Table 5.1 và Table 5.2 khớp JSON sau rounding;
- PDF 32 trang A4, fonts/render các trang nội dung chính sạch, không có placeholder
  `??` hoặc black-box font lỗi;
- PR wording đã bỏ claim anti-averaging tuyệt đối và mô tả đúng hai nhánh đều có
  xác suất dương ở finite horizon;
- README đã tách thesis family khỏi legacy family rõ hơn vòng trước.

Đây là evidence tích cực đáng giữ lại ở vòng sửa tiếp theo.

### 1.2 Các blocker khiến chưa thể final-verify

1. **Averaging population/primary key bị lặp.** Cấu hình ghi 40 stay-points nhưng
   chỉ có 39 cặp `(user, full-precision coordinate)` và 38 `home_id` duy nhất sau
   rounding. `1,600` raw rows chỉ còn `1,520` key
   `(mechanism, home_id, seed)` duy nhất. Một stay của user `014` bị lặp đúng byte;
   hai stay user `023` cách nhau khoảng `0.0068m` nhưng collision sau rounding.
2. **So sánh mechanism không còn paired.** RNG cho nhiễu GPS ngoại sinh chứa tên
   mechanism, nên mỗi mechanism nhận một jitter realization khác nhau cho cùng
   home/seed. Với chỉ 8 seed, random imbalance đi thẳng vào difference giữa methods.
3. **`on_road_rate` không đo khoảng cách tới road edge.** Code đo nearest road
   vertex nhưng Ch5 diễn giải phần bù là “điểm ngoài lưới đường”. Independent
   point-to-edge recomputation cho Planar cho off-edge `18.0%`, `12.4%`, `7.95%`
   ở epsilon `0.01/0.02/0.05`, không phải `54.1%`, `48.7%`, `41.4%` suy từ cột hiện tại.
4. **Định lý SM-REM chưa literal-complete và không khớp cache-miss production.** Nó
   không định nghĩa cell metric/cặp cell/transcript inequality, đồng thời gọi mẫu
   đầu là REM dù code có thể gọi T-REM nếu chỉ cache của cell rỗng nhưng temporal
   history vẫn còn.
5. **PR API nhận budget âm/0/NaN.** Có thể tạo object, phát output và quảng cáo
   `privacy_cost_per_step_max=0.0` với `epsilon_release=-0.01` và
   `epsilon_test=0.01`.
6. **Test xanh nhưng còn load-bearing blind spots.** RNG test có `... or True`;
   REM distribution test lấy expected probability từ chính production logits nên
   mutation hệ số `epsilon/2 -> epsilon` vẫn pass; không có SM lifecycle/geometry
   hoặc exhaustive production transcript tests.
7. **Attacker/metric prose vẫn overclaim.** Ch5 còn suy “riêng tư hơn” từ proxy;
   HMM simulator bỏ `logZ`; averaging MLE không exact dưới jitter/restricted search;
   posterior mean không optimal cho Euclidean loss.
8. **Provenance sạch nhưng chưa đủ tá lập.** Không pin GeoLife bytes, dependencies,
   cwd/interpreter; `started_at_utc` và git state được lấy sau computation; raw
   benchmark thiếu counts để reconstruct cột HMM coverage.
9. **Repo-wide formal/narrative sweep chưa xong.** Supporting docs và simulator
   vẫn gán Euclidean Geo-I/memoization/composition sai cho SM/PR, và gọi nearest-
   vertex evidence là proof chống map matching.
10. **Novelty positioning chưa cập nhật văn liệu tháng 7/2026.** Một bài Computer
    Networks 2026 đã nghiên cứu chính Bayesian attack cho stationary users, HMM cho
    mobile users, cùng ba defense Memoization/Replay/Replication. Overlap với SM/PR
    phải được thảo luận và novelty hạ về road-constrained metric integration, không
    dùng câu “chưa ai làm/chưa có công bố chiếm giữ”.

### 1.3 Release decision

Không nên dùng snapshot này như số liệu final/submission package. Ít nhất phải sửa
R4-001, R4-002 và R4-003 rồi rerun `O5`; không được chỉ sửa Table/PDF thủ công vì
raw rows, weighting, CI và paired design đều bị ảnh hưởng.

Các câu sau chưa an toàn để đưa vào abstract/contribution/conclusion:

- “40 stay-point x 8 seed” hoặc “1,600 independent observations”;
- “RNG order-independent/fair across mechanisms” nếu jitter vẫn mechanism-specific;
- “41--54% Planar outputs off-road”;
- “T-REM riêng tư hơn REM” dựa trên HMM proxy;
- “MLE chính xác cho REM” mà không kèm `vertex-secret, no-jitter, iid, full search`;
- “SM-REM đạt epsilon-Geo-I mức ô” mà không phát biểu metric/kernel/cell pair;
- “PR-SM-REM/memoization là novelty chưa có trong literature”;
- “provenance đầy đủ/tất cả table cells reconstruct được”;
- “19+10 tests” (đúng là 19 total, trong đó 10 production tests).

Framing an toàn hiện tại:

> Đây là industry-aware research prototype. REM/T-REM có ideal single-step metric
> arguments và road-vertex output structure; SM là exact cache defense cho static
> same-cell trong lifecycle hẹp nhưng không có Euclidean/trajectory guarantee; PR là
> previous-release probabilistic reuse với ideal per-step bound. Evaluation hiện là
> exploratory, attacker chủ yếu là proxy, và result averaging cần rerun sau khi sửa
> identity/paired-input bugs.

## 2. Status matrix của R3-001...R3-015

| ID | Response tại `bc75c14` | Verified status | Kết luận ngắn |
|---|---|---|---|
| R3-001 | RESOLVED | **PARTIAL** | Hai lỗi gốc (base Geo-I assumption, support mask) đã sửa; tỷ số cho mọi measurable event vẫn có case `0/0` |
| R3-002 | RESOLVED | **PARTIAL verification** | Semantic RNG implementation tốt; test có tautology, không test runner output; rounded home-ID collision và unpaired jitter còn |
| R3-003 | RESOLVED + caveat | **PARTIAL** | Clean source pin pass; data/env/start/command/unique record provenance thiếu |
| R3-004 | RESOLVED | **PASS** | PR đã được hạ đúng finite-horizon và both-branches-probabilistic |
| R3-005 | RESOLVED | **FAIL/PARTIAL** | Cell theorem underdefined, REM/T-REM initialization mismatch, không có SM tests, docs/UI contradiction |
| R3-006 | RESOLVED | **PARTIAL** | Tests gọi production code nhưng không mutation-sensitive/exhaustive như acceptance criterion |
| R3-007 | NARROWED | **PARTIAL/NARROWED** | Caveat tốt hơn; Ch5/README/Ch2/simulator vẫn overclaim hoặc dùng attacker khác |
| R3-008 | PARTIAL | **PARTIAL** | Runtime graph fail-closed pass; clean-clone downloader/source hash/env recipe chưa đúng như docs |
| R3-009 | RESOLVED | **PASS, minor handoff debt** | Family split rõ; README intro vẫn ưu tiên internship/legacy artifact |
| R3-010 | PARTIAL | **REGRESSED/PARTIAL** | Raw rows có, nhưng duplicate stay/ID, unpaired jitter, one-seed và pooled success CI |
| R3-011 | RESOLVED | **FAIL** | Nearest-vertex bị diễn giải thành off-road, nhiều formal/attacker/web contradictions còn |
| R3-012 | RESOLVED | **PARTIAL** | Keyword-only/2E footgun fixed; non-positive/non-finite budgets và state wording chưa fixed |
| R3-013 | PARTIAL | **PARTIAL** | Profile multimodal đã disclose; 35.28% explicit foot/cycle/path/pedestrian/steps, chưa support drive-only deployment |
| R3-014 | RESOLVED | **PARTIAL** | Canonical runner pass; project venv không có pytest, runtime deps broad `>=`, không CI/locked env |
| R3-015 | NARROWED/PLANNED | **PARTIAL/PLANNED** | Comparator thiếu; absolute novelty claims và literature update 2026 chưa xử lý |

### Gate decision độc lập

| Gate | Response expectation | Verifier decision |
|---|---|---|
| G0 Formal | PASS | **FAIL/PARTIAL** |
| G1 Production/tests | PASS | **PARTIAL** |
| G2 Provenance | PASS except raw clone | **FAIL/PARTIAL** |
| G3 Evaluation validity | PASS | **FAIL** |
| G4 Statistical evidence | PARTIAL | **FAIL/PARTIAL with progress** |
| G5 Manuscript consistency | PASS | **FAIL** |
| G6 Industry/novelty | PARTIAL | **PARTIAL** |

## 3. Methodology và positive evidence

### 3.1 Commit/worktree integrity

Review bắt đầu trên branch `verifier`, HEAD/upstream cùng `bc75c14`, working tree
sạch. Chuỗi commit là:

```text
4aea94b  Round 4 C4: formal/API/RNG/provenance/test/claims fixes
7f3b8d2  Round 4 O4: regenerate outputs from clean C4 + tables + PDF
bc75c14  Round 4 R4: response to verification_c5385ec.md
```

`git diff --check` trên candidate range pass. `R4` chỉ thay response file; output
bytes không đổi từ `O4`.

### 3.2 Tests và production smoke

Canonical command:

```bash
venv/bin/python -m tests.run_all
```

Kết quả: `19 passed, 0 failed`. Con số response “19+10” sai vì 10 production-
coupled tests nằm trong tổng 19.

Hai đường pytest không phải canonical-green:

```text
venv/bin/python -m pytest -q tests  -> venv không cài pytest
python3 -m pytest -q               -> collection fail vì system Python thiếu networkx
```

Quick benchmark clean-C4 chạy hết, raw rows finite. Independent regeneration toàn
bộ utility/QoS cho 18 official rows khớp tuyệt đối. Đây là evidence cho deterministic
utility path; full expensive attacker suite không được verifier rerun hai lần end-to-end.

### 3.3 Output reconciliation

- `benchmark_results.json`: 360 raw rows = 18 mechanism/epsilon cells x 20
  trajectories; semantic record keys unique; aggregate mean fields reconstruct với
  max discrepancy `1.821e-5` từ raw rounded values.
- `averaging_multi_results.json`: formulas cho median, user-cluster bootstrap CI và
  pooled success rate reconstruct đúng output hiện có.
- Tuy nhiên “formula reconstructs committed aggregate” không chứng minh sample
  design hợp lệ: duplicate identities và mechanism-specific jitter nằm trước bước
  aggregation.
- Benchmark raw rows không chứa per-trajectory `hmm_covered/hmm_total`, nên không
  reconstruct được cột `cov` chỉ từ raw rows.

### 3.4 Graph artifact

| Thuộc tính | Verified value |
|---|---:|
| GZip SHA-256 | `ed0af579fb32c7164de85bca945a10432f818dd27eb80d6d5a7afa62213b0407` |
| XML SHA-256 | `5691186b8b3bd9078915faefb7fb639ff2c7f45b4e8b1efd1de4b9ebdad4c8a5` |
| Graph SHA-256 | `9e44ec2a48d6249f3534bb8511010925428018e528f5116d5f9eb24d86a2d487` |
| Nodes / edges | `13,813 / 41,040` |
| Weak components | `1` |
| Latitude extent | `39.960011 -- 40.019992` |
| Longitude extent | `116.290003 -- 116.359997` |

Artifact local pass không đồng nghĩa rolling URL/build recipe của clean clone sẽ tái
tạo đúng bytes. Explicit foot/cycle/path/pedestrian/steps chiếm 14,480 edges
(`35.28%`); unknown highway types thêm `22.97%`. Vì vậy chỉ gọi là multimodal graph.

### 3.5 PDF QA

`thesis/main.pdf` có 32 trang A4, fonts hiển thị bình thường; tables thực ở các trang
nội dung không clipping. Tuy nhiên trang vật lý 5 (`Danh sách bảng`, printed page iv)
có hai caption Table 5.1/5.2 tràn mép phải và cắt tên JSON/page number. Build log báo
overfull boxes khoảng `90.08pt` và `107.88pt` từ captions ở
`ch5_thucnghiem.tex:57-64,174-180`. Visual main-body pass nhưng submission QA và
semantic QA fail.

## 4. Findings mới R4 và suggest improvements

### R4-001 — P0 release blocker: 40 stay records nhưng chỉ 38 stable IDs

**Evidence**

- `data/geolife.py:166-200` append detected stays và cap per user nhưng không dedupe,
  không giữ source file/segment/stay index;
- `experiments/run_averaging_multi.py:89-90` tạo `home_id` bằng user + coordinates
  làm tròn 6 chữ số;
- `outputs/averaging_multi_results.json:55-56` lặp
  `014@39.985369,116.320213`;
- `outputs/averaging_multi_results.json:67-68` lặp
  `023@39.969793,116.300581`.

Independent count:

```text
selected entries                         40
unique (user, full-precision coordinate) 39
unique rounded home_id                   38
raw rows                               1600
unique (mechanism, home_id, seed)       1520
colliding composite-key groups            80
byte-identical colliding groups            40
```

User `014` là exact duplicate (`0m`); cả 5 mechanisms x 8 seeds = 40 duplicate
row-pairs byte-identical. Hai điểm user `023` cách nhau khoảng `0.00678m`, nhưng
rounding làm cùng ID và cùng RNG streams. Raw row composite key do đó không phải
primary key, provenance không truy ngược raw stay, và bootstrap/median double-weight
ít nhất một location.

Sensitivity khi gộp các sub-centimetre duplicates cho median ở `n=100`:

| Mechanism | committed (m) | deduplicated sensitivity (m) |
|---|---:|---:|
| Planar Laplace | 23.598 | 23.268 |
| REM | 29.106 | 28.232 |
| T-REM | 29.638 | 28.232 |
| SM-REM | 155.922 | 152.046 |
| PR-SM-REM | 51.162 | 50.997 |

Direction chính chưa đảo, nhưng exact cells/CIs không còn là final evidence.

**Required fix**

1. Định nghĩa estimand: unique significant location, visit episode, hay user-level
   location cluster. Không được vô tình coi cùng stay là independent location.
2. Giữ intrinsic source identity: GeoLife file path/hash, segment time bounds,
   within-segment stay index, full-precision coordinate; UUID/hash có thể derive từ
   tuple này.
3. Dedupe exact/near-identical stays theo một threshold có giải thích, hoặc cluster
   repeated visits và dùng cluster weight rõ ràng.
4. Assert `len(selected_ids)==len(set(selected_ids))` và uniqueness của raw composite
   key trước khi write output.
5. Rerun JSON, raw rows, Table 5.2, captions/conclusions và PDF từ clean source.

**Acceptance test**

- zero duplicate selected IDs và raw keys;
- config count bằng actual unique study units;
- raw row trace được về source record;
- raw-to-aggregate test pass sau rerun;
- manuscript không hard-code “40” nếu population thực tế khác.

### R4-002 — P1: exogenous GPS jitter phụ thuộc mechanism, mất paired fairness

**Evidence**

`experiments/run_averaging_multi.py:113-116` dùng:

```python
rng_from_key("avg-jitter", SEED, eps, m, hid, s)
```

Tên mechanism `m` phải thuộc RNG của mechanism, nhưng không nên thuộc RNG của GPS
jitter ngoại sinh. Hiện Planar/REM/TREM/SM/PR nhận năm input-noise realizations khác
nhau cho cùng home/seed. Ví dụ bốn standard-normal draws đầu ở home `000`, seed `0`
khác hoàn toàn giữa mechanisms (Planar `[-3.1165, 9.5654, ...]`, REM
`[-7.7719, 3.7724, ...]`, SM `[14.8359, 9.9020, ...]`).

**Required fix**

```text
jitter key    = (schema, root_seed, epsilon, stable_home_id, replicate)
mechanism key = (schema, root_seed, epsilon, mechanism_name, stable_home_id, replicate)
```

Sinh noisy input sequence một lần cho mỗi `(home, seed)` rồi feed cùng sequence cho
tất cả mechanisms. Nếu design cố ý unpaired, phải tăng replication, báo independent-
group inference và không gọi matched comparison; paired design hợp lý hơn ở đây.

**Acceptance test**

- instrument runner và assert input coordinates byte-identical across mechanisms
  cho cùng `(home, seed, report index)`;
- đảo/chèn mechanism không đổi cả exogenous input lẫn output của mechanisms cũ;
- rerun O5 sau fix, không reuse O4 numbers.

### R4-003 — P1: `on_road_rate` là nearest-vertex rate, không phải on-road rate

**Evidence**

`evaluation/metrics.py:76-85` tự ghi rõ nearest **road VERTEX**, nhưng column/caption
và Ch5 `:107-110` diễn giải `1-rate` thành percentage “rơi ngoài lưới đường” và liên
hệ trực tiếp RAoPT/map matching. Một point giữa road segment dài có thể xa mọi vertex
nhưng vẫn nằm trên edge.

Verifier tái sinh đúng full C4 Planar streams (20 trajectories/858 points) rồi dùng
projected nearest-edge distance ở cùng threshold 25m:

| epsilon | committed near-vertex | true near-edge | true off-edge |
|---:|---:|---:|---:|
| 0.01 | 0.4590 | 0.8196 | **18.0%** |
| 0.02 | 0.5132 | 0.8760 | **12.4%** |
| 0.05 | 0.5860 | 0.9205 | **7.95%** |

Structural statement “road mechanisms output graph vertices” vẫn đúng. Numerical
statement “41--54% Planar releases off-road” sai semantics và phóng đại actual
off-edge mass khoảng 2--5 lần.

**Required fix**

- Option A: đổi tên metric/cột/prose thành `near_vertex_rate`, bỏ mọi inference về
  actual off-road/map-matching surface;
- Option B (khuyến nghị): implement projected point-to-edge distance, add unit test
  với point ở giữa một long edge, rerun outputs/table/PDF;
- không gọi structural output support là bằng chứng RAoPT-resistance; muốn claim đó
  phải chạy attack/map matcher trực tiếp.

### R4-004 — P1 formal: SM cell theorem underdefined và initialization mismatch

**Evidence**

`thesis/chapters/ch4_phuongphap.tex:293-303` cố định một `x,C` rồi nói transcript
“thừa hưởng epsilon-Geo-I mức-ô”, nhưng không:

- định nghĩa space/metric trên cells;
- lượng hóa hai cells `C,C'`;
- viết transcript event inequality;
- nói rõ temporal history phải empty/common, không chỉ cache cell `C` empty.

Theorem đặt `Z ~ REM(rep(C))`; production `StayMemoizedREM` kế thừa TREM và cache
miss gọi `super().perturb` (`core/mechanisms.py:350`). Nếu cache C rỗng nhưng previous
public release từ cell khác còn tồn tại, mẫu đầu là T-REM conditioned on history,
không phải REM.

Một form tối thiểu cần lượng hóa:

```text
Pr[M_T(C) in S | h] <=
exp(epsilon * d_cell(C,C')) Pr[M_T(C') in S | h]
```

với `d_cell(C,C') = d(rep(C), rep(C'))`, common public history/reset rõ ràng, và
kernel first-release thực tế (REM hoặc T-REM) thống nhất với code.

Không có test nào cho `StayMemoizedREM`, corner/edge geometry, A-B-A trace, cache
lifecycle hay common-history condition.

**Required fix**

- viết theorem ở kernel/event level với cell metric và pair quantification;
- hoặc reset cả temporal state ở scope theorem, hoặc phát biểu first release là
  `TREM(rep(C) | common public history)`;
- thêm exhaustive tiny-graph tests cho same-cell repeat, edge/corner factor,
  A-B-A counterexample và reset/lifecycle;
- sửa `docs/problem_formulation.md:46-49,64-75` cùng simulator wording.

### R4-005 — P1 executable API: budget âm/0/NaN vẫn được chấp nhận

**Evidence**

Constructor `PrivateReuseSMREM` (`core/mechanisms.py:445+`) chỉ kiểm mutual exclusion
và `None`, không validate positivity/finiteness. Counterexample:

```python
m = PrivateReuseSMREM(
    rn, epsilon_release=-0.01, epsilon_test=0.01
)
print(m.privacy_cost_per_step_max)  # 0.0
m.perturb(...)                      # vẫn phát output
```

Negative release epsilon đảo preference của exponential mechanism, nhưng public API
lại quảng cáo privacy cost bằng zero. `epsilon_step_cap=0`, negative và `NaN` cũng
được nhận. Existing footgun test không cover các case này.

**Required fix / acceptance**

- reject mọi epsilon/cap không finite hoặc `<=0` trước khi tạo state;
- kiểm `epsilon_test + epsilon_release` finite và không overflow;
- parameterized tests cho `0`, âm, `NaN`, `+/-inf`, wrong type;
- không dùng `max(0, sum)` hoặc arithmetic accidental để “sanitize” invalid budget.

### R4-006 — P1 test validity: production-coupled nhưng chưa mutation-sensitive

**Evidence**

- `tests/test_production_mechanisms.py:34+` lấy expected softmax từ chính
  `production logits`; mutation coefficient `-epsilon*d/2` thành `-epsilon*d` vẫn
  cho test pass;
- sign test chỉ bắt dấu, không bắt hệ số `1/2`;
- analytic PR test vẫn copy ideal kernel riêng thay vì enumerate production output;
- RNG test tại `:167-180` chỉ test helper, không runner output, và dòng assertion
  unrelated mechanism có `... or True`, nên không thể fail;
- state reuse test không kiểm `n_resample`; resample tình cờ trả lại cùng vertex có
  thể trông như reuse;
- không có two-step same-public-history composition test hoặc SM tests.

**Required test set**

1. Hand-computed tiny graph probability kiểm đúng coefficient `epsilon/2`, không
   derive expected từ production logits.
2. Exhaustive production PR one-step cho `z=h` và `z!=h`, cùng two-step transcript
   dưới common public history.
3. State-machine test kiểm explicit branch counters/trace, không infer branch chỉ từ
   equality của output.
4. Actual runner permutation/insertion test cho mechanism và epsilon lists.
5. SM geometry/lifecycle/counterexample tests từ R4-004.
6. Xóa mọi `or True`; chứng minh mutation checklist bằng ít nhất targeted mutation
   run hoặc parameterized witnesses.

19/19 hiện tại là smoke evidence tốt, nhưng chưa đủ acceptance cho formal coupling.

### R4-007 — P1: attacker labels và inference conclusions chưa claim-safe

**Evidence**

- Averaging MLE exact chỉ trong vertex-secret/no-jitter/iid với full search;
  `evaluation/attacks.py:252-294` dùng candidate truncation 1200m chưa được chứng minh,
  official study thêm jitter 10m, nhưng Ch5 `:159-177` vẫn gọi “chính xác cho REM”.
- HMM có exact REM **emission family** khi dùng `logZ`, không phải toàn attacker
  optimal: transitions/candidate truncation là modeling choices và estimator dùng
  posterior mean (`attacks.py:193-197`), không phải geometric median tối ưu cho
  Euclidean loss.
- Offline HMM trừ `logZ` (`attacks.py:159-161`); `online_estimates` bỏ nó
  (`:200-232`); simulator gọi online path không truyền normalizer
  (`web/simulator.py:108-109`). Simulator vì vậy đánh giá một attacker khác/yếu hơn.
- Ch5 `:112-116` vẫn kết luận T-REM “không mất riêng tư” và “riêng tư hơn” từ HMM
  proxy, trái caveat `:225-233` và response G3.
- Ch2/README còn dùng “optimal for REM” quá rộng; module header `attacks.py:13-16`
  stale so với actual implementation.

**Required fix**

- dùng đúng cụm: “REM likelihood/emission exact under vertex-secret, no-jitter,
  iid assumptions”; không gọi end-to-end HMM/MLE optimal/exact ngoài scope đó;
- pass/fix `logZ` trong online simulator và test offline/online parity;
- bỏ mọi privacy ranking từ proxy; chỉ gọi diagnostic error;
- nếu muốn comparative privacy, implement mechanism-aware, jitter-aware sequential
  likelihood/attacker và geometric-median decision rule cho Euclidean loss;
- audit `README`, Ch2, Ch5, docs, web labels cùng một registry.

### R4-008 — P1 provenance: clean commit pass, experiment identity chưa đủ

**Positive evidence**

Hai JSON pin `source_commit=4aea94b`, `source_dirty_before_run=false`, graph hash,
root seed, epsilons và selected records. Đây là sửa đúng root cause dirty-parent ở
vòng trước.

**Residual evidence**

- `experiments/provenance.py:73-94` không lưu GeoLife file/version/hash hoặc exact
  segment time bounds;
- không lưu installed package versions/lock hash; runtime requirements dùng broad
  `>=`; `requirements-dev.txt` chỉ pin pytest nhưng comment gọi environment pinned;
- `command=sys.argv` thiếu interpreter, `-m` form, cwd và relevant environment;
- runner gọi `provenance()` sau computation (`run_benchmark.py:175+`, averaging
  `:154+`), nên `_git_state()` không phải “before run”; `started_at_utc` cũng thực
  chất gần end time. Benchmark ghi `12:23:57Z`, trong khi tổng cell runtimes khoảng
  `870.5s`, suy ra computation bắt đầu gần `12:09:26Z`;
- duplicate/colliding home IDs không source-backed;
- benchmark raw rows thiếu `hmm_covered/hmm_total`, nên table coverage không raw-
  reconstruct được.

**Required fix**

- capture immutable run context và start time trước loop; capture end separately;
- pin dataset bytes/version, loader args, intrinsic segment/stay IDs;
- lưu interpreter, cwd, canonical command, platform, package versions hoặc lock hash;
- lưu stream keys/semantic IDs đã resolve;
- bổ sung raw numerator/denominator cho mọi displayed aggregate;
- schema test phải fail nếu missing/duplicate ID, dirty source hoặc graph mismatch.

### R4-009 — P2 formal wording: fixed-public proposition còn case `0/0`

`ch4_phuongphap.tex:98-118` đã thêm hai correction quan trọng: base kernel phải
epsilon-Geo-I và conditioned density có indicator `1_A`. Tuy nhiên statement nói
“mọi measurable S” rồi viết ratio `K_A(S|x)/K_A(S|x')`. Với common kernel
`delta_0`, `A={0}`, `S={1}`, cả hai probabilities bằng zero; displayed ratio là
`0/0`, không xác định.

**Required fix**

Phát biểu DP inequality trực tiếp:

```text
K_A(S | x) <= exp(2 epsilon d(x,x')) K_A(S | x')
```

cho mọi event, rồi chỉ dùng density/pointwise ratio khi có common dominating measure
và denominator dương. Test comment đang nói witness “hits” `e^(2epsilon)` nhưng
numeric example chỉ cho thấy bound epsilon không đủ; sửa thành “witnesses factor-2
necessity / approaches or lies below the bound” theo đúng evidence.

### R4-010 — P1 manuscript/web consistency: response R3-011 chưa RESOLVED

Các contradiction representative:

- `docs/problem_formulation.md:46-49` nói cả SM/PR đạt Euclidean Geo-I và PR là
  memoization/static exact-repeat, nhưng H2 `:64-73` và code nói ngược lại;
- `docs/problem_formulation.md:74-75` nói decision memoize không rò, trong khi
  revisit hit/miss là secret-dependent infinite-ratio channel;
- `docs/system_model_and_threats.md:51-52` suy REM có thể đạt usefulness `delta=1`
  dù committed QoS 200m thấp hơn 1;
- `:109` gọi REM/T-REM cải thiện worst-case `T epsilon`, trái Ch4 dùng đúng trần
  composition `T epsilon`;
- `:167-173` nói T-REM “lấp” trajectory/correlation guarantee dù chỉ có conditional
  per-step theorem;
- `:251,259-265` gọi nearest-vertex metric là off-road proof và nói cả PR đã memoize;
- simulator/UI hiển thị `epsilon toàn quỹ đạo=n epsilon` đồng dạng cho SM/baseline dù
  SM có revisit infinite-ratio leak còn baseline không có epsilon guarantee hợp lệ;
- `core/mechanisms.py:252-255` nói averaging REM/T-REM hội tụ về `x`, trái phân tích
  bias trong Ch4; `:287-290` gọi private reuse là future work dù PR class nằm ngay sau.

**Required fix**

Tạo machine-readable claim registry hoặc ít nhất một canonical Markdown table với
fields: mechanism, ideal guarantee, executable caveat, state/lifecycle, evaluation
attacker, prohibited claim. Dùng `rg` sweep toàn repo và add regression test cho các
phrase cấm quan trọng (`all mechanisms ... Euclidean Geo-I`, `PR memoization`,
`privacy higher` từ proxy, `off-road` từ vertex metric).

### R4-011 — P1 novelty/literature: overlap 2026 chưa được position

Bài của Simitçioğlu & Gürsoy, *Privacy risks of continuous location sharing under
local differential privacy: Inference attacks and defenses*, Computer Networks 284
(July 2026), Article 112333, DOI `10.1016/j.comnet.2026.112333`, nghiên cứu:

- Bayesian attacks cho stationary/near-stationary users;
- HMM attacks cho mobile users;
- ba defense **Memoization, Replay, Replication**;
- Replay lưu location và perturbed counterpart trong cache;
- Replication chỉ lưu perturbed counterpart trước đó và reuse khi location lặp.

Nguồn chính thức: [ScienceDirect article](https://www.sciencedirect.com/science/article/pii/S1389128626003452).
Đây là overlap trực tiếp về threat split và defense pattern của SM/PR, dù paper dùng
categorical LDP protocols chứ không road-metric Geo-I. Novelty defensible có thể là
road-constrained metric integration/formalization/evaluation, không phải phát minh
memoization/previous-release reuse.

Ngoài ra `docs/problem_formulation.md:82-85` đặt một câu trong ngoặc kép và nói road
PTPPM “tự thừa nhận” stateless repeated-release vulnerability. Trang/source của
[road-network PTPPM](https://arxiv.org/abs/2511.21020) mô tả Permute-and-Flip mỗi
timestamp nhưng không support exact quotation đó. Ch4 đã sửa thành “suy luận của
luận văn”; supporting doc phải đồng bộ, bỏ quote/attribution hoặc dẫn đúng passage.

**Required fix**

- thêm 2026 paper vào related work và bảng comparator dimensions;
- map SM vs Memoization/Replay, PR vs Replication/predictive mechanisms, nêu khác biệt
  metric/output/state/guarantee/threat;
- bỏ “chưa ai làm”, “khoảng trống chưa có công bố chiếm giữ”, “đối thủ gần nhất” nếu
  chưa có systematic-search protocol;
- nếu gọi PTPPM stateless/repeated-release gap, ghi rõ là inference từ algorithm và
  không dùng fabricated quotation;
- comparator executable vẫn PLANNED, không gọi superiority.

### R4-012 — P2 graph recipe: artifact verified, clean build chưa verified

`data/build_beijing_graph.py:1-8` và `data/README.md:19-25` nói script tải source khi
thiếu và verify canonical hash. Thực tế code `:42-47` exit nếu GZip thiếu; phần hash
chỉ tính/ghi hash của bất kỳ local file nào đang có, không so với expected immutable
source hash. URL BBBike rolling và OSMnx dependency broad `>=` tiếp tục làm clean
rebuild không deterministic.

**Required fix**

- hoặc implement atomic download từ dated/immutable URL + expected hash fail-closed;
- hoặc sửa docs thành manual artifact procedure, kèm exact hash và licensing/source;
- pin compatible OSMnx/runtime lock, validate manifest keys, add corrupted-source
  negative test;
- không claim drive-only/ride-hailing behavior tới khi build filtered graph và rerun.

### R4-013 — P2 environment/CI: canonical runner pass nhưng chưa reproducible env

Positive: `tests.run_all` là một command rõ và pass trong existing runtime venv.
Residual:

- exact venv không có pytest dù `requirements-dev.txt` nêu canonical pytest setup;
- system Python thiếu runtime dependencies;
- `requirements.txt` dùng lower bounds, không lock exact resolved versions;
- không có CI workflow;
- không có canonical quick benchmark/provenance schema smoke trong suite;
- response ghi sai 19+10.

**Acceptance**: build clean locked environment, chạy `tests.run_all`, pytest nếu còn
advertise, graph negative tests và quick experiment/provenance smoke; CI lưu exact
commands/versions; response/report count phải là 19 total hoặc số mới thực chạy.

### R4-014 — P2 statistics: đúng là exploratory, cần giữ PARTIAL

Moving benchmark dùng một root seed, 20 trajectories/858 points nhưng chỉ 7 users,
equal-weight trajectories, không multi-seed hoặc user-cluster inference. RNG schema
change đã làm một số estimates dịch đáng kể (ví dụ PR epsilon .01 displacement
`576.1 -> 623.5m`, HMM `324.2 -> 370.7m`), cho thấy Monte Carlo sensitivity có ý
nghĩa.

Averaging bootstrap median cluster theo user được tái tạo đúng, nhưng success
probability vẫn pooled/no cluster CI; duplicate study units làm current interval không
final. Cần:

- nhiều independent root seeds;
- user-cluster CI/sensitivity cho moving benchmark;
- cluster CI cho averaging success;
- khai báo estimand và weighting user/stay/trajectory;
- ablation theta/grid/horizon nếu dùng PR/SM trade-off làm contribution;
- tránh infer population/general deployment từ 7/21 GeoLife users.

### R4-015 — P1 submission QA: PDF visual-main pass, ToC/semantic fail

Hai table captions quá dài làm `Danh sách bảng` tràn/cắt ở physical page 5. Sửa bằng
short optional captions (`\caption[Short caption]{Full caption}`) hoặc dedicated
source/provenance note ngoài caption. Sau khi rerun numbers, render toàn PDF lại và
inspect list-of-tables, Table 5.1/5.2, cross-references và conclusion pages.

Semantic PDF hiện còn chứa R4-001/R4-003/R4-007 claims, nên ngay cả khi layout được
sửa, PDF vẫn chưa submission-ready cho tới khi source narrative được sửa và rebuild.

## 5. Thứ tự sửa khuyến nghị cho Claude agents

### Phase A — dừng propagation của số liệu sai

1. Fix stay identity/dedup + uniqueness assertions (R4-001).
2. Fix common paired jitter streams (R4-002).
3. Fix/rename nearest-edge metric (R4-003).
4. Add raw schema fields/provenance captured pre-run (R4-008).
5. Add targeted tests; chạy test trước khi expensive rerun.

Không cập nhật bảng/PDF trước khi Phase A pass.

### Phase B — formal/executable correctness

1. Rewrite SM theorem and align reset/history with production (R4-004).
2. Validate all PR budgets (R4-005).
3. Replace circular/vacuous tests and add production transcript/lifecycle tests
   (R4-006, R4-009).
4. Fix online HMM normalization/parity (R4-007).

### Phase C — rerun clean package

Khuyến nghị tiếp tục three-commit protocol:

```text
C5 = code/tests/docs/thesis prose, clean and no regenerated outputs
O5 = outputs/tables/PDF generated from clean C5
R5 = response to R4-001...R4-015, pins C5/O5/verifier commit
```

Trước O5, assert working tree clean và capture provenance/start. Sau O5:

- run raw uniqueness/reconciliation scripts;
- compare table cells programmatically;
- render all PDF pages và inspect physical page 5;
- report exact commands, environment lock/hash và result artifact hashes.

### Phase D — manuscript/novelty sweep

1. Apply one claim registry repo-wide (R4-007/R4-010).
2. Add July-2026 Computer Networks work and narrow novelty (R4-011).
3. Correct graph/build/deployment claims (R4-012/R4-014).
4. Use optional short captions and rebuild PDF (R4-015).

## 6. Acceptance checklist cho vòng kế tiếp

Một response mới chỉ được ghi `RESOLVED` khi evidence bên phải cùng tồn tại:

| Finding | Minimum acceptance evidence |
|---|---|
| R4-001 | Unique source-backed stay IDs + unique raw keys + dedup policy + rerun |
| R4-002 | Same noisy inputs across mechanisms + permutation/insertion runner test |
| R4-003 | Point-to-edge metric/test, hoặc all prose renamed to near-vertex without off-road inference |
| R4-004 | Cell metric/event theorem + REM/TREM history alignment + SM lifecycle tests |
| R4-005 | Reject non-positive/non-finite budgets with parameterized tests |
| R4-006 | No tautology/circular oracle; coefficient, branch and transcript mutations caught |
| R4-007 | Exact-scope labels; online/offline logZ parity; no proxy privacy ranking |
| R4-008 | Pre-run provenance + data/env/command hashes + all table cells raw-reconstructable |
| R4-009 | Event inequality handles null events; no unrestricted ratio `0/0` |
| R4-010 | Repo-wide claim sweep + docs/web/PDF consistent with registry |
| R4-011 | 2026 overlap discussed; unsupported quote/absolute novelty removed |
| R4-012 | Pinned verified download/build, or truthful manual procedure + lock |
| R4-013 | Clean locked environment and CI/canonical commands green |
| R4-014 | Multi-seed/user-cluster evidence or conclusions explicitly remain exploratory |
| R4-015 | Full PDF render including list-of-tables passes after semantic rerun |

## 7. Commands/checks đã dùng

Representative verifier commands:

```bash
git status --short --branch
git log --oneline --decorate -8
git diff --check 7cbb042..bc75c14
venv/bin/python -m tests.run_all
venv/bin/python -m pytest -q tests
python3 -m pytest -q
```

Ngoài ra verifier đã:

- parse/reconcile cả hai JSON và audit composite-key uniqueness;
- load raw GeoLife local để so full-precision vs rounded stay identities;
- verify graph hashes/counts/components/extent/highway distribution;
- regenerate full benchmark utility streams;
- independently compute projected nearest-edge distances;
- render toàn bộ PDF và inspect targeted pages/build log;
- audit source/prose theo line và đối chiếu literature primary pages.

## 8. Giới hạn của verification này

- Verifier không rerun toàn bộ expensive attacker suite end-to-end lần thứ hai; đã
  rerun quick path, full utility path và raw aggregate reconciliation.
- Raw GeoLife/graph nằm ngoài Git; audit dùng artifacts local, nên chính limitation
  clean-clone provenance là một finding chứ không được giả vờ đã giải quyết.
- Empirical enumeration/sampling không thay proof của ideal kernels.
- Independent nearest-edge numbers là audit sensitivity chính xác trên regenerated
  C4 Planar streams; official results vẫn phải do project runner mới tạo lại sau fix.
- Literature search không phải systematic review hoàn chỉnh; việc tìm thấy một paper
  overlap trực tiếp đủ để bác bỏ absolute novelty wording, nhưng không chứng minh đã
  liệt kê hết related work.

## 9. Verdict cuối

`bc75c14` **không pass final verification**. Package đáng ghi nhận ở clean commit
lineage, deterministic regeneration, graph integrity, raw aggregate reconciliation và
claim narrowing của PR. Tuy nhiên averaging artifact hiện tại có duplicated study
units/keys và unpaired input noise; on-road narrative dùng sai metric; formal/API/test,
attacker/provenance, repo-wide consistency và novelty positioning đều còn blocker.

Decision: **NEEDS REVISION; sửa C5 rồi rerun O5, không patch số liệu O4 tại chỗ.**
