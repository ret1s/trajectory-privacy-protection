# Verification vòng 4 — audit commit `c5385ec`

## Hồ sơ review

| Trường | Giá trị |
|---|---|
| Branch được review | `verifier` |
| Commit kết quả | `c5385ecaa205aa9ffe9c1db93afc84ca8d86c302` |
| Commit message | `Harden round 3 (R2-001..R2-015): claim-safe path + order-independent rerun` |
| Commit cha | `d0a068a381f219a8755e5d9120b2abf1ebebf634` |
| Ngày review | 2026-08-23 (Asia/Ho_Chi_Minh) |
| Phạm vi | Formal guarantees, production implementation, tests, RNG, graph/data provenance, attackers, experiments, thesis/docs/PDF consistency |
| Kết luận | **NEEDS REVISION — ideal-kernel progress verified; implementation/result package chưa final-verified** |

Tài liệu này là review độc lập, commit-scoped. Mọi line reference trỏ tới nội dung
của `c5385ec`, không phải một working tree đã sửa sau review. Commit phản hồi
`adb9387` không được dùng làm evidence rằng candidate đã tự giải quyết các finding
dưới đây.

Raw graph dùng cho spot-check tồn tại cục bộ tại
`data/raw/beijing_graph.pkl`, SHA-256
`9e44ec2a48d6249f3534bb8511010925428018e528f5116d5f9eb24d86a2d487`.
Raw graph và raw GeoLife không nằm trong Git, nên verifier có thể audit artifact
cục bộ nhưng clean clone chưa thể tái tạo toàn bộ evidence chỉ từ commit.

### Quy ước trạng thái

- **RESOLVED**: root cause, evidence và manuscript đều đạt acceptance criteria.
- **NARROWED**: claim đã được hạ về phạm vi evidence hiện có; capability mạnh hơn
  chưa được hiện thực hoặc chứng minh.
- **PARTIAL**: có tiến bộ thực chất nhưng còn ít nhất một acceptance criterion fail.
- **OPEN/FAIL**: claim hoặc implementation hiện tại bị phản ví dụ trực tiếp.
- **PLANNED**: chỉ được đặt trong future work, chưa được gọi là contribution đã có.

Review tách bốn lớp không được nhập làm một:

1. **Ideal theorem** — kernel real-arithmetic có proof đúng không?
2. **Executable mechanism** — float/RNG/state/code có hiện thực kernel đó không?
3. **Evaluation evidence** — attacker, sampling protocol và statistics có support
   conclusion không?
4. **Artifact provenance** — người khác có xác định và tái chạy đúng source state,
   data, environment và commands không?

Một test toy pass ở lớp 1 không tự động verify lớp 2--4.

## 1. Kết luận điều hành

Commit `c5385ec` có tiến bộ rõ ràng:

- proof PR-SM-REM đã dùng bound đúng
  `(epsilon_test + epsilon_release)` ở ideal-kernel level và xử lý cả case `z=h`;
- runner chính dùng matched hard-step budget `epsilon/2 + epsilon/2`;
- finite-precision limitation của Gumbel-max đã được ghi thẳng thay vì tiếp tục
  claim executable pure Geo-I;
- graph local có bbox/count/hash đúng và toàn bộ Table 5.1/5.2 khớp JSON sau làm
  tròn;
- ba formal regression modules đều chạy pass;
- quick benchmark chạy hết, không có lỗi runtime hoặc giá trị không hữu hạn;
- PDF mới có 31 trang, render sạch, không thấy clipping/overlap nghiêm trọng.

Tuy nhiên candidate chưa thể pass vì bảy blocker load-bearing:

1. proposition fixed-public rejection vẫn thiếu giả thiết base kernel thỏa
   `epsilon`-Geo-I và thiếu indicator `1_A(z)`;
2. RNG gọi là order-independent nhưng seed phụ thuộc ordinal `mi` và `ei`;
3. hai result JSON được sinh từ parent `d0a068a`, trong đó benchmark ghi
   `git_dirty: true`, nên không pin được source state sạch của `c5385ec`;
4. core/doc vẫn overclaim PR-SM-REM "defeats averaging", dù noisy test cho xác suất
   resample dương ở mọi bước hữu hạn;
5. problem formulation còn claim trajectory/budget cho SM-REM trái với chính
   counterexample likelihood-ratio vô hạn trong Ch4;
6. attacker non-REM vẫn là proxy nhưng Ch5 còn suy ra superiority từ proxy error;
7. graph builder/README nói tự tải và verify hash, trong khi code chỉ fail nếu thiếu
   source và chỉ ghi hash của bất kỳ file local nào đang có.

### Release decision

Không nên dùng các câu sau như final thesis claim tại snapshot này:

- "Benchmark seed độc lập với thứ tự chạy."
- "`benchmark_results.json` và `averaging_multi_results.json` là output sạch của
  commit `c5385ec`."
- "Proposition fixed-public conditioning đã được chứng minh như phát biểu."
- "PR-SM-REM đánh bại averaging" hoặc "static user luôn reuse release đầu".
- "SM-REM tốn budget đúng một lần mỗi distinct cell trên arbitrary trace."
- "SM-REM/PR-SM-REM riêng tư hơn đáng kể" dựa trên REM-emission proxy.
- "Script graph tự tải và verify canonical BBBike extract."
- "SM-REM giữ 150.5m ở mọi `n`."
- "Simulator đánh giá truy vấn ba hiệu thuốc thật."

Framing an toàn cho candidate:

> `c5385ec` là một industry-aware research prototype. Ideal REM/PR kernels có
> single-step metric-privacy arguments đã được thu hẹp hợp lý. Executable sampler
> vẫn chỉ là finite-precision approximation; SM-REM chỉ có static same-cell/cache-
> lifetime anti-averaging property; PR-SM-REM là previous-release predictive reuse
> có per-step ideal bound và finite-horizon empirical mitigation. Mechanism-aware
> attacks, clean provenance, mobile implementation và non-trivial window accountant
> vẫn là phần chưa hoàn tất.

## 2. Status matrix của R2-001...R2-015

| ID | Response tại `c5385ec` | Verified status | Kết luận ngắn |
|---|---|---|---|
| R2-001 | RESOLVED ideal | **RESOLVED for ideal kernel** | PR one-step bound và case `z=h` đúng; production-coupled test còn thiếu |
| R2-002 | PARTIAL/honest | **NARROWED / PARTIAL** | Chưa có window budget manager; trivial composition được ghi tương đối đúng |
| R2-003 | PARTIAL improved | **PARTIAL** | Runners split `eps/2 + eps/2` đúng; API dễ dùng thành `2eps`, chưa có sweep/assertion |
| R2-004 | PARTIAL | **PARTIAL — blocker** | Label proxy tốt hơn; HMM/PR/SM attacks vẫn misspecified, manuscript còn superiority claim |
| R2-005 | PARTIAL improved | **PARTIAL — blocker** | Artifact local đúng; clean-clone build/download/hash verification fail |
| R2-006 | RESOLVED claim-safe | **NARROWED** | Pure Geo-I chỉ claim ideal; executable zero-support được document, chưa fix |
| R2-007 | PARTIAL | **PARTIAL** | Ch4 nói previous-release reuse đúng; docs/core vẫn nhập nhằng memoization |
| R2-008 | PARTIAL | **PARTIAL** | Ch5 caveat stay-point tốt; runner vẫn gọi `home`, population không representative |
| R2-009 | PARTIAL improved | **FAIL** | Seed dùng list index nên không order-independent; one seed/no CI/raw rows |
| R2-010 | claimed resolved | **OPEN — broad blocker** | README, problem formulation, Ch1/Ch4/Ch5 và simulator còn contradiction |
| R2-011 | claimed resolved | **FAIL — formal blocker** | Test đúng hơn statement; proposition thiếu assumptions và support mask |
| R2-012 | PLANNED | **PLANNED** | Thesis đã scope server artifact/on-device target; implementation mobile chưa có |
| R2-013 | PARTIAL | **FAIL/PARTIAL** | Output mang dirty parent commit; dataset/env/raw-row provenance thiếu |
| R2-014 | claimed resolved | **OPEN/PARTIAL** | Eclipse/RAPPOR chỉ prose; chưa có aligned executable comparator |
| R2-015 | PARTIAL | **PARTIAL/FAIL narrative** | Tables/PDF render pass; claim số liệu và POI narrative còn sai |

Không được suy ra "overall verified" từ việc R2-001, budget wiring, graph hash và
formal toy tests pass.

## 3. Findings mới cần response theo ID R3

| ID | Severity | Finding | Action class |
|---|---|---|---|
| R3-001 | P0 / Critical | Proposition fixed-public conditioning sai như phát biểu | Sửa theorem + test statement |
| R3-002 | P1 / High | Seed derivation phụ thuộc mechanism/epsilon ordinal | Sửa implementation + permutation test |
| R3-003 | P1 / High | Result artifacts không pin clean source state | Rerun sạch + provenance schema |
| R3-004 | P1 / High | PR overclaim anti-averaging asymptotic | Sửa docstring/thesis + horizon analysis |
| R3-005 | P1 / High | SM guarantee/budget contradiction và sai factor biên grid | Sửa formal scope repo-wide |
| R3-006 | P1 / High | Formal tests không gọi production mechanisms | Thêm implementation-coupled tests |
| R3-007 | P1 / High | Attacker proxy không support comparative privacy claims | Implement attackers hoặc hạ conclusion |
| R3-008 | P1 / High | Graph recipe mô tả sai, source/build chưa immutable | Fix downloader/hash/profile/lock |
| R3-009 | P1 / High | README legacy pipeline claim formal guarantee sai | Tách legacy/demo khỏi thesis method |
| R3-010 | P1 / High | Moving/stay experiments chưa đủ statistical evidence | Multi-seed/user-cluster/raw tidy |
| R3-011 | P1 / High | Thesis/PDF chứa numerical và use-case claims sai | Repo-wide semantic cleanup |
| R3-012 | P2 / Medium | PR API budget footgun và state wording quá rộng | Redesign constructor + clarify state |
| R3-013 | P2 / Medium | Road graph profile chứa pedestrian/cycle edges | Define target network profile |
| R3-014 | P2 / Medium | Không có repo-wide green test command/environment | Dev dependencies + CI + warnings |
| R3-015 | P2 / Medium | Chưa có aligned comparator/ablation cho novelty | Add baselines hoặc narrow novelty |

## 4. Scope, evidence và methodology

### 4.1 Commit inventory

Diff `d0a068a..c5385ec` thay đổi 23 files, 1,107 insertions và 530 deletions:

- mechanism code, attacker code và two experiment runners;
- graph builder, graph manifest và data README;
- three formal test modules;
- two result JSONs;
- README và four supporting research documents;
- Ch1, Ch4, Ch5, Ch6 và `thesis/main.pdf`.

Static checks:

- `git diff --check c5385ec^ c5385ec`: pass;
- changed Python files compile trong available Python environment: pass;
- committed JSON files parse: pass;
- local graph/source hashes khớp manifest: pass.

### 4.2 Formal regression execution

Các command theo module chạy trong project environment:

```bash
venv/bin/python -m tests.test_pr_privacy_bound
venv/bin/python -m tests.test_rejection_conditioning
venv/bin/python -m tests.test_sampler_support
```

Kết quả:

- PR analytic example: ratio `4.4817`, vượt bound cũ `e^1 = 2.7183` và thỏa bound
  mới `e^2 = 7.3891`;
- 2,000 random PR checks: worst normalized ratio khoảng `0.9834`, không thấy
  counterexample cho ideal one-step bound;
- fixed-public conditioning toy: ratio `4.2 > e^epsilon = 3` và
  `4.2 <= e^(2epsilon) = 9`;
- secret-dependent accept set: ratio vô hạn;
- Gumbel float witness: observed span `17.1098 <= 40.3406`; logit gap `45` không
  thắng trong 200,000 draws dù ideal probability `2.863e-20 > 0`.

`python3 -m pytest -q tests` chạy `7 passed`, kèm bốn
`PytestReturnNotNoneWarning` vì một số test functions trả value thay vì chỉ assert.
Repo-wide `python3 -m pytest -q` fail lúc collection tại `legacy/test_app.py` vì
system environment thiếu Flask. Exact project venv có dependencies runtime nhưng
không cài pytest. Vì vậy chưa tồn tại một canonical green test command.

### 4.3 Experiment spot-check

Verifier chạy quick benchmark trên clean archive của `c5385ec` với raw data được
link vào read-only. Sáu mechanisms đều hoàn tất ở `epsilon=0.02`:

| Mechanism | Mean displacement (m) | Bayes/proxy error (m) | HMM/proxy error (m) |
|---|---:|---:|---:|
| Planar Laplace | 94.27 | 96.18 | 77.15 |
| Baseline TT2 surrogate | 77.10 | 77.43 | 72.07 |
| REM | 171.34 | 168.98 | 122.13 |
| T-REM | 191.91 | 191.95 | 127.42 |
| SM-REM | 190.59 | 189.90 | 149.08 |
| PR-SM-REM | 257.37 | 259.74 | 147.76 |

Independent regeneration của utility path cho toàn bộ 18 official benchmark rows
khớp committed `mean_disp` và `qos` với max absolute difference `0`. Điều này là
positive evidence cho deterministic utility generation; nó không sửa stale/dirty
provenance và không independently reproduce toàn bộ expensive attack/averaging
artifacts.

### 4.4 Graph artifact

Local artifact khớp manifest:

| Thuộc tính | Giá trị |
|---|---:|
| GZip SHA-256 | `ed0af579fb32c7164de85bca945a10432f818dd27eb80d6d5a7afa62213b0407` |
| XML SHA-256 | `5691186b8b3bd9078915faefb7fb639ff2c7f45b4e8b1efd1de4b9ebdad4c8a5` |
| Graph SHA-256 | `9e44ec2a48d6249f3534bb8511010925428018e528f5116d5f9eb24d86a2d487` |
| Nodes / edges | 13,813 / 41,040 |
| Weak components | 1 |
| Latitude extent | 39.960011--40.019992 |
| Longitude extent | 116.290003--116.359997 |

Đây là verification của local artifact, không phải verification rằng current
rolling BBBike URL hôm nay sẽ trả đúng source bytes đó.

### 4.5 Output/PDF QA

- Table 5.1 khớp 18 benchmark rows sau rounding.
- Table 5.2 khớp averaging JSON sau rounding.
- `thesis/main.pdf`: 31 trang A4, fonts embedded, không thấy undefined `??`, bảng
  không clip/overlap trong full-document render.
- Visual QA pass nhưng semantic QA fail vì các claim R3-007/R3-011 đã được render
  nguyên trạng.

### 4.6 Giới hạn của verifier

- Full expensive attacker/averaging suite không được rerun end-to-end lần thứ hai;
  utility/QoS được regenerated toàn bộ, attack paths được smoke/spot-check.
- Raw data bị gitignore nên clean archive vẫn cần local raw artifacts.
- Toy enumeration support ideal theorem, không phải exhaustive proof của float
  production implementation.
- Không có mobile device benchmark; mọi on-device feasibility claim vẫn cần evidence
  riêng.

## 5. Findings chi tiết và suggest improvements

### R3-001 — P0: proposition fixed-public conditioning sai như phát biểu

**Evidence:**

- `thesis/chapters/ch4_phuongphap.tex:98-106`;
- `tests/test_rejection_conditioning.py:6-9,80-85`.

Proposition hiện viết

```text
K_A(z|x) = K(z|x) / K(A|x)
```

nhưng:

1. không giả sử base kernel `K` thỏa `epsilon`-Geo-I;
2. không giới hạn `z in A` hoặc nhân `1_A(z)`;
3. vì vậy công thức không định nghĩa một distribution đúng trên toàn output space.

Hai counterexample tối thiểu:

- nếu không có Geo-I assumption, chọn
  `K(.|a)=delta_0`, `K(.|b)=delta_1`, `A={0,1}`; acceptance dương nhưng ratio vô hạn;
- nếu thiếu indicator, chọn `K=(1/2,1/2)`, `A={0}`; công thức hiện tại cho mass
  `(1,1)`, tổng bằng 2.

**Required fix:**

Phát biểu rõ:

\[
K_A(z\mid x)=\frac{\mathbf 1_A(z)K(z\mid x)}{K(A\mid x)},
\qquad K(A\mid x)>0,
\]

với `K` thỏa `epsilon`-Geo-I trên cùng metric space. Kết luận có thể phát biểu cho
mọi `z in A`, hoặc tốt hơn cho mọi measurable event `S`:

\[
K_A(S\mid x)\le e^{2\epsilon d(x,x')}K_A(S\mid x').
\]

Giữ riêng ba cases:

- fixed public `A`: generic safe upper bound `2epsilon`;
- secret-dependent `A_x`: có thể infinity;
- finite retry + fallback: kernel khác, phải phân tích riêng.

**Acceptance:**

- theorem có đủ assumption và support mask;
- test có negative regression cho missing-assumption counterexample;
- test function chỉ assert, không return;
- response không được mark R2-011/R3-001 resolved chỉ vì toy array cũ vẫn pass.

### R3-002 — P1: RNG chưa order-independent

**Evidence:** `experiments/run_benchmark.py:98-111` và claim tại
`thesis/chapters/ch5_thucnghiem.tex:52-54`.

Current seed:

```python
ei = EPSILONS.index(eps)
for mi, mech in enumerate(build_mechanisms(...)):
    mech.rng = default_rng(SeedSequence([SEED, ei, mi]))
```

`mi` là vị trí mechanism, không phải identity; `ei` là vị trí epsilon, không phải
giá trị epsilon. Đảo mechanism list đã làm ba REM releases đầu đổi hoàn toàn. Thêm
một mechanism trước REM hoặc reorder `EPSILONS` cũng đổi result.

**Required fix:**

- derive seed từ stable semantic key `(schema_version, root_seed, exact epsilon,
  mechanism_name[, trajectory_id])`;
- không dùng built-in `hash()` vì Python hash randomization;
- dùng explicit integer registry hoặc SHA-256/BLAKE2 digest với byte encoding được
  version hóa;
- truyền independent RNG ngay lúc constructor thay vì tạo mọi mechanism bằng shared
  RNG rồi overwrite sau;
- ghi `rng_schema`, root seed và resolved seed/substream ID vào provenance.

Ví dụ design:

```python
payload = f"rng-v1|{root_seed}|{eps:.17g}|{mechanism_name}".encode()
words = np.frombuffer(hashlib.sha256(payload).digest()[:16], dtype="<u4")
rng = np.random.default_rng(np.random.SeedSequence(words.tolist()))
```

**Acceptance:**

- permutation test đảo mechanisms và epsilons cho output per-key byte-identical;
- test thêm một unrelated mechanism nhưng existing mechanisms không đổi;
- test chạy mechanism subset nhưng common keys vẫn không đổi;
- thesis chỉ gọi order-independent sau khi tests trên pass.

### R3-003 — P1: result provenance không pin clean source state

**Evidence:**

- `outputs/benchmark_results.json:3-4` ghi
  `git_commit=d0a068a...`, `git_dirty=true`;
- `outputs/averaging_multi_results.json:3` ghi cùng parent và không có dirty flag;
- `experiments/run_benchmark.py:170-194`;
- `experiments/run_averaging_multi.py:138-162`.

Committed outputs có thể đã được tạo từ working tree chứa patch tương lai của
`c5385ec`, nhưng artifact không lưu patch đó. Vì vậy người khác không thể chứng
minh source state tạo output bằng commit hash được ghi.

Provenance còn thiếu:

- hashes/IDs của selected GeoLife files, user IDs và trajectory/stay IDs;
- command, timestamp, quick/full mode;
- Python và complete dependency lock;
- RNG derivation schema;
- raw per-trajectory/per-location rows;
- averaging dirty state, bootstrap seed và number of draws;
- semantic graph hash checked at load time.

**Required fix — two/three-commit workflow:**

1. Commit `C`: code/tests/docs setup, no regenerated outputs; tree clean.
2. Chạy experiments từ `C`; provenance phải ghi `source_commit=C` và
   `source_dirty=false` trước khi output write.
3. Commit `O`: chỉ raw/aggregate outputs + updated tables/PDF.
4. Commit `R`: response file pin cả `C`, `O` và verifier target.

Không regenerate output từ một dirty checkout rồi lấy future commit hash bằng mô tả
thủ công.

**Suggested provenance schema:**

```json
{
  "schema": "msc-experiment-v1",
  "source_commit": "...",
  "source_dirty_before_run": false,
  "command": ["python", "-m", "experiments.run_benchmark"],
  "started_at_utc": "...",
  "python": "3.11.12",
  "environment_lock_sha256": "...",
  "graph_sha256": "...",
  "dataset_files_sha256": {"...": "..."},
  "selected_record_ids": ["..."],
  "rng_schema": "sha256-semantic-key-v1",
  "root_seeds": [42]
}
```

**Acceptance:**

- both JSONs report a real clean source commit containing the executed code;
- averaging records dirty state too;
- clean-clone run with pinned raw artifacts reproduces aggregate JSON or a documented
  tolerance-normalized comparison;
- raw tidy evidence makes every table cell independently recomputable.

### R3-004 — P1: PR-SM-REM overclaims asymptotic anti-averaging

**Evidence:** `core/mechanisms.py:354-401`, especially `:396-398`; Ch5
`thesis/chapters/ch5_thucnghiem.tex:121-130,198-202`.

For finite `theta`, `epsilon_test` and distance `d`, noisy test gives

\[
0 < q_x(h)=F_{Lap}(\theta-d(x,h)) < 1.
\]

Static input therefore vẫn có positive resample probability mỗi step. Khi
`d(x,h)=theta`, `q=1/2`; probability không resample trong 100 later steps là
`2^-100`. Ngược lại, moving point ngoài threshold vẫn có positive reuse probability.

PR làm giảm số fresh samples hữu hiệu trong finite horizon; nó không memoize tuyệt
đối và không làm averaging impossible asymptotically.

**Required fix:**

- đổi docstring thành "partially mitigates averaging over the evaluated horizon";
- nói rõ cả reuse/resample branches đều probabilistic;
- không nhập PR vào cùng category "static exact-repeat memoization" của SM;
- báo `n_resample`, effective unique releases và inference error theo horizon dài;
- thêm analytic/empirical curve cho `T -> 10^k` hoặc bound expected resamples;
- nếu muốn persistent anti-averaging, thiết kế lifecycle/memo state khác và phân tích
  privacy của revisit channel, không gọi current PR là persistent memoization.

**Acceptance:**

- không còn chuỗi "static user reuses the first release so averaging is defeated";
- unit test xác nhận cả branches có nonzero probability trong representative cases;
- conclusion phân biệt exact SM protection, PR finite-horizon mitigation và
  asymptotic behavior.

### R3-005 — P1: SM guarantee contradiction và grid-bound thiếu `sqrt(2)`

**Evidence:**

- correct caveat: `core/mechanisms.py:275-291` và
  `thesis/chapters/ch4_phuongphap.tex:292-315`;
- contradictory claims: `docs/problem_formulation.md:46-49,64-73`;
- quantization implementation: `core/mechanisms.py:317-348`.

`problem_formulation.md` nói budget cộng một lần mỗi distinct cell và trigger
memoization "an toàn, không rò rỉ". Ch4 lại đúng khi cho counterexample
`X=(a,a)`, `X'=(a,b)`, event `{z2 != z1}` có ratio vô hạn. Không thể giữ cả hai.

Với square grid cạnh `g`, representatives của hai diagonal cells chạm tại corner
cách nhau `sqrt(2)g`. Bound adjacent-cell safe là

\[
e^{\epsilon d(rep(C),rep(C'))}\le e^{\epsilon\sqrt{2}g},
\]

không phải `e^(epsilon*g)` như docstring/Ch4 hiện tại.

Static theorem cũng nên nói rõ initial cache/history. `Z ~ REM(rep(C))` và transcript
`(Z,...,Z)` chỉ đúng khi cell chưa có memo entry ở đầu scope hoặc khi conditioning
trên một common public reset/cache state thích hợp.

**Required fix:**

- xóa arbitrary-trace budget claim khỏi problem formulation;
- scope theorem: one cell, one cache epoch, empty/common initial state, exact repeated
  cell key, cache persistence/lifecycle được định nghĩa;
- sửa grid bound thành actual representative distance; tốt hơn, phát biểu trực tiếp
  bằng cell pseudometric thay vì một scalar cell width thiếu điều kiện;
- phân biệt input points qua edge và qua corner;
- định nghĩa cache reset, principal, app/site partition, TTL và whether cache content
  survives sessions.

**Acceptance:**

- repo-wide không còn claim `O(#distinct cells)*epsilon` cho arbitrary traces;
- grid bound có proof/test geometry cho edge/corner;
- theorem static ghi initial-state assumptions;
- A-B-A/cross-session regression minh họa đúng behavior và limitation.

### R3-006 — P1: formal tests chưa ràng buộc production code

**Evidence:**

- `tests/test_pr_privacy_bound.py:25-38` dựng toy kernel riêng;
- `tests/test_sampler_support.py:26-47` gọi NumPy trực tiếp;
- rejection test chỉ dùng toy arrays;
- không test nào import `PrivateReuseSMREM`, `RoadExponential` hoặc production
  `_sample` path.

Đổi dấu threshold, sai scale, quên `eps/2`, thay state transition hoặc sửa sampler
sai vẫn có thể để cả 7 tests pass.

**Required tests:**

1. Fake two-/three-vertex `RoadNetwork` gọi trực tiếp `PrivateReuseSMREM`.
2. Exhaustive one-step kernel/transcript test cho `z=h` và `z!=h`.
3. Two-step public-history composition test qua production state transition.
4. Direct `RoadExponential._sample` support witness với two secrets/logit vectors.
5. Direct finite-Laplace threshold probabilities cho reuse/resample.
6. Budget wiring assertion cho every runner/factory/simulator.
7. Permutation RNG regression của R3-002.

Tests cần kiểm behavior, không chỉ copy lại formula từ theorem. Nếu cần exact
probabilities, inject deterministic/mock randomness hoặc expose a pure kernel helper
được production `perturb()` dùng chung.

**Acceptance:**

- mutation test đổi một trong sign/scale/budget/state làm ít nhất một test fail;
- tests import production classes;
- no test returns a non-`None` value;
- canonical test command chạy clean trong declared environment.

### R3-007 — P1: attacker proxy không support comparative privacy claims

**Evidence:**

- `evaluation/attacks.py:150-232,277-294`;
- `experiments/run_benchmark.py:59-79,113-120`;
- `web/simulator.py:108-109`;
- `thesis/chapters/ch5_thucnghiem.tex:38-46,91-130,185-214`.

Các hạn chế còn lại:

- HMM report dùng posterior mean trong khi metric là Euclidean distance; posterior
  mean không phải Bayes action cho Euclidean norm — geometric median mới phù hợp;
- benchmark HMM subtracts input-dependent `logZ`, nhưng online simulator path bỏ nó;
- averaging MLE chỉ search vertices trong 1,200m quanh centroid; likelihood family
  đúng cho ideal iid REM nhưng global maximization chưa được đảm bảo;
- T-REM, SM-REM và PR-SM-REM có history/state-dependent kernels; REM-emission proxy
  không phải optimal/mechanism-aware attacker của chúng;
- GPS jitter chưa được marginalize trong likelihood;
- observed proxy error là upper bound trên error của optimal attacker, nên error cao
  không chứng minh mechanism mạnh hơn.

Ch5 vẫn gọi `SM-REM` "duy nhất thật sự" và PR "tốt hơn REM đáng kể" trước khi caveat
rằng attacker PR/SM chưa đúng family. Kết luận superiority không được support.

**Option A — minimum claim-safe:**

- đổi toàn bộ kết luận thành descriptive proxy diagnostics;
- không dùng proxy columns để rank privacy giữa non-REM mechanisms;
- mô tả HMM estimator chính xác là posterior mean nếu chưa đổi loss/action;
- dùng wording "under this fixed REM-emission proxy" tại mọi comparison.

**Option B — stronger evidence:**

- T-REM likelihood: include public previous release/time, temporal weight và
  step-specific normalizer;
- SM likelihood: model cache state/equality/revisit transcript;
- PR likelihood: sequential mixture kernel gồm reuse mass và fresh REM branch;
- jitter-aware model: marginalize latent true GPS observation or use documented
  Monte Carlo/quadrature approximation;
- use weighted geometric median for Euclidean Bayes loss;
- make simulator and offline evaluator share the same normalized emission code;
- verify restricted MLE against full-`V` argmax or remove unproven truncation.

**Acceptance:**

- no "optimal/exact" label beyond the exact family/search conditions;
- no superiority claim for non-REM from REM proxy alone;
- synthetic tests recover known secrets under each mechanism-specific kernel;
- simulator/offline likelihood parity test;
- attacker ablation/convergence documented.

### R3-008 — P1: graph recipe không download/verify canonical source

**Evidence:**

- `data/build_beijing_graph.py:1-8` nói downloads-if-absent và deterministic;
- actual `data/build_beijing_graph.py:42-47` exits nếu `.gz` thiếu;
- `data/build_beijing_graph.py:63-80` tính hash nhưng không compare expected hash;
- `data/README.md:19-25` nói script "tải, verify hash";
- `requirements.txt:1-11` dùng broad lower bounds.

Current BBBike URL là rolling extract. Cùng URL ở thời điểm khác có thể cho graph
khác. Manifest ghi hash của whatever local file exists, không chứng minh file đó là
canonical source đã được review. Pickle bytes cũng có thể khác giữa dependency
versions dù semantic graph gần giống.

**Required fix:**

- script thực sự download atomically nếu absent, hoặc README nói rõ manual step;
- hard-code/load an expected source hash và fail closed khi mismatch;
- pin immutable dated source/artifact URL, release asset hoặc repository with DOI;
- pin exact environment (`==`, lockfile/container digest), không chỉ `>=`;
- check graph hash/count/extent at runtime before official experiment;
- consider canonical GraphML/Parquet edge/node artifact plus semantic sorted hash
  thay vì chỉ pickle hash;
- make manifest generated format byte-identical with builder output.

**Acceptance:**

- clean directory run downloads exact expected bytes or emits one unambiguous
  documented command;
- altered source byte causes build to fail before graph generation;
- pinned environment rebuild passes semantic graph checks;
- official experiment refuses a graph not matching declared manifest.

### R3-009 — P1: README legacy pipeline overclaims formal privacy

**Evidence:**

- `README.md:64` chạy old `experiments.run_averaging` thay vì multi-stay evidence;
- `README.md:78-118` gọi legacy web apps cùng core algorithm và nói
  `trajectory_privacy.py` cung cấp formal privacy cùng road/building/QoS constraints;
- `core/trajectory_privacy.py:126-145,180-184,226-232` dùng secret-dependent
  constraints/rejection;
- Ch1/Ch4 đã giải thích capped/rejected baseline không có claimed guarantee.

Repo hiện có ít nhất hai method families:

1. legacy internship/demo pipeline;
2. thesis REM/T-REM/SM-REM/PR mechanisms và experiment runners.

README nhập hai family thành một, khiến user chạy sai artifact và hiểu sai guarantee.

**Required fix:**

- đặt một authoritative "Thesis artifact" section với exact commands;
- relabel legacy apps là demo/surrogate, no formal guarantee;
- update averaging command thành `experiments.run_averaging_multi`;
- không nói legacy core "matches Algorithm 1" nếu algorithm/current thesis khác;
- table mapping mechanism -> code path -> theorem scope -> executable caveat ->
  official evaluation;
- add `make verify`/script cho canonical checks.

**Acceptance:**

- một reader mới không thể nhầm legacy pipeline với formally analyzed mechanisms;
- all README commands exist and generate the artifacts named in Ch5;
- formal claim links tới đúng implementation/test/theorem.

### R3-010 — P1: statistical evidence chưa đủ cho population/comparison claims

**Evidence:**

- `experiments/run_benchmark.py:82-165`;
- `data/geolife.py:43-50,101-133,166-200`;
- `experiments/run_averaging_multi.py:64-135`.

Moving benchmark:

- one root seed `42`;
- 20 first-qualifying trajectories from only 7 users;
- trajectories aggregate equally, dù nhiều trajectories thuộc cùng user;
- no multi-seed CI, user-cluster CI hoặc raw rows.

Stay experiment:

- 40 first-qualifying stay-points / 21 users, max two per user;
- không phải ground-truth residence/home;
- median CI cluster-resamples users — positive;
- success probabilities pool 320 indicators without user-cluster CI;
- raw rows bị discard;
- proxy attacker limitation vẫn load-bearing.

**Required fix:**

- predefine sampling population và keep stable record IDs;
- multi-seed moving evaluation, cluster bootstrap by user;
- report median/mean with CI and distribution, not only one realization;
- cluster CI for success probabilities;
- write tidy raw rows with user/trajectory/location IDs pseudonymized;
- call points `stay_points` unless residence ground truth/filter exists;
- add budget-split × `theta` × horizon sensitivity;
- separate exploratory, confirmatory and robustness results.

**Acceptance:**

- every headline comparison has uncertainty over appropriate independent units;
- no home/residence generalization from unlabeled stay-points;
- table cells reproducible from committed raw rows;
- result remains qualitatively stable across seeds/users or conclusion is narrowed.

### R3-011 — P1: numerical/use-case/semantic claims còn sai trong PDF

**Evidence examples:**

- `thesis/chapters/ch1_gioithieu.tex:72-74` vẫn gọi evaluator "Bayes tối ưu";
- `thesis/chapters/ch4_phuongphap.tex:178-194` overclaim on-road removes attack
  information và arithmetic mean convergence on bounded road support;
- `thesis/chapters/ch4_phuongphap.tex:318` heading gọi "đảm bảo w-event" dù
  `:387-398` nói chưa có manager;
- `thesis/chapters/ch4_phuongphap.tex:421-425` nói sub-millisecond mà không có
  isolated/mobile measurement;
- `thesis/chapters/ch5_thucnghiem.tex:195-196` nói SM-REM "150.5m ở mọi n", trong
  khi Table 5.2 là khoảng 147.1--164.5m;
- `thesis/chapters/ch5_thucnghiem.tex:223-225,237-241` gọi POIs là ba hiệu thuốc
  và claim recall khoảng 81%, nhưng simulator dùng synthetic generic POIs;
- `docs/problem_formulation.md:46-49,64-86` gộp PR với exact memoization và claim
  novelty tuyệt đối;
- `docs/system_model_and_threats.md:202-208,287` còn graph size cũ;
- `core/mechanisms.py:147-149` còn comment `|V|≈78k`.

**Required fix:**

- designate one claim registry/source-of-truth table:
  `claim -> formal scope -> code -> evidence -> limitation -> allowed wording`;
- repo-wide search và sửa các token stale:
  `optimal`, `exact`, `memoize`, `w-event`, `150.5`, `78k`, `77,727`,
  `sub-millisecond`, `pharmacy`, `home`, `order-independent`, `RESOLVED`;
- distinguish structural on-road property from empirical attack resistance;
- distinguish arithmetic sample mean from mechanism-aware MLE consistency;
- remove unsupported exact recall/use-case labels or add real POI dataset/protocol;
- rebuild PDF and inspect rendered pages, not only LaTeX source.

**Acceptance:**

- no load-bearing contradiction across README/docs/code/thesis/PDF;
- every headline number maps to a JSON/raw field and exact protocol;
- visual and semantic PDF QA both pass;
- response document does not mark a finding resolved while cited source remains
  contradictory.

### R3-012 — P2: PR API budget footgun và privacy-state wording

**Evidence:** `core/mechanisms.py:381-409,412-437` và Ch4 `:336-337`.

Constructor mặc định:

```python
PrivateReuseSMREM(epsilon, ..., eps_test=None)
# eps_test defaults to epsilon
```

Natural call `PrivateReuseSMREM(E, rn)` có release budget `E` cộng test budget `E`,
tức hard-step cap `2E`, trong khi caller có thể hiểu `E` là total per-step epsilon.
Official runners split đúng nhưng API không enforce semantics.

Ngoài ra predictor state `_prev_choice/_prev_release_xy` là function của public
history, nhưng toàn executable state không hoàn toàn public-history-only:

- `n_resample` phụ thuộc hidden branch;
- RNG consumes different numbers of draws theo branch;
- cùng public release có thể đến từ reuse hoặc reject+REM collision.

Ideal proof vẫn dùng fresh independent randomness và private counters không affect
output, nhưng wording cần nói "privacy-relevant predictor state", không phải mọi
internal state.

**Required fix:**

- redesign constructor thành keyword-only
  `epsilon_release`, `epsilon_test`, hoặc nhận `epsilon_step_cap` + split;
- expose `privacy_cost_per_step_max` và assert official comparator equality;
- document counters là diagnostic, not released/not used for decisions;
- use branch-independent random substreams if executable state reasoning cần đơn
  giản và reproducible hơn;
- condition theorem on same public history and fresh independent randomness.

**Acceptance:**

- impossible to accidentally label `2E` mechanism as total budget `E`;
- test all official factories/simulator budget equality;
- theorem/docstring only claim properties of privacy-relevant state.

### R3-013 — P2: road profile chưa phù hợp mọi industry scenario

**Evidence:** `data/build_beijing_graph.py:52-55` gọi `graph_from_xml` mà không có
network filter; `core/road_network.py:39-60` dùng mọi vertex làm candidate.

Audit local graph thấy nhiều edges thuộc pedestrian/non-driving profiles, gồm khoảng:

- 11,754 `footway`;
- 1,285 `cycleway`;
- 652 `path`;
- 616 `steps`.

Điều này không nhất thiết sai cho pedestrian/sport use case, nhưng không support
unqualified drive/ride-hailing realism. Một candidate set chung cũng có thể trả
release lên steps/path cho vehicle scenario.

**Required fix:**

- define target service mode: pedestrian, bicycle, drive hoặc multimodal;
- build/filter graph theo declared OSM highway profile;
- nếu hỗ trợ nhiều scenario, version separate candidate graphs/profiles;
- report edge-class distribution và disconnected-component policy;
- evaluate on-road validity theo point-to-edge/profile, không chỉ nearest vertex.

**Acceptance:**

- manifest lưu network profile/filter;
- use-case claims khớp profile;
- tests reject excluded edge classes cho each profile.

### R3-014 — P2: chưa có canonical green test environment

**Evidence:** targeted test results trong Mục 4.2, `requirements.txt:1-11`.

Runtime dependencies dùng broad `>=`; pytest không nằm trong exact venv. System
pytest collection lại phụ thuộc Flask ngoài test scope. Bốn tests trả non-`None`
values tạo warnings và có thể thành errors ở future pytest.

**Required fix:**

- add `pyproject.toml`/dev requirements với exact tested versions;
- define canonical command, ví dụ `python -m pytest -q tests`;
- configure or move `legacy/test_app.py` để repo-wide collection predictable;
- replace return values bằng asserts/logging;
- add CI jobs: unit/formal, quick experiment, provenance/schema, LaTeX build;
- preserve expensive full experiments as manual/release job with cached raw data.

**Acceptance:**

- fresh declared environment runs canonical tests green with zero warnings;
- CI executes the same command;
- quick benchmark smoke test validates schema and finite outputs.

### R3-015 — P2: comparator/ablation chưa đủ cho novelty và industry positioning

**Evidence:** `experiments/run_benchmark.py:42-52` chỉ có Planar, legacy surrogate,
REM, T-REM, SM-REM và PR-SM-REM; Eclipse/RAPPOR là prose ở Ch5.

Không bắt buộc implement mọi paper để hoàn thành luận văn, nhưng không được nói khe
hở "chưa ai chiếm" hoặc superiority nếu chưa có systematic literature/experimental
comparison. Memoization/replay/replication defenses trong repeated LDP/location
sharing là vùng overlap gần, dù metric/road setting có thể khác.

**Required options:**

- implement one aligned long-term-observation baseline nếu feasible; hoặc
- build a precise comparator table với dimensions:
  threat, domain, adjacency/metric, state, memo lifecycle, output support,
  formal guarantee, attacker, data và utility;
- add ablations: REM vs T weighting, exact memo vs noisy reuse, matched-budget split,
  `theta`, grid, horizon;
- state novelty narrowly as integration/evaluation in road-constrained metric setting,
  not invention of memoization or predictive reuse.

**Acceptance:**

- novelty sentence survives comparison table without absolute unsupported wording;
- each experimental advantage has aligned budget/threat/protocol;
- missing executable comparator is explicitly listed as limitation if not added.

## 6. Positive verification summary

Các điểm sau có thể giữ trong response, với đúng qualifier:

1. **PR ideal one-step proof:** pass cho same public history và ideal randomness.
2. **Matched budget in official runners:** pass, `eps_test=eps_release=eps/2`.
3. **Finite-precision honesty:** pass ở mức claim narrowing; executable pure Geo-I
   vẫn không được claim.
4. **Local graph geometry/hash:** pass cho artifact đang có.
5. **Utility numerical replay:** pass cho 18 rows `mean_disp` và `qos`.
6. **JSON/Table rounding:** pass.
7. **PDF visual layout:** pass.
8. **Runtime smoke:** quick benchmark pass.

Các positive points này không override R3-001...R3-015.

## 7. Acceptance gates

### G0 — Formal statements: **FAIL**

Pass khi:

- R3-001 theorem được sửa đầy đủ;
- SM grid/cell scope R3-005 nhất quán;
- all formal claims state ideal/executable scope explicitly.

### G1 — Production implementation/tests: **FAIL**

Pass khi:

- production-coupled privacy tests R3-006 green;
- RNG semantic-key permutation tests green;
- budget API cannot be misused silently;
- canonical test environment green, zero warnings.

### G2 — Reproducibility/provenance: **FAIL**

Pass khi:

- official outputs record clean source commit;
- graph/data/dependency/RNG/record provenance pinned;
- raw rows and clean rerun evidence committed.

### G3 — Evaluation validity: **FAIL**

Pass theo một trong hai đường:

- mechanism-aware attackers implemented/validated; hoặc
- all non-REM attack columns and conclusions consistently narrowed to proxy
  diagnostics, không rank privacy.

### G4 — Statistical evidence: **FAIL**

Pass khi headline moving/stay claims có multi-seed/user-cluster uncertainty, correct
population labels và raw-row reproducibility.

### G5 — Manuscript/artifact consistency: **FAIL**

Pass khi repo-wide semantic sweep, source-table mapping và rebuilt PDF đều không còn
contradictions/numerical claims R3-009/R3-011.

### G6 — Industry/novelty scope: **PARTIAL**

Pass khi graph profile, deployment target, cache lifecycle và comparator positioning
được định nghĩa đủ để map algorithm vào một concrete system without overclaim.

## 8. Ordered remediation plan cho Claude agents

### Phase 0 — freeze và chia commit đúng

1. Giữ `c5385ec` và report này immutable.
2. Tạo implementation commit `C4` chỉ chứa code/tests/docs setup.
3. Không sửa output/PDF trong `C4`.
4. Chỉ chạy official experiments khi checkout `C4` clean.
5. Commit outputs/PDF thành `O4`; response riêng thành `R4` pin `C4/O4`.

### Phase 1 — formal blockers trước mọi rerun

1. Sửa proposition R3-001.
2. Sửa SM boundary/cell theorem R3-005.
3. Hạ PR anti-averaging wording R3-004.
4. Redesign budget API/state wording R3-012.
5. Thêm production-coupled formal tests R3-006.

Không rerun expensive experiments trước khi mechanism/API/seed schema được freeze.

### Phase 2 — deterministic experiment framework

1. Semantic-key RNG R3-002.
2. Provenance schema R3-003.
3. Tidy raw output schemas.
4. Canonical test/dev environment R3-014.
5. Graph download/hash/profile fixes R3-008/R3-013.

### Phase 3 — evaluation decision

Chọn rõ một path:

- **Claim-safe thesis path:** giữ attackers là proxy, hạ mọi comparative privacy
  conclusions; tập trung contribution formal/structural/finite-horizon; hoặc
- **Strong evidence path:** implement mechanism-aware sequential attackers và
  jitter-aware likelihood trước khi kết luận superiority.

Không dùng path ở giữa: caveat attacker ở cuối section nhưng vẫn dùng proxy error
để claim mechanism tốt hơn ở đầu section.

### Phase 4 — statistical rerun

1. Freeze selected record IDs.
2. Multi-seed moving benchmark.
3. User-cluster CIs cho trajectory metrics và stay success rates.
4. Budget/grid/theta/horizon ablations.
5. Store raw tidy rows + aggregate script.
6. Run from clean `C4`, capture complete provenance.

### Phase 5 — synchronize story

1. Fix README family split R3-009.
2. Create claim registry.
3. Sweep Ch1/Ch4/Ch5/Ch6/docs/comments.
4. Fix numerical/use-case errors R3-011.
5. Narrow novelty/comparator R3-015.
6. Rebuild PDF only after source/data/claims freeze.

### Phase 6 — response closure

Response agent phải trả lời từng `R3-001...R3-015` với ba cột:

| Finding | Code/test evidence | Manuscript evidence | Status |
|---|---|---|---|

Chỉ dùng **RESOLVED** khi acceptance criteria trong report này đều pass. Nếu chọn
claim-safe path, dùng **NARROWED**, không dùng RESOLVED cho capability chưa có.

## 9. Required regression checklist trước verification tiếp theo

### Formal/kernel

- [ ] Fixed-public conditioning theorem has base-Geo-I assumption and `1_A` mask.
- [ ] Negative rejection counterexamples committed.
- [ ] PR one-step production kernel enumeration.
- [ ] PR two-step same-public-history transcript bound.
- [ ] Direct finite sampler support witness.
- [ ] SM edge/corner representative-distance test.
- [ ] SM A-A and A-B-A lifecycle tests.
- [ ] All test functions assert and return `None`.

### RNG/provenance

- [ ] Mechanism-order permutation invariance.
- [ ] Epsilon-order permutation invariance.
- [ ] Existing results invariant after adding unrelated mechanism.
- [ ] Both outputs record clean source commit and dirty flag.
- [ ] Data/graph/environment hashes pinned and checked.
- [ ] Selected record IDs stored.
- [ ] Raw tidy rows regenerate aggregate tables.

### Attackers

- [ ] REM full-graph likelihood normalization test.
- [ ] HMM Bayes action matches declared loss.
- [ ] Offline/simulator normalized-emission parity.
- [ ] Restricted MLE checked against full-graph argmax.
- [ ] Non-REM mechanism-aware attack or explicit proxy-only conclusion.
- [ ] GPS jitter handling documented/tested.

### Experiments/statistics

- [ ] Multi-seed moving results.
- [ ] User-cluster uncertainty for headline metrics.
- [ ] Cluster CI for success rates.
- [ ] Stay-point labels used consistently.
- [ ] Budget split/grid/theta/horizon sensitivity.
- [ ] Exact commands and runtime recorded.

### Docs/PDF

- [ ] README distinguishes legacy/demo and thesis mechanisms.
- [ ] No stale `78k/77,727`, `optimal`, `150.5 at every n`, pharmacy or
  order-independent claims.
- [ ] Claim registry maps theorem/code/test/output/limitation.
- [ ] Tables regenerate from committed raw rows.
- [ ] PDF visual and semantic QA pass.

## 10. Definition of done

Vòng tiếp theo chỉ nên nhận verdict **PASS within declared scope** khi:

1. R3-001 và R3-002 không còn counterexample;
2. R3-004/R3-005 statements nhất quán repo-wide;
3. tests bind trực tiếp production code và canonical test suite green;
4. outputs pin clean source commit, exact graph/data/environment/RNG schema;
5. conclusions không vượt attacker/statistical evidence;
6. README/docs/thesis/PDF dùng cùng terminology, numbers và limitation;
7. response pin rõ implementation commit, output commit và verifier target.

Nếu mechanism-aware attacks, mobile implementation, exact finite-precision sampler
hoặc non-trivial w-event accountant chưa được làm, chúng vẫn có thể ở future work.
Điều kiện là thesis không gọi chúng là capability/contribution đã verify.

