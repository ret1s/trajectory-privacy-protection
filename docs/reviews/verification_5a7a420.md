# Verification vòng 3 — audit commit `5a7a420`

## Hồ sơ review

| Trường | Giá trị |
|---|---|
| Branch được review | `verifier` |
| Commit kết quả | `5a7a420b5ffce7133c94dfab8dd3c0e1202ba3ab` |
| Commit message | `Address verification round 2 (R2-001..R2-015): fix graph bbox, sampler, budget, claims` |
| Commit cha | `4a7086ed6d6f66b63f59caf7c22a3316185907c0` |
| Ngày review | 2026-08-23 (Asia/Ho_Chi_Minh) |
| Phạm vi | Formal privacy, finite-precision implementation, graph/data provenance, attackers, experiments, thesis/docs/PDF consistency |
| Kết luận | **NEEDS REVISION — numerical rerun verified, final claims chưa verified** |

Tài liệu này là review độc lập, commit-scoped. Mọi line reference trỏ tới commit
`5a7a420`, không phải một working tree được sửa sau review. Verifier vòng trước
`verification_0fafd4d.md` được giữ nguyên như provenance lịch sử; tài liệu này đánh
giá candidate đã phản hồi vòng đó.

Raw graph dùng cho các spot-check tồn tại cục bộ tại
`data/raw/beijing_graph.pkl`, SHA-256
`9e44ec2a48d6249f3534bb8511010925428018e528f5116d5f9eb24d86a2d487`.
Artifact này và raw BBBike extract đều bị `.gitignore` loại khỏi commit. Vì vậy
review xác nhận được artifact local và kết quả local, nhưng một clean clone chưa
thể tái lập chúng chỉ từ Git. Đây là một phần của R2-005/R2-013.

### Quy ước trạng thái

- **RESOLVED**: root cause và acceptance criteria liên quan đã pass.
- **NARROWED**: claim đã được hạ đúng về phạm vi evidence hiện có; capability mạnh
  hơn chưa được cài hoặc chứng minh.
- **PARTIAL**: có tiến bộ thực chất nhưng ít nhất một acceptance criterion còn fail.
- **OPEN/PLANNED**: chưa có implementation hoặc evidence mới đủ để review.

Review tách hai câu hỏi không được nhập làm một:

1. **Numerical reproducibility:** JSON/PDF mới có thật sự được tạo từ code và graph
   hiện tại hay không?
2. **Claim validity:** experiment, attacker, executable sampler và theorem có support
   kết luận privacy/industry mà manuscript đang đưa ra hay không?

Commit này pass câu hỏi thứ nhất ở local workspace, nhưng chưa pass câu hỏi thứ hai.

## 1. Technical summary

Commit `5a7a420` sửa được nhiều root cause quan trọng:

- bbox OSMnx đã dùng đúng thứ tự và graph local mới thực sự nằm trong vùng GeoLife;
- official benchmark và averaging outputs đã được rerun trên graph mới;
- PR-SM-REM đã dùng matched worst-case budget `epsilon/2 + epsilon/2`;
- bound một bước của PR đã cộng cả test cost và fresh REM release;
- Planar Laplace không còn nhận road-REM normalizer;
- Ch5 gọi 40 điểm là stay/significant locations thay vì ground-truth homes;
- PDF là bản build mới và bảng Ch5 khớp JSON.

Tuy nhiên candidate vẫn chưa thể coi là final verified result vì sáu nhóm blocker:

1. **Executable sampler vẫn có input-dependent zero support.** NumPy Gumbel-max
   hữu hạn độ chính xác không cho mọi candidate cơ hội thắng. Vì vậy pure Geo-I chỉ
   đúng cho ideal real-arithmetic kernel, không đúng cho executable hiện tại.
2. **Attacker được gọi là optimal/consistent nhưng vẫn misspecified end-to-end.**
   Objective với zero normalizer đúng family fixed-centre Planar trong mô hình
   no-jitter, và road-normalized form đúng cho vertex-secret iid REM; official
   protocol lại ép hypothesis lên road vertices, thêm GPS jitter, và dùng cùng proxy
   cho T-REM/SM-REM/PR. Diagnostic độc lập tìm được attacker mạnh hơn rõ rệt.
3. **Graph artifact đúng nhưng build recipe/provenance chưa đúng.** README không lấy
   largest weakly connected component, ghi sai counts, raw artifact không tracked và
   source/dependency chưa pin đủ.
4. **Formal rejection proposition vẫn sai/tự mâu thuẫn.** Fixed-public conditioning,
   secret-dependent conditioning và finite retry chưa được tách thành các mệnh đề có
   giả thiết đúng.
5. **Moving benchmark vẫn phụ thuộc mechanism order.** Một mutable RNG dùng chung,
   một seed, 7 users, không raw rows và không cluster uncertainty.
6. **Thesis/docs chưa đồng bộ với code.** Ch1/Ch4/Ch6 còn claim w-event, cutoff,
   graph cũ, optimal attacker, one-home/one-seed và on-device implementation trái
   với artifact hiện tại.

### Release decision

Không nên dùng các câu sau như kết luận cuối hoặc abstract claim:

- “Gumbel-max làm mọi road vertex reachable nên executable thỏa pure Geo-I.”
- “PR-SM-REM đạt trajectory-level `w`-event `epsilon_w`-Geo-I thật.”
- “47.8m / 54% là kết quả trước attacker tối ưu của PR-SM-REM.”
- “SM-REM giữ 150.5m ở mọi `n`.”
- “Recipe trong `data/README.md` tái tạo graph 13,813/41,040.”
- “Artifact hiện tại là implementation client-side/on-device.”
- “R2-003, R2-005, R2-006, R2-007, R2-008, R2-011 và R2-014 đã RESOLVED.”

Framing an toàn cho snapshot này:

> Commit `5a7a420` là một industry-aware research prototype với numerical outputs
> đã được tái lập cục bộ. Ideal REM/PR kernels có các bound real-arithmetic đã được
> thu hẹp hợp lý, nhưng executable finite-precision pure privacy, mechanism-aware
> attacks, reproducible clean build và manuscript consistency vẫn chưa hoàn tất.

## 2. Status matrix R2-001...R2-015

| ID | Severity | Response `5a7a420` | Verified status | Kết luận ngắn |
|---|---|---|---|---|
| R2-001 | P0 | RESOLVED | **RESOLVED for ideal kernel** | Bound mới đúng; thiếu committed formal/property test |
| R2-002 | P0 | PARTIAL | **NARROWED / repo-wide PARTIAL** | Ch4 nói trivial composition; Ch1/Ch6 vẫn overclaim/mâu thuẫn |
| R2-003 | P0 | RESOLVED | **PARTIAL** | Runner matched budget đúng; output config/test/sweep thiếu |
| R2-004 | P0 | PARTIAL | **PARTIAL — blocker** | Planar normalizer sửa; attacker family/jitter vẫn sai |
| R2-005 | P0 | RESOLVED | **PARTIAL** | Local graph đúng; README recipe và clean-clone provenance fail |
| R2-006 | P0 | RESOLVED (mitigated) | **PARTIAL — G0 blocker** | Gumbel-max vẫn có unreachable outputs |
| R2-007 | P1 | RESOLVED | **NARROWED / PARTIAL** | Scope cục bộ đúng; tên/framing/experiment chưa sync |
| R2-008 | P1 | RESOLVED | **NARROWED / PARTIAL** | Ch5 caveat stay-point; code/config/headings vẫn “home” |
| R2-009 | P1 | PARTIAL | **OPEN / PARTIAL** | Shared RNG, one seed, 7 users, no CI/raw rows |
| R2-010 | P1 | PARTIAL | **PARTIAL — broad** | Nhiều contradiction load-bearing còn trong thesis/docs |
| R2-011 | P1 | RESOLVED | **PARTIAL — formal blocker** | Proposition không đổi và vẫn sai như phát biểu |
| R2-012 | P1 | PLANNED | **OPEN / PLANNED** | Server-side prototype, chưa có on-device/lifecycle |
| R2-013 | P2 | PARTIAL | **PARTIAL** | Manifest là tiến bộ; result/data/env provenance thiếu |
| R2-014 | P2 | RESOLVED | **OPEN / PARTIAL** | Không có comparator table hoặc aligned benchmark |
| R2-015 | P2 | PARTIAL | **OPEN / PARTIAL** | PDF mới nhưng còn nhiều claim/count/layout stale |

Không được suy ra “overall verified” từ việc R2-001 và numerical rerun pass. Các
acceptance gates G0--G5 ở Mục 7 đều còn fail.

## 3. Scope, evidence và methodology

### 3.1 Commit inventory

Diff `4a7086e..5a7a420` thay đổi 16 files, 698 insertions và 516 deletions, gồm:

- mechanism/simulator/bbox code;
- graph manifest và data README;
- benchmark/averaging runners và hai JSON outputs;
- Ch4, Ch5 và `thesis/main.pdf`;
- response file `docs/reviews/response_0fafd4d.md`.

Static QA:

- `git diff --check 4a7086e 5a7a420`: pass;
- tám Python files thay đổi compile syntactically: pass;
- JSON parse: pass;
- không có test mới được commit; test duy nhất tìm thấy là `legacy/test_app.py`.

### 3.2 Graph artifact

Independent load trong environment tương thích với manifest xác nhận:

| Thuộc tính | Giá trị |
|---|---:|
| SHA-256 | `9e44ec2a48d6249f3534bb8511010925428018e528f5116d5f9eb24d86a2d487` |
| Nodes | 13,813 |
| Edges | 41,040 |
| Weak components | 1 |
| Latitude extent | 39.9600113--40.0199922 |
| Longitude extent | 116.2900027--116.3599974 |

Manifest khai báo Python 3.11.12, OSMnx 2.0.5, NetworkX 3.5, NumPy 2.3.2 và
SciPy 1.16.1. Local pickle hash và counts khớp manifest. Tất cả bbox callsites đã
dùng `(left,bottom,right,top)`.

Điều này xác nhận root bbox bug đã được sửa trong local artifact. Nó chưa xác nhận
clean-clone reproduction; xem R2-005/R2-013.

### 3.3 Numerical reproduction

Audit tái chạy hoặc đối chiếu độc lập các aggregate sau:

- toàn bộ 18 benchmark rows và 8 utility/realism metrics mỗi row: max absolute
  difference `0`;
- Bayes metrics tại `epsilon=0.02`: max difference `2.84e-14`;
- Planar HMM: calculated = saved = `76.82328616929541`, coverage `1.0`;
- toàn bộ averaging output, 5 mechanisms x 2 estimators x 7 values of `n`, gồm
  median, bootstrap CI và success rates: max rounded difference `0`;
- Ch5 tables khớp committed JSON sau rounding.

Kết luận đúng là: **outputs không stale đối với local code/graph hiện tại**. Việc
tái lập aggregates không làm attacker trở thành exact, không chứng minh privacy
của sampler và không bù cho raw/provenance records còn thiếu.

### 3.4 PDF QA

`thesis/main.pdf` là bản build mới, 30 trang A4, chứa counts và tables mới. Render
toàn bộ trang không thấy clipping, overlap hoặc black boxes nghiêm trọng; citations
và cross-references không undefined. Build log vẫn có bốn overfull boxes khoảng
2.3pt, 24.77pt, 17.68pt và 8.86pt. Nội dung stale vẫn hiện trực tiếp trong PDF,
đặc biệt ở Ch4, heading averaging và Ch6; xem R2-010/R2-015.

### 3.5 Giới hạn của chính verifier

Một số diagnostics dưới đây là audit-only scripts/runs chưa được check in. Chúng đủ
để falsify claim “resolved/optimal”, nhưng không được copy vào Ch5 như final result
cho đến khi method, environment, raw rows và command được version-control. Mục 10
ghi provenance và giới hạn của từng diagnostic.

## 4. Findings chi tiết

### R2-001 — bound PR một bước đã sửa đúng cho ideal kernel

**Status: RESOLVED for ideal real-arithmetic kernel.**

Evidence:

- `core/mechanisms.py:362-371`;
- `thesis/chapters/ch4_phuongphap.tex:322-348`.

Với previous public release `h`, kernel là:

\[
K_x(z\mid h)=
\begin{cases}
q_x(h)+(1-q_x(h))R_x(h), & z=h,\\
(1-q_x(h))R_x(z), & z\ne h.
\end{cases}
\]

Đặt `d=d(x,x')` và
`c=exp((eps_test+eps_release)d)`. Noisy threshold và ideal REM cho:

\[
q_x\le e^{\epsilon_{\mathrm{test}}d}q_{x'},\qquad
1-q_x\le e^{\epsilon_{\mathrm{test}}d}(1-q_{x'}),\qquad
R_x(z)\le e^{\epsilon_{\mathrm{release}}d}R_{x'}(z).
\]

Với `z != h`, tích hai chặn cho ngay `K_x(z|h) <= c K_x'(z|h)`. Với
`z=h`, hai summands cùng chịu hệ số `c`:

\[
\begin{aligned}
K_x(h\mid h)
&=q_x+(1-q_x)R_x(h)\\
&\le e^{\epsilon_{\mathrm{test}}d}q_{x'}
  +e^{(\epsilon_{\mathrm{test}}+\epsilon_{\mathrm{release}})d}(1-q_{x'})R_{x'}(h)\\
&\le c\,[q_{x'}+(1-q_{x'})R_{x'}(h)]
 = cK_{x'}(h\mid h).
\end{aligned}
\]

Đây là complete ideal-kernel argument mà proof sketch trong thesis nên viết tường
minh; không được lấy “ratio của tổng” bằng cách nhân ratio một cách không giải thích.

Hai-vertex spot-check với `a=0m`, `b=100m`, `h=a`, `theta=200m` và
`eps_test=eps_release=0.01/m` cho event `z=b`:

- `K_a(b) = 0.0255472867`;
- `K_b(b) = 0.1144949955`;
- ratio `= 4.481689 = exp(1.5)`;
- bound cũ `exp(1)` fail;
- bound mới `exp(2)` pass.

Hai nghìn random finite-domain checks một bước và adaptive two-step checks không tìm
thấy phản ví dụ cho bound mới. Condition trên cùng public transcript, internal state
`h` giống nhau; sau mỗi release `_prev_choice/_prev_release_xy` là hàm của public
release, nên adaptive-state composition hợp lệ ở ideal kernel level.

Hardening trước final release, không làm thay đổi kết luận theorem ideal ở trên:

- proof sketch nên nói rõ trong case `z=h`, cả hai summands chịu cùng hệ số
  `c=exp((eps_test+eps_release)d)`, nên tổng cũng chịu hệ số đó;
- theorem cần nói explicit “conditioned on the same public history”;
- biến exact enumeration thành committed unit/property test để chống regression;
- không nói bound ideal này đã chứng minh executable finite-precision mechanism;
  R2-006 vẫn độc lập và nghiêm trọng.

Regression evidence nên thêm ở vòng sau:

- `test_pr_two_vertex_privacy_bound` fail bound cũ và pass bound mới;
- `test_pr_full_transcript_bound_by_enumeration`;
- proof không phụ thuộc hidden state/counter được expose.

### R2-002 — Ch4 đã narrow claim, nhưng repo chưa nhất quán

**Status: NARROWED; repo-wide PARTIAL.**

`thesis/chapters/ch4_phuongphap.tex:350-361` phát biểu đúng:

- mỗi later step có worst-case `eps_test + eps_release`;
- first window có
  `eps_release + (w-1)(eps_test+eps_release)`;
- một full later window có `w(eps_test+eps_release)` dưới `D_infinity`;
- không có budget manager/privacy filter;
- `n_resample` chỉ là post-hoc diagnostic.

Đây là một scope correction hợp lệ, không phải implementation của non-trivial
`w`-event accountant. Repo vẫn mâu thuẫn:

- Ch1 `ch1_gioithieu.tex:42-46` claim “trajectory-level w-event epsilon_w-Geo-I
  thật”;
- Ch6 `ch6_tongket.tex:31-33` nói w-event còn là future work;
- `core/mechanisms.py:45-47` vẫn mô tả mọi mechanism tốn epsilon mỗi point;
- Ch4 `ch4_phuongphap.tex:224-228` vẫn nói SM siết composition theo số distinct
  locations, dù section SM sau đó bác bỏ arbitrary-trace theorem.

Required fix:

1. Chọn một trong hai scope:
   - thesis-safe: chỉ claim primitive per-step + trivial composition; hoặc
   - cài `w, epsilon_w`, sliding-window ledger/filter và behavior khi hết budget.
2. Định nghĩa neighboring trajectories và metric trước khi dùng scalar `epsilon_w`.
3. Đồng bộ Ch1, Ch4, Ch6, README và simulator budget display.

Acceptance:

- không claim một `epsilon_w` không định nghĩa, privacy saving theo realized branch,
  hoặc một enforced accountant khi chúng chưa tồn tại;
- trivial guarantee `epsilon_w=w(eps_test+eps_release)` được phép gọi là w-event
  bound sau khi length-`w` adjacency, `D_infinity`, session và metadata scope được
  định nghĩa rõ;
- test window composition cho mọi hard-branch transcript;
- post-hoc resample count không được gọi là guarantee ex-ante.

### R2-003 — matched budget đúng trong runners, chưa đủ đóng finding

**Status: PARTIAL.**

Positive evidence:

- `experiments/run_benchmark.py:49-51`;
- `experiments/run_averaging_multi.py:58-59`;
- `web/simulator.py:63`.

Các path chính construct
`PrivateReuseSMREM(eps/2, eps_test=eps/2)`. Vì fresh REM release có exponent
`-(eps/2)/2 * d = -0.25 eps d`, `EMISSION_SCALE["pr_sm_rem"] = 0.25` tại
`run_benchmark.py:59-70` là đúng. Hard step worst-case bằng total epsilon, khớp REM.

Acceptance criteria còn fail:

- `outputs/benchmark_results.json` chỉ lưu mechanism/epsilon cùng aggregates;
- averaging config tại `run_averaging_multi.py:140-142` không lưu `eps_test`,
  `eps_release`, `theta`, `w` hoặc `epsilon_w`;
- caption/setup Ch5 không khai báo đầy đủ PR parameters;
- không có formal-budget equality assertion;
- không có sweep `theta x budget split`.

API còn một hazard: `core/mechanisms.py:395-397` mặc định
`eps_test=epsilon`. Vì positional `epsilon` thực chất là release budget, direct call
`PrivateReuseSMREM(E, rn)` có hard-step cost tới `2E`. Official runners đã đúng,
nhưng public API vẫn dễ bị dùng sai.

Response `response_0fafd4d.md:28` nói split đã nằm trong “caption/config”, nhưng
committed output config không support câu đó.

Required fix:

- đưa resolved parameters vào result manifest và Ch5 setup;
- assert tổng formal cost trước khi chạy comparison;
- đổi API thành keyword-only `epsilon_release`, `epsilon_test`, hoặc nhận
  `epsilon_step_cap` rồi derive split với invariant machine-checkable;
- chạy sensitivity theo `theta`, split và utility/privacy, hoặc hạ claim về một
  single operating point exploratory.

### R2-006 — Gumbel-max vẫn tạo zero-support hữu hạn độ chính xác

**Status: PARTIAL; G0 blocker. Response đánh RESOLVED là sai.**

Evidence:

- `core/mechanisms.py:166-177`;
- NumPy 2.3.2 `random_gumbel` dùng uniform sinh từ 53-bit `next_double`;
- official source:
  <https://github.com/numpy/numpy/blob/v2.3.2/numpy/random/src/distributions/distributions.c#L478-L485>.

Docstring nói Gumbel-max “draws exactly from softmax” và “Every candidate therefore
stays reachable”. Với executable NumPy float64, hai câu này không đúng.

Vì accepted uniform nhỏ nhất/lớn nhất là hữu hạn, standard Gumbel executable nằm
trong khoảng xấp xỉ:

\[
[-3.60377899,\ 36.73680057],
\]

nên chênh lệch tối đa giữa hai Gumbel draws là `40.34057956`. Nếu một candidate có
logit thấp hơn leader quá mức này, nó không thể thắng `argmax(logit + g)`.

Trên graph 13,813 vertices mới, tại góc SW của bbox:

| Total/release epsilon dùng trong REM score | Logit range | Candidates không thể thắng |
|---:|---:|---:|
| 0.01 | 43.1377 | 46 |
| 0.02 | 86.2754 | 9,685 |
| 0.05 | 215.6885 | 13,277 |

Vertex index `5630`, coordinate `(40.0190986, 116.3596198)`, cách SW khoảng
8,852m, có logit gap `43.1377` tại epsilon `0.01`. Ideal probability từ SW là
khoảng `8.84e-21 > 0`, nhưng executable probability bằng 0. Tại input ở chính
vertex đó, `default_rng(64)` có thể chọn nó. Support vì vậy phụ thuộc secret input
và có một likelihood ratio vô hạn.

Lỗi này cũng ảnh hưởng PR matched-budget từ total `epsilon=0.02`, vì release kernel
dùng `epsilon_release=0.01`. Việc graph nhỏ hơn làm lỗi ít hơn ở epsilon thấp nhưng
không loại bỏ nó.

#### Finite Laplace threshold của PR cũng cần audit riêng

PR dùng `rng.laplace` tại `core/mechanisms.py:414`. NumPy Laplace cũng được sinh từ
một finite 53-bit uniform grid, nên noise executable bị chặn xấp xỉ bởi

\[
|L|\le \frac{52\ln 2}{\epsilon_{\mathrm{test}}}.
\]

Với matched totals và `theta=200m`:

| Total `E` | `eps_test=E/2` | Max absolute Laplace noise | Reuse probability chắc chắn bằng 0 khi `d(x,h)` lớn hơn |
|---:|---:|---:|---:|
| 0.01 | 0.005 | 7,208.73m | 7,408.73m |
| 0.02 | 0.010 | 3,604.37m | 3,804.37m |
| 0.05 | 0.025 | 1,441.75m | 1,641.75m |

Các khoảng cách này xuất hiện trong graph dài khoảng 8.85km. Binary threshold test
vì vậy không phải exact ideal Laplace mechanism trên declared domain. Điều này chưa
tự nó chứng minh output-only `K` có zero mass, vì resample collision có thể che test
bit; nhưng ideal composition proof không được áp thẳng cho executable test. Cần một
direct finite-kernel proof/test ngay cả sau khi release sampler được sửa.

Ngoài ra RNG state của PR tiến một số draws phụ thuộc branch: reuse bỏ qua Gumbel
draws còn resample consume chúng. Ideal proof giả định fresh independent randomness;
finite PRNG state là một lý do bổ sung không được đồng nhất executable với ideal
stochastic kernel, dù counterexample zero-support đã đủ bác pure-DP claim.

Required decision — phải chọn và ghi rõ một trong các hướng:

1. **Ideal-only claim:** theorem chỉ áp cho mathematical kernel; executable được gọi
   numerical approximation, không claim pure Geo-I.
2. **Approximate privacy:** định lượng uniform `(epsilon, delta)` hoặc numerical
   privacy loss cho declared bbox/epsilon/domain.
3. **Exact/verified discrete sampling:** dùng một algorithm có chứng minh sampling
   từ computable weights với unbounded random-bit refinement/interval arithmetic,
   rồi property-test actual implementation. Chỉ đổi sang một float sampler khác
   không đủ.

Acceptance:

- docstring bỏ “exact/every candidate reachable” nếu chưa chứng minh;
- property test trên declared domain falsify được SW/far-vertex counterexample;
- `test_pr_finite_laplace_threshold_privacy` kiểm tra direct output kernel, không chỉ
  binary test bit;
- theorem/manuscript tách ideal distribution khỏi executable implementation;
- mọi epsilon trong benchmark thỏa privacy notion đã khai báo.

### R2-011 — rejection proposition vẫn sai như phát biểu

**Status: PARTIAL; formal blocker.**

`thesis/chapters/ch4_phuongphap.tex:91-107` không thay đổi so với parent ở phần
mệnh đề. Nó định nghĩa một fixed set `F`, rồi kết luận acceptance normalizer không
được bounded; ngay sau đó lại thừa nhận conditioning trên fixed public allowed set
có bound `2 epsilon`. Hai đoạn tự mâu thuẫn.

Với fixed public allowed set `A` và `K(A|x)>0`:

\[
\frac{K_A(z\mid x)}{K_A(z\mid x')}
\le e^{\epsilon d(x,x')}
   \frac{K(A\mid x')}{K(A\mid x)}
\le e^{2\epsilon d(x,x')}.
\]

Finding gốc không phải “mọi public rejection đều phá privacy”, mà là cần phân biệt:

- fixed public `A` với positive acceptance;
- secret/input-dependent `A_x` hoặc QoS check quanh `x`;
- zero acceptance ở một input;
- finite retry và fallback path, nơi output distribution khác infinite conditioning.

Minimal secret-dependent witness: cho base kernel `K` uniform trên output
`{0,1}` với mọi input, nên là 0-DP. Đặt allowed sets `A_a={0}` và `A_b={1}`. Sau
conditioning, `K_A(0|a)=1` nhưng `K_A(0|b)=0`, cho likelihood ratio vô hạn. Notation
`A_x` là điều còn thiếu trong proposition hiện tại.

Required fix:

- viết lại proposition bằng notation `A` và `A_x` riêng;
- thêm giả thiết positive acceptance;
- tách finite retry/fallback thành một mechanism khác;
- thêm toy finite-domain normalization/privacy test.

Response `response_0fafd4d.md:62-63` không được giữ status RESOLVED.

### R2-004 — attacker vẫn chưa mechanism-aware hoặc optimal

**Status: PARTIAL; P0 evaluation blocker.**

Positive changes đã verify:

- Planar và baseline dùng constant/zero normalizer thay vì road-REM normalizer tại
  `experiments/run_benchmark.py:59-79,105-116`;
- averaging Planar cũng dùng zero normalizer tại
  `experiments/run_averaging_multi.py:47-49,86-99`;
- PR emission scale `0.25` khớp matched release budget;
- discrete REM likelihood normalize tới sai số tối đa `2.22e-16` trên các vertices
  được spot-check.

Nhưng code/manuscript vẫn dùng wording vượt evidence:

- `evaluation/attacks.py:240-264`: “CORRECT adversary estimator”, “consistent
  maximum-likelihood” và kết luận convergence cho T-REM/SM;
- `evaluation/attacks.py:281-286`: argmax “provably” trong radius 1,200m mà không có
  proof/test;
- `experiments/run_averaging_multi.py:2-16`: “consistent mechanism-aware MLE”;
- `thesis/chapters/ch5_thucnghiem.tex:135-145`: “MLE nhất quán” và heading
  “attacker tối ưu”;
- `evaluation/attacks.py:200-232`: online HMM bỏ `logZ`, không cùng likelihood với
  benchmark HMM.

Baseline cũng bị misspecified: `core/mechanisms.py:76-130` dùng exponential radius
có cap, temporal smoothing và conditional road snap, trong khi runner chấm bằng
memoryless `exp(-epsilon*d)` với zero normalizer. Đây không phải exact baseline
likelihood.

Benchmark HMM tại `evaluation/attacks.py:193-197` trả posterior mean nhưng metric là
Euclidean distance; Bayes action cho loss này là posterior geometric median. HMM
vẫn là một fixed diagnostic attack hữu ích, không phải loss-optimal attacker.

Objective hiện tại

\[
-a\sum_i d(x,z_i)-n\log Z(x)
\]

với zero normalizer là fixed-centre Planar log-likelihood; với input-dependent road
normalizer và đúng coefficient, nó là exact iid REM likelihood trên vertex-secret
domain. Official experiment lại ép Planar hypotheses lên road vertices, dùng
continuous stay centres với Gaussian input jitter, còn T-REM có temporal weights,
SM có cache/cells và PR có sequential equality/reuse events. Vì vậy official rows
không phải optimal end-to-end attackers; T-REM/SM/PR cụ thể là **REM-form proxies**.

#### Planar diagnostic

Trên official 40 locations x 8 seeds x 100 jittered reports:

| Estimator | Median error | Success within 50m |
|---|---:|---:|
| Official road-vertex, jitter-ignorant Planar-likelihood proxy | 25.2m | 91.2% |
| Sample mean | 10.135m | 100% |
| Fixed-centre Planar geometric median | 9.045m | 100% |

Geometric median là analytic MLE cho fixed-centre Planar Laplace nếu bỏ input
jitter. Diagnostic trên vẫn jitter-ignorant, nên chưa phải exact Gaussian--Laplace
convolution MLE; tuy nhiên nó đủ bác bỏ claim official 25.2m row là “optimal”.

#### PR sequential diagnostic

Trên current graph, matched parameters `eps_release=eps_test=0.01`, `theta=200m`,
`n=100`, static/no-jitter candidate-domain protocol:

| PR attacker | Median error | Success within 50m |
|---|---:|---:|
| iid REM proxy | 44.939m | 58.75% |
| Exact sequential PR kernel | 34.375m | 74.06% |

Sequential likelihood dùng

\[
P(z_1\mid x)=R_x(z_1),\qquad
P(z_{2:T}\mid x,z_{1:T-1})=\prod_{t=2}^T K_x(z_t\mid z_{t-1}),
\]

với equality mass `q_x(h)+(1-q_x(h))R_x(h)` và inequality mass
`(1-q_x(h))R_x(z)`. Diagnostic chưa marginalize jitter và hypothesis vẫn là road
vertices, nhưng chênh lệch đủ lớn để refute “optimal proxy”.

Candidate-radius spot-check trên 40 locations, seed 0, `n=100` không thấy local
1,200m search khác full-V argmax cho Planar/REM/PR proxy. Đây là empirical check,
không phải proof cho chữ “provably”.

Impact:

- Planar row đang đánh giá privacy quá lạc quan;
- PR proxy bỏ run-length/equality evidence và có thể đánh giá defense quá lạc quan;
- không được dùng `47.8m vs 29.8m` để claim PR vượt REM trước optimal attacker;
- current table vẫn hữu ích như proxy study nếu label và caveat đúng.

Required fix có hai đường hợp lệ:

**Minimum claim-safe path:** giữ current proxy table, đổi heading/docstrings/caption
thành restricted Planar/REM likelihood hoặc REM-form proxy theo từng row; hạ mọi
optimal/comparative-superiority claim và coi table là exploratory descriptive result.

**Stronger evidence path:** cài attackers đúng mechanism/jitter rồi rerun official
comparison:

1. Định nghĩa per-mechanism `log_likelihood(history, candidate_secret, params)`.
2. Planar: continuous/declaratively discretized secret domain và jitter marginalization.
3. Baseline: exact pipeline likelihood nếu khả thi, hoặc label heuristic proxy và
   không dùng optimal claim.
4. REM: nói rõ vertex-secret hay continuous-secret model.
5. T-REM: dùng history-dependent `Z_t(x,z_{t-1})`.
6. SM: model cell representative, cache state, equality và cell spill.
7. PR: dùng sequential kernel và latent jitter.
8. Đồng bộ benchmark HMM với online HMM và dùng Bayes action khớp loss, hoặc label
   posterior-mean HMM là fixed heuristic.
9. Cho tới lúc đó, label theo row: restricted Planar likelihood, exact
   vertex/no-jitter REM khi applicable, và mechanism proxy cho baseline/T/SM/PR.

Acceptance cho **minimum claim-safe path**:

- heading/caption/code docstrings không còn “optimal/consistent/correct”;
- mỗi row được label đúng restricted likelihood/proxy;
- baseline và HMM được label heuristic;
- threat model, secret domain, jitter omission và loss/estimator nằm ngay trước table;
- không dùng proxy differences để claim comparative superiority.

Acceptance bổ sung nếu giữ **strong optimal-attacker claims**:

- brute-force toy-domain likelihood test cho từng mechanism;
- correct normalizer tests cho Planar/REM/HMM;
- baseline có tested pipeline likelihood;
- HMM estimator khớp declared loss;
- PR equality/inequality likelihood khớp exact enumeration;
- jitter-aware recovery test;
- official table được rerun từ attacker implementations cuối.

### R2-005 — graph local đúng nhưng recipe chưa tái lập artifact

**Status: PARTIAL. Response RESOLVED là quá mạnh.**

Positive evidence:

- local hash/count/extent/one-WCC khớp manifest;
- mọi OSMnx bbox callsite đã dùng `(left,bottom,right,top)`;
- Ch5 graph description khớp local artifact;
- numerical outputs reconcile với local graph/code.

Independent execution của đúng recipe `data/README.md:26-36` với local
`Beijing.osm` cho:

| Stage | Nodes | Edges | Weak components |
|---|---:|---:|---:|
| `graph_from_xml` | 335,097 | 886,427 | not recorded |
| README bbox truncation | 19,634 | 47,745 | 4,313 |
| Largest WCC | 13,813 | 41,040 | 1 |

Manifest `data/beijing_graph.manifest.json:3` nói có largest-WCC step, nhưng README
code không làm bước này. `data/README.md:39` còn ghi `~78k/~209k`.

Local source hashes, chưa nằm trong manifest:

| Artifact | SHA-256 |
|---|---|
| `data/raw/Beijing.osm.gz` | `ed0af579fb32c7164de85bca945a10432f818dd27eb80d6d5a7afa62213b0407` |
| `data/raw/Beijing.osm` | `5691186b8b3bd9078915faefb7fb639ff2c7f45b4e8b1efd1de4b9ebdad4c8a5` |
| `data/raw/beijing_graph.pkl` | `9e44ec2a48d6249f3534bb8511010925428018e528f5116d5f9eb24d86a2d487` |

Các gaps còn lại:

- `data/raw/` bị ignore;
- BBBike URL mutable, không có download date/hash trong committed manifest;
- không có tracked executable build script;
- manifest thiếu explicit component count và pickle-relevant versions như Shapely,
  Pandas/GeoPandas;
- `requirements.txt` chỉ pin lower bounds;
- local `.venv` Python 3.9/NumPy 1.x không load được pickle sinh bởi NumPy 2.x.

Required fix:

1. Thêm tracked deterministic graph-build script.
2. Verify compressed/decompressed source hashes trước build.
3. Truncate bbox rồi largest-WCC bằng explicit deterministic code.
4. Generate manifest từ artifact thay vì nhập tay.
5. Pin complete environment hoặc dùng stable cross-version serialization.
6. Sửa README counts và chạy clean-clone test.

Acceptance:

- clean clone + pinned source tạo đúng graph hash;
- README command chạy nguyên trạng;
- manifest/component/bbox/hash tests pass;
- runner assert expected graph hash trước experiment;
- không còn graph counts cũ ngoài historical verifier.

### R2-007 — previous-release predictive reuse đã scoped đúng cục bộ

**Status: NARROWED / PARTIAL.**

Đã sửa đúng:

- implementation chỉ lưu `_prev_release_xy/_prev_choice` tại
  `core/mechanisms.py:401-426`;
- Ch4 `ch4_phuongphap.tex:364-369` và docstring `mechanisms.py:385-390` nói rõ
  leave-and-return `A-B-A` không được nhớ;
- state là previous public release, không phải persistent secret-location cache.

Chưa đồng bộ:

- tên `PR-SM-REM` vẫn dễ được hiểu là private persistent SM cache;
- Ch1 `ch1_gioithieu.tex:42-46` nói thay memoization exact và đạt true w-event,
  không giải thích previous-release-only scope;
- Ch5 setup và Ch6/README không mô tả PR nhất quán;
- experiment chỉ là contiguous `A,A,...`, chưa có `A-B-A`, cache lifecycle hoặc
  cross-session scenario.

Với code hiện tại, tên/framing an toàn là **Predictive-REM / previous-release private
reuse**. Nếu giữ `PR-SM-REM`, phải giải thích ngay lần đầu rằng `SM` chỉ là nguồn gốc
thiết kế, không phải persistent cache semantics.

Acceptance:

- một state-machine description duy nhất trong Ch1/Ch4/Ch5/Ch6/README/code;
- không claim home-return/cross-session defense;
- chỉ cần A-B-A test nếu muốn nâng scope sang leave-and-return protection.

### R2-008 — population đã relabel trong Ch5 nhưng chưa repo-wide

**Status: NARROWED / PARTIAL.**

Positive change: `ch5_thucnghiem.tex:149-152` gọi đúng 40 stay/significant
locations, 21 users và nói không phải ground-truth residences.

Loader `data/geolife.py:166-200` vẫn:

- dùng `home`/`n_homes` fields;
- lấy generic 200m/20min stays;
- chọn first qualifying stays theo sorted order;
- không có nighttime, recurrence, residence validation hoặc random sampling.

Stale naming còn ở `experiments/run_averaging_multi.py:2-16,79-109` và JSON config.
Effective population là 21 user clusters, không phải 320 independent samples.

GPS jitter đi qua nhiều grid cells trong 100 reports:

| Statistic | Distinct SM grid cells |
|---|---:|
| Min | 1 |
| Median | 3 |
| Mean | 3.09375 |
| Max | 4 |

Do đó claim “SM 150.5m ở mọi n” sai; actual row là
`164.5, 162.8, 157.8, 149.4, 147.1, 152.3, 150.5m` cho
`n=1,2,5,10,20,50,100`.

Audit-only user-cluster bootstrap cho success within 50m:

| Mechanism | Point estimate | Approx. 95% CI |
|---|---:|---:|
| SM-REM | 11.25% | 7.81%--15.06% |
| PR-SM-REM | 53.75% | 44.59%--62.19% |

JSON có user-cluster median CI cho mọi `n`, nhưng Ch5 chỉ trình bày point medians dù
setup nói dùng bootstrap. Success CIs thì không tồn tại cả trong JSON lẫn manuscript;
raw rows cũng không được lưu để audit heterogeneity.

Required fix:

- repo-wide rename sang stay/significant-location;
- random/stratified location sampling;
- raw user/location/seed rows và user-cluster CI cho median lẫn success;
- boundary/cell-count stratification;
- nếu giữ “home” claim, thêm nighttime + recurrence + cross-session validation.

### R2-009 — moving benchmark vẫn one-seed và order-dependent

**Status: OPEN / PARTIAL.**

`experiments/run_benchmark.py:106-107` tạo một `default_rng(42)` rồi truyền cùng
mutable object tuần tự qua sáu mechanisms. Empirical permutation:

| Execution | First REM candidate-array index |
|---|---:|
| Normal mechanism order | 11,852 |
| REM run first | 594 |

Benchmark gồm 20 trajectories, 858 points, 7 users và một seed. Không có independent
per-mechanism seed derivation, repeated seeds, user-cluster CI, raw per-trajectory
rows hoặc selected user/file manifest.

Moving population cũng là deterministic convenience sample:
`data/geolife.py:43-50,101-133` lấy 20 qualifying trajectories đầu tiên theo sorted
user/file order, tối đa ba trajectories mỗi user. Không có random/stratified sampling
hoặc population-generalization design.

Exact aggregate reproduction chỉ chứng minh deterministic output không stale; nó
không cung cấp uncertainty hoặc order-invariant comparison.

Required fix:

- derive stable seeds từ `(root,mechanism,epsilon,user,trajectory,replicate)`;
- paired nhưng independent streams per mechanism;
- multiple replicates và user-cluster intervals;
- random/stratified trajectory selection hoặc explicit convenience-sample framing;
- raw rows + aggregate rebuild;
- test permuting mechanism order không đổi per-mechanism output.

### R2-010 — cross-file consistency vẫn là broad P1 blocker

**Status: PARTIAL / OPEN.**

Load-bearing contradictions còn trong thesis:

1. Ch1 `ch1_gioithieu.tex:24-47` nói “hai cơ chế” nhưng liệt kê bốn.
2. Ch1 claim true `w`-event; Ch4 nói chỉ trivial composition; Ch6 nói future.
3. Ch4 `ch4_phuongphap.tex:224-228` nói SM siết budget theo distinct locations,
   trái caveat arbitrary-trace tại lines 275-286.
4. Ch4 algorithm `ch4_phuongphap.tex:372-387` vẫn dùng secret-centred `V_c`, graph
   77,727 vertices và `<10ms`, trái full-V code/manifest.
5. Ch5 setup `ch5_thucnghiem.tex:21-27` liệt kê 5 mechanisms nhưng table có 6 và
   không khai báo PR split/theta.
6. Ch5 heading line 144 gọi attacker tối ưu dù lines 200-205 caveat proxy.
7. Ch6 lines 26-42 còn cutoff, distinct-cell composition, one-home/one-seed và
   likelihood state cũ.

Repo/support docs còn stale:

- README bỏ PR và nói benchmark 5 mechanisms (`README.md:21,36-58`);
- module docstring `core/mechanisms.py:8-47` chỉ mô tả bốn mechanism và nói mọi
  mechanism tốn epsilon per point;
- `docs/research_notes.md:84-117` chứa bảng/câu chuyện cũ;
- `docs/problem_formulation.md:63-90` claim distinct-cell composition/public grid;
- `docs/system_model_and_threats.md:252-264` và
  `docs/attack_scenarios.md:284` nói mechanisms chưa memoize;
- `data/README.md:39` chứa graph counts cũ;
- Ch3 chưa định nghĩa taxonomy S1--S8 mà Ch4 dùng.

Acceptance:

- repo-wide terminology/count/algorithm/scope sweep;
- historical verifier files được loại khỏi stale-string gate;
- Ch1--Ch6, README, code docstrings và grounding docs kể cùng một story;
- Ch4 pseudocode phải khớp full-support implementation.

### R2-012 — on-device là target architecture, chưa phải artifact hiện có

**Status: OPEN / PLANNED.**

Ch4 `ch4_phuongphap.tex:21-32` viết mechanism “chạy client-side/on-device” như một
implementation fact. Artifact thực tế là Python/Flask server-side simulator
(`README.md:26-27`, `web/simulator.py:49-106`). Không có mobile/OS integration,
secure cache, TTL/eviction/restart/map-update policy, identifier rotation, energy
benchmark hoặc SDK contract.

Graph size `33MB` ở Ch4 cũng stale; current local bbox pickle khoảng 5.9MB. Runtime
JSON đo cả mechanism, metrics và attackers nên không support sub-ms/<10ms on-device.

Required wording:

> Artifact hiện tại là một on-device-targeted research prototype/reference
> mechanism với server-side Flask simulator. Kiến trúc đích đặt mechanism trước
> điểm gửi tới LBS, nhưng chưa được hiện thực hoặc benchmark trên mobile client.

Acceptance:

- tách mathematical mechanism, Python reference và target mobile deployment;
- cache/identifier lifecycle được định nghĩa dù có thể out-of-scope;
- không dùng latency/memory claim nếu chưa có isolated hardware benchmark.

### R2-013 — manifest là tiến bộ nhưng output provenance chưa đủ

**Status: PARTIAL.**

`outputs/benchmark_results.json` là bare aggregate array. Averaging config tại
`run_averaging_multi.py:140-142` chỉ lưu `n_homes`, count seeds, epsilon, jitter và
report count.

Thiếu:

- Git commit và dirty-tree flag;
- graph/source/data hashes;
- selected users/files/stays;
- actual derived seeds;
- `theta`, test/release split, grid và temporal params;
- root seed, epsilon list, loader interval/filter parameters;
- QoS radius;
- POI `n_pois`, `k`, `k_prime` và seed;
- HMM candidate radius/max candidates/`v_typ`;
- MLE candidate radius;
- `KS`, success radii, bootstrap reducer/draws/seed;
- exact command và generation timestamp;
- dependency versions/lock;
- bootstrap seed/draw count;
- raw replicate rows;
- hardware/runtime context.

Required fix:

1. Mỗi run xuất immutable manifest với code/input/config/environment identity.
2. Lưu tidy raw rows và aggregate từ raw bằng tracked script.
3. Pin dependencies và graph build.
4. Runner fail nếu graph hash/config không match expected manifest.
5. Không overwrite verified outputs mặc định.

Acceptance: clean environment có thể xác định chính xác artifact/config tạo JSON và
rebuild aggregate byte-for-byte hoặc numerically theo declared tolerances.

### R2-014 — novelty comparator chưa tồn tại

**Status: OPEN / PARTIAL. Response RESOLVED là sai.**

Response nói đã có “bảng đối thủ Ch4/section 5.5”, nhưng:

- Ch4 `ch4_phuongphap.tex:42-61` chỉ có requirements table;
- Ch5 `ch5_thucnghiem.tex:200-209` chỉ có một paragraph Eclipse/RAPPOR;
- Ch3 `ch3_lienquan.tex:37-49` không phân tích predictive mechanism, RAPPOR,
  LP-Doctor, Release-GeoInd hoặc memoization precedents;
- Eclipse bị gọi “đối thủ gần nhất” dù chưa align threat model/benchmark.

Novelty defensible hiện tại:

> Road-native instantiation/integration của previous-release predictive reuse với
> REM, cùng corrected formal accounting và threat-grounded repeated-location
> evaluation. REM là Euclidean-metric specialization/sibling của GEM; predictive
> reuse và memoization có precedents, không phải primitive mới.

Comparator table tối thiểu phải có: protected unit, state/lifetime, reuse decision,
formal guarantee, road-native output và aligned evaluation cho Predictive Mechanism,
RAPPOR PRR, LP-Doctor, GEM/GG-I, Eclipse, SM-REM và current Predictive-REM. Chỉ thêm
nguồn khác sau khi primary metadata được verify.

Acceptance:

- comparator table xuất hiện trong thesis, không chỉ response/support docs;
- gap/novelty claim scoped theo tập literature và ngày review;
- Eclipse không gọi direct/nearest competitor nếu chưa có aligned comparison.

### R2-015 — tables đúng nhưng narrative/PDF vẫn stale

**Status: OPEN / PARTIAL.**

Numerical reconciliation của tables pass, nhưng các claim sau còn trong source/PDF:

- graph `77,727/~78k/~209k`, secret cutoff và `<10ms`;
- projection “sai số dưới 1m” categorical;
- five-mechanism setup thiếu PR;
- SM “150.5m ở mọi n”;
- PR “tốt hơn REM đáng kể” trước proxy attacker;
- noisy test chỉ cho “vài resample lọt qua”;
- “pharmacy”/recall khoảng 81% dù POIs là synthetic random road vertices;
- on-road metric bị diễn giải như literal point-to-road-edge validity;
- T-REM “đóng” velocity-linkage dù chưa chạy direct reconstruction attack;
- Ch6 nói one-home/one-seed dù current study là 40x8.

Projection diagnostic, 50,000 sampled vertex pairs, seed 0:

| Statistic | Absolute projected-vs-haversine error |
|---|---:|
| Median | 0.0876m |
| P95 | 0.7596m |
| P99 | 1.2043m |
| Sample max | 2.1825m |
| Fraction at most 1m | 97.818% |

Wording supportable là “p95 dưới 1m trong sampled-pair diagnostic”, không phải mọi
pair đều dưới 1m.

Current `epsilon=0.02` synthetic k-NN recall trải từ 0.776 tới 0.985 tùy mechanism;
không có một unqualified 81% pharmacy result. Benchmark protocol dùng 500 synthetic
POIs, true `k=5`, returned `k_prime=10` (`evaluation/metrics.py:108`), trong khi
simulator dùng 300 synthetic POIs, `k=3`, `k_prime=3` (`web/simulator.py:54`). Vì
vậy benchmark range không thay thế trực tiếp được một con số cho simulator “three
pharmacies”. Nếu muốn pharmacy use case, phải load real OSM amenities và pin query
protocol.

Official PR jittered 40x8 run có `n_resample` min 4, median 49, mean 48.975, p95 74
và max 91 trên 100 reports. Counter gồm first mandatory release; trừ bước đầu vẫn
còn median 48 later resamples. Câu `ch5_thucnghiem.tex:196-197` rằng chỉ có “vài
resample lọt qua” là sai lớn. Diagnostic này chưa được lưu trong output, củng cố
R2-013.

`evaluation/metrics.py:76-82` đo khoảng cách tới nearest road **vertex** trong 25m,
không phải distance tới road edge/polyline. Vì vậy tỷ lệ 42--50% của Planar không
được diễn giải thẳng thành literal off-road/map-matching surface. REM-family 100%
vẫn là structural vì outputs chính là vertices; tên metric/prose phải nói đúng proxy.

Required fix:

- repo-wide stale claim search;
- Ch4 algorithm/full support và Ch6 limitations rewrite;
- Ch5 six-mechanism config + exact attacker labels;
- synthetic POI labeling;
- nearest-vertex metric labeling hoặc đổi implementation sang point-to-edge distance;
- sửa resample narrative và lưu resample diagnostics trong output;
- isolated runtime benchmark hoặc bỏ timing claim;
- rebuild/render PDF và kiểm tra content, không chỉ compile.

## 5. Calculation spot-check summary

| Claim/metric | Verification status | Evidence/result |
|---|---|---|
| Local graph hash/count/extent | **Verified** | Hash `9e44...d487`; 13,813/41,040; one WCC |
| README recipe rebuilds that graph | **False** | Recipe stops at 19,634/47,745 and 4,313 WCCs |
| All bbox callsites use OSMnx 2.x order | **Verified** | Core/web/data callsite sweep pass |
| Release-derived benchmark metrics are current | **Verified locally** | 18 rows x 8 metrics, max diff 0; attacks spot-checked separately |
| Bayes attack columns at `epsilon=0.02` | **Spot-check pass** | All six rows, max absolute diff `2.84e-14` |
| Planar HMM at `epsilon=0.02` | **Spot-check pass** | Exact match `76.82328616929541`, coverage 1.0 |
| Averaging aggregate is current | **Verified locally** | 5x2x7 medians/CIs/success, rounded diff 0 |
| Ch5 numbers match JSON | **Verified** | Tables reconcile after rounding |
| PR and REM official rows use matched step cap | **Verified in constructors** | PR split `E/2 + E/2`; emission scale 0.25 |
| Output artifact records matched split/theta | **False** | Config fields absent |
| R2-001 new ideal bound | **Verified** | Ratio `exp(1.5)` passes `exp(2)`, fails old `exp(1)` |
| Current Gumbel sampler is exact/full-support | **False** | Bounded Gumbel span; input-dependent unreachable vertices |
| Planar 25.2m row uses optimal attacker | **False** | Mean/geometric-median diagnostics reach about 10.1m/9.0m |
| PR 47.8m row uses exact sequential attacker | **False** | No-jitter sequential diagnostic beats proxy materially |
| SM is exactly 150.5m at every `n` | **False** | Actual row varies 164.5m to 147.1m to 150.5m |
| 40 locations are ground-truth homes | **False** | Generic 200m/20min stay points, no residence validation |
| Moving comparison is seed/order robust | **False** | Shared RNG; REM first output changes under order permutation |
| Projection error is always below 1m | **False** | p99 1.2043m, sampled max 2.1825m |
| PDF compiled from current source | **Verified** | 30 pages, current tables, no undefined refs |
| PDF content is internally consistent | **False** | Cutoff/old graph/optimal/one-home claims remain |

## 6. PDF và presentation QA

Build mechanics pass nhưng content QA fail.

- `thesis/main.pdf`: 30 pages, A4, 819,181 bytes, created 2026-08-23 13:20 +07.
- Render toàn bộ pages: không có obvious clipping/overlap/black squares.
- Không có undefined citations/references.
- Bảng Ch5 fit và match JSON.
- Bốn overfull boxes còn trong `thesis/main.log`:
  - khoảng 2.3pt ở Ch1;
  - 24.77pt ở Ch4 requirements table;
  - 17.68pt ở Ch4 proof;
  - 8.86pt ở Ch4 heading.

Visual inspection cũng xác nhận committed PDF chứa chính các contradiction:

- Ch4 page thuật toán còn cutoff và graph 77,727;
- section title “đa-home với attacker tối ưu”;
- Ch6 còn cutoff, distinct-cell/w-event và one-home/one-seed story.

Không cần thêm chart để xác định blockers hiện tại. Khi rerun final nên có:

1. privacy--utility trade-off dưới matched formal budget;
2. averaging error vs `n` với correct attacker và user-cluster intervals;
3. attacker success vs radius với CI;
4. `theta x budget split x resample rate` sensitivity;
5. grid-boundary/cell-count stratification;
6. moving result uncertainty theo user/seed;
7. graph extent/build provenance visualization hoặc machine-checkable report.

## 7. Acceptance gates

### Gate G0 — data và executable kernel: **FAIL**

Đã đạt:

- corrected local graph hash/count/extent verified;
- bbox callsites corrected;
- output aggregates rerun locally.

Chưa đạt:

- finite Gumbel sampler có unreachable outputs;
- finite Laplace threshold của PR chưa có direct output-kernel proof;
- không có sampler reachability/property tests;
- graph/source artifact không tracked hoặc clean-rebuildable;
- output manifest không pin commit/graph/config/raw rows.

### Gate G1 — formal privacy: **FAIL**

Đã đạt:

- R2-001 ideal per-step theorem corrected;
- trivial `D_infinity` length-`w` bound được nêu trong Ch4.

Chưa đạt:

- R2-011 rejection proposition;
- explicit adjacency/window/session/metadata definition;
- executable privacy statement consistent với R2-006;
- executable PR threshold/release kernels chưa được chứng minh theo privacy notion
  đã khai báo;
- public empty-cache/reset precondition của SM đủ explicit;
- Ch1/Ch4/Ch6 scope thống nhất.

### Gate G2 — evaluation validity: **FAIL**

Đã đạt:

- official PR/REM matched step cap;
- Planar road-normalizer bug fixed.

Chưa đạt:

- per-mechanism and jitter-aware likelihoods;
- exact sequential PR official attacker;
- continuous/off-vertex secret model;
- formal-budget equality assertion và config provenance;
- online/benchmark HMM likelihood consistency.

### Gate G3 — statistical evidence: **FAIL**

Đã đạt:

- averaging uses 40 stays, 21 users, 8 seeds;
- cluster bootstrap cho median;
- stay-point caveat trong Ch5.

Chưa đạt:

- moving benchmark multiple independent seeds/users uncertainty;
- order-invariant RNG;
- raw replicate rows và success CIs;
- random/stratified stay sampling;
- leave-return/cross-session evidence nếu home-defense scope được giữ.

### Gate G4 — consistency và artifact QA: **FAIL**

Chưa đạt:

- Ch1/Ch4/Ch6 consistency;
- Ch4 pseudocode equals full-support code;
- README/docs/code docstrings reflect six mechanisms/current graph/current PR;
- response statuses/evidence honest;
- PDF stale-string/content gate.

### Gate G5 — scope và positioning: **FAIL**

Chưa đạt:

- PR naming/state story repo-wide;
- target-on-device vs server-side artifact distinction;
- comparator table and scoped novelty;
- all P0/P1 findings RESOLVED hoặc NARROWED với mandatory caveat.

Chỉ khi G0--G5 pass mới nên nâng overall status từ **Needs revision** lên
**Share with caveats** hoặc **Ready to share**.

## 8. Ordered remediation plan cho Claude Code

### Phase 0 — freeze source of truth

1. Pin candidate `5a7a420`, graph hash `9e44...d487` và current JSON hashes.
2. Không overwrite current JSON trong prose-only phase.
3. Dùng workflow hai commit: implementation/result commit `C` trước; response commit
   `R` sau, với `docs/reviews/response_5a7a420.md` map đủ R2-001...R2-015 và tham
   chiếu `C`. Một file không thể pin hash của chính commit chứa nó.
4. Theo dõi ba dimensions riêng cho mỗi finding: code, evidence/test, manuscript.
5. Chốt mechanism name/scope: previous-release Predictive-REM hay persistent private
   cache. Với code hiện tại, chọn previous-release scope.

**Dependency rule:** không regenerate Ch5 trước khi graph/environment/output manifest
và attacker protocol được freeze. Formal toy work có thể chạy song song.

### Phase 1 — resolve formal blockers

1. R2-006:
   - hạ executable pure-DP claim ngay;
   - chọn ideal-only, quantified approximate privacy hoặc exact verified sampler;
   - add declared-domain property tests.
2. R2-011:
   - fixed-public lemma with positive acceptance;
   - secret-dependent counterexample;
   - finite retry separately.
3. Harden R2-001 proof sum case và add exact test.
4. R2-002:
   - định nghĩa adjacency/metric/session;
   - giữ trivial window bound hoặc implement real accountant.

**Gate:** không còn theorem/proposition tự mâu thuẫn; executable and ideal claims tách
rõ; counterexamples cũ được committed tests falsify.

### Phase 2 — reproducible graph và result manifests

1. Tracked deterministic `build_beijing_graph` script.
2. Pin BBBike source hashes/date và full build environment.
3. Add largest-WCC step và manifest generation.
4. Add runner graph-hash assertion.
5. Stable seed derivation, raw tidy rows và complete output manifest.
6. Aggregate scripts rebuild JSON từ raw rows.

**Gate:** clean clone xác định và tái tạo chính xác graph/config/aggregate provenance.

### Phase 3 — correct attackers

Phase này là bắt buộc nếu luận văn muốn giữ optimal-attacker hoặc comparative-
superiority claims. Với minimum claim-safe path, thay bằng repo-wide proxy relabel,
explicit limitations và không dùng table để kết luận superiority.

1. Per-mechanism likelihood interface.
2. Planar and REM exact declared-domain baselines.
3. T-REM temporal normalizer.
4. SM cache/cell/jitter likelihood.
5. PR sequential likelihood including equality events.
6. GPS jitter marginalization hoặc separate no-jitter exact experiment.
7. Align online HMM and benchmark HMM.
8. Validate all likelihoods on toy finite domains.

**Gate:** “optimal/consistent” chỉ xuất hiện khi domain/prior/loss/likelihood đều
được khai báo và test. Nếu không, retain proxy label.

### Phase 4 — rerun experiments fairly

Full attacker rerun là dependency của stronger evidence path. Một prose-only
minimum path có thể giữ current aggregates như exploratory proxy snapshot, nhưng
vẫn phải sửa RNG/provenance nếu dùng moving comparison như evidence định lượng.

1. Same formal step/window cap across comparable rows.
2. Mechanism-order-independent seeds.
3. Multi-seed moving study, user-cluster uncertainty.
4. Raw stay results, success intervals, random/stratified sample.
5. `theta`, split, grid, jitter and cache/reuse sensitivity.
6. A-B-A and cross-session only if persistent-home claim remains.
7. Archive old outputs by commit/graph/protocol identity; do not silently replace.

### Phase 5 — synchronize research story

1. Ch1: mechanism count, PR scope, w-window wording, attacker names.
2. Ch3: define S1--S8 and add comparator table.
3. Ch4: full-V algorithm, correct graph/runtime, formal propositions.
4. Ch5: six-mechanism config, exact params/attacker labels, remove unsupported flat/
   significant-superiority/pharmacy claims.
5. Ch6: current results and limitations, no old cutoff/one-home story.
6. Sync README, module docstrings, `research_notes.md`, `problem_formulation.md`,
   `system_model_and_threats.md`, `attack_scenarios.md`, simulator budget display.
7. State target-on-device vs Python/Flask artifact explicitly.

### Phase 6 — PDF và response closure

1. Rebuild PDF.
2. Check undefined refs, overfull boxes and visual render.
3. `pdftotext` and stale-string search, excluding historical verifier files.
4. Response commit `R` phải pin exact implementation/result commit `C`, tests,
   output hashes và exact evidence; không yêu cầu `R` tự pin hash của chính nó.
5. Chỉ mark RESOLVED theo handoff protocol ở Mục 11.

## 9. Required tests trước vòng verification tiếp theo

### Formal/kernel tests

- `test_rem_probability_normalizes_on_declared_domain`
- `test_rem_metric_privacy_ratio_on_toy_domain`
- `test_executable_sampler_declared_privacy_notion`
- `test_sampler_sw_far_vertex_counterexample`
- `test_trem_weighted_logit_reachability`
- `test_pr_release_sampler_reachability_for_all_reported_eps`
- `test_pr_finite_laplace_threshold_privacy`
- `test_pr_two_vertex_privacy_bound`
- `test_pr_equal_and_unequal_output_cases`
- `test_pr_full_transcript_bound_by_enumeration`
- `test_pr_state_depends_only_on_public_history`
- `test_pr_window_worst_case_composition`
- `test_fixed_public_conditioning_positive_acceptance`
- `test_secret_dependent_rejection_counterexample`
- `test_finite_retry_distribution_if_retained`
- `test_sm_static_repeat_public_empty_cache_scope`

### Graph/provenance tests

- `test_graph_source_hash`
- `test_graph_nodes_within_bbox`
- `test_graph_is_expected_largest_wcc`
- `test_all_osmnx_bbox_calls_use_left_bottom_right_top`
- `test_graph_manifest_matches_artifact`
- `test_output_manifest_contains_commit_params_seeds_hashes`
- `test_runner_rejects_wrong_graph_hash`
- `test_aggregate_rebuilds_from_raw_rows`

### Attacker tests

- Planar log-likelihood matches analytic density under declared protocol;
- `test_planar_benchmark_uses_constant_normalizer`;
- REM likelihood matches direct normalized kernel;
- T-REM sequential likelihood matches enumeration;
- PR equality/inequality likelihood matches exact kernel;
- SM duplicate releases are not treated as iid fresh draws;
- jitter-aware likelihood recovers simulated centres under its stated conditions;
- `test_online_hmm_uses_declared_normalizer`;
- local candidate search equals full search or carries a proven bound.

### Experiment QA

- mechanism-order permutation leaves seeded per-mechanism outputs unchanged;
- multiple replicates and derived seeds are recorded;
- CIs cluster-resample users, not correlated reports;
- success-rate CI is saved, not only median CI;
- all comparison rows pass formal-budget equality assertion;
- Ch5 table generation checks graph/output manifest hashes;
- current aggregate rebuilds exactly from raw rows.

## 10. Provenance của verifier diagnostics

Các rows sau dùng để falsify/triage. Không copy chúng vào thesis tables như final
evidence trước khi script, raw data và environment được tracked.

| Diagnostic | Inputs/method | Reproducibility status |
|---|---|---|
| Commit/diff audit | Exact commit `5a7a420b5ffce7133c94dfab8dd3c0e1202ba3ab` | Fully commit-scoped |
| Local graph audit | Ignored pickle hash `9e44...d487`; Python 3.11.12/OSMnx 2.0.5 stack | Hash preserved; clean clone missing artifact |
| README graph build | Local BBBike source hashes above; exact README steps, then explicit largest WCC | Method preserved here; standalone script absent |
| Benchmark reproduction | Official loader/order/shared RNG; 20 trajectories, 858 points, 7 users | Aggregates reproduced; raw audit rows not checked in |
| Averaging reproduction | 40 stays, 21 users, 8 seeds, 100 reports, jitter 10m, runner seeds | Aggregates reproduced; raw audit rows absent |
| Two-vertex PR bound | Two vertices 100m apart; theta 200m; both eps 0.01 | Fully specified; should become unit test |
| Finite-domain PR checks | 2,000 random one-step plus adaptive two-step enumeration | Audit-only script not checked in |
| Gumbel reachability | NumPy 2.3.2 53-bit bounds; SW input; full graph; eps 0.01/0.02/0.05 | Fully specified counterexample; add property test |
| Planar attacker diagnostic | Official reports; mean and fixed-centre geometric median; jitter ignored in likelihood | Falsification only; not final jitter-aware MLE |
| PR sequential diagnostic | Current graph; vertex hypothesis; static/no jitter; 40x8 protocol | Falsification only; not continuous/jitter-aware final result |
| PR resample-count diagnostic | Official jittered 40x8 runs; counter includes first mandatory release | Audit-only; counts absent from committed JSON |
| Success CI | User-cluster bootstrap, seed 0, 10,000 draws | Audit-only; raw rows/draws not committed |
| Projection check | 50,000 random vertex pairs, seed 0, projected vs haversine distance | Audit-only; supports percentile wording only |
| PDF QA | Rendered all 30 pages; inspected content and LaTeX warnings | Render temp removed; PDF committed, build log local/ignored |

## 11. Handoff protocol cho response agent

Response tiếp theo nên là `docs/reviews/response_5a7a420.md`. Agent phải tạo một
implementation/result commit `C`, rồi commit response trong commit `R` riêng và để
response tham chiếu `C`; không thể yêu cầu một file pin hash của chính commit chứa
nó. Không sửa hoặc xóa verifier này để làm status trông tốt hơn.

Mỗi finding chỉ được đánh `RESOLVED` khi response có đủ evidence cần thiết trong bốn
nhóm sau:

1. **Code/root cause:** exact file/line và behavior mới.
2. **Proof/test:** committed test hoặc complete proof falsify counterexample cũ.
3. **Result/provenance:** output mới từ correct frozen graph/config, kèm raw/manifest.
4. **Manuscript/consumers:** Ch1--Ch6, README/docs/PDF/UI đồng bộ nếu finding ảnh
   hưởng các consumer đó.

Status rules:

- use `RESOLVED` chỉ khi all applicable acceptance criteria pass;
- use `NARROWED` khi capability mạnh hơn chưa implement nhưng claim đã hạ đúng;
- use `PARTIAL` khi root cause progress có thật nhưng gate còn fail;
- use `PLANNED` khi chưa có code/evidence;
- use `OUT-OF-SCOPE` chỉ với impact và mandatory caveat explicit.

Response hiện tại `docs/reviews/response_0fafd4d.md` có các handoff errors cần tránh:

- không pin candidate commit `5a7a420`;
- không dẫn tests/output hashes cho từng status;
- mark R2-003/R2-005/R2-006/R2-007/R2-008/R2-011/R2-014 quá mạnh;
- mục “Còn mở” bỏ sót nhiều finding PARTIAL/PLANNED;
- nói “không còn claim trong repo” dù archived response hợp lệ vẫn chứa historical
  claim; phải nói “không còn trong active code/manuscript”.

### Definition of done

Vòng sau chỉ nên đề xuất **Share with caveats** khi:

- G0--G5 pass;
- không còn P0/P1 PARTIAL/OPEN dưới claims intended for final thesis;
- exact executable privacy notion được stated honestly;
- correct attacker protocol tạo official comparison tables, hoặc current proxy table
  được relabel/narrowed và không support optimal/superiority claim;
- clean provenance rebuilds graph/results;
- standalone reader của PDF + README + response hiểu đúng cùng scope như code.

Cho tới lúc đó, status chính thức của commit `5a7a420` là:

> **NEEDS REVISION — numerical rerun verified locally; formal executable privacy,
> attacker validity, reproducibility và manuscript consistency chưa pass.**
