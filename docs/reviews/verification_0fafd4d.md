# Verification vòng 2 — PR-SM-REM, attacker MLE và multi-stay study

## Hồ sơ review

| Trường | Giá trị |
|---|---|
| Branch được review | `verifier` |
| Commit kết quả | `0fafd4dda4f4615ae4222db16e444367bfe0d44a` |
| Commit message | `Harden P1: PR-SM-REM (w-event guarantee), MLE attacker, multi-home study` |
| Commit cha | `b77fe27` |
| Ngày review | 2026-08-23 (Asia/Ho_Chi_Minh) |
| Phạm vi | Formal guarantee, implementation, graph/data provenance, attacks, experiments, thesis/docs consistency, industry scope |
| Kết luận | **NEEDS REVISION — chưa được dùng như final verified result** |

Tài liệu này là review độc lập, commit-scoped. Mọi line number trỏ tới commit
`0fafd4d`, không phải một working tree đã được sửa sau review. Verifier cũ
`verification_5b39226.md` được giữ nguyên như lịch sử của vòng trước; tài liệu này
không thay thế provenance của vòng đó.

Phần review code, outputs, thesis và support documents là commit-scoped. Riêng các
spot-check cần raw road graph dùng artifact đang có trong workspace tại thời điểm
review: `data/raw/beijing_graph.pkl`, SHA-256
`72359fe3847cba962ba2b3e537218c82d644777e55e269243d4292e9659125b1`. File này bị
`.gitignore` loại khỏi Git, nên người review ở một clean clone chưa thể tự tái lập
spot-check chỉ từ commit; đây cũng là một phần của finding R2-013.

## 1. Kết luận điều hành

Bản mới đã sửa được nhiều vấn đề quan trọng của vòng trước:

- REM/T-REM đã bỏ explicit secret-centred cutoff trong code và tính logits trên
  toàn bộ array `V`; đây là positive result ở mức ideal real-arithmetic kernel,
  còn executable support chịu caveat R2-006;
- cache miss của SM-REM đã sample tại public representative `rep(cell)`;
- theorem SM-REM đã được thu hẹp đúng về static exact-repeat với public
  empty-cache/reset initial state, không còn tuyên bố arbitrary-trajectory theorem
  trong chính section SM-REM;
- PR-SM-REM dùng một noisy distance test hợp lệ về nguyên lý; ở ideal
  real-arithmetic kernel, equality/revisit channel không còn likelihood ratio vô hạn;
- benchmark table và averaging table trong Chương 5 khớp các JSON đã commit;
- multi-stay protocol (40 stay-points, 21 users, 8 seeds, GPS jitter) tốt hơn rõ
  rệt so với one-home/one-seed illustration;
- HMM candidate coverage đã tăng lên 0.97--1.00.

Tuy nhiên, sáu blocker load-bearing vẫn còn:

1. PR-SM-REM chưa cài một `w`-window budget manager/filter; guarantee cố định của
   code hiện chỉ là worst-case composition, không phải realized-resample budget;
2. bound riêng cho event `{z2 != z1}` bỏ quên privacy loss của fresh REM release;
3. PR-SM-REM được benchmark với `eps_test = epsilon`, nên hard step có privacy
   cost tối đa `2 * epsilon` nhưng vẫn được so với REM dưới cùng nhãn `epsilon`;
4. estimator gọi là "consistent/optimal MLE" thực tế là REM-form iid likelihood áp
   cho tất cả mechanisms: sai mechanism family cho 4/5 rows và không exact end-to-end
   cho cả 5/5 rows trong official continuous-secret/GPS-jitter protocol;
5. road graph 77,727 nodes được build bằng sai thứ tự bbox của OSMnx 2.x. Graph
   thực tế trải khoảng 98 km, không phải bbox GeoLife; toàn bộ normalizer và
   benchmark phải chạy lại sau khi sửa graph;
6. float64 `exp` underflow và cumulative-probability rounding tạo input-dependent
   unreachable outputs. Vì vậy real-number proof chưa đủ để claim pure Geo-I cho
   executable sampler trên declared domain.

### Release decision

Không nên dùng các câu sau làm kết luận cuối của luận văn ở trạng thái hiện tại:

- "PR-SM-REM đạt w-event epsilon_w-Geo-I thật" nếu không kèm primitive composition,
  scalar bound derive từ metric/adjacency rõ ràng (ví dụ `D_infinity`) hoặc budget
  manager đã enforce;
- "PR-SM-REM đạt 47.6 m / 55% trước attacker tối ưu";
- "REM/T-REM tệ hơn Planar Laplace trước averaging MLE vì 25.2 m < 34.4 m";
- "graph 77,727 vertices là graph của cùng bbox GeoLife";
- "full-support implementation thỏa pure Geo-I" mà không nêu numerical domain
  và finite-precision caveat;
- "PR-SM-REM là private persistent memoization cho các lần quay lại nhà".

Framing an toàn tạm thời:

> Commit `0fafd4d` là một industry-aware research prototype. Ideal REM kernel trên
> fixed public support có proof đúng. SM-REM có static-repeat anti-averaging trong
> một cell/cache lifetime. PR-SM-REM là một road-native instantiation của predictive
> reuse làm revisit decision hữu hạn-private, nhưng window accountant, matched-budget
> evaluation và mechanism-aware attacks chưa hoàn tất.

## 2. Bảng findings ưu tiên

| ID | Severity | Trạng thái finding cũ | Kết luận vòng này |
|---|---|---|---|
| R2-001 | P0 / Critical | V-003 claimed resolved | Bound `{z2 != z1}` sai; phải cộng test + REM |
| R2-002 | P0 / Critical | V-003 claimed resolved | Không có w-event accountant; chỉ có trivial worst-case composition |
| R2-003 | P0 / Critical | Mới | PR benchmark không matched privacy budget |
| R2-004 | P0 / Critical | V-004 claimed resolved | Sai family cho 4/5; official protocol không exact cho 5/5 rows |
| R2-005 | P0 / Critical | Mới | Graph bbox build sai; output và normalizer cần chạy lại |
| R2-006 | P0 / Critical | V-001 claimed resolved | Underflow/CDF rounding tạo unreachable outputs |
| R2-007 | P1 / High | V-003 redesign | PR bỏ persistent cache và T-REM; chưa cover leave-and-return home threat |
| R2-008 | P1 / High | V-005 claimed resolved | Multi-stay tốt hơn, nhưng không phải ground-truth homes và claims còn quá mạnh |
| R2-009 | P1 / High | V-012 open | Moving benchmark vẫn one-seed, 7 users, shared sequential RNG |
| R2-010 | P1 / High | V-015 partial | Ch4/Ch6/README/docs/review-response tự mâu thuẫn |
| R2-011 | P1 / High | V-010 partial | Proposition rejection fixed-public set sai như đang phát biểu |
| R2-012 | P1 / High | V-006/V-009 open | Artifact chưa phải on-device system; cache/identifier lifecycle chưa có |
| R2-013 | P2 / Medium | V-012 open | Dependency/data/result provenance chưa đủ clean rerun |
| R2-014 | P2 / Medium | Mới | Novelty của PR phải position như predictive mechanism + REM integration |
| R2-015 | P2 / Medium | V-007/V-011 partial | Một số numerical/narrative claims không khớp output hoặc construct |

## 3. Methodology review và findings chi tiết

### R2-001 — P0: bound cho event khác previous release đang thiếu `epsilon`

**Evidence**

- `core/mechanisms.py:387-405`
- `core/mechanisms.py:355-363`
- `thesis/chapters/ch4_phuongphap.tex:322-344`
- `docs/reviews/response_5b39226.md:27-34`

Đặt:

- `h = z_{t-1}` là previous public release;
- `q_x(h) = P[d(x,h) + Lap(1/eps_test) <= theta]` là xác suất reuse;
- `R_x(z)` là REM kernel với privacy parameter `eps_release`.

Kernel thực tế của PR-SM-REM là:

\[
K_x(z\mid h)=
\begin{cases}
q_x(h)+(1-q_x(h))R_x(h), & z=h,\\
(1-q_x(h))R_x(z), & z\ne h.
\end{cases}
\]

Với event `z != h`, likelihood chứa **cả** test-fail và REM:

\[
K_x(z\mid h)=(1-q_x(h))R_x(z).
\]

Do đó upper bound tổng quát là

\[
\frac{K_x(z\mid h)}{K_{x'}(z\mid h)}
\le \exp((\epsilon_{test}+\epsilon_{release})d(x,x')),
\]

không phải chỉ `exp(eps_test * d)` như thesis và docstring hiện nói.

**Counterexample tối thiểu**

Hai vertices `a=0m`, `b=100m`; previous release `h=a`; `theta=200m`;
`eps_test=eps_release=0.01/m`. Với output event `z_t=b`:

- dưới input `a`: probability (bỏ common previous-step factor) = `0.025547`;
- dưới input `b`: probability = `0.114495`;
- ratio = `4.4817 = exp(1.5)`;
- bound đang claim = `exp(eps_test * 100) = exp(1) = 2.7183`;
- bound composition đúng = `exp((eps_test+eps_release)*100) = exp(2)`.

**Impact**

Noisy test vẫn là một component đúng; lỗi nằm ở theorem/example và accounting của
hard branch. Đây không tái tạo infinite-ratio leak cũ, nhưng làm headline theorem
và comparison budget sai.

**Suggested fix**

1. Đổi mọi per-step statement thành `(eps_test + eps_release)`-Geo-I trong worst
   case; first step chỉ tốn `eps_release`.
2. Trình bày joint mechanism `(reuse_bit, output)` rồi dùng sequential composition;
   output hiện tại là post-processing vì bit không được trả riêng.
3. Nếu muốn path-dependent saving, phải dùng bounded budget manager/filter và lấy
   supremum trên mọi possible public run, không chỉ run đã quan sát.

**Acceptance criteria**

- Có proof cho cả hai cases `z=h` và `z!=h`.
- Toy two-vertex exact enumeration pass bound mới và fail bound cũ.
- Không còn chuỗi `ratio <= exp(eps_test` cho event resample trong repo.

### R2-002 — P0: code chưa cài w-event privacy accountant

**Evidence**

- `core/mechanisms.py:374-405`
- `thesis/chapters/ch4_phuongphap.tex:322-353`
- `web/simulator.py:134-145`

`PrivateReuseSMREM.__init__` chỉ nhận `epsilon`, `eps_test`, `theta`. Không có:

- `w`;
- `epsilon_w`;
- sliding-window ledger;
- privacy filter/odometer;
- dynamic budget allocation;
- behavior khi remaining budget không đủ.

Hai nguồn được cite trong thesis đặt điều kiện mạnh hơn implementation hiện tại:

- Predictive mechanism định nghĩa run-dependent cost nhưng guarantee toàn cơ chế là
  `sup_r epsilon_beta(r)` và dùng một bounded budget manager:
  <https://petsymposium.org/2014/papers/Chatzikokolakis.pdf>.
- Kellaris Theorem 3 yêu cầu `sum epsilon_i <= epsilon_w` trong **mọi** sliding
  window, không phải chỉ đếm hậu nghiệm số hard branches:
  <https://www.vldb.org/pvldb/vol7/p1155-kellaris.pdf>.

Vì Laplace test có full support, resample có xác suất dương ở mọi applicable step.
Condition trên một fixed public pre-window transcript `h`, primitive adaptive
sequential composition trên một window cho:

\[
\frac{P(y_{1:w}\mid x_{1:w},h)}{P(y_{1:w}\mid x'_{1:w},h)}
\leq
\exp\!\left(\sum_{t=1}^{w}\epsilon_t d(x_t,x'_t)\right).
\]

Muốn viết một scalar `epsilon_w`, phải khai báo adjacency/window metric. Dưới
convention `D_infinity(x,x') = max_t d(x_t,x'_t)`, guarantee cố định của code với
constant parameters cho một full window chỉ có thể lấy worst case:

\[
\epsilon_w = w(\epsilon_{test}+\epsilon_{release}),
\]

Nếu window đầu bắt đầu tại `t=1`, nơi initial step chỉ fresh-release:

\[
\epsilon_{release}+(w-1)(\epsilon_{test}+\epsilon_{release}).
\]

Biểu thức đầu giả định test có thể chạy ở mọi step; biểu thức thứ hai tách initial
release. Với một product metric hoặc adjacency khác, phải giữ weighted sum nguyên
thủy và derive scalar bound tương ứng; không được dùng công thức
`w * epsilon` mà không định nghĩa metric. `n_test` và `n_resample` chỉ là counters
sau khi chạy; chúng không enforce một guarantee ex-ante.

**Observed diagnostic, không phải guarantee**

Trên đúng 40 stay-points x 8 seeds, 100 reports, `epsilon=eps_test=0.02`:

| Statistic | Value |
|---|---:|
| PR resamples / 100, min | 1 |
| median | 10 |
| mean | 11.55625 |
| p95 | 26 |
| max | 42 |
| Executed-operation approximation, median | 2.18 = `99*0.02 + 10*0.02` |
| Code-counter/thesis-style conservative account | 2.20 = `100*0.02 + 10*0.02` |
| REM 100-release composition | 2.00 |
| Valid PR ex-ante worst case, first window under `D_infinity` | 3.98 |

`n_test` hiện tăng cả ở initial release dù step đó không sample Laplace test; vì vậy
counter-style `2.20` cao hơn executed-operation approximation `2.18`. Cả hai đều là
post-hoc diagnostics, không phải ex-ante privacy guarantee.

**Suggested fix — hai lựa chọn**

**Lựa chọn A, thesis-safe và ít code:**

- không claim budget saving hình thức;
- phát biểu per-step worst case `eps_test + eps_release`;
- phát biểu product-metric composition trước, rồi mới derive w-window scalar bound
  theo metric/adjacency đã khai báo;
- gọi observed resample count là empirical diagnostic, không gọi privacy guarantee.

**Lựa chọn B, contribution mạnh hơn:**

- thêm `w`, `epsilon_w` và per-step allocated bounds;
- trước mỗi step, tính budget còn lại sau các allocations trong `w-1` timestamps;
- chọn `eps_test_t`, `eps_release_t` sao cho bound của step không làm window vượt
  `epsilon_w`;
- khi không còn budget, output một public/state-only fallback hoặc dừng release theo
  policy đã định nghĩa;
- proof phải lấy supremum trên mọi public transcript/branch.

**Acceptance criteria**

- Unit test trượt qua stream dài hơn `3w` và assert mọi window ledger `<=epsilon_w`.
- Adversarial test force hard branch ở mọi step vẫn không vượt budget.
- UI/report hiển thị `epsilon_w`, `w`, split test/release và actual publication rate;
  không dùng `T*epsilon` chung cho mọi mechanism.

### R2-003 — P0: PR-SM-REM không được so sánh ở cùng privacy guarantee

**Evidence**

- `core/mechanisms.py:374-377`
- `experiments/run_benchmark.py:42-64`
- `experiments/run_averaging_multi.py:38-56`
- `thesis/chapters/ch5_thucnghiem.tex:17-23,152-188`

Default `eps_test = epsilon`. Runner gọi `PrivateReuseSMREM(eps, rn)` mà không
override. Vì vậy:

- REM hard release: `epsilon`;
- PR reuse step: `eps_test = epsilon`;
- PR resample step: `eps_test + eps_release = 2 * epsilon`.

Nhưng Chương 5 đặt PR cùng block `epsilon=0.01/0.02/0.05` như thể guarantee giống
REM. Utility advantage của PR có thể một phần đến từ việc chi nhiều privacy hơn.

**Suggested fix**

Chọn một public comparison budget, ví dụ `epsilon_step_total` hoặc
`(w, epsilon_w)`, rồi split:

\[
\epsilon_{test}=\rho\epsilon_{step,total},\qquad
\epsilon_{release}=(1-\rho)\epsilon_{step,total}.
\]

Nếu dùng window accountant, mọi mechanism phải được so trên cùng protected unit và
cùng worst-case `epsilon_w`, không cùng tên biến constructor.

Chạy sensitivity sweep cho `rho` và `theta`; một point `theta=200m` không đủ để
kết luận đây là optimal/balanced design.

**Acceptance criteria**

- Table config ghi rõ `eps_test`, `eps_release`, `theta`, `w`, `epsilon_w`.
- Mọi row trong một comparison group có cùng formal worst-case guarantee.
- Có ablation/sweep `theta x budget split`, kèm utility, resample rate và attack error.

### R2-004 — P0: "consistent/optimal MLE" vẫn misspecified

**Evidence**

- `evaluation/attacks.py:240-328`
- `evaluation/attacks.py:75-120,200-225`
- `experiments/run_averaging_multi.py:82-101`
- `experiments/run_benchmark.py:84-99`
- `outputs/averaging_multi_results.json`
- `thesis/chapters/ch5_thucnghiem.tex:124-194`
- `docs/reviews/response_5b39226.md:36-44`

`AveragingAttack._mle` luôn tối ưu:

\[
-a\sum_i d(x,z_i)-n\log Z_{REM}(x),
\]

trên candidate `x` là road vertices. Runner dùng cùng estimator cho Planar Laplace,
REM, T-REM, SM-REM và PR-SM-REM.

Nó chỉ là iid REM MLE khi đồng thời:

- emission đúng là REM;
- true secret domain được giới hạn ở road vertices;
- observations iid;
- không có GPS jitter hoặc jitter đã được đưa vào likelihood.

Các điều kiện đó không đúng cho bốn mechanism còn lại, và thậm chí true stay-point
centres của REM experiment là continuous coordinates, không phải vertices. Vì GPS
jitter cũng không nằm trong likelihood, không row nào trong 5 official rows hiện
thỏa toàn bộ assumptions cần để gọi estimator là exact end-to-end MLE.

#### Planar Laplace discrepancy

Trong JSON đã commit tại `n=100`:

- sample mean median = `10.1m`, CI `[9.2, 11.1]`;
- estimator được gọi là MLE median = `34.4m`;
- reported success within 50m = `73.1%`.

Reproduction độc lập trên cùng 40 locations x 8 seeds:

| Planar attacker | Median error @100 | P(error <= 50m) |
|---|---:|---:|
| Stored REM-form MLE | 34.4m | 73.1% |
| Jitter-ignorant road-vertex Planar diagnostic, constant normalizer | 19.90m | 91.9% |
| Jitter-ignorant fixed-centre geometric-median diagnostic | 9.04m | 100% |
| Stored sample mean | 10.13m | 100% |

Vì vậy câu "REM/T-REM còn tệ hơn Laplace ở n=100: 25.2m so 34.4m" không được
support. Hai diagnostic mạnh hơn không phải exact MLE của Gaussian-jitter convolution,
nhưng đủ cho thấy Planar hiện dễ bị recover hơn reported estimator gợi ý.

Point/HMM benchmark cũng truyền discrete road normalizer cho Planar tại
`experiments/run_benchmark.py:84`, rồi `BayesianPointAttack` trừ `log Z(x)`. Planar
continuous density có normalizer constant. Tại `epsilon=0.02`, committed Bayes error
là `100.5319m`; bỏ road normalizer cho Planar cho `99.3311m`. Chênh lệch headline nhỏ
nhưng construct "exact Planar likelihood" vẫn sai; HMM Planar cũng nhận normalizer
không phù hợp. Web `online_estimates` lại không dùng `log Z`, nên benchmark và online
attacker hiện chưa cùng declared likelihood.

#### PR-SM-REM sequential-likelihood diagnostic

Sequential kernel đúng, nếu bỏ jitter và coi một static secret `x`, là:

\[
P(z_1\mid x)=R_x(z_1),\qquad
P(z_{2:T}\mid x,z_1)=\prod_{t=2}^{T}K_x(z_t\mid z_{t-1}),
\]

với `K` ở R2-001. Equality/run-length tự nó là evidence về khoảng cách từ `x` tới
previous release. Iid REM proxy bỏ hoàn toàn evidence này.

Diagnostic không-jitter trên 40 locations x 8 seeds:

| PR attacker | Median error @100 | P(error <= 50m) |
|---|---:|---:|
| Current iid REM proxy | 47.81m | 53.1% |
| Correct PR sequential kernel on current candidate domain, no jitter | 34.37m | 73.1% |

Đây **không phải** exact continuous-domain attacker hay replacement cho table chính
thức vì official protocol có `sigma=10m` jitter. Nó là falsification test cho claim
"attacker tối ưu": chỉ cần dùng đúng sequential kernel trên candidate domain hiện tại
đã tạo ra attacker mạnh hơn đáng kể.

#### Likelihood cần cài theo mechanism

1. **Planar:** continuous density có normalizer constant; với jitter, tích phân hoặc
   Monte Carlo convolution `p(z|home)=int p(z|x)p(x|home)dx`.
2. **REM:** xác định rõ secret domain là `R^2` hay `V`. Nếu `R^2`, không được gọi
   vertex-restricted optimizer là exact MLE.
3. **T-REM:** dùng history-dependent normalizer
   `Z_t(x,z_{t-1}) = sum_v w_t(v;z_{t-1}) exp(-a d(x,v))`.
4. **SM-REM:** model cell representative, cache state, equality events và GPS fixes
   crossing cell boundaries; không nhân duplicate memoized release như iid evidence.
5. **PR-SM-REM:** dùng sequential kernel ở trên và jitter-aware latent inputs.

**Acceptance criteria**

- Mỗi mechanism có explicit `log_likelihood(history, candidate_secret, params)`.
- Toy domains được brute-force enumeration để đối chiếu log-likelihood.
- Planar attacker không dùng road REM normalizer.
- Thesis chỉ dùng "optimal/consistent" khi threat model, domain và likelihood khớp.
- Chạy lại toàn bộ averaging table sau khi attacker và graph đều được sửa.

### R2-005 — P0: road graph được build bằng sai thứ tự bbox

**Evidence**

- `data/README.md:17-39`
- `data/geolife.py:24-27`
- `core/road_network.py:17-55`
- `core/trajectory_privacy.py:50`
- `web/app.py:117`
- `web/app_optimized.py:158`
- `web/trajectory_privacy_optimized.py:57`
- `thesis/chapters/ch5_thucnghiem.tex:6-15`

`BEIJING_BBOX` có format:

```text
(min_lat, min_lon, max_lat, max_lon)
= (39.96, 116.29, 40.02, 116.36)
```

Recipe hiện truyền:

```python
bbox=(b[2], b[0], b[3], b[1])
```

Sai ordering không chỉ nằm trong recipe data: các call `graph_from_bbox` và
`features_from_bbox` ở core/web files liệt kê trên cũng dùng convention cũ. Chỉ sửa
benchmark graph sẽ để app và trajectory pipeline tiếp tục lấy sai vùng.

OSMnx 2.0.5 `truncate_graph_bbox` yêu cầu
`(left, bottom, right, top)`, tức đúng ra phải là:

```python
bbox=(b[1], b[0], b[3], b[2])
```

Graph pickle local nêu trong hồ sơ review (không được Git track) có extent đo được:

| Dimension | Actual graph | Intended bbox |
|---|---:|---:|
| latitude | 39.9600034 -- 40.3549195 | 39.96 -- 40.02 |
| longitude | 115.3293228 -- 116.3599998 | 116.29 -- 116.36 |
| projected bounding-box diagonal | khoảng 98,113m | khoảng 8--9km |
| nodes | 77,727 | 19,634 khi cắt đúng current graph |
| edges | 208,994 | 47,745 khi cắt đúng current graph |

Graph hiện giữ phần lớn vùng phía tây/bắc và đặt GeoLife box ở gần góc đông-nam.
Điều này ảnh hưởng:

- fixed output domain `V`;
- REM/T-REM/SM/PR normalizers;
- boundary bias và MLE;
- runtime/memory claims;
- projection accuracy;
- underflow và actual numerical support;
- mọi committed experimental number.

Claim projection "well under 1m over a 10km box" cũng không còn áp cho graph 98km.
Sampling 50,000 vertex pairs cho projected-vs-haversine absolute error median
`1.45m`, p95 `12.40m`, sampled max `148.62m`.

**Suggested fix**

1. Sửa bbox order và regenerate graph.
2. Quyết định public output domain: toàn graph đã crop hay largest connected road
   component; ghi rõ và giữ độc lập với secret.
3. Lưu graph theo format/version ổn định hơn pickle nếu có thể, hoặc pin đầy đủ
   Python/NumPy/Shapely/OSMnx.
4. Ghi extent, node/edge count, component count và SHA-256 vào manifest.
5. Chạy lại precomputed normalizers, benchmark, averaging và simulator artifacts.

**Acceptance criteria**

- Assert mọi graph node nằm trong bbox với tolerance đã nêu.
- Repo-wide test/search xác nhận mọi OSMnx 2.x bbox call dùng
  `(left, bottom, right, top)`, gồm cả graph và feature queries.
- Manifest chứa source URL/date/hash, build command, dependency versions và graph hash.
- Thesis node/edge count khớp artifact mới.
- Không reuse JSON hiện tại sau khi graph thay đổi.

### R2-006 — P0/Critical: formula full-support nhưng executable có unreachable outputs

**Evidence**

- `core/mechanisms.py:161-170`
- `thesis/chapters/ch4_phuongphap.tex:169-179`

`_sample` trừ max logit rồi gọi `np.exp`. Với float64, sufficiently negative logits
underflow về đúng `0`. Đây là một input-dependent support failure ở executable dù
ideal formula có positive mass trên mọi vertex.

Diagnostics trên graph sai-extent hiện tại:

- tại một boundary vertex, `epsilon=0.02`: 56,880 / 77,727 outputs có numerical
  probability zero;
- tại `epsilon=0.05`, ngay trong intended GeoLife bbox, support mask thay đổi theo
  input corner: SW 26 zero entries, SE 1,051, NW 6, NE 13;
- do có vertex zero dưới một input nhưng positive dưới input khác, executable không
  thỏa literal pure Geo-I trên domain đó.

Sửa graph về bounding-box diagonal khoảng 9km giảm logit range và loại `np.exp == 0`
trong spot-check, nhưng **không** tự khôi phục executable reachability. Với induced
correct crop 19,634 vertices, input ở góc tây-nam:

| `epsilon` | `exp`-zero entries | CDF zero-width bins |
|---:|---:|---:|
| 0.01 | 0 | 1,434 / 19,634 |
| 0.02 | 0 | 14,791 / 19,634 |
| 0.05 | 0 | 18,979 / 19,634 |

Current `np.random.Generator.choice(p=...)` dùng finite-precision cumulative
probabilities. Nhiều positive `p_i` quá nhỏ để làm `cumsum` tăng, nên category tương
ứng có interval width bằng zero và không reachable qua sampler. Tại `epsilon=0.05`,
minimum probability khoảng `6.88e-97`; 19,120 entries nhỏ hơn `2^-53`.

**Suggested fix**

- sửa graph trước;
- thiết kế sampler có declared finite-randomness semantics và kiểm tra actual
  reachability; log-domain/Gumbel sampling tự nó không phải proof vì RNG vẫn hữu hạn;
- thêm property test cho toàn declared input domain/epsilon grid;
- nếu không thể chứng minh finite-precision pure DP, phân biệt rõ "ideal mathematical
  kernel" với "floating-point implementation" và nêu approximate/numerical caveat.

**Acceptance criteria**

- Với graph/epsilon domain được publish, kiểm tra cả `p_i > 0`,
  `diff(cumsum(p)) > 0` và actual sampler reachability/proven bound; không dùng một
  log-domain rewrite đơn thuần để đóng finding.
- Test chạy ở bbox corners, all observed true points và worst-distance vertices.
- Thesis không suy guarantee executable chỉ từ real-number proof nếu test này fail.

### R2-007 — P1: PR-SM-REM đổi bài toán, chưa privately memoize old locations

**Evidence**

- `core/mechanisms.py:336-405`
- `core/mechanisms.py:228-333`
- `thesis/chapters/ch4_phuongphap.tex:301-353`

`PrivateReuseSMREM` subclass `RoadExponential`, không subclass T-REM/SM-REM. State
chỉ giữ **immediately previous released vertex**. Khi resample ở nơi khác, old home
release bị quên. Cơ chế vì vậy là parrot predictive reuse, không phải private lookup
trong persistent location cache.

Hệ quả:

- cover tốt hơn cho một contiguous static dwell;
- không tự cover pattern `home -> commute -> work -> commute -> same home`;
- 100 consecutive reports ở một stay-point không test nightly/cross-session home
  inference mà industry threat model nhấn mạnh;
- fresh branch là REM, không dùng T-REM reachability weighting;
- SM-REM cũ trên cache hit cũng bypass current reachability penalty và có thể jump
  về cached vertex cũ.

**Suggested fix**

Chọn và nói rõ một trong hai scope:

1. **Predictive REM:** rename/position PR là private reuse của previous release cho
   consecutive correlated reports. Đây là scope phù hợp code hiện tại.
2. **Private persistent memoization:** thiết kế private cache query/selection, cache
   lifecycle và reachability interaction. Đây là một bài toán khó hơn; không thể chỉ
   noisy-test khoảng cách tới previous release.

Thêm experiment interleaved return pattern và cross-session reset/persistence trước
khi claim home-revisit defense.

**Acceptance criteria**

- Threat scenario trong thesis khớp state machine thật.
- Có experiment `A,B,A,B,...` hoặc nightly home sequence, không chỉ `A,A,A,...`.
- Nếu giữ tên SM-REM, giải thích cache nào tồn tại; nếu không có cache, đổi tên để
  tránh làm reviewer hiểu sai.

### R2-008 — P1: multi-stay experiment là improvement nhưng chưa phải home evidence

**Evidence**

- `data/geolife.py:136-200`
- `experiments/run_averaging_multi.py:38-138`
- `outputs/averaging_multi_results.json`
- `thesis/chapters/ch5_thucnghiem.tex:140-194`

Loader tìm generic stay-point 200m/20min rồi đặt field name `home`. Không có:

- ground-truth residence label;
- nighttime filter;
- recurrent-home criterion;
- address/residential-landuse validation;
- random population sampling.

Nó lấy 40 qualifying stays đầu tiên theo sorted user/file order, tối đa 2/user.
Do đó manuscript nên gọi là **stay/significant locations**, không gọi 40 homes thật.

SM-REM row cũng không "flat ở mọi n": MLE proxy thay đổi `159.2m -> 147.6m` vì
GPS jitter đưa input qua nhiều grid cells. Reproduction cho thấy distinct cached cells
trong 100 reports: min 1, median 3, mean 3.153, max 4.

Cluster bootstrap median implementation nhìn chung đúng hướng. Tuy nhiên:

- table thesis bỏ CI dù JSON có CI;
- success probability chưa có CI;
- aggregate JSON không cho audit heterogeneity theo user/home/boundary distance.

Diagnostic cluster CI cho success@50m:

| Mechanism | Point estimate | Approx. 95% user-cluster CI |
|---|---:|---:|
| SM-REM | 10.6% | 6.9% -- 14.5% |
| PR-SM-REM | 55.0% | 45.6% -- 64.3% |

Các CI này chỉ mô tả current proxy attacker/current graph; phải recompute sau fixes.

**Suggested fix**

- relabel dataset unit thành stay-point;
- nếu muốn home claim, định nghĩa và validate residence inference protocol;
- random/stratified sample users/locations;
- lưu raw per-location x seed x n x mechanism errors và metadata không nhạy cảm;
- report CI cả median curve lẫn success probability;
- stratify theo distance-to-grid-boundary và number of distinct cells.

### R2-009 — P1: moving benchmark chưa có uncertainty và independent seeding

**Evidence**

- `data/geolife.py:81-133`
- `experiments/run_benchmark.py:67-145`
- `outputs/benchmark_results.json`

20 trajectories thực tế chỉ thuộc 7 users. Benchmark dùng một root seed và truyền
cùng một mutable RNG object tuần tự qua mechanisms. Kết quả phụ thuộc order của
mechanism list; không có per-user cluster CI hoặc repeated mechanism seeds.

Positive verification: chạy lại utility/realism metrics tại `epsilon=0.02` khớp
committed JSON đến sai số dưới `3e-14`. Nghĩa là output không stale, nhưng
reproducibility của một deterministic run không thay thế uncertainty/generalization.

**Suggested fix**

- derive seed độc lập từ tuple `(root_seed, mechanism, epsilon, user, trajectory,
  replicate)` bằng `SeedSequence` hoặc stable hash;
- dùng paired seeds across mechanisms nhưng không dùng shared mutable stream;
- nhiều replicates và cluster CI theo user;
- lưu selected user/file IDs và raw per-trajectory metrics;
- tránh causal wording "do reuse" nếu chưa có ablation.

### R2-010 — P1: thesis và support documents chưa đồng bộ

**Evidence examples**

- `thesis/chapters/ch4_phuongphap.tex:224-229` vẫn nói SM siết composition theo
  distinct locations, trái `:275-287`;
- `thesis/chapters/ch4_phuongphap.tex:356-370` vẫn mô tả KD-tree secret-centred
  `V_c = V intersect B(x,R)`, trái code/full-support proof;
- `thesis/chapters/ch5_thucnghiem.tex:17-23` liệt kê năm mechanisms, bỏ PR params;
- `thesis/chapters/ch6_tongket.tex:12-42` vẫn nói T-REM đóng velocity channel,
  cutoff còn tồn tại, w-event là future, averaging one-home/seed;
- `thesis/chapters/ch1_gioithieu.tex:24-47` nói "hai cơ chế" nhưng itemize bốn cơ
  chế, đồng thời claim PR đạt w-event thật;
- `thesis/chapters/ch4_phuongphap.tex:45-58` dùng taxonomy S1--S6 và trỏ Ch3,
  nhưng Ch3 không định nghĩa taxonomy S1--S8 nên PDF chưa self-contained;
- `README.md:18-22,36-60` chưa đưa PR vào mechanism/command count;
- `docs/research_notes.md:84-117` còn bảng cũ, nói T-REM đóng attack và chưa có
  SM/PR;
- `docs/problem_formulation.md:38` gọi approximate-geolocation đúng là Google/WICG
  proposal, nhưng `:46-49` lại gọi nó là W3C anti-averaging spec;
- chính `docs/problem_formulation.md:46-49` còn gộp SM-REM vào scope thỏa Euclidean
  R3, trái caveat/pseudometric discussion ở Ch4;
- `docs/problem_formulation.md:63-72` còn claim distinct-cell composition và public
  grid không leak;
- `docs/system_model_and_threats.md` và `docs/attack_scenarios.md` vẫn mô tả snapshot
  trước memoization/private reuse;
- `docs/reviews/response_5b39226.md` trỏ file `outputs/averaging_results.json` đã xóa,
  có V-005 vừa RESOLVED vừa FUTURE, và vẫn liệt kê multi-home là open;
- `core/mechanisms.py:8-47` module docstring nói mọi mechanism spend epsilon/point,
  nhưng không liệt kê SM/PR.

PDF hiện build được 29 trang và không có undefined references, nhưng có overfull
boxes. Đây là presentation caveat, không phải formal blocker; contradictions ở trên
mới là blocker.

**Suggested fix**

Sau khi code/results ổn định, chọn một source-of-truth và sync theo thứ tự:

1. formal definitions/theorems;
2. implementation and tests;
3. experiment configs/raw outputs;
4. Ch1/Ch3/Ch4/Ch5/Ch6;
5. README/web/docs;
6. một `response_0fafd4d.md` map từng finding của file này sang commit/test/output.

Không sửa verifier lịch sử để làm finding cũ trông như đã pass từ đầu.

### R2-011 — P1: proposition về fixed-public rejection đang sai

**Evidence**

- `thesis/chapters/ch4_phuongphap.tex:91-107`
- `core/trajectory_privacy.py:38-110,126-141`

Proposition hiện định nghĩa một forbidden set `F` rồi nói conditioning trên
`z notin F` nói chung không còn finite Geo-I. Ngay paragraph sau lại thừa nhận fixed,
public conditioning giữ một `2*epsilon` upper bound.

Nếu `K` là epsilon-metric-private, `A=F^c` là fixed/public và
`K(A|x) > 0` cho mọi input trong declared domain, thì:

\[
K_A(z\mid x)=\frac{K(z\mid x)}{K(A\mid x)}
\]

nhận một factor `exp(epsilon*d)` từ numerator và tối đa một factor nữa từ acceptance
normalizer. Vì vậy fixed-public conditioning không support proposition/title hiện tại.

Trong internship code, building/water/road geometry là public output predicate.
Input-dependent QoS cap/check, alternative-road search và finite retry/fallback cần
phân tích riêng; radius cap đã đủ để phá pure Geo-I.

**Suggested fix**

- rút proposition fixed-F hiện tại;
- tách ba cases: fixed public conditioning, secret-dependent support/predicate, finite
  retry/fallback;
- chỉ claim điều đã proof cho đúng original pipeline;
- giữ radius-cap proposition làm formal failure chính nếu không cần mở rộng.

**Acceptance criteria**

- Corrected proposition ghi rõ fixed/public `A` và positive acceptance condition.
- Toy test đối chiếu bound `2*epsilon`; secret-dependent/zero-acceptance cases không
  bị nhập chung với fixed-public conditioning.

### R2-012 — P1: industry architecture là target, chưa phải implemented system

**Evidence**

- `thesis/chapters/ch4_phuongphap.tex:21-40`
- `web/simulator.py:49-106,134-145`
- `core/mechanisms.py:294-333,380-405`

Thesis nói mechanism chạy client-side/on-device, nhưng artifact demonstrator là Flask
server nhận raw coordinates và chạy Python mechanism. Đây là một architecture target,
không phải deployed mobile implementation.

Các vấn đề industry còn chưa cài:

- privacy principal và identifier rotation;
- cache persistence/TTL/eviction/restart/map-update;
- secure local storage và deletion;
- context/purpose-based granularity;
- metadata/timestamp/query leakage;
- latency/energy/memory trên phone;
- app SDK/API contract và accuracy semantics.

On-road output loại một cue off-road nhưng không loại topology/prior/transition hoặc
learned reconstruction. RAoPT chưa được chạy. T-REM là soft Euclidean displacement
regularizer, không phải hard road reachability guarantee.

**Required wording**

Gọi artifact là "on-device-targeted research prototype" hoặc "reference mechanism
and server-side simulator". Không gọi là shipped client-side system.

### R2-013 — P2: reproducibility/provenance chưa đủ clean rerun

**Evidence**

- `requirements.txt`
- `data/README.md`
- `outputs/benchmark_results.json`
- `outputs/averaging_multi_results.json`

Dependencies chỉ có lower bounds, không lock. `.venv` local với NumPy 1.24.3 không
load được graph pickle được tạo dưới NumPy 2 layout (`numpy._core`); `venv` khác với
NumPy 2.3.2 load được. Pickle phụ thuộc Python/scientific-stack versions.

JSON chưa lưu:

- commit hash;
- graph/data hashes và extents;
- selected users/files/stay IDs;
- full mechanism params (`theta`, `eps_test`, grid, temporal params);
- derived seeds;
- dependency versions;
- raw replicate records;
- wall-clock/hardware context riêng khỏi attack runtime.

**Suggested fix**

- pin/lock environment;
- tạo `outputs/manifest.json` hoặc embed manifest trong mỗi output;
- lưu graph bằng reproducible build script và stable format;
- lưu raw tidy results (`jsonl`, CSV hoặc Parquet) rồi derive aggregate JSON;
- thêm command `make reproduce`/script không overwrite verified results trừ khi explicit;
- CI smoke test load graph, run toy mechanisms và validate schema.

### R2-014 — P2: novelty cần position lại

PR-SM-REM hiện dùng đúng ba components đã có trong predictive-mechanism paper:

- parrot predictor = previous release;
- private noisy threshold quality test;
- fresh noise mechanism khi test fail.

Primary source: Chatzikokolakis, Palamidessi, Stronati, PETS 2014:
<https://petsymposium.org/2014/papers/Chatzikokolakis.pdf>.

Do đó contribution defensible hơn là:

> instantiation/integration của predictive reuse với a road-output REM kernel, cộng
> threat-aligned evaluation trên repeated stay observations.

Không nên position private reuse primitive hoặc parrot mechanism là hoàn toàn mới.
Tương tự, REM là Euclidean-metric specialization/sibling của graph exponential
mechanism; SM exact memoization có precedent ở predictive/memoization/location
privacy literature. Novelty có thể nằm ở tổ hợp, formalization đúng, implementation
road-native và evaluation, nhưng cần comparator/direct related-work table.

Claude agents nên lập bảng nearest-precedent tối thiểu cho:

- predictive mechanism PETS 2014 (parrot/private test/budget manager);
- RAPPOR permanent randomized response (stability/memoization precedent, không phải
  location mechanism trực tiếp);
- LP-Doctor per-place/per-protection-level cached release, với primary source đã link
  trong `verification_5b39226.md:235`;
- Release-GeoInd kiểu repeat-until-moved và công trình Computer Networks 2026 về
  memoization/replay/replication đã được literature audit trước gợi ra; hai item này
  phải được verify lại bằng primary metadata/source trước khi đưa vào thesis.

Mỗi row nên phân biệt protected unit, state lifetime, reuse decision, formal
guarantee, road support và attacker/evaluation. Eclipse là related work đang được
cite, chưa phải "direct competitor" nếu chưa có implementation-aligned benchmark.

Nếu muốn claim algorithmic novelty mạnh hơn, cần chỉ ra component/theorem nào không
suy trực tiếp từ predictive mechanism + REM + standard composition.

### R2-015 — P2: numerical/narrative và construct discrepancies

Các ví dụ cần cleanup sau rerun:

- Ch5 nói T-REM có 0% speed violation ở `epsilon=0.02`; JSON là `0.02238` (~2.2%);
- Ch5 simulator use case nói k-NN recall khoảng 81%; current JSON tại epsilon=0.02
  khoảng 97--100%, và POIs là synthetic road vertices, không phải pharmacies;
- "SM-REM flat ở mọi n" trái chính row `159.2 -> 147.6`;
- HMM/Bayes attacker dùng REM-form likelihood; ngay cả REM official row còn lệch
  continuous-secret/jitter protocol, và PR bị bỏ khỏi caveat list trong setup;
- nearest-road metric là nearest-vertex, không point-to-edge;
- single benchmark runtime bao gồm attacks, không chứng minh mechanism latency dưới
  1ms; chưa có isolated microbenchmark manifest đủ warm-up/repetitions/hardware;
- graph 98km làm projection claim under-1m không còn đúng;
- Strava endpoint-zone recovery là route-boundary/road-gate inference, không phải
  direct evidence cho iid arithmetic averaging; Ch4/Ch5 phải tách hai constructs;
- Eclipse chỉ nên gọi related comparator cho đến khi được benchmark cùng threat model;
- success rate cần CI và attacker-direction label rõ: higher success = worse privacy.

## 4. Calculation spot-check summary

| Claim/metric | Status | Evidence |
|---|---|---|
| Ch5 benchmark values match committed JSON | Verified | Rounded values reconcile; epsilon=0.02 utility rerun matches <3e-14 |
| Ch5 averaging medians SM=147.6, PR=47.6 under current code | Verified | Reproduced 40 stays x 8 seeds |
| These are results against optimal mechanism-aware attacker | **Not verified / contradicted** | Planar and sequential-PR diagnostics above |
| PR event `{z2!=z1}` costs only eps_test | **False** | Two-vertex ratio 4.4817 > exp(1) |
| Current PR has non-trivial enforced w-event budget | **False** | No w/accountant/filter; worst case 3.98 at w=100, eps=.02 |
| PR and REM rows use matched privacy guarantee | **False** | PR hard branch costs 2epsilon by default |
| Graph 77,727 nodes is same GeoLife bbox | **False** | Correct crop gives 19,634 nodes / 47,745 edges |
| Ideal fixed-support REM theorem | Verified at real-number kernel level | Triangle-inequality proof is valid |
| Executable full-support at all benchmark eps | **False** | `exp` underflow and CDF zero-width bins make categories unreachable |
| SM static exact-repeat theorem | Verified only for ideal kernel, public empty-cache/reset state | Transcript is post-processing of one sample |
| Planar benchmark uses its declared exact likelihood | **False** | Point/HMM receive a road-REM normalizer; web path differs again |
| SM remains exactly flat with 10m GPS jitter | **False** | Median 3 distinct cells; errors change with n |
| Multi-"home" population is ground-truth residences | **False** | Generic 200m/20min stays, no residence validation |
| PDF references compile | Verified | 29-page PDF, no undefined refs; overfull boxes remain |

## 5. Visualization/presentation review

Không cần thêm chart để xác định formal blockers; table trên là đủ. Khi rerun final,
nên có tối thiểu:

1. privacy-utility Pareto plot với **matched formal budget**;
2. averaging error vs `n`, CI cluster theo user, tách đúng attacker per mechanism;
3. attacker success vs radius với CI;
4. resample rate/budget split vs `theta`;
5. distribution theo distance-to-grid-boundary;
6. moving benchmark CI theo user, không chỉ point estimate;
7. graph extent map để chứng minh data domain đúng.

Chart title phải nói rõ `current graph`, `attacker`, `epsilon unit`, `w`, `theta`, số
users, seeds và liệu higher/lower là tốt cho privacy.

## 6. Kế hoạch sửa đề xuất cho Claude agents

### Phase A — Freeze và chỉnh scope trước khi sửa code

1. Không chỉnh/xóa verifier lịch sử.
2. Tạo `docs/reviews/response_0fafd4d.md` map R2-001...R2-015.
3. Quyết định mechanism cuối là:
   - predictive previous-release reuse; hay
   - private persistent cache.
4. Quyết định protected unit:
   - per release;
   - product-metric trajectory;
   - hoặc `(w, epsilon_w)` sliding window.
5. Định nghĩa secret/input domain và public output graph domain.
6. Gắn current JSON với old graph SHA-256 và archive snapshot trước khi sinh output mới.

### Phase B — Fix graph và numerical kernel trước mọi rerun

1. Sửa bbox order.
2. Regenerate graph và manifest.
3. Validate extents/components/projection.
4. Fix/test numerical sampler support.
5. Archive old derived outputs bằng recoverable workflow dưới old graph hash; không
   overwrite và không để old/new graph results trộn nhau.

**Hard dependency cho multi-agent execution:** freeze corrected graph hash và locked
environment trước mọi formal-numeric, attacker hoặc experiment run dùng để đóng
findings. Không để Phases C--E chạy số trên các graph artifacts khác nhau. Symbolic
proof/toy-domain work có thể song song, nhưng manuscript agent không được regenerate
Ch5 cho tới khi artifacts của Gates G0--G3 đã freeze.

### Phase C — Formalize PR mechanism

Minimum viable theorem:

- first step: `eps_release`-Geo-I;
- later step conditional on public history:
  `(eps_test + eps_release)`-Geo-I;
- trajectory product-metric composition stated explicitly;
- fixed-parameter window guarantee stated as worst case.

Stronger option:

- implement `w, epsilon_w` accountant/filter;
- prove bound for all transcripts and forced hard branches;
- expose allocated budgets in output manifest.

### Phase D — Build exact/declared attackers

1. Define threat model and prior before observation.
2. Implement mechanism-specific likelihoods.
3. Validate each on toy finite domains by brute force.
4. Handle GPS jitter as latent measurement noise or remove it from the "exact MLE"
   experiment and make a separate robustness experiment.
5. Separate:
   - exact mechanism-aware attacker;
   - deliberately approximate HMM;
   - naive mean baseline.

### Phase E — Rerun experiments fairly

1. Same formal privacy guarantee across mechanisms.
2. Correct graph and locked environment.
3. Multi-user/multi-seed moving and stay experiments.
4. Raw records + manifest + aggregate script.
5. User-cluster CI and success CI.
6. `theta`, budget split, grid-size and cache-lifecycle ablations.
7. Add leave-return and cross-session scenarios.

### Phase F — Sync thesis and artifact

1. Fix Ch1 contribution/count claims and define S1--S8 self-contained in Ch3.
2. Rewrite Ch4 algorithm/theorems from actual code.
3. Regenerate Ch5 tables/figures only from frozen outputs đã pass G0--G3.
4. Rewrite Ch6 from current state, including remaining limitations.
5. Sync README, `research_notes.md`, simulator budget display và grounding docs.
6. Build PDF; run undefined-ref/overfull checks.
7. Complete response file with links to code, tests, outputs and thesis lines.

## 7. Required tests trước khi đóng findings

### Formal/property tests

- `test_rem_support_is_public_and_fixed`
- `test_rem_probability_normalizes_on_declared_domain`
- `test_rem_metric_privacy_ratio_on_toy_domain`
- `test_sampler_has_nonzero_cdf_bin_width_or_declares_numerical_caveat`
- `test_pr_two_vertex_privacy_bound`
- `test_pr_equal_and_unequal_output_cases`
- `test_pr_full_transcript_bound_by_enumeration`
- `test_pr_fixed_parameter_worst_case_window_bound` (Lựa chọn A)
- `test_pr_window_accountant_all_hard_branches` (chỉ bắt buộc cho Lựa chọn B)
- `test_pr_state_depends_only_on_public_history`
- `test_sm_static_repeat_is_one_sample_postprocessing`
- `test_sm_public_reset_state_matches_theorem`
- `test_sm_arbitrary_revisit_is_not_claimed`
- `test_fixed_public_conditioning_requires_positive_acceptance`

### Data/provenance tests

- `test_graph_nodes_within_bbox`
- `test_all_osmnx_bbox_calls_use_left_bottom_right_top`
- `test_graph_manifest_matches_artifact`
- `test_projection_error_within_declared_tolerance`
- `test_output_manifest_contains_commit_params_seeds_hashes`

### Attacker tests

- Planar log-likelihood matches analytic density;
- `test_planar_benchmark_uses_constant_normalizer`;
- `test_online_hmm_uses_declared_normalizer`;
- REM likelihood matches brute-force normalized kernel;
- T-REM sequential likelihood matches direct enumeration on toy graph;
- PR equality/inequality likelihood matches Monte Carlo and exact toy probabilities;
- duplicated SM outputs do not get treated as independent fresh draws;
- jitter-aware likelihood recovers simulated static centres as `n` grows.

### Experiment QA

- mechanism order permutation does not change a mechanism's seeded outputs;
- aggregate table can be rebuilt exactly from saved raw rows;
- CI code cluster-resamples users, not individual correlated reports;
- all compared rows pass a formal-budget equality assertion;
- no table is generated from a graph hash different from its manifest.

## 8. Acceptance gates cho vòng verification tiếp theo

### Gate G0 — Data/kernel

- Correct bbox graph is a versioned/hash-addressed artifact or is reproducibly built
  from a pinned source/environment.
- Actual sampler reachability is tested; any finite-precision limitation is explicit.
- Output manifest complete.

### Gate G1 — Formal privacy

- R2-001 counterexample handled by corrected theorem.
- Window bound enforced or honestly reduced to primitive composition plus a scalar
  bound under an explicitly defined window metric/adjacency.
- Protected adjacency/metric/metadata scope explicitly defined.
- SM theorem declares public reset/session state; R2-011 fixed-public proposition
  includes positive acceptance and keeps secret-dependent cases separate.

### Gate G2 — Evaluation validity

- Mechanism-aware likelihood exists for every mechanism used in an "optimal" claim.
- PR/REM compared at matched guarantee.
- Jitter and secret domain are included in attacker model.
- Planar/REM/HMM/web attacker paths use their declared and tested normalizers.

### Gate G3 — Statistical evidence

- Multiple users/independent derived seeds with raw results and cluster CI;
  mechanism-order permutation leaves each seeded result unchanged.
- Stay-point vs home terminology correct.
- Leave-return/cross-session scenario evaluated if persistent-home claim remains.

### Gate G4 — Consistency

- Ch1 contribution/count claims are internally consistent; Ch3 defines the S1--S8
  taxonomy used later in the PDF.
- Ch4 algorithm equals code and its Euclidean/pseudometric scope matches grounding docs.
- Ch5 equals newly generated outputs.
- Ch6/README/`research_notes.md`/docs no longer describe old
  cutoff/one-home/future-PR state; S1--S8 and WICG/W3C wording are self-consistent.
- `response_0fafd4d.md` maps every finding to evidence and does not mark partial work
  as RESOLVED.

### Gate G5 — Scope/positioning

- PR chỉ được gọi persistent cache nếu implementation có persistent cache semantics;
  nếu không, gọi đúng là predictive previous-release reuse.
- Simulator được gọi server-side demonstrator cho on-device target, không phải shipped
  client-side system.
- Novelty table đối chiếu named precedents; Eclipse không được gọi direct competitor
  nếu chưa benchmark cùng threat model.
- Mọi P0/P1 finding có status `RESOLVED` hoặc `NARROWED` với caveat bắt buộc; không
  còn P0/P1 ở `PARTIAL/PLANNED` khi claim ready-to-share.

Chỉ khi G0--G5 pass mới nên đổi overall assessment từ **Needs revision** sang
**Share with caveats** hoặc **Ready to share**.

## 9. Required caveats nếu phải trình bày snapshot này ngay

- Số liệu là descriptive results của 20 trajectories / 7 users / one moving-benchmark
  seed và 40 generic stays / 21 users / 8 averaging seeds.
- Road graph hiện bị crop sai và mọi numerical result phải được xem là provisional.
- PR `epsilon` row không matched guarantee với REM vì test budget cộng thêm.
- `47.6m / 55%` dùng iid REM proxy, không phải exact PR attacker.
- SM static theorem giả định ideal kernel và public empty-cache/reset state; không
  cover GPS jitter crossing cells, prior hidden cache state, arbitrary traces hoặc
  secret revisit pattern.
- PR chỉ reuse previous release, không persistent-memoize old home.
- w-event statement hiện là conditional theorem/trivial worst-case composition,
  chưa phải enforced implementation.
- On-road 100% là structural property, không phải trực tiếp chứng minh kháng RAoPT,
  map matching hoặc re-identification.
- Flask simulator là server-side demonstrator cho một on-device target architecture.
- PR novelty là integration/instantiation của predictive mechanism với REM, trừ khi
  một component/theorem mới được chứng minh rõ.

## 10. Handoff protocol cho agent sửa tiếp

Mỗi finding chỉ được đánh dấu `RESOLVED` khi response có đủ bốn loại evidence:

1. **Code:** file/line hoặc commit thực hiện root-cause fix;
2. **Test/proof:** test name hoặc proof mới falsify được counterexample cũ;
3. **Result:** output mới được sinh từ correct graph/config/manifest;
4. **Manuscript:** Ch4/Ch5/Ch6 wording và table đã sync.

Các trạng thái nên dùng:

- `RESOLVED`: đủ cả bốn evidence, nếu finding cần cả bốn;
- `NARROWED`: claim đã thu hẹp đúng nhưng capability chưa implement;
- `PARTIAL`: có root-cause progress nhưng acceptance criteria chưa pass;
- `PLANNED`: chưa có implementation/evidence;
- `WONTFIX/OUT-OF-SCOPE`: phải giải thích impact và required caveat.

Không dùng kết quả rerun trên old graph để đóng R2-004/R2-008/R2-009. Không chỉ sửa
wording để đóng R2-001/R2-002 nếu thesis vẫn claim stronger implementation. Không dùng
post-hoc observed resample count làm formal privacy budget. Không overwrite current
JSON: archive dưới old graph hash, freeze corrected graph/environment, rồi mới cho các
agent tạo formal-numeric evidence, attacker results và Ch5 theo dependency ở Phase B.

## 11. Provenance của diagnostics trong verifier

Các diagnostics dưới đây dùng để falsify/triage, không phải final experiment outputs.
Claude agents không được copy số vào Ch5 như verified result nếu chưa chuyển method
thành script/test được track và chạy trên frozen G0 artifact.

| Diagnostic | Inputs/method đã biết | Reproducibility status |
|---|---|---|
| Commit/thesis/output review | Git commit `0fafd4dda4f4615ae4222db16e444367bfe0d44a` | Commit-scoped; line references phải refresh nếu code đổi |
| Raw graph extent/count/hash | Local ignored pickle SHA-256 `72359fe3847cba962ba2b3e537218c82d644777e55e269243d4292e9659125b1` | Hash preserved here; artifact/build source absent from clean clone |
| Correct induced crop | Intended bbox `(39.96,116.29,40.02,116.36)`; OSMnx 2.x order `(left,bottom,right,top)` | 19,634 nodes / 47,745 edges is a diagnostic crop of current wrong graph, not a freshly downloaded canonical graph |
| Projection error | Seed 0; 50,000 sampled vertex pairs; projected Euclidean vs `haversine_vector` | Method/seed preserved; standalone audit script not checked in |
| Utility JSON reconciliation | Current graph; official runner semantics; `epsilon=0.02`; absolute difference `<3e-14` | Confirms committed JSON is not stale; does not validate graph/methodology |
| Multi-stay medians/resamples/cells | 40 sorted stays, 21 users, 8 seeds, 100 reports, `sigma=10m`; runner seeds `1000*home_index+s` and jitter seeds `7000*home_index+s` | Core outputs reproduced; raw rows and audit command not retained |
| Official median CI | `bootstrap_ci`, user-cluster resampling, seed 0, 1,000 bootstrap draws | Code exists, but aggregate JSON omits raw rows |
| Success-rate CI diagnostic | User-cluster resampling, seed 0, 10,000 draws | Audit-only; code/raw bootstrap draws not checked in |
| Two-vertex PR counterexample | Two vertices 100m apart; `theta=200m`; `epsilon=eps_test=0.01`; exact finite kernel enumeration | Fully specified in R2-001; convert to unit test |
| Planar diagnostics | Current 40 x 8 sample set; fixed-centre/no-jitter-form likelihood; geometric median `9.0405m`, success 100% | Iteration/tolerance script not retained; audit-only, not jitter-aware MLE |
| PR sequential diagnostic | Current road-vertex hypotheses; exact PR transition kernel; no jitter; `47.81/53.1%` proxy vs `34.37/73.1%` sequential | Candidate-search script not retained; audit-only falsification, not continuous-home MLE |
| Numerical support | Wrong graph `exp` zeros plus correct-induced-crop `diff(cumsum(p))` zero-width bins at SW input | Counts preserved in R2-006; must become sampler property tests on canonical graph |
| Dependency/runtime context | OSMnx API observed as 2.0.5; one local NumPy 1.24.3 env cannot load the NumPy-2-layout pickle | Environment/hardware not locked; do not treat timing as verified |

Ưu tiên đầu tiên của response agent cho các audit-only rows là tạo checked-in
`tests/` hoặc `experiments/audit_*` script, lưu exact CLI/config/environment và xuất
raw evidence dưới graph hash. Nếu rerun không khớp, giữ finding mở và ghi discrepancy;
không chỉnh số trong verifier lịch sử để khớp result mới.
