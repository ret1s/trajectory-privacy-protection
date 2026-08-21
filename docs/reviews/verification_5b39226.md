# Nhận xét kiểm chứng kết quả SM-REM hiện tại

## Hồ sơ review

| Trường | Giá trị |
|---|---|
| Branch được review | `verifier` |
| Commit kết quả | `5b39226cc6efb4268d769a0bd82f3c55f5d667bb` |
| Commit message | `Snapshot current SM-REM results for verification` |
| Commit cha | `ec6821f` |
| Ngày review | 2026-08-21 (Asia/Ho_Chi_Minh) |
| Phạm vi | Formal claims, implementation, experiments, thesis consistency, industry traceability |
| Kết luận | **Chấp nhận làm prototype snapshot; chưa thể coi là kết quả đã verify về mặt hình thức** |

Tất cả file và line number trong tài liệu này đều trỏ tới commit `5b39226`, không trỏ tới working tree hoặc một phiên bản sửa sau này. Review này chỉ ghi nhận hiện trạng; không âm thầm sửa kết quả đang được đánh giá.

## Kết luận ngắn

Framing hiện tại đã có industry awareness khá rõ: road-constrained release, temporal plausibility, repeated-report averaging và stateful protection đều là những vấn đề có thật trong hệ thống vị trí. Phần lõi toán học của REM lý tưởng trên một output set cố định cũng hợp lý; code chạy được và đủ tái lập để làm research prototype.

Tuy nhiên, ba nhóm claim đang chặn việc coi đây là final thesis result:

1. cutoff có tâm tại true location làm output support thay đổi theo dữ liệu bí mật, nên implementation của REM/T-REM/SM-REM không còn pure Geo-I;
2. SM-REM không cài đúng cell-pseudometric kernel đã phát biểu, và exact memoization làm lộ equality/revisit pattern, nên trajectory theorem cùng claim composition theo `số distinct cells × epsilon` không đúng;
3. Bayesian/HMM evaluator chưa dùng likelihood thật của REM, T-REM hay SM-REM, vì vậy attack error hiện tại chưa thể được gọi là Bayes-optimal hoặc mechanism-aware.

Framing an toàn nhất cho commit này là:

> REM/T-REM là các idealized fixed-support road-output kernels có lập luận per-release conditional Geo-I. Cutoff trong implementation hiện là một experimental approximation. T-REM làm giảm một chỉ báo speed implausibility nhưng chưa chứng minh correlation robustness. SM-REM là exploratory stateful defense ngăn variance reduction của arithmetic averaging khi cùng một location được báo chính xác lặp lại trong một cache lifetime; chưa có trajectory-level DP proof.

## Tổng hợp mức độ

| ID | Mức độ | Nhận xét | Hướng xử lý |
|---|---|---|---|
| V-001 | Nghiêm trọng | Cutoff phụ thuộc true location phá pure Geo-I | Phải sửa hoặc hạ claim |
| V-002 | Nghiêm trọng | First release của SM-REM trái với cell pseudometric | Phải sửa |
| V-003 | Nghiêm trọng | Memoization làm lộ revisit pattern; trajectory theorem sai | Redesign hoặc thu hẹp scope |
| V-004 | Nghiêm trọng | “Optimal/mechanism-aware” attacks dùng sai likelihood | Sửa evaluator và wording |
| V-005 | Cao | Averaging evidence mới là one-home/one-seed exact-repeat test | Mở rộng evaluation |
| V-006 | Cao | Cache lifetime chưa cover cross-session industry threat | Định nghĩa và cài lifecycle |
| V-007 | Cao | Claim của T-REM/on-road mạnh hơn evidence | Thu hẹp claim, chạy attack thật |
| V-008 | Cao | HMM state truncation gây bias không đều giữa mechanisms | Thêm coverage/convergence checks |
| V-009 | Cao | Requirement table và artifact hiện có không khớp | Tách target/theory/code/evidence |
| V-010 | Cao | Benchmark baseline không phải pipeline được mô tả là faithful | Cài lại hoặc đổi tên |
| V-011 | Trung bình | Một số evaluation constructs bị đặt sai tên hoặc tính sai | Sửa metrics |
| V-012 | Trung bình | Provenance của kết quả chưa đủ để verify | Ghi hashes, seeds, versions, CIs |
| V-013 | Trung bình | Industry evidence và citations cần chỉnh | Sửa nguồn và wording |
| V-014 | Trung bình | Composition và metadata scope chưa được định nghĩa đủ | Định nghĩa protected object/metric |
| V-015 | Thấp | Còn các lỗi consistency và LaTeX nhỏ | Editorial cleanup |

## Nhận xét chi tiết

### V-001 — Nghiêm trọng: cutoff trong implementation phá pure Geo-I

**Bằng chứng:** `core/mechanisms.py:135-157`; `thesis/chapters/ch4_phuongphap.tex:121-169`; `docs/problem_formulation.md:57-62`.

Proof hiện tại đúng khi output set \(V\) là cố định và công khai. Implementation lại lấy mẫu trên

\[
V_x = V \cap B(x,R),
\]

nên support phụ thuộc private input. Chọn một vertex \(v\) và hai input rất gần nhau nhưng nằm ở hai phía của cutoff boundary đối với \(v\). Khi đó \(P[M(x)=v]>0\), trong khi \(P[M(x')=v]=0\). Likelihood ratio của event này là vô hạn, bất kể hai cutoff disks overlap nhiều đến đâu. Tail mass nhỏ không thể khôi phục pure-DP guarantee.

Claim định lượng về tail cũng không đúng trên chính graph và 858 điểm của benchmark đã commit:

| epsilon | Full-support mass bị bỏ |
|---:|---:|
| 0.005 | 4.08%–17.94%; mean 8.60% |
| 0.010 | 0.0817%–1.318%; mean 0.369%; median 0.291% |

Các số này không support claim tail nhỏ hơn `1e-4` với mọi `epsilon >= 0.005`.

**Cần làm:** dùng toàn bộ \(V\), hoặc một public candidate set cố định và độc lập với \(x\). Nếu vẫn giữ cutoff, cần gọi nó là approximation và suy ra một uniform approximate-DP \(\delta\) cụ thể; không gọi implementation hiện tại là pure epsilon-Geo-I.

### V-002 — Nghiêm trọng: SM-REM không cài đúng cell mechanism đã phát biểu

**Bằng chứng:** `core/mechanisms.py:240-261,286-305`; `thesis/chapters/ch4_phuongphap.tex:237-263`; `docs/problem_formulation.md:63-69`.

Cell hiện chỉ được dùng làm cache key. Khi cache miss, code gọi T-REM với exact private coordinates `(lat, lon)`, không dùng một fixed public representative `rep(cell)`.

Với hai điểm khác nhau \(x,x'\) trong cùng một cell, proposed pseudometric cho \(\widetilde d(x,x')=0\). Metric privacy khi đó bắt buộc hai output distributions giống hệt nhau. Thực tế chúng khác nhau vì distance, support, logits và normalizer vẫn phụ thuộc exact point. Một reproduction tối thiểu với hai vertices, `grid=60m`, `epsilon=0.02`, hai input cùng cell tại 1m và 29m cho xác suất output vertex đầu lần lượt là `0.641067` và `0.505000`.

**Cần làm:** nếu muốn có single-release cell-level statement, hãy quantize input về một fixed public representative trước khi sampling, đồng thời dùng fixed output support. Chỉ sửa bước này vẫn chưa đủ để có trajectory privacy.

### V-003 — Nghiêm trọng: exact memoization làm lộ revisit pattern

**Bằng chứng:** `core/mechanisms.py:242-261,293-305`; `thesis/chapters/ch4_phuongphap.tex:255-276`; `docs/problem_formulation.md:64-72`; `thesis/chapters/ch6_tongket.tex:31-33`.

Xét hai secret traces \(X=(a,a)\) và \(X'=(a,b)\), trong đó `a` được revisit còn `b` nằm ở cell khác. Định nghĩa transcript event

\[
E = \{z_2 \ne z_1\}.
\]

Với \(X\), exact cache reuse cho \(P(E\mid X)=0\). Với \(X'\), cell thứ hai được sample mới và \(P(E\mid X')>0\). Do đó một chiều của likelihood ratio là vô hạn. Grid công khai không làm secret cache hit/miss trở thành thông tin công khai. Hệ quả:

- arbitrary-trajectory theorem hiện tại sai;
- composition theo số distinct secret cells chưa được chứng minh;
- caveat chỉ giới hạn loss ở `exp(epsilon * cell width)` cũng chưa đúng.

Đây cũng là loại leakage mà predictive-mechanism literature đã cảnh báo: reuse có thể làm lộ trạng thái không di chuyển, nên chính reuse decision phải được bảo vệ. Xem [Chatzikokolakis et al., PETS 2014](https://petsymposium.org/2014/papers/Chatzikokolakis.pdf).

**Safe theorem còn giữ được:** nếu protected secret là một location cố định \(x\) lặp \(T\) lần và \(Z\sim M(x)\), việc release `(Z, ..., Z)` chỉ là deterministic post-processing của một sample và thừa hưởng guarantee của \(M\). Kết quả này chỉ cover static exact-repeat scenario.

**Cần làm:** hoặc thu hẹp contribution về scenario trên, hoặc redesign reuse decision bằng một predictive/stateful mechanism có privacy rõ ràng và định nghĩa trajectory adjacency đầy đủ.

### V-004 — Nghiêm trọng: evaluated attackers chưa mechanism-aware hoặc Bayes-optimal

**Bằng chứng:** `evaluation/attacks.py:4-29,34-67,70-136`; `experiments/run_benchmark.py:51-57`; `thesis/chapters/ch2_cosolythuyet.tex:67-72`; `thesis/chapters/ch5_thucnghiem.tex:32-36`.

Attacker hiện gần như chỉ dùng `exp(-epsilon * distance / 2)` cho road mechanisms. Nó bỏ qua:

- input-dependent normalizer và support indicator của truncated REM;
- history-dependent weight và normalizer của T-REM;
- cache state và equality likelihood của SM-REM;
- exact behavior của simplified baseline.

Candidate “prior” còn được chọn sau khi quan sát \(z\), nên không phải một fixed prior theo nghĩa Bayesian thông thường. Posterior mean tối ưu cho squared Euclidean loss, không tối ưu cho Euclidean-distance loss đang được report; geometric median mới là estimator phù hợp với loss đó.

Trên committed graph tại `epsilon=0.01`, candidate normalizers thay đổi khoảng `3.933×`; thêm term `-log Z(x)` bị thiếu làm một posterior estimate dịch `84.1m`. Đây không phải sai khác có thể bỏ qua.

**Cần làm:** cài exact likelihood cho từng mechanism, prior cố định trước observation, state-aware sequential attacker, và estimator khớp evaluation loss. Trước khi hoàn thành, nên đổi tên thành “heuristic Bayesian remapper” và “approximate-emission HMM”.

### V-005 — Cao: averaging result mới có external validity rất hẹp

**Bằng chứng:** `experiments/run_averaging.py:48-78`; `evaluation/attacks.py:178-222`; `thesis/chapters/ch5_thucnghiem.tex:104-142`.

Experiment chỉ dùng một home, một random seed, một cache lifecycle và cùng một coordinate chính xác lặp tối đa 100 lần. Flat curve của SM-REM là hệ quả tất định của việc trả cached value; đây là unit/sanity test hữu ích, nhưng chưa phải bằng chứng về general home protection.

Các targeted diagnostics cho thấy uncertainty rất quan trọng:

- qua 30 independent seeds, SM-REM error tại `n=100` dao động `44.4m`–`893.7m`, median `215.4m`;
- home được chọn chỉ cách grid boundary khoảng `6m`;
- thêm Gaussian GPS jitter `sigma=10m` tạo 2–4 distinct cells trong 100 reports;
- first release được cache vẫn có thể tình cờ gần secret, nên “không có thêm averaging gain” không đồng nghĩa “home đã an toàn”.

Với road kernels, arithmetic sample mean hội tụ về \(E[Z\mid x]\), không nhất thiết về \(x\). Exact output-mean bias tại home đang dùng là `83.18m` khi `epsilon=0.01` và `27.33m` khi `epsilon=0.02`. T-REM outputs còn phụ thuộc temporal state, nên iid claim `O(1/sqrt(n))` không áp dụng trực tiếp.

**Cần làm:** đánh giá nhiều homes/users và seeds; report confidence intervals và success probabilities tại các radius có ý nghĩa; stratify theo khoảng cách tới grid boundary; thêm GPS jitter, cache reset/expiry, cross-session reports và mechanism-aware repeated-release estimator.

### V-006 — Cao: cache lifecycle chưa cover industry threat đã đặt ra

**Bằng chứng:** `core/mechanisms.py:281-305`; `experiments/run_benchmark.py:87-92`; `web/simulator.py:95-102`.

Cache chỉ nằm trong memory và `reset()` xóa toàn bộ. Benchmark reset theo trajectory; web endpoint tạo mechanism mới theo request. Vì vậy cùng một địa điểm có thể nhận fresh samples sau một trip, request, process restart hoặc application session—đúng setting mà long-term observer có thể average releases.

Cache hit còn trả vertex cũ mà không áp current T-REM reachability weighting. Trong benchmark, 316/858 reports (36.8%) là cache hits, nên temporal plausibility khi revisit không phải edge case.

**Cần làm:** định nghĩa privacy principal và cache key, persistence boundary, TTL/rotation, eviction, restart và map-update semantics, concurrent access, secure storage và deletion policy. Cần tách “reset temporal state” khỏi “reset protected memoization state”, rồi đánh giá trade-off giữa memoization và current reachability.

### V-007 — Cao: claim của T-REM và on-road mạnh hơn evidence

**Bằng chứng:** `core/mechanisms.py:202-211`; `thesis/chapters/ch1_gioithieu.tex:28-35`; `thesis/chapters/ch4_phuongphap.tex:156-177`; `thesis/chapters/ch5_thucnghiem.tex:82-101`; `thesis/chapters/ch6_tongket.tex:22-25`.

T-REM dùng một soft positive penalty dựa trên straight-line distance. Nó không enforce hard reachable set và không dùng road shortest-path/travel-time distance, nên barrier hoặc disconnected components vẫn có thể tạo jump phi thực tế. Tại `epsilon=0.01`, speed violations vẫn khoảng 6%, và HMM attacker error giảm từ `265.1m` ở REM xuống `233.6m` ở T-REM—tức attacker tốt hơn theo chính metric của luận văn.

Tương tự, emit road vertices chỉ loại bỏ off-road-support cue; nó không loại bỏ topology, transition, prior hay learned reconstruction features. RAoPT chưa được chạy. On-road rate 100% của mechanism có output domain là road vertices là structural property, chưa phải attack-resistance evidence.

Nguồn liên quan: [RAoPT paper](https://erikbuchholz.de/wp-content/papercite-data/pdf/bsw%2B22.pdf).

**Cần làm:** mô tả T-REM là “soft straight-line displacement regularizer” và chỉ claim mức giảm speed violation đã đo. Thay “đóng/loại bỏ map-matching hoặc velocity attack” bằng “loại bỏ một cue; mức kháng attack cần được test trực tiếp”.

### V-008 — Cao: HMM state truncation làm comparison không công bằng

**Bằng chứng:** `evaluation/attacks.py:76-93`.

HMM giới hạn khoảng 2,100 candidates xuống 800 điểm gần observed point nhất. Tại `epsilon=0.01`, nearest road vertex của true location được giữ lại với tỷ lệ xấp xỉ:

| Mechanism | True-proxy state được giữ |
|---|---:|
| Laplace | 98.1% |
| simplified baseline | 100.0% |
| REM | 83.6% |
| T-REM | 91.5% |
| SM-REM | 89.6% |

State approximation vì vậy có thể làm attacker error của road mechanisms tăng mạnh hơn comparators.

**Cần làm:** dùng adaptive/larger state sets và report candidate coverage cùng convergence sensitivity trước khi so sánh attack errors.

### V-009 — Cao: requirement table đang mạnh hơn artifact hiện có

**Bằng chứng:** `thesis/chapters/ch4_phuongphap.tex:21-58`; `docs/problem_formulation.md`; `web/simulator.py:98-102,140`.

Artifact hiện tại chưa thực hiện đầy đủ một số mục đang được trình bày như đã đạt:

- client-side/on-device execution mới là architecture assumption; demo chạy mechanism trên Flask và trả raw coordinates;
- context-based privacy granularity chưa được cài; epsilon là global;
- speed consistency mới là soft empirical improvement;
- averaging protection bị giới hạn ở exact-repeat và một cache lifetime;
- endpoint protection chưa có endpoint detector hay policy riêng;
- pure Geo-I chỉ đúng với ideal fixed-support mechanism, không đúng với cutoff code.

Thesis còn nhắc R8–R12 và S1–S8 nhưng chưa định nghĩa đầy đủ các ID đó ngay trong thesis PDF. Web UI luôn report `T * epsilon`, kể cả khi budget này chưa có ý nghĩa hình thức hoặc mâu thuẫn với claim của SM-REM.

**Cần làm:** tách matrix thành `Design target`, `Formal ideal`, `Implemented`, `Empirically evaluated`; đưa threat/requirement taxonomy vào thesis; chỉ hiển thị privacy budget đã được định nghĩa và chứng minh.

### V-010 — Cao: benchmark baseline không phải pipeline được mô tả là “faithful”

**Bằng chứng:** `core/mechanisms.py:15-20,76-127`; `core/trajectory_privacy.py:126-141`; `thesis/chapters/ch4_phuongphap.tex:66-96`.

Simplified benchmark baseline bỏ alternative-road search và building/water validity/rejection pipeline của method trước, nhưng vẫn được gọi là faithful. Sai khác này có thể thay đổi cả utility lẫn attack surface, vì vậy comparison hiện tại chưa phải controlled comparison với internship-2 method.

Theoretical rejection statement cũng đang quá mạnh. Conditioning một epsilon metric-private kernel trên một fixed public allowed set có thể không giữ nguyên epsilon, nhưng acceptance normalizer cho general `2 * epsilon` upper bound dưới các giả định chuẩn. Kết luận “không còn finite guarantee” không đúng nói chung với fixed public conditioning; location-dependent support và bounded-retry/fallback behavior phải được phân tích riêng.

**Cần làm:** hoặc cài lại original pipeline, hoặc đổi tên thành “simplified TT2 surrogate” và thêm ablation. Sửa proposition để phân biệt fixed public conditioning, private-input-dependent support và finite retry behavior.

### V-011 — Trung bình: một số evaluation constructs bị đặt sai tên hoặc tính sai

**Bằng chứng:** `evaluation/metrics.py:58-121`; `core/road_network.py:84-86`; `web/simulator.py:52`; `web/templates/simulator.html:117`; `thesis/chapters/ch5_thucnghiem.tex:147-171`.

- “distance to road” thực tế là nearest-vertex distance, không phải point-to-edge distance. Midpoint nằm đúng trên một road edge dài 100m vẫn có thể bị tính là cách road 50m.
- DTW được mô tả là normalize theo warping-path length, nhưng code chia cho `n + m` mà không track path. Hai aligned trajectories cách nhau đúng 1m có thể trả `0.5m`.
- “pharmacies” là random road vertices, không phải OSM pharmacy POIs. Claim recall khoảng 81% không trace được tới committed result table; việc trả 10 candidates để so với ground-truth top 5 cũng làm recall dễ cao hơn.
- `max_disp` là mean của per-trajectory maxima, không phải global maximum.

**Cần làm:** tính nearest-edge distance; sửa hoặc đổi tên DTW normalization; label POIs là synthetic hoặc load `amenity=pharmacy`; nêu rõ `k` và returned-list size; đổi tên hoặc sửa maximum aggregation.

### V-012 — Trung bình: provenance chưa đủ để verify kết quả

**Bằng chứng:** `experiments/run_benchmark.py:75-127`; `experiments/run_averaging.py:62-78`; `requirements.txt`; committed JSON outputs.

Một RNG object được consume tuần tự qua các mechanisms, nên kết quả phía sau phụ thuộc mechanism order và số random draws phía trước. JSON chưa ghi commit, master/derived seeds, data/graph checksums, selected files/users, full config, package versions, aggregation unit, repetitions hoặc confidence intervals. OSM/data inputs được generate hoặc gitignore và chưa pin vào immutable snapshot; dependency constraints mới là lower bounds chứ chưa lock.

**Cần làm:** derive independent deterministic seed theo mechanism/epsilon/trajectory/repetition; thêm manifest chứa commit và input hashes; lock dependencies; ghi dataset selection và hardware; report user-level uncertainty.

### V-013 — Trung bình: một số industry/source claims cần sửa

**Bằng chứng:** `docs/problem_formulation.md:38,48,76-79`; `thesis/chapters/ch4_phuongphap.tex:229-276`; `thesis/refs.bib:65-69`.

- “15-minute cached value” nằm trong Google-authored approximate-geolocation explainer/proposal, không phải rule của W3C Geolocation Recommendation hiện tại. W3C API có caller-controlled `maximumAge`, không phải privacy obfuscation guarantee. Xem [W3C Geolocation](https://www.w3.org/TR/geolocation/) và [Approximate geolocation explainer](https://github.com/explainers-by-googlers/approximate-geolocation).
- AOSP hiện dùng `MIN_ACCURACY_M = 200m` và một slowly changing random offset; 60m không xấp xỉ giá trị này. Xem [AOSP LocationFudger](https://android.googlesource.com/platform/frameworks/base/%2B/897c0e136b4d99b8fd393683688b826340f43103/services/core/java/com/android/server/location/fudger/LocationFudger.java).
- Strava default hidden start/end distance là 200m, không phải 60m. Xem [Strava privacy defaults](https://support.strava.com/en-us/articles/15401763-your-privacy-defaults-when-you-create-a-strava-account).
- LP-Doctor là precedent hợp lệ nhưng nên cite trực tiếp: hệ thống tính anonymized location một lần theo location/protection level rồi reuse. Xem [Fawaz & Shin, USENIX Security 2015](https://www.usenix.org/system/files/conference/usenixsecurity15/sec15-paper-fawaz.pdf).
- PTPPM bibliography metadata đang sai, và review không tìm thấy việc tác giả tự thừa nhận repeated-location limitation như câu hiện tại. Nên trình bày đây là gap do luận văn suy ra, không phải quote hoặc author admission. Đối chiếu [arXiv metadata](https://arxiv.org/abs/2511.21020).
- Strava endpoint-zone inference dùng route-boundary geometry, road gates và metadata; không nên dùng làm direct evidence cho iid Geo-I arithmetic averaging. Xem [Dhondt et al. summary](https://lepoch.at/publication/epz-inference-attacks).

**Cần làm:** thêm primary citation ngay cạnh từng platform claim, sửa bibliography metadata, và phân biệt rõ statement của source với inference của luận văn.

### V-014 — Trung bình: trajectory composition và metadata scope chưa rõ

**Bằng chứng:** `core/mechanisms.py:45-47`; `docs/system_model_and_threats.md:105-109,231-238`; `docs/problem_formulation.md:21-27,87-90`; `thesis/chapters/ch4_phuongphap.tex:21-32,215-219`.

Với fixed public metadata và một valid fixed-support adaptive kernel, transcript bound tự nhiên là

\[
\frac{P(Z\mid X)}{P(Z\mid X')} \le
\exp\!\left(\epsilon\sum_t d(x_t,x'_t)\right).
\]

Đây là epsilon-privacy theo product metric \(D_1=\sum_t d_t\), hoặc có thể upper-bound thành `T * epsilon` dưới một \(D_\infty\) neighboring relation được định nghĩa rõ. Chỉ viết `epsilon_traj = T * epsilon` làm protected object và adjacency bị mơ hồ.

Coordinate Geo-I cũng không tự động bảo vệ joint release `(z, metadata(x))`. Timestamp, query, linkable identity và content phải là fixed/public trong neighboring comparison, được sanitize bằng mechanism khác, hoặc được đưa vào joint analysis.

**Cần làm:** định nghĩa event-level hay user-level adjacency, trajectory metric, protected radius, observation window, public/secret metadata và cache state mà adversary nhìn thấy.

### V-015 — Thấp: documentation và layout consistency

**Bằng chứng:** `README.md`; `thesis/chapters/ch5_thucnghiem.tex:74`; LaTeX build log.

- README vẫn chỉ liệt kê bốn mechanisms và bỏ SM-REM; đồng thời còn imply formal guarantees cho pipeline cũ.
- Chapter 5 viết “bốn quan sát” nhưng enumerate năm.
- Thesis build không có unresolved references, nhưng còn các overfull boxes đáng kể tại Chapter 4 requirements table/proof/theorem.
- `kellaris2014wevent` được khai báo `@inproceedings` nhưng mang journal-style volume/number metadata.
- README dùng lệnh `python`, trong khi reviewed environment có `python3` nhưng không có `python`.

## Những phần đã pass

- Ideal REM proof đúng khi output domain là một fixed public finite set.
- Weight của T-REM dựa trên previous release/timestamp là public đối với current secret sau khi condition trên history; với fixed full support, per-step conditional proof có thể giữ lại.
- Planar Laplace sample radius theo `Gamma(shape=2, scale=1/epsilon)` đúng như kỳ vọng.
- Trong cùng một mechanism object và cùng cell, cache trả ổn định đúng một road vertex.
- Categorical sampler dùng numerically stable logits bằng cách trừ maximum.
- Averaging output được reproduce byte-for-byte trong isolated rerun; quick benchmark chạy hoàn tất.
- `python3 -m compileall -q core data evaluation experiments web` pass.
- `latexmk -xelatex -interaction=nonstopmode -halt-on-error` pass; thesis có 27 trang và không còn unresolved references/citations ở final pass.
- Threat-model documents đã nêu một số limitation quan trọng; hướng LP-Doctor/stateful offset có industry precedent thật.

## Thứ tự xử lý đề xuất

### P0 — trước khi claim method đã được verify

1. Bỏ private-input-centred support cutoff, hoặc formally hạ implementation xuống một approximate guarantee cụ thể.
2. Chọn scope thật sự của SM-REM:
   - hướng bảo thủ: exact repeated static location trong một cache lifetime;
   - hướng nghiên cứu sâu hơn: redesign input quantization và private state/reuse semantics, rồi chứng minh trajectory guarantee.
3. Thay heuristic attacker kernels bằng exact mechanism likelihoods; thêm state và candidate-coverage checks.
4. Sửa wording của thesis ngay để không trộn ideal construction, implemented approximation và empirical observation.

### P1 — trước khi dùng numerical conclusions làm thesis evidence

5. Chạy multi-home, multi-user, multi-seed evaluation với user-level confidence intervals, GPS jitter, boundary strata và cross-session/cache-lifecycle scenarios.
6. Cài persistent, scoped memoization semantics; test revisit-versus-reachability behavior.
7. Sửa baseline và evaluation constructs, sau đó chạy mechanism-aware và reconstruction attacks trực tiếp.
8. Thêm immutable experiment provenance và independent derived RNG seeds.

### P2 — trước khi nộp luận văn

9. Sửa platform/paper citations và bibliography metadata.
10. Làm threat/requirement taxonomy self-contained, sửa terminology/metrics và dọn LaTeX layout warnings.

## Quyết định verification

Commit `5b39226cc6efb4268d769a0bd82f3c55f5d667bb` nên được giữ nguyên như **pre-verification prototype snapshot**. Không nên tag commit này là proved/final SM-REM result. Với candidate tiếp theo, có thể đối chiếu trực tiếp các finding ID V-001–V-015 và đánh dấu từng mục là `resolved`, `accepted limitation` hoặc `out of scope`, kèm fixing commit tương ứng.
