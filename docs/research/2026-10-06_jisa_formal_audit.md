# Geo-I cho hướng bài JISA: audit lý thuyết và novelty

Ngày 06/10/2026. Đây là audit phương pháp và phạm vi tuyên bố, không sửa core,
experiment, artifact hoặc report đã chốt. Kết luận hiện tại: nền tảng Geo-I
phù hợp để giữ lại; phần có triển vọng thành đóng góp là **chọn tập truy vấn
khả thi trên đường để phục hồi nhiều mục đích POI trong một protocol không
phụ thuộc purpose riêng tư**, dưới cap toàn epoch. Việc ghép ledger, cache,
Geo-I và dummies tự nó chưa chứng minh novelty. Audit này là đối chiếu có
trọng tâm, chưa là systematic review hoặc dự đoán khả năng được nhận bài.

## 1. Những điểm cần giải quyết theo thứ tự

| Ưu tiên | Điểm chặn tuyên bố | Hành động cụ thể, giữ nền tảng Geo-I |
|---|---|---|
| 1 | Executable dùng float64 Gumbel/Laplace và NumPy PRNG, chưa có chứng chỉ pure Geo-I | Phân biệt theorem ideal với simulator; xây adapter sampler/entropy được kiểm chứng hoặc định lượng sai số và công bố approximate guarantee. Không đổi số đã chốt. |
| 2 | S7 hiện là request noninterference có điều kiện, chưa là che intent suy từ tuyến đường/activation | Định nghĩa server view, giữ cùng GPS/lịch khi chứng minh noninterference; đánh giá riêng intent tương quan với GPS và history. |
| 3 | C=.23 m⁻¹ là cap hợp lệ nhưng rất lỏng ở khoảng cách có ý nghĩa | Sweep cap công khai; báo đồng thời C·r và likelihood ratio tại r=50/100/500m, utility tail, traffic và attacker. |
| 4 | Novelty của road-awareness, prediction, dummies, Bayesian recovery, cache đã có prior art | Đặt câu hỏi nghiên cứu ở constrained query-set optimization; so với public-prior, Geo-I cùng anchors và planner đối chứng. |
| 5 | Belief là surrogate, không phải posterior attacker được hiệu chuẩn | Đánh giá calibration/prior mismatch; nếu thêm risk-aware planner thì dùng chỉ protected history và public/shadow model, chấm risk trên dữ liệu độc lập. |
| 6 | Cap không tự áp dụng qua epoch, reinstall hoặc nhiều ledger/device | Nêu rõ subject scope, persistence/anti-rollback assumption và tổng cap qua epoch/device; không suy identity anonymity từ sổ ngân sách. |
| 7 | Logical clock chưa là chứng minh packet timing thực tế không rò | Nêu threat model logical transcript; nếu đưa runtime/timing vào claim, đo hoặc padding theo lịch công khai, kể cả lỗi/retry. |
| 8 | Evidence hiện tại đã được dùng để phát triển và một số comparator là adaptation | Khóa candidate/selector rồi xác nhận trên nguồn chưa xem; giữ labels development và fidelity/N/A rõ ràng. |

Các giới hạn ở hàng 1 không có nghĩa nghiên cứu không thể nộp JISA. Chúng giới
hạn **tuyên bố về implementation đã triển khai**; theorem có điều kiện và đánh
giá simulator vẫn có giá trị nếu viết đúng phạm vi.

## 2. Prior art gần nhất và phần không nên nhận là mới

| Primary source | Nội dung liên quan đã xác nhận | Cách đặt đóng góp của ta |
|---|---|---|
| [Andrés et al., Geo-I, CCS 2013](https://arxiv.org/abs/1212.1984) | Metric privacy cho vị trí và Planar Laplace | Geo-I là nền tảng kế thừa. |
| [Chatzikokolakis et al., predictive mechanism, PETS 2014](https://petsymposium.org/2014/papers/Chatzikokolakis.pdf), Theorem 1 | Private test, prediction và bounded budget manager cho d∞-privacy | Reuse test và bounded composition không phải privacy primitive mới. |
| [Takagi et al., Geo-Graph-Indistinguishability](https://arxiv.org/html/2010.13449v1) | Graph metric, graph exponential mechanism, postprocessing và utility optimization | REM hiện tại dùng Euclidean score trên fixed road support; không đồng nhất với graph-distance GEM. |
| [Efficient Utility Improvement for Location Privacy, PoPETs 2017](https://petsymposium.org/popets/2017/popets-2017-0051.pdf) | Bayesian remapping để phục hồi utility | Protected belief/remapping riêng lẻ không mới. |
| [Atmaca et al., IEEE OJVT 2024](https://ieeexplore.ieee.org/document/10416404/) | Approximate Geo-I, dummy queries cho charging stations và Bayesian occupancy estimation | “Geo-I + dummies + hữu ích cho xe” đã có; đối chiếu output/service contract trước khi so số. Primary publisher abstract đã đọc; full-paper method audit vẫn cần. |
| [TransProtect/VehiTrack, SIGSPATIAL 2024](https://arxiv.org/html/2409.09495v1), §3–4 | Context-aware inference và candidate locations dựa trên real trajectory/learned mobility | Road-realistic candidates chưa đủ mới. Khác biệt có thể kiểm chứng là planner của ta không lấy raw history ngoài filter. Secret-dependent candidate restriction không tự kế thừa proof full-support REM. Đây là nhận định của audit, không kết luận paper kia vi phạm theorem của họ. |
| [Privacy-Aware Remapping, SAC 2024](https://www.privateer-project.eu/wp-content/uploads/2024/02/SAC24-A_Privacy_Aware_Remapping_Mechanism_for_Location_Data.pdf) | Discretization/remapping và location re-identification | Không đồng nhất giảm re-identification thực nghiệm với cap Geo-I nhỏ hơn. |
| [PRIVIC, PoPETs 2024](https://petsymposium.org/popets/2024/popets-2024-0033.php) | BA/IBU, elastic metric, QoS và statistical utility | LBS retrieval utility khác utility phân phối thu thập; Bayesian update và incremental collection đã có. |
| [LR-Geo, PoPETs 2025](https://petsymposium.org/popets/2025/popets-2025-0046.php) | Locally relevant LP, coefficient upload, Benders decomposition, guarantee có xác suất | Không thay full support bằng private local support rồi giữ nguyên theorem. Trust boundary và guarantee khác nhau. |
| [Fake queries trong continuous LBS, 2026](https://link.springer.com/article/10.1007/s44443-025-00438-z), §3.1/4 | Chèn all-dummy queries giữa real queries để chống correlation/reachability | Dummy-only events và continuous query mixing đã có. Protocol của ta cần được so theo tổng events/bytes và việc có gửi true coordinate hay không. |
| [Semantic correlation of moving paths, 2026](https://link.springer.com/article/10.1007/s44443-026-00899-w) | Dummies/semantic trajectory correlation | Không nhận semantic realism hoặc ASR/DER của paper thành bảo đảm của ta. |
| [Chen và Pang, query privacy, 2014](https://orbilu.uni.lu/handle/10993/10297) | User profiles và query dependency ảnh hưởng query privacy | “Không gửi purpose field” chưa xử lý intent/location correlations hoặc ownership unlinkability. |

Search còn thấy một lead rất gần về road adaptation/mobility prediction tại
EuroS&PW 2026, DOI `10.1109/EuroSPW72509.2026.00012`. Chưa lấy được primary
paper trong audit này nên **chưa dùng secondary abstract để kết luận novelty**;
cần đọc bản tác giả/publisher trước khi hoàn tất related works. Không cần giới
hạn nền tảng lý thuyết vào ba năm: các theorem kế thừa phải dẫn nguồn gốc.

## 3. Định nghĩa cần dùng trong bài

Giữ một public context P cố định: graph và projection, support V, POI catalogue
version, prior/model đã chốt độc lập với trace đang bảo vệ, K/L/categories,
public epochs, session starts/closes, admission policy và tick schedule. Hai
trace so sánh có **cùng P**, cùng số public ticks; khác tọa độ GPS x_t. Metric:

\[
d_P(x,x')=\|\Pi(x)-\Pi(x')\|_2,\qquad
D_\infty(X,X')=\max_{s,t}d_P(x_{s,t},x'_{s,t}).
\]

Đây là metric trong projected coordinates của implementation; chưa là theorem
theo WGS84 geodesic toàn cầu hoặc directed road distance. Không snap raw GPS
trước primitive rồi mặc nhiên gọi đó là Euclidean Geo-I: snapping có thể làm
thay đổi metric/sensitivity.

Server view V_pub gồm public requests (Q coordinates đã xáo, timestamp,
categories, L, schema), response transcript và public admission/timing signals
được mô hình hóa. Nó **không** gồm GPS, internal Z, reuse branches/counters,
ledger/key, local QuerySpec, local top-k P hoặc click/navigation telemetry.
Nếu thực tế gửi một trong các trường này thì phải mở rộng view và phân tích lại.
Account/IP được thấy hoặc loại khỏi scope một cách tường minh; chưa có theorem
identity anonymity. “Z đã bảo vệ” là internal protected history, không nhất
thiết là history server trực tiếp quan sát.

Hai property tách biệt:

1. **Coordinate transcript privacy:** mọi measurable A thỏa
   Pr[V_pub(X,u)∈A] ≤ exp(C·D∞(X,X′)) Pr[V_pub(X′,u′)∈A], dưới giả định bên dưới.
2. **Conditional purpose noninterference:** với cùng X/P, distribution của
   V_pub(X,u) bằng V_pub(X,u′) cho mọi private purpose/category/radius/destination
   histories u,u′. Local answer có thể khác; network view không đổi.

Property 2 tương đương zero added information về u **khi đã điều kiện hóa X/P**.
Nó không nói I(U;V_pub|P)=0 khi X hoặc activation phụ thuộc U. Một route chỉ đi
qua bệnh viện vẫn cho attacker tín hiệu intent dù request không có chữ “hospital”.
Nếu hai intent tạo các GPS distributions khác nhau, phải đánh giá marginal
intent inference và auxiliary knowledge riêng. Chỉ dưới một assumption thêm
về khoảng cách giữa các trace, coordinate theorem mới chuyển thành một bound
cho so sánh intent đó.

## 4. Các theorem có thể phát biểu và proof sketch

### T1 — REM trên fixed public support

Với public V, u>0 và independent ideal randomness:

\[
R_u(v\mid x)=\frac{\exp[-u\,d_P(x,v)/2]}{
\sum_{w\in V}\exp[-u\,d_P(x,w)/2]}.
\]

Kernel này là u-Geo-I theo d_P. Triangle inequality chặn log ratio của score
bởi u·d/2; normalizer thêm tối đa u·d/2. Tổng là u·d. Mapping vertex thành
coordinate, kể cả nhiều vertices cùng coordinate, giữ bound bằng marginalization.
Input-centred cutoff hoặc true-GPS QoS rejection không thuộc proof này.

### T2 — Fixed-cap predictive transcript, gồm hidden branches

Prediction h_t là hàm của protected extended history/public context; test
d_P(x_t,h_t)+Lap(1/u)≤θ với θ công khai. Extended transcript E chứa branch
và fresh protected anchor. First fresh anchor tính một unit; later reuse tính
một test unit; later refresh tính test+release, hai units. Filter kiểm tra
**trước đọc GPS** rằng worst next cost còn đủ. Khi thiếu cap chỉ dự đoán từ
protected state. Mọi feasible path chi tối đa U units.

Với fixed extended path e, h_t, pacing/read/stopping decisions giống nhau khi
tính likelihood dưới X và X′. Pass probability và complement của Laplace test
đều có ratio ≤exp(u·d_t); fresh REM thêm ≤exp(u·d_t). Do đó

\[
\frac{p_X(e)}{p_{X'}(e)}\le
\exp\!\left(\sum_t c_t(e)u\,d_t\right)
\le\exp(UuD_\infty(X,X')).
\]

Marginalize branches rồi áp mọi public/protected-state postprocessor để được
bound C_s=Uu cho public Q transcript. Đây là instantiation của predictive
bounded-manager argument, không theorem privacy mới. Realized branch accounting
đúng **nhờ prospective filter và extended-path proof**, không phải vì đếm lại
fresh samples sau khi chạy một mechanism khác.

Trong code matched filter, U=2H−1 và u=B/(2H). **H không phải hard maximum số
GPS reads** của branch-charged filter. Với H≥2, reuse-heavy path có thể đọc tới
U−1=2H−2 lần vì later reuse chỉ tốn một unit nhưng mỗi lần phải reserve hai.
H12 có thể đọc 22 lần; H1 đọc một lần. Nếu cần giới hạn đúng H reads thì thêm
public read-count ceiling riêng; không thay nghĩa cap hoặc sửa số đã chốt.

### T3 — Persistent fixed-epoch composition và mục đích truy vấn

Public epoch có N slots, mỗi session được dành C/N **trước** factory/GPS, không
thu hồi unused credit theo private branch. Với H cố định, đặt
u=C/[N(2H−1)]. Nếu mỗi admitted session thực thi T2 và dùng independent ideal
randomness thì bound cho linked multi-session view là C·D∞. Denied sessions
không đọc GPS/gửi Q; việc denied phải chỉ do cùng public policy/history P.
SQLite reserve-before-use/crash-without-refund thực hiện accounting invariant;
không tự tạo theorem về security của filesystem hay cross-device sharing.

Belief, reachable selector, progress/slack, private Q-order permutation, server
response theo Q và cache không đọc lại private GPS/intent cho network decisions.
Chúng là postprocessing của E/P. T1–T2 do đó áp cho V_pub, với mọi u/u′.
Fix X/P và couple cùng independent randomness/server strategy để các requests
giống từng bước: đây là proof của conditional purpose noninterference. Có thể
bao gồm adaptive server responses nếu server không có thêm secret channel và
local answers/retries không tạo feedback network theo private demand.

Local ranking dùng raw current GPS **không** là safe postprocessing cho một
output công bố công khai. Nó được phép vì P:top-k chỉ ở thiết bị/người dùng,
không nằm trong V_pub. Đưa P/click vào server view làm theorem trên không còn
áp dụng tự động.

### T4 — Per-step coverage, quotient và slack

Tại fixed protected history, mỗi internal track j có public reachable set A_j.
Ground set dùng labeled pairs (j,v), mỗi partition chọn một pair. Public service
response signature S(v) và nonnegative normalized POI weights w định nghĩa

\[
F(Q)=\sum_p w_p\,1\{p\in\cup_{q\in Q}S(q)\}.
\]

F monotone submodular. Global greedy với exact marginal oracle dưới partition
matroid có classical 1/2 bound; đây là kết quả kế thừa, không 1−1/e cho code
hiện tại. [Primary theorem reference](https://research.ibm.com/publications/maximizing-a-monotone-submodular-function-subject-to-a-matroid-constraint).
Equal ordered signatures cho equal marginals và equal objective; giữ public
tie-best representative trong mỗi A_j bảo toàn greedy/exchange decisions.
Exchange chỉ nhận cải thiện; service-equivalent progress bảo toàn signatures;
slack stage kiểm tra full final objective F_final≥F_base−σ. Vì vậy, trong exact
arithmetic tại cùng history:

\[
F(Q_{final})\ge\tfrac12\max_{Q\ feasible}F(Q)-\sigma.
\]

Đây là bound **surrogate một bước**. Nó không đảm bảo actual Recall, toàn route,
counterfactual future states hoặc calibrated adversarial risk. Float oracle/tie
có thể cần một numerical tolerance term để phát biểu cho executable. Current
weights còn trộn service-reference profiles, finite location grid và public
purpose/destination approximation; chưa bằng actual user query distribution.

Nếu sau này chứng minh một objective thật G trên cùng feasible sets và
sup_Q|F(Q)−G(Q)|≤η, bound chuyển thành G(Q_final)≥.5 OPT_G−σ−1.5η.
TV calibration của joint location/purpose distribution có thể cho η khi utility
nằm trong [0,1]. Hiện chưa có calibration certificate đó; không gán η=0.

### T5 — Static local cache monotonicity, không thêm network

Fix immutable public catalogue version, purpose/GPS và total lexical ranking.
Tại t, static cache chỉ chứa IDs đã nhận ở public ticks ≤t trong cùng valid
public epoch. Nếu candidate union C_t chứa current-reply set R_t, local top-k
từ C_t có Recall@k với full static reference không thấp hơn từ R_t: reference
items có trong candidate luôn đứng trước non-reference items trong cùng ranking.
No additional request/GPS follows because neither local answer nor expiry
changes public schedule/Q. Đây là correctness lemma cho static metadata, không
novel cache algorithm. Live availability không monotone theo old IDs; thiếu
current status phải là unknown, không suy “còn mở/còn chỗ” từ cache.

## 5. Implementation scope còn thiếu

`core/mechanisms.py` đã ghi rõ float Gumbel-max có bounded RNG span và có thể
tạo input-dependent zero support. Không thể chứng minh pure Geo-I của executable
bằng Monte Carlo, log-domain arithmetic, support clipping hoặc chỉ tăng precision.
Vấn đề finite precision đã được phân tích bởi [Mironov, CCS 2012](https://www.microsoft.com/en-us/research/publication/on-significance-of-the-least-significant-bits-for-differential-privacy/).
[Ilvento, base-2 exponential implementation](https://arxiv.org/abs/1912.04222)
cho một hướng exact implementation cho lớp score/parameter thích hợp; không
là chứng chỉ tự động cho continuous Euclidean REM và noisy threshold của ta.

Operational ledger dùng private HMAC domain separation rồi `default_rng`.
Điều này tránh cùng public seed giữa sessions trong experiment, nhưng NumPy
nêu rõ [RNG của họ dành cho simulation, không dành cho cryptographic security](https://numpy.org/doc/stable/reference/random/index.html).
Paper nên giữ simulator RNG để reproducibility, tách proposed production
adapter dùng OS/CSPRNG/unbounded random-bit refinement với sampler proof.
Đổi entropy source riêng vẫn chưa sửa finite-precision distribution. Theorem
trên giả định independent ideal bits; cryptographic computational claim cần
assumption và chứng minh riêng, không gọi finite PRNG là independent ideal draws.

Một lựa chọn khác là approximate bound có certificate. Ví dụ, nếu toàn
implemented transcript kernel cách ideal kernel tối đa η trong total variation
**uniformly cho mọi allowed input**, thì trên pairs D∞≤r:

\[
\Pr[\widehat V(X)\in A]\le e^{Cr}\Pr[\widehat V(X')\in A]
 +(1+e^{Cr})\eta.
\]

Proof cộng/subtract ideal probabilities. Đây chỉ là conditional transfer lemma;
η chưa được đo/chứng minh. Một uniform per-primitive error cộng với bounded
primitive count có thể cho η qua coupling; phải bao gồm test, score rounding,
sampling và adaptive state, không chỉ tail mass của một REM draw. Hiện chưa
có executable (ε,δ)-certificate để công bố.

Public clock trong API là timestamp đã đưa vào, không đảm bảo actual packet
release times độc lập với hidden branch. Private-read/reuse paths, CPU contention
do local query và error/retry có thể thay actual latency. “Zero delay” hiện
nghĩa không dùng endpoint holdback/suppression; không nghĩa computation bằng
không hoặc timing side-channel đã được giải quyết.

## 6. Restart, devices và con số cap

Trong một ledger/public epoch, reservations tồn tại qua process restart;
repeated token bị chặn và crash không hoàn cap. Nhưng rollback/xóa DB, clone
state, independent ledgers cho cùng người/xe, hoặc new public epoch không được
theorem che giấu. Với epochs e và devices d, bound thông thường cộng thành
Σ_{e,d}C_{e,d} hoặc Σ_{e,d}C_{e,d}D∞^{e,d}. Cross-device claim cần shared trusted
accounting scope hoặc public preallocation; ledger không phát hiện person/vehicle
identity. Khóa HMAC/diagnostic logs cũng phải là private local state.

C=.23 m⁻¹ cho toàn epoch cho phép likelihood ratio exp(Cr). Tại r=100m:
exp(23)≈9,74×10⁹. Endpoint20 single-session cap .0575 m⁻¹ cho exp(5,75)≈314.
Các bound hợp lệ này không tự đảm bảo nhỏ Hit100, không tạo khoảng cách nhiễu
tối thiểu, cũng không là lower bound cho attacker MAE. Cần công bố public cap
sweep và privacy–utility–cost frontier tại khoảng cách có ý nghĩa; chọn cap theo
threat/utility requirement trước confirmation. Nếu muốn odds amplification≤a
tại r thì C≤log(a)/r là calibration minh họa, không một lựa chọn tối ưu đã biết.

## 7. Hướng cải tiến có đóng góp rõ hơn, không đổi REM

Ưu tiên một hypothesis hẹp: **calibrated risk-aware reachable query-set
selection**, cùng primitive, cap, clocks, K/L và retrieval contract. Candidate
planner lấy protected belief/history và public/shadow-trained risk model, không
nhận true GPS hiện tại, true endpoint, future GPS hoặc actual private purpose.
Risk có thể là estimated public-attacker Hit probability/uncertainty ở các
targets đã khai báo, và cần chấm calibration trước khi dùng làm selector.
Không dùng evaluator truth để chọn Q.

Để có một claim kiểm chứng được, tạo nhiều feasible Q actions bằng public
postprocessing, chỉ nhận action thỏa coverage floor F≥F_base−σ rồi chọn theo
risk estimate đã chốt. T2–T3 giữ nguyên dù risk estimate sai: bảo đảm formal
đến từ inherited cap, **empirical risk gain** là hypothesis riêng. Coverage floor
vẫn có bound của T4; một objective F−λ·risk tùy ý có thể mất submodularity,
nên không tự giữ 1/2 theorem cho objective mới. λ/σ/model bank chọn chỉ trên
development/selection và khóa trước nguồn mới.

Các ablation bắt buộc: cùng Geo-I anchors không planner; public-prior planner;
current protected-belief coverage; candidate risk-aware planner; và Planar
Geo-I baseline có cùng cap/retrieval contract. Giữ matched-cap, K/L/bytes/timing
hoặc công bố trade-off riêng. Vary prior mismatch, first trip chưa có cache,
long trip, linked histories và geometry-aware attacker. Purpose stress cần cả
payload control và intent correlated với location/history. Known limitations
và failed configurations phải giữ, không chọn lại attacker/defense theo test.

## 8. Paper outline khả thi

1. Định nghĩa constrained online multi-purpose LBS problem, server view và
   phân biệt coordinate privacy, purpose noninterference, identity inference.
2. Algorithm có pseudocode tách private filter với public/protected planner,
   fixed schema, local ranking và static/live cache semantics.
3. T1–T3 là inherited conditional privacy correctness; T4/quotient/slack là
   scoped optimizer correctness. Nêu rõ những theorem/bounds đã có prior art.
4. Novel method hypothesis và calibration evidence; một metric không được thay
   thế likelihood ratio, và formal cap không thay empirical attack evaluation.
5. Matched ablations, native-paper metrics đúng contract, mới independent
   confirmation, utility tails/traffic/device latency và negative results.
6. Artifact versioning cùng sampler/entropy scope và deployment assumptions.

Các nguồn code audit: `core/mechanisms.py`, `core/session_budget.py`,
`benchmark/anchor_belief.py`, `benchmark/public_purpose_belief.py`,
`benchmark/engines/{filtered_cover,matched_filter,paced_guard,service_cover,fair_cover,quotient_cover,progress_cover,slack_progress}.py`,
`benchmark/{query_purpose,query_order,budgeted_geoi_lbs,versioned_static_geoi_lbs,versioned_static_poi_cache}.py`.
Các argument sẵn có [predictive filter](predictive_filter_argument.md) và
[service-cover review](service_cover_literature_review.md) nhất quán với narrow
ideal-kernel claim; không biến các observed attack scores thành theorem.
