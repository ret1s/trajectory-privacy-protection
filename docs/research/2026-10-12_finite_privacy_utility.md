# Geo-I với lựa chọn Q hữu hạn: chứng minh privacy và utility

Bản mở rộng giữ Geo-I/REM, noisy reuse, cap Cs=0,23/m mỗi phiên, K5/L30 và
lịch bảo vệ GPS tối thiểu60s. Chỉ lớp chọn ba khung Q ở0/20/40s thay đổi.
Mục tiêu là đóng khoảng trống số học float của kênh Q và cải thiện pool POI;
không thay backbone hoặc đổi GPS/nhãn của dữ liệu để tạo điểm đẹp.

## 1. Vì sao floor cũ thất bại

Bảng cũ có validity v(x,p)≤1. Những public cases không có reference được
đặt0 trong score availability rồi trộn đều bốn purpose. Ngay cả khi U chứa
toàn catalogue, g(x,U)≤mean_p v(x,p). Trên dữ liệu công khai hiện tại:

\[
\min_x\frac14\sum_pv(x,p)=0{,}722222\ldots<0{,}75.
\]

Vì vậy floor0,75 bất khả thi từ định nghĩa score, chưa cần chạy một quỹ đạo.
N/A không phải dịch vụ thất bại: ví dụ bán kính không có POI thì đáp án hợp
lệ là danh sách rỗng. Bản mới chuẩn hóa các reference **công khai, có định
nghĩa** tại từng state/purpose trước khi trộn belief. Thao tác này tạo một
bảng cố định G(x,a)∈[0,1], vẫn tuyến tính theo belief; không chuẩn hóa lại
mẫu số theo b hoặc request ψ của người dùng.

Một floor mới phải ghi rõ phạm vi: trung bình của các public reference cases,
không phải minimum của mọi private purpose, mọi radius và mọi GPS trên đường.

## 2. P1 — Kernel hữu hạn, không tính xác suất bằng float

Giữ transcript tăng cường của các hành động Q đã lấy mẫu trước đó cố định
(bao gồm nhãn lane giả; P2 giải thích phép chiếu xuống transcript server). Thư viện đoạn Q, số hành động
n và các integer masses m_a>0 đều công khai. Đặt M=Σm_a và w_a=m_a/M.
Từ belief Geo-I, tính score rồi chuyển thành F_a∈{0,...,S}, S=65536. Phép
biến đổi private có thể dùng float: chứng minh privacy chỉ cần F bị chặn,
không cần score là posterior đúng hoặc float được làm tròn chính xác.

Với allowance ε≥0 công khai, đặt:

\[
z_a=F_a/S,\quad\mu=\sum_aw_az_a,\quad
\alpha=\frac\varepsilon{2+\varepsilon},\quad
p_a=w_a[1+\alpha(z_a-\mu)].
\]

**Chuẩn hóa và support.** Σp_a=1 vì Σw_a(z_a−μ)=0. Do z_a,μ∈[0,1] và
α<1, w_a(1−α)≤p_a≤w_a(1+α), nên mọi hành động đều có xác suất dương.

**Tỷ số likelihood.** Với hai lịch sử private bất kỳ E,E′, cùng public prefix:

\[
\boxed{\frac{p_a(E)}{p_a(E')}\le
\frac{1+\alpha}{1-\alpha}=1+\varepsilon\le e^\varepsilon.}
\]

Không cần bound độ nhạy của một phép chuẩn hóa private mới; mọi vector
integer trong miền đã khai báo đều thỏa bất đẳng thức này. Thứ tự Q và K
là một phần hành động chung. Triển khai hiện giữ K5, không quyết định K từ GPS.

**Triển khai đúng PMF hữu hạn.** Viết α=N/D bằng số nguyên, không rút về float:

\[
W_a=m_a\left[DMS+N\left(MF_a-\sum_cm_cF_c\right)\right],\quad
\sum_aW_a=DM^2S=T_{public}.
\]

Mọi W_a là số nguyên dương. Lấy J đều trong {0,...,T_public−1}, rồi chọn
interval cumulative có độ dài W_a. Đúng W_a giá trị J chọn a, nên xác suất
**chính xác** là W_a/T_public=p_a. Python integer không tràn fixed-width;
không có exp, softmax hoặc xác suất float trong bước lấy mẫu. T_public và
số bit/draw bound không phụ thuộc F, tránh private-dependent normalizer.

Production policy mặc định dùng `secrets.randbelow`, với giả thiết bit OS
đều và độc lập. Rejection với bound công khai không làm số lần gọi RNG phụ
thuộc vào score. Simulator dùng private seeded PCG streams để ghép thí nghiệm;
**không** gọi PRNG này là randomness đã chứng nhận information-theoretic DP.
Mã Python chưa là một chương trình hardware constant-time được kiểm chứng.

## 3. P2 — Cả phiên và cận Bayes bằng số hữu tỷ

Lịch công khai giữ ε_j=Γ/[j(j+1)]. SQLite commit lựa chọn/bộ đếm trước khi
công bố; khởi động lại tiếp tục các khung đã lưu và không cấp lại allowance.
**Support phải điều kiện đúng thông tin.** Hai lane giả có thể cùng tọa độ;
server không nhất thiết nhìn được nhãn lane, nhưng thư viện bước sau sử dụng
nhãn đã chọn. Vì thế không giả sử thư viện là một hàm xác định của riêng
coordinate prefix. Xét transcript tăng cường O*: công bố về mặt phân tích
**hành động Q đã lấy mẫu**, gồm nhãn lane giả và cả ba khung đã chọn. Đây
là đầu ra kernel đã được bảo vệ; không gồm lane GPS thật, Z, belief, F hay
private ledger. Khi O* trước đó cố định, support/masses bước sau thực sự chung.
Cận P1 đúng với mọi E điều kiện cùng prefix O*. Hỗn hợp các E dưới hai
quỹ đạo X,X′ cũng giữ cận; nhân các conditional likelihood, không cần IID:

\[
\boxed{\frac{P(O=o\mid X,C)}{P(O=o\mid X',C)}
\le R_n:=\prod_{j=1}^n(1+\varepsilon_j)
\le e^{\sum_j\varepsilon_j}\le e^\Gamma.}
\]

Áp dụng chain rule cho O*; transcript wire thực tế O=f(C,O*) là phép chiếu,
trả tọa độ theo lịch, gọi server và giữ cache. Cộng xác suất trên những nhãn
lane/khung tương thích vẫn giữ cùng R_n. Không cần server biết nhãn lane giả.
Không dùng số candidate của một lịch sử lane ẩn để công bố posterior bound
chặt hơn riêng cho coordinate transcript; R_n ở đây độc lập lịch sử đó.

Thư viện có thể đổi theo **hành động Q ngẫu nhiên đã được bảo vệ**; không đổi theo GPS/Z/b/ψ chưa công bố.
Cận exact PMF này không phụ thuộc việc upstream Geo-I lấy mẫu float tốt đến
đâu, miễn upstream chỉ tác động qua bounded scores và không công bố lỗi,
neo, ledger hoặc kênh riêng khác. Nó không chứng nhận toàn bộ sensor/hardware
stack. Runtime API yêu cầu valid protected histories và lịch packet logic cố định.

Ở Γ=1 và11 đoạn được chọn trong0–600s:

\[
R_{11}=10763282027953/4817387520000\approx2{,}234257.
\]

Cận này chặt hơn dùng e^Γ≈2,718282. Mã làm tròn allowance xuống một số nhị
phân hữu tỷ trước khi tạo α; product runtime không lớn hơn R_n ở trên.
Budget vẫn reserve allowance gốc theo lịch công khai, không tái dùng credit
căn cứ private score hoặc private branch.

**Cận mọi thời lượng còn chặt hơn e^Γ.** Với cutoff J công khai và
s=Γ/(J+1)<1, tổng allowance sau J không quá s. Khai triển tích tail:
mỗi elementary symmetric sum bậc k không quá s^k, nên với mọi n≥J,

\[
R_n\le R_J\sum_{k\ge0}s^k=\frac{R_J}{1-s}.
\]

Ở Γ=1,J=11, toàn phiên có cận hữu tỷ
R∞≤(12/11)R11≈2,43737<e. Không cần biết trước phiên dài bao lâu và không
dùng giả thiết độc lập. Với hai hypotheses có prior đều, posterior cho một
hypothesis không quá khoảng0,7091 (prefix11 khoảng0,6908); prior mạnh vẫn
có thể cho posterior cao. Đây là cận coordinate/query channel, không phải
cam kết ẩn account/biển số hoặc utility cao cho mọi chuyến dài.

Nếu Geo-I upstream lý tưởng cho α_G=Cs·D∞, cùng transcript còn thỏa:

\[
\alpha(X,X')\le\min\{C_sD_\infty(X,X'),\log R_n\}.
\]

Hai cận của cùng đầu ra, không cộng Cs với Γ. Cs có đơn vị1/m; Γ không có
đơn vị. Nhiều phiên liên kết composition bằng tổng Cs·D∞ và tích R_n;
không có cap tám phiên và không coi privacy tổng lịch sử tự reset.

Với identity người/phương tiện hoặc future route Y, điều kiện cùng C và
side information, không có explicit-ID channel, Bayes cho:

\[
\frac{P(Y=y\mid O,C)}{P(Y=y'\mid O,C)}\le
R_n\frac{P(Y=y\mid C)}{P(Y=y'\mid C)},\qquad
P(Y=y\mid O,C)\le\frac{R_n\pi_y}{R_n\pi_y+1-\pi_y}.
\]

Đây là cận tăng suy luận, không phải anonymity/không thể đoán đúng. Prior=1,
account, IP, biển số, session activation/close và click gửi thêm nằm ngoài
những mục tiêu được bảo đảm ở đây. ψ không đi vào selector: với cùng
protected history và randomness, đổi request local không đổi transcript.
Nếu ψ tương quan với vị trí, server vẫn có thể suy luận gián tiếp qua vị trí;
không suy ra ψ độc lập hoàn toàn với O chỉ vì request được xử lý local.

## 4. U1 — Belief Geo-I có đóng góp utility được chứng minh

Điểm dưới đây là **planning score** từ protected belief; không đồng nhất với
Recall thật. Với z ở P1, tính trực tiếp:

\[
\boxed{\mathbb E_p[z]=\mu+\alpha\operatorname{Var}_w(z)\ge\mathbb E_w[z].}
\]

Σw_a z_a(z_a−μ)=E_w[z²]−μ² chứng minh đẳng thức. Khi α>0 và variance>0,
ưu tiên bằng protected scores cải thiện kỳ vọng so với cùng public base.
Nếu mọi action giống nhau, gain=0; không gọi đó là đóng góp của Geo-I.
Đây là so sánh một bước tại **cùng prefix/support**. Trung bình các gains
điều kiện vẫn không âm trên lịch sử của policy đang xét, nhưng không suy ra
cả quỹ đạo protected-belief có tổng utility cao hơn quỹ đạo Γ0 độc lập:
hai policy có thể đến các prefix/thư viện khác nhau. Benchmark controls kiểm
tra riêng kết luận thực nghiệm toàn phiên đó.

Với raw planning score V và quantizer đơn điệu F(V), còn có:

\[
\mathbb E_p[V]-\mathbb E_w[V]
=\alpha\operatorname{Cov}_w(V,z)
=\frac\alpha2\sum_{a,c}w_aw_c(V_a-V_c)(z_a-z_c)\ge0.
\]

Quantizer tương đối dùng min/max private để tận dụng S levels, nhưng chỉ
thay vector F bị chặn: không thay support, public budget hoặc nội dung query.
Các scores hữu hạn và monotone mapping là điều kiện của raw-score gain;
privacy P1 vẫn đúng với bất kỳ bounded F nào. Certificate lưu gain exact
bằng Fraction cho binary64 scores đã tính, không lấy một test score đẹp để
hiệu chỉnh ε. NaN/inf được chuyển thành bounded values trong quantizer;
không suy ra utility tốt cho dữ liệu số lỗi.

Score chính giữ90% POI-reference objective và10% độ gần hình học với các
public latent states. Vế thứ hai dùng belief Geo-I để ưu tiên placement có
ích cho các bước kế tiếp, trên các paths vốn đã hợp lệ theo đường có hướng.
Đây là secondary placement proxy, chưa là chứng minh giảm latency hoặc tăng
actual Recall. Đối chứng dùng public prior thay b và Γ0 giúp kiểm tra phần
cải thiện nào đến từ pool, phần nào từ protected adaptation.

## 5. U2 — Invariant POI pool, không nới floor theo vị trí private

Tạo portfolio từ reply signatures trên **toàn mạng công khai**. Các state
trả cùng POI pool được giảm về representative; beam width24 chọn các gói K5
có public-reference coverage cao. Beam là heuristic, không tuyên bố tìm được
optimum của bài toán set cover. Không đọc route labels/heldout GPS/attack scores.

Gọi η_i là public POI-reference mass, coi các binary64 coefficients đã công
bố như số hữu tỷ rồi chuẩn hóa chính xác. Q(U)=Σ_{i∈U}η_i. Đặt một lần:

\[
\rho=\max_{a\in\mathcal A_{initial}}Q(U_a)-1/200.
\]

Chỉ giữ đoạn có Q(U_{a,τ})≥ρ ở mọi khung. Nếu khung đã gửi trước đó đạt
ρ, đoạn đứng yên cũng đạt; do đó support không rỗng. Induction từ khung đầu
cho invariant ở mọi bước, không phụ thuộc private belief và không nới floor
theo thời gian. So sánh floor bằng Fraction, không dùng tolerance làm tròn
một action thiếu QoS thành đạt. Invariant này là weighted PUBLIC-reference
coverage, **không** là lower bound cho mọi actual GPS/purpose.

Goal công khai hiện tăng maximum public-reference quality từ90,6925% ở thư
viện cũ lên93,7269%, nên targetρ mới khoảng93,2269%. Cần benchmark để biết
quality này chuyển thành service thực tế bao nhiêu; không tự δ_b=δ_proxy=0.

## 6. U3 — Chứng chỉ theo request thật và service ceiling

R_ψ(x) là reference top-5 của mục đích thật, U là pool đã nhận, A là union
của POI có thể nhận qua mọi point query công khai ở cùng L/catalogue.
Rỗng là N/A. Vì U⊆A:

\[
\boxed{H_\psi(x,U)=\frac{|R\cap U|}{|R|}
=\frac{|R\cap A|}{|R|}-\frac{|R\cap(A\setminus U)|}{|R|}.}
\]

Đây là bridge không cần calibrated belief hoặc đúng private-purpose prior.
Vế đầu là attainable service ceiling, vế thứ hai là phần thiếu do Q pool.
Nó đúng cho mọi purpose/radius/destination có reference hợp lệ, với catalogue,
ranking, tie-break và status validity như đã khai báo. Chỉ tính certificate
thật ở local/evaluator; không dùng kết quả thiếu để trigger một request thêm.

Nếu c=|A\setminus U| biết từ public pool, có cận đơn giản:

\[
H_\psi(x,U)\ge\max\{0,(|R\cap A|-c)/|R|\}.
\]

Cận này có thể bằng0 khi pool còn thiếu nhiều POI; phải báo đúng, không thay
bằng public averageρ. Local sorting giữ **danh sách đầy đủ**; lấy top-5 chỉ
trong metric. Ranking lemma bảo đảm intersection bằng overlap trong metric.
L30 là độ sâu mỗi category mỗi Q, không phải150 POI tổng tất cả categories.

Trong benchmark, toàn bộ418 POI có thể đạt bằng một số point query, nhưng
một gói K5 chỉ trả một phần. Nếu provider cho bulk download catalogue tĩnh,
control download một lần có thể tránh tọa độ queries. Vì vậy phải ghi hợp
đồng point-query hữu hạn, cùng K/L và chi phí; không nói hơn mọi LBS.

## 7. U4 — Di chuyển Q không làm mất kết quả cho request thật

Bản thứ hai dùng thư viện `DominanceMotionLibrary`. Từ khung Q giả đã chọn,
chỉ chấp nhận khung tiếp theo nếu đi được trong20s trên đồ thị có hướng và:

\[
\boxed{U_{t-1}\subseteq U_t.}
\]

Ở mỗi bước, pool được tính từ reply signatures công khai của provider, không
từ GPS, ψ, belief hay reference thật. Đoạn đứng yên giữ support không rỗng;
8 hướng công khai sinh thêm đường đi. Mỗi track được di chuyển một lần trong
mỗi20s, sau đó kiểm tra tiếp điều kiện union; không cộng ba bước đi vào cùng
một khung. Maximize số POI rồi tiến theo heading là heuristic công khai,
không là nghiệm tối ưu và không là đóng góp riêng của Geo-I.

**Bảo đảm pointwise.** Với bất kỳ GPS x, request ψ và reference hợp lệ R:

\[
H_\psi(x,U_t)-H_\psi(x,U_{t-1})
=\frac{|R\cap(U_t\setminus U_{t-1})|}{|R|}\ge0.
\]

Ở đây so sánh **cùng x và ψ**, không khẳng định Recall của hai thời điểm có
GPS khác nhau phải tăng. Induction U_t⊇U_0 cho:

\[
\boxed{H_{\psi_t}(x_t,U_t)\ge H_{\psi_t}(x_t,U_0)\quad\forall t.}
\]

Vế phải là control giữ nguyên Q ban đầu **cùng một lần lấy mẫu**, được đánh
giá tại GPS/request hiện tại. Không phải baseline Geo-I cũ. Bảo đảm không
cần posterior b calibrated, không cần sai số proxy bằng0; đúng cả purpose
ngoài bốn purpose khảo sát nếu ranking/reference đáp ứng cùng lemma local
sorting. Rỗng vẫn là N/A. Nội dung ứng dụng là danh sách phù hợp sắp xếp
đầy đủ; không bắt người dùng nhận top-5.

Lấy kỳ vọng có điều kiện theo lịch sử, rồi trung bình cả phiên, giữ nguyên
bất đẳng thức; không giả sử IID. Union/cache chỉ bổ sung POI nên bảo đảm
cũng đúng cho within-epoch cache của thí nghiệm. Chỉ có catalogue/status cố
định và response contract đã khai báo: expiry, category/status thay đổi hoặc
server trả replies khác phá giả thiết và cần chứng chỉ riêng. Road state và
coordinate resnap phải trả cùng pool; verifier kiểm tra điều này thay vì
mặc định các lane tại tọa độ trùng nhau luôn có reply giống nhau.

Điều kiện U_t⊇U_{t−1} cũng giữ public floor vì η≥0; không nới floor. Nó
có thể hạn chế khả năng di chuyển và thích nghi. Một pool ban đầu thiếu POI
vẫn có thể thấp hơn baseline; nếu không mở rộng được thì chỉ bảo đảm bằng
static control. Không cam kết QoS cao ở mọi chuyến dài với budget hữu hạn.

## 8. Thí nghiệm và ranh giới kết luận

Source/config được đóng băng trước khi sinh cohort native mới: train24,
selection12,test24 families. Train/selection có một private draw mỗi trip,
test có ba draws; mọi phương án nhận cùng realization Geo-I cho mỗi draw.
Không fit defense trên test, không đổi mẫu hay loại draws xấu. Giữ baseline,
legacy PQB, finite kernel trên thư viện cũ, portfolio Γ0, portfolio public-prior
Γ1 và portfolio protected-belief Γ1. Primary chọn trước là phương án cuối.

Attacker fit TRAIN, chọn theo loss ở SELECTION, lưu trước TEST. Utility CI
bootstrap theo family, ba draws nằm bên trong family; gate−2pp cả macro và
mỗi purpose, thêm kiểm tra từng draw. Benchmark S2 là coordinate proxy của
cửa sổ dừng, S4–S6 là hệ quả suy luận có điều kiện, không có metric identity
vật lý mới hoặc claim anonymity. S9/S10 vẫn công khai start/close.

Sau bản đầu, đóng băng riêng bản dominance trước khi sinh cohort thứ hai:
seed công khai2026101243, loại84 junction groups của original24 và development60,
loại route gần trùng theo Jaccard đã khai báo. Giữ đủ baseline, legacy PQB,
portfolio cũ, static matched-initial, dominance Γ0, dominance public-prior Γ1
và dominance protected-belief Γ1. Primary chọn trước là dominance cuối.
Đây vẫn là synthetic cùng bản đồ/generator, không phải xác nhận cross-city.

Các giả thiết về uniform independent OS bits, trusted local ledger và logical
request clock là điều kiện của bảo đảm; không có chứng nhận phần cứng,
constant-time Python, bộ Geo-I float hoặc secrecy của packet metadata khác.
Code được chứng minh về PMF hữu hạn và kiểm tra bằng finite oracles; chưa
là một proof assistant/machine-verified implementation.

## 9. Định lý nối với mã

| Kết quả | Mã bảo đảm | Kiểm tra |
|---|---|---|
| P1 PMF/support/ratio | `FiniteCenteredKernel.integer_weights` | enumerate scores và mọi integer draw trên mô hình nhỏ |
| P2 composition | lịch Fraction, `PersistentQueryBudget` | mọi prefix, cached segment/restart, product exact |
| P2 mọi thời lượng | `all_duration_ratio` dùng bound tail hữu tỷ | prefix1..1000 ở Γ0/0,5/1/2; posterior prior0/1 |
| U1 variance/covariance | integer F và exact probabilities | Fraction identity và raw planning gain |
| N/A | `DefinedPublicPurposeTable` | defined public masses=1, N/A không đổi thành Recall0 |
| U2 floor invariant | public signatures, beam và Fraction quality | support public, holds, từng khung qua restart |
| P2 support tăng cường | fake lane IDs/full sampled segments → wire projection | trùng tọa độ, support khác theo lane ẩn; product bound |
| U4 actual utility ≥ static | `DominanceMotionLibrary` chỉ giữ union supersets | mọi khung, đường có hướng; cùng initial draw và từng actual-purpose Recall |
| U3 real request | `reference_certificate` | hit=attainable−missing, N/A và service ceiling |
| Bound score errors | quantizer bounded | NaN/inf/extreme finite values không phá phạm vi F |
| Secret content | giữ QuerySpec/GPS ở local | network schema unchanged; mọi category/L30 cố định |

Nguồn background: [Mironov, lỗi do floating point trong DP](https://www.microsoft.com/en-us/research/publication/on-significance-of-the-least-significant-bits-for-differential-privacy/)
và [Canonne–Kamath–Steinke, exact sampling cho discrete DP](https://arxiv.org/abs/2004.00010).
Công thức centered tilt và các derivations ở đây dành cho prototype này;
không gọi nó là discrete-Gaussian mechanism hoặc thuật toán lấy nguyên từ
hai paper. Nguyên tắc composition/postprocessing là background DP/Geo-I.

[Bằng chứng mới](../../artifacts/benchmarks/finite_pool_20261012_v1/)
và [bản dominance](../../artifacts/benchmarks/dominance_motion_20261012_v1/)
được ghi riêng; [bản10/10](2026-10-10_multistep_privacy_utility.md) và các số
liệu cũ được giữ nguyên. Chỉ các phép so sánh cùng cohort/config mới là paired.
