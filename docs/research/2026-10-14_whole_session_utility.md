# Geo-I + bounded Q: cận utility cho cả phiên thích nghi

## Phát biểu và phạm vi

Giữ Geo-I/REM và bộ lọc ngân sách mỗi phiên. Lớp Q lấy mẫu chung ba khung,
K=5, L=30 mỗi category, gửi mỗi20s và chọn đoạn mỗi60s. Support, các mục
tiêu đi trên đường, cache đã nhận và trọng số public có thể đổi theo lịch sử.
Chỉ điểm số bị chặn nhận belief Geo-I; request và GPS thật dùng để trả kết
quả ở local. Chứng minh dưới đây áp dụng cả exploration cũ và targeted
exploration mới, không yêu cầu belief là posterior đúng.

So sánh **hai phiên bản của cùng planner**: phiên bản P dùng Γ>0 để nghiêng
phân phối theo belief, và phiên bản W dùng Γ=0 với cùng quy tắc public. Hai
phiên có thể lấy mẫu ra Q, cache và support khác nhau; không bắt chúng phải
có cùng realized prefix. Đây là điểm mở rộng so với cận từng bước trước.

Điều kiện: cùng bản đồ, catalogue, public clock, quy tắc support/masses và
provider law; bit ngẫu nhiên đều độc lập. Không có kênh trực tiếp GPS, ψ,
private ledger, lỗi phụ thuộc private score hoặc identity gửi server. Quỹ
đạo và quy tắc request dùng để đánh giá phải cố định trong phép so sánh,
hoặc tuân theo cùng một môi trường tương tác đã được mô hình hóa. Mã PCG
trong benchmark và Geo-I float chưa được chứng nhận như một toàn bộ stack.

## U3 — Cận χ² trên toàn đường đi

Với cùng augmented public prefix H, gọi w_a(H)>0 là public base. Điểm số
z_a∈[0,1], μ=Σ_aw_az_a và α_j=ε_j/(2+ε_j). Kernel hữu hạn là:

\[
p_a=w_a[1+\alpha_j(z_a-\mu)],\qquad
\varepsilon_j=\frac{\Gamma}{j(j+1)}.
\]

Belief có thể còn ngẫu nhiên khi điều kiện H. Lấy kỳ vọng điểm số dưới
phân phối private-state của P cho \bar z_a∈[0,1]. Do w cố định ở H, kernel
marginal vẫn là w_a[1+α_j(\bar z_a−Σ_cw_c\bar z_c)]. Vì vậy:

\[
\chi^2(p_j\Vert w_j\mid H)
=\alpha_j^2\operatorname{Var}_{w_j}(\bar z)
\le\frac{\alpha_j^2}{4}.
\]

Đặt L_j=P(H_j)/W(H_j). Common support bảo đảm tỷ số tồn tại. Tại prefix
H_{j−1}, E_W[(L_j/L_{j−1})²|H_{j−1}]≤1+α_j²/4. Nhân với L_{j−1}²
rồi lấy kỳ vọng lặp, không cần độc lập giữa các bước:

\[
\boxed{\chi^2(P_n\Vert W_n)
=\mathbb E_W[L_n^2]-1
\le\prod_{j=1}^{n}\left(1+\frac{\alpha_j^2}{4}\right)-1.}
\]

Responses có cùng conditional provider law ở cùng query/history, nên không
thêm factor. Công bố từng khung, cache và wire thực tế là projection hoặc
hậu xử lý; không tăng TV. Điều kiện support chung được áp dụng lên
**augmented sampled fake-lane/segment history**, không giả định rằng tọa độ
server nhìn thấy xác định duy nhất lane.

Với bất kỳ h∈[0,1] là hàm của toàn phiên và cùng private evaluation inputs:

\[
\boxed{|\mathbb E_P[h]-\mathbb E_W[h]|
\le\operatorname{TV}(P_n,W_n)
\le\frac12\sqrt{\prod_{j=1}^{n}(1+\alpha_j^2/4)-1}.}
\]

Chọn h là Recall của một thời điểm, Recall trung bình cả phiên, một purpose
cụ thể hoặc indicator một sự kiện dịch vụ. Cận không chỉ áp cho planning
score, không cần sai số belief bằng0 và không giả sử GPS points IID.
Nó không áp trực tiếp cho latency/cost chưa được chặn, không chứng minh
baseline Geo-I cũ có cùng W, và không nói belief luôn cải thiện utility.

## Cận mọi thời lượng và số học chứng chỉ

Sau prefix J, α_j≤Γ/[2j(j+1)], nên:

\[
s_J=\sum_{j>J}\frac{\alpha_j^2}{4}
\le\frac{\Gamma^2}{48J^3}.
\]

Dùng tổng elementary symmetric sums không lớn hơn s_J^k; nếu s_J<1,
tail product≤1/(1−s_J). Với Γ1,J64, code tính prefix bằng Fraction và
chặn tail bằng số hữu tỷ. Square-root display không phải chứng chỉ float;
code dùng integer square root để tạo **cận hữu tỷ làm tròn lên**.

| Γ | 11 selections,0–600s: cận shift (pp, display) | Mọi thời lượng (pp, display) |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0,5 | 3,029 | 3,030 |
| 1 | 5,523 | 5,526 |
| 2 | 9,453 | 9,459 |

Ở Γ1, cận hữu tỷ mọi thời lượng nhỏ hơn **553/10000**, tức **5,53pp**.
Đây là theorem về ideal Q PMF, không phải khoảng tin cậy benchmark. Cận
đúng cả khi proxy sai; đổi lại adaptation có mức ảnh hưởng hữu hạn. Không
cam kết service tốt khi public base không truy hồi được POI cần thiết.
Nếu chứng nhận được E_W[h]≥c thì E_P[h]≥max(0,c−δ); hiện chưa chứng nhận
c tuyệt đối cho toàn bộ miền GPS/purpose. Mean hay bootstrap CI của W không
được thay vào c để biến một thực nghiệm thành bảo đảm toán học.

[Code exact certificate](../../benchmark/whole_path_chi_square.py).
Oracle duyệt PMF toàn đường đi trên support/masses thích nghi, kiểm tra χ²,
TV và mọi bounded utility qua hàm indicator đạt supremum. Cận coupling/KL
độc lập được giữ tại [session_utility_bounds.py](../../benchmark/session_utility_bounds.py).

## Privacy và utility nền tảng vẫn giữ

1. **P1/P2:** tại cùng H, p_a(X)/p_a(X′)≤1+ε_j. Chain rule và projection
   cho R_n=Π_j(1+ε_j), mọi thời lượng Γ1 có R≤2,437371. Giới hạn tăng
   posterior odds cần target→private trajectory→wire khi điều kiện public
   context; prior mạnh, account/IP hoặc biển số vẫn có thể làm identity rõ.
2. **Geo-I:** protected upstream history phải bao gồm các private noisy-test
   branches khi phân tích. Bộ lọc prospective integer-cost giữ Σc_tu≤Cs;
   public pacing và Q là hậu xử lý của history này. Trong ideal upstream,
   log-likelihood ratio wire≤min{Cs·D∞,logR_n}. Không dùng calibration của
   belief làm giả thiết privacy; không gọi Geo-I float là đã certified.
3. **U1:** static valid cache U_t⊇U_0. Với cùng GPS/request tại t, local
   sorted Recall(U_t)≥Recall(U_0). Không cần belief, không so hai GPS khác
   nhau. Expiry/status động có thể phá inclusion của valid cache.
4. **U3:** bounded private adaptation so với cùng adaptive public base trên
   cả phiên. Đóng khoảng trống mà một snapshot certificate không giải quyết.

## Định lý → giả thiết → mã → kiểm tra

| Kết quả | Giả thiết/cơ chế | Kiểm tra |
|---|---|---|
| Full-session privacy | public support/masses; bounded integer score; persisted Q ledger | independent PMF/support/path audit |
| Geo-I spatial bound | ideal extended test/anchor kernel; prospective cap Cs | core integer ledger + conditional argument |
| Cache utility tại GPS thật | static valid records; fixed ranking/ties | rebuild replies/union, recompute actual Recall |
| Full-path utility distortion | same public-base policy at matched prefix; uniform bits | exact adaptive PMF/χ²/TV oracle |
| Numerical proxy trouble | uniform protected-score fallback; same public support/clock | malformed-belief tests; flag stays local |
| Restart/retry | trusted cache/ledger, replay committed Q before clock advance | crash injection; same-frame retry; changed-policy rejection |
| Whole pipeline | original Geo-I engine fully restored; query/GPS local separation | real-engine component-loop equivalence test |

[Chi tiết privacy/cache trước](2026-10-12_balanced_cache_proof.md) và
[predictive filter](predictive_filter_argument.md) được giữ nguyên.

## I1 — Chặn attacker mạnh tùy ý, không cần fit một identity classifier

Gọi Y là target riêng tư có trước kênh đang phân tích: identity người/phương
tiện, private intent hoặc một planned future choice. Điều kiện cùng C,
Y đi vào observation qua private inputs/score generator; không có ID channel
trực tiếp. W là public-base policy, nên O_W độc lập Y. Target thực tế do
chính public output gây ra không có tính độc lập này; không áp kết luận prior
cho một future action như vậy.

U3 áp cho mỗi conditional law P(O|Y=y,C), kể cả hỗn hợp các private
trajectories tương thích y. Do tính lồi, với prior π_y bất kỳ:

\[
\operatorname{TV}(P_{Y,O\mid C},\ \pi\otimes W)
=\sum_y\pi_y\operatorname{TV}(P_{O\mid y,C},W)\le\delta_\Gamma.
\]

Với attacker f(O,C) và reward r(Y,f)∈[0,1], chênh lệch expected reward giữa
hai joint laws≤δ. Dưới π⊗W, quan sát O không cải thiện best constant/prior
strategy. Vì vậy **mọi attacker**, không chỉ bank đã fit, thỏa:

\[
\boxed{\mathrm{Success}^{*}_{after\ Q}
\le\mathrm{Success}^{*}_{prior}+\delta_\Gamma.}
\]

Với Γ1, có thể dùng cận hữu tỷ δ=0,0553 cho mọi thời lượng một phiên.
Với 0–1 classification, prior success=max_yπ_y; với Hit100, đó là prior mass
lớn nhất trong một ball100m; không mặc định bằng1/số điểm. Bounded Bayes loss
có cận giảm risk tương ứng. MAE mét chưa được chặn không trực tiếp là reward
[0,1]; cần public finite-domain diameter hoặc một loss normalization riêng.

- Hai planned route choices có prior50/50: **accuracy trung bình≤55,53%**.
- Hai mươi identity candidates có prior đều: **accuracy trung bình≤10,53%**.
- Prior đã biết gần chắc identity: cận có thể vacuous; Q không xóa tri thức
  đã có. Nhiều phiên liên kết cần composition khác.

Các số này là conditional examples, không là một benchmark identity mới.
Chúng không giới hạn posterior của *từng* wire ở55,53%; cận posterior
pointwise từ R∞ vẫn có thể tới70,91% với prior50/50 trên rare outcomes.
Average success và pointwise posterior là hai đại lượng khác nhau.

### I2 — Lượng thông tin riêng tư trung bình

Tại cùng prefix, KL(p_j||w_j)≤χ²(p_j||w_j)≤α_j²/4. Chain rule cho KL
cho K_n=Σ_jα_j²/4. Information-radius identity, với một common W độc lập X,
cho private exogenous input X:

\[
I(X;O\mid C)
=\mathbb E_X KL(P_{O\mid X,C}\Vert W)-KL(P_{O\mid C}\Vert W)
\le K_n.
\]

Do data processing, cùng cận áp cho target Y qua input model nêu trên.
Không dùng I của một future trajectory chứa chính public output để suy ra
kết luận này. Với Γ1 và rational tail sau64 selections, K∞/ln2<**0,018bit**.
Code dùng ln2>0,69 (hai terms dương đầu của atanh series cho ln2 đã đủ),
nên bit bound là **số hữu tỷ làm tròn bảo thủ**, không dựa vào log float.
Đây là additional information do **kênh Q logical một phiên**, không gồm
account, hardware timing, logs hay lịch sử các phiên khác.

[Certificate](../../benchmark/bounded_inference.py) không mặc định prior
uniform: chưa cung cấp prior reward bound thì `average_success_upper=None`.
Oracle kiểm tra optimal Bayes attacker bằng PMF exact, cả prior không đều.

## G1 — Vai trò Geo-I ở mức từng GPS fix

Cs=0,23m⁻¹ là worst-case cap của **cả history**, không là epsilon của từng
GPS fix. Trong ideal extended history, first fresh release có cost1, mỗi
private read sau có test+refresh cost≤2; c_0=1 vàΣc≤23. Với các distances
d_i giữa hai scheduled GPS-input histories:

\[
\log LR_{GeoI}\le u\left[d_0+
\operatorname{TopSum}_{22}(d_1,d_1,d_2,d_2,\ldots)\right]
\le C_s\max_i d_i.
\]

Tại cùng extended history, mọi spending/stop decisions agree. Chi phí của
một later input chỉ là0/1/2 units; chọn22 largest candidate unit-distances
là upper bound cho mọi feasible adaptive allocation. Không giả sử independence.

Chỉ first fix khác10m: log-ratio Geo-I≤0,1; chỉ một later scheduled fix khác
10m:≤0,2. Nếu mọi fix đều khác1m, cap≤0,23. Whole wire nhận **minimum**
của cận Geo-I này và logR_n của Q. Điều này chỉ ra phần Geo-I đóng góp ở
local/event adjacency, trong khi Q thêm global disclosure cap giữa các
trajectories xa nhau. Các GPS chỉ dùng cho utility local không thêm private
transcript factor. Đây vẫn là ideal-upstream claim; float implementation
chưa được gọi là certified.

[Code](../../benchmark/geo_i_event_bound.py) và oracle duyệt mọi0/1/2 unit
allocation. Không công bố distances chọn theo hidden GPS như một bound của
một wire cụ thể; đây là adjacency/pairwise privacy profile.

## U4 — Không thể có utility cao tùy ý với retrieval nhỏ và Γ hữu hạn

Giả sử một category có M equiprobable private regions, mỗi region có một
reference set R_i gồm k POI, **các R_i pairwise disjoint**. Gọi C là số records
khác nhau thật sự available, gồm cả metadata đã có offline. Với W độc lập
private region:

\[
\mathbb E_W[Recall]
=\mathbb E_W\frac{\sum_i|R_i\cap U|}{Mk}
\le\min\{1,C/(Mk)\}.
\]

Joint-TV bound I1/U3 áp cho bounded local Recall dù local client biết region:

\[
\boxed{\mathbb E_{P_\Gamma}[Recall]\le C/(Mk)+\delta_\Gamma,
\qquad C\ge Mk(r_{target}-\delta_\Gamma).}
\]

Ví dụ M1000,k5,Γ1,desired Recall95% cần ít nhất khoảng**4474** available
unique records trong mô hình disjoint này. Một fresh category pool K5×L60≤300
không đủ: upper bound≤11,53%. Cache theo thời gian, side information public
thu hẹp miền, các reference sets overlap, hoặc một provider contract khác
có thể đổi bài toán. Không áp con số4474 trực tiếp cho catalogue418 có các
reference overlaps trong benchmark; đây là necessary lower bound có điều kiện.

Đây là lý do phải báo cáo privacy–utility–data cost và không suy rộng99,60%
sang mọi city/catalogue. Bulk catalogue được cấp offline có C=N, không có
mâu thuẫn khi nó đạt100% và không cần point queries. Không lấy0 queries làm
C=0 nếu metadata đã được cung cấp sẵn.

[Certificate](../../benchmark/retrieval_cost_bound.py); oracle kiểm tra mọi
cache mask trên reference sets disjoint.
