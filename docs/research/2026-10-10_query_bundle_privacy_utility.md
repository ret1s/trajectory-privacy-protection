# Geo-I–PQB: chứng minh privacy và utility có điều kiện

Đây là phần hoàn thiện của [cơ chế chọn gói Q](2026-10-10_probabilistic_query_bundles.md).
Geo-I/REM, noisy reuse, ngân sách GPS từng phiên và xử lý ψ tại local giữ nguyên.
PQB vẫn là prototype opt-in; các định lý sau không tự áp dụng cho planner xác
định cũ hoặc các kênh account/IP. Tất cả cận privacy là số thực lý tưởng;
mã và audit cung cấp kiểm tra số, chưa chứng nhận pure-DP của sampler float.

**Kết quả chính:** thay cận β chung bằng một cận tính từ bảng công khai; có cận
utility kỳ vọng và cận từng gói nếu thư viện có coverage floor. Số Q không cần
cố định, nhưng tăng K không phải điều kiện để những chứng minh này đúng.

## 1. Đặc tả và miền áp dụng

Giữ ngữ cảnh công khai C, một vòng chọn gói và thư viện hữu hạn A cố định.
`E` là lịch sử tọa độ đã được Geo-I bảo vệ; `b(E)` là phân bố trên N đại diện
vị trí công khai. Gói a gồm **toàn bộ tọa độ, thứ tự và K(a)**. Đặt

\[
g_{xa}\in[0,1],\quad c_a=K(a)/K_{\max},\quad
s_b(a)=\frac{\sum_x b_xg_{xa}+\lambda(1-c_a)}{1+\lambda},\qquad
T(a\mid b)=\frac{w_a e^{\eta s_b(a)}}{\sum_{a'}w_{a'}e^{\eta s_b(a')}},
\quad\eta=\beta/2.
\]

w_a>0 và tổng bằng 1; λ, β, A, g và w đều được xác định công khai. Không có
GPS thật hoặc ψ trong điểm s. Đây là exponential selection áp dụng trên đầu
ra Geo-I, dựa trên [Dwork–Roth, §3.4](https://privacytools.seas.harvard.edu/sites/g/files/omnuum6656/files/privacytools/files/the_algorithmic_foundations_of_differential_privacy.pdf)
và [Geo-I, §3](https://arxiv.org/html/1212.1984v3).
Các định lý dưới đây được suy trực tiếp từ kernel này; không đề xuất một định
nghĩa riêng tư mới hoặc cho rằng exponential mechanism là đóng góp mới.

Trong audit, g là nearest-reference Recall@5, L30, tất cả danh mục, trên 36 vị
trí công khai. Tập 36 vị trí là miền **ước lượng/utility**, còn miền đầu ra REM
vẫn gồm toàn bộ 66.189 trạng thái đường. Privacy của kernel đúng cho mọi b
trên miền ước lượng, không cần GPS thật nằm trong 36 vị trí. **Utility trên
GPS ngoài lưới cần thêm cận sai số đại diện**, không suy từ privacy.

## 2. Định lý P1: cận privacy theo biến thiên tương đối của điểm

Với hai beliefs b,b′, đặt d_a=s_b(a)−s_b′(a). Tỷ số chuẩn hóa là:

\[
\frac{Z_b}{Z_{b'}}=\sum_a T(a\mid b')e^{\eta d_a}.
\]

Vì đây là trung bình có trọng số,
`η min_a d_a ≤ log(Z_b/Z_b′) ≤ η max_a d_a`. Do đó

\[
\left|\log\frac{T(a\mid b)}{T(a\mid b')}\right|
\le\eta\,[\max_a d_a-\min_a d_a].
\]

Điểm thay đổi cùng một lượng ở mọi gói không làm thay đổi phân phối chọn.
Vì vậy, chặn **biến thiên tương đối giữa các gói** tốt hơn chặn từng điểm riêng.
Phần chi phí λ(1−c_a) triệt tiêu khi lấy hiệu giữa b và b′.

Định nghĩa hằng số hoàn toàn công khai:

\[
\boxed{\kappa=\frac1{1+\lambda}
\max_{a,a'}\left[\max_x(g_{xa}-g_{xa'})-\min_x(g_{xa}-g_{xa'})\right].}
\]

Với mọi cặp b,b′, hiệu giữa d_a và d_a′ là hiệu hai trung bình của
`(g_xa−g_xa′)/(1+λ)`, nên không lớn hơn range của dãy này. Suy ra

\[
\boxed{T(a\mid E)\le e^{\varepsilon_Q}T(a\mid E'),\qquad
\varepsilon_Q=\eta\kappa=\frac\beta2\kappa
\le\frac\beta{1+\lambda}\le\beta.}
\]

Hằng số κ là supremum đúng của oscillation của hiệu điểm trên toàn simplex:
chọn b và b′ là point masses tại hai x đạt max/min cho một cặp gói đạt κ sẽ
đạt oscillation đó. Nhưng **ε_Q vẫn có thể lớn hơn privacy loss chính xác của
kernel**, vì ta chặn mẫu số bằng min/max. Không gọi ε_Q là privacy loss tối thiểu.

Nếu κ=0, phân phối gói độc lập với b dù các điểm riêng có thể thay đổi theo
vị trí. Nếu κ>0, có thể chọn β từ một yêu cầu công khai ε_target:

\[
\beta\le 2\varepsilon_{\rm target}/\kappa.
\]

Đây là calibration theo cấu trúc bảng công khai; không fit theo accuracy của
attacker trên test. Prototype giới hạn β≤40 để giữ support số dương trong
float; đó là giới hạn triển khai, không phải hằng số trong định lý số thực.

## 3. Định lý P2: quỹ đạo, đầu ra server và nhãn suy luận

Tại vòng t, giữ transcript trước đó o_<t cố định. Thư viện, w_t, bảng g_t và
β_t có thể phụ thuộc **C và o_<t công khai**, nhưng không được prune theo E_t
hoặc ψ. Tính κ_t từ bảng của vòng đó. Mỗi conditional kernel có cận
ε_Q,t=β_tκ_t/2 cho mọi lịch sử E_t.

Ngay cả khi phân bố E_t điều kiện theo X và o_<t khác nhau, phân phối của gói
là một hỗn hợp các kernel T_t có cùng support và cùng cận. Các hỗn hợp đó có
cận ε_Q,t: mọi giá trị T_t(a|E_t) nằm trong một khoảng có tỷ số max/min không
quá exp(ε_Q,t), và trung bình bất kỳ cũng nằm trong khoảng đó.

Nhân các conditional likelihood của những vòng đã quan sát:

\[
\Gamma_Q=\sum_t\varepsilon_{Q,t},\qquad
P(O=o\mid X)\le e^{\Gamma_Q}P(O=o\mid X').
\]

Nếu toàn bộ O còn là hậu xử lý của một cơ chế Geo-I cho quỹ đạo với cận
`α_G=Σ_s C_s D∞(X_s,X_s′)`, phải đồng thời thỏa hai cận:

\[
\boxed{\alpha(X,X')\le\min\left\{
\sum_s C_sD_\infty(X_s,X_s'),\ \Gamma_Q\right\}.}
\]

Không cộng ngân sách GPS với ε_Q: đây là **hai cận của cùng đầu ra**. Các lần
đọc GPS vẫn tiêu ngân sách Geo-I như trước; các lần lấy mẫu gói mới cộng vào
Γ_Q. Khi dùng lại một gói đã lấy mẫu tại cadence công khai, không tự thêm một
ε_Q mới. Hết Γ_Q không cho phép đọc GPS hoặc tạo gói mới miễn phí.

Nếu β/κ hoặc số vòng thích nghi theo transcript công khai, một cam kết trước
phải chặn `sup_o Σ_t ε_Q,t(o_<t)` trên mọi đường đi, hoặc dùng một bộ kiểm tra
ngân sách công khai trước mỗi vòng. Không lấy tổng ở đường đi thuận lợi nhất.
Session stop/start riêng, cờ fresh/reuse, ledger GPS, Z, b hoặc debug log gửi
trực tiếp có thể vượt qua PQB; cận Γ_Q không tự bao phủ các kênh đó.

Đầu ra server, thứ tự Q, K, bytes và retries chỉ được đưa vào O với cận trên
khi chúng là hậu xử lý của gói/C/lịch sử công khai, với cùng kernel server
(kể cả server có chiến lược thích nghi). Click hay account được gửi thêm cần
phân tích riêng. Transition K động cần thư viện đường khả thi công khai khi
giữ cùng o_<t; static audit hiện tại chưa triển khai phần transition đó.

Với prior π và ε chung cho mọi ứng viên:

\[
P(X=x\mid O)\le
\frac{e^\varepsilon\pi(x)}{e^\varepsilon\pi(x)+1-\pi(x)}.
\]

Chứng minh bằng Bayes: các likelihood l_x′ đều ít nhất e^(−ε)l_x; thay vào
mẫu số. Khi prior có ít nhất hai ứng viên có khối lượng dương và ε hữu hạn,
posterior của từng ứng viên không thể bằng 1. Cận số mới là phần có ý nghĩa:
“nhỏ hơn 1” đơn thuần vẫn có thể là 99,999%.

Với nhãn Y như người, phương tiện vật lý, next edge hay destination, nếu Y
ảnh hưởng kênh đang xét qua lịch sử E, cùng C và không lộ qua định danh khác,
các hỗn hợp P(O|Y) cũng có cận Γ_Q. Prior đã gồm side information đang điều
kiện; đây là cận **mức tăng odds**, không xóa identity/route đã biết. Không
chuyển ε_Q của một snapshot thành cận cho mọi chuyến đã liên kết.

### Số học hữu hạn: phát biểu có điều kiện

Nếu phân phối triển khai T_hat cho mọi E gần kernel lý tưởng trong TV không
quá ζ, thì với mọi biến cố B:

\[
\widehat T_E(B)\le T_E(B)+\zeta
\le e^{\varepsilon_Q}T_{E'}(B)+\zeta
\le e^{\varepsilon_Q}\widehat T_{E'}(B)+(1+e^{\varepsilon_Q})\zeta.
\]

Đây là một cận approximate-DP **nếu đã chứng minh ζ riêng**. Audit float hiện
tại chưa chứng minh ζ=0 hay một ζ dương cụ thể. Phải kiểm chứng cả REM và RNG
để chứng nhận một triển khai đầy đủ; test inequality pass không thay công việc đó.

## 4. Định lý U1: điểm utility kỳ vọng và cận regret tốt hơn

Giữ E, b và thư viện cố định. Đặt p_a=T(a|b). Với một phân phối so sánh ν trên
cùng thư viện, khai triển KL(ν||p)≥0 cho

\[
\eta\mathbb E_p s-\operatorname{KL}(p\|w)
\ge\eta\mathbb E_\nu s-\operatorname{KL}(\nu\|w).
\]

Đây là tính chất tối ưu của Gibbs distribution; suy trực tiếp từ KL, không
giả sử planner tìm được một optimum hình học trên mọi tập Q ngoài thư viện.

Đặt s*=max_a s(a), `B_d={a:s(a)≥s*−d}`, `W_d=Σ_(a∈B_d)w_a`. Chọn ν là w
điều kiện trên B_d; khi đó KL(ν||w)=log(1/W_d) và Eνs≥s*−d. Bỏ số hạng
KL(p||w)≥0 để được:

\[
\boxed{r:=s^*-\mathbb E_p s\le
\min_{d\ge0}\left[d+\frac{\log(1/W_d)}\eta\right].}
\]

Chỉ cần xét các gap s*−s(a); lấy d=max gap cho r≤max gap, kể cả khi β rất nhỏ.
Với β=0, p=w và tính kỳ vọng trực tiếp; không chia cho η=0. Cận mới dùng tổng
khối lượng các gói **gần tốt nhất**, thay vì chỉ gói tối ưu và bỏ được số hạng
“+1” của cận tích phân đuôi cũ. Có thể tính chính xác r từ p tại local.

Cận xác suất cho một gói vẫn có dạng:

\[
P(s^*-s(A)\ge\tau)\le
\min\{1,e^{-\eta(\tau-d)}/W_d\},\quad0\le d<\tau.
\]

Chứng minh: tử số của gói kém không quá e^(η(s*−τ)); mẫu số ít nhất
W_d e^(η(s*−d)). Tất cả giá trị b, regret hoặc chứng chỉ local phải giữ local;
gửi chúng lên server mở thêm một kênh E không nằm trong định lý P2.

## 5. Định lý U2: từ belief và surrogate tới dịch vụ thật

Giữ E và một mục đích utility ψ được đánh giá. μ là phân bố thực của vị trí
ứng viên trong bài toán utility đó; b là phân bố thiết bị dùng chọn Q. Đặt
`H_ψ(a)=E_μ h_ψ(X,a)`, `G_b(a)=Σ_x b_xg_xa`. Giả sử:

ψ được quyết định trước lần randomization mới; randomness chọn A độc lập
với X và ψ khi giữ E,C. Vì thế phân phối A vẫn là p và utility thật có dạng
`Σ_a p_a H_ψ(a)`. Nếu người dùng chọn ψ sau khi nhìn A, không được giữ nguyên
phép conditioning này; phải định nghĩa và phân tích lại utility tương tác.

\[
\operatorname{TV}(\mu,b)\le\delta_b,\qquad
\sup_{x,a}|h_\psi(x,a)-g_{xa}|\le\delta_p.
\]

δ_p bao gồm sai lệch đại diện lưới, mục đích, vị trí local, catalogue và phép
xếp hạng; không mặc nhiên bằng 0. Với GPS liên tục, thay điều kiện thứ hai
bằng một ánh xạ công khai φ và `|h_ψ(X,a)−g_(φ(X),a)|≤ω_ψ(X)`; có thể dùng
δ_p=E[ω_ψ(X)|E] nếu được xác lập độc lập. Chặn TV của phân bố φ(X) với b.

Vì g nằm trong [0,1], TV chặn sai số trung bình theo g; cộng phần proxy cho:

\[
\boxed{\sup_a|H_\psi(a)-G_b(a)|\le
\delta:=\min\{1,\delta_b+\delta_p\}.}
\]

Phân phối p đã cố định theo E nên nhân p rồi cộng cho cận **utility kỳ vọng
thật**, không chỉ điểm tối ưu surrogate:

\[
\boxed{\mathbb E_{A\sim p}H_\psi(A)
\ge\max\{0,\mathbb E_pG_b(A)-\delta\}.}
\]

Định lý không cần μ là posterior của attacker; μ chỉ là phân bố mục tiêu
utility. Khi conditioning thêm thông tin local hoặc ψ, phải dùng μ và δ_b
phù hợp conditioning đó. S7 vẫn yêu cầu p không phụ thuộc ψ thật.

Để so với gói tốt nhất, đặt `F_ψ(a)=H_ψ(a)+λ(1−c_a)`. Sai lệch giữa F_ψ và
(1+λ)s_b không quá δ. Áp dụng trước và sau regret:

\[
\mathbb E_pF_\psi(A)\ge\max_aF_\psi(a)-(1+\lambda)r-2\delta.
\]

Nếu `H*=max_a H_ψ(a)` và `a*` đạt H*, suy ra

\[
\boxed{\mathbb E_pH_\psi(A)\ge
H^*-(1+\lambda)r-2\delta-\lambda(1-\mathbb E_pc_A).}
\]

Dùng `c_a*≤1` cho số hạng chi phí bảo thủ. Nếu cùng K ở mọi gói thì c=1 và
số hạng này bằng 0. **Không thay H* bằng 1 để tạo lower bound tuyệt đối**:
thư viện có thể không có gói utility=1. Đây là so sánh với optimum trong cùng
thư viện, không phải optimum toàn chuyến hay toàn bộ tập POI thành phố.

### Vì sao reference coverage có thể là utility thật?

Cho R_ψ(x) là tập POI reference đúng với mục đích ψ, U(a) là union phản hồi
sau dedup. Nếu local filter/ranking có cùng mục đích, trạng thái, metadata và
tie rule với oracle, thì

\[
\operatorname{Recall@k}(x,a)=|R_\psi(x)\cap U(a)|/|R_\psi(x)|.
\]

Mọi POI reference có trong U đều đứng trước các POI ngoài top-k theo cùng thứ
tự nên không bị loại khỏi local top-k. Reference rỗng là N/A. Đây là lemma
cho **metric đánh giá**, không yêu cầu giao diện chỉ hiển thị top5; full sorted
list có thể được trả local. Nếu utility là completeness của toàn bộ danh sách
matching, phải đổi R sang toàn bộ tập matching, không gọi Recall@5 là completeness.

Với trọng số POI reference w_ψ,x và proxy w_x đều tổng bằng 1, sai lệch
coverage của mọi U(a) không quá `TV(w_ψ,x,w_x)`. Vì vậy có cách định lượng
δ_p, thay vì đồng nhất mục đích khác nhau. Cận top-k stability theo graph có
hướng trong [`current_formal_ranking_robustness.tex`](../../thesis/current_formal_ranking_robustness.tex)
cho một điều kiện để sai lệch reference bằng 0: cùng miền khả thi, sai số điểm
có cận và khoảng tách thứ hạng đủ lớn. GPS Euclid gần không tự chứng minh các
điều kiện đó. Catalogue động, đích detour riêng và POI unavailable cần cận riêng.

## 6. Định lý U3: thư viện công khai có utility floor

Trước khi có người dùng hiện tại, chọn:

\[
\mathcal A_\rho=\{a\in\mathcal A:\min_xg_{xa}\ge\rho\}.
\]

Thư viện này dùng toàn bộ miền utility **công khai**, không prune theo Z hoặc
b. Nếu không rỗng, với mọi b, μ và mọi gói a được chọn:

\[
G_b(a)\ge\rho,\qquad H_\psi(a)\ge\max\{0,\rho-\delta_p\}.
\]

**Chứng minh:** g_xa≥ρ với mọi x nên mọi trung bình theo b hay μ đều ≥ρ; trừ
phần proxy error. Sai lệch belief δ_b không cần xuất hiện ở cận floor này.
Đây là bảo đảm từng gói theo utility đã định nghĩa, mạnh hơn bảo đảm kỳ vọng
trên A. Cận privacy P1 giữ nguyên khi tính κ lại trên A_ρ và chuẩn hóa một w
công khai dương mới. Không giữ ε cũ sau khi đổi bảng rồi nói đã chứng nhận.

Nếu proxy allowance là uniform theo từng x, cận floor cũng đúng cho từng
input x trong miền. Nếu chỉ có allowance dạng kỳ vọng của ω(X), cận là cho
H_ψ(a) điều kiện theo E; không nâng thành cam kết cho mỗi GPS riêng lẻ.

Với nhiều mục đích, có thể yêu cầu `min_(x,ψ∈Ψ_public) g_ψ(x,a)≥ρ`, với tập
ψ/prototypes công khai cố định. Chứng minh giống nhau và không gửi ψ thật.
**Audit hiện tại mới áp dụng floor nearest**, chưa có chứng nhận floor cho
fastest/radius/detour hoặc mọi radius/destination có thể. Floor lớn có thể làm
thư viện rỗng; khi đó báo không khả thi, không đổi dữ liệu hoặc bí mật hạ floor.

Nếu floor cùng transition đường làm tập gói khả thi tại một vòng rỗng, chưa
có bảo đảm chuyến đi. Mỗi vòng cần cùng thư viện khi giữ o_<t; không gán một
floor snapshot cho toàn bộ hành trình. Khi điều kiện từng vòng được chứng
minh, cận kỳ vọng trung bình theo thời gian là trung bình các lower bound,
không cần độc lập giữa các bước. Một xác suất thất bại r_t cho mỗi vòng cho
cận “mọi vòng tốt” ít nhất `1−Σ_t r_t` bằng union bound, không nhân khi chưa
có độc lập; chỉ áp dụng cho số vòng hữu hạn đã chặn công khai.

## 7. Định lý U4: điều kiện để privacy và utility cùng đạt được

Không nên hứa một lower bound utility tùy ý khi đã cố định privacy và dung
lượng phản hồi. Xét M vị trí giả thuyết đồng prior, với utility đúng tại mỗi
vị trí h_x(a)∈[0,1]. Giả sử toàn bộ kênh X→A có likelihood ratio không quá
e^ε cho mọi cặp vị trí. Khi đó posterior mỗi vị trí không quá
`p_max=e^ε/(e^ε+M−1)`. Đặt `B=max_a Σ_x h_x(a)`. Điều kiện theo gói a:

\[
\mathbb E[h_X(a)\mid A=a]
=\sum_x P(X=x\mid a)h_x(a)\le p_{\max}\sum_xh_x(a).
\]

Lấy trung bình theo A cho cận cần thiết:

\[
\boxed{\mathbb E[h_X(A)]\le
\min\left\{1,\frac{B e^\varepsilon}{e^\varepsilon+M-1}\right\}.}
\]

Ví dụ M=2 và mỗi gói chỉ phục vụ được một trong hai vùng, tức
h_0(a)+h_1(a)≤1, thì utility trung bình không vượt logistic(ε). Với ε=1,
không thể yêu cầu bảo đảm utility95% trong mô hình này: ceiling chỉ73,11%.
Đây là một điều kiện tương thích, không phải kết luận tất cả LBS có ceiling đó.

Nếu một gói có thể trả POI phù hợp cho **nhiều vị trí cùng lúc**, B lớn và
ceiling có thể bằng1. Khi ε=0, ceiling là B/M: output độc lập vị trí vẫn có
utility cao nếu candidate pool chứa kết quả của hầu hết các vị trí. Điều này
giải thích vai trò của **thu hồi POI rộng rồi lọc local** trong mô hình chúng
ta. Lớp randomization không tạo thông tin dịch vụ mới; nó chọn một pool đã
có khả năng phục vụ nhiều giả thuyết. K/L/cost và thư viện quyết định khả
năng đó. B của surrogate không tự là B của một mục đích riêng ψ; khi dùng
cận này cho mục đích đó phải dùng h_ψ thật hoặc allowance thích hợp.

## 8. Kiểm tra số và phạm vi kết luận

Mã mới: [`query_bundle_bounds.py`](../../benchmark/query_bundle_bounds.py).
Tests: [`test_query_bundle_bounds.py`](../../tests/test_query_bundle_bounds.py).
Audit: [`query_bundle_certificate_audit_20261010.py`](../../experiments/query_bundle_certificate_audit_20261010.py).
Bằng chứng: [`query_bundle_certificates_20261010_v1`](../../artifacts/benchmarks/query_bundle_certificates_20261010_v1/).

Giữ nguyên map, REM, L30, 36 ứng viên và thư viện ban đầu của audit trước.
Tính mọi REM output, không Monte Carlo. Công bố 30 trường hợp khả thi gồm
K5/K động, floor 0/0,75/0,8/0,9, β=2 hoặc calibration công khai ε_target=1,
prior đồng đều/lệch80%, và một trường hợp **client prior đồng đều nhưng truth
prior lệch80%**. Không fit attacker; Bayes optimal được tính từ channel R×T.

Với prior đồng đều, một fresh REM, β=2:

| Thư viện/kernel | Recall kỳ vọng | Mean K | Cận ε_Q | Bayes đoán đúng grid | Floor proxy thực |
|---|---:|---:|---:|---:|---:|
| K5, thư viện gốc | 90,64% | 5,000 | 0,4800 | 3,09% | 66,67% |
| K5, lọc floor công khai 0,75 | 92,51% | 5,000 | 0,2933 | 2,93% | 76,67% |
| K động, lọc floor công khai 0,75 | 93,67% | 5,981 | 0,2933 | 2,96% | 76,67% |
| K động, lọc floor công khai 0,80 | 94,81% | 7,000 | 0,2400 | 2,91% | 83,33% |

Ở cùng cận ε_target=1, calibration cho K5/floor0,75 dùng β≈6,8182 và đạt
Recall 92,66%, so với 90,85% ở K5/thư viện gốc. Cận ε là lý thuyết theo bảng,
không phải accuracy. Giá trị floor thực cao hơn ngưỡng yêu cầu do các gói
hữu hạn có giá trị coverage rời rạc. Các phép so sánh này là **development**
trên cùng map, không thay bảng trajectory benchmark hoặc là chứng minh SOTA.

K động/floor0,80 chỉ còn K7; tên mode cho phép K động không có nghĩa kết quả
thực sự dùng nhiều K. Chi phí query tăng 40% so với K5. Floor0,90 không khả
thi trong cả hai modes; K5/floor0,80 cũng không được chứng nhận bởi phép so
sánh float nghiêm ngặt. Một giá trị tính 0,799999... không được làm tròn lên
để tuyên bố đạt floor0,80. Không có directed-rounding certificate cho bảng float.

Ở client prior sai, K5/floor0,75/calibration ε=1 có Recall thực 98,49%, trong
khi lower bound có allowance TV là 86,72%. TV trung bình theo E là 0,1948.
Đây là kiểm tra bridge với một mismatch đã khai báo, không đo calibration
của bộ lọc mobility production. Privacy không cần client prior khớp truth;
utility certificate nói rõ điều kiện và phần sai số.

Kết quả đủ để giữ hướng **thư viện công khai + kernel xác suất + chứng chỉ
coverage/cost/privacy**, thay vì tăng K vì lý do ẩn danh. Trước khi đưa vào
mô hình chính, còn cần transition có hướng qua nhiều bước, floor cho mục đích
khác, grid/sensor allowance, fresh moving-family evaluation và sampler hữu
hạn được kiểm chứng. Các điều kiện đó không được giả định đã thỏa chỉ vì
snapshot hiện tại cho số tốt.

## 9. Lời trình bày trước hội đồng

“Geo-I bảo vệ các lần đọc vị trí. Lớp truy vấn của chúng tôi chọn ngẫu nhiên
một gói Q từ thư viện công khai, với điểm cân bằng độ phủ POI và chi phí.
Chúng tôi chứng minh cận likelihood của gói từ độ biến thiên tương đối của
điểm, và cộng dồn cận đó theo những lần chọn mới. Về utility, điểm độ phủ là
một kỳ vọng có định nghĩa rõ; sai lệch belief và surrogate được đưa vào cận,
không giả sử bằng 0. Đặc biệt, nếu lọc thư viện bằng coverage floor công khai,
ta có bảo đảm độ phủ từng gói trên miền đã chứng nhận mà vẫn giữ privacy.
Kiểm tra hiện tại cho thấy K5 có thể tăng utility và giảm mức suy luận nhờ
chọn thư viện tốt hơn. Phần bảo đảm ngoài lưới và trên toàn chuyến đi đang
được đánh giá tiếp, nên chúng tôi tách rõ định lý, giả thiết và benchmark.”
