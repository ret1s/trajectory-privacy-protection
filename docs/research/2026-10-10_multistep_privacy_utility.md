# Geo-I và lựa chọn đoạn Q: chứng minh cho cả phiên

Bản này nối chứng minh snapshot với prototype chạy nhiều bước. Geo-I/REM,
noisy reuse và cap GPS **Cs = 0,23/m mỗi phiên** giữ nguyên. Cận chọn Q là
một lớp bổ sung; chưa thay engine chính. Không có cap số phiên.

## 1. Kênh quan sát và các giả thiết bắt buộc

Ký hiệu X là quỹ đạo GPS; E là lịch sử đã bảo vệ gồm noisy anchor và trạng
thái nội bộ đã được cơ chế Geo-I bảo vệ; b(E) là belief bất kỳ. C là bản đồ,
POI, cấu hình, lịch gửi và ranh giới phiên công khai. O là toàn bộ transcript
server thấy. Nhu cầu ψ và GPS dùng để sắp xếp POI chỉ tồn tại tại local.

Chuỗi nhân quả:

\[
X\longrightarrow E\longrightarrow A_j=(Q_{j,0},Q_{j,1},Q_{j,2})
\longrightarrow O\longrightarrow\text{phản hồi/cache}.
\]

Tại mỗi lịch sử công khai o trước đó, thư viện A(C,o), trọng số nền w,
bảng điểm g, floor và ngân sách phải giống nhau cho mọi E. Tất cả hành động
được giữ có w>0. Các RNG là randomness riêng, độc lập với X; seed công khai
trong test không phải cấu hình triển khai an toàn. Server có thể thích nghi,
nhưng phản hồi/retry chỉ được dùng C, O và trạng thái server công khai.

Bảo đảm điều kiện trên lịch kích hoạt/kết thúc đang quan sát được; không bảo
vệ account, IP, click gửi thêm, nhật ký debug, hoặc thời điểm mở/đóng ứng dụng.
Không suy luận rằng bảo vệ tọa độ đầu/cuối S9/S10 đồng nghĩa giấu activation.

## 2. P1 — Cận một lần chọn cả đoạn Q

Mỗi A gồm ba khung K=5, cách nhau 20 giây. Từ khung đã công bố trước đó,
mỗi track đi theo đường có hướng với thời gian không quá 20 giây. Hành động
đứng yên luôn nằm trong thư viện trước khi lọc floor. Thư viện được dựng từ
các goal công khai; không dùng GPS, Z, b hoặc ψ để chọn support.

Đặt g(x,a) bằng trung bình điểm POI của ba khung, với bốn purpose có trọng
số cố định. Mỗi bảng được xây từ reference POI công khai. Điểm trường hợp
không có reference là 0 **cho mục tiêu availability của planner**, với validity
mask riêng; Recall thực nghiệm của trường hợp đó vẫn là N/A. Không điều kiện
lại trọng số purpose theo b. Vì vậy g thuộc [0,1] và tuyến tính theo belief.

\[
s_b(a)=\frac{\sum_x b_xg(x,a)}{1+\lambda},\qquad
T(a\mid b)=\frac{w_a\exp(\beta s_b(a)/2)}{\sum_cw_c\exp(\beta s_b(c)/2)}.
\]

K cố định nên term tiết kiệm cardinality của kernel cũ bằng 0. Giữ λ=0,25
để đối chiếu cùng công thức. Định nghĩa:

\[
\kappa=\frac{1}{1+\lambda}\max_{a,c}
\left(\max_x[g(x,a)-g(x,c)]-\min_x[g(x,a)-g(x,c)]\right).
\]

Với d_a=s_b(a)-s_{b'}(a), range_a(d_a)≤κ: d_a-d_c là hiệu hai trung bình
của cùng dãy g(x,a)-g(x,c). Viết lại tỷ số normalizer như trung bình của
exp(-βd_c/2) cho thấy:

\[
\log\frac{T(a\mid b)}{T(a\mid b')}\le\frac\beta2\left(d_a-\min_cd_c\right)
\le\frac\beta2\kappa=\varepsilon_Q.
\]

Không yêu cầu b là posterior thật. Tính một κ_upper công khai và đặt
β≤2ε_allowed/κ_upper. **Một lần lấy mẫu bảo vệ cả đoạn**, không nhân ε_Q
thêm ba lần chỉ vì gửi ba khung của đoạn đã lấy mẫu.

Nếu κ=0, mọi hiệu điểm giữa hai hành động độc lập b. T độc lập b với mọi β;
β=40 cải thiện lựa chọn utility công khai mà không tiêu privacy. Không coi
một κ rất nhỏ là 0. Cận ε_Q là cận trên, không phải privacy loss tối thiểu.

## 3. P2 — Composition thích nghi và ngân sách cả phiên

Giữ o trước đó cố định. Các conditional kernel dưới mọi E có cùng support
và tỷ số max/min không vượt exp(ε_Q,j). Phân phối dưới X và X' là các hỗn hợp
khác nhau của các giá trị ấy, nên cũng có cùng cận. Nhân các conditional
likelihood, **không giả sử E hay các vòng độc lập**, được:

\[
P(O=o\mid X,C)\le\exp\left(\sum_j\varepsilon_{Q,j}\right)P(O=o\mid X',C).
\]

Chọn trước Γ_Q, rồi dùng lịch công khai:

\[
\varepsilon_{Q,j}=\frac{\Gamma_Q}{j(j+1)},\qquad
\sum_{j=1}^{n}\varepsilon_{Q,j}=\Gamma_Q\frac n{n+1}\le\Gamma_Q.
\]

Điều này đúng với mọi độ dài phiên. Mốc chọn đoạn là 0,60,120,... giây,
không phụ thuộc fresh/reuse của Geo-I. Arm chọn từng bước chia allowance của
mỗi block thành ba phần bằng nhau để so sánh ở **cùng ngân sách phiên**.

SQLite ghi atomic lựa chọn, số thứ tự, allowance đã reserve và các khung
trước khi công bố. Khởi động lại dùng khung lưu, retry không lấy mẫu mới.
Token/start/context/Γ không được đổi trong cùng file. File/key nằm ở local.
Xóa hoặc rollback ledger làm mất bảo đảm vận hành; không có cách tự chứng
minh chống rollback chỉ bằng một file SQLite trên thiết bị không đáng tin.

Nếu upstream Geo-I thực sự bảo đảm Cs D∞(X,X') cho E và mọi kênh O đi qua
lớp trên, hậu xử lý và P2 đồng thời cho:

\[
\alpha(X,X')\le\min\{C_sD_\infty(X,X'),\Gamma_Q\}.
\]

Đây là **hai cận của cùng đầu ra**, không phải cộng Cs với Γ_Q. Cs có đơn vị
1/m; Γ_Q không có đơn vị. Phiên liên kết vẫn composition: tổng các Cs D∞
và tổng Γ_Q; không đặt lại tổng lịch sử thành 1 sau mỗi chuyến.

## 4. P3 — Cận suy luận cho người, phương tiện và tương lai

Với nhãn Y (người, phương tiện vật lý, hướng đi tiếp theo), điều kiện trên
cùng C và side information S. Nếu không có kênh lộ Y trực tiếp và ảnh hưởng
Y đến O chỉ đi qua E, các hỗn hợp cũng thỏa likelihood ratio exp(Γ_Q).
Do Bayes, với hai giả thuyết có prior dương:

\[
\frac{P(Y=y\mid O,C,S)}{P(Y=y'\mid O,C,S)}
\le e^{\Gamma_Q}\frac{P(Y=y\mid C,S)}{P(Y=y'\mid C,S)}.
\]

Với prior π_y, posterior một ứng viên bị chặn bởi:

\[
P(Y=y\mid O,C,S)\le\frac{e^{\Gamma_Q}\pi_y}{e^{\Gamma_Q}\pi_y+1-\pi_y}.
\]

Ví dụ Γ_Q=1: 10 ứng viên đồng đều có cận posterior 23,20%; prior đã là 80%
thì cận vẫn khoảng 91,58%. Đây là hạn chế **mức tăng suy luận**, không phải
chứng minh attacker không đoán được identity. Các nhóm có context/public
identifier khác nhau không tự thỏa định lý này. S4–S6 là hệ quả có điều kiện,
không tuyên bố đạt k-anonymity hoặc unlinkability của phương tiện.

## 5. U1 — Lấy POI rộng, sắp xếp đầy đủ tại local

Gọi R_ψ(x) là reference top-5 theo purpose ψ và tie-break lexical, U là union
POI đã nhận và còn hiệu lực. Client sắp xếp **toàn bộ** POI trong U thỏa ψ.
Một phần tử thuộc R_ψ(x)∩U luôn nằm trong top-5 của danh sách đó: không thể
có hơn bốn phần tử đứng trước nó trong toàn bộ catalogue. Vì vậy Recall@5
của local ranking đúng bằng |R_ψ(x)∩U|/|R_ψ(x)| khi reference không rỗng.
Reference rỗng là N/A. k=5 chỉ thuộc metric; giao diện không cắt câu trả lời.

Thêm POI hợp lệ vào cache không thể làm giảm intersection. Kết quả chỉ áp
dụng trong version/epoch mà server bảo đảm còn hợp lệ; không áp dụng cho
POI đã đóng hoặc provider thay catalogue ngoài hợp đồng hiện tại. L30 là
độ sâu **mỗi category mỗi Q**; 150 là tối đa mỗi category trước dedup ở K5,
không phải tổng tất cả category.

## 6. U2 — Floor qua nhiều vị trí và bridge sang utility thật

Để tránh lấy floor của một vị trí giả định cố định qua ba khung, dùng:

\[
f(a)=\min_{\tau\in\{0,1,2\},\,x\in\mathcal X_{public}}
\frac14\sum_{p=1}^{4}g_p(x,Q_{a,\tau}).
\]

Giữ a khi f(a)≥ρ. Với **bất kỳ chuỗi vị trí** trong miền public được kiểm tra,
điểm composite ở từng khung vẫn ≥ρ. Đây không bảo đảm từng purpose riêng,
cũng không bảo đảm cho radius/destination ngoài prototype. Không suy diễn
floor trên 36 state đại diện thành cận cho cả mặt phẳng GPS hoặc toàn mạng.
Nếu thư viện khả thi không đạt ρ, giữ thư viện gốc và báo floor thực; không
refetch bí mật, tăng budget hoặc sửa raw sample.

Cho mỗi khung, μ_τ là phân phối vị trí thật điều kiện E; b là belief đầu đoạn.
Giả sử TV(μ_τ,b)≤δ_b,τ và bảng điểm với utility thật sai không quá δ_p,τ
cho mọi hành động. Vị trí/ψ được xác định trước randomness của đoạn, hoặc
các allowances phải đúng cả khi điều kiện theo hành động. Với utility [0,1]:

\[
\mathbb E[H_\tau\mid E]\ge
\mathbb E_{A\mid E}\sum_x b_xg(x,Q_{A,\tau})-\delta_{b,\tau}-\delta_{p,\tau}.
\]

Điều này theo |E_μg-E_bg|≤TV(μ,b), rồi trừ sai số proxy. Trung bình ba khung
và dùng tower property cho cả phiên cho trung bình các cận, không cần IID.
Sai số belief còn gồm **di chuyển sau lúc chọn đoạn**, không chỉ sai số GPS.
Giá trị actual_expected_utility_lower để null khi chưa có đủ allowances.

Có ba mức rõ ràng: `public_table_only`, `conditional_actual_bound` với
allowances đã khai báo, và `unverified_actual_utility`. Benchmark đo actual
Recall riêng; điểm proxy tốt không tự chứng nhận calibration của filter.

Cận regret Gibbs/KL trong bản chứng minh snapshot tiếp tục áp dụng cho
thư viện đoạn và g trung bình: so với thư viện khả thi công khai hiện tại,
không so với optimum của mọi quỹ đạo tương lai. Ở K cố định, utility score
regret được nhân (1+λ); bridge thêm sai số đúng phạm vi đã khai báo.

## 7. Giới hạn cần trình bày cùng định lý

Nếu κ_j bị chặn dưới bởi hằng số dương, allowance giảm làm β_j tiến về 0:
khả năng thích nghi theo belief giảm, phân phối tiến về trọng số nền. Nếu
κ_j=0, vẫn tối ưu utility công khai miễn phí. Không thể vừa cam kết budget
hữu hạn vừa cam kết độ chính xác cao trong mọi miền và mọi phiên dài.

Cận cần thiết đã chứng minh trước: với M giả thuyết đồng đều và mỗi gói chỉ
phục vụ tổng B đơn vị utility trên M giả thuyết, utility trung bình không quá
min(1,B e^Γ/(e^Γ+M-1)). Với M=2,B=1,Γ=1 ceiling là 73,11%. Pool POI phục
vụ đồng thời nhiều vị trí có thể tăng B; tăng randomization không tạo POI mới.
Không khẳng định prototype đạt utility gate trước khi đọc kết quả cohort mới.

κ_upper dùng rational/outward allowance cho **bảng binary-float được coi là
số thực chính xác**. Nó chưa chứng nhận sai số tạo bảng, logsumexp, RNG hoặc
REM. Nếu sampler thực gần kernel trong TV≤ζ thì một vòng có approximate-DP
δ≤(1+e^ε)ζ; ζ chưa được chứng minh cho NumPy/PCG64 hiện tại. Tests likelihood
pass là kiểm tra số, không thay chứng nhận sampler.

## 8. Định lý → giả thiết → mã bảo đảm

| Kết quả | Giả thiết chính | Cơ chế/mã và kiểm tra |
|---|---|---|
| P1: một đoạn | cùng support, g tuyến tính bounded, trọng số public | `PublicSegmentLibrary`, `FixedPublicPurposeTable`, `SegmentPqbPolicy`; finite oracle |
| P1: calibration | κ_upper không thấp hơn κ | `oscillation_upper` và làm tròn β xuống; rational oracle, không tolerance zero |
| P2: cả phiên | lịch public, mọi đường đi tổng ≤Γ | `PersistentQueryBudget`, Fraction allowance, transaction trước publish; test300frames |
| P2: không refill | cùng policy/file, state/key giữ local | immutable policy, cached frames và retry; restart/reorder tests |
| Geo-I hậu xử lý | prototype chỉ nhận E/b; upstream giữ cap | chung một `PacedSlackProgressLaneDummy` trong study; Cs.23/m; không extra GPS |
| directed feasibility | Q nối từ khung đã gửi | shortest directed travel≤20s và hold; kiểm tra state thật, không resnap lane trùng tọa độ |
| ψ không có trong request | ψ/GPS local, không điều khiển support/cadence | `SegmentCoverClient` + request-schema equality với hai request/GPS khác nhau |
| U1: ranking | cùng catalogue/objective/tie/validity | `local_sorted_pois`; full sorted answer, metric-only top5 |
| U2: floor | tất cả τ,x trong miền public | `floor_table` min frame/state; báo degraded nếu infeasible |
| U2: utility thật | allowances TV/proxy đã thiết lập | `actual_utility_bridge`: mặc định None; không tự δ=0 |
| P3: identity/forecast | common side information, không explicit ID | posterior odds corollary; không tuyên bố anonymous/không thể suy ra |

## 9. Nguồn và bằng chứng

Khái niệm Geo-I, distance-dependent likelihood bound và hậu xử lý bắt nguồn
trong [Andrés et al., Geo-Indistinguishability](https://arxiv.org/html/1212.1984v3).
Noisy reuse và quản lý budget cho mobility traces là background đã có trong
[Chatzikokolakis et al., Predictive DP](https://arxiv.org/html/1311.4008v2).
Lịch telescoping, đoạn Q và các bridge ở đây là suy luận cho prototype này;
không gọi chúng là định lý hay thuật toán được sao chép nguyên từ hai paper.

Chứng minh snapshot/regret/compatibility: [bản trước](2026-10-10_query_bundle_privacy_utility.md).
Protocol và kết quả mới: [query_segments_20261010_v2](../../artifacts/benchmarks/query_segments_20261010_v2/).
Cohort native mới: [query_segment_fresh_native_20261010_v2](../../artifacts/datasets/query_segment_fresh_native_20261010_v2/).

Hợp đồng dịch vụ của prototype là point-query có độ sâu hữu hạn L30. Trong
catalogue tĩnh nhỏ này, nếu provider cho phép bulk download toàn bộ catalogue
còn hiệu lực thì tải một lần và lọc local có thể tránh truy vấn tọa độ hoàn
toàn; đây là control đã ghi ở luận văn, không bị thay thế bởi PQB. Do đó kết
quả mới không được dùng để khẳng định PQB tối ưu cho mọi hợp đồng LBS.

Ở prefix kết thúc 600s, joint reserve toàn allowance của block11 còn step
mới tiêu một phần ba của block11. Hai arm có cùng cap Γ và cùng allocation
ở mỗi block hoàn tất; không tuyên bố mọi prefix dở block có chi phí bằng nhau.

Geo-I vẫn có vai trò riêng trong cận min: Cs·D∞ tiến về0 khi hai quỹ đạo tiến
lại gần nhau, còn Γ_Q là một cap không phụ thuộc khoảng cách. Lớp chọn Q tự
cho cận Γ_Q không thay thế bảo đảm Geo-I theo khoảng cách; giữ backbone tạo
cận chặt hơn ở cả vùng gần và vùng xa, đồng thời cung cấp lịch sử đã bảo vệ
cho bộ ước lượng utility.

Sau khi hoàn tất benchmark, `calibrated_beta` của helper snapshot cũ được
siết thêm bằng binary rationals cho cả κ và phép làm tròn β. Cách này tránh
trường hợp phép trừ float xóa một κ rất nhỏ rồi chọn β40 với budget0. Kernel
đoạn đã có kiểm tra exact-zero/outward riêng từ lúc đóng băng. Replay nguồn
đóng băng giữ provenance của số liệu; thay đổi helper không được gọi trong
study được ghi riêng, không sửa kernel, dữ liệu hay kết quả.
