# Cơ sở toán học của GeoI-BQ: riêng tư cả phiên và chất lượng dịch vụ

GeoI-BQ kết hợp hai mức bảo vệ: Geo-I kiểm soát việc tiết lộ từ GPS sang
thông tin đã bảo vệ; kernel Q kiểm soát việc tiết lộ từ thông tin đó sang
các truy vấn công khai trong cả phiên. Việc lấy POI dùng một giao thức chung,
còn nhu cầu thật và việc sắp xếp kết quả được xử lý tại thiết bị.

Bản này thống nhất các chứng minh thành một chuỗi **cơ chế → transcript →
suy luận → utility**, để trình bày trước benchmark. Kết quả lý thuyết áp
dụng cho kernel được đặc tả dưới đây; benchmark là kiểm chứng chất lượng
dịch vụ của cấu hình [GeoI-BQ v1](2026-10-14_main_model.md).
Không thay cấu hình, dữ liệu hoặc các kết quả đã đóng băng.

## 1. Mô hình, ký hiệu và giả thiết

Một **phiên** là một lần chạy với cùng session token, ledger và lịch công
khai. Khởi động lại tiếp tục phiên đó. Ngân sách của phiên này không được
hiểu là ngân sách chung cho mọi phiên của một người dùng.

| Ký hiệu | Ý nghĩa |
|---|---|
| \(C\) | Ngữ cảnh công khai cố định: bản đồ, catalogue, cấu hình, lịch và provider law |
| \(X=(x_0,x_1,\ldots)\) | GPS ở những mốc được đưa vào cơ chế bảo vệ |
| \(E\) | Lịch sử nội bộ đã bảo vệ: nhánh noisy test và các neo REM \(Z\) |
| \(B_j\) | Belief được tính từ lịch sử đã bảo vệ; không cần là posterior đúng |
| \(H_{j-1}\) | Lịch sử phân tích gồm các đoạn Q đã chọn, nhãn lane giả và phản hồi trước bước \(j\) |
| \(\mathcal A_j(H_{j-1})\) | Thư viện các đoạn Q khả thi trên đường ở bước \(j\) |
| \(w_{j,a}>0\) | Xác suất nền công khai trên thư viện, \(\sum_a w_{j,a}=1\) |
| \(z_{j,a}\in[0,1]\) | Điểm số riêng đã lượng tử hóa cho đoạn \(a\) |
| \(O_n\) | Transcript máy chủ: Q có thứ tự, các mốc gửi, phản hồi và retry logic |
| \(U_t\) | Tập metadata POI đã nhận và còn hợp lệ ở thời điểm \(t\) |
| \(\psi_t\) | Nhu cầu thật ở local: purpose, category, radius hoặc destination |
| \(u\), \(C_s\) | Ngân sách một đơn vị và cap Geo-I của phiên, đơn vị \(\mathrm m^{-1}\) |
| \(\varepsilon_j\), \(\Gamma\) | Allowance bước Q và cap Q cả phiên, không có đơn vị |

Các định lý dùng sáu giả thiết:

1. **Support chung.** Khi giữ cùng \(C,H_{j-1}\), thư viện, thứ tự hành động
   và masses chỉ phụ thuộc bản đồ, các Q đã chọn và phản hồi công khai.
   GPS, Z, belief và nhu cầu không loại ứng viên. Có đoạn đứng yên khả thi.
2. **Geo-I lý tưởng.** REM và noisy test dùng các kernel xác suất được đặc
   tả ở Mệnh đề 1. Belief và Q chỉ đọc thông tin đã bảo vệ từ upstream.
3. **Kernel Q bị chặn.** Mọi điểm số vào kernel nằm trong \([0,1]\);
   lấy mẫu PMF hữu hạn bằng bit đều độc lập. Privacy không cần độ chính xác
   của phép tính belief/điểm số trước lượng tử hóa.
4. **Ledger và clock.** Commit trước khi gửi, không rollback/reset allowance,
   retry dùng lại Q đã chọn. Khung thứ hai và thứ ba của đoạn không lấy mẫu lại.
5. **Transcript chung.** Tại cùng Q/history, provider có cùng conditional law.
   Request thật, GPS local, belief, nhánh fresh/reuse và diagnostics không
   đi vào bản tin hoặc quyết định gửi riêng. Clock/mở/đóng phiên được giữ cùng
   nhau khi so sánh, hoặc dừng theo một quy tắc chỉ dùng transcript công khai.
6. **Utility.** Catalogue xét trong chứng minh cache có metadata ổn định còn
   hợp lệ. Lọc và sắp xếp dùng cùng GPS/nhu cầu tại thời điểm đánh giá, cùng
   quy tắc tính điểm và xử lý tie. Utility được chuẩn hóa trong \([0,1]\).

Các giả thiết này xác định bài toán toán học. Chứng chỉ integer của PMF Q
và invariant ledger là các bảo đảm của mã; sampler Geo-I dùng float và thời
gian thực thi trên phần cứng không được suy ra từ định lý kernel lý tưởng.

## 2. Mệnh đề 1 — Geo-I cho lịch sử GPS thích nghi

Với tập trạng thái đường công khai hữu hạn \(V\), tọa độ \(a_v\), REM là

\[
R_u(v\mid x)=\frac{e^{-u\|x-a_v\|/2}}{\sum_{w\in V}e^{-u\|x-a_w\|/2}}.
\]

Đặt \(d=\|x-x'\|\). Bất đẳng thức tam giác chặn tỷ số của mỗi số hạng
giữa \(e^{-ud/2}\) và \(e^{ud/2}\). Cộng theo cùng \(V\) cho cùng cận
đối với normalizer. Do đó

\[
e^{-ud}\le \frac{R_u(v\mid x)}{R_u(v\mid x')}\le e^{ud}.
\tag{1}
\]

Tại cùng lịch sử đã bảo vệ, tham chiếu \(h\) giống nhau. Hàm
\(g_h(x)=\|x-h\|\) là 1-Lipschitz. Test
\(g_h(x)+\operatorname{Lap}(1/u)\le\theta\) vì vậy thỏa \(u\)-Geo-I
cho **cả giữ lẫn làm mới**. Gọi xác suất giữ là \(q_x\). Kernel chung là

\[
K_x(\mathrm{reuse})=q_x,
\qquad K_x(\mathrm{fresh},v)=(1-q_x)R_u(v\mid x).
\tag{2}
\]

Lần đầu chỉ có REM: một đơn vị. Nhánh reuse: một đơn vị test. Nhánh fresh:
hai đơn vị test và REM. Không đọc GPS: không thêm chi phí. Phải tính xác
suất nhánh trong (2), không chỉ đếm số lần tạo Z.

Cho \(d_i=\|x_i-x'_i\|\), cap integer \(M\), với lần đầu bắt buộc tốn
một đơn vị. Prospective filter bảo đảm \(c_0=1\), \(c_i\in\{0,1,2\}\)
và \(\sum_i c_i\le M\) trên mọi lịch sử. Chain rule tại cùng lịch sử mở
rộng cho \(\log LR\le u\sum_i c_i d_i\), từ đó

\[
\boxed{G(X,X')=u\left[d_0+
\operatorname{TopSum}_{M-1}(d_1,d_1,d_2,d_2,\ldots)\right]
\le C_sD_\infty(X,X'),\quad C_s=uM.}
\tag{3}
\]

TopSum lấy các phần tử lớn nhất, tối đa số lượng đang có. Đây là cận trên
của mọi phân bổ chi phí khả thi, không cần các GPS độc lập. Tại cùng
extended history, quyết định pacing/cap giống nhau, nên các event không đọc
GPS có kernel giống nhau. Cộng các history tương thích bảo toàn cận.

Ở cấu hình chính, \(u=0{,}01\), \(M=23\), \(C_s=0{,}23\). Nếu chỉ GPS
đầu khác 10 m, \(G\le0{,}1\); nếu chỉ một GPS sau khác 10 m,
\(G\le0{,}2\). Con số 0,23 là cap **cả phiên**, không phải epsilon cho
từng fix. Nguyên lý Geo-I và private prediction được kế thừa từ
[Andrés et al.](https://arxiv.org/abs/1212.1984) và
[Chatzikokolakis et al.](https://petsymposium.org/2014/papers/Chatzikokolakis.pdf).

## 3. Mệnh đề 2 — Kernel Q chuẩn hóa, support chung và lấy mẫu integer

Tại cùng \(H_{j-1}\), bỏ chỉ số \(j\) để viết gọn. Đặt

\[
\mu=\sum_aw_az_a,\qquad
\alpha=\frac{\varepsilon}{2+\varepsilon},\qquad
p_a=w_a[1+\alpha(z_a-\mu)].
\tag{4}
\]

Vì \(\sum_aw_a(z_a-\mu)=0\), ta có \(\sum_ap_a=1\). Vì
\(z_a,\mu\in[0,1]\) và \(\alpha<1\),
\(w_a(1-\alpha)\le p_a\le w_a(1+\alpha)\), nên mọi ứng viên vẫn có
xác suất dương. Với **mọi** hai private score vectors,

\[
\boxed{\frac{p_a}{p'_a}\le
\frac{1+\alpha}{1-\alpha}=1+\varepsilon.}
\tag{5}
\]

Kể cả khi private belief còn ngẫu nhiên sau khi điều kiện theo \(H\),
kernel marginal vẫn có dạng (4): thay \(z_a\) bằng
\(\bar z_a=\mathbb E[z_a\mid H,X]\in[0,1]\). Vì \(w\) không đổi ở
cùng \(H\), kỳ vọng đi qua phép centered tilt. Hai input có thể cho hai
phân phối belief hoàn toàn khác nhau; (5) vẫn đúng.

**Đúng PMF bằng số nguyên.** Viết \(w_a=m_a/T\), \(T=\sum_am_a\),
\(z_a=F_a/S\), \(\alpha=N/D\). Đặt

\[
W_a=m_a\left[DTS+N\left(TF_a-\sum_cm_cF_c\right)\right],
\quad \sum_aW_a=DT^2S.
\tag{6}
\]

Mọi \(W_a\) là integer dương. Lấy một integer đều trong
\(\{0,\ldots,DT^2S-1\}\), chọn cumulative interval dài \(W_a\), cho
đúng xác suất \(W_a/(DT^2S)=p_a\). Tổng số nguyên không phụ thuộc private
scores. Phép chọn **cả đoạn ba khung**, gồm thứ tự năm Q, là một output;
gửi riêng từng khung sau đó là hậu xử lý của output đó.

Điểm raw hiện tại được chuẩn hóa min–max trước khi lượng tử hóa. Vì vậy
không mặc định \(|z_a-v_a|\le1/S\) với raw score \(v\). Privacy chỉ dùng
\(z\in[0,1]\); utility sẽ dùng covariance thực sự ở Mệnh đề 7.

## 4. Định lý 1 — Privacy end-to-end của cả phiên

Phân bổ công khai \(\varepsilon_j=\Gamma/[j(j+1)]\) cho lần chọn đoạn
thứ \(j\). Vì \(1/[j(j+1)]=1/j-1/(j+1)\),

\[
\sum_{j=1}^n\varepsilon_j=\Gamma\frac{n}{n+1}\le\Gamma.
\tag{7}
\]

Trong mã, allowance là số hữu tỷ; tham số kernel được chuyển về một
binary64 value làm tròn xuống rồi lấy lại đúng Fraction, nên
\(\widetilde\varepsilon_j\le\varepsilon_j\). Mệnh đề 2 dùng tham số
kernel thực sự; các cận cả phiên dùng allowance lớn hơn và vẫn đúng.
Đặt \(\alpha_j=\varepsilon_j/(2+\varepsilon_j)\),
\(\widetilde\alpha_j=\widetilde\varepsilon_j/(2+\widetilde\varepsilon_j)
\le\alpha_j\). Không giả định binary64 lưu đúng allowance.

Ở cùng augmented history, các conditional kernels thỏa (5). Chain rule
cho \(R_n=\prod_{j=1}^n(1+\varepsilon_j)\le e^\Gamma\).
Provider responses có cùng conditional law nên không thêm factor.
Cache là hàm của các replies; retry lặp bản tin đã commit; nhãn lane giả
và các khung chưa gửi chỉ được bỏ đi khi chiếu xuống wire. Tích phân/cộng
trên các history cùng tạo một wire cũng bảo toàn tỷ số.

Mặt khác, cả Q process là hậu xử lý của \(E\) và public context; (3) cũng
chặn tỷ số wire. Hai cận cùng đúng cho **cùng output event**, nên lấy cận
nhỏ hơn, không cộng ngân sách khác đơn vị:

\[
\boxed{\Pr[O_n\in A\mid X,C]
\le e^{\min\{G(X,X'),\log R_n\}}
\Pr[O_n\in A\mid X',C].}
\tag{8}
\]

Hoán đổi \(X,X'\) cho cận ngược. Không cần suy ngược Z trước khi áp dụng
(8). Công thức xác suất đúng là marginal hóa **toàn bộ** protected history:

\[
\Pr[O\in A\mid X,C]
=\int \Pr[O\in A\mid E=e,C]\,d\Pr[E=e\mid X,C].
\tag{9}
\]

Cho một snapshot đơn giản, (9) trở thành
\(\sum_z\Pr[Q=q\mid Z=z,C]\Pr[Z=z\mid X,C]\). Không được nhân hai
"xác suất suy ra đúng" để kết luận privacy, vì Z là biến ẩn và attacker
có thể suy luận trực tiếp từ toàn transcript.

### Phiên dài và quy tắc dừng công khai

Sau cutoff \(J\), \(\sum_{j>J}\varepsilon_j=\Gamma/(J+1)=s<1\).
Khai triển tích, mỗi elementary symmetric sum không vượt \(s^k\), nên

\[
R_\infty\le\frac{\prod_{j=1}^J(1+\varepsilon_j)}{1-\Gamma/(J+1)}.
\tag{10}
\]

Với \(\Gamma=1,J=11\), cận hữu tỷ là
\(10763282027953/4415938560000<2{,}437372\).
Giá trị 2,437371 trong bản trước là hiển thị xấp xỉ; khi ghi một **cận
thập phân**, phải làm tròn lên thành 2,437372 hoặc giữ phân số chính xác.
Các prefix distributions xác định một law trên sequence space. Bất đẳng
thức cùng \(R_\infty\) đúng trên mọi finite-prefix event; mở rộng qua
monotone class cho mọi event của cả sequence. Vì vậy một thời điểm dừng
được chọn từ transcript công khai, kể cả dừng khi attacker thấy một mẫu
"dễ đoán", là hậu xử lý của sequence và giữ cận toàn phiên này.

Quy tắc chi phí dự toán là một privacy filter; cách quản lý này liên quan
đến [Rogers et al.](https://arxiv.org/abs/1605.08294). Lịch Q ở đây được
ấn định trước, nên (7)–(10) được chứng minh trực tiếp bằng tích kernel.
Ledger bền vững làm invariant này có nghĩa qua restart.

## 5. Định lý 2 — Giới hạn suy luận của mọi attacker

### 5.1 Posterior odds và posterior của một mục tiêu

Gọi \(Y\) là identity, planned next step hoặc planned destination. Tại
cùng \(C\), target đi vào wire qua các private inputs của cơ chế; không
có một định danh trực tiếp bổ sung trong observation. Với một sự kiện
target \(A\), prior \(\pi=\Pr[Y\in A\mid C]\in(0,1)\), hai
conditional laws là các hỗn hợp của các trajectory laws. Từ cận Q giữa
mọi trajectory pair, tích phân theo hai phân phối mixing bất kỳ cho

\[
\frac1{R_n}\le
\frac{\Pr[O=o\mid Y\in A,C]}{\Pr[O=o\mid Y\notin A,C]}
\le R_n.
\tag{11}
\]

Bayes cho posterior odds bằng prior odds nhân likelihood ratio. Do đó

\[
\boxed{\frac{\pi}{\pi+R_n(1-\pi)}
\le \Pr[Y\in A\mid O=o,C]
\le \frac{R_n\pi}{1-\pi+R_n\pi}.}
\tag{12}
\]

Cận áp cho identity **người và phương tiện** nếu observation chỉ có kênh
trajectory đã mô hình hóa. Khi mọi pair giữa hai target classes còn thỏa
\(G(X,X')\le g_A\), có thể thay \(R_n\) bằng
\(\min\{e^{g_A},R_n\}\). Đây là vai trò bổ sung của Geo-I: cận gần
theo khoảng cách; Q thêm cận toàn cục ngay cả giữa các trajectory xa nhau.

Ở \(\Gamma=1\), prior 50/50 có posterior từng wire không quá khoảng
70,91%. Đây là cận của **mỗi observation**, không phải accuracy trung bình.
Với prior 0 hoặc 1, posterior tương ứng vẫn 0 hoặc 1.

### 5.2 Accuracy trung bình và lượng thông tin

Gọi \(W\) là law của **cùng public-base planner** khi \(\Gamma=0\).
Tại cùng history, \(W\) dùng \(w\); nó không đọc private scores và độc
lập với target riêng được mô hình hóa ở trên. Từ (4),

\[
\chi^2(p_j\Vert w_j\mid H)
=\widetilde\alpha_j^2\operatorname{Var}_{w_j}(\bar z)
\le\alpha_j^2/4.
\tag{13}
\]

Đặt \(L_n=dP_n/dW_n\). Tower property, không giả sử độc lập, cho
\(\mathbb E_W[L_n^2]\le\prod_{j\le n}(1+\alpha_j^2/4)\). Bởi
\(TV=\tfrac12\mathbb E_W|L_n-1|\), Cauchy–Schwarz cho

\[
\boxed{TV(P_n,W_n)\le\delta_n
:=\frac12\sqrt{\prod_{j\le n}(1+\alpha_j^2/4)-1}.}
\tag{14}
\]

Với mọi prior \(\pi_y\), mỗi law \(P_{O\mid Y=y,C}\) và các hỗn hợp
private states thỏa cùng (14). Vì
\(TV(P_{Y,O},\pi\otimes W)=\sum_y\pi_yTV(P_{O\mid y},W)\),
với mọi attacker và reward trong \([0,1]\),

\[
\boxed{\mathrm{Success}^{*}_{\mathrm{after}}
\le\mathrm{Success}^{*}_{\mathrm{prior}}+\delta_n.}
\tag{15}
\]

Dưới \(\pi\otimes W\), quan sát độc lập với Y nên không tốt hơn best
prior strategy. Với classification, prior success là \(\max_y\pi_y\).
Với Hit100, nó là prior mass lớn nhất có thể phủ bởi một ball100m; không
mặc định là \(1/N\). (15) không yêu cầu chọn một attacker classifier.

Chain rule của KL và \(KL(p\Vert w)\le\chi^2(p\Vert w)\) cho
\(K_n=\sum_{j\le n}\alpha_j^2/4\). Information-radius identity là

\[
I(Y;O\mid C)
=\mathbb E_Y KL(P_{O\mid Y,C}\Vert W)
-KL(P_{O\mid C}\Vert W)\le K_n.
\tag{16}
\]

Sau \(J\), \(\sum_{j>J}\alpha_j^2/4\le\Gamma^2/(48J^3)\).
Chặn tích bằng \(1/(1-s)\) như (10) cho cận \(\delta_\infty\).
Likelihood-ratio martingale có second moment bị chặn đồng đều, nên hội tụ
trong \(L^1\); áp dụng (14) cho law toàn sequence. KL trên sigma-algebra
tăng dần hội tụ đến KL của sequence, vì vậy (16) cũng giữ cho mọi thời
lượng và các phép chiếu/dừng công khai.

**Hệ quả số:** tại \(\Gamma=1\), \(\delta_\infty<0{,}0553\),
\(K_\infty/\ln2<0{,}018\) bit. Hai lựa chọn prior đều cho average
accuracy không quá55,53%; 20 identity candidates prior đều cho không
quá10,53%. Đây là hệ quả có điều kiện của định lý, không phải benchmark S4–S6.
Future target phải có trước kênh hoặc có causal model tương thích; một hành
động tương lai được tạo bởi chính Q output không tự thỏa giả thiết này.

## 6. Mệnh đề 3 — Nội dung truy vấn giữ ở local

Giữ cùng trajectory \(X\), public context và randomness. Nếu thay
\(\psi\) mà không đổi score/support/cadence/category/depth, các bản tin,
phản hồi và cache giống nhau theo quy nạp. Do đó

\[
\boxed{\mathcal L(O\mid X,\psi,C)=\mathcal L(O\mid X,C),
\qquad I(\Psi;O\mid X,C)=0.}
\tag{17}
\]

Đây là payload noninterference: cùng vị trí, đổi mục đích không làm đổi
request server. Nếu nhu cầu có tương quan với trajectory, attacker vẫn có
thể học nhu cầu qua trajectory; kênh suy luận đó chịu (11)–(16) với đúng
target model. Không nhầm noninterference có điều kiện với independence
vô điều kiện của intent và trajectory.

## 7. Mệnh đề 4 — Từ POI được truy hồi đến danh sách trả cho người dùng

Gọi \(\mathcal D\) là catalogue hợp lệ và \(T_t\) là top-k reference
theo GPS thật, nhu cầu \(\psi_t\) và một total order cố định. Ứng dụng
lọc \(U_t\subseteq\mathcal D\) theo \(\psi_t\), rồi trả **toàn bộ**
danh sách phù hợp đã sắp xếp. Top-k chỉ được lấy khi đánh giá.

Mỗi item thuộc \(T_t\cap U_t\) có rank trong subset không lớn hơn rank
trong catalogue. Vì vậy nó nằm trong top-k subset và

\[
\operatorname{Recall@k}(U_t;x_t,\psi_t)
=\frac{|T_t\cap U_t|}{|T_t|}\in[0,1].
\tag{18}
\]

Các reference rỗng là N/A theo protocol, không dùng mẫu số0. Union/reply
cache cho \(U_t\supseteq U_0\); tại cùng GPS/nhu cầu,

\[
\boxed{\operatorname{Recall@k}(U_t;x_t,\psi_t)
\ge\operatorname{Recall@k}(U_0;x_t,\psi_t).}
\tag{19}
\]

Đây là bảo đảm pointwise, không cần belief đúng. Lấy kỳ vọng hoặc trung
bình có trọng số cố định qua các tick giữ bất đẳng thức. Nếu \(T_t\subseteq
U_t\), local top-k trùng reference; với toàn danh sách, chỉ có các items
đã truy hồi được đưa vào trả lời.

## 8. Định lý 3 — Utility cả phiên của private adaptation

Với cùng private evaluation inputs và cùng public-base rule, gọi
\(h(O,X,\psi)\in[0,1]\) là utility: Recall một tick, Recall trung bình
cả phiên hoặc Recall của một purpose cố định. Áp dụng (14) cho **toàn
path**, không ép hai lần chạy phải có cùng realized Q/cache:

\[
\boxed{|\mathbb E_{P_\Gamma}h-\mathbb E_Wh|\le\delta_n
\le\delta_\infty.}
\tag{20}
\]

Mỗi input cố định đều có cùng cận, nên có thể tích phân qua một phân phối
evaluation inputs chung. Tại \(\Gamma=1\), private adaptation làm expected
bounded utility lệch tối đa5,53 điểm phần trăm so với **cùng** public planner
\(\Gamma=0\), dù belief sai. Không phải cận giữa hai planner có support
khác nhau. Nếu \(\mathbb E_Wh\ge c\) đã được chứng nhận đúng miền, thì
\(\mathbb E_{P_\Gamma}h\ge\max(0,c-\delta_n)\).

## 9. Mệnh đề 5 — Khi nào điểm belief cải thiện utility thật?

Tại một history cố định, gọi \(v_a\) là raw planning score đã tính,
\(z_a\) là điểm thực sự đưa vào kernel, và \(r_a\in[0,1]\) là
conditional expected actual utility của action. Từ (4), chính xác

\[
\Delta_r=\mathbb E_pr-\mathbb E_wr
=\alpha\operatorname{Cov}_w(z,r).
\tag{21}
\]

Nếu \(\max_a|r_a-v_a|\le\eta\) và
\(D_z=\mathbb E_w|z-\mu|\), thì

\[
\boxed{\Delta_r\ge\alpha\left[
\operatorname{Cov}_w(z,v)-\eta D_z\right].}
\tag{22}
\]

Chứng minh: viết \(r=v+(r-v)\), dùng
\(\mathbb E_w[(z-\mu)(r-v)]\ge-\eta\mathbb E_w|z-\mu|\).
Nếu quantizer đơn điệu, \(\operatorname{Cov}_w(z,v)\ge0\), vì

\[
\operatorname{Cov}_w(z,v)=\tfrac12\sum_{a,b}w_aw_b
(z_a-z_b)(v_a-v_b)\ge0.
\tag{23}
\]

**Tiêu chí utility có điều kiện:** adaptation không giảm utility tại
prefix này nếu \(\operatorname{Cov}_w(z,v)\ge\eta D_z\). Khi không có
cận \(\eta\), chỉ kết luận gain của planning score; (20) vẫn đúng cho
actual utility mà không cần \(\eta\). Mã certificate tính covariance từ
raw binary64 values và actual integer scores, tránh giả định sai về lỗi
min–max quantization. Gain từng prefix không tự chứng minh gain giữa hai
whole policies: support/cache về sau có thể khác nhau.

Một cách đủ để kiểm soát \(\eta\) là TV error của belief so với law mục
tiêu, cộng uniform proxy error và objective mismatch. Với bảng cố định
\(G(s,a)\in[0,1]\), chênh lệch hai kỳ vọng không vượt
\(TV(b,\rho)\). Nhưng score đang trộn purpose trung bình và proximity;
muốn chứng minh cho một request cụ thể phải tính mismatch đó, không tự
đặt bằng0. Sai số chưa chứng nhận được để là unknown.

## 10. Định lý 4 — Chuyển public-reference floor sang utility thực tế

Đây là bước bổ sung giúp phân biệt rõ một chứng chỉ trên bảng tham chiếu
và một bảo đảm cho các GPS/nhu cầu thật. Gọi \(\rho_0\) là phân phối
trên service cases của miền công khai; mỗi case chứa state/purpose/category
và các tham số liên quan. Chứng chỉ initial pool và (19) cho, với **mọi**
sampled history \(H_t\) đang xét,

\[
\mathbb E_{s\sim\rho_0}g_t(s,H_t)\ge c_t,
\qquad g_t\in[0,1].
\tag{24}
\]

Giả sử law của actual service case, **điều kiện theo history/cache này**,
là \(\rho_t(\cdot\mid H_t)\), với

\[
TV(\rho_t(\cdot\mid H_t),\rho_0)\le\tau_t,
\qquad |r_t(s,H_t)-g_t(s,H_t)|\le\nu_t.
\tag{25}
\]

Miền ngoài bảng có thể đưa vào không gian chung với public mass0; proxy
error phải có nghĩa trên miền so sánh đó. Tính chất biến phân của TV và
(24)–(25) cho

\[
\mathbb E[r_t\mid H_t]\ge
\mathbb E_{\rho_t}g_t-\nu_t
\ge c_t-\tau_t-\nu_t.
\]

Với time weights \(\lambda_t\ge0\) cố định/công khai,
\(\sum_t\lambda_t=1\), tower property cho

\[
\boxed{\mathbb E\bar r
\ge\sum_t\lambda_t\max(0,c_t-\tau_t-\nu_t)=:\underline c,
\qquad \bar r=\sum_t\lambda_t r_t.}
\tag{26}
\]

Không cần các tick độc lập. Với \(b<1\), áp dụng Markov cho deficit
\(1-\bar r\) còn cho

\[
\Pr[\bar r<b]\le
\min\left(1,\frac{1-\underline c}{1-b}\right).
\tag{27}
\]

**Vì sao phải điều kiện theo history?** Hai service cases và hai cache,
mỗi cache chỉ chứa đáp án của một case, đều có public-reference mean0,5.
Nếu kernel nghiêng nhẹ về cache thiếu đáp án của case thật, actual mean có
thể là0,45 dù marginal cases vẫn50/50. TV vô điều kiện bằng0 không cứu
được kết luận. TV của case **sau khi biết cache** là0,05; (26) cho đúng0,45.
Oracle exact giữ ví dụ này để ngăn một phép chuyển floor sai do selection bias.

Định lý 4 dùng floor **mỗi path** của chính cache, nên không cần trừ thêm
\(\delta_n\). Định lý 3 dùng expected utility của matched public-base
policy và trừ \(\delta_n\). Hai cách chứng nhận khác nhau; không lấy mean
benchmark hoặc public mean99% thay vào một baseline guarantee chưa được
chứng nhận. Với cấu hình chính, (24) được chứng nhận trên public mixture;
\(\tau_t,\nu_t\) cho actual cohort chưa được chứng nhận bằng0.

## 11. Các scenario nhận được bảo đảm nào?

| Scenario | Mục tiêu | Kết quả toán học |
|---|---|---|
| S1–S3 | Vị trí tại các mốc, vị trí dừng/proxy và quỹ đạo đã quan sát | (8) cho cả transcript; (12), (15) cho inference với prior/target tương ứng |
| S4 | Liên kết danh tính; người/thiết bị và phân tích phương tiện | (11)–(16) chặn thông tin thêm từ Q, với identity target và cùng public context |
| S5 | Next step/edge đã dự định | Cùng inference bounds khi target causal model thỏa giả thiết |
| S6 | Future destination đã dự định | Cùng inference bounds; không cần giải một bài toán dự đoán để chứng minh likelihood cap |
| S7 | Nội dung nhu cầu truy vấn | (17) bảo vệ payload trực tiếp; phần intent tương quan vị trí chịu inference bounds |
| S8 | Suy luận từ người đồng hành | Cần joint observation model; không gán định lý single-user cho joint transcript |
| S9–S10 | Điểm đầu và cuối chuyến | Là các private GPS inputs của (8); chỉ đổi endpoint có cận event-sensitive (3) |

S1–S3 và S9–S10 là trọng tâm thực nghiệm. Với S4–S6, đóng góp lý thuyết
là một cận suy luận độc lập thuật toán attacker; không chuyển cận đó thành
khẳng định attacker không thể biết danh tính hoặc tương lai. S9–S10 nhận
bảo đảm từ whole-session mechanism hiện tại, không cần gán cho nó một cơ
chế delay mới chưa tồn tại trong release.

## 12. Bảng nối định lý với mã và kiểm chứng

| Kết quả | Cơ chế bảo đảm trong mã | Kiểm chứng |
|---|---|---|
| (1)–(3), Geo-I | `RoadExponential`, `PrivateReuseSMREM`, prospective integer cap | oracle mọi0/1/2 cost allocation; định lý kernel lý tưởng |
| (4)–(6), Q PMF | `FiniteCenteredKernel`; bounded scores; public integer total | exact normalization/PMF/privacy oracles |
| (7)–(10), cả phiên | `BoundedQuerySession`; persistent ledger; ba khung chọn chung | support/path audit; restart/retry; hidden-state/alias/public-stop oracle |
| (11)–(16), inference | `finite_session_bounds`, `bounded_inference`, `whole_path_chi_square` | posterior event oracle; Bayes reward/χ²/TV exact enumeration |
| (17), query content | interface GPS/ψ local tách khỏi protected query interface | same protected information/randomness → same wire tests |
| (18)–(19), cache | union replies; `local_sorted_pois`; fixed ranking | actual utility recomputation từ reply/cache đã lưu |
| (20), utility cả phiên | exact rational prefix/tail của `whole_path_chi_square` | whole adaptive path oracle, không so chỉ một snapshot |
| (21)–(23), gain có điều kiện | `conditional_actual_gain`; giữ raw v và quantized z | duyệt actual utility vectors, unknown error không thành0 |
| (24)–(27), reference transfer | `reference_transfer`; bound phải history-conditional | coupled case/cache counterexample; exhaustive joint-law oracle |

[Chứng chỉ và oracle bổ sung](../reviews/2026-10-14_mathematical_completion.json)
có thể tái lập bằng:

```sh
/private/tmp/trajectory-query-segment-venv/bin/python -m experiments.verify_geoi_bq_mathematics_20261014 --check
```

Chứng minh kernel không dựa vào việc benchmark đẹp. Benchmark hiện tại
99,595% Recall xác nhận một lựa chọn cấu hình có chất lượng dịch vụ tốt
trong protocol; các định lý ở trên giải thích **điều gì được bảo vệ, được
bảo vệ với mức nào, và vì sao các cơ chế của mô hình tạo được bảo đảm đó**.
Đóng góp được trình bày là cấu trúc và phân tích end-to-end của GeoI-BQ
trên nền Geo-I; không coi Geo-I, Bayes, postprocessing hay composition là
các nguyên lý do đề tài phát minh.
