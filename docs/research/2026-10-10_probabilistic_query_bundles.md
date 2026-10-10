# Geo-I + chọn gói truy vấn xác suất: số Q linh hoạt và cận suy luận

Trạng thái: **cơ chế phát triển opt-in**, đã có mã kernel và kiểm tra xác suất;
chưa thay thế planner K5 của benchmark quỹ đạo đã công bố. Geo-I/REM,
noisy reuse, ngân sách GPS từng phiên và xử lý nhu cầu tại thiết bị được giữ nguyên.

Phần hoàn thiện tiếp theo: [privacy theo score oscillation, utility bridge và
coverage floor công khai](2026-10-10_query_bundle_privacy_utility.md). Cận β và
cận regret dưới đây vẫn đúng nhưng bảo thủ hơn các cận trong phần bổ sung.

## 1. Điều chỉnh nào có ý nghĩa về toán học?

Planner cũ tối ưu độ phủ POI từ phân bố ước lượng `b` rồi chọn năm Q gần như
xác định. Đây là hậu xử lý hợp lệ của Geo-I, nhưng chưa có một cận riêng chứng
minh Q làm khó việc khôi phục Z hơn. Không gửi Z trực tiếp không đồng nghĩa với
không thể suy ra Z từ quy tắc chọn Q đã biết.

Đề xuất **Geo-I–PQB** (*Probabilistic Query Bundles*): chọn ngẫu nhiên **cả gói
truy vấn**, gồm số lượng và tọa độ Q, từ một thư viện công khai. Gói phủ POI tốt
hơn được ưu tiên; mọi gói trong thư viện vẫn có xác suất dương. Số lượng có thể
là 3/5/7 trong prototype, hoặc một tập hữu hạn bất kỳ do cấu hình công khai quy
định. Không có gì đặc biệt về số 5 trong định lý.

Động lực của K linh hoạt là cân bằng độ phủ dịch vụ với chi phí. **Động lực của
randomization là giới hạn suy luận**. Hai việc này cần được phân biệt. K7 không
tự bảo vệ hơn K3: nhiều mẫu độc lập tập trung quanh Z có thể giúp attacker ước
lượng Z chính xác hơn. Ví dụ nhị phân, mỗi mẫu đoán đúng Z với xác suất 0,75:
một mẫu cho độ chính xác 75%; ba mẫu và biểu quyết đa số cho 84,375%.

Nền tảng lựa chọn là *exponential mechanism*, không phải một định nghĩa riêng
tư mới: [Dwork–Roth, §3.4](https://privacytools.seas.harvard.edu/sites/g/files/omnuum6656/files/privacytools/files/the_algorithmic_foundations_of_differential_privacy.pdf).
Geo-I giữ vai trò bảo vệ phép đọc tọa độ:
[Andrés et al., §3](https://arxiv.org/html/1212.1984v3).
Đóng góp đang nghiên cứu là áp dụng một kernel chung cho **hình học Q, K,
độ phủ POI và chi phí**, kèm cận suy luận và cách cộng dồn đúng.

## 2. Luồng và các biến

```text
GPS thật X ──Geo-I / kiểm tra tái sử dụng──> lịch sử tham chiếu được bảo vệ E
                                                     │
                  bản đồ + POI công khai ────────────> b(E)
                                                     │
                 thư viện gói công khai A ──PQB──────> A_t=(K_t,Q_1,...,Q_Kt)
                                                     │
                               server trả L30 cho mọi danh mục từ từng Q
                                                     │
                        union/dedup ──GPS + nhu cầu ψ tại local──> lọc/sắp xếp
```

- `E`: toàn bộ lịch sử được bảo vệ, không chỉ Z cuối cùng. Với một lần REM
  mới và không có lịch sử trước đó, có thể viết đơn giản `E=Z`.
- `b(E)`: phân bố vị trí thiết bị ước lượng từ E và ngữ cảnh công khai. `b`
  dùng để ưu tiên dịch vụ, **không mặc nhiên là posterior thật của attacker**.
- `A`: thư viện hữu hạn các gói Q, dựng trước từ bản đồ/POI công khai. Một gói
  `a` chứa K(a) điểm đường hợp lệ. Thứ tự gửi cũng là một phần của gói.
- `w(a)>0`: trọng số công khai, tổng bằng 1. Prototype phân bổ tổng khối lượng
  đều cho các nhóm K, rồi đều cho các gói trong mỗi nhóm để tránh thiên lệch do
  một nhóm có nhiều gói hơn.
- `g(x,a)∈[0,1]`: chất lượng của tập POI từ gói a tại vị trí ứng viên x. Prototype
  dùng độ phủ reference nearest Recall@5 theo các danh mục có POI; bản khái quát
  có thể dùng một tổ hợp **công khai, cố định** của nhiều mục đích.
- `λ≥0`: mức phạt chi phí; `β≥0`: độ tập trung của phép chọn gói, không có đơn
  vị. β nhỏ bảo vệ mạnh hơn; β lớn ưu tiên gói có utility tốt hơn.

K, λ, β, w, L và quy tắc dựng thư viện phải là cấu hình công khai. Không chọn
β từ GPS hoặc nhu cầu riêng. Không truyền b, điểm utility hoặc debug log Z cho
server. Nhu cầu thật ψ tiếp tục chỉ được dùng sau khi nhận POI.

## 3. Phân phối chung của Q và K

Đặt chất lượng mong đợi và điểm đã chuẩn hóa:

\[
G_b(a)=\sum_x b(x)g(x,a),\qquad
s_b(a)=\frac{G_b(a)+\lambda(1-K(a)/K_{\max})}{1+\lambda}\in[0,1].
\]

Chọn **một gói duy nhất** theo phân phối:

\[
\boxed{T_\beta(a\mid E)=
\frac{w(a)\exp\{\beta s_{b(E)}(a)/2\}}
{\sum_{a'\in\mathcal A}w(a')\exp\{\beta s_{b(E)}(a')/2\}}.}
\]

Vì K là một phần của a nên

\[
P(K=k\mid E)=\sum_{a:K(a)=k}T_\beta(a\mid E).
\]

K thích ứng **theo xác suất**, tùy độ phủ có thể đạt được dưới b. Không phải
một quy tắc “vượt ngưỡng mật độ thì luôn K7”; quy tắc như vậy có thể tự tiết lộ
ngưỡng. Đây cũng không phải năm phép lấy mẫu độc lập nên không được nhân năm
phân phối riêng rồi mặc nhiên kết luận tốt hơn.

Thuật toán một lần chọn:

1. Cập nhật b từ đầu ra Geo-I và chuyển động công khai; không đọc thêm GPS.
2. Tính G cho **mọi** gói trong thư viện chung.
3. Tính log-probability bằng log-sum-exp và lấy mẫu một gói với randomness bí mật.
4. Gửi L30, cùng tập danh mục và quy tắc công khai cho từng Q trong gói.
5. Dedup, lọc và sắp xếp tại local theo ψ và GPS hiện tại.

**Ví dụ nhỏ minh họa K thích ứng**, không phải số đo từ dữ liệu: hai vùng A/B,
ba gói K3/K5/K7 có độ phủ tương ứng (0,90/0,30), (0,95/0,75), (1,00/1,00).
Với w đều, β=2 và λ=0,25, cùng công thức cho:

| Phân bố b(A), b(B) | P(K3) | P(K5) | P(K7) |
|---|---:|---:|---:|
| 1,00; 0,00 | 33,91% | 33,33% | 32,76% |
| 0,50; 0,50 | 29,57% | 34,11% | 36,32% |
| 0,00; 1,00 | 25,54% | 34,57% | 39,88% |

Ở A, K3 đã phủ tốt nên được ưu tiên hơn; ở B, K7 có lợi hơn. β=2 giữ xác suất
của các lựa chọn khác tương đối lớn để hạn chế lộ vị trí. Không thể đồng thời
đòi “luôn chọn đúng gói tối ưu theo vị trí” và cận global β nhỏ.

## 4. Định lý 1: cận chung, bao gồm cả K

**Giả thiết:** cùng ngữ cảnh công khai, cùng thư viện và w; s nằm trong [0,1]
cho mọi lịch sử E; phép lấy mẫu thực hiện đúng phân phối lý tưởng.

Với mọi E, E′ và mọi gói a:

\[
\boxed{e^{-\beta}\le
\frac{T_\beta(a\mid E)}{T_\beta(a\mid E')}
\le e^\beta.}
\]

**Chứng minh.** Chênh lệch điểm của một gói không quá 1 nên tỷ số các tử số
không quá `exp(β/2)`. Với mọi gói a′, điều tương tự đúng; cộng theo w(a′) cho
tỷ số hai mẫu số cũng không quá `exp(β/2)`. Nhân hai cận được `exp(β)`.
Đổi vai E và E′ cho chiều ngược lại. Cận giữ nguyên khi cộng xác suất trên một
tập gói, nên việc attacker chỉ quan sát K cũng được bảo vệ.

Cận này không cần b chính xác và không cần các Q độc lập. Nó áp dụng cho mọi
cặp lịch sử E, kể cả ở xa nhau. Sai số b ảnh hưởng chất lượng dịch vụ, nhưng
không làm sai định lý nếu vẫn là một phân bố chuẩn hóa và thư viện chung.

λ cố định làm phần chi phí giống nhau giữa E và E′; thực tế có thể siết cận
hơn vì `|s_b(a)−s_b′(a)|≤1/(1+λ)`. Tài liệu và API dùng cận β bảo thủ để không
phụ thuộc cách chọn objective về sau; không xem giá trị số kiểm tra nhỏ hơn β
là một chứng nhận floating-point.

## 5. Định lý 2: nối GPS → Z → Q và posterior

Với một fresh REM, đặt `R(z|x)=P(Z=z|X=x)`. Không cần biết Z có khôi phục được
từ Q hay không:

\[
P(A=a\mid X=x)=\sum_z R(z\mid x)T_\beta(a\mid z).
\]

Do T(a|z) có tỷ số lớn nhất/nhỏ nhất không quá e^β, mọi trung bình có trọng số
của các giá trị này cũng nằm trong cùng khoảng. Vì thế

\[
\boxed{\frac{P(A=a\mid X=x)}{P(A=a\mid X=x')}
\le \exp\!\{\min(u\,d(x,x'),\beta)\}.}
\]

Cận `ud` đến từ hậu xử lý của REM u-Geo-I; cận β đến từ định lý 1. Phải thỏa
cả hai nên lấy **min**, không cộng u với β. β không tiêu thêm lượt GPS; đây là
một mức hạn chế thông tin trong lớp hậu xử lý.

Cho π là prior của attacker. Công thức đúng cho suy luận vị trí thật là

\[
\boxed{P(X=x\mid A=a)=\sum_z P(X=x\mid Z=z)P(Z=z\mid A=a).}
\]

Đây là quy tắc nhân kết hợp tổng theo Z ẩn. Không được chỉ nhân hai xác suất
ở một giá trị Z đoán được. Hơn nữa, với mọi x trong prior và cận chung α:

\[
\boxed{P(X=x\mid A=a)\le
\frac{e^\alpha\pi(x)}{e^\alpha\pi(x)+1-\pi(x)},\quad\alpha\le\beta.}
\]

**Chứng minh cận posterior:** đặt l_x=P(a|x). Cận tỷ số cho
`l_x′≥e^(−α)l_x` với mọi x′. Thay vào mẫu số Bayes, giữ nguyên π(x)l_x ở tử số,
rồi rút gọn. Muốn dùng `α=min(ur,β)` cho tất cả ứng viên phải có khoảng cách
giữa mọi cặp ứng viên không quá r. Dù không có r, vẫn dùng được α=β.

Ví dụ **lý thuyết**, β=1 và hai vị trí có prior 50/50 cho posterior không quá
73,11%; 36 vị trí đồng prior cho không quá `e/(e+35)=7,21%`. Với β=2,
hai vị trí cho 88,08% và 36 vị trí cho 17,43%. Đây không phải tỷ lệ đúng đo được
trong benchmark, cũng không phải suy luận “bất khả thi”. Nếu prior đã 99% thì
β=1 vẫn cho phép posterior tới 99,63%; cơ chế không xóa thông tin đã biết.

Nếu muốn thiết kế từ một yêu cầu thay vì tùy ý chọn β, đặt upper bound p cho
prior của một giả thuyết và mục tiêu posterior τ, với 0<p≤τ<1. Điều kiện đủ là:

\[
\boxed{\Gamma\le\log\frac{\tau(1-p)}{(1-\tau)p}.}
\]

Ví dụ p=1/36 và τ=20% cho Γ≤log(8,75)≈2,169. Một lần chọn β=2 thỏa điều kiện;
10 vòng công khai phải chia tổng β_t≤2,169. Nếu không biết prior upper bound,
chỉ được cam kết hệ số tăng odds e^Γ, không hứa posterior tuyệt đối ≤20%.

Một cận bổ sung cho attacker tối ưu trên tập vị trí hữu hạn:

\[
V(X\mid A)=\sum_a\max_x\sum_z\pi(x)R(z\mid x)T(a\mid z)
\le\sum_z\max_x\pi(x)R(z\mid x)=V(X\mid Z).
\]

Đẩy `max` vào trong tổng cho bất đẳng thức; dùng `Σ_a T(a|z)=1` để rút gọn.
Do vậy Q không làm tăng xác suất đoán tối ưu trung bình so với việc công khai
Z. **Không nhất thiết giảm nghiêm ngặt**, và không có đảm bảo cho mỗi lần chạy
riêng so với Z thật ở lần đó. Cần đo trade-off, không tuyên bố vượt trội mặc định.

## 6. Định lý 3: nhiều bước di chuyển và ngân sách

Với phiên hoàn chỉnh phải thay Z bằng **toàn bộ lịch sử được bảo vệ E**. Planner
lưu b, Z trước đó và Q trước đó, nên chỉ viết `X_t→Z_t→Q_t` không điều kiện sẽ
bỏ sót lịch sử. Quan hệ đúng là `X_1:T→E→A_1:m`, với ngữ cảnh công khai cố định.

Tại mỗi vòng công khai t, conditional kernel của gói mới có cận β_t cho mọi E,
E′ **khi giữ nguyên các gói đã quan sát trước đó**. Nhân các conditional likelihood:

\[
\Gamma=\sum_{t=1}^{m}\beta_t,\qquad
\boxed{\alpha_{\rm session}\le
\min\{C_s\,D_\infty(X,X'),\Gamma\}.}
\]

`C_s` là ngân sách Geo-I từng phiên hiện có (cấu hình làm việc: 0,23/m); không
đổi ngân sách GPS và không giới hạn số phiên. Γ là cận cộng dồn **số lần lấy mẫu
gói mới**, đơn vị không chiều, hoàn toàn khác C_s. Hai phiên liên kết thì cộng
hai Γ và hai cận Geo-I tương ứng. Không được lấy β của một gói làm cận cho
toàn chuyến đi hoặc toàn bộ lịch sử của một người.

Các điều kiện triển khai cần giữ:

- Vòng chọn gói/cadence/max-round phải cố định công khai; không chọn lại chỉ vì
  GPS vừa gây refresh, vì chính thời điểm refresh có thể lộ thông tin.
- Thư viện một vòng có thể phụ thuộc Q trước đó **đã gửi** và elapsed time
  công khai để giữ tính khả thi trên đường; phải giống nhau cho mọi E khi giữ
  transcript trước đó cố định. K thay đổi cần quy tắc sinh/kết thúc track công
  khai. Prototype hiện tại mới kiểm tra một snapshot, chưa triển khai phần này.
- Chọn lại nhiều lần với randomness mới tiêu thêm β dù Z giữ nguyên; điều này
  khác việc tiêu ngân sách Geo-I cho GPS. Khi Γ hết, có thể dùng lại gói đã
  lấy mẫu tại một lịch công khai; không phải đọc lại GPS để tạo gói mới miễn phí.
- Gửi lại một gói đã lấy mẫu, hoặc xử lý dữ liệu phản hồi bằng cùng kernel
  server/adversary, là hậu xử lý, không phải thêm một lần lấy mẫu độc lập.
  Không có cận này cho thời gian gửi/đóng phiên do hành vi riêng quyết định.
- RNG phải độc lập với private mechanism và không công khai seed. K, thứ tự,
  category, byte count, retry và các kênh được đưa vào transcript phải tuân
  quy tắc trên. Log/debug và account/IP ngoài threat model không được bảo vệ.
- Cận Γ chỉ áp dụng khi mọi đầu ra phụ thuộc E được gửi qua kernel gói hoặc
  hậu xử lý của gói. Nếu gửi trực tiếp Z, cờ fresh/reuse hay ledger GPS, kênh đó
  có thể vượt qua lớp PQB; không được giữ cận Γ cho transcript mở rộng như vậy.

Ví dụ m=10 vòng công khai, β_t=0,1 cho Γ=1; **đây là phân bổ minh họa**, không
phải tham số được xác nhận QoS. Giữ β_t=2 ở 10 vòng cho Γ=20, cận suy luận lại
yếu. Muốn dùng kết quả ở hội đồng phải nêu cận theo đúng cửa sổ quan sát.

## 7. Liên hệ S4, S5, S6 và S7

Giả sử một nhãn Y (người, phương tiện vật lý, next route hoặc destination) chỉ
ảnh hưởng đầu ra qua quỹ đạo/ước lượng vị trí, trong **cùng ngữ cảnh công khai**.
Vì kernel có cận Γ cho mọi E, các hỗn hợp `P(A|Y=y)` cũng có tỷ số không quá
e^Γ, dù hai nhãn có các phân bố quỹ đạo khác nhau. Không còn cần ép các quỹ đạo
của hai nhãn nằm gần nhau để có **cận Γ này**.

\[
\frac{P(Y=y\mid A)}{P(Y=y'\mid A)}
\le e^\Gamma\frac{P(Y=y)}{P(Y=y')},\qquad
P(Y=y\mid A)\le\frac{e^\Gamma\pi_y}{e^\Gamma\pi_y+1-\pi_y}.
\]

Với hai lớp đồng prior, balanced Bayes accuracy không quá logistic(Γ). Với M
lớp đồng prior, posterior từng lớp không quá `e^Γ/(e^Γ+M−1)`; **M là số giả
thuyết nhãn, không phải K điểm Q**.

- **S4 identity:** cận áp dụng riêng cho nhãn người và nhãn phương tiện; không
  bảo vệ khi username, biển số, account hoặc IP đã định danh nhãn. Với linkage
  hai transcript, phải tính chi phí của cả hai và lịch sử liên kết được.
- **S5 next route:** tính Γ cho toàn bộ prefix attacker quan sát. Không cần xây
  một predictor mới để chứng minh cận, nhưng prior tuyến đi quen thuộc vẫn tồn tại.
- **S6 destination:** tương tự, bao gồm lịch sử quan sát được có liên quan; không
  chứng minh “không biết nơi đến” chỉ từ một snapshot.
- **S7 query content:** giữ ψ ngoài s và ngoài mọi request. Khi cố định quỹ đạo
  và ngữ cảnh, phân phối transcript giống nhau cho ψ và ψ′. Không tự bảo vệ
  sự tương quan giữa ý định với quỹ đạo, hay click sau đó được gửi lên server.

Đây là bảo vệ **mức tăng suy luận do transcript**, có điều kiện và định lượng;
không tuyên bố đã giải quyết hoàn toàn mọi dạng identity/future inference.

## 8. Cận utility và giới hạn

Cho `s*=max_a s_b(a)`, và `w*` là tổng trọng số công khai của các gói đạt s*.
Với β>0 và τ≥0, tử số của các gói có điểm ≤s*−τ bị chặn bởi
`exp(β(s*−τ)/2)`; mẫu số ít nhất `w*exp(βs*/2)`. Vì vậy:

\[
P(s^*-s_b(A)\ge\tau)\le
\min\{1,e^{-\beta\tau/2}/w^*\}.
\]

Tích phân cận đuôi cho:

\[
\mathbb E[s^*-s_b(A)]\le
\min\{1,\tfrac{2}{\beta}(\log(1/w^*)+1)\}.
\]

Đây là cận cho **điểm ước lượng trên thư viện**, không phải cam kết Recall thật
trên mọi quỹ đạo hay mọi nhu cầu. Với β nhỏ hoặc thư viện lớn, cận có thể rất
yếu. Thư viện không chứa gói tốt thì randomization không sửa được sự thiếu hụt.
Phải đo QoS thật, chi phí và mục đích truy vấn trước khi tích hợp.

Các cách cần tránh:

1. Chỉ giữ các gói gần Z hoặc cắt xác suất thấp về zero: mất common support,
   không còn cận β; Geo-I hậu xử lý vẫn có thể còn nếu chỉ dùng E, nhưng không
   được giữ tuyên bố về cận mới.
2. Xem K lớn là k-anonymity hoặc M danh tính có thể: Q không phải người thật.
3. Tính cận một lần rồi bỏ qua việc gói mới được chọn tại nhiều bước.
4. Thay kết quả Q hiện có và tiếp tục ghi benchmark cũ như bằng chứng của cơ chế mới.

## 9. Mã, kiểm chứng và phạm vi kết quả

- Kernel: [`benchmark/probabilistic_query_bundle.py`](../../benchmark/probabilistic_query_bundle.py).
- Test: [`tests/test_probabilistic_query_bundle.py`](../../tests/test_probabilistic_query_bundle.py).
- Exact audit: [`experiments/query_bundle_audit_20261010.py`](../../experiments/query_bundle_audit_20261010.py).
- Protocol và toàn bộ frontier: [`query_bundle_kernel_20261010_v2`](../../artifacts/benchmarks/query_bundle_kernel_20261010_v2/).

Audit dùng bản đồ SUMO native và danh mục POI công khai đã có, L30, một fresh
REM u=0,01/m, một tập ứng viên GPS lấy từ grid công khai. **Toàn bộ trạng thái
đường của map** vẫn là miền đầu ra REM, không cắt thành grid. Tính chính xác
channel `R×T`, kỳ vọng Recall, K, Bayes optimal trên prior hữu hạn và posterior
lớn nhất, không fit một attacker và không phụ thuộc lần random may mắn.

Control K5 là maximizer trên **cùng thư viện**; không phải full planner cũ có
ràng buộc dummy motion. Toàn bộ frontier β=0/1/2/4/8 được công bố; β=2 chỉ là
điểm minh họa định trước, chưa chọn bằng test hay đề xuất đưa thẳng vào production.
Có cả prior đồng đều và prior lệch 80% để nhìn rõ giới hạn khi side information mạnh.
Trong audit này, belief của thiết bị dùng cùng prior và cùng REM kernel với
attacker; đây là mô hình **matched** thuận lợi cho objective ước lượng, không
phải kiểm chứng calibration của bộ lọc hiện tại. Định lý β không cần sự khớp
này, nhưng utility ở một prior hoặc mobility model khác cần đánh giá riêng.

**Kết quả đã tính**: 66.189 trạng thái đầu ra REM, 36 ứng viên vị trí và 48 gói
công khai. Với prior đồng đều, một fresh REM và β=2:

| Cơ chế trên cùng thư viện | Expected Recall@5 | Expected K | Bayes đoán đúng vị trí grid | Posterior lớn nhất |
|---|---:|---:|---:|---:|
| Control tối ưu xác định, K5 | 99,10% | 5,000 | 33,91% | 81,78% |
| PQB ngẫu nhiên, giữ K5 | 90,64% | 5,000 | 3,09% | 3,35% |
| PQB ngẫu nhiên, K3/5/7 | 89,12% | 4,993 | 3,09% | 3,43% |

Baseline đoán theo prior là 1/36=2,78%. “Đoán đúng vị trí grid” ở đây là lựa
chọn một trong 36 giả thuyết hữu hạn, **không phải Hit100 hoặc accuracy S4**.
Expected Recall tính trên vị trí thật ứng viên và toàn bộ channel R×T; không
chỉ chấm điểm mục tiêu trên b. Đây là kỳ vọng một query snapshot, không có CI
gia đình quỹ đạo hoặc phép so sánh với bảng benchmark cũ.

Tỷ số likelihood lớn nhất đo được ở K động là exp(0,348), dưới cận exp(2).
Cận posterior lý thuyết cho prior 36 vị trí là 17,43%; giá trị thực tính 3,43%
nhỏ hơn cận này. Cận bảo thủ và kết quả tính được có vai trò khác nhau.

Ở prior lệch 80%, Bayes accuracy của PQB vẫn là 80% vì đoán theo prior đã rất
mạnh; posterior lớn nhất của K động là 81,27%. Vì vậy **không tuyên bố cơ chế
ngăn được mọi suy luận danh tính hay tuyến đi quen thuộc**.

Kết luận phát triển: đã có một đánh đổi xác suất–utility rõ ràng; randomization
có đóng góp đáng đo trên diagnostic này. **K động chưa tạo lợi ích thuyết phục
so với PQB giữ K5**: utility thấp hơn khoảng 1,52 điểm %, số query trung bình gần
như không giảm. Chưa chốt K động là cấu hình chính. Phải phát triển thư viện
và transition trước khi đánh giá trên chuyến đi hoàn chỉnh.

Kết quả chỉ là **static development diagnostic** của lớp chọn gói. Chưa có
chứng cứ moving-trajectory S1/S2/S3/S9/S10 mới, transition K động, b trên lịch
sử dài, người/phương tiện với dữ liệu nhãn thật, hoặc xác nhận generalization
đa bản đồ. Đây là bước tiếp theo của nghiên cứu, không phải cớ tuyên bố fully.

Pure-DP trong các định lý là số thực lý tưởng. Prototype kiểm tra xác suất
không zero và các bất đẳng thức bằng float, nhưng chưa có sampler số học hữu
hạn được chứng nhận. Không gọi việc test pass là chứng minh triển khai pure-DP.

## 10. Cách trình bày ngắn trước hội đồng

“Geo-I vẫn bảo vệ phép đọc GPS. Chúng tôi bổ sung một lớp chọn ngẫu nhiên cả
gói truy vấn thay vì luôn lấy năm điểm bằng tối ưu xác định. Gói có thể chứa
số điểm khác nhau, được ưu tiên theo độ phủ POI và chi phí. Một phân phối
chung cho cả tọa độ lẫn số điểm cho phép chứng minh likelihood ratio bị chặn
bởi e^β, rồi chuyển thành cận posterior và cộng dồn qua các bước. Như vậy,
bảo vệ không dựa vào việc attacker không biết quy tắc hoặc không thấy Z;
nó dựa vào cận xác suất dù attacker biết cơ chế. Chúng tôi đang kiểm tra đánh
đổi utility và cost trước khi đưa lớp này vào benchmark quỹ đạo đầy đủ.”
