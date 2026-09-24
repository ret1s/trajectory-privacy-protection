# Ưu tiên đạt đủ dịch vụ ở nhiều vị trí khả dĩ

Vòng 25 thay hàm mục tiêu của planner, giữ nguyên metrics đánh giá, K5, L10,
neo, ngân sách và đường đi. Mục tiêu cũ tối đa hóa Recall trung bình theo belief.
Mục tiêu mới giảm ưu tiên cho một vùng khi coverage dự đoán của vùng đó đã đạt
90%, để dành truy vấn cho các vùng khả dĩ khác còn thiếu dịch vụ.

## Ví dụ

Giả sử người dùng có thể ở hai vùng, mỗi vùng xác suất 50%:

| Tập truy vấn | Recall vùng 1 | Recall vùng 2 | Trung bình | Mục tiêu chặn ở .9 trước khi trung bình |
|---|---:|---:|---:|---:|
| A | 100% | 80% | 90% | 85% |
| B | 90% | 90% | 90% | 90% |

Mục tiêu cũ coi A/B ngang nhau. Mục tiêu mới thích B vì cả hai vùng đều đủ mức
dịch vụ. Đây chỉ là ví dụ để giải thích hàm mục tiêu, không phải số benchmark.

## Công thức và ranh giới thông tin

Gọi bᵢ là belief đã bảo vệ, wᵢₚ là trọng số POI p trong top-5 tham chiếu của
vị trí công khai i, và U(A) là hợp POI server trả cho tập truy vấn A:

`Rᵢ(A) = Σₚ wᵢₚ 1[p ∈ U(A)]`

`Fτ(A) = Σᵢ bᵢ min(Rᵢ(A), τ)`, với τ=.9.

Trọng số w là công khai và chuẩn hóa theo các category có tham chiếu; POI trùng
chỉ tính một lần. Cap áp dụng ở từng vị trí trước khi lấy kỳ vọng. Điều này
khác cap trên coverage category đã trung bình qua mọi vị trí. Số .9 lấy từ yêu
cầu dịch vụ đã chốt, không quét tham số bằng các điểm S1.C.

Không có nhãn scenario, GPS mới hoặc vị trí tương lai trong phép chọn. Chỉ
belief từ neo đã bảo vệ cùng POI/bản đồ công khai. Cận lý tưởng của cơ chế neo
giữ nguyên theo hậu xử lý; điều này không chứng minh privacy thực nghiệm tăng.

Cách đọc tương đương: `τ − Fτ(A) = E_b[(τ − Rᵢ(A))₊]`, tức thiếu hụt dịch vụ
trung bình so với ngưỡng. Ví dụ A ở trên thiếu trung bình .05, còn B thiếu 0.
Với δ>0, bất đẳng thức Markov cho
`P_b(Rᵢ(A) ≤ τ−δ) ≤ (τ−Fτ(A))/δ`.
Đó là hệ quả toán học quen thuộc trên **belief đang dùng**, không phải cận xác
suất thất bại ngoài thực tế. Không thể thay Recall thực nghiệm bằng số này khi
belief chưa được hiệu chuẩn.

## Tính chất tối ưu có thể kiểm tra

Mỗi Rᵢ là coverage có trọng số không âm. Khi tập đã chọn lớn hơn, phần POI mới
do một truy vấn bổ sung không tăng, và khoảng còn lại tới cap cũng không tăng.
Do đó Fτ có lợi ích biên giảm dần, không giảm và bằng 0 ở tập rỗng. Đây là áp
dụng tính submodular đã biết, không nhận một định lý greedy mới.

Tiền lệ trực tiếp là [Krause et al., JMLR 2008, mục 4 và 7.5](https://www.jmlr.org/papers/volume9/krause08b/krause08b.pdf):
SATURATE dùng trung bình các mục tiêu bị chặn và xét đánh đổi trung bình/worst-case.
Ta dùng trọng số belief, cap cố định và miền đi tới được riêng cho từng track.
Không chuyển bảo đảm worst-case của SATURATE sang planner này, cũng không nhận
phép chặn mục tiêu là novelty. Đây là nguồn nền tảng thuật toán; phần related
works ứng dụng vẫn cần các công trình gần đây.

Greedy chọn một vị trí cho mỗi track trên miền khả thi cố định; cận 1/2 của
greedy partition-matroid áp dụng trong số học lý tưởng với marginal chính xác.
Exchange cải thiện objective đó. Bước progress bảo toàn toàn bộ POI replies nên
bảo toàn Fτ tại thời điểm ấy. Slack .03 có thể làm mất tối đa .03 objective theo
belief; không chuyển cận này thành Recall thật, chất lượng cả chuyến hoặc cận
global sau các lựa chọn làm thay đổi miền di chuyển ở tương lai.

## Hiện thực và phạm vi thử

`benchmark/capped_service_objective.py` dùng ma trận sparse giữa POI và vị trí
khả dĩ, gộp các chữ ký phản hồi giống nhau, cache theo hợp POI đã phủ. Không
loại khối lượng belief nhỏ. Marginal được kiểm tra bằng vét cạn các ví dụ nhỏ.
Tắt cap phải cho đúng output/neo/ledger của bản cha; cap=1 bằng expected-union
utility ở mức công thức, không mặc định bitwise tie của hai cách tính giống nhau.

Hai biến thể chốt trước: cap .9, slack 0 và .03. Profile dùng prefix từ auxiliary
train, ghi riêng chi phí dựng index. Screen đã chạy đủ 15 ca core, cùng đồng hồ
và neo ghép cặp: đạt 12/15 và 13/15 ngưỡng. S1.C là 73,33% / 75,83%, chưa tốt hơn
hai control tương ứng. Không chọn các bản này để tiếp tục confirmation. Chưa
có attacker học tương ứng. [Đầy đủ các ca và đánh đổi](capped_planar_results.md).

## Gộp chính xác nhu cầu dịch vụ để giảm phép tính

Nếu hai vị trí công khai có cùng hàng trọng số POI W, chúng có cùng R(A) với
mọi tập truy vấn A. Gom chúng thành nhóm g và cộng belief:

`Fτ(A) = Σg (Σi∈g bᵢ) min(Wg · covered(A), τ)`.

Đây là đẳng thức trong số học thực, giữ toàn bộ khối lượng xác suất. Gộp nhu cầu
ở đây khác phép gộp ứng viên có cùng **phản hồi**: một phép giảm số vị trí phải
tính kỳ vọng, phép kia giảm số lựa chọn phải so sánh. Không gộp chỉ vì hai vị
trí gần nhau hoặc có cùng category nhưng khác POI.

Trên bản đồ hiện tại, 2.073 hàng tham chiếu còn 458 hàng khác nhau. Profile
12 thời điểm của một chuyến auxiliary-train cho 688,80 ms/bước với phép tính
trực tiếp và 196,63 ms/bước sau gộp. Hai bản chọn cùng trạng thái ở cả 12 bước;
neo cũng giống nhau. Đây là profile tuần tự, cache đã nạp, không phải benchmark
hiệu năng độc lập hoặc kết quả trên điện thoại. Đổi thứ tự cộng float có thể
đổi quyết định khi hòa điểm; không khẳng định bitwise tương đương trên mọi phiên.
Chi phí dựng chỉ mục ghi riêng trong
[profile](../../artifacts/benchmarks/research_loop/iteration25_capped_grouped_profile.json).

Đây không phải tối ưu worst-case: một vùng có belief rất nhỏ vẫn có thể bị bỏ
qua. Belief sai hoặc chuyển động khó theo kịp cũng có thể làm Recall thật thấp.
