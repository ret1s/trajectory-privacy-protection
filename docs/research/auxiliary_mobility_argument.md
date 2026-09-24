# Học mô hình di chuyển để planner dự đoán sát hơn

Vòng 20 thay mô hình chuyển động của belief, giữ nguyên neo riêng tư, epsilon,
ngân sách, POI tham chiếu và ràng buộc chuyển động của output. Đây là một ablation
đang phát triển, chưa phải bằng chứng đã giải quyết S1.C.

Mô hình cũ khuếch tán xác suất quanh vị trí hiện tại, với khối lượng lớn ở trạng
thái đứng yên. Xe thật có thể di chuyển có hướng dọc đường; một belief quá chậm
có thể tiếp tục ưu tiên POI phía sau. Chỉ tối ưu chính xác objective của belief
không sửa được dự đoán vị trí không phù hợp.

## Dữ liệu và thuật toán

Giữ lưới 2.073 trạng thái công khai. Từ 128 chuyến của 64 nhóm auxiliary, đếm:

- Tᵢ: tổng thời gian quan sát trong ô i, tính bằng giây.
- Nᵢⱼ: số lần chuyển từ ô i sang ô j, với i khác j.

Tốc độ chuyển được làm trơn bằng mô hình công khai cũ:

`qᵢⱼ = (Nᵢⱼ + τ₀ × Pᵢⱼ(20)/20) / (Tᵢ + τ₀)` với i ≠ j;
`qᵢᵢ = −Σⱼ≠ᵢ qᵢⱼ`.

P(20) là chuyển tiếp 20 giây của mô hình cũ. τ₀ là lượng thời gian giả định dành
cho prior: dữ liệu ít thì dựa nhiều vào prior; có nhiều dữ liệu thì dùng tốc độ
quan sát được. Belief sau Δt được dự đoán bằng `b × exp(QΔt)`, rồi cập nhật bằng
likelihood của neo riêng tư như trước. Đồng hồ query có thể không đều.

Tổng dữ liệu fit là 136.561 giây và 6.732 lần đổi ô; 977/2.073 ô có dữ liệu.
Các ô chưa gặp dùng tốc độ từ prior, không bị loại khỏi hỗ trợ. Prior không gian
ban đầu, likelihood REM/reuse và trọng số POI không đổi. Đây là mô hình chuyển
động gần đúng trên lưới; đường đi của output vẫn phải qua bộ kiểm tra trên làn
có hướng, không lấy chuyển ô của belief làm chứng minh khả thi vật lý.

## Chọn tham số ngoài các case đánh giá

Chốt trước τ₀ ∈ {20, 100, 500} giây. Chọn bằng negative log likelihood (NLL)
trung bình theo 16 nhóm auxiliary khác, gồm 1.682 chuyển tiếp 20 giây. Không dùng
S1.C hoặc nhãn của bộ core/expanded để chọn τ₀. NLL thấp hơn là dự đoán tốt hơn.

| Mô hình | NLL theo nhóm (nat) |
|---|---:|
| Mô hình rời rạc cũ | 2,5297 |
| Chỉ chuyển prior cũ thành mô hình thời gian liên tục, chưa học | 2,6710 |
| Học, τ₀ = 20 s | 2,2214 |
| **Học, τ₀ = 100 s** | **2,1961** |
| Học, τ₀ = 500 s | 2,3269 |

Control không học được bổ sung sau khi đã chọn τ₀, để phân biệt tác dụng học dữ
liệu với việc đổi cách biểu diễn thời gian; không thay lựa chọn đã chốt. Cải
thiện NLL chưa chứng minh posterior đã calibrated hay attacker khó suy luận hơn.

## Kết quả tích hợp (vòng 20)

Đã chạy 132 lượt mới, replay đúng 198 control, trên đủ 15 ca của bộ core. Cả hai
bản mới đạt 14/15 ngưỡng Recall 90%; S1.C vẫn thiếu. Với cùng các neo và ledger:

| Bản paced | S1.C cũ | S1.C dùng chuyển động học từ auxiliary |
|---|---:|---:|
| Không slack | 74,17% | 77,50% |
| Slack .03 | 82,50% | 78,33% |

Học chuyển động có ích ở một số ca, nhưng không cải thiện mọi phối hợp. Đây vẫn
là validation phát triển chỉ hai nhóm, đã được xem nhiều lần. Chưa chạy attacker
học phù hợp cho output mới hoặc kiểm tra mở rộng đủ ca. Không chuyển kết luận
privacy của vòng 18 sang hai bản mới. Control thời gian liên tục không học mới
được kiểm tra likelihood; chưa có ablation defender toàn phiên cho control đó.

## Privacy và contribution

Khi chạy trên một người dùng, predictor chỉ nhận belief trước đó, thời gian
công khai và neo đã bảo vệ. Dữ liệu auxiliary ở đây là SUMO tổng hợp, không phải
lịch sử riêng tư của chính người đang được bảo vệ. Học từ dữ liệu người thật sẽ
cần một giả định/quy trình dữ liệu riêng; kết quả này không tự cấp bảo đảm cho
dữ liệu huấn luyện đó.

Mô hình Markov và dự đoán từ lịch sử đã được nghiên cứu trước, chẳng hạn
[Xiao–Xiong](https://arxiv.org/abs/1410.5919) và
[predictive Geo-I](https://arxiv.org/abs/1311.4008). Không nhận đây là một primitive
privacy mới. Câu hỏi của ablation là: mô hình chuyển động phù hợp hơn có giúp
bộ lập kế hoạch K truy vấn với ràng buộc đường đạt dịch vụ tốt hơn ở cùng
ngân sách hay không, và đánh đổi suy luận của attacker là gì?

Nguồn: `iteration20_mobility_fit.json`, `iteration20_mobility_null.json`,
`iteration20_mobility_training.npz` và các generator trong
`artifacts/benchmarks/research_loop/`. Runner tích hợp là
`experiments/research_loop_mobility_cases.py`; output thay đổi phải được kiểm tra
bằng attacker học phù hợp, không dùng lại số privacy của planner cũ.
