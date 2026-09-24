# Đối chiếu primitive tạo neo trong cùng planner

Vòng 26 thay riêng kernel tạo neo và likelihood tương ứng trong belief.
Planner, K5, L10, top-5 tham chiếu, đồng hồ, giãn đọc 60 giây, ngưỡng reuse
200 m và trần .23 m⁻¹ giữ nguyên. Hai biến thể có slack 0 và .03.

## Vì sao cần phép so sánh này?

Neo REM hiện tại chọn một điểm trong toàn bộ mạng đường với xác suất tỷ lệ
`exp(-ε d(x,v)/2)`. Hệ số 1/2 kiểm soát cả tử số và hằng số chuẩn hóa phụ thuộc
vị trí thật. Planar Laplace chuẩn có mật độ trên toàn mặt phẳng
`ε²/(2π) exp(-ε d(x,z))`; chuẩn hóa không phụ thuộc x. Cùng cận ε không có nghĩa
hai kernel có cùng sai lệch, chất lượng dịch vụ hoặc độ khó suy luận thực tế.

Planar Laplace lấy bán kính Gamma(2,1/ε), góc đều. Đây là cơ chế gốc của
[Andrés et al., CCS 2013, mục 4.1](https://arxiv.org/html/1212.1984).
Private test để dùng lại đầu ra trước dựa trên
[Chatzikokolakis et al., PETS 2014, mục 3–4](https://arxiv.org/html/1311.4008).
Phép thử này kế thừa các primitive đó; chưa tái lập toàn bộ giao thức dịch vụ
và chính sách ngân sách của hai paper.

Nếu đổi kernel đã tạo gain lớn, không được gán gain ấy cho contribution planner.
Nếu Recall tăng nhưng attacker đoán endpoint đúng hơn, phải báo cả đánh đổi.
So cùng cận hình thức là một phép kiểm soát; vẫn cần so privacy thực nghiệm.

## Hiện thực và ranh giới

Nhiễu được cộng vào tọa độ XY trong phép chiếu công khai cố định. Không snap
GPS thật trước, không cắt bán kính, không ép neo lên đường. Chỉ các vị trí truy
vấn cuối cùng của planner phải đi được trên mạng đường. Cận dùng khoảng cách
Euclidean trong XY này, không tự chuyển thành khoảng cách địa lý trên toàn cầu.

Belief dùng cùng 2.073 vị trí khả dĩ và prior/chuyển động cũ. Khi có private read:

- Lần đầu: cập nhật bằng mật độ planar Laplace.
- Neo trùng neo trước: dùng xác suất private test chấp nhận reuse.
- Neo khác: dùng xác suất test từ chối nhân mật độ fresh.

Fresh liên tục có xác suất bằng 0 tại một điểm chính xác. Vì thế không cộng
“mật độ fresh” vào “khối xác suất reuse” như với REM rời rạc. Đây là likelihood
của kernel lý tưởng; belief lưới và mô hình chuyển động vẫn chỉ gần đúng.

Lượt đầu tốn .01; private test tốn .01; fresh sau test thêm .01. Chỉ đọc GPS
khi ngân sách đủ cho nhánh đắt nhất. Hết ngân sách, tiếp tục planner từ lịch sử
đã bảo vệ. Kernel khác có thể đổi kết quả test và số lần đọc, nên không yêu cầu
ledger của planar giống REM. Hai bản planar ghép cùng RNG phải có cùng neo/ledger.

Các sampler float không chứng minh pure DP chính xác của chương trình. Cận
toàn phiên chỉ áp dụng theo các giả định đồng hồ cố định của
[lập luận filter](predictive_filter_argument.md); không bảo vệ timestamp mở/đóng.

## Phạm vi bằng chứng

Screen đã chạy đủ 15 ca core, 33 chuyến nguồn × 2 RNG × 2 biến thể = 132
lượt mới; replay 198 control. Hai nhóm validation đã dùng trong phát triển,
không phải confirmation. Riêng S9.B và S10.A chỉ có một nhóm đủ điều kiện;
các ca còn lại có hai. Attacker học trên REM không được chuyển sang coi là
matched attacker của planar. Chưa kết luận cải thiện S9/S10 từ screen này.

Sáu test kiểm tra sampler, atom/density, tỷ số likelihood trên lưới nhỏ,
causality, đường đi và không đọc GPS khi hết ngân sách. Test tính đúng không
thay thế đánh giá hiệu quả bảo vệ. Runtime của screen có thể chịu ảnh hưởng
job vòng 25 chạy cùng lúc, không dùng để suy ra tăng tốc so với REM.

## Kết quả core

| Kernel / planner | Recall toàn phiên | S1.C | S2.C | Ca đạt 90% |
|---|---:|---:|---:|---:|
| REM, paced | 92,15% | 74,17% | 93,33% | 14/15 |
| Planar Laplace, paced | 93,67% | 73,33% | 94,17% | 14/15 |
| REM, paced + slack .03 | 93,12% | 82,50% | 91,57% | 13/15 |
| Planar Laplace, paced + slack .03 | 94,44% | 77,50% | 87,41% | 13/15 |

Trung bình toàn phiên tăng nhưng không khắc phục S1.C. Bản planar + slack
cải thiện S1.B từ 89,17% lên 99,17%, đồng thời làm S2.C tụt dưới ngưỡng.
S10.A còn có Hit50=1/2 ở bản này, so với 0/2 của REM + slack, dưới bộ đối thủ
hình học được chọn từ core train. Riêng ca này chỉ có một record thuộc một nhóm
tuyến validation và hai RNG; không phải hai người dùng độc lập. Đây là
counterexample, không phải ước lượng ASR tổng thể đáng tin cậy.
Các Hit ở từng bán kính dùng bộ chọn riêng; không đọc chúng như một CDF chung.

**Quyết định:** giữ làm đối chiếu kernel; chưa chọn làm phiên bản tốt hơn hoặc
chạy confirmation. Gain từ primitive chuẩn không phải contribution mới của ta.
Chưa có attacker chuỗi học lại hoặc expanded screening cho planar ở vòng này;
probe một truy vấn bên dưới đã được học riêng.
Đã kiểm tra lại toàn bộ neo bằng replay sampler và ledger; utility được tính
lại từ đúng tọa độ gửi đến server, kèm kiểm tra đường đi và cửa sổ tấn công.

[Artifact đầy đủ](../../artifacts/benchmarks/research_loop/iteration26_planar_anchor_cases.json).

## Attacker học riêng: truy vấn đầu phiên

Sinh lại đầu ra planar tại 2.000 vị trí công khai, dùng cùng bank kNN/ExtraTrees
và bộ ước lượng theo bán kính của vòng 16; chọn attacker trên 16 nhóm auxiliary.
Chấm 16 chuyến core validation × 2 RNG, nhưng chỉ có **hai nhóm tuyến**;
32 lượt này không phải 32 mẫu độc lập.
Hai bản planar có cùng đầu ra đầu tiên nên dùng chung probe này.

| Neo / attacker được học tương ứng | MAE attacker | Hit100 | Hit500 |
|---|---:|---:|---:|
| REM | 606,64 m | 0/32 | 12/32 |
| Planar Laplace | 404,55 m | 2/32 | 26/32 |
| Raw: đọc tọa độ đầu tiên | 0 m | 32/32 | 32/32 |

MAE thấp hơn và Hit cao hơn đều bất lợi privacy. Cùng .23 m⁻¹ toàn phiên không
đồng nghĩa hai cơ chế bảo vệ tốt như nhau trong phép tấn công này. Số đo REM
được replay từ vòng 16 sau khi xác minh 66 truy vấn core đầu tiên giống hệt nguồn
hiện tại; không lấy model đã học REM để chấm planar.

Probe cho phép nhìn truy vấn đầu tiên của toàn phiên, **không phải** kết quả
S9.A/B/C đã cắt hoặc giới hạn cửa sổ. Có thêm 2.032 lượt sinh một truy vấn
(2.000 train + 32 selection); không đếm chúng thành 2.032 chuyến SUMO hoặc
lượt bảo vệ toàn phiên. [Nguồn và mọi sai số](../../artifacts/benchmarks/research_loop/iteration26_first_query_attack.json).
