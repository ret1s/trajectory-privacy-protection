# Hai chuyến cùng site: giữ phân phối thay vì chỉ giữ một dự đoán

Vòng 24 bổ sung một đối thủ cho S9.C/S10.C. Kết quả đáng chú ý: với phương pháp
`response_paced`, đối thủ này đoán origin trong 100 m ở **3/24 cặp record–RNG**,
trong khi attacker được bộ chọn cũ chọn cho Hit100 có 0/24. Đây là 12 nhóm và
hai RNG, không phải 24 người dùng độc lập. Không dùng số 0 cũ để kết luận không
có rò rỉ origin.

## Cách suy luận và giới hạn

Mỗi view được chuyển thành feature chỉ từ tọa độ/time được phép thấy. Từ 64
nhóm auxiliary, lấy 64 hàng feature gần nhất và tạo phân phối KDE trên 2.073
điểm lưới công khai. Thử bandwidth 100/250/500 m, có thành phần prior để tránh
khối lượng bằng 0. Với hai view cùng site, so hai cách:

- Pool: trung bình hai phân phối, giữ các vùng khả dĩ của cả hai.
- Product: nhân hai phân phối và chia prior một lần, ưu tiên vùng cùng giải
  thích được hai quan sát mà không đếm prior hai lần.

Đây là **score thực nghiệm gần đúng**: các view có thể tương quan, KDE/priors
không được chứng minh calibrated. Không gọi kết quả là posterior chính xác.
Các hành động gồm mean, MAP, geometric median lặp hữu hạn và điểm lưới tối đa
khối lượng trong từng bán kính. Grid hit decision chỉ tối ưu trên tập điểm lưới.

Chọn cấu hình và hành động theo từng metric trên 16 nhóm auxiliary khác; sau
đó chấm đủ 12 nhóm expanded, hai RNG, cả S9.C/S10.C. Giữ raw và hai cơ chế đã
có shadow phù hợp. Không dùng nhãn expanded để fit/chọn; tập expanded vẫn là
development đã xem trong vòng cải tiến defender.

## Kết quả

Bộ chọn trên toàn bộ bank (cũ + KDE) vẫn chọn các attacker cũ cho cả sáu cặp
method–case. Do đó, readout chính theo bộ chọn đó không đổi. Khi báo cáo các
thành phần đã định riêng, có những khác biệt đáng chú ý:

| Method, case | Bank cũ: MAE / Hit100 | Product KDE: MAE / Hit100 |
|---|---:|---:|
| Raw, S9.C | 195 m / 0% | 950 m / 0% |
| Raw, S10.C | 190 m / 16,67% | 867 m / 0% |
| Paced, S9.C | 590 m / 0% | 782 m / **12,50%** |
| Paced, S10.C | 1.306 m / 0% | 1.653 m / 0% |
| Paced + slack, S9.C | 514 m / 0% | 818 m / 4,17% |
| Paced + slack, S10.C | 1.200 m / 0% | 1.726 m / 0% |

MAE và Hit100 có bộ chọn theo metric; bảng không mặc định hai cột luôn dùng
cùng một estimator. Riêng Paced/S9.C, auxiliary chọn đúng một cấu hình cho mọi
metric trong product bank: bandwidth 500 m, posterior mean. Ba lần Hit100 là:

| Nhóm / RNG | Sai số origin |
|---|---:|
| family-704 / 0 | 71,67 m |
| family-706 / 1 | 92,93 m |
| family-708 / 1 | 7,15 m |

Không chọn lại cấu hình trên ba kết quả này. Có nhiều estimator được kiểm tra,
nên đây là counterexample mô tả, chưa là ước lượng tần suất tổng quát hóa chính
xác. KDE kém raw controls cũ; không quảng bá nó là attacker mạnh hơn toàn diện.
Prior-only trên S9.C có MAE khoảng 2.246 m, Hit500=0%; product dùng quan sát đã
khác prior-only, nhưng cũng chưa thay thế được đối thủ theo đường/ngữ nghĩa.

## Hệ quả cho contributions

Phải giữ đồng thời MAE và các hit radius, cùng raw controls và nhiều attacker.
Attacker có MAE kém vẫn có thể thành công sát mục tiêu ở một phần record. Bộ
chọn trên auxiliary nhỏ cũng có thể bỏ sót một đối thủ hữu ích ở development;
không diễn giải metric của một attacker được chọn thành bảo đảm trước mọi đối thủ.
Chưa có kết quả KDE phù hợp cho output mới của vòng 20/21/23.

Code: `evaluation/site_density.py`. Artifacts: `iteration24_site_density.json`
và `iteration24_site_density_components.json`. Verifier replay toàn bộ 144 dòng
expanded với training/selection tách nhóm và kiểm tra lại bộ chọn theo metric.
