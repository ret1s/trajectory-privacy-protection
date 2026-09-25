# Algorithm improvement loop — checkpoint 24/09/2026

> Kiểm tra mở rộng 25/09: [S9/S10 trên 32 nhóm tuyến mới](endpoint_calendar_results.md), giữ nguyên kế hoạch/attacker đã khóa và thêm lịch gửi công khai trước/sau chuyến. Recall 100% của tập thật + dummy cần đọc cùng privacy và byte; các số lịch sử dưới được giữ riêng.

## Vòng 29–30: thêm nhánh truy vấn công khai theo loại

Đã hoàn tất 30 vòng. [Kết quả và lập luận](category_cover_results.md): plan 30
truy vấn loại–tọa độ, tại 19 tọa độ công khai, đạt Recall 95,67%, 15/15 ca trên
tập phát triển. S1.C 95,57% thay 80,18% của bản K5/slack/cache. Không gọi đây
là cùng K=5 hoặc cùng byte; trung bình utility không tăng có ý nghĩa rõ so với
bản cũ, còn S2 và một số ca S10 giảm.

Plan đã khóa trước khi sinh bốn nhóm 901–904, 88/88 chuyến SUMO hoàn tất.
Năm scenario dùng 32 chuyến, 56 record. Kiểm tra mới ở availability 80% đạt
94,61%, 15/15; ở 95% chỉ qua 11/15. Plan rộng 67 đã khóa từ trước đạt 100% trên
mọi ca/mức availability đã chấm, với byte truyền cao hơn. Không có chứng nhận
phủ toàn danh mục; tám POI còn ngoài hợp top-L tĩnh của plan rộng.

Lập luận privacy mới: network policy không nhận GPS; với cùng vùng/danh mục,
đồng hồ và trạng thái server, thay GPS không đổi transcript. Không suy ra
Hit=0 hoặc bảo vệ metadata; chưa có bộ attacker timing/identity mới. Giữ cả
hai plan và phương pháp Geo-I cũ với đúng giao diện/chi phí của từng nhánh.

Có 3.276 lượt đánh giá dịch vụ tất định mới; không tăng số lượt sinh nhiễu
4.162 đã có. Tổng chuyến SUMO vật lý là 600 sau khi thêm 88 chuyến. Verifier
dựng lại plan và qua 8.532 đối chiếu Dijkstra, 315 dòng tổng hợp, 14 kiểm tra
luồng dữ liệu. Xem `iteration29_*`, `iteration30_*` và dataset
`research_loop_confirmation_v1`; đây là kiểm tra nhóm mới nhỏ trên cùng thành
phố/generator, chưa phải xác nhận cuối cùng của luận văn.

## Checkpoint vòng 28 (giữ để đối chiếu)

**Cấu hình làm việc cho dịch vụ:** paced + slack 0,03 + cache theo khoảng hiệu
lực, K5/L10. Trên workload POI khả dụng mới, Recall trung bình theo 15 ca đạt
**95,60%**, so với **82,07%** của fixed K5. Cache tăng riêng **0,24 điểm phần trăm**
với cùng transcript và chi phí truyền. Vẫn đạt 14/15 ngưỡng utility; S1.C 80,18%.
[Bảng kết quả, thiết lập và phạm vi](live_service_results.md).

Đã hoàn tất 28 vòng. Vòng 27 thêm truy vấn công khai không cải thiện privacy và
vẫn thiếu S1.C; [bảng chi phí](public_supplement_results.md). Vòng 28 chuyển sang
workload availability mô phỏng theo lựa chọn được người dùng giao, giữ riêng
kết quả POI tĩnh trước đó. Đối chứng tải toàn bộ bitmap 419 trạng thái vẫn đạt
100% mà không gửi tọa độ; không nhận ưu thế so với API bulk.

Tổng lượt thực thi stochastic toàn phiên vẫn là **4.162**; vòng 27 và 28 chỉ
replay dịch vụ, không tăng số chuyến SUMO. Vòng 28 có 14.688 lượt đánh giá dịch
vụ, 24.912 hàng record trên chín cấu hình trạng thái. **83 test liên quan** đã
qua, gồm tám test cache/supplement chạy cùng nhau. Có 648 đối chiếu dịch vụ mới
với Dijkstra xuôi; chưa có confirmation độc lập hoặc so mới đầy đủ sáu paper.

Các kết quả 25–26 và probe đầu phiên được giữ trong
[bảng 15 ca](capped_planar_results.md). Riêng core S9.B/S10.A chỉ có một nhóm
validation đủ điều kiện; không suy mẫu số từ tổng nhóm của dataset. Protocol:
[algorithm_improvement_loop.md](algorithm_improvement_loop.md).

## Nguồn và phạm vi

- Khôi phục được SUMO 1.27.1 và OSM mới ngày 20/09. Hash khác nguồn lịch sử;
  benchmark mới tách riêng, không ghép với số cũ.
- Mạng mới: 102.404 trạng thái làn, 110.945 cung, 419 POI.
- Đã chạy 88 chuyến SUMO mới, hoàn tất 88/88; 142 record, có đủ 30 loại A/B/C
  trong toàn bộ dataset. Có dữ liệu cho một loại **không có nghĩa đã đánh giá đủ
  loại đó**. Các vòng 3–9 dùng base/repeat/access_endpoint; vòng 10 đã dùng đủ 15 ca A/B/C của năm scenario, với bộ đối thủ hữu hạn.
- Hai nhóm train (501/502), hai nhóm validation phát triển (601/602).
  Đây là tập dùng trong vòng cải tiến; chưa có kết quả confirmation độc lập.
- Đến vòng 15 có 1.864 lượt thực thi toàn phiên, gồm 640 lượt tạo nguồn attacker
  phụ trợ. Các control và RNG lặp không phải mẫu độc lập mới. Vòng 15 tái sử dụng
  thêm 264 control đã lưu, không đếm chúng thành lượt chạy mới.

## Các vòng đã chạy

| Vòng | Thay đổi / phép kiểm tra | Kết quả | Quyết định |
|---|---|---|---|
| 1 | Gộp trạng thái có cùng danh sách POI trước tối ưu | 54 lượt ghép cặp, transcript giống hệt; 24,10 → 11,11 ms/bước, p95 158,43 → 51,93 ms | Giữ cải tiến tính toán; không gọi là cải thiện privacy |
| 2 | Đối chứng toàn bộ POI tĩnh lưu cục bộ | 1.740 truy vấn giống tham chiếu, Recall 100%, không truy vấn vị trí từ xa | Phải giữ đối chứng này và chốt lại thông tin chỉ máy chủ có |
| 3 | Lõi toàn phiên; tách head60, tail60, cả hai | Cả hai cho Recall 60,55%, phục vụ 83,60%, trễ 60 s | Không chốt cấu hình cắt/buffer60 |
| 4 | Privacy filter dùng ngân sách của nhánh private test | Đọc 13–18 vị trí thay vì 12, Recall vẫn 81,86% | Chưa có lợi ích utility riêng; giữ làm ablation |
| 5 | Di chuyển tới vùng hữu ích trong các vị trí có cùng POI | Progress 83,79%; filter + progress 84,58%, so với nền 81,86% | Có tín hiệu tương tác giữa ngân sách và chuyển động; chưa xác nhận |
| 6 | Cùng tăng L5 → L10 cho mọi phương pháp, vẫn chấm top-5 | Filter + progress 90,57%; nền 89,18%; phản hồi ID khoảng 4,51 kB thay 2,74 kB | Công bố đánh đổi chi phí; không gọi tăng L là contribution thuật toán |
| 7 | Cho phép mất tối đa 0,01 hoặc 0,03 objective theo belief để đi tiếp | Slack .01: L5 84,87%, L10 90,75%; slack .03: L5 84,13%, L10 90,91% | Chưa có người thắng rõ; không chọn riêng metric thuận lợi |

**Vòng 8 — so cùng cận privacy chặt:** lõi cũ cấu hình B=.24 nhưng thực tế có
cận .23. Đã chạy lại filter với tối đa 23 đơn vị .01, giữ cùng cận .23. Khi đó
progress + filter, không slack, đạt L5 **84,42%**, L10 **90,54%**, so với nền
**81,86% / 89,18%**. Hai bản slack chỉ đạt L10 khoảng 89,84–89,85%, nên chưa được
chọn. Bản không slack là ứng viên tiếp tục, không phải cấu hình đã xác nhận.
Phiên thấp nhất của nó ở L10 là 87,63%. S10 MAE 1.736 m vẫn thấp hơn 1.952 m của
nền (đánh đổi privacy), Hit100 vẫn 0/6 dưới bộ đối thủ hữu hạn. Các số vòng 4–7
chỉ so cùng trần cấu hình .24, chưa so bằng cận chặt như vòng 8.

Chi phí của **toàn bộ ứng viên có planner** cao hơn bản quotient đứng yên:
22,48 so với 3,97 ms/bước trên workload toàn phiên vòng 8; p95 40,25 so với
5,22 ms. Không lấy mức tăng tốc 2,17 lần của riêng phép gộp ở vòng 1 để nói toàn
bộ planner nhanh hơn. Giá trị hiện thấy là utility tốt hơn trong chi phí vẫn đo
được; chưa có kết quả trên thiết bị di động.

Các số Recall vòng 3–9 trên là trung bình toàn phiên ở validation phát triển, không phải
kết quả đủ 15 ca A/B/C của năm scenario. Kết quả theo ca mới nằm ở mục dưới. Phiên thấp nhất ở L10/slack .01 là 89,03%;
đây là chẩn đoán đuôi phân phối, **không thay thế ngưỡng 90% theo từng ca** đã chốt.
Cần chạy đủ từng ca trước khi kết luận pass/fail ngưỡng đó.

## Privacy: tín hiệu có thật và giới hạn

Bộ đối thủ mới được chọn trên hai nhóm train rồi áp lên validation. Raw control:
S9 Hit100=100%, S10 Hit100=83,33%; vì vậy phép thử mới phân biệt được dữ liệu thô,
khác vấn đề floor ở phép thử S10 lịch sử. Các cơ chế bảo vệ đều đạt Hit100=0/6 ở
sáu phiên validation này, nhưng chỉ có hai nhóm độc lập và bộ đối thủ hữu hạn.

Không dùng số 0 này để kết luận đã giải quyết endpoint. S9 vẫn có Hit500=50% ở
nhiều biến thể. Progress làm S10 MAE từ 1.952 xuống 1.648 m (bất lợi privacy).
Slack .01 làm S9 MAE từ 680 xuống 464 m nhưng S10 MAE lên 2.122 m. Đây là đánh đổi
phụ thuộc target/metric; chưa có Pareto dominance. Chưa chạy mạnh đủ các đối thủ
học từ shadow, ràng buộc chuyển tiếp đường và gộp nhiều chuyến cùng site.

Các vòng dùng trần .24 mỗi phiên có cận cặp .48. Từ vòng 8, cận chặt .23
mỗi phiên cho cận cặp .46. Không gắn nhãn .23 cho cả cá nhân hoặc cả cặp lặp. Thời điểm mở/đóng phiên và các
telemetry khác cũng chưa thuộc bảo đảm tọa độ với đồng hồ cố định.

## Lập luận contributions hiện có

1. **Tối ưu tính toán có tính chất kiểm tra được:** gộp theo chữ ký dịch vụ và
   chọn đúng đại diện theo tie rule; giữ nguyên bộ chọn ban đầu và phân phối
   transcript. Có bằng chứng tốc độ trên mạng thành phố, chưa phải số đo điện thoại.
2. **Thiết kế tiến về mục tiêu với ràng buộc utility:** đổi vị trí trong cùng chữ
   ký POI để tránh kẹt; bản slack nới giới hạn objective một cách định lượng.
   Ràng buộc này áp dụng cho belief gần đúng, không phải bảo đảm Recall thật.
3. **Tích hợp bộ lọc ngân sách với bộ chọn có thể di chuyển:** có lợi ích tương
   tác trong ablation, nhưng quản lý ngân sách predictive không phải ý tưởng mới.
   [Lập luận riêng của implementation](predictive_filter_argument.md).
4. **Đánh giá toàn phiên:** đo cả truy vấn không được phục vụ và độ trễ thực;
   bác bỏ cấu hình biên không đạt yêu cầu thay vì chỉ chấm phần còn công bố.

Các điểm 2–4 vẫn là contributions ứng viên, cần dữ liệu và đối chứng mạnh hơn.
Không nhận phần Geo-I, predictive reuse, greedy coverage hay truncation là phát minh.
Predictive budget management đã có trong [PETS 2014](https://arxiv.org/abs/1311.4008),
và tối ưu utility dưới Geo-I đã có trong [CCS 2014](https://arxiv.org/abs/1402.5029).
[Privacy-Aware Remapping 2024](https://www.privateer-project.eu/wp-content/uploads/2024/02/SAC24-A_Privacy_Aware_Remapping_Mechanism_for_Location_Data.pdf)
cũng nghiên cứu gộp ô vị trí để thay đổi privacy/quality. Gộp của ta hiện dựa trên
**kết quả dịch vụ và miền đi tới được**, không được đánh đồng với việc tiền xử lý
GPS thật bằng cách lượng tử hóa. Chưa đủ khảo sát để khẳng định ưu tiên phát minh.

## Điểm phải chốt về dịch vụ

Nếu mục tiêu chỉ là tìm POI tĩnh công khai, đối chứng local cache có lý do tồn tại
và không thể loại chỉ vì nó không gửi dummy. Trong benchmark, bản đồ và danh mục
POI đã là tài nguyên của bộ chọn. Cache có chi phí: khoảng 12,7 MB chỉ mục, 20,1 MB
SUMO XML và 57,6 kB danh mục POI (các thành phần chưa nén, không phải payload mạng).

Câu hỏi đã gửi: dịch vụ thực cần thông tin POI cập nhật, tuyến đường/ETA theo giao
thông, hay chỉ POI/khoảng cách tĩnh? Chưa đổi workload để tránh làm đẹp kết quả.
Nếu mục đích là bắt buộc công bố telemetry cho một tác vụ khác, cần nêu tác vụ đó
và utility tương ứng; “bắt buộc gửi” đơn thuần không đủ giải thích giá trị dịch vụ.

## Phần còn thiếu trước khi chốt

- Thông tin chỉ máy chủ có và workload dịch vụ; kiểm soát cập nhật/cache nếu có.
- Tăng số nhóm cho các A/B/C đã chạy; sửa các ca utility chưa đạt; kiểm tra
  các giả định đồng hồ công bố và quyền quan sát bằng workload dịch vụ đã chốt.
- Attacker mạnh, lặp theo site; đủ số nhóm để ước lượng bất định đáng tin.
- So sánh cùng giao diện/budget/K/độ sâu phản hồi; phân biệt paper nguyên bản,
  bản thích nghi và trường hợp không áp dụng.
- Confirmation mới sau khi khóa phương pháp/metric, không dùng lại validation
  đã xem để tuyên bố tổng quát hóa.

## Tái lập

Python thực nghiệm: `/tmp/trajectory-research-env/bin/python` (3.11), NumPy/SciPy,
scikit-learn 1.7.1, SUMO và sumo-data 1.27.1. Môi trường `/tmp` cần tạo lại nếu bị
xóa; dữ liệu OSM/net và cache cũng không mặc định nằm trong Git.

Các runner ở `experiments/research_loop_*.py`; mỗi vòng ghi artifact riêng dưới
`artifacts/benchmarks/research_loop/`. Các runner từ chối ghi đè vòng hoàn tất.
`python -m experiments.verify_research_loop` kiểm tra phiên hoàn tất, chỉ số mẫu,
ngân sách, kế toán công bố và control trùng giữa các vòng. Đây không phải chứng
nhận hiệu quả privacy. Đã chạy gộp và qua **45 tests** liên quan đến các module
của các vòng cải tiến ban đầu, gồm guard, pacing, biên dữ liệu attacker, objective top-k/top-L,
oracle phủ POI và planning slack kết hợp pacing.

## Vòng 9–11: kiểm tra mạnh hơn và những điểm chưa đạt

**Vòng 9 — lõi mới nhất.** Đã đưa switching belief (dừng/di chuyển) hiện có
vào so sánh. Ở workload 20 s toàn phiên, lõi switching đạt L10 89,18%; ứng viên
filter + progress một mode đạt 90,54%; tích hợp cả switching đạt 89,81%. Không
có bằng chứng rằng cứ thêm switching là tốt hơn. Các kết quả vẫn là phát triển.

**Vòng 10 — đủ cửa sổ A/B/C, không reset ở mỗi record.** Sinh một luồng cơ chế
xuyên suốt từng chuyến, rồi trích chính xác các event mà record cho phép. Đồng hồ
thử là hợp lịch nền 20 s và thời điểm truy vấn của các record; nó được cố định
cho mọi phương pháp. Bảo đảm chỉ điều kiện trên đồng hồ này, không che giấu thời
gian yêu cầu dịch vụ hoặc scenario membership. Lịch khác vòng 9 nên không ghép
hai bộ số. Đã chạy 264 lượt (33 phiên × 2 RNG × 4 phương pháp), tạo 448 dòng ca.

| Ca | Số nhóm validation | Switching cũ: Recall L10 | Filter + progress: Recall L10 | Qua 90%? |
|---|---:|---:|---:|:---:|
| S1.A | 2 | 94.17% | 90.00% | Có |
| S1.B | 2 | 89.17% | 86.67% | Chưa |
| S1.C | 2 | 73.33% | 74.17% | Chưa |
| S2.A | 2 | 100.00% | 100.00% | Có |
| S2.B | 2 | 100.00% | 100.00% | Có |
| S2.C | 2 | 99.17% | 100.00% | Có |
| S3.A | 2 | 88.33% | 89.17% | Chưa |
| S3.B | 2 | 88.33% | 89.17% | Chưa |
| S3.C | 2 | 88.33% | 89.58% | Chưa |
| S9.A | 2 | 88.76% | 88.22% | Chưa |
| S9.B | 1 | 90.86% | 90.86% | Có |
| S9.C | 2 | 87.34% | 88.28% | Chưa |
| S10.A | 1 | 94.25% | 95.04% | Có |
| S10.B | 2 | 98.33% | 99.10% | Có |
| S10.C | 2 | 89.83% | 90.65% | Có |

Ứng viên qua **8/15 ca**, so với 7/15 của switching cũ, còn thiếu bảy ca.
S1.C là điểm yếu rõ nhất. S1.A giảm từ 94,17% xuống 90%; không có ưu thế trên
mọi ca. Chỉ hai nhóm, riêng S9.B/S10.A có một nhóm; chưa đủ để suy rộng.

Có 11/946 event validation mỗi lần lặp không có POI tham chiếu ở bất kỳ category
nào. Bản tổng hợp đầu sinh NaN; bản checked giữ nguyên transcript/sai số privacy,
đổi thành null và công khai số event rỗng theo quy ước sẵn có của
`evaluation/retrieval_frontier.py`. Mọi phương pháp có cùng eligibility, xác định
từ tham chiếu. Recall trên là trên các truy vấn có tham chiếu; event rỗng không
được cho điểm 1. Ngưỡng 0,9 dùng dung sai số thực 1e-12, không nới yêu cầu dịch vụ.
Bản đầu và source trước sửa vẫn lưu để kiểm toán; dùng `iteration10_cases_checked.json`.

**Vòng 11 — học suy ngược tập truy vấn đầu tiên.** Sinh 2.000 mẫu trên prior
đường công khai và cơ chế thật để huấn luyện kNN/loss-aware neighbors/ExtraTrees.
Không dùng nhãn endpoint validation để huấn luyện. Trên sáu phiên validation toàn
luồng ở vòng 9, kNN15 định trước đạt S9 MAE **287 m**, Hit100 **3/6**, Hit200 **4/6**.
Tất cả kết quả của 16 bộ ước lượng học được được lưu, không chỉ kNN15.

Bộ chọn attacker theo hai nhóm train vẫn chọn estimator hình học cũ, nên metric
“train-selected” của nó còn MAE 680 m / Hit100=0. Đây là hạn chế tổng quát hóa
của việc **chọn attacker**, không phải bằng chứng rằng các attacker khác không
đoán được. kNN15 là một phép thử định trước, nhưng việc nêu nó ở đây là chẩn đoán
phát triển; không đổi nó thành attacker confirmation sau khi đã xem validation.

Tập truy vấn đầu tiên giống nhau ở mọi biến thể có bảo vệ trong vòng 9: planner
chưa tác động vào bước đầu. Vì vậy thêm planner chưa giải quyết được điểm yếu S9
này. S10 vẫn cần attacker học chuỗi, ràng buộc đường có hướng và nhiều chuyến cùng
site. Bộ gộp hình học S9.C/S10.C của vòng 10 chỉ là bước đầu.

**Quyết định:** chưa chọn phiên bản cuối cùng, chưa dùng tập confirmation mới.
Vẫn cần chốt thông tin chỉ máy chủ có trước khi kết luận giá trị dịch vụ, vì local cache
đang giải chính xác workload POI tĩnh mà không công bố vị trí. Sau đó kiểm tra
phân bổ ngân sách đầu phiên để xử lý điểm yếu S9, sự cạn ngân sách trước truy vấn
POI hiếm, và đo lại cả utility lẫn endpoint attacks. Không tiếp tục sửa workload
hoặc tăng tải phản hồi chỉ để biến các ca chưa đạt thành đạt.

Định vị so với sáu đối chứng và phạm vi bảo đảm:
[contribution_positioning_20260924.md](contribution_positioning_20260924.md).

## Vòng 12–13: bảo vệ đầu phiên và phân bổ lần đọc theo thời gian

**Vòng 12 — giảm epsilon ở đầu phiên.** Thử epsilon release/test bằng một phần
tư (.0025 thay .01) ở lần đầu hoặc 60 giây đầu; giữ cận toàn phiên không quá .23.
Lõi thực tế là Road Exponential Mechanism (REM) với private reuse test, không phải
lấy nhiễu Laplace hai chiều. Belief sử dụng đúng emission của từng pha. Đã chạy
60 lượt toàn phiên; công thức và giới hạn ở [origin_guard_argument.md](origin_guard_argument.md).

Attacker phải được huấn luyện lại trên cơ chế mới: thêm 2.000 mẫu đầu phiên.
kNN15 cố định chuyển từ MAE 287 m / Hit100 3/6 sang MAE 1.421 m / Hit100 0/6;
nhưng ExtraTrees vẫn đoán trong 100 m ở 1/6 phiên. Đây là thách thức riêng từng
attacker, không phải xác nhận privacy. Guard lần đầu làm Recall L10 toàn phiên
giảm từ 90,54% xuống **86,17%**; guard 60 giây còn **84,82%**. Chưa chọn hai bản này.

**Vòng 13 — đọc GPS tối đa một lần mỗi 60 giây công khai.** Vẫn trả mọi truy vấn
đã lên lịch. Giữa hai lần đọc, planner chỉ sử dụng trạng thái đã bảo vệ; thời gian
trôi qua không làm ngân sách được cấp lại. Đã chạy 330 lượt toàn phiên trên đúng
33 phiên nguồn, hai RNG và năm phương pháp; trích 560 dòng ca A/B/C.

| Cấu hình | Ca có Recall L10 ≥ 90% | S1.C | S3.A | S9.A | S10.C |
|---|---:|---:|---:|---:|---:|
| Filter + progress | 8/15 | 74,17% | 89,17% | 88,22% | 90,65% |
| Thêm khoảng cách đọc 60 s | **14/15** | **75,00%** | **90,50%** | **91,45%** | **93,05%** |
| Thêm cả guard lần đầu | 1/15 | 75,00% | 78,29% | 78,39% | 83,21% |

Đây là mức cải thiện utility đáng tiếp tục, vẫn chỉ trên hai nhóm validation
phát triển, có ca chỉ một nhóm. Bản giãn lần đọc không đổi truy vấn đầu tiên,
nên vẫn có điểm yếu S9 đã thấy ở vòng 11. Không gộp bảng này với workload 20 s
thuần của vòng 12; không gọi 14/15 là đã bảo vệ đủ năm scenario.

Chẩn đoán S1.C cho thấy các POI nhà hàng/café gần vị trí cần phục vụ bị thiếu.
Ở hai mẫu kiểm tra, một số truy vấn giả cách vị trí thật 1–4 km, dù chưa hết
ngân sách. Các site nằm trong thành phần liên thông mạnh lớn; không có bằng
chứng rằng thiếu kết nối mạng là nguyên nhân. Khoảng cách tới các mục tiêu có ích
và giới hạn tốc độ của query tracks là giả thuyết cần kiểm tra tiếp. Không tăng
tốc độ, cho phép nhảy vị trí hoặc loại ca khó chỉ để qua ngưỡng.

**Vòng 14 — nguồn attacker độc lập với các nhóm chính.** Đã mô phỏng 80 nhóm
phụ trợ mới, mỗi nhóm gồm một chuyến thường và một chuyến dừng/quay lại:
160/160 chuyến SUMO hoàn tất, 170.687 điểm FCD. Sau đó chạy bốn biến thể cơ chế,
tổng 640 lượt toàn phiên. 64 nhóm huấn luyện, 16 nhóm chọn attacker. Nguồn phụ trợ
dựa trên bản đồ công khai, không dùng endpoint của tập chính để lên tuyến.

Attacker dùng kNN, ExtraTrees trực tiếp/phần dư, các quyết định tối ưu MAE/Hit
trên láng giềng và phép chiếu lên đường. Đầu vào chỉ gồm các tọa độ và thời gian
trong cửa sổ được công bố. S1 chọn trên snapshot; S9/S10 chọn theo cửa sổ che biên,
prefix hoặc hai chuyến liên kết. Đây là attacker học đặc trưng chuỗi, chưa phải
HMM/Viterbi với chuyển tiếp đường có hướng hoặc mô hình ngữ nghĩa POI.

Các nguồn phụ trợ chỉ dành cho attacker: **không biến hai nhóm validation đã
xem nhiều lần thành tập confirmation độc lập cho defender**. Đồng hồ nguồn phụ
trợ 20 s cũng khác đồng hồ có bổ sung event của các ca chính; cần công khai lệch
phân phối này khi diễn giải kết quả.

Kết quả vòng 14 cho thấy việc liên kết nhiều chuyến thực sự quan trọng. Ở S9.C,
attacker ExtraTrees phần dư, chiếu lên đường rồi gộp hai chuyến, được chọn trên
16 nhóm phụ trợ, đạt MAE **258 m**, Hit100 **25%** đối với filter + progress.
Với pacing, attacker được chọn có MAE **308 m**, Hit100 0 nhưng Hit200 25%.
Một attacker hình học khác trong bộ đã đạt Hit100 25% trên chính pacing; đây là
chẩn đoán sau khi xem kết quả, không được gọi là lựa chọn ngoài test. Các số đều
dựa trên rất ít nhóm chính và không phải xác suất rủi ro đã được xác nhận.

S1.B của pacing cũng có Hit100 25% dưới attacker được chọn. Không có cơ sở nói
14 ca qua ngưỡng utility đồng nghĩa 14 ca đã được bảo vệ. Trên các cửa sổ che biên,
raw geometric control có Hit100 0 ở S9.A/C và cả ba ca S10; riêng S9.A/C có
Hit200 100%. Vì vậy phải xem MAE và nhiều bán kính, đồng thời tăng cường raw
control, thay vì dùng Hit100 0 của protected làm bằng chứng duy nhất.

Mỗi bán kính chọn một attacker riêng trên nguồn phụ trợ. Các Hit50/100/200/500
không phải bốn điểm của cùng một CDF; trên tập chính nhỏ có thể xuất hiện Hit100
lớn hơn Hit200. Artifact lưu tên attacker của từng metric và sai số của mọi
estimator, để đọc đúng khác biệt do chọn attacker.

Vòng 14 chưa thay thế phép học suy ngược lần đầu của vòng 11. Ví dụ, bộ học đặc
trưng chuỗi trên auxiliary có thể yếu hơn kNN học riêng từ 2.000 vị trí ngẫu nhiên
ở bước đầu. Giữ cả hai thách thức; không bỏ phản ví dụ cũ chỉ vì bộ mới cho số đẹp.

**Đã tăng cường raw control:** huấn luyện cùng loại bộ học endpoint trên các
quỹ đạo raw của 64 nhóm phụ trợ, chọn trên 16 nhóm còn lại. Không chỉ cho protected
gặp learner mà raw chỉ gặp đối thủ hình học. Kết quả riêng lưu ở
`iteration14_raw_sequence_audit.json`. Raw S10.A có MAE 168 m / Hit200 100%; S10.C
có MAE 402 m / Hit500 100%. Riêng S10.B còn MAE 1.033 m, Hit100 và Hit500 đều 0
với các attacker được chọn; đó là phép thử còn yếu hoặc thiếu thông tin, chưa
thể dùng để chứng nhận bảo vệ S10.B. Thêm learner vào bộ chọn cũng có thể đổi
attacker và làm kết quả trên tập chính giảm, do lệch phân phối và mẫu nhỏ.

## Vòng 15: tối ưu đúng phản hồi đã được phép dùng

Giữ K=5, mỗi query nhận top-L=10, và chấm phục hồi top-k=5 như trước. Thay đổi
duy nhất ở objective là tối ưu hợp top-10 thực nhận; trọng số POI vẫn là xác suất
thuộc top-5 tham chiếu theo belief đã bảo vệ. Không tăng L, đổi mẫu số hoặc ngưỡng.
Xem [công thức và kiểm tra](response_aware_objective.md).

Đã chạy 198 lượt mới, tái sử dụng 264 control của vòng 13; tổng 784 dòng ca,
trong đó 336 dòng thuộc ba cấu hình mới. Kết quả validation phát triển:

| Cấu hình | Ca qua 90% L10 | S1.C | S3.A | S9.A | S10.C |
|---|---:|---:|---:|---:|---:|
| Filter + progress cũ | 8/15 | 74,17% | 89,17% | 88,22% | 90,65% |
| Objective theo phản hồi top-10 | **14/15** | **75,00%** | **90,33%** | **92,07%** | **92,59%** |
| Objective top-10 + pacing | **14/15** | 74,17% | 91,17% | 91,43% | 93,32% |
| Objective top-10 + pacing + guard | 4/15 | 73,33% | 80,10% | 85,49% | 88,91% |

Hai cách sửa khác nhau đều nâng utility từ 8 lên 14 ca, nhưng ghép lại chưa làm
S1.C đạt. Bản guarded có phục hồi một phần utility so với 1/15 ở vòng 13, vẫn
chưa đạt yêu cầu. Không chọn riêng S3/S9 tăng để bỏ qua S1.C giảm.

**Privacy của các output mới chưa được xác nhận bằng attacker học phù hợp cơ
chế mới.** Vòng 15 mới có bộ hình học cùng quy trình chọn ngoài validation;
không chuyển số privacy của vòng 14 sang các cấu hình mới. Giữ các ứng viên để
so sánh tiếp, chưa khóa người thắng. Cùng K/L là cùng giới hạn phản hồi, không
đồng nghĩa số byte thực nhận bằng nhau. Runner lưu số POI trả về từng query;
runtime ở vòng này là số đo phát triển, chưa phải phép đo hiệu năng cô lập.

Sau 15 vòng, ba điểm quyết định vẫn mở: workload cần máy chủ, nguyên nhân và
cách sửa S1.C trên thêm nhóm phát triển, và attacker endpoint đủ mạnh (đặc biệt
S10.B). Chưa chạy confirmation mới hoặc đưa ra kết luận đạt yêu cầu luận văn.

## Vòng 16–17: kiểm tra cấu hình mới và xác định giới hạn của planner

**Attacker đã được huấn luyện lại.** Thêm 320 lượt cơ chế trên 80 nhóm phụ trợ
cho hai bản objective top-10, vẫn tách 64 nhóm fit / 16 nhóm chọn attacker. Đánh
giá đúng A/B/C và riêng toàn phiên. Đây là việc hoàn thiện phép thử cho output
đã thay đổi, không phải một tập confirmation mới cho defender.

Thêm 2.000 mẫu public-road đầu phiên cho bộ chọn top-10. So ghép cặp trên cùng
16 phiên validation nguồn, hai RNG, vẫn chỉ hai nhóm:

| Đầu phiên S9 | Pacing objective cũ | Pacing objective top-10 |
|---|---:|---:|
| MAE của attacker chọn trên auxiliary | 540 m | 607 m |
| Hit100 của attacker chọn trên auxiliary | 12,5% | 0% |
| MAE của kNN15 cố định | 484 m | 525 m |
| Hit100 của kNN15 cố định | 9,375% | 0% |

Số lớn hơn ở MAE là khó suy luận hơn. Tuy nhiên kNN5 cố định vẫn có Hit100 12,5%
trên bản top-10. Không kết luận rủi ro bằng 0 và không so trực tiếp bảng này với
MAE 287 m ở vòng 11, vốn dùng tập phiên/RNG khác. Bộ học chuỗi trên toàn phiên
của top-10+pacing có S9 MAE 557 m / Hit100 3,125%, cho thấy quan sát thêm lịch sử
có thể thay đổi kết quả. Các attacker khác nhau phải được giữ trong cùng audit.

Ở A/B/C, bản top-10+pacing có S9.C MAE 664 m / Hit200 25%. Các ca S10 có Hit100
0 dưới bộ chọn hiện tại, nhưng giới hạn raw-control S10.B vẫn còn; chưa chứng
nhận S10.B bằng các số này. Chi tiết mọi estimator nằm trong artifacts, không
chỉ những estimator có số thuận lợi.

**Oracle S1.C.** Đã replay 24 prefix, kiểm tra trùng output đã lưu và giải MILP
để tìm Recall thật tốt nhất trong miền mà mỗi track có thể đi tới ở bước cuối.
Với pacing cũ, cả Recall thật và oracle validation đều 75%, dù các mục tiêu toàn
mạng từ belief có thể cho 95%. Chỉ thay bộ chọn tại bước cuối không đủ; cần sửa
chuyển động từ trước. Đây là giới hạn của lịch sử track hiện tại, không phải bất
khả thi của toàn bài toán. Xem [chẩn đoán planner](planning_failure_diagnosis.md).

**Vòng 17 — chấp nhận giảm tối đa .03 objective để tiến về mục tiêu.** Chạy
66 lượt mới, tái sử dụng 198 control. S1.C tăng **74,17% → 82,50%**, S3.A tăng
**91,17% → 94,10%**, nhưng S1.B giảm **91,67% → 89,17%**. Tổng chỉ 13/15 ca qua
ngưỡng, so với 14/15; chưa có người thắng trên mọi ca. Đây vẫn là ứng viên có
đánh đổi đáng phân tích, không bị xóa vì làm một metric xấu đi. Ngân sách, K, L,
ngưỡng và constraint đường đều giữ nguyên. Cận .03 chỉ áp dụng objective theo
belief, không phải Recall thật.

## Vòng 18 hoàn tất: mở rộng dữ liệu trước khi chỉnh tiếp

Đã tạo thêm **12 nhóm, 264/264 chuyến SUMO hoàn tất**, 154.045 điểm FCD và 415
record. Có 173 record cho năm scenario đang nghiên cứu. S1.C có 12 nhóm; S10.B
có 8 nhóm đủ điều kiện; S3.B có 11, S9.B có 10. Giữ rõ các trường hợp không đủ
điều kiện, không tự điền thành record hợp lệ.

Danh sách năm cấu hình đã khóa trước khi chấm gồm raw và bốn ứng viên: pacing
cũ, objective top-10 unpaced, objective top-10+pacing, và bản có slack .03.
Đã chạy 1.020 lượt toàn phiên, chấm 1.730 hàng case. Giữ các lựa chọn attacker
từ dữ liệu cũ khi screening hữu hạn. Thêm 160 lượt auxiliary cho slack và học
bộ đối thủ phù hợp, rồi chấm cả bốn ứng viên bằng các learner fit/chọn ngoài
12 nhóm mới. Dữ liệu có nhãn **expanded_development**, chưa phải confirmation
cuối cùng; không tuyên bố site/map mới độc lập chỉ từ seed mới.

Tất cả đạt 14/15 ngưỡng utility trung bình theo ca. S1.C lần lượt 75,69%, 71,81%,
76,25% và 78,75%. Thêm slack giúp utility ở một số ca nhưng làm attacker đoán
chính xác hơn ở S1.A/S3.A. Các khoảng bootstrap ghép cặp theo nhóm cũng chưa
chứng minh gain S1.C của slack ổn định. Xem [bảng và phạm vi thống kê](expanded_development_results.md).

S10.B raw MAE khoảng 1.367 m, Hit500 12,5%; chưa có phép thử nhạy để kết luận
bảo vệ tốt. [Phân tích prefix](s10_prefix_identifiability.md) tách sự mơ hồ vốn
có trước ngã rẽ khỏi tác dụng riêng của defender.

## Vòng 19: lịch đọc giãn dần theo ngân sách còn lại

Giữ nguyên epsilon và cap; thời gian tối thiểu giữa hai lần đọc là
`60 × tổng đơn vị / đơn vị còn lại`. Quyết định chỉ dùng đồng hồ và lịch sử nhánh
đã bảo vệ, không nhìn trước đích hoặc thời lượng chuyến. Có hai ablation với và
không có slack .03. Ba test mới và bốn test liên quan đã qua.

Đã chạy 132 lượt mới trên tập core và replay đúng 198 control. Bản không slack
có S1.C 73,33% (control 74,17%); bản có slack 78,33% (control 82,50%). Chưa chọn
thay thế bản cũ. Do S1.C core chỉ ở khoảng 530/643 giây, đã kiểm tra tiếp tất
cả 12 nguồn S1.C mở rộng, hai RNG: 48 lượt mới và 48 control replay, giữ nguyên
cấu hình và đồng hồ toàn phiên. Đây là chẩn đoán chuyến dài, không phải
confirmation hoặc kết quả privacy học lại cho hai output mới.

Trên các nguồn mở rộng, lịch mới giảm tuổi của lần đọc GPS gần nhất tại mẫu từ
199,83 xuống 98,38 giây. Không slack: Recall S1.C 76,25% → 74,44%. Có slack:
78,75% → 80,83%, nhưng Recall toàn phiên trên chính các nguồn này 87,53% →
87,31%. Khoảng bootstrap ghép cặp 95% cho gain S1.C là [−1,94; 7,22] điểm phần
trăm; chưa có bằng chứng tăng ổn định. Kết quả âm cũng được giữ nguyên.

Tổng tới vòng 19: **3.610 lượt thực thi toàn phiên** (gồm các lượt control đã
thật sự chạy, không đếm replay thành chạy mới), trên nguồn core, auxiliary và
expanded tách biệt. Số này không phải cỡ mẫu độc lập; các nguồn SUMO có tổng
512 chuyến vật lý hoàn tất. Chưa có confirmation site/map độc lập hoặc một
ứng viên đủ mọi tiêu chí. Bước tiếp theo cần tập trung planner và hợp đồng dịch
vụ, không suy ra rằng cứ tăng số vòng tìm tham số sẽ tạo được contribution.

## Vòng 20–22: dự báo, lookahead và đối chứng không dùng GPS

**Vòng 20:** học generator chuyển động trên 64 nhóm auxiliary, chọn smoothing
bằng NLL trên 16 nhóm khác. NLL 2,5297 → 2,1961; không dùng nhãn core/expanded
để fit/chọn. Hai bản mới giữ nguyên neo/ledger, 132 lượt mới trên đủ ca core;
cả hai vẫn 14/15. S1.C bản không slack 74,17% → 77,50%, có slack 82,50% → 78,33%.
[Thuật toán và ablation](auxiliary_mobility_argument.md).

**Vòng 21:** planner tối ưu phản hồi hiện tại và sau 120 giây, dự báo từ belief
đã bảo vệ. Ba bản chốt trước, 24 lượt toàn phiên mới trên bốn nguồn S1.C. Bản
forecast/slack đạt S1.C validation 80,83%, vẫn dưới 90%; Recall toàn phiên cũng
thấp hơn control slack. Chưa có matched learned attacks cho vòng 20–21.
[Cơ chế và kết quả](lookahead_service_planning.md). MPC/location forecasting đã
có trong nghiên cứu 2023–2025, không nhận đây là ý tưởng mới.

**Vòng 22:** fixed public queries không dùng GPS, đủ 173 record mở rộng. K5
đạt Recall toàn phiên 81,77%, S1.C 81,94%, 0/15 ngưỡng; K12 đạt 96,54%, 15/15,
với 2,4 lần số tọa độ/slots phản hồi. Bản thích nghi K5/slack hơn control K5 ở
14 ca (khoảng 10,40–22,08 điểm phần trăm), nhưng kém 3,19 điểm ở S1.C. Bootstrap
chỉ là thăm dò trên tập phát triển đã xem. Không có privacy dominance với control
không đọc GPS. Local cache vẫn là đối chứng chính xác ở workload POI tĩnh.

Tới đây có **3.766 lượt chạy cơ chế toàn phiên**, không đếm control replay thành
lượt mới; thêm 510 phép đánh giá protocol fixed-query tất định được ghi riêng.
Nguồn SUMO vẫn là 512 chuyến hoàn tất. **57 tests** liên quan đã qua; verifier
kiểm tra provenance, ngân sách, neo ghép cặp, các view và phép chấm service. Các
kiểm tra này không chứng nhận hiệu quả privacy hoặc mức sẵn sàng luận văn.

## Vòng 23–24: coverage công khai và tấn công site lặp

**Vòng 23:** giữ tổng K5, chia hai truy vấn công khai cố định và ba truy vấn
thích nghi để phủ phần POI còn thiếu. S1.C core đạt 90% ở cả hai bản mới, nhưng
S1.A giảm 94,17% → 86,67%. Không slack chỉ đạt 7/15 ngưỡng; có slack đạt 13/15,
vẫn thiếu S1.A/B. Recall toàn phiên cũng giảm. Chưa chọn tiếp tục hoặc đưa số
S1.C riêng lẻ thành claim. [Cơ chế, cận và đủ bảng 15 ca](public_backbone_argument.md).

**Vòng 24:** KDE kết hợp hai view cùng site, chia prior để tránh đếm lặp. Fit
trên 64 nhóm auxiliary, chọn trên 16 nhóm, chấm 12 nhóm expanded cho hai cơ chế
cũ và raw. Bank gộp vẫn chọn attacker cũ; kết quả chính không đổi. Trong các
thành phần chốt trước, product KDE/mean 500 m đạt S9.C Hit100 **3/24** cho paced,
so với 0/24 của attacker được chọn trước, dù MAE kém hơn. Đó là counterexample
cho diễn giải quá mạnh từ Hit100=0, không chứng minh KDE mạnh hơn mọi mặt.
[Chi tiết và raw controls](repeated_site_density_attack.md).

Tới vòng 24: **3.898 lượt chạy cơ chế toàn phiên**, cộng 510 phép đánh giá
fixed-query service tất định đã ghi riêng. Không sinh thêm nguồn SUMO, vẫn 512
chuyến vật lý. **63 tests** liên quan đã qua; verifier replay toàn bộ 144 dòng
KDE mở rộng, kiểm tra các track/Recall/neo/ngân sách và control của vòng 23.
Giới hạn về hợp đồng dịch vụ, confirmation độc lập và đối chứng paper vẫn mở.
