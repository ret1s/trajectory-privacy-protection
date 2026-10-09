# Kiểm chứng truy hồi POI theo nhiều mục đích

**Kết luận:** chưa thay cấu hình Geo-I/REM Epoch8/L30. Biến thể gửi đủ bốn template L10 cố định có Recall trung bình 86,34%, thấp hơn L30 92,69%, trong khi byte phản hồi tăng 57,52%. Kết quả âm ổn định trên ba lượt nhiễu. Có mã dịch vụ thử nghiệm để tiếp tục nghiên cứu; chưa bật trong mô hình hiện tại.

## Giả thuyết và ranh giới riêng tư

Giả thuyết của người dùng: thay một truy hồi nearest sâu bằng nhiều truy hồi top-L nhỏ theo các mục đích khác nhau, hợp ứng viên rồi xếp hạng local. Điểm đúng trong lập luận là nhiều ứng viên nearest không bảo đảm chứa top-5 của fastest hoặc detour. Bộ chọn Q hiện tại cũng tối ưu surrogate nearest, chưa trực tiếp tối ưu bốn purpose.

Thí nghiệm giữ nguyên Q, Z, belief, lịch, ngân sách và lần đọc Geo-I đã lưu. Không sinh thêm điểm nhiễu, không fit mô hình bảo vệ, không thay sample hoặc chia lại tập. Tất cả category và template được request tại mọi event, không phụ thuộc câu hỏi thật. Bán kính truy hồi là 1.000 m quanh Q. Template detour dùng **chín đích mẫu công khai**: các trạng thái belief gần lưới 3×3 của bounding box bản đồ tại quantile 0,15/0,5/0,85. Top-L của mỗi category lấy bằng interleave theo hạng, loại trùng và cắt L trên các đích mẫu. Không gửi đích thật.

Do đó `public_detour_bank` là **truy hồi ứng viên theo nhiều đích mẫu**, không phải câu trả lời detour cho đích riêng của người dùng. Đích thật chỉ dùng tại bộ đánh giá/xếp hạng local như benchmark trước. Gửi đích thật hoặc bán kính riêng mà chưa có cơ chế bảo vệ sẽ không giữ được ranh giới S7 hiện tại.

## Lập luận trước benchmark

1. **Request bất biến theo nhu cầu thật.** Với cùng Q và ngữ cảnh công khai, bản tin có dạng `W = F(Q, public_plan)`, không nhận `QuerySpec` thật. Vì vậy thay purpose/category/radius/destination local không đổi request. Đây là mở rộng cùng lập luận hậu xử lý và noninterference có điều kiện, không là bảo đảm mới cho ý định tương quan với tuyến đường, account, click hoặc giờ bật ứng dụng. Không tạo thêm lần đọc riêng tư nên không tự cộng phí Geo-I; payload/response mới vẫn có thể làm thay đổi khả năng suy luận thực nghiệm.
2. **Radius top-L có thể hoàn toàn dư thừa.** Khi cùng Q, category, distance và quy tắc phá hòa, `TopL(distance ≤ r)` là tập con của `TopL(nearest)`. Nó không đem lại POI mới nếu hai query đều dùng cùng L. Muốn radius tạo độ phủ mới phải đổi ngữ nghĩa truy hồi, chẳng hạn vùng công khai rộng hơn hoặc phân bổ theo các ô vùng; chỉ đổi nhãn query không đủ.
3. **Fastest chỉ giúp khi thứ tự ứng viên khác nearest.** Nếu mọi cạnh có cùng tốc độ v, thì `T(Q,p) = D(Q,p)/v`, nên hai thứ tự giống nhau. Graph hiện tại có tốc độ khác nhau nhưng dùng free-flow với cap 8 m/s, không có ùn tắc; mức đa dạng phải được đo, không suy từ tên mục đích.
4. **Điều kiện top-5 chính xác không thay đổi.** Gọi R là top-5 thật theo nhu cầu local và A là hợp POI đã nhận. Với cùng metadata/trạng thái đúng, thứ tự xếp hạng và vị trí local chính xác, đáp án đúng khi và chỉ khi `R ⊆ A`. Nhiều template top-L hữu hạn không tự bảo đảm điều này cho mọi GPS, radius và đích riêng.

Một kiểm tra trên graph đồ chơi có đường chậm/ngắn và đường nhanh/dài cho thấy nearest1 + fastest1 có thể nâng Recall macro top-1 từ 50% lên 100% so với nearest2, với cùng trần hai bản ghi. Đây là **phản ví dụ cho kết luận “đa mục đích luôn vô ích”**, không phải kết quả thực nghiệm thay cho SUMO.

## Giao thức

- Dùng lại cohort synthetic cùng bản đồ đã dùng trong xác nhận L30: 24 TRAIN × 1 draw, 12 SELECTION × 1 draw, 24 TEST × 3 draw; tám chuyến mỗi nhóm. Đây là replay giả thuyết trên cohort hiện có, **không phải xác nhận độc lập trên dữ liệu mới**.
- Đóng băng source/config trước tính score. Chọn trên SELECTION và ghi `selection_freeze.json` trước replay TEST. Giữ mọi biến thể, kể cả thất bại. Không điều chỉnh template theo TEST.
- Primary: current replies only; trung bình category có reference không rỗng, rồi nhóm và bốn purpose. Draw và chuyến nằm trong family. Reference rỗng là N/A. CI bootstrap 10.000 lần theo 24 family TEST, không coi các tick/draw là người dùng độc lập.
- Ba candidate: nearest15 + fastest15; bốn template L10; nearest20 + fastest10 + detour-bank10. Đối chứng nearest L10/L20/L30/L40. L30 đối chiếu candidate 30 bản ghi tối đa/category/Q; L40 đối chiếu candidate 40. Trần bằng nhau không đồng nghĩa byte bằng nhau.
- Gate chọn đã khai báo: macro tăng ≥1 điểm %, mỗi purpose giảm không quá 0,5 điểm %, byte phản hồi ≤1,25× L30, macro không kém nearest có cùng trần. Candidate được chọn còn phải qua xác nhận TEST, CI dưới >0 và gain dương mỗi draw. **Không candidate nào qua SELECTION.** TEST của các candidate là readout mô tả đã khai báo, không là lựa chọn mới.
- Chi phí là compact JSON application payload với cùng trường POI. Mỗi template là request riêng cho mỗi Q; bản ghi trùng vẫn tính byte. Chưa đo HTTP/TLS, latency, CPU server hoặc pin điện thoại. Batch/gộp response có thể đổi chi phí nhưng chưa được thử.

## Kết quả TEST

Recall macro bốn mục đích; byte phản hồi tính trên toàn TEST. MB = 10⁶ byte. Các số dưới dùng phiên bản v2 đã qua kiểm chứng.

| Truy hồi | Recall (%) | Phản hồi (MB) | So byte với L30 | POI duy nhất trung bình/event |
|---|---:|---:|---:|---:|
| Nearest L10 | 83,41 | 317,55 | 0,49× | 110,54 |
| Nearest L20 | 89,71 | 494,71 | 0,76× | 168,28 |
| **Nearest L30 hiện tại** | **92,69** | **648,58** | **1,00×** | **209,02** |
| Nearest L40 | 94,43 | 802,35 | 1,24× | 238,78 |
| Nearest15 + fastest15 | 87,29 | 835,57 | 1,29× | 142,39 |
| **Bốn template L10** | **86,34** | **1.021,61** | **1,58×** | **142,09** |
| Nearest20 + fastest10 + detour-bank10 | 90,97 | 1.129,85 | 1,74× | 188,93 |

| Candidate so với L30 | ΔRecall (điểm %) | CI 95% theo family | Gain ở từng draw (điểm %) |
|---|---:|---|---|
| Nearest15 + fastest15 | −5,39 | [−6,53; −4,26] | −5,13 / −5,60 / −5,46 |
| Bốn template L10 | −6,35 | [−7,46; −5,26] | −6,21 / −6,52 / −6,32 |
| Nearest20 + fastest10 + detour-bank10 | −1,72 | [−2,21; −1,25] | −1,68 / −1,61 / −1,87 |

Số request của L30 là 90.570; bốn template là 362.280. Cả ba candidate có Recall thấp hơn và byte phản hồi cao hơn L30. Chênh lệch không chỉ do request header: byte bảng chỉ tính response. Các CI là từng phép so sánh đã khai báo, chưa điều chỉnh kiểm định nhiều lần; không cần dùng chúng để chọn candidate vì gate SELECTION đã từ chối toàn bộ.

**Chẩn đoán trên toàn bộ 90.570 Q TEST, mọi category:** fastest15 trả 4.890.564 bản ghi, trong đó 4.883.270 đã có trong nearest15 tại cùng Q/category (**99,85% trùng**). Radius10 không thêm bản ghi nào ngoài nearest10. Những số này đo kênh truy hồi public, không dùng GPS thật hay đích thật. Detour mẫu làm tăng một phần độ đa dạng so với L10, nhưng chưa bù được việc giảm độ sâu nearest và tải lặp.

## Quyết định và hướng tiếp theo

Giữ **Geo-I/REM Epoch8/L30** và bốn cách xếp hạng local. Không gọi chúng là bốn query type đã gửi server. Không gộp kết quả privacy Endpoint20 vào thí nghiệm utility này. Trong trình bày, nói rõ L30 cải thiện xác suất thu hồi đáp án trên cohort đã đo, không bảo đảm hoàn toàn mọi purpose.

Lập luận ứng dụng: với dịch vụ tìm POI lân cận, độ gần là tiêu chí hữu ích để thu ứng viên chung; sau đó người dùng có thể ưu tiên thời gian, bán kính hoặc độ đi vòng bằng xếp hạng local. Không diễn đạt thành “mọi mục đích luôn chọn POI gần nhất”: fastest và detour có thể ưu tiên POI xa hơn. Chưa thêm một ràng buộc khoảng cách mới cho tất cả purpose; benchmark vẫn dùng reference đầy đủ đã khai báo, không đổi reference thành pool nearest. Đáp án local là tốt nhất trong ứng viên đã thu hồi; Recall kiểm tra mức đầy đủ của pool đối với top-5 thật.

Hướng đáng nghiên cứu là **ứng viên bổ sung có độ phủ khác nhau**, thay vì bốn query giống ứng viên: tránh radius trùng nearest; đánh giá fastest trên mạng có khác biệt thời gian thực; thiết kế đích/vùng công khai hoặc bảo vệ đích trước khi request detour. Mỗi hướng cần đặc tả ranh giới riêng tư trước, rồi chọn trên development và xác nhận trên cohort mới. Không nới gate hoặc đổi sample để ép bốn template hiện tại thắng.

## Tái lập và kiểm chứng

Mã thử: `benchmark/multi_purpose_retrieval.py`. Chạy `python -m experiments.multi_purpose_retrieval_20261009 --stage all` chỉ với thư mục output/work mới theo quy ước write-once; evidence hiện có không được ghi đè. Kiểm chứng: `python -m experiments.verify_multi_purpose_retrieval_20261009`; chạy diagnostic bằng `python -m experiments.diagnose_multi_purpose_retrieval_20261009`, cũng ghi write-once. Để kiểm chứng lại evidence, giữ nguyên receipt cũ và chạy verifier vào receipt mới theo cùng source/snapshot, không xóa hoặc sửa dữ liệu đã chốt.

Evidence: [protocol](../../artifacts/benchmarks/multi_purpose_retrieval_20261009_v2/protocol.json), [selection freeze](../../artifacts/benchmarks/multi_purpose_retrieval_20261009_v2/selection_freeze.json), [readout](../../artifacts/benchmarks/multi_purpose_retrieval_20261009_v2/readout.json), [validation](../../artifacts/benchmarks/multi_purpose_retrieval_20261009_v2/validation.json), [overlap diagnostic](../../artifacts/benchmarks/multi_purpose_retrieval_20261009_v2/template_overlap.json).

Independent verifier đối chiếu **27.172 event L30** với score/cost đã chốt, tính lại mọi family aggregate, byte tổng và CI. Native oracle tính đường đi theo chiều forward kiểm tra **45 request ở event đầu của ba candidate**, không kiểm tra mọi state. Các source/input hashes, Q/anchor/ledger hashes, split/draw/event inventory được giữ và kiểm tra. Bộ test còn có graph phản ví dụ, invariant nhu cầu local không đổi wire, tham số wire không hợp lệ và selection cho phép không có winner.

**Lịch sử sửa:** v1 giữ đầy đủ protocol, source snapshot và kết quả nhưng không được coi là validated: native oracle phát hiện tie-order detour do dùng reverse-source distance khác số học forward của local ranking khi triệt tiêu gần 0. v2 dùng forward-source cho detour, giữ nguyên hypotheses, gate, dữ liệu, Q và split; chạy lại toàn bộ rồi qua oracle. [Failure receipt v1](../../artifacts/benchmarks/multi_purpose_retrieval_20261009_v1/failure.json). Đây là sửa tính đúng của phép tính, không fine-tune theo score.
