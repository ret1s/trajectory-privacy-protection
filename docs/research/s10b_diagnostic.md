> **Phạm vi hiện tại:** S10 chỉ gồm A/C; xem [kết quả và lập luận đã cập nhật](active_scope_results.md). Nội dung dưới lưu kết quả/chẩn đoán của phiên bản trước; không dùng số tổng hợp A/B/C cũ thay cho phạm vi hiện tại.

# S10.B: chẩn đoán và quyết định phạm vi

Cập nhật 25/09/2026. **Tạm dừng phát triển S10.B như một đóng góp endpoint riêng; chuyển hướng này sang S6. Phần kết luận hiệu quả S10 hiện tập trung A/C.** Giữ nguyên dataset, các hàng B và trung bình A/B/C đã công bố. Đây là quyết định sau khi xem kết quả, không phải thay đổi một benchmark đã khóa để nhận điểm tốt hơn.

## 1. Vướng mắc nằm ở đặc tả nào?

S10.B cho đối thủ xem tiền tố đến trước một chỗ phân nhánh, rồi yêu cầu đoán điểm cuối bị giấu. Không cấp thời gian còn lại, tổng quãng đường chuyến hay danh sách hai đích thật.

Ví dụ: xe đi tới một giao lộ. Từ đây xe còn chạy vài phút và có nhiều đích hợp lệ. Cho biết “chuyến đã kết thúc” nhưng không cung cấp thêm dữ liệu về phần sau giao lộ không giúp xác định xe đã dừng ở đâu.

**Trong bộ sinh hiện tại, S6.A và S10.B là cùng một bài toán thống kê:** cùng chuyến, cùng chỉ số quan sát, cùng nhãn đích và cùng chính sách dữ liệu. Khác biệt nằm ở nhãn diễn giải `online_destination` / `offline_hidden_endpoint`. Việc đặt tên thời điểm khác nhau chưa tạo ra hai phép thử độc lập.

| Tập dữ liệu | S10.B có dữ liệu tác vụ trùng S6.A |
|---|---:|
| Phát triển ban đầu | 4/4 |
| Phát triển mở rộng | 8/8 |
| Bốn nhóm xác nhận trước đây | 2/2 |
| Mở rộng 32 nhóm | **25/25** |

25 record B của tập mở rộng chứa 50 chuyến mục tiêu; đơn vị độc lập để gộp vẫn là **25 nhóm**, không phải số điểm GPS hay số lần chạy cơ chế.

| Thuộc tính phần bị giấu, chỉ dùng để chẩn đoán | Trung vị | Khoảng min–max |
|---|---:|---:|
| Thời gian từ mẫu cuối được xem tới endpoint | 304,5 giây | 153–588 giây |
| Khoảng cách thẳng từ mẫu cuối tới endpoint | 1.123,7 m | 151,5–2.844,8 m |
| Khoảng cách giữa hai endpoint trong cặp | 1.860,4 m | 700,5–3.128,0 m |

Hai vị trí cuối được quan sát gần nhau không có nghĩa hai tiền tố hoàn toàn giống nhau: tốc độ, thời gian và làn vẫn có thể mang thông tin. Vì vậy **không kết luận mọi attacker đều bất lực**.

## 2. Đã thử cải thiện thực sự

Bank cũ có 131 decoder B, gồm hình học, học máy, ngoại suy và posterior theo mạng đường. Vòng này bổ sung **40 decoder chuyên cho B**:

- Extra Trees dự đoán trực tiếp tọa độ đích.
- Extra Trees dự đoán độ dời tới đích trong hệ trục theo hướng di chuyển.
- kNN theo hai cách biểu diễn trên, với k = 1/3/5/9.
- Prior đích học riêng từ B, vị trí quan sát cuối và các bản chiếu về mạng đường.

Chỉ truyền tọa độ/thời gian tương đối của tiền tố được phép vào hàm dự đoán. Không dùng ID nhóm/chuyến, nhãn vai trò, thời gian đuôi thật, điểm cuối hay tuyến tương lai làm feature. Nhãn thật chỉ dùng để học trên tập fit và chấm điểm ngoại tuyến.

Trong 10 nhóm fit và 10 nhóm selection đã định sẵn, mỗi bên có **7 nhóm đủ điều kiện B**, tương ứng 14 chuyến mỗi bên. Chọn decoder riêng cho MAE và từng ngưỡng Hit trên selection; khóa lựa chọn rồi áp dụng vào 25 nhóm B mở rộng. Vì tập mở rộng đã được xem trước vòng này, kết quả mới là **chẩn đoán thăm dò**, không được gọi là xác nhận trên một tập test hoàn toàn mới.

## 3. Kết quả raw control

Đây là khả năng attacker khi nhận **vị trí chưa qua cơ chế bảo vệ**, vẫn giới hạn ở tiền tố được phép. Với control này, Hit cao hơn và MAE thấp hơn nghĩa là attacker tốt hơn.

| Bộ attacker | MAE ↓ | Hit100 ↑ | Hit200 ↑ | Hit500 ↑ |
|---|---:|---:|---:|---:|
| Bank gốc, lựa chọn cũ | 1.220,3 m | 0% | 0% | 8% |
| 40 decoder chuyên B, chọn trên selection | 1.178,3 m | 0% | 0% | 12% |
| Bank hợp nhất 171 decoder, chọn trên selection | **1.178,3 m** | **0%** | **0%** | **8%** |

MAE giảm 42 m, khoảng 3,4%. Bank chuyên B đạt Hit500 cao hơn trên tập đánh giá, nhưng khi chọn giữa toàn bộ decoder trên selection, decoder Hit500 cũ vẫn thắng và kết quả giữ ở 8%. **Không lấy 12% thay cho 8% bằng cách chọn lại theo tập đánh giá.** Các ngưỡng có thể dùng decoder khác nhau nên các cột Hit không phải một CDF chung.

Kết quả chưa đủ tạo một raw control hữu ích ở độ phân giải 100–200 m. Do đó, việc phương pháp bảo vệ cũng đạt Hit100 = 0% ở B **không chứng minh nó đã làm giảm khả năng xác định đích** ở ngưỡng này. Cải thiện MAE nhỏ cũng chưa phải bằng chứng cải thiện có ý nghĩa thống kê.

Artifact còn lưu một kiểm tra dùng đáp án để chọn decoder tốt nhất cho từng mục tiêu: bank 171 có ít nhất một dự đoán trong 100 m ở 4% mục tiêu. Đây chỉ là chẩn đoán độ phủ của **bank hữu hạn**; không phải attacker triển khai được, không phải benchmark và không phải cận cho mọi thuật toán.

## 4. Tập trung S10 ở đâu để lập luận vững hơn?

**S10.A/C** chỉ giấu đoạn cuối ngắn theo quy tắc 60 giây; C còn cho phép kết hợp các chuyến liên kết. Raw control xác định endpoint trong 100 m ở 12,90% (A) và 21,88% (C), vì vậy có tín hiệu để kiểm tra cơ chế bảo vệ có làm giảm suy luận hay không.

Bảng dưới là **phân rã mô tả từ kết quả đã khóa**, gộp đều A và C, không chọn/chạy lại đối chứng. A có 31 nhóm, C có 32; không cộng thành 63 nhóm độc lập.

| Phương pháp | S10.A/C Hit100 ↓ | S10.A/C MAE ↑ |
|---|---:|---:|
| Vị trí thật | 17,39% | 211,6 m |
| DLS adapter | 11,16% | 221,6 m |
| RDG adapter | 10,75% | 224,5 m |
| TransProtect, predictor Markov | 13,31% | 259,3 m |
| Semantic, predictor thực nghiệm | 11,82% | 292,3 m |
| Fake-query adapter | 26,11% | 213,1 m |
| Đề xuất 30 + lịch | **0%** | **1.916,3 m** |
| Đề xuất 67 + lịch | **0%** | **1.916,3 m** |

Lập luận được giữ ở mức: trong cùng dịch vụ, tập dữ liệu và bank đối thủ đã thử, phương pháp theo lịch làm giảm suy luận endpoint ở A/C, với chi phí truyền cao hơn. Hai cấu hình lịch trùng kết quả privacy A/C không có nghĩa chi phí và utility trùng nhau. AnotherMe là tham chiếu offline có lỗi và mẫu số khác, nên không đưa vào bảng cùng mẫu số này. Các adapter không tương đương tái lập đầy đủ paper nguyên bản.

Khoảng tin cậy A/B/C trong báo cáo cũ **không được chuyển sang A/C**. Bảng này không thêm kiểm định mới. Bảo đảm độc lập payload vẫn phụ thuộc vùng/lịch đăng ký công khai cố định và không bao gồm IP, account, click hay kích hoạt theo chuyến.

## 5. Quyết định cho vòng tiếp theo

1. Giữ S10.B trong dataset và bảng đầy đủ như ca tham chiếu; không tiếp tục dùng nó làm bằng chứng riêng cho đóng góp endpoint.
2. Gộp hướng dự đoán đích từ tiền tố này vào S6.A, tránh tính một phép thử thành hai bằng chứng khác nhau.
3. Ưu tiên hoàn thiện S1/S2/S3, S9 và S10.A/C. Đổi decoder trong chẩn đoán này không thay kết quả cơ chế bảo vệ đã báo cáo.
4. Chỉ mở một biến thể B mới nếu có câu hỏi khác và dữ liệu phụ trợ hợp lý, ví dụ tổng quãng đường bị rò trong metadata. Phải định nghĩa lại quan sát, huấn luyện/chọn lại attacker công bằng và dùng cohort chưa xem; không cấp endpoint thật để làm control dễ hơn.

Hướng dùng metadata có tiền lệ: nghiên cứu [Dhondt và cộng sự, CCS 2022](https://lepoch.at/publication/epz-inference-attacks) khai thác thông tin khoảng cách rò rỉ, mạng đường và vị trí đi vào vùng che để suy luận endpoint. Đây là chế độ thông tin khác B hiện tại; không mượn tỷ lệ thành công của paper để suy ra kết quả của ta.

## Tái lập và kiểm tra

- [Protocol](../../artifacts/benchmarks/s10b_diagnostic_v1/protocol.json), [selection](../../artifacts/benchmarks/s10b_diagnostic_v1/selection.json), [readout](../../artifacts/benchmarks/s10b_diagnostic_v1/readout.json), [verification](../../artifacts/benchmarks/s10b_diagnostic_v1/verification.json).
- `python -m experiments.diagnose_s10b select` rồi `evaluate`: từ chối ghi đè selection/readout đã hoàn tất. Code fit lại tất định từ dữ liệu lịch sử; không cần pickle model mới.
- `python -m experiments.verify_s10b`: tính lại metric/lựa chọn, kiểm tra tách nhóm, đối chiếu dữ liệu S6.A, bảo toàn 131 decoder cũ và hash nguồn. Hai raw repetitions được xác nhận giống nhau, chỉ tính một lần.
- `pytest tests/test_prefix_destination_attack.py`: kiểm tra hệ trục tương đối, loại bỏ metadata/clock tuyệt đối, cửa sổ đứng yên và hợp đồng input raw.

Lần chạy evaluate đầu gặp lỗi đọc container JSON trước khi dự đoán trên tập đánh giá; bản sửa chỉ mở khóa bọc `rows`, được ghi trong `implementation_amendment.json`, giữ nguyên selection. Benchmark và dataset gốc không bị sửa.
