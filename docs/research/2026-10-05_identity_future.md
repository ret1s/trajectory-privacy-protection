# S4, S5, S6 — vòng chẩn đoán sau buổi gặp 03/10

Đã có bộ đánh giá chạy được cho **liên kết danh tính tổng hợp S4**, **vị trí tương lai
20 giây là đại diện không gian của S5**, và **đích đến S6 từ prefix công bố đã đóng
băng**. Đây là kết quả phát triển trên dữ liệu SUMO; chưa phải bộ kiểm định đầy đủ
S4/S5/S6 hoặc bằng chứng nhận diện người thật.

## 1. Cần phân biệt ba scenario

Theo [`data/threat_scenarios.py`](../../data/threat_scenarios.py):

| Scenario | Attacker cần suy luận gì? | Đã chạy trong vòng này |
|---|---|---|
| **S4** | Hai phiên có thuộc cùng người/thiết bị hay không? | Tách riêng **cùng người tổng hợp** và **cùng phương tiện vật lý tổng hợp** |
| **S5** | Đoạn đường tiếp theo | Proxy vị trí GPS sau **20 giây**; bài toán đúng ID đoạn đường bị chặn bởi thiếu bộ giải mã cạnh công khai |
| **S6** | Đích đến từ phần chuyến đã quan sát | Vị trí cuối của trace từ prefix S3 đã đóng băng; chưa chạy toàn bộ các tình huống fork/lịch sử của S6 |

S5 và S6 không phải hai dạng nhận diện danh tính. Cần đánh giá cả danh tính người
và phương tiện trong S4, rồi đánh giá suy luận tương lai bằng các target riêng.

## 2. Dữ liệu và ranh giới attacker

Giữ nguyên [`research_loop_expanded_v1`](../../artifacts/datasets/research_loop_expanded_v1/)
và toàn bộ các file benchmark cũ. Chia **12 route family** theo quy tắc đóng băng:

- Train: family-701–706.
- Chọn attacker/ngưỡng: family-707–709.
- Readout: family-710–712.

Mọi phiên, người tổng hợp, phương tiện và lần lặp RNG của một family nằm trong
cùng phần. Không để hai phiên của một danh tính xuất hiện ở train và test.
Attacker chỉ nhận thời gian công bố, ID công khai và tọa độ Q. Person ID, vehicle
ID, GPS thật, route plan, mốc thời gian gốc và ledger chỉ nằm ở phía evaluator.

S4 chạy **72 phiên**, gồm sáu vai trò mỗi family: base, repeat, new_device,
shared_device, companion và partial. Mọi cặp không thứ tự trong sáu phiên được
chấm: **180 cặp mỗi phương pháp**; 90 train, 45 selection, 45 test. Chỉ **20%** cặp
cùng danh tính, nên đoán tất cả “khác người” đã đạt accuracy 80%. Vì vậy dùng
**balanced accuracy**, macro-F1 và ROC-AUC; không dùng accuracy một mình.

GeoI-Slack chạy thực sự với K=5, B=0,24/m, H=12, ngưỡng tái dùng 200 m, lịch đọc
60 giây và slack=0,03. Mỗi phiên có RNG độc lập; ngân sách hiệu lực không quá
0,23/m. S4 không bật che đầu/cuối, để xét riêng khả năng liên kết.

Cache lane gốc không còn. Các lần chạy mới dùng mạng công khai tái dựng từ toàn
bộ **9.138 polyline lưu trữ**, 22.106 state, 907 ô ước lượng và 418 POI. Khoảng
cách state 40 m, tốc độ 8 m/s và kết nối junction là giả thiết công khai; không
khôi phục được đầy đủ luật rẽ/lane ban đầu. Đây là một tích hợp phát triển riêng,
không thay thế benchmark gốc. S5/S6 dùng nguyên prefix công bố trong benchmark
cũ, không cần chạy lại cơ chế bảo vệ.

## 3. S4: cải thiện attacker trước khi đọc kết quả bảo vệ

Lần đầu dùng ExtraTrees/kNN với threshold mặc định 0,5. Raw vehicle chỉ đạt
balanced accuracy **44,44%**, thấp hơn mức đoán ngẫu nhiên cân bằng. Lần này không
đủ để kết luận mô hình bảo vệ tốt; lưu nguyên kết quả thất bại.

Vòng tiếp theo thêm so khớp hình dạng các track Q, giữ tính bất biến khi đổi tên
track, và chọn threshold từ lưới 0; 0,05; …; 1 trên **selection**. Không chọn
threshold, feature hoặc phương pháp theo test score. Readout vẫn là dữ liệu đã
dùng trong vòng phát triển trước, nên không gọi là confirmation độc lập.

| Target S4 | Raw: balanced accuracy / AUC | GeoI-Slack: balanced accuracy / AUC |
|---|---:|---:|
| Cùng người tổng hợp | **69,44% / 0,778** | **43,06% / 0,532** |
| Cùng phương tiện vật lý tổng hợp | **62,50% / 0,718** | **52,78% / 0,448** |

Control đảo nhãn train đạt lần lượt 51,39% và 52,78% balanced accuracy trên raw.
Như vậy attacker raw đã có tín hiệu vượt control. Tuy nhiên:

- Điểm dưới 50% không có nghĩa “bảo vệ tốt hơn ngẫu nhiên”; nó có thể là lỗi
  hiệu chỉnh/quyết định của attacker trên nhóm mới.
- GeoI-Slack **vẫn có một test family đạt AUC liên kết người 0,861**. AUC theo
  family dao động 0,278–0,861; số trung bình không chứng minh mọi route đều được
  bảo vệ. Với ba family, 45 cặp có phụ thuộc không tương đương 45 mẫu độc lập.
- Dataset chủ ý cho base/repeat/new_device/shared_device cùng route nhưng nhãn
  người và phương tiện khác nhau. Tọa độ không đủ để suy ra chắc chắn “cùng
  người” hay “cùng xe”. Đây là nhãn thiết kế SUMO, không có quan sát người thật.
- Bảo vệ Q không xóa account, IP, biển số, phần cứng hay click. Những kênh đó
  cần threat model và cơ chế riêng; không được gộp vào bảo đảm Geo-I.

Kết quả hiện tại cho thấy giảm tín hiệu liên kết ở mức trung bình dưới attacker
đã định nghĩa, đồng thời chỉ rõ một nhóm còn rò. Chưa đủ để công bố giải quyết
toàn bộ S4.

## 4. S5/S6: suy luận tương lai từ prefix đã bảo vệ

Bank gồm ExtraTrees, kNN, vị trí công bố cuối và vận tốc suy ra từ phần đã công
bố. Vận tốc còn được chiếu vào catalogue đường công khai có 48.844 điểm, cách
nhau tối đa 20 m trên từng polyline. MAE lớn hơn hoặc Hit100 thấp hơn có nghĩa
attacker khó định vị target hơn, trong phạm vi bank này.

Prefix S3 thường kết thúc sát GPS cuối chuyến, nên phép hỏi “20 giây sau” ở cuối
prefix không có target cho nhiều family. Giữ lại lần thất bại này. Phép sửa cố
định là **bỏ đúng một event công bố cuối khỏi mọi prefix**, rồi dự đoán GPS sau
20 giây; event tương lai không được đưa vào feature. Không cắt theo kết quả
mong muốn hoặc chọn family có số đẹp.

Mỗi phương pháp có 12 dòng train, 6 selection, 6 test; các lần lặp RNG vẫn gom
theo ba test family. Các biến thể ở đây là control nội bộ, không phải paper
đối chứng bên ngoài.

| Phương pháp | S5 proxy: MAE / Hit100 | S6 destination: MAE / Hit100 |
|---|---:|---:|
| **Raw** | **93,5 m / 66,67%** | **132,8 m / 33,33%** |
| Filter-Paced | 1.391,4 m / 0% | 2.609,8 m / 0% |
| GeoI-Paced | 1.719,6 m / 0% | 2.560,9 m / 0% |
| **GeoI-Slack** | **860,2 m / 0%** | **942,0 m / 0%** |
| Response-Progress | 1.719,6 m / 0% | 2.770,3 m / 0% |

GeoI-Slack gây sai số cao hơn raw, nhưng một số biến thể khác gây sai số cao
hơn GeoI-Slack. Vì vậy kết quả này không chứng minh GeoI-Slack vượt tất cả
phương pháp về privacy; cần đọc cùng utility và chi phí của từng biến thể.

**S5 đúng ID cạnh chưa qua gate.** Không một test edge ID nào xuất hiện trong
train; classifier đóng có accuracy 0% cả với raw. Không được lấy số đó làm
bằng chứng bảo vệ. Catalogue công khai mới có ID polyline tái dựng, chưa có ánh
xạ sang ID cạnh SUMO gốc. Proxy GPS sau 20 giây vẫn là phép đo không gian hợp
lệ, nhưng không thay cho next-edge accuracy tại các fork.

**S6 chỉ là target vị trí cuối từ prefix S3.** Khoảng từ prefix đến đích ở test
là 16–87 giây. Chưa kiểm định S6 tại fork chung hoặc dự đoán routine/rare từ
lịch sử nhiều ngày; không trình bày bảng này như kết quả đầy đủ S6.

## 5. Gate tiếp theo và tái lập

1. Giữ nguyên control yếu và repair đã lưu; tăng số family và bổ sung một tập
   confirmation mới sau khi đóng băng bank và threshold.
2. Với S4, bổ sung auxiliary history có nhãn người/xe rõ ràng, cặp “cùng đường
   nhưng khác người/xe”, và đánh giá riêng family còn có AUC cao. Không dùng ID
   evaluator làm feature công bố.
3. Với S5, phục hồi mapping cạnh công khai và candidate-edge decoder; đo accuracy
   tại fork và one-successor riêng, có raw/majority/permutation control.
4. Với S6, tạo public transcript cho đúng các record fork/history đã có trong
   dataset, giữ lịch sử ngày 1–6 riêng khỏi target ngày 7–8.

```bash
python -m experiments.identity_future_eval --future-only
python -m experiments.identity_future_eval --linkage-only
python -m experiments.identity_future_refine
python -m experiments.identity_future_prefix_cut
python -m experiments.verify_identity_future
# Ghi một bản kiểm tra mới; path chưa tồn tại, giữ nguyên validation.json đầu tiên.
python -m experiments.verify_identity_future --validation-output /private/tmp/identity_future_recheck.json
python -m pytest -q tests/test_identity_future.py
```

Các lệnh tạo kết quả từ chối ghi đè bằng chứng đã hoàn thành. Validator có thể
chạy lại: mặc định kiểm tra toàn bộ rồi giữ nguyên `validation.json`; tùy chọn
`--validation-output` ghi bản mới vào path chưa tồn tại. Bản kiểm tra đầu tiên
giữ hash validator tại lúc tạo, bản recheck ghi hash code hiện tại. Không đổi
bank, threshold, split hoặc đọc lại test để chọn tham số ở bước recheck.
Các protocol và output nằm ở
[`artifacts/benchmarks/identity_future_20261005`](../../artifacts/benchmarks/identity_future_20261005/).
`future_results.json` và `linkage_results.json` giữ lần đầu, `future_results_v2.json`
sửa eligibility từng target, `refinement_results.json` chứa attacker S4 hiệu
chỉnh, `prefix_cut_results.json` chứa S5 proxy sau cắt cố định. File public riêng
không chứa nhãn danh tính; evaluator file được đánh dấu rõ. Validator kiểm tra
hash, split, ngân sách, nhãn cặp, prefix giữ nguyên, event tương lai bị giữ lại
và phép tính metrics. **10 kiểm thử và validator đã qua.**
