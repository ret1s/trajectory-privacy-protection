# Hồ sơ bằng chứng cho báo cáo GVHD ngày 06/10/2026

Báo cáo mới dùng [evidence.json](../supervisor_meeting/2026-10-06_brief/evidence.json). Bảng so đối chứng giữ nguyên paper-v2; các vòng mới về query purpose, endpoint, danh tính và ngân sách nhiều phiên được trình bày riêng. Không sửa report cũ, trace, nhãn hoặc điểm số để làm mô hình thắng.

## Related works: một bảng, cửa sổ ba năm

Cửa sổ là **06/10/2023–06/10/2026**, xét ngày công bố online đầu tiên. Sáu paper dưới đây đều nằm trong cửa sổ. **Fully** nghĩa paper trực tiếp xét và đánh giá task/threat theo scenario, trong giả định của paper; **Partially** nghĩa xử lý cơ chế liên quan hoặc bối cảnh hẹp hơn. Đây là đối chiếu của nhóm nghiên cứu, không phải nhãn tác giả và không có nghĩa chống mọi attacker. Scenario không liệt kê: chưa tìm thấy bằng chứng đánh giá trực tiếp.

| Paper và ngày online | Output công bố | Fully | Partially | Metrics gốc: privacy; utility; cost |
|---|---|---|---|---|
| [TransProtect, 14/09/2024](https://arxiv.org/html/2409.09495v1) | Một tọa độ nhiễu mỗi mốc; tuyến ứng viên nằm bên trong phương pháp | S1, S3 | S2 | EIE ↑; chênh chi phí tới đích Δc ↓; chưa khóa metric vận hành riêng |
| [Semantic correlation, 04/06/2026](https://link.springer.com/article/10.1007/s44443-026-00899-w) | GPS thật + K−1 dummy hợp lý theo đường/ngữ nghĩa | S1 | S2, S3 | ASR ↑; DER ↑ là chất lượng dummy; thời gian tạo ↓ |
| [Fake queries, 03/01/2026](https://link.springer.com/article/10.1007/s44443-025-00438-z) | Tập thật + dummy; xen tập query giả chỉ có dummy | S1, S3 | S2 | Số đường khó phân biệt/ASR ↑; chưa chấm POI Recall; thời gian/RSS ↓ |
| [CPCROK, 09/04/2025](https://uhra.herts.ac.uk/id/eprint/25705/) | Beacon thật + tuyến beacon giả; đổi pseudonym ở mix-zone | — | S4 | Tỷ lệ ghép đúng trace xe ρ ↓; chưa chấm POI; số tuyến giả cần thiết ψ ↓ |
| [DP-FETC, 12/11/2025](https://www.aimspress.com/article/doi/10.3934/era.2025293) | Bộ trace tổng hợp offline, chỉnh OD của tuyến tương quan | — | S3, S8, S9, S10 | MI ↓ và ε-DP; JSD vị trí/thời lượng/quãng đường ↓; phân tích độ phức tạp |
| [EV Geo-I querying, 30/01/2024](https://wrap.warwick.ac.uk/id/eprint/183198/) | Một vị trí AGeoI + dummy; Edge trộn query, xe chọn trạm sạc locally | — | S1, S2, S3 | (ε,δ)-AGeoI; CoP ↓, tỷ lệ CoP=0 ↑, Wasserstein của IBU ↓; chưa tái lập bytes/query |

Các mức Partial cần giải thích ngắn: Semantic chưa có phép đo tái dựng cả đoạn đường theo S3 của ta; ngữ nghĩa địa điểm không đồng nghĩa giấu ý định query S7. CPCROK đánh giá một lần đổi pseudonym xe, chưa tách người/xe/thiết bị qua nhiều phiên. DP-FETC bảo vệ dữ liệu offline và quan hệ tương quan, chưa có attacker định vị từ người đi cùng hoặc suy endpoint từ cửa sổ online. EV querying nêu location/journey threats nhưng thực nghiệm tập trung utility, và giả định reset threat khi đổi Edge; vì vậy không mang nhãn Fully từ bảng cũ sang bảng mới.

Nguồn kiểm tra: TransProtect §3.1, §4.4 Eq.13, §5.1.4; Semantic §3.2, §4 và §6.3–6.5; Fake queries §3–5, §6.3–6.6; CPCROK §3.2, §5.4; DP-FETC §4.3–4.4, §5.2 Eq.19–21; EV querying §V.C–E, §VI Definition 6.1, §VII. Ba PDF không đọc được qua web tool đã được tải trực tiếp từ publisher/kho tác giả và đọc toàn văn; manifest giữ URL và SHA-256 của PDF đã đọc.

[Geo-I 2013](https://doi.org/10.1145/2508859.2516735), [DLS 2014](https://doi.org/10.1109/INFOCOM.2014.6848002), và [RDG — preprint 2018, issue 2021](https://arxiv.org/abs/1805.06104) là nguồn nền tảng, ghi riêng ngoài cửa sổ. AnotherMe issue 2024 không tự động là paper mới trong ba năm: metadata có sẵn ghi online 11/09/2023, trước cửa sổ. Publisher hiện không truy cập được; Crossref ghi ngày tạo DOI 11/09/2023 nhưng ngày tạo DOI không thay thế bằng chứng ngày online. Không đưa AnotherMe vào sáu paper gần đây.

## Vì sao cần metric chung và vẫn giữ metric gốc?

- **Output khác nhau.** ASR của tập chứa GPS thật cần posterior của thành viên thật. Tập Q-only không có thành viên đó; không được gán ASR=100%. Offline synthetic dataset cũng không thể chấm bằng cách giả làm một query online.
- **Thước đo trả lời câu hỏi khác nhau.** Entropy đo phân bố trọng số; DER đo dummy hợp lý; ε là bảo đảm của cơ chế. Chúng chưa nói attacker đoán đúng bao nhiêu hoặc dịch vụ trả đúng POI hay không.
- **So công bằng bằng hai lớp đo.** Dùng cùng threat/attacker với Hit100/MAE, cùng dịch vụ với Recall@5, kèm bytes/thời gian; thêm metric gốc khi đủ dữ liệu và đúng hợp đồng. Thiếu thì giữ N/A và lý do, không coi là 0.

CoP của EV querying đo quãng đường tới trạm sạc được chọn từ vị trí nhiễu so với trạm tối ưu từ vị trí thật; không phải sai số attacker. JSD của DP-FETC đo fidelity của phân bố dataset; không phải Recall của POI online. Không xếp các số này cạnh nhau rồi gọi cùng một privacy score.

## Bảng đối chứng cùng protocol: paper-v2 lịch sử

Nguồn đóng băng: [results.json](../../artifacts/benchmarks/paper_benchmark/results.json), SHA-256 `2df043bfb73c3b2f630046381daf5b4bc9f6941fb5be925b0967d6de98f27538`. **12 chuyến test, ba seed SUMO 81–83; K=5; top-5 POI, sáu category; 60 cửa sổ scenario có tương quan.** Mỗi cell là mean đều theo 12 chuyến. Các phương pháp có cùng trace, service contract, split và họ attacker, nhưng khác số tọa độ công bố và không có cùng bảo đảm ε. Dấu * là local adaptation, chưa tái lập đủ mô hình học/protocol gốc.

Attacker không chỉ có một Shadow KNN: bank gồm centroid, prior, continuity, road filter, full-window path decoder, mean và **Shadow KNN (3 neighbors)**; S2 có thêm running mean và intersection cho output chứa GPS thật. Fit KNN trên shadow trips, chọn decoder riêng để tối thiểu MAE hoặc tối đa Hit100 trên attack validation, rồi khóa trước test. Không dùng đáp án test để chọn attacker. Raw S9/S10 cũng chỉ thấy cửa sổ còn lại sau che đầu/cuối; sai số Raw bao gồm độ khó của masking.

Trong mỗi ô: **Hit100% ↓ / MAE m ↑ / Recall@5% ↑**.

| Phương pháp | S1 | S2 | S3 | S9 | S10 |
|---|---:|---:|---:|---:|---:|
| Raw | 100 / 13 / 100 | 100 / 39 / 100 | 96,43 / 23 / 100 | 41,67 / 120 / 100 | 8,33 / 630 / 100 |
| DLS* | 58,33 / 1.318 / 100 | 100 / 39 / 100 | 31,21 / 1.688 / 100 | 8,33 / 1.582 / 100 | 0 / 2.029 / 100 |
| TransProtect* | 83,33 / 84 / 97,78 | 66,67 / 88 / 97,19 | 69,28 / 99 / 97,93 | 25 / 184 / 97,27 | 8,33 / 680 / 98,52 |
| Semantic* | 100 / 11 / 100 | 100 / 42 / 100 | 40,58 / 257 / 100 | 33,33 / 132 / 100 | 16,67 / 731 / 100 |
| **BR-Dummy / Geo-I** | **33,33 / 299 / 95,83** | **22,22 / 247 / 97,01** | **7,44 / 776 / 87,67** | **0 / 316 / 96,60** | **0 / 2.315 / 88,33** |

BR có Hit100 không cao hơn ba adaptations trong năm scenario, với tie DLS ở S10. EIE/MAE lớn hơn TransProtect trong cả năm, nhưng DLS có EIE lớn hơn BR ở S1/S3/S9. **Chưa thắng mọi metric**; Recall S3/S10 của BR dưới 90%. Cost cũng phải đọc cùng: macro mean năm scenario là khoảng 3.237 JSON bytes/mốc và 5,40 ms/mốc cho BR, so với khoảng 720 bytes/mốc và 0,35 ms/mốc cho TransProtect. Đây là thời gian máy benchmark cũ; không phải độ trễ network hoặc điện thoại. Manifest giữ số chính xác theo từng scenario và mẫu số.

Đã tính lại **175 điểm hiển thị** từ per-trip rows và kiểm tra **37/37 source pins** còn nguyên. Năm phương pháp hiển thị và hai internal Geo-I controls đều 12/12 hoàn thành mỗi scenario. AnotherMe S1/S2/S9/S10 là N/A theo complete-route contract; S3 thành công 11/12, còn một failure. Không thay N/A/failure bằng điểm bảo vệ tốt. Các số này là dữ liệu phát triển lịch sử, không là confirmation mới cho cấu hình hiện tại.

## Metric gốc: có số nào thực sự mới?

[Native readout](../../artifacts/benchmarks/native_metrics_20261005/readout.json) tính lại **431 dòng paper-v2 K=5 thành công** từ tọa độ ước lượng/đáp án; EIE point estimate khớp MAE. **Khi cùng decoder, khoảng cách và phép gộp, EIE này là MAE**, không phải bằng chứng độc lập thứ hai. Posterior expected inference error cần posterior đã hiệu chuẩn; archive chưa có.

Δc gốc TransProtect còn **N/A** trong paper-v2: thiếu cost tables/prior đích và cache gốc; Q-only còn khác singleton output. Entropy/ASR/DER/path count thiếu prior, posterior, Sim hoặc arrival-time/direction rule tương ứng. CPCROK cần beacon/pseudonym trace; DP-FETC cần paired synthetic dataset; EV querying cần available charging stations và IBU logs. Không dựng các đầu vào đó từ GPS test để có số đẹp.

Một phép đo mới thuộc **metric family của Eq.13** đã thực hiện trên cohort riêng: Raw Δc=0 m; Geo-I Endpoint20 trung bình cả 5 Q là **1.386,74 m**, median Q **1.379,69 m**, p95 **2.340,46 m**; **170/170 mốc, 850 Q, bốn family, tám chuyến × hai seed**. Prior đều trên 418 POI công khai, chi phí là đường có hướng trên mạng tái dựng; cost tables được giữ để kiểm tra. [Readout và giới hạn](../../artifacts/benchmarks/native_cost_diagnostic_20261005/readout.json) ghi rõ đây là road-length extension, không lấp N/A của cost gốc, không phải head-to-head TransProtect, và **số lớn hơn nghĩa méo chi phí lớn hơn, không phải privacy tốt hơn**. Đánh giá dịch vụ cuối vẫn theo POI sau merge/local ranking.

## Ranh giới cho renderer

Chỉ nhóm cùng cohort/protocol/attacker trong một bảng. Các vòng endpoint robust selection, query purpose, ngân sách nhiều phiên và native SUMO S5/S6 có nguồn riêng do parent bổ sung vào manifest; không ghép điểm mới vào paper-v2. Cấu hình đã chọn trên dữ liệu xem nhiều lần phải gọi development/diagnostic. Nêu cả tradeoff và phép đo chưa đủ; không diễn giải AUC gần 0,5 như đã giấu account/IP, và không đổi backbone Geo-I để cải thiện bảng.
