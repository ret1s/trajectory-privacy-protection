# Cập nhật nghiên cứu sau buổi hướng dẫn 05/09/2026

Ngày rà soát: 07/09/2026. Bản chính: `thesis/main.tex`.

## Phạm vi và kết luận được phép dùng

Đợt này cập nhật lập luận, nguồn nghiên cứu, sơ đồ và đặc tả benchmark; không sửa thuật toán, chạy lại benchmark hay sinh đủ dataset mới. Các kết quả smoke test cũ không trở thành kết quả S1–S10 chỉ vì đặc tả đã được mở rộng.

- Giữ bốn chương: kiến trúc; đô thị; targets/scenarios/dataset; đối chứng và giao thức đánh giá.
- Thêm ngữ cảnh công khai trên thiết bị, tách khỏi lịch sử bí mật. Cơ chế bảo vệ được tô nổi và có sơ đồ chức năng riêng.
- Duy trì bốn nhóm target; mười scenario là thiết kế thí nghiệm của luận văn, không phải taxonomy chuẩn lấy nguyên từ một paper.
- S2–S3 là trọng tâm phát triển ban đầu; S1 là nền. Các scenario còn lại là hướng mở rộng/đánh giá theo chức năng, chưa có tuyên bố đã bảo vệ.
- Không phát biểu rằng lĩnh vực chưa có framework đánh giá. Shokri et al. đã xây dựng framework; đóng góp cần khảo sát là cấu hình thống nhất cho urban online dummy-generation, có tác vụ và chi phí cụ thể.
- Khảo sát có mục tiêu, chưa phải systematic review đầy đủ. Chưa đủ bằng chứng tuyên bố không phương pháp nào khác bao phủ cùng các kịch bản.

## Ánh xạ mã kịch bản: tránh đọc nhầm artifacts cũ

| Mã cũ | Mã mới | Thay đổi |
|---|---|---|
| S1 | S1 | Định vị từ một lần gửi. |
| S2 | S2 | Target là vị trí điểm dừng, không toàn bộ hành trình. |
| S3 | S3 | Target là chuỗi vị trí đã đi; tách với dự báo tương lai. |
| S4 | S4 | Phân biệt person/device/vehicle/session; liên kết phiên khác nhận diện danh tính. |
| S5 | S5 và S6 | Tách bước/đoạn tiếp theo ở horizon cố định khỏi đích đến chưa tới. |
| S6 | S7 | Ý định/nội dung truy vấn. |
| S7 | S8 | Đặt tên suy luận qua đồng hành; population prior trở thành nguồn phụ trợ chung, không một scenario riêng. |
| Chưa có | S9 | Suy điểm đầu đã bị che từ phần còn lại được quan sát. |
| Chưa có | S10 | Suy điểm cuối đã xảy ra nhưng bị che; khác S6 dự đoán trước khi đến. |

`threat_records.tex`, `notes/threat_model_mapping_draft.md`, snapshots và kết quả benchmark cũ vẫn giữ mã cũ để bảo toàn ý nghĩa lịch sử. Khi triển khai mới phải có trường phiên bản taxonomy; không relabel CSV/JSON bằng thay chuỗi toàn cục.

## Sổ nguồn: kịch bản, năng lực đối thủ và công trình bảo vệ

Các liên kết dưới đây là nguồn gốc/nguồn tác giả, không dựa vào đoạn trả lời của công cụ tìm kiếm để gán kết quả thực nghiệm.

| Nguồn | Phần được sử dụng | Giới hạn bằng chứng |
|---|---|---|
| [Shokri et al., S&P 2011](https://www.comp.nus.edu.sg/~reza/files/Shokri-SP2011.pdf) | Framework mô hình người dùng, LPPM, đối thủ, quan sát và sai số suy luận. | Không gọi entropy, khoảng cách dummy và xác suất tấn công là cùng một metric. |
| [Niu et al., INFOCOM 2014](https://mcn.cse.psu.edu/paper/other/infocom-ben-niu14.pdf) | DLS và enhanced-DLS; chọn dummy có xét xác suất truy vấn, entropy/phân tán. | Nền cho S1, không bằng chứng bảo vệ chuỗi S2/S3. |
| [Sun et al., ASA 2017](https://epic.hust.edu.cn/minchen/min_paper/2017/2017-FGCS-2-ASA.pdf) | Tấn công thống kê dài hạn và thống kê vùng; cơ chế chống thống kê. | S2 dừng tại chỗ là trường hợp thiết kế của ta, không phải một benchmark stationary được chép từ ASA. |
| [Shaham et al., TMC 2021](https://doi.org/10.1109/TMC.2020.2993599), [bản tác giả](https://arxiv.org/abs/1805.06104) | Viterbi, entropy chuyển tiếp, Robust Dummy Generation (RDG). | Phải đọc và hiện thực đúng quy tắc chuỗi trước khi gọi là RDG reproduction. |
| [Beresford & Stajano, 2004](https://www.cl.cam.ac.uk/~arb33/papers/BeresfordStajano-MixZones-PerSec2004.pdf) | Mix zones, đổi bí danh và nguy cơ nối người di chuyển qua quan sát. | Cần giao thức định danh; không có tác dụng nếu account/device ID thật đã gửi nguyên vẹn. Không phải dummy baseline trực tiếp. |
| [de Montjoye et al., 2013](https://www.nature.com/articles/srep01376) | Tính nhận diện/duy nhất của mẫu di chuyển. | Duy nhất trong dữ liệu không tự cho biết tên người nếu thiếu ánh xạ phụ trợ. |
| [Ziebart et al., AAAI 2008](https://publications.ri.cmu.edu/storage/publications/pub_files/2008/7/AAAI2008-bziebart.pdf) | Route/destination inference từ partial trajectory; cơ sở năng lực S5/S6. | Nghiên cứu dự báo, không phải kiểm chứng tấn công trên đầu ra LPPM của ta. |
| [Predictive DP mechanism, PETS 2014](https://arxiv.org/abs/1311.4008) | Bảo vệ chuỗi bằng cơ chế có dự đoán/lịch sử. | Không thuộc dummy-generation; không nhầm predictor của cơ chế bảo vệ với attacker dự báo đích. |
| [PriSTE, ICDE 2019](https://www.cs.emory.edu/site/aims/pub/cao19icde.pdf) | Khái niệm sự kiện không gian–thời gian và yêu cầu bảo vệ suy luận từ chuỗi. | Chỉ định hướng chức năng S5/S6, không khẳng định đúng hai đầu dự đoán của ta đã được giải quyết. |
| [Wu et al., WWW 2021](https://link.springer.com/article/10.1007/s11280-020-00830-x) | Chuỗi truy vấn giả nhằm bảo vệ cả location và query privacy. | Đọc metadata/abstract; chưa trích công thức metrics hoặc gán kết quả toàn văn. Đã sửa tên tài liệu: “Dummy Query Sequences”, không “Fake Query Sequences”. |
| [Olteanu et al., TMC 2017](https://api.unil.ch/iris/server/api/core/bitstreams/f8701973-7f51-4425-b15d-08f88eded8ef/content) | Interdependent privacy, co-location, Bayesian network/belief propagation. | Là phân tích/tấn công, không phải bằng chứng một bộ sinh dummy bảo vệ được S8. |
| [Dhondt et al., CCS 2022](https://doi.org/10.1145/3548606.3560616), [abstract hội nghị](https://www.sigsac.org/ccs/CCS2022/proceedings/ccs-proceedings.html) | Endpoint Privacy Zones và suy luận điểm đầu/cuối trong ứng dụng fitness. | Chuyển nguyên lý đe doạ sang đô thị, không chuyển số liệu tấn công hoặc tuyên bố chạy trên ô tô. |

Khoảng trống cần khảo sát tiếp: bảo vệ đồng hành S8 bằng phối hợp nhóm trong kiến trúc client–LSP. Không được lấy việc chưa tìm đủ tài liệu trong đợt này làm bằng chứng không có prior work.

## Kiểm tra chỉ số công trình gốc

### TransProtect, SIGSPATIAL 2024

Nguồn toàn văn: [Yadav et al.](https://arxiv.org/html/2409.09495v1), Mục 4.4, công thức (13), Mục 5.1.4 và bảng kết quả.

- EIE đo sai số giữa ước lượng của attacker và vị trí thật; bảng dùng km, không phải khoảng cách giữa thật và đầu ra cơ chế.
- Utility loss là kỳ vọng sai lệch chi phí tới đích dịch vụ, có trọng số đích; chi phí mạng đường có thể là thời gian. Không phải POI Recall@k.
- Để tái lập số liệu cần chốt estimator/loss/averaging theo mã gốc. Common location MAE là phép đo bổ sung được đặc tả rõ, không tự nhận là đã tái lập chính xác EIE của paper.
- TransProtect là lớp lựa chọn ứng viên kết hợp cơ chế Laplace/LP; số ứng viên nội bộ không phải số vị trí/gói bắt buộc công bố.

### AnotherMe, TDSC 2024 (early access 2023)

Nguồn metadata: [DOI](https://doi.org/10.1109/TDSC.2023.3314200). Nguồn tác giả: [repository](https://github.com/fang-zhiyou/AnotherMe/tree/0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f), đối chiếu `docs/reproduction/anotherme.md`.

- Mã tác giả có bộ sinh VTGA và các gói thí nghiệm phát hiện quỹ đạo với LSTM/CNN/TSHN. Detector dùng học sâu không làm VTGA trở thành deep generator.
- Không truy cập được toàn văn qua trang IEEE trong đợt này; Crossref chỉ xác nhận metadata và liên kết xuất bản, không cung cấp công thức metrics.
- **Chưa xác minh đầy đủ** công thức, mẫu số, phân chia dữ liệu, bảng accuracy/mobile overhead của bài. Không điền recognition rate từ nguồn thứ cấp, không tự gán ADE/FDE/POI recall cho paper.
- Cần bổ sung PDF tác giả/bản thư viện truy cập hợp lệ trước khi chốt original-metric parity. Mốc 50% trong thesis chỉ giải thích phép đoán ngẫu nhiên của hai lớp cân bằng, không phải kết quả AnotherMe đã được kiểm chứng.

### Semantic correlation, JKSUCIS 2026

Toàn văn: [Liu, Peng & Zhou](https://link.springer.com/article/10.1007/s44443-026-00899-w), Mục 6.2–6.5.

- ASR: tỷ lệ yêu cầu đạt điều kiện ẩn danh; định nghĩa liên hệ khả năng LSP nhận ra thật không quá 1/K. Cần attacker/posterior operationalization, không chỉ `len(candidates) == K`.
- DER: nguồn nêu trung bình similarity rồi mô tả tỷ lệ dummy vượt ngưỡng. Phải giải quyết ambiguity trước code parity: giữ `mean_similarity` và `valid_dummy_fraction` riêng; chỉ đồng nhất nếu similarity là indicator nhị phân và mẫu số trùng nhau.
- MSE/semantic prediction accuracy là chỉ số học; độ trễ sinh tập là chi phí. Chưa thấy phép đo đầu cuối top-k POI utility trong phần đánh giá này.

### Các hướng không dùng học sâu

| Paper | Vị trí định nghĩa gốc | Cách đưa vào giao thức chung |
|---|---|---|
| [Cache-based, Entropy 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9955910/) | Mục 6.2 data availability; 6.3 cache hit; 6.4 query time; 6.5 xác suất lộ vị trí sau các bước tấn công. | Giữ riêng chỉ số gốc. Đánh giá ứng dụng phải xét chất lượng/độ mới cache; không đồng nhất công thức availability của paper với POI recall. |
| [LSPPM-SI, Scientific Reports 2025](https://www.nature.com/articles/s41598-025-88553-9) | Evaluation indicators, công thức 12–14: AE entropy; IL tổng diện tích vùng điểm dừng; SD trung bình số semantic categories. | IL không phải mét hay ADE; semantic diversity không chứng minh chống query-intent classifier. Phải kiểm tra nhân quả khi xử lý điểm dừng toàn hành trình. |
| [Fake queries, JKSUCIS 2026](https://link.springer.com/article/10.1007/s44443-025-00438-z) | Mục 6.3 indistinguishable moving paths; 6.4 average computation delay; 6.5 ASR theo attack; 6.6 mobile time/RSS. | Ghi rõ path count cục bộ không phải posterior toàn chuỗi. Lần gửi dummy-only được chèn giữa các truy vấn thật: không đồng nhất cả phương pháp với một fixed dummy-only track. |

Các paper gốc có thể dùng GeoLife. Điều đó chỉ mô tả xuất xứ kết quả paper; benchmark của luận văn tiếp tục SUMO + OSM, không dùng GeoLife để sinh/học dữ liệu benchmark.

## Hợp đồng đánh giá mới cần hiện thực

1. **Đầu vào:** public urban graph + POI snapshot + các quỹ đạo SUMO/LBS events. Ghi phiên bản map, hệ tọa độ và quy tắc snapping/access point. Chỉ dùng public traffic đã có lúc chạy.
2. **Taxonomy/version:** mã S1–S10 đi cùng phiên bản; fixture kiểm tra rõ target, observation window và auxiliary knowledge.
3. **Causal replay:** prefix invariance với cùng random stream; cơ chế và attacker dự báo không nhận route file, điểm cuối hoặc tốc độ tương lai.
4. **Event stream:** tách real service events khỏi public emissions. Hỗ trợ 0, 1 hoặc nhiều lần gửi; thời gian/im lặng là quan sát. Nhãn true-vs-fake event thuộc evaluator, không cấp LSP.
5. **Privacy head:** location/stop/endpoints → point error + hit radius; identity → person/device top-1 riêng hoặc pair-link F1; future edge → top-1 tại horizon; destination → top-1/error; query → macro-F1. S8 bật/tắt co-location trong cùng cấu hình.
6. **Attacker controls:** dùng cùng auxiliary knowledge và tập kiểm thử; attacker biết cơ chế, được fit đúng output contract. Có baseline chỉ dùng prior và baseline không bảo vệ. Không buộc chọn true member khi phương pháp không công bố true member.
7. **Utility:** end-to-end POI Recall@k sau local filtering. Deduplicate theo POI ID, cùng top-k/tie break/category/routing. Không có đáp án → N/A và coverage; có đáp án nhưng service fail → 0, không loại khỏi denominator.
8. **Costs:** request/real-query, coordinates/real-query, bytes gồm response; generation và end-to-end latency tách, p50/p95. Ghi toàn bộ cache policy và dummy requests.
9. **Validity:** road/mode/turn/speed/time/stop constraints, unreachable edges, missing POI. Không dùng feasibility làm privacy score.
10. **Statistics:** chia train/validation/test theo người/chuyến và route/seed phù hợp. Bootstrap theo đơn vị độc lập người/chuyến, không bootstrap từng điểm tương quan. So sánh tại utility/cost tương đương và báo đường đánh đổi theo từng scenario, không tổng hợp thành một điểm chung không có ý nghĩa.

Các thresholds (r, horizon, số POI, độ mới cache), số người/chuyến, số seeds và budget sweep chưa được chốt bằng thực nghiệm. Phải chọn trên train/validation trước khi xem test; chưa gọi bộ benchmark hoàn thiện.

## Thứ tự thực hiện tiếp theo

1. Bổ sung toàn văn AnotherMe; chốt definition/denominator ASR và DER từ nguồn hoặc liên hệ tác giả nếu cần.
2. Chuyển bảng scenario thành cấu hình có thể kiểm tra bằng máy. SUMO tạo chuyển động; lớp LBS tự quản lý user/device/session, queries, co-location và endpoint masks.
3. Xây attacker S2/S3 trước; thêm DLS/RDG và fake-query insertion 2026 theo source-mapped implementation, có kiểm tra lịch gửi.
4. Cài metric adapters + common service evaluator, rồi mới chạy sweep. Giữ original-paper metrics ở bảng riêng.
5. Đánh giá module bật/tắt theo target; chỉ viết contribution về coverage khi có evidence vượt comparator dưới protocol chung. Chứng minh Geo-I (nếu áp dụng) là việc riêng, không suy từ entropy/error hoặc số test phần mềm.

## Kiểm tra bản tài liệu

- Build XeLaTeX với `latexmk`: thành công; PDF cuối 36 trang A4, giữ format/bìa canonical thesis.
- Kiểm tra tĩnh: bốn chương, đúng mười hàng S1–S10; 24 khóa tài liệu được trích dẫn, không khóa thiếu, không nhãn tham chiếu thiếu/trùng và không còn S1--S7 trong bản chính.
- Log cuối không có cảnh báo undefined reference/citation, author undefined hay overfull. Còn các thông báo underfull tại bibliography; đã kiểm tra không bị tràn/lỗi hiển thị.
- Đã render và kiểm tra tổng thể các trang, xem riêng hai sơ đồ, bảng kịch bản, bảng công trình/metrics và các trang đổi bố cục; sửa trang chỉ còn một đoạn cuối chương và khoảng ngắt trang thừa.
- `git diff --check` đạt. Đây là kiểm tra tài liệu và ranh giới phát biểu, không phải chứng nhận độ đúng của thuật toán hoặc tái lập tất cả công trình.
- Không sửa hoặc chạy lại benchmark implementation trong đợt cập nhật này.
