# Ghi chú phát triển và kiểm chứng luận văn

Tệp này lưu các chi tiết phục vụ quá trình phát triển, kiểm chứng và trao đổi. Các nội dung ở đây không được đưa nguyên văn vào bản luận văn chính nếu chúng chỉ mô tả trạng thái code, giới hạn của bản chạy thử hoặc hướng công việc tiếp theo.

## Quy ước biên tập

- **Cập nhật 07/09/2026:** bản chính dùng S1--S10. Xem [hồ sơ nghiên cứu và ánh xạ mã cũ/mới](research_update_2026-09-07.md). Các mô tả S1--S7 và trạng thái phần mềm bên dưới là ghi chú lịch sử; không đại diện cho việc đã triển khai đủ mười kịch bản. Định nghĩa metrics mới là đặc tả đánh giá, chưa phải kết quả chạy.

- Bản chính hiện tập trung bốn chương: kiến trúc, phạm vi đô thị, mục tiêu/đe doạ/dataset SUMO, và phương pháp đối chứng. Phương pháp đề xuất cùng kết quả được tạm giữ ngoài PDF chính.
- [Snapshot đầy đủ trước khi rút gọn](snapshots/graduation_thesis_full_2026-09-05.tex) giữ nguyên nội dung cũ để tham chiếu; các số chương và mô tả thuật toán trong những ghi chú bên dưới thuộc bản đầy đủ, không phải cấu trúc hiện tại.
- AnotherMe adaptation sử dụng điểm cuối, tốc độ toàn hành trình và căn chỉnh theo toàn chuỗi; chưa được coi là đối chứng trực tuyến nhân quả đã đạt yêu cầu. Theo phạm vi biên tập hiện tại, Chương 4 tập trung vào tác giả, năm công bố và nguyên lý phương pháp gốc; phần bàn riêng về giới hạn AnotherMe được giữ tại draft này và hồ sơ tái cài đặt, không trình bày trong mục mô tả phương pháp.
- `thesis/main.tex` là bản luận văn chính và chỉ chứa lập luận khoa học, phương pháp, thiết kế thực nghiệm, kết quả và giới hạn có ý nghĩa học thuật.
- Ghi chú như “chưa giả định thuật toán”, “prototype cũ”, số test, endpoint web, hash artifact, commit và checklist triển khai được đặt tại đây hoặc trong `docs/reviews/`.
- Các phát biểu chưa được thực nghiệm hỗ trợ phải dùng phạm vi chính xác; không biến một kiểm tra tích hợp thành kết luận về riêng tư hay SOTA.

## Trạng thái phương pháp cần tiếp tục kiểm chứng

- Chương 1 phân biệt truy vấn thật với nội dung công bố; không cấp truy vấn thật cho attacker nếu module đã bảo vệ trường đó. Điểm ra của LSP là kết quả dịch vụ; utility phải được đánh giá sau bước xử lý cục bộ.
- Chương 2 định nghĩa miền trên cạnh đường, trạng thái cạnh/offset, cận thời gian đi và tập chuyển động khả thi. Đây là đặc tả mục tiêu, không xác nhận code đã kiểm tra đủ hạn chế rẽ, gia tốc, tín hiệu hoặc dừng.
- Các ví dụ trong `thesis/threat_records.tex` dùng mạng nhỏ do tác giả dựng và nhiều giao diện đầu ra để giải thích attack. Chúng không phải output của REM, SUMO hay một SOTA cụ thể.
- S3 có một chuỗi ứng viên khả thi duy nhất; S7 giữ ba chuỗi khả thi (B-C, E-C, E-F) để chỉ ra giá trị của nguồn phụ trợ. S5 tách hẳn hàng tương lai khỏi đầu vào attacker. S6 giả định thêm lịch sử tương tác, cần bật riêng khi đánh giá.

- Cơ chế neo REM hiện được lập luận trong mô hình xác suất số thực lý tưởng theo từng lần phát hành.
- Cần kiểm tra ảnh hưởng của lấy mẫu số thực hữu hạn, tách dòng randomness giữa tầng neo và tầng hậu xử lý, và cơ chế quản lý ngân sách theo cửa sổ thời gian.
- Tầng sinh quỹ đạo giả đang dùng điều chuẩn động học cục bộ. Bản hoàn chỉnh cần kiểm tra reachability theo đường đi có hướng và đưa likelihood từ POI/population prior vào hàm mục tiêu.
- Bảo đảm vị trí theo Geo-I không tự động suy rộng sang danh tính, tuyến tương lai hoặc nội dung truy vấn; mỗi mục tiêu cần module và attacker riêng.

## Trạng thái các phương pháp đối chứng

Ba pipeline đang chạy là các bản tái cài đặt độc lập có ánh xạ về công trình gốc, chưa được coi là tái lập đầy đủ kết quả paper:

- **TransProtect:** pipeline cục bộ dùng Markov proxy thay GCN--Transformer đã huấn luyện; còn thiếu checkpoint, training recipe, split Rome/San Francisco, VehiTrack EIE và table parity.
- **AnotherMe:** dùng router và endpoint WGS84 cục bộ thay AMap/GCJ-02; còn thiếu virtual-user/POI workflow, classifier và đánh giá mobile.
- **Semantic correlation 2026:** dùng thống kê tần suất và loại đường OSM khi không có weights/AMap semantic tree; còn thiếu posterior LSP đã hiệu chỉnh, ASR/DER labels và table parity.

Các chi tiết đối chiếu nguồn nằm trong:

- `docs/reproduction/transprotect.md`
- `docs/reproduction/anotherme.md`
- `docs/reproduction/semantic_correlation.md`

## Bản chạy kiểm tra tích hợp hiện có

- SUMO 1.27.1, một phương tiện đánh giá, tám sự kiện, 19 phương tiện nền và 133 chuyển tiếp.
- Mạng đường gồm 4.892 đỉnh và 9.138 cạnh có hướng; quỹ đạo SUMO và miền ứng viên dùng cùng graph.
- Kết quả hiện tại chỉ kiểm tra output contract, ranh giới ground truth, metric, bản đồ và khả năng chạy end-to-end.
- Chưa dùng để xếp hạng riêng tư vì chưa có nhiều user/seed, khoảng tin cậy, attacker chung và mức utility/overhead tương đương.

Artifact tham chiếu:

- `artifacts/benchmarks/dummy_benchmark_results.json`
- `artifacts/benchmarks/dummy_benchmark_map.html`
- `artifacts/benchmarks/dummy_benchmark_preview.png`

## Dữ liệu và đánh giá còn cần hoàn thiện

- Phần chuyên sâu của bảng ánh xạ S1--S7 được lưu tại [Bản draft ánh xạ threat model](threat_model_mapping_draft.md); bảng trong luận văn chỉ giữ diễn giải đơn giản để trình bày các kịch bản và định hướng xây dựng dataset.
- Sinh đầy đủ S1--S7 bằng SUMO kết hợp scenario generator: dwell, repeated reports, revisit, future route, query intent và co-mobility.
- Tách train/test theo user, route và seed; chạy nhiều giá trị $K$, ngân sách riêng tư và mức utility.
- Bổ sung POI/query workload và metric ứng dụng: top-$k$ POI recall/overlap, route error, travel-time error và task success.
- Xây attacker S2--S3 trước, sau đó attacker cho identity, future route, query content và group correlation.
- Báo attacker advantage so với prior/random, phân phối theo quỹ đạo và khoảng tin cậy.
- Thực hiện ablation cho REM anchor, stable offset, reachability/speed term, population prior và semantic term.

## Kiểm chứng phần mềm và tái lập

- Bộ test hiện có 124 test; con số này chỉ phản ánh kiểm chứng phần mềm, không thay thế validation khoa học.
- Web evaluator phải giữ ground truth ngoài public/attacker view. Nếu dùng dữ liệu nhạy cảm thật, cần vendor tài nguyên bản đồ và áp dụng CSP trước khi hiển thị trong trình duyệt có mạng.
- Hash, commit và lệnh tái tạo artifact được lưu trong các verifier ở `docs/reviews/`; chúng không thuộc phần nội dung chính của luận văn.
