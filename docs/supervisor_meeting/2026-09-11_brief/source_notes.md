# Nguồn, lựa chọn đối chứng và cách kiểm tra bản brief 11/09/2026

## Phạm vi

Bản ngắn dành cho GVHD: khép các yêu cầu ngày 05/09 trước, sau đó trình bày phương
pháp và kết quả đã có. Không chạy thêm/tinh chỉnh thuật toán, không sửa ground
truth, không thay bản luận văn 107 trang bằng bản ngắn. Nguồn định lượng khóa ở
commit `c47f61bd4036918b993e1a3a4eb1c804cb35ae16`; hash từng đầu vào nằm trong
`evidence.json`. Không dùng dữ liệu GeoLife để sinh, học hay kiểm thử trong brief.

## Kết luận lựa chọn đối chứng cho milestone này

- Nhóm hiện đại chính: TransProtect (2024), semantic correlation (2026),
  fake-query insertion (2026). Lý do tương ứng: ngữ cảnh đường, tương quan
  ngữ nghĩa/thời gian, phá liên kết bằng lịch công bố; có cả học sâu và không học sâu.
- Nền: DLS/enhanced-DLS (một paper, hai cấu hình) và RDG (chống Viterbi).
- Tham chiếu quỹ đạo: AnotherMe. Giữ phương pháp đã khảo sát, nhưng không trộn
  kết quả adapter đọc toàn đoạn vào bảng trực tuyến nhân quả.
- Chọn vào giao thức không có nghĩa đã chạy đủ hoặc đã tái lập bài gốc.
  Fake queries và RDG chưa có kết quả so sánh ở vòng mới. Ba adapter cũ vẫn là
  `paper_adaptation`; neural layers có trong mã không chứng minh lượt benchmark
  đã dùng mạng gốc. Hồ sơ `docs/reproduction/` là nguồn trạng thái hiện thực.
- ASA, mix zones, CkiDel, Wu, EPZ và công trình đồng hành dùng để xác định khoảng
  trống/chức năng theo kịch bản. Không đưa mọi citation thành comparator trực tiếp.
- Không tuyên bố các phương pháp được chọn là mạnh nhất mọi điều kiện. Không dùng
  xếp hạng tạp chí/hội nghị làm bằng chứng chất lượng thực nghiệm.

## Đối chiếu nguồn gốc, truy cập 11/09/2026

Chỉ sử dụng nguồn tác giả, nhà xuất bản và kho cơ quan cho các khẳng định kỹ thuật.
Tên ngắn [DLS], [SC]... trong bảng ánh xạ sang liên kết ngay dưới bảng và trong
`evidence.json`. Các hình là sơ đồ tổng hợp mới, không chép hình từ publisher.

| Nguồn | Phần đã đối chiếu | Kết luận được phép / hạn chế |
|---|---|---|
| Niu et al., INFOCOM 2014, DOI 10.1109/INFOCOM.2014.6848002 | Đọc lại text của paper lưu `cache/research_2026_09_07/dls2014.pdf`, abstract, §II-III và định nghĩa entropy | Entropy truy vấn, vùng che phủ CR của bản enhanced; không suy ra riêng tư chuỗi. |
| Sun et al., FGCS 2017, DOI 10.1016/j.future.2016.06.017 | Publisher abstract/highlights; tác giả HUST/UBC công bố PDF | MNAME/SNAME và phân bố dummy theo vùng chống LSA/RSA. Liên quan S2, không phải phép thử điểm dừng S2 nguyên bản. |
| Shaham et al., TMC 2021, DOI 10.1109/TMC.2020.2993599 | Kho KAUST và arXiv 1805.06104; abstract tác giả | RDG, entropy chuyển tiếp và chống Viterbi. Bản preprint sớm khác danh sách tác giả; dùng metadata TMC cho citation. |
| Yadav et al., SIGSPATIAL 2024, DOI 10.1145/3678717.3691211 | https://arxiv.org/html/2409.09495v1, §4.2-4.4, Eq.13, §5.1.4 | GCN/Transformer hỗ trợ chọn vị trí; EIE và sai lệch chi phí hành trình. Chưa đồng nhất EIE kỳ vọng với mọi cách tính MAE. |
| Li et al., TDSC 21(4), 2024, DOI 10.1109/TDSC.2023.3314200 | IEEE abstract/index (online 11/09/2023); https://github.com/fang-zhiyou/AnotherMe; hồ sơ mã đã ghim | Đã xác nhận chỉ số nhận ra quỹ đạo giả, đáp ứng và pin từ abstract; chưa tiếp cận đủ toàn văn để xác nhận công thức/mẫu số và mọi bảng gốc. Không tự gán AUC/F1 hay POI Recall cho bài. |
| Liu/Peng/Zhou, JKSUCIS 38:478, 2026, DOI 10.1007/s44443-026-00899-w | Publisher toàn văn §6.3-6.5 | ASR: N_success/N_total, với nhận diện thật ≤1/K. DER vừa có trung bình Sim vừa tỷ lệ đạt ngưỡng; cần công bố operationalization. Thiếu posterior/Sim cụ thể nên không coi chỉ sinh đủ K là ASR. |
| Liu/Hu/Zhou, JKSUCIS 38:53, 2026, DOI 10.1007/s44443-025-00438-z | Publisher toàn văn §6.3-6.6, published 03/01/2026 | Số đường nối khó phân biệt, ASR trước tương quan/khả năng tới/ngữ nghĩa, độ trễ và tài nguyên thiết bị. Chưa có định nghĩa posterior chung để coi ASR ngang với SC. |
| Beresford/Stajano, PerSec 2004 | Bản tác giả https://www.cl.cam.ac.uk/~fms27/papers/2004-BeresfordSta-mix.pdf; DOI đúng 10.1109/PERCOMW.2004.1276918 | Mix zones là bảo vệ liên kết qua bí danh, không che tài khoản ổn định được gửi. |
| Theodorakopoulos, Shokri et al., WPES 2014, “Prolonging the Hide-and-Seek Game” | Bản tác giả https://arxiv.org/abs/1409.1716 và https://carmelatroncoso.com/papers/Shokri-WPES2014.pdf | Bổ sung cơ chế tối ưu theo mô hình di chuyển cho S5. Có giả định về mobility/adversary; không gán kết quả của paper cho cạnh kế tiếp SUMO. |
| Xue et al., 2017, DOI 10.1177/1550147716685421 | Publisher: method CkiDel/DPCD, “Protection ability evaluation” | Bảo vệ đích đến bằng xóa check-in lịch sử; chấm predictive accuracy và aggregated distance error. Đúng mục tiêu S6 hơn chỉ dẫn paper dự báo; khác bối cảnh và không là dummy comparator. |
| Wu et al., WWW 24:25-49, 2021, DOI 10.1007/s11280-020-00830-x | Publisher abstract, published online 06/07/2020 | Chuỗi dummy che cả query locations và attributes. Không coi việc che loại POI đủ che mọi truy vấn văn bản; toàn văn nguồn này bị giới hạn truy cập. |
| Olteanu et al., TMC 2017, DOI 10.1109/TMC.2016.2561281; PoPETs 2019-0017 | Bản tác giả TMC, PoPETs abstract | Bằng chứng nguy cơ đồng vị trí và mô hình quyết định chia sẻ. Không ghi là bộ sinh dummy đã giải quyết S8. Khảo sát này chưa xác minh comparator bảo vệ đúng S8; không kết luận không có trong lĩnh vực. |
| Dhondt et al., CCS 2022, DOI 10.1145/3548606.3560616 | PDF tác giả, abstract/introduction và mô tả countermeasures | Không chỉ paper tấn công: có sáu biện pháp và đánh đổi, trong đó làm thô quãng đường. EPZ trên ứng dụng thể thao không đồng nhất mặt nạ 60 giây xe đô thị. |

## Nguồn định lượng và kiểm tra

- `artifacts/datasets/urban_fresh_v2/dataset.json`: đếm lại families, traces, FCD,
  records và từng `(scenario, split, A/B/C)`. Không dùng số đếm trong prose cũ.
- `artifacts/benchmarks/fresh_switching/readout.json`: chỉ bốn biến thể nội bộ,
  S1-S3, 53 bản ghi xác nhận, 636 lượt, 3984 sự kiện công bố.
- `selection.json`: cả L=5 và L=10 có `chosen=null`; quy tắc không được thay khi
  soạn báo cáo. Chọn theo Recall từng ca >=90%, sau đó Hit của bộ đối thủ.
- Chạy lại `experiments.verify_fresh_readout`: toàn bộ phép tính/tables được so
  khớp; sink xuất bị thay bằng so sánh receipt cũ để không ghi đè bằng chứng bất biến.
  Lệnh trực tiếp ban đầu tới cuối đã gặp FileExistsError ở thao tác ghi receipt;
  đó không phải phép so khớp thất bại và không được báo là đã chạy thành công lệnh đó.
- Thang đơn vị: Recall/Hit trong input là [0,1]; biểu đồ đổi sang phần trăm;
  chênh lệch theo nhóm là điểm phần trăm. k=5 không đổi khi L tăng 5→10.
- Trung bình kết quả theo ca rồi nhóm/lần lặp; S3.B có 5 nhóm thay vì 6.
  Không lấy trung bình trực tiếp 6 giá trị “family macro” để thay global macro.
- Chi phí 64,7% là tỷ số byte danh sách ID ở mean_exchange, không HTTP đầy đủ.
- Số lượt chạy không phải số người/nhóm độc lập. Không tạo CI hoặc p-value từ
  3984 sự kiện tương quan. Biểu đồ theo nhóm thể hiện kết quả trái chiều.

## Hình và cấu trúc

Audience: technical, giải thích bằng tiếng Việt. Một canonical artifact HTML và
PDF được in từ chính HTML, không có runtime biểu đồ thứ hai.

- Tóm tắt và kiến trúc: sơ đồ khái niệm trong nội dung React, không phải chart dữ liệu.
- Hai bảng S1-S10: exact lookup cho mục tiêu / dữ liệu / related protection / tình trạng.
- Bảng dataset: exact count cho 30 ca, chia hai vai trò. Không heatmap tô “bảo vệ”.
- Bảng comparator: phân biệt lựa chọn với bằng chứng tái lập; ghi metrics gốc ngắn.
- Bộ đo chung: privacy, utility, cost và giới hạn cho targets khác.
- Phương pháp và pipeline: sơ đồ khái niệm, không chụp hình từ paper.
- Recall: grouped bar, 4 cấu hình × 2 mức L, baseline 0, cùng %, hai màu,
  nhãn trực tiếp và legend L; không gọi Recall@10. Một hàng = cấu hình × L.
- Riêng tư: bảng vì cần đối chiếu ba chỉ số khác đơn vị/hướng, không trộn trục.
- Chênh Hit theo nhóm: bar có dấu, 6 nhóm; baseline 0, âm tốt hơn; màu chỉ hỗ trợ,
  dấu và nhãn truyền đạt đầy đủ. Không đánh đồng win-count với kiểm định thống kê.

Đã sắp xếp definitions/method trước kết quả theo yêu cầu người dùng. Further
questions và next steps gộp vào trang cuối; không thêm phần kết quả lịch sử dài.
Nguồn đầy đủ giữ tại đây để bản trình bày không biến thành nhật ký nghiên cứu.

## Tái tạo

Xem lệnh đầy đủ trong `README.md`. `build_report.py` tạo nội dung và reviewed
snapshot từ bằng chứng. `ReportContent.jsx` dùng các thành phần DataComponent,
DataTable, EvidenceChart và RichNarrative của Data Analytics; không có renderer
biểu đồ song song. `render_pdf.py` in trực tiếp compiled app với cùng dữ liệu và
SVG qua Chromium, không vẽ lại đồ thị. Mỗi bảng tối đa tám hàng; bảng đếm mười
scenario tách hai nửa năm hàng để không mất S9-S10 do phân trang.

Plugin Data Analytics đổi từ 0.2.10 sang 1.0.2 trong khi thực hiện. Lần package
theo bản cũ bị từ chối vì đòi SQL; không bịa SQL cho nguồn JSON. Bản cuối dùng
runtime 1.0.2 và ghi nguồn file / phép biến đổi Python đúng thực tế. Mã hạ tầng
plugin nằm ở thư mục tạm, chỉ commit phần nội dung, bằng chứng và bản xuất.

Không có trình duyệt giao diện trong phiên này (iab unavailable; browser list
empty). Chỉ xác nhận kiểm tra runtime khi chuyển PDF, hình vẽ và văn bản bản in;
không tuyên bố đã kiểm thử tương tác editor, hover hoặc kích thước điện thoại.

Các thiếu hụt còn lại: toàn văn AnotherMe chưa đủ; S8 chưa có comparator trực tiếp
đã xác minh; chưa có bảng so sánh SOTA trên bộ xác nhận mới. Đây là giới hạn nghiên
cứu hiển thị trong brief, không được diễn giải “tất cả yêu cầu đã giải quyết xong”.
