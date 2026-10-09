# Kiểm tra chương phương pháp của luận văn

Ngày kiểm tra: 06/10/2026. Phạm vi là chương 5 từ
[`thesis/current_method.tex`](../../thesis/current_method.tex), gồm hình kiến
trúc, công thức, tham số, bảng chức năng và giới hạn các phát biểu. Việc kiểm tra
này không thay thế kiểm chứng sampler, đánh giá deployment hoặc peer review.

## Kết quả trên bản dựng đã hoàn tất

**PASS trong phạm vi nội dung và bố cục chương phương pháp.** Bản được kiểm tra
là [`build/thesis/reviewed_main.pdf`](../../build/thesis/reviewed_main.pdf), 67
trang, SHA256
`2c739564dcebc6d04999e51153c654a5cd035700a0e44a2c2b361fd2415bac60`.
Module phương pháp có SHA256
`2a2690289899e329df507affb252dbe7a10fb6f938307276e58c99ce1700ce28`.

Đã kiểm tra **toàn bộ mười trang PDF 39–48**, tương ứng số in 33–42; hình kiến
trúc ở trang PDF 40. Từng trang đã được xem trực tiếp sau sửa. Sau lần dựng
cuối để rút gọn danh mục hình/bảng, render lại cả mười trang bằng cùng PyMuPDF
và hệ số 1,6: cả mười PNG giống từng byte với các trang đã xem trước đó, chỉ
đổi vị trí trong PDF. Danh mục viết tắt ở trang PDF 6 và tóm tắt ở trang PDF 7
cũng được xem trực tiếp trên bản cuối; REM dùng tên nhất quán Road Exponential
Mechanism.

Hình 5.1 không còn chữ đè hộp/tiêu đề; GPS và nhu cầu riêng đều là đầu vào ngoài
vùng mô hình; nhu cầu riêng chỉ tới ranking; Q đi qua dải trống, không xuyên
hộp đầu ra. Server nằm ngoài vùng mô hình. Bảng purpose không còn ngắt câu.
Không thấy ký tự thiếu, citation chưa giải quyết, công thức bị cắt, footnote
tràn lề hoặc nội dung bị mất. Trang cuối chương còn một đoạn kết ngắn; đây là
phân trang thưa, không phải thiếu nội dung hoặc lỗi tràn trang.

Kiểm tra chỉ cấp PASS cho bố cục và sự phù hợp của phát biểu với nguồn đã đọc.
Nó không cấp chứng chỉ privacy thực thi, khẳng định mọi S1–S10 đã giải quyết,
hoặc đánh giá toàn bộ 67 trang. Không sửa source/artifact đã chốt trong quá
trình review; chỉ sửa bố cục của module mới theo thống nhất với người tích hợp.

## Bản PDF đầu tiên và sửa bố cục

Bản đầu tiên của [`build/thesis/main.pdf`](../../build/thesis/main.pdf) có 65
trang, SHA256
`ea57bb0692b235bdfd814d936f34c621f9524cb4a297470b27a6484a4df09315`.
Đã sao chép thành snapshot riêng, render bằng PyMuPDF ở hệ số 1,6 và xem trực
tiếp **cả chín trang PDF 40–48**. Các trang tương ứng số in 34–42; số trang PDF
được dùng làm định danh trong kiểm tra này.

Các công thức, bảng tham số, ký tự tiếng Việt, citation và footnote đường dẫn
đều đọc được và nằm trong vùng trang. Hình 5.1 ở trang PDF 41 có ba lỗi bố cục:
tiêu đề vùng mô hình đè hộp lọc, nhãn Z bị chật giữa hai hộp và đường Q đi qua
hộp kết quả cục bộ. Bảng purpose ở trang PDF 45 còn ngắt một câu giữa hai từ.

Sau khi thông báo và được tác giả tích hợp đồng ý, chỉ sửa bố cục của module:

- Đưa GPS thật, ngữ cảnh công khai và nhu cầu riêng thành ba đầu vào nằm ngoài
  vùng mô hình; nhu cầu riêng chỉ có mũi tên tới bước xếp hạng cục bộ.
- Chừa dải riêng cho tiêu đề mô hình, tăng khoảng cách các bước, bỏ hai nhãn
  mũi tên không đủ chỗ và đặt đường Q có nhãn trong dải trống giữa mô hình và
  hộp đầu ra. Mũi tên kết quả cục bộ có lớp trắng tại chỗ giao đường để không
  bị hiểu là nối vào request.
- Giữ bảng purpose tại vị trí nguồn bằng `[H]`.

Không sửa công thức, tham số, nguồn, code cơ chế hoặc artifact đã chốt.
`git diff --check` đạt sau bản sửa. Trạng thái cuối và SHA của bản được xem đã
ghi ở phần đầu; SHA trong phần lịch sử này chỉ nhận diện lần dựng đầu tiên.

## Đối chiếu nội dung với nguồn

Đã đối chiếu các phát biểu quan trọng với
[cấu hình đã xác nhận](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/recommended_configuration.json),
[factory bộ chọn](../../benchmark/engines/public_service_planner.py),
[bộ lọc](../../benchmark/engines/filtered_cover.py),
[cap khớp](../../benchmark/engines/matched_filter.py),
[sổ epoch](../../core/session_budget.py),
[ranking cục bộ](../../benchmark/query_purpose.py) và
[protocol base-Q](../../artifacts/benchmarks/qplanner_depth_base_q_generalization_20261006_v1/protocol.json).

| Nội dung | Kết luận kiểm tra |
|---|---|
| Tên cấu hình hiện tại | Geo-I / REM, L=30; planner giữ `legacy_l10`; K=5 và local top-k=5. L là giới hạn mỗi loại mỗi Q, không phải mã model lịch sử. |
| Cap qua phiên | Epoch công khai `[0,12000)`, tám slot; H=12, U=23, u=0,00125/m; cap phiên hiệu lực 0,02875/m, danh nghĩa 0,03/m; cap epoch hiệu lực 0,23/m, danh nghĩa 0,24/m. |
| Thu phí và supplier | Dành trước 1/2 đơn vị rồi mới đọc; nhánh đầu/giữ/mới thu 1/1/2. H không được mô tả sai thành ceiling 12 lần GPS. |
| Miền và metric | REM có support đường công khai cố định nhưng điểm số dùng Euclid của phép chiếu; khoảng cách đường có hướng thuộc utility, không bị đồng nhất với metric riêng tư. |
| Ước lượng và Q | Belief là xấp xỉ phục vụ utility; Q dùng dữ liệu công khai hoặc đã bảo vệ, có miền tới được theo directed free-flow; không buộc chứa GPS hoặc có K điểm khác nhau. |
| Cận tối ưu | Cận greedy 1/2 dưới matroid được đối chiếu [Calinescu và cộng sự](https://epubs.siam.org/doi/10.1137/080733991); slack chỉ áp cho surrogate một bước dưới oracle/số học chính xác; không gán cận 1−1/e, Recall thực hay tối ưu cả trajectory. |
| Kernel và mã chạy | Tách các mệnh đề lý tưởng khỏi float64/Gumbel/Laplace/NumPy PRNG và thời gian packet thực. Chương không tuyên bố simulator có chứng chỉ pure Geo-I. |
| S7 | Cùng vị trí, lịch và ngữ cảnh thì đổi nhu cầu không đổi request; không suy ra intent độc lập vị trí/account/activation/click. Request luôn mọi loại, bốn purpose chỉ xếp hạng local. |
| Sensor và utility | Supplier bảo vệ có pacing 60 s; utility dùng ground-truth synthetic tại mỗi public event và đích thực làm local oracle. Không nhận kết quả đó làm đo năng lượng hay utility chỉ dùng sensor fix 60 s. |
| S9/S10 | Không warmup/holdback/future endpoint detector; Endpoint20 là cấu hình và cohort riêng. Không chuyển evidence endpoint thành privacy của vòng L30. |
| Đóng góp | Tích hợp và hợp đồng có nguồn; tăng L là utility/cost tĩnh có trả thêm bandwidth. Không nhận hai vòng planner thất bại làm cải tiến, không tuyên bố superiority comparator cùng chi phí. |

Lập luận lý tưởng đã được đọc theo từng bước: tỷ số kernel REM có hai hệ số
tam giác u/2; transcript mở rộng cố định nhánh/neo và quyết định đọc tiếp; mỗi
đường có tổng đơn vị không quá NU; lấy biên và hậu xử lý giữ cận. Phát biểu S7
giữ điều kiện cùng activation/schedule và loại retry phụ thuộc bí mật. Không
có phát biểu mới cho sampler hữu hạn hoặc ẩn account/IP.

[Ghi chú tích hợp và nguồn sơ cấp](../../thesis/notes/current_method_integration_2026-10-06.md)
giữ danh sách chi tiết các claim lịch sử cần bỏ khỏi PDF đang hoạt động.

## Đọc bổ sung các chương liên quan

Đã đọc riêng [`current_dataset.tex`](../../thesis/current_dataset.tex) và
[`current_conclusions.tex`](../../thesis/current_conclusions.tex). Các con số
60 nhóm, 480 chuyến, 312.480 điểm FCD, 240 chuyến hiệu chuẩn và 360 archive khớp
[metadata kiểm chứng cohort](../../artifacts/datasets/qplanner_fresh_native_20261006_v2/validation.json);
419 POI nguồn/418 POI dịch vụ cũng khớp. Cả bốn chênh lệch purpose current-only
của L30 trên TEST đều dương trong
[paired readout](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/paired_readout.json).
Các chương giữ rõ cùng bản đồ/tổng hợp, draw lồng trong nhóm, chi phí phản hồi
và scope của các thử nghiệm cũ. Câu S7 có thể bị hiểu GPS là công khai đã được
người tích hợp sửa thành ``cùng vị trí thật và lịch công khai''; reviewer không
sửa những module do người khác sở hữu.
