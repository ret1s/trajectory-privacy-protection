# Kiểm tra PDF: chứng minh độ chính xác của neo nội bộ

Ngày kiểm tra: 07/10/2026. **PASS trong phạm vi công thức, số minh họa, giới
hạn phát biểu và khả năng đọc của phần accuracy.** PDF cuối được giữ tại
[`reviewed.pdf`](../../artifacts/reports/thesis_formal_review_20261007_v2/reviewed.pdf),
95 trang, SHA256
`d3d51ad7f97995cc098b3ba7a0546e679214a36501c137e885b8d0ca14e3f920`.
Vị trí scratch đã xem là `build/thesis/main.pdf`, với hash candidate
`e033ab14504258c81f7da15a2bac3c96c2fd3d160bdf0486ccc4889f49b0a8c3`.
Đã kiểm tra trực tiếp cả năm ảnh trang PDF **57–61**, tương ứng số in
50–54. Mục 5.9 bắt đầu ở PDF 57, phần công thức và ví dụ ở PDF 58–60;
PDF 61 bắt đầu mục service 5.10. Đây không phải kiểm tra toàn bộ luận văn.

Nguồn [`current_formal_accuracy.tex`](../../thesis/current_formal_accuracy.tex)
có SHA256
`06df4f8f0654d5230a13a969a0e45e10914f7f111337acbb85b5201bd6f27bf4`,
khớp các byte đã được review toán độc lập trước tích hợp. Không có sửa nguồn,
code, scoring hoặc artifact trong lần kiểm tra PDF này.

Các công thức (5.30)–(5.38) hiện đủ dấu, chỉ số và mẫu số: CDF/quantile của
REM giữ multiplicity của ID; cận khối lượng gần/xa dùng đúng chiều đơn điệu
`B/(A+B)`; kernel hỗn hợp gồm `q·1[d>r] + (1−q)T`; failure rate kết hợp là
`αR+αT−αRαT`. Dấu max và ba điều kiện bán kính ở (5.36) không bị cắt.
Cận event không đọc tách lỗi neo cũ khỏi chuyển dịch của tọa độ đo. Tổng
union ở (5.38) dùng biến cố chung đọc-và-lỗi cùng cận số lần đọc theo đường;
không gán confidence của một lần đọc cố định cho last-read được chọn thích
nghi. Số phương trình và tham chiếu trong đoạn giải thích khớp nhau.

Ví dụ ba trạng thái được đánh dấu rõ là minh họa toán học, không phải số
benchmark. Các số đã kiểm tra độc lập khớp bản in: CDF tại 1.000 m là
0,9092; xác suất giữ ở 3.000 m là 0,0151; tail hỗn hợp là 0,1045;
`200+800·log20≈2.596,6 m` và failure rate 0,049375. Văn bản yêu cầu tiếp
tục lấy max với quantile REM của chính bản đồ/input đang xét, không nhận
bán kính này làm cận 95% của cohort thực nghiệm.

Các giả định và giới hạn đều đọc rõ: kernel lý tưởng, vị trí đo trong hệ
phẳng, không áp hằng số planar Laplace cho REM, measured-input displacement
là giả định thêm, pacing tối thiểu 60 s không cho tuổi neo tối đa 60 s,
Gaussian không có hard noise bound, và cap có thể ngừng đọc. Kết luận cuối
chỉ áp cho neo Z, không cho từng Q, posterior calibration, Recall, GPS thật
hoặc sampler float. Không thấy ký tự thiếu, công thức/footnote tràn lề,
chữ chồng, citation chưa giải quyết hoặc nội dung bị mất trong phạm vi
trang được xem. Phần no-read ngắt qua trang 59–60 nhưng liền mạch và đủ
nội dung. `git diff --check` đạt sau khi thêm ghi nhận này.

## Xác nhận bản PDF cuối

PDF cuối vẫn có **95 trang**, SHA256
`d3d51ad7f97995cc098b3ba7a0546e679214a36501c137e885b8d0ca14e3f920`.
Đã xác thực trực tiếp hash và số trang của PDF hoàn tất, rồi so từng byte
của cả năm PNG trang 057–061 trong `qa_formal_v2_20261007_final/` với
render candidate trong `qa_formal_v2_20261007/`. **Cả năm ảnh giống từng
byte**, nên phần accuracy trên bản cuối giữ nguyên toàn bộ nội dung và
bố cục đã xem trực tiếp ở candidate. Source accuracy cũng vẫn có SHA
`06df4f8f0654d5230a13a969a0e45e10914f7f111337acbb85b5201bd6f27bf4`.

**PASS phần accuracy trên PDF cuối này**, với cùng phạm vi PDF 57–61,
mục 5.9 và các công thức (5.30)–(5.38). Hash candidate `e033ab14…9b0a8c3`
ở phần đầu nhận diện lần kiểm tra hình ảnh trước sửa bố cục bridge 69/70;
không nhầm nó với hash toàn PDF cuối. Kiểm tra này không tự cấp PASS cho
mọi trang còn lại hoặc đánh giá lại chứng minh/scoring. Không sửa TeX,
code, dữ liệu hay số thực nghiệm trong lần xác nhận cuối.
