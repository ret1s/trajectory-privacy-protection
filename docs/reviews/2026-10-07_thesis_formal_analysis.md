# Phần chứng minh trước benchmark — 07/10/2026

Đã mở rộng phần lý thuyết thành Mục **5.7–5.8**, trước Chương 6 của [luận văn hiện tại](../../artifacts/reports/graduation_thesis.pdf). PDF đã biên dịch và kiểm tra đủ **83 trang**. Đây là chứng minh theo mô hình lý tưởng và review bằng assistant độc lập, không phải chứng nhận bảo mật của executable hoặc phản biện khoa học của hội đồng.

## Các kết quả đã chứng minh

| Thành phần | Kết quả | Điều kiện chính |
| --- | --- | --- |
| REM trên đường | Tỷ số xác suất bị chặn bởi `exp(u·d_E)` | Support và phép chiếu công khai cố định; tính cả normalization; sampler lý tưởng. |
| Kiểm tra giữ/làm mới | Giữ tốn `u`; test rồi làm mới tốn `2u`; neo đầu tốn `u` | Chặn xác suất **chung** của nhánh và neo tại cùng lịch sử; không công bố noise/coins gốc. |
| Cap qua phiên | Toàn transcript mạng logic thỏa `exp(C·D_infinity)` trong một epoch | Dự toán trước GPS; mọi đường chi không quá `N·U`; cùng lịch/ngữ cảnh và quan sát đã khai báo. |
| Chỉ thay một mẫu GPS | Toàn transcript thỏa `exp(2u·r)` | Mọi tọa độ khác, lịch và ngữ cảnh giữ nguyên; không thay một thuộc tính xuất hiện nhiều lần trên tuyến. |
| Q và phản hồi | Giữ cận riêng tư qua hậu xử lý | Kernel hậu xử lý chung, coins độc lập; không thêm raw GPS, local answer, retry theo nhu cầu hoặc kênh timing chưa mô hình hóa. |
| Nhu cầu riêng / S7 | Cùng vị trí/lịch, đổi nhu cầu local không đổi phân phối bản tin mạng | Purpose/category/radius/destination chỉ dùng local; không suy intent độc lập với tuyến hoặc metadata. |
| Track Q | Tồn tại chuỗi trạng thái khả thi theo graph có hướng | Chọn/thay luôn trong miền tới được; không chứng minh giao thông thực, tránh va chạm hoặc mọi cách nối Q đều khả thi. |
| Greedy và slack | `F_final ≥ F_opt/2 − slack` cho surrogate cùng bước | Trọng số không âm, oracle chính xác, partition matroid; không là cận Recall hoặc tối ưu cả chuyến. |
| Local top-k | Đúng reference nếu và chỉ nếu thu hồi đủ reference | Cùng miền hợp lệ, vị trí/ranking đúng, catalogue/version và tie-break. Reference rỗng là N/A. |
| Độ sâu phản hồi | L30 không giảm Recall so L20 | Cùng Q, response prefix, reference/miền và ranking đúng. Có phản ví dụ khi GPS local sai hoặc cache/clock khác nhau. |

Hệ quả Bayesian chặn mức thay đổi **odds**, tức tỷ số xác suất của hai giả thuyết sau quan sát so với trước quan sát. Không suy một tỷ lệ Hit100 cố định hoặc MAE tối thiểu từ epsilon. Đã thêm các phản ví dụ cho support theo GPS, giữ neo không trả phí và gửi request theo raw GPS.

## Ví dụ tham số hiện tại

`N=8`, `H=12`, `C=0.23 m^-1` cho `U=23`, `u=0.00125 m^-1`, cap hiệu lực mỗi phiên `0.02875 m^-1`; ngân sách danh nghĩa engine là `0.03 m^-1`.

- Ba lần đọc tạo đầu → giữ → làm mới chi `(1+1+2)u=0.005 m^-1`.
- Với ngưỡng 200 m, xác suất giữ là khoảng **58.55%** ở khoảng cách 50 m, **14.33%** ở khoảng cách 1,200 m. Đây là test có nhiễu, không phải cận cứng 200 m.
- Hai trace chỉ khác một mẫu GPS cách 100 m có hệ số tối đa `exp(0.25)≈1.284` cho toàn transcript lý tưởng.
- Chỉ xét riêng một đầu ra REM, hai giả thuyết cách 100 m có prior 50/50 cho posterior tối đa khoảng **53.12%**. Không áp số này cho nhánh test+làm mới hoặc toàn epoch.
- Toàn epoch với `D_infinity=100 m` có cận `exp(23)`, rất lỏng. Cận này giải thích hợp thành, không thay kết quả attacker thực nghiệm.

## Nguồn và đối chiếu AnotherMe

Đã đọc [Geo-I, bản tác giả v3](https://arxiv.org/pdf/1212.1984v3), Definition 3.1 và phần trace/Bayesian, cùng [predictive privacy v2](https://arxiv.org/pdf/1311.4008v2), Theorem 1 và Appendix B. Các primitive và cận greedy là nền tảng kế thừa; luận văn chứng minh điều kiện áp dụng vào pipeline đang chạy, không nhận chúng là nguyên lý mới.

**Chưa truy cập được toàn văn AnotherMe** từ publisher, repository tác giả hoặc kho trường. Vì vậy chưa xác minh theorem/section hay loại bảo đảm của paper, và chưa hoàn tất đối chiếu proof với AnotherMe. [Hồ sơ nguồn](../research/2026-10-07_anotherme_theory_comparison.md) giữ các đường dẫn và giới hạn truy cập; không suy đoán AnotherMe có hoặc không có DP.

## Kiểm chứng và bảo toàn evidence

- [Review nguồn riêng tư độc lập](2026-10-07_formal_privacy_source_review.md): REM, phép thử, pathwise composition, postprocessing, Bayesian và hệ quả một mẫu GPS đạt theo giả định đã nêu.
- [Review dịch vụ/PDF độc lập](2026-10-07_formal_service_pdf_review.md): proof khả thi, greedy/slack, purpose, local top-k và điều kiện đơn điệu đạt; phản ví dụ chạy trên ranker thật cho Recall `1→0.8` khi vị trí local sai.
- Tectonic hoàn tất với 0 lỗi, 0 cảnh báo overfull/undefined. Còn underfull và cảnh báo bỏ toán khỏi bookmark; chúng không làm mất công thức trong trang. Mọi font nhúng; 330 liên kết không có đích nội bộ lỗi; không có text vượt biên trang. Đã xem 14 contact sheet của toàn 83 trang và chi tiết phần proof.
- Hai exporter `--check` đạt: 29 JSON/8 TeX và 10 readout/3 TeX. Không chấm benchmark hoặc tạo khóa/draw mới. Chín evidence pin và hai export manifest của review trước giữ nguyên.
- Không thay code cơ chế, attacker, dataset, protocol hoặc scores. Chỉ đổi nguồn luận văn, lời kết, khoảng cách danh sách nguồn và handoff. Không chạy lại toàn suite cho thay đổi tài liệu; kết quả suite **988 passed / 9 skipped** thuộc revision trước, được lưu riêng trong review trước.

PDF cuối: SHA-256 `e5f1981f2788ed7cac851c44ceca0528b2a38cac1ab0b9ded267303225662a4f`. [Manifest, source snapshot và build log](../../artifacts/reports/thesis_formal_review_20261007_v1/) lưu 24 nguồn TeX và bytes PDF đã kiểm. Bản **74 trang trước khi mở rộng proof** vẫn giữ nguyên tại `artifacts/reports/thesis_review_20261007/reviewed.pdf`, cùng manifest/source snapshot cũ.

Hook đánh dấu thao tác PDF của skill đã được thử một lần trước authoring, nhưng môi trường không có `node`; không báo hook thành công. Biên dịch dùng Tectonic và visual QA dùng PyMuPDF/PIL. Việc đọc IEEE qua UI cũng không thực hiện được vì quyền Accessibility/Screen Recording đang thiếu; không thay quyền hệ thống hoặc vượt kiểm soát truy cập.
