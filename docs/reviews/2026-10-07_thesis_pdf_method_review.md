# Visual review PDF: thuật toán, tài nguyên và S8 — 07/10/2026

Hồ sơ chỉ bổ sung; mỗi lần render được ghi riêng theo hash. Review này kiểm tra bố cục và tính đọc được, không chấm lại model, xác nhận chứng minh hoặc đo tài nguyên. Không chỉnh thesis hay artifact.

## Snapshot đã xem lần đầu

- PDF: `build/thesis/reviewed_20261007.pdf`.
- SHA-256: `25b330967aea24a4a931bba910c97383366a036dbdf8e357d6a579c6b3eb955c`.
- Số trang xác thực bằng PyMuPDF: 74.
- Đã xem PNG đầy đủ của 16 trang PDF: **39–50, 59–60, 64–65**, từ `build/thesis/qa_20261007/`.
- Page locator bên dưới là trang vật lý PDF, không phải số trang in ở footer.

| Trang PDF / trang in | Nội dung được xem | Nhận xét |
| --- | --- | --- |
| 39–40 / 33–34 | Phạm vi phương pháp và Hình 5.1 | Hộp thiết bị/server, vùng phương pháp, GPS/Z/Q/POI và các mũi tên đọc được. Caption đầy đủ; không chồng chữ hoặc cắt đường nối. Hình ở trang riêng có khoảng trắng, phù hợp một float toàn trang. |
| 41–44 / 35–38 | REM, thử giữ neo, cap, belief, objective và local purpose | Công thức, bảng 5.1–5.2, chỉ số/đơn vị và footnote rõ. Các dòng/cột không tràn lề; chữ tiếng Việt không mất dấu hoặc biến thành ô vuông. |
| 45–46 / 39–40 | Tiểu mục **5.6.1: Trình tự xử lý một sự kiện công khai** | Bảy bước có số và tên in đậm. Bước 4 hiển thị đúng cập nhật likelihood khi đọc riêng tư, kể cả giữ Z; event không đọc chỉ dự đoán. Bước 5 nối bình thường qua trang 46; không mất dòng hoặc nhầm số. Phân biệt nguồn vị trí và giới hạn chi phí tính toán nằm ngay sau thuật toán. |
| 47–49 / 41–43 | Giới hạn lý tưởng, S7, S9–S10 và bảng vai trò | Equations và bảng 5.3 đọc được, không tràn. Giới hạn float/PRNG/network timing và phạm vi đóng góp hiện rõ; S8 không được trình bày như cơ chế đã giải quyết đầy đủ. |
| 50 / 44 | Chuyển sang Chương 6 | Tiêu đề và hierarchy bắt đầu chương rõ; không còn số trang/caption của chương trước chen vào. |
| 59–60 / 53–54 | Tiểu mục 6.6.3 và Bảng 6.10 về S8 lịch sử | Chosen attackers in đậm, cột MAE/Hit100 rõ. Ba family, năm cặp, 41 pair-event và 24 event mục tiêu duy nhất hiển thị đầy đủ. Giới hạn public seed, lịch sử/không current Epoch8-L30 và không suy độc lập/quyền riêng tư nhóm nằm sát kết quả; không bị đẩy thành footnote khó thấy. |
| 64–65 / 58–59 | Mục 6.9 và Bảng 6.14 về tài nguyên | L10 planner riêng và prefix phản hồi L20/L30/L40 giữ allocation L60 được phân biệt rõ. Hàng bytes/MB căn đúng, đọc được số 2.168.881.152; caption ghi MB thập phân và không phải peak RAM. Các qualifier “nếu materialize”/“nếu đầy” hiện đủ. Đường dẫn nguồn xuống dòng trong lề, không bị cắt. |

**Kết luận của snapshot này: PASS về visual/readability trong phạm vi đã xem; không có blocker.** Chưa xem toàn bộ 74 trang. Ngắt trang giữa bước 5 và phần còn lại là bình thường, không yêu cầu chỉnh thuật toán.

Root sau đó thông báo sẽ chỉnh vị trí Figure 6.1 và font của phần provenance ở những trang khác. Snapshot đầu này được giữ như một record riêng; receipt cuối cần pin hash mới và xem lại các phần bị dịch trang. Không dùng pass này như xác nhận tự động cho một PDF khác hash.

## Receipt của PDF cuối

- PDF: `build/thesis/reviewed_20261007_final.pdf`.
- SHA-256 xác thực: **`8b36dae14084ce0a762488f762e2a2417bfd7bbb2b24b1ea0e03271311ff54b7`**.
- Số trang xác thực bằng PyMuPDF: **74**.
- Render được đọc: `build/thesis/qa_20261007_final/`.
- Đã xem lại toàn bộ sáu trang được yêu cầu: **45, 46, 59, 60, 64, 65**, và phần kết S8 ở đầu trang **61**.
- So sánh byte PNG xác nhận **toàn bộ trang 39–50 không đổi** so với snapshot đầu đã xem. Vì vậy nhận xét sơ đồ, chương phương pháp và chuyển chương ở trên áp dụng nguyên vẹn cho các render này. Trang 59/60/64/65 có reflow mới và đã được xem lại, không giả định giống snapshot cũ.

Kết quả recheck:

1. Tiểu mục 5.6.1 vẫn ở trang 45–46, bảy bước có thứ tự rõ; không cắt/chồng ký hiệu hay chữ. Cách phân biệt phép thử giữ neo với event không đọc GPS được giữ nguyên.
2. S8 bắt đầu ở trang 59; cảnh báo lịch sử/public seed và số pair-event/unique event ở đầu trang 60, gần Bảng 6.10. Attacker được chọn vẫn in đậm, năm view và đơn vị MAE/Hit100 rõ. Phần kết nối sang trang 61 đúng nội dung, không thiếu câu. Chẩn đoán này vẫn được trình bày riêng khỏi xác nhận cấu hình hiện tại.
3. Mục 6.9 ở trang 64–65. Bảng 6.14 nằm trọn trên trang 65; các cột bytes/MB và qualifier “nếu đầy”/“nếu materialize” đọc được. Đoạn về view L60 tiếp tục sau table với planner L10 riêng và cảnh báo không phải peak RAM; không có lỗi render hoặc mất dòng. Các đường dẫn provenance xuống dòng nhưng nằm trong lề.
4. Hash source vẫn đúng bản đã review: `current_algorithm.tex` = `e732a65530c1263e30ca28f3195d264dfaf7ada10104334c8995f20be7682cae`; `current_resources.tex` = `add9fd52e6da82e84159a59249d6e867663475ac3bf089114143f6e7a8bfe77e`.

**Final visual verdict: PASS trong phạm vi nêu trên; không có blocker cần sửa thesis.** Receipt này không tuyên bố review toàn bộ PDF, chấm lại số liệu, đo hiệu năng hoặc kiểm chứng định lý. Không chỉnh PDF, thesis hay artifact; chỉ bổ sung hồ sơ review.
