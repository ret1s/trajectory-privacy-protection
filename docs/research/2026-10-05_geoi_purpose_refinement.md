# Mục tiêu chọn Q công khai cho nhiều purpose: giữ cấu hình hiện tại

**Vẫn giữ Geo-I với α=0, L=20.** Vòng này chỉ thử thay trọng số POI để chọn Q; không thay REM, phép thử tái sử dụng, ε, prior vị trí hay ngân sách. Các trọng số mới có thể tăng Recall, nhưng chưa vượt qua cả kiểm tra suy luận đầu/cuối và chi phí.

`α` trộn mục tiêu chọn Q cũ với một mục tiêu công khai cho bốn purpose: gần nhất, thời gian đi nhanh nhất, trong bán kính 1.000m, và ít vòng đường nhất. Detour trong mục tiêu công khai dùng tám địa điểm công khai được chọn theo độ phủ bản đồ; **không dùng điểm đến riêng của người dùng**. α=0 giữ mục tiêu cũ, α=0,5 trộn hai mục tiêu, α=1 dùng mục tiêu mới. L là số POI/category/Q nhận từ server.

**α=0 vẫn phục vụ cả bốn purpose ở client.** Sau một lần nhận cùng Q/replies, client hợp các POI còn hiệu lực rồi xếp hạng theo purpose, GPS, radius và destination riêng tư ở local. Đổi purpose lúc trả lời không tạo thêm truy vấn hoặc lần gọi bảo vệ GPS. Thay α/L công khai có thể đổi Q; chỉ **Z, lịch đọc và nhánh tính ngân sách** giống nhau giữa các cấu hình thử.

## Chọn bằng development, không bằng điểm test

Đã ấn định α∈{0;0,5;1}, L∈{10;20;40}, K=5, ε tạo/thử `0,0025/m`, cap `0,0575/m` mỗi phiên. Sáu family fit, ba selection, ba test; lấy phiên đầu theo thứ tự ID ở mỗi family và hai lần ngẫu nhiên. Giữ nguyên tất cả chín cấu hình và raw trong artifact.

Cấu hình mới phải có worst-purpose Recall≥90%, tăng ít nhất 1 điểm phần trăm; với cả S9/S10, MAE không được giảm quá 10% và Hit100 không tăng quá 5 điểm phần trăm. Bytes không vượt 1,5×baseline, trừ khi tăng Recall ít nhất 5 điểm phần trăm. Attacker chọn riêng trên selection theo MAE và từng ngưỡng Hit.

| Cấu hình selection | Worst-purpose Recall | MAE S9 / S10 (m) | Bytes/tick | Kết luận |
|---|---:|---:|---:|---|
| α0, L20 | 96,45% | 995 / 849 | 7.695 | Giữ baseline |
| α0, L40 | 99,80% | 1.253 / 887 | 12.139 | Không qua chi phí |
| α0,5, L20 | 97,60% | 1.065 / 661 | 7.693 | Không qua S10 |
| α1, L20 | 97,95% | 836 / 777 | 7.692 | Không qua S9 |

L40 tăng bytes khoảng 58% nhưng tăng worst-purpose Recall dưới 5 điểm phần trăm. Các mục tiêu trộn ở L40 còn không qua kiểm tra endpoint. Không chọn chúng chỉ vì một cột utility đẹp hơn. **Chỉ baseline và raw được chạy trên test**; không có điểm test của cấu hình bị loại.

## Kết quả của cấu hình được giữ

| Test development | Gần nhất | Nhanh nhất | Trong radius | Ít vòng đường nhất | Mean bốn purpose |
|---|---:|---:|---:|---:|---:|
| Geo-I, α0 L20 | 96,33% | 96,33% | 90,10% | 94,57% | **94,33%** |
| GPS gốc, L20 | 100% | 100% | 100% | 95,94% | 98,98% |

Geo-I dùng khoảng **7.678 bytes/tick**, raw 1.542; chỉ tính JSON requests và ID replies, chưa HTTP/TLS. S9 MAE **1.497m**, S10 **1.046m**; Hit100 đều 0/6, trên ba family × hai lần ngẫu nhiên. Đây là mẫu nhỏ đã được dùng trong các vòng phát triển, không chứng minh attacker mới sẽ thất bại.

Giới hạn chính:

- Bản đồ dựng lại có **cùng tốc độ 8m/s trên mọi cạnh**: gần nhất và nhanh nhất thực tế trùng nhau trong cohort này. API có hỗ trợ thời gian khác khoảng cách, nhưng bảng này chưa kiểm tra đường có tốc độ/ùn tắc khác nhau.
- Danh sách POI server vẫn lấy theo khoảng cách với độ sâu hữu hạn. Ngay raw cũng chỉ đạt 95,94% cho detour; chưa bảo đảm nhận đủ POI tối ưu của mọi purpose.
- Radius không có POI phù hợp được giữ là N/A; không đổi thành 100%. Có 8.532 dòng reference rỗng trong toàn vòng.
- Mục tiêu công khai không phải posterior của intent. Có 33 latent profile không tới được POI và có khối lượng mục tiêu bằng 0.
- Seed của vòng này được công bố để tái lập simulation; deployment cần randomness riêng tư. Tọa độ Geo-I không ẩn account/IP, thời điểm bắt đầu/kết thúc hoặc identity metadata. Sampler hữu hạn vẫn có giới hạn kernel lý tưởng.

## Kiểm tra độc lập và artifact

[Artifact được promote](../../artifacts/benchmarks/geoi_purpose_refinement_20261005/README.md) là bản sao byte-identical của toàn bộ thư mục tạm, giữ các cấu hình bị loại. [Verifier](../../experiments/verify_geoi_purpose_refinement.py) không gọi hàm tổng hợp hoặc chọn cấu hình của runner: tính lại 43.872 dòng Recall, quy tắc selection, các MAE/Hit từ error rows; tái dựng POI khả dụng và directed-road ranking để kiểm tra **87.744 danh sách reference/returned**, toàn bộ **192 byte totals**, và **3.744 lỗi centroid/median/OLS**. Z/ledger khớp ở 168 lần so sánh qua α/L; các file backbone đã pin giữ nguyên. Test riêng xác nhận đổi cả bốn purpose/radius/destination/GPS local không thêm traffic sau cùng một fetch.

Các predictor học và Viterbi chỉ có error rows, không lưu fitted model: verifier tính lại metric từ lỗi đã lưu, không tuyên bố tái fit độc lập những estimator đó. Protocol pin các nguồn chính; một số helper phụ thuộc gián tiếp chưa được pin riêng. Verifier lưu hash hiện tại của chúng **chỉ để audit**, không gọi đó là cam kết đã có trước khi chạy.

Chạy: `python -m experiments.verify_geoi_purpose_refinement`. Nguồn kết quả: [readout.json](../../artifacts/benchmarks/geoi_purpose_refinement_20261005/readout.json), [selection.json](../../artifacts/benchmarks/geoi_purpose_refinement_20261005/selection.json), [verification.json](../../artifacts/benchmarks/geoi_purpose_refinement_20261005/verification.json).
