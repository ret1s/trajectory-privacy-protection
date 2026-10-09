# Kiểm tra độc lập phần chứng minh riêng tư — 07/10/2026

Review chỉ đọc `thesis/current_formal_privacy.tex`, nguồn primary và các lớp thực thi liên quan; không thay thesis, cơ chế, dữ liệu hoặc kết quả đã khóa. **Không thấy lỗi toán học chặn dưới các giả định kernel lý tưởng và hậu xử lý có kernel chung.** Không coi kết luận này là chứng nhận sampler executable.

## Đối chiếu nguồn

- [Predictive mechanism, arXiv 1311.4008v2](https://arxiv.org/pdf/1311.4008v2): statement Fact 1, Lemma 1 và Theorem 1 ở trang PDF 10; Appendix B, trang 24–25, chứng minh phép thử riêng tư và xác suất chung của run. Theorem 1 lấy supremum chi phí qua các run, không dùng mức chi trung bình hoặc chỉ số neo fresh quan sát được. Đã đọc text và render ba trang này. PDF HTML có thể đánh lại số theorem; số trên PDF là Theorem 1.
- [Geo-I, arXiv 1212.1984v3](https://arxiv.org/pdf/1212.1984v3): Definition 3.1 ở trang PDF 4, đặc trưng Bayesian ở trang 5; §3.3, trang 5–6, dùng khoảng cách lớn nhất giữa các cặp điểm trong tuple và giải thích hợp thành. Định nghĩa giới hạn tỷ số phân phối, không gán một Hit/MAE tuyệt đối.

## Những bước đã kiểm

1. **REM:** support hữu hạn cố định, score `−u·distance/2`; hệ số score và normalization mỗi bên trả tối đa `u·d/2`, cho tổng `u·d`. Không bỏ normalization hay cắt support bằng GPS thật.
2. **Phép thử:** khoảng cách tới cùng neo trong cùng prefix là 1-Lipschitz. Cả xác suất giữ và không giữ đều được chặn. Biểu thức xác suất giữ và hai số minh họa khớp; cận giữ-xa là cận event chung, không là xác suất có điều kiện hoặc cận Recall.
3. **Nhánh:** giữ tốn một đơn vị; tạo mới sau test tốn hai; neo đầu tốn một. Đây là cận xác suất chung cùng outcome/prefix. Fresh trùng tọa độ neo cũ vẫn khác nhánh giữ trong transcript mở rộng rồi được marginalize.
4. **Cap và supplier:** code dự toán một/hai đơn vị trước gọi GPS; pacing và gate không đọc raw GPS để quyết định. `matched_filter` cho `U=2H−1`. Sổ dùng `u=C/(N·U)`, cấp trước từng slot, không hoàn phần dư qua phiên. Do mọi đường chi tối đa `N·U`, tích tỷ số cho `exp(C·D_infinity)`; dừng dựa trên prefix không cần cộng phí riêng. `H` không là số lần GPS cứng.
5. **Bayes:** công thức odds là hệ quả đúng cho hai hypothesis có prior dương, cùng context; mixture đòi mọi cặp trace giữa hai support đều nằm trong bán kính khai báo. Đổi đơn vị `m` và `m^-1` nhất quán. Cận epoch `exp(23)` ở 100 m được gọi là rất lỏng, không chuyển thành thành công tấn công.

## Hai điểm cần giữ tường minh

**Coins hậu xử lý:** cần chúng độc lập coins REM/test trong mô hình lý tưởng, hoặc giả định trực tiếp `X → E → O` với kernel chung `T(O|E)` không phụ thuộc input. Chỉ nói RNG hậu xử lý không đọc GPS chưa đủ nếu nó dùng lại coins của cơ chế. Ví dụ `Y=X xor R` với `R` đều là đầu ra riêng tư, nhưng gửi thêm `R` làm lộ `X`. Đã gửi agent chính đề nghị làm rõ giả định này; code tách luồng anchor/dummy không thay chứng minh độc lập lý tưởng bằng một chứng chỉ PRNG/HMAC.

**Miền GPS:** `d_E` đo hai tọa độ đầu vào trong phép chiếu công khai cố định, không bao gồm altitude, accuracy, heading hoặc timing. Không tự suy cận đối với vị trí vật lý thật khi nguồn GPS có lỗi hoặc lấy mẫu theo dữ liệu. Lịch mở/đóng, metadata, real packet timing và local answer đã được loại khỏi observation cụ thể; mở rộng observation phải có phân tích mới. Không có kết luận toàn bộ S1–S10 hoặc group privacy từ định lý này.

## Pins của lượt review

| File | SHA-256 |
| --- | --- |
| `thesis/current_formal_privacy.tex` | `f6d302103c85d8c2716687931373957c930041428607810656fc26d54949dc27` |
| `core/mechanisms.py` | `2c950335809507f0551eeeea62089260e74fa94624b3a84eb64b4666ca0cadfa` |
| `core/session_budget.py` | `7bbb7ed8fd16252d0ed6280375c212e67c9216128d65e2a67f9e87e2e0039171` |
| `benchmark/engines/filtered_cover.py` | `59633ff32d36de8be5d20e88d62fa579d6c7bc24f9f081904326140ddc00ee7c` |
| `benchmark/engines/matched_filter.py` | `147c298c6f128c680ce899465a6dc2f980bb8f8456d38c2cc98726f47cd7e5e8` |
| `benchmark/engines/paced_guard.py` | `740b6b8eabce33a8b50bca55f33b22b868f5f42211f86600924d4d8d43b6ecf0` |
| `benchmark/engines/quotient_cover.py` | `93db80c1acce17b50e6dcdf4c711189604b003735b41a087c4d8d9ccaada8ed2` |

Primary PDF cache tạm: `/private/tmp/predictive-1311.4008v2-primary-20261007.pdf`, 27 trang, SHA-256 `47a3d5fdd619b2ae22567a175c9f318422b9907e3b7d0989fe121e4699620ee9`; `/private/tmp/geoi-1212.1984v3-primary-20261007.pdf`, 15 trang, SHA-256 `41969623c248a18be01b977fdca30c489d5d0b6967d1d064598441181fd3d452`. URL/version/hash là locator bền vững; chưa vendor các PDF vào repo.

## Receipt kiểm tra candidate cuối sau sửa giả định

Đã đọc lại phiên bản `thesis/current_formal_privacy.tex` SHA-256 `b510825b5116e580a1488bd02afd4a01e7452148244eabf33a9afa155038cbf0`. Đây là phiên bản **sau** lượt review đầu; pin đầu và các nhận xét trước được giữ nguyên ở trên. Giả định thứ hai nay nói tường minh randomness hậu xử lý độc lập randomness REM/phép thử. Miền so sánh nay nói rõ GPS đưa vào cơ chế, chưa tự bảo vệ vị trí vật lý thật khi cảm biến sai. Định nghĩa Geo-I được nêu riêng, đúng chiều tỷ số và đơn vị. Hai điểm làm rõ của lượt đầu đã được xử lý trong candidate này.

Đã tính độc lập ví dụ Bayes: `exp(0.125)/(1+exp(0.125)) = 0.5312093733737563`, làm tròn **53,12%** đúng cho một đầu ra riêng của kernel REM, hai vị trí cách 100 m, `u=0.00125 m^-1` và prior 50/50 tại lịch sử đang xét. Đây không là cận cho cả nhánh test rồi tạo neo: cận joint hai đơn vị tương ứng `exp(0.25)/(1+exp(0.25)) = 56,22%` với cùng loại prior. Cũng không áp 53,12% cho cả epoch hoặc attacker benchmark. Đã gửi agent chính đề nghị làm rõ cụm “quan sát REM mới” thành đầu ra riêng của kernel để tránh nhầm nhánh refresh; không sửa canonical source trong review.

Ví dụ ba lần đọc đầu/giữ/mới cho `(1+1+2)×0.00125 = 0.005 m^-1` cũng khớp. Sinh Q và phản hồi chỉ giữ cùng cận khi thỏa giả định hậu xử lý đã nêu, không cộng phí hoặc miễn phí cho một kênh raw-GPS khác.

Đã kiểm SHA-256 `build/thesis/main.pdf` là `bac481ed81dfbec1d195005f84e6b45e4637d15746856c91c09b63b5e67b4580` và số trang bằng **85**. Receipt này kiểm source/maths và hash/count, **không** tuyên bố visual QA toàn bộ PDF 85 trang. Không chỉnh thesis/core, không regenerate dữ liệu hoặc scores. Không có thêm nguồn AnotherMe full text, nên vẫn chưa phân loại hay xác minh theorem của AnotherMe.

## Receipt cuối: bản 83 trang và hệ quả một mẫu GPS

Đã kiểm phiên bản thực tế cuối `thesis/current_formal_privacy.tex` SHA-256 `b3a6f09db962ea0211addc5166ca6495aaef0004e8a4f80ed9b3b0dae3b6dd3a`, cùng `build/thesis/main.pdf` SHA-256 `e5f1981f2788ed7cac851c44ceca0528b2a38cac1ab0b9ded267303225662a4f`, **83 trang**. Hai receipts cũ mô tả đúng các candidate trước và được giữ lại; chúng không là pin cho PDF cuối.

Ví dụ 53,12% nay ghi rõ **chỉ xét riêng một đầu ra REM, không kèm kết quả phép thử**. Các giả định independence, support cố định, đo GPS đầu vào, lịch công khai và ranh giới quan sát vẫn hiện diện. Đoạn số học hữu hạn được rút gọn nhưng vẫn phân biệt proof kernel lý tưởng với float/PRNG/HMAC, giữ giả định thiết bị tin cậy và yêu cầu chứng nhận sampler/approximate riêng. Không còn điểm làm rõ đang chờ từ review trước.

Hệ quả mới khi hai trace chỉ khác mẫu `i0` là đúng: trên mỗi đường mở rộng chung, các khoảng cách `d_i` khác bằng 0, nên `u·sum_i c_i(e)·d_i = u·c_i0(e)·r ≤ 2u·r`. Sau đó cộng đường và áp kernel hậu xử lý chung. Dù hai lần chạy thực tế có neo và các đầu ra sau khác nhau, phép so xác suất dùng **cùng prefix** nên không trả thêm phí tại các mẫu có đầu vào bằng nhau. Đây là cận toàn transcript cho adjacency một tọa độ đo được trong cùng lịch, không là cận `2u` cho hai hành trình thay đổi tại nhiều thời điểm. Với `u=0.00125 m^-1` và `r=100 m`, `2ur=0.25`, `exp(0.25)=1.2840254167`; số **1,284** trong source khớp. Timing, metadata, local answer và suy nhà từ toàn tuyến không được đưa vào hệ quả này.

**Kết luận source/maths: PASS theo các giả định đã khai báo; không có blocker.** Không chỉnh canonical source/evidence, không chấm benchmark mới; hash/count của PDF không thay visual QA toàn bộ 83 trang. Trạng thái thiếu toàn văn AnotherMe vẫn giữ nguyên.
