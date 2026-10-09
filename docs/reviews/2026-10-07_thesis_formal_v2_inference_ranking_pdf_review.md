# Review PDF formal-v2: inference, ranking và belief bridge — 07/10/2026

**PASS trong phạm vi được giao: không thấy lỗi toán học hoặc lỗi hiển thị chặn.** Không chỉnh thesis, code, dữ liệu hoặc evidence. Review này không tuyên bố kiểm trực quan toàn bộ PDF.

## Bản PDF và trang đã xem

- PDF cuối: `build/thesis/main.pdf`, **95 trang**, SHA-256 `d3d51ad7f97995cc098b3ba7a0546e679214a36501c137e885b8d0ca14e3f920`.
- Bản lưu bền vững: [PDF đã review](../../artifacts/reports/thesis_formal_review_20261007_v2/reviewed.pdf). Đã kiểm bản lưu trùng byte với `build/thesis/main.pdf` và cùng SHA-256 nêu trên.
- Đã xem ảnh thực tế các **trang vật lý 54–57 và 65–70**. Số in dưới trang tương ứng 47–50 và 58–63.
- Trang 54–57, 65–68 được xem từ `build/thesis/qa_formal_v2_20261007/`; trang 69–70 được xem lại từ `build/thesis/qa_formal_v2_20261007_final/` sau sửa layout.
- Đã đối chiếu byte của cả 95 PNG candidate/final: **chỉ 69 và 70 thay đổi**. Tám trang đã xem trước đó giữ nguyên byte, nên nhận xét trực quan chuyển sang PDF cuối có căn cứ.
- Candidate trước có PDF SHA `e033ab14504258c81f7da15a2bac3c96c2fd3d160bdf0486ccc4889f49b0a8c3`; đây không là pin cho bản cuối.

## Kết quả đọc source và đối chiếu implementation

Agent viết review này là tác giả module inference. Các phép tính tự kiểm của inference không được gọi là review source độc lập; module đó đã được agent `boundary_noise` đọc độc lập. Module ranking do `boundary_noise` viết, còn belief bridge do agent chính viết; các nhận xét ranking/bridge dưới đây là đọc độc lập của reviewer hiện tại. Review ảnh PDF được thực hiện trực tiếp trên mười trang nêu trên.

- **Inference, trang 54–57:** các công thức TV, balanced Bayes, expected MAP với prior bất kỳ và multicandidate hiển thị đủ dấu/mẫu số. Câu phân biệt expected success với posterior theo event nằm cạnh công thức; all-pair diameter, `M` khác `K`, conditional full protected history và reset không xóa quan sát cũ đều đọc rõ. Hệ quả sensor còn gắn với coupling có điều kiện và entropy independence, không gán certification cho GPS thực/local-ranking diagnostic.
- **Ranking, trang 65–68:** đọc source và công thức `MultiPurposeRoadRanking.scores`. Khoảng sai số có hướng `[-d_+,d_-]` đúng chiều tam giác; `e_p-e_a` của detour và việc triệt tiêu `e_a` khi so hai POI cho cận pair-error `b_D`. Cận generic `gap>2η_s` giữ **tập**, không hứa thứ tự nội bộ. Cận phần reference đã nhận và còn ổn định đúng theo đếm hạng: mỗi phần tử đó có tối đa `r_R−1` đối thủ khác trong reference đứng trước. Với radius, miền ước lượng nằm trong `E∪Bρ`; `G_r` kiểm cả membership chắc chắn và mọi entrant có thể có. Recall N/A khi reference rỗng được tách khỏi Recall 0 khi reference có nghĩa nhưng đáp án rỗng. Không bỏ khoảng cách snap vào điểm: code dùng trạng thái origin và vertex POI theo đúng graph D/T được phân tích.
- **GPS/road scope, trang 68:** cảnh báo Euclid 15 m không cho road-error 15 m nằm ngay trước bridge; Gaussian mỗi trục không bị trình bày thành hard bound. Mutual reachability, graph/catalogue/destination/status cố định và giới hạn float vẫn được khai báo. Không thấy proof mới cho dữ liệu thực hoặc privacy từ robustness.
- **Belief bridge, trang 68–70:** đã đối chiếu `_poi_weights`, wrapper response-aware và objective không category-cap của cấu hình legacy. Hàng latent rỗng đóng góp 0 vào surrogate, không đổi N/A benchmark thành 0. Reference5, signature planner10 và reply30 được tách rõ. `|F_b-F_mu|≤ζ` đúng khi `f_Q∈[0,1]`; áp sai lệch trước/sau greedy cho `½ optimum − σ − 3ζ/2`. Giả định calibration chưa đo và scope nearest tại representative được ghi cạnh cận. Không suy utility của GPS liên tục, fastest/radius/detour hoặc tối ưu toàn chuyến từ đó.

## Layout cuối

Không thấy công thức bị cắt, chồng chữ, thiếu glyph, tràn margin hoặc footnote đè nội dung. Các công thức ranking dài trên trang 67 vẫn nằm trong khung. Sửa layout của agent chính đã đặt trọn nhãn **“Hệ quả (cận độ phủ khi belief có sai lệch)”**, điều kiện và công thức (5.59) trên trang 70; không còn tách nhãn bold qua 69/70. Chuyển đoạn theo trang ở những nơi khác không làm mất giả định hoặc một vế bất đẳng thức. Footnote/source paths đọc được; các scope/caveat nằm gần công thức liên quan.

## Source pins khi chốt review

| File | SHA-256 |
| --- | --- |
| `thesis/current_formal_inference.tex` | `b07255e86c447d9142743124e71045af420d836889a720dfbcdac4a4bc2a47c0` |
| `thesis/current_formal_ranking_robustness.tex` | `138963ee3cf727e4e437f8bd5cf8f9a1a01de8df7c6179646108d3b988c6a784` |
| `thesis/current_formal_belief_bridge.tex` | `0297177f143b77a890cf9ac1e776192338191c4934677bce0d23ffe0f4aa0054` |
| `benchmark/query_purpose.py` | `110ce445bb1a89283728b263c1994eda2cf601b3c6ae1eab49e31b2b6bad4443` |
| `evaluation/lane_travel.py` | `111e1bc109dd53e29922d44bfbc16bded8f1409e467e43faba3723026dc45bde` |
| `benchmark/anchor_belief.py` | `9a4d99261ce43c3ebc6d0ce118e974ea3778ca82d0a02ac36b428e669d43dc61` |
| `benchmark/response_aware_belief.py` | `2f75b486472430260fa5a112cf5f49cb2bfd18d67da378d8c62954c72a43f070` |
| `benchmark/engines/fair_cover.py` | `c739b71374a6682f6468588720660152bf73ad11c825237d90ed9a213cb8d8a6` |

## PNG pins của mười trang được review

Các locator bên dưới là `build/thesis/qa_formal_v2_20261007_final/page_NNN.png`.

| Trang vật lý | SHA-256 PNG |
| --- | --- |
| 54 | `d4e57871b2f6a14d86c195c28bdb7821c4013dec2c2fb2cc348c19bfedf498a7` |
| 55 | `b3f4c6d56d26a2d2b4b8d66431a0f4f74a9d8d1b69d21dbf6e7bf0115cb63fdd` |
| 56 | `c62cfd253ce53f0f696a0dfabe72c71e53aeaf5443bb0b0ba0626847f5b6d499` |
| 57 | `5b9da76330bc39a441290884b91704cbe92927e0551dd8233bf100b5c3ff0675` |
| 65 | `294870ca20a0f7874c6746ecfd2d841c851070889eda905969663f5b8f3cb21d` |
| 66 | `c68b951029bfe530ad9a196af51a7f7aab43ebe1cd0449699479c79bedfc5c44` |
| 67 | `56024fa97e7524516314478ae6780ee26690f9eb9e37f38e620dff8b2b796688` |
| 68 | `47633d9d2d26b43693e52e08efc9de7eaa2202a910b158e6e36066c8497ccf4a` |
| 69 | `961f66925c43dfca7368129efa265342c63b396dd2b51d013a0ed6f7d8313d7b` |
| 70 | `661a73c87a6d62ed597e42ee37a62068bce50c0c33292f166b5e53e6a3224981` |

Receipt này kiểm source/maths trong phạm vi nêu, hash/count và readability/layout của mười trang. Không chạy hoặc sửa benchmark, không chứng nhận executable sampler/entropy/timing, và không thay việc đọc full text AnotherMe còn thiếu.
