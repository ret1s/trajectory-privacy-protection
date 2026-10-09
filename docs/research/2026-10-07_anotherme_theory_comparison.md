# AnotherMe và Geo-I: trạng thái xác minh lý thuyết — 07/10/2026

**Chưa có toàn văn AnotherMe để xác minh phần chứng minh.** Vì vậy, số section, theorem, trang và loại bảo đảm toán học của paper hiện **chưa xác minh**. Không được biến giới hạn truy cập này thành kết luận “AnotherMe không có proof”, “chỉ có thực nghiệm”, hoặc “AnotherMe chứng minh DP”.

## Nguồn đã xác minh

Đúng công trình cần đọc là Yuanfei Li, Xiong Li, Xiangyang Luo, Zhetao Li, Hongwei Li và Xiaosong Zhang, *AnotherMe: A Location Privacy Protection System Based on Online Virtual Trajectory Generation*, IEEE TDSC 21(4), 2552–2567, 2024. Thông tin khớp [metadata do publisher gửi Crossref](https://api.crossref.org/works/10.1109%2FTDSC.2023.3314200) và [trang tác giả Xiong Li tại UESTC, mục 47](https://faculty.uestc.edu.cn/lixiong/zh_CN/index.htm).

| Nguồn primary | Đã đọc được | Giới hạn |
| --- | --- | --- |
| [DOI / IEEE document 10246991](https://doi.org/10.1109/TDSC.2023.3314200) | DOI, publisher và metadata | Web trả trang kiểm tra JavaScript; HTTP document trả body rỗng. Chưa đọc body paper. |
| [PDF publisher được Crossref chỉ định](https://ieeexplore.ieee.org/ielx7/8858/10592103/10246991.pdf?arnumber=10246991) | Xác minh đường dẫn trong metadata | HTTP 420; endpoint staging tương ứng HTTP 418; stamp PDF cũng không lấy được PDF. |
| [Kho của trường tác giả Zhetao Li, kết quả 43](https://ir.specialsci.cn/jnlib/scholar/result?page=5&scholar=c5350752a0c479e6%3AZhetao+LI) | Abstract về virtual user, ánh xạ POI, AMap và đánh giá nhận diện thật/ảo | [Record chi tiết](https://ir.specialsci.cn/jnlib/detail/f59fba80-b42f-4892-94ed-0d204a28621f) chuyển sang login. Abstract không thay phần proof. |
| [Repository chính thức, revision đã dùng](https://github.com/fang-zhiyou/AnotherMe/tree/0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f) | README, VTGA và danh sách file hoàn chỉnh | Tree không có PDF/doc/TeX; tám ZIP công bố cũng không có những định dạng này. Không tìm thấy manuscript. |

Không tìm thấy bản paper PDF trong repo, kể cả file ignored, hoặc cache `/private/tmp`. Browser tích hợp báo không khả dụng và danh sách browser được điều khiển trả rỗng. Kiểm tra UI bổ sung của agent chính thấy ứng dụng Chrome, nhưng truy cập bị chặn do quyền Accessibility/Screen Recording còn thiếu; không mở thêm luồng cấp quyền hệ điều hành. Chỉ mục OpenAlex/Semantic Scholar được dùng để tìm đường tới bản tác giả, đều không trả OA URL; chúng không được dùng để suy nội dung theorem. Không sử dụng mirror bên ngoài, vượt login hoặc gửi yêu cầu cho tác giả.

## So sánh có thể nói được lúc này

AnotherMe tạo hành trình ảo có mẫu di chuyển/POI phù hợp. Abstract báo recognition rate trung bình 53,8%; đây là kết quả của phép thử nhận diện được mô tả trong abstract, **không đủ để xác định toàn bộ bảo đảm toán học của paper**. [Abstract tại kho trường tác giả](https://ir.specialsci.cn/jnlib/scholar/result?page=5&scholar=c5350752a0c479e6%3AZhetao+LI).

Geo-I có tiêu chí phân phối rõ: với mọi vị trí `x,x′` và event đầu ra `E`,

```text
Pr[K(x)∈E] ≤ exp(ε·d_E(x,x′)) · Pr[K(x′)∈E].
```

Đây là **Definition 3.1, §3.1, trang PDF 4** của bản arXiv v3; **Theorems 3.1–3.2, §3.2, trang PDF 5** đưa các đặc trưng Bayesian cho mọi prior. Cận hạn chế thông tin thêm do quan sát, không bảo đảm attacker có sai số tuyệt đối lớn khi prior đã mạnh. [Bản tác giả Geo-I](https://arxiv.org/html/1212.1984v3), [PDF v3](https://arxiv.org/pdf/1212.1984v3).

| Câu hỏi khi so lý thuyết | AnotherMe | Geo-I / phần hiện tại của luận văn |
| --- | --- | --- |
| Secret, observation, metric, lớp attacker và lượng từ của guarantee là gì? | Cần đọc statement/assumptions trong toàn văn; chưa xác minh | Nền tảng dùng secret tọa độ và metric Euclid; luận văn khai báo riêng public clock, observation và cap epoch. |
| “Không phân biệt được” có nghĩa gì? | Không đồng nhất mô tả abstract/recognition rate với event-wise probability bound | Definition 3.1 là ràng buộc phân phối, không là tỷ lệ nhận diện 50% hay Hit100. |
| Có proof DP/Geo-I/composition hay một phân tích khác? | Chưa phân loại được | Chương 5 hiện nêu cận kernel/transcript lý tưởng; không coi float/PRNG simulator là triển khai đã được chứng nhận. |
| Có thể suy paper thua từ local benchmark không? | Không; implementation trong repo là adaptation | Điểm số/cap/output/cost và mẫu số phải khớp; proof tọa độ cũng không tự giải quyết identity, intent hoặc S8. |

Mã [VTGA `geo_obf`, dòng 136–144](https://github.com/fang-zhiyou/AnotherMe/blob/0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f/VTGAs/gen_virtual_traj.py#L136-L144) thêm nhiễu rời rạc `randint(-20,20)/800000` cho hai tọa độ. Việc có nhiễu không tự chứng minh DP. Nhưng helper này cũng **không đủ để phủ nhận** một guarantee khác của toàn hệ thống/paper; phải đọc đúng theorem và miền giả định trước.

## Hướng cập nhật chương khi có toàn văn

1. Trích **statement, section, trang, định nghĩa privacy và assumptions** của AnotherMe. Phân biệt proof bảo đảm riêng tư với phân tích độ phức tạp, tính hợp lý quỹ đạo hoặc tính đúng thuật toán. Một bảo đảm toán học không nhất thiết là DP.
2. So secret/observation, side information, phạm vi một chuyến/nhiều phiên, tham số và lượng từ của hai guarantee. Không chọn metric để chứng minh loại bảo đảm này mạnh hơn loại kia khi chúng khác bài toán.
3. Giữ phần kế thừa Geo-I tách khỏi đóng góp triển khai/hợp thành của luận văn. Giữ caveat kernel lý tưởng so với sampler thực, không quảng bá cap hữu hạn thành xác suất lộ vị trí hoặc bảo vệ đầy đủ S1–S10.

Các đoạn AnotherMe hiện tại ở `thesis/main.tex` (giới thiệu hệ thống và metric thật/ảo) không nói paper “không có proof”; phần metric đã yêu cầu đối chiếu toàn văn. Nên giữ cách diễn đạt này. Câu có thể dùng trong ghi chú nghiên cứu: **“Phân tích toán học của AnotherMe cần được đối chiếu theo đúng statement và giả định trong toàn văn; hiện chưa đủ nguồn để phân loại hoặc so độ mạnh với cận Geo-I.”** Chưa chỉnh thesis trong audit này.

## Pins và phần còn thiếu

- Revision tác giả: `0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f`.
- SHA-256 nội dung VTGA lấy từ raw GitHub tại revision đó: `bd154cc7e0e88306e942a35acd9a71c6fb5ff3127f21e3014a30766ea4f21af7`.
- `docs/reproduction/anotherme.md`: `ea9d58acc416697b3195f2df0d08ab4315c5532b7dfc63568bda21a3e798b93e`.
- `benchmark/engines/anotherme.py`: `a3cbcc26110b2f5a53a9e11830ad07f22854589e21b2762e06a7a3c4f2b13041`; adaptation, không dùng nó thay toàn văn. [Hồ sơ tái lập](../reproduction/anotherme.md).
- Geo-I PDF primary tải để đọc tại `/private/tmp/geoi-1212.1984v3-primary-20261007.pdf`: 15 trang; SHA-256 `41969623c248a18be01b977fdca30c489d5d0b6967d1d064598441181fd3d452`. Đã đối chiếu text và render trang 4–5. Đây là cache tạm, URL/version/hash là locator bền vững.
- **AnotherMe full-text SHA / section / theorem / proof pages: chưa có.** Cần PDF publisher được cấp quyền hoặc manuscript do tác giả công bố để hoàn tất phần này. Không gán số theorem, công thức hoặc assumptions dựa vào suy đoán.
