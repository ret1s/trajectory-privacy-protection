# Verification — Location & Trajectory Privacy Foundations v1

Ngày kiểm tra: 2026-09-03

Trạng thái: **PASS — sẵn sàng để đọc và dùng làm checklist review luận văn**

## 1. Phạm vi artifact

- LaTeX nguồn: `docs/study_guides/location_trajectory_privacy_foundations.tex`
- PDF giao: `output/pdf/location_trajectory_privacy_foundations.pdf`
- Kích thước: 43 trang A4; text layer có thể tìm kiếm; font được nhúng.
- Mục đích: tài liệu tự học và cầu nối để review graduation thesis, không phải một chương của luận văn và không tự nhận là systematic review/SOTA survey năm 2026.

Nội dung đã được tổ chức theo chuỗi câu hỏi: bài toán và trust boundary → spatial
`k`-anonymity/spatial cloaking → Casper → continuous attacks → dummy generation →
attacker-aware metrics → Geo-Indistinguishability → đối chiếu các lớp bảo vệ → checklist
áp dụng cho luận văn hiện tại.

## 2. Kiểm tra nguồn và claim

Các claim cốt lõi đã được đối chiếu với paper gốc hoặc bản author-hosted:

- Chow–Mokbel 2011: taxonomy trajectory privacy, snapshot/tracking attacks và survey các hướng bảo vệ.
- The New Casper 2006 và Casper* 2009: privacy profile, spatial cloaking, candidate list và trusted-side refinement.
- Chow–Mokbel 2007: query sampling/query tracking, `k_l`, `k_q`, `k`-sharing và memorization.
- Chow–Mokbel–Liu 2011: peer-to-peer cloaking và giả định trust của peer.
- Chow–Mokbel–Bao–Liu 2011: road-network cloak và trade-off candidate-list/query cost.
- You–Peng–Lee 2007: dummy trajectories và ba metric SD/TD/DD lịch sử.
- Samarati 2001, Sweeney 2002: nguồn gốc `k`-anonymity.
- Andrés et al. 2013: Geo-Indistinguishability và planar Laplace mechanism.

Các chỉnh lý quan trọng so với materials cũ:

1. Internship 2 chỉ trực tiếp trích một bài của Mokbel: Chow–Mokbel 2011; survey này hệ thống hoá lĩnh vực chứ không phải nguồn phát minh mọi kỹ thuật.
2. Metadata bibliography cũ của bài survey sai. Metadata đúng là *ACM SIGKDD Explorations Newsletter* 13(1), 19–29, DOI `10.1145/2031331.2031335`.
3. New Casper dùng profile `(k, A_min)`; `A_max` xuất hiện trong Casper* như một extension, không được gán ngược cho New Casper.
4. `K` dummy của một client không đồng nghĩa spatial `k`-anonymity dựa trên `k` người thật; xác suất `1/K` cũng không tự động đúng nếu attacker xếp hạng không cân bằng.
5. Dummy trajectories cổ điển dùng contract **truth + dummies**. Thesis hiện tại dùng **dummy-only** nên attacker task và metric phải được định nghĩa lại.
6. Ví dụ TD trong survey có lỗi nội tại: text ghi `N_k=3`, nhưng liệt kê tám đường và tính `1/8`; tài liệu này giải thích nhất quán với `N_k=8`.
7. Bán kính planar Laplace là `Gamma(shape=2, scale=1/epsilon)`, không phải một biến Exponential đơn. Cơ chế Exponential-radius cũ trong Internship 2 vì vậy không phải đúng planar Laplace.
8. Geo-I theo từng event không tự trở thành trajectory-level epsilon; composition/budget theo chuỗi release phải được phát biểu riêng.
9. “Query quality” trong road-network paper là candidate target list/query-execution cost, không được đổi tên thành POI recall hoặc NDCG.

## 3. Kiểm tra build và nội dung PDF

Build command đã chạy thành công:

```bash
cd docs/study_guides
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=/private/tmp/location_privacy_foundations_build \
  location_trajectory_privacy_foundations.tex
```

Kết quả kiểm tra kỹ thuật:

- XeLaTeX exit code `0`; không có citation/reference chưa resolve và không có overfull box.
- Có một underfull box nhẹ trong ô bảng; không gây tràn, che chữ hay sai bố cục.
- `pdfinfo`: 43 trang, A4, không mã hoá, không JavaScript, metadata title/author hợp lệ.
- `pdffonts`: các font chính đều được embed; text extraction không có replacement character.
- `pdftotext -layout`: thành công, 14,117 token dạng từ theo `wc`; văn bản có thể tìm kiếm.
- SHA-256 của PDF giao: `b8e4e20ae5954a57483ad9c7811fb001dce7e6f6df7d1c3e069ace3284c13a47`.

## 4. Visual QA

- Đã render toàn bộ PDF bằng Poppler ở 110 DPI.
- Đã kiểm tra 43/43 trang: cover, mục lục, bảng, công thức, hyperlink text, header/footer và bibliography.
- 42 trang đầu byte-identical với vòng render đã duyệt trước đó.
- Trang bibliography cuối đã được chỉnh từ hai trang thành một trang và kiểm tra lại ở kích thước gốc.
- Không phát hiện clipping, overlap, hình mờ/khó đọc hoặc trang trắng ngoài ý muốn.
- Toàn bộ sơ đồ TikZ là hình minh hoạ tự dựng theo khái niệm; không sao chép figure có bản quyền từ paper.

## 5. Giới hạn còn lại

- Đây là nền tảng để hiểu và audit luận văn, không thay thế việc đọc paper gốc.
- Tài liệu không tuyên bố ba implementation hiện tại là reproduction chính thức hoặc SOTA năm 2026.
- Các guarantee chỉ đúng dưới threat model/output contract được ghi cạnh từng lớp kỹ thuật.
- Claim thực nghiệm của thesis vẫn phải được kiểm chứng bằng dataset/simulation, calibrated attacker, uncertainty và ablation riêng.

## 6. Cách dùng để review thesis

Nên đọc Chương 1–6 trước, sau đó dùng ma trận R1–R12 ở Chương 9 để kiểm tra lần lượt:
secret/output/trust boundary, snapshot hay continuous, online hay offline, ý nghĩa `k/K`, side
information của attacker, task của attacker, phạm vi Geo-I, post-processing, utility thực tế,
uncertainty và comparator contract. Chỉ review code/benchmark sau khi các câu hỏi này đã có câu
trả lời nhất quán.
