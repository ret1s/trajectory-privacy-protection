# Demo kết quả luận văn GeoI-BQ v1

Demo phát lại **output đã đóng băng** của phương pháp cuối cùng và sáu arm đối
chiếu. Dữ liệu là hành trình **SUMO tổng hợp**, không phải GPS trực tiếp của
người dùng. Demo không fit attacker, không lấy mẫu lại defense và không thay
đổi benchmark.

## Mở demo

Tại thư mục gốc repository:

```sh
python3 -m web.thesis_demo --port 5011
```

Mở <http://127.0.0.1:5011>. Chỉ cần Python 3.9 trở lên; không cần Flask,
Node, API key hoặc Internet khi trình bày. Dừng bằng Ctrl+C.
Server chỉ phục vụ những tệp trong manifest của demo, bind loopback và
không chạy thực nghiệm qua HTTP.

## Trình bày trong khoảng 5–7 phút

1. **Theo dõi thuật toán → Hướng dẫn trình bày.** Mốc 0 giây: chỉ GPS thật
   dùng tại local; REM tạo Z nội bộ; belief phục vụ việc chọn đoạn ba khung Q.
   Bấm từng component để xem vai trò và số liệu đã lưu.
2. **Mốc 20 giây.** Năm Q của khung tiếp theo được phát theo clock. Không đọc
   bảo vệ GPS mới, không chọn lại đoạn; allowance của kênh Q không tăng.
3. **Mốc 60 giây.** Có private read theo pacing, noisy test có thể giữ hoặc
   tạo Z. Nhánh thực tế của sample và ledger hiện bên phải. Không dùng một
   sample để khẳng định mọi phiên đều có cùng nhánh.
4. **Phía LBS server.** GPS/Z, ledger và đáp án local được ẩn. Server nhìn
   tọa độ công bố, clock và common cover request. Đây chỉ là góc nhìn minh
   hoạ: toàn bundle có ground truth để phân tích, không phải triển khai cách
   ly client/server hay endpoint production.
5. **Trở lại local.** Đổi purpose, loại POI và cache; bản tin không đổi và
   không phát sinh request mạng. Danh sách đầy đủ được lọc/sắp xếp theo GPS
   hiện tại, có phân trang “Xem thêm”. Recall@5 chỉ là phép đo chất lượng.
6. **Kết quả thực nghiệm.** Đọc Hit100 thấp hơn, MAE cao hơn cùng với Recall,
   CI và chi phí. Đổi S1–S3/S9/S10; bảng bổ sung có S4–S8 và phạm vi từng
   target. Dấu * chỉ adaptation đã audit, chưa phải implementation tác giả.
7. **Bảo đảm toán học.** Bốn mệnh đề và cận toàn transcript. Thay prior để
   minh hoạ vì sao cùng cận likelihood ratio vẫn cho posterior cao nếu prior
   đã mạnh. Công thức đầy đủ và giả thiết nằm trong “Mở luận văn”.

Space phát/tạm dừng; ←/→ đổi mốc khi focus ngoài ô nhập. Kéo/cuộn bản đồ để
pan/zoom. “Theo GPS” chỉ có ở góc nhìn phân tích; “Toàn cảnh” dùng extent
bản đồ công khai. URL giữ sample, draw, phương pháp và mốc để mở lại.

## Dữ liệu và cách đọc

- Bundle: `artifacts/demos/geoi_bq_20261111_v1/`. Gồm đủ 20 TEST families,
  40 native sessions và 120 draws; mặc định sample đầu theo ID, không chọn
  theo điểm. Mỗi phiên 31 mốc trong 600 giây. Các fake packet giữa mốc vẫn
  nằm trong history và được tính đủ ở chi phí.
- Sáu arm: Raw GPS, REM-Z K1, TransProtect Markov adaptation, Semantic
  correlation OSM/empirical adaptation, Fake-query road/history adaptation,
  GeoI-BQ v1. Raw gửi vị trí thật, REM-Z công bố Z; không áp hợp đồng “chỉ Q,
  Z nội bộ” của GeoI-BQ cho tất cả đối chứng.
- K=5 và L60 **mỗi loại/tọa độ** là cấu hình bản cuối đang được đối chiếu,
  không phải L30 của các slide lịch sử. Reply được union theo POI ID và cache
  tĩnh; “Đoạn 60 giây” là epoch cache, “Cả phiên” là session cache.
- Catalogue có 418 POI với ID/category nguồn, không có tên doanh nghiệp.
  Distance/time/detour và full ordering dùng đúng bảng reference benchmark;
  detour có đích local đã khai báo trong workload. Không có reference hợp lệ
  thì Recall là N/A, không tự điền 0 hoặc 100%.
- Overlay centroid chỉ minh hoạ S1 ở packet hiện tại; không phải estimator
  toàn quỹ đạo, không thay các attacker riêng theo loss ở bảng tổng hợp.
- Không dựng belief heatmap: frozen output không giữ vector belief đầy đủ.
  Score lượng tử hoá của ứng viên là score planner đã lưu, không phải
  xác suất GPS, probability chọn đoạn hay Recall thực tế.
- Đường đứt nối các mốc Q chỉ thể hiện thứ tự; không mô phỏng tuyến đường
  nối giữa chúng. Bản đồ vẽ một passenger lane mỗi external edge cho dễ xem.
- Chứng minh tách kernel lý tưởng, integer PMF và giới hạn float Geo-I/
  upstream restore. Không gọi việc replay này là chứng nhận triển khai
  pure-DP. Cận posterior minh hoạ có điều kiện, không phải attack accuracy.
- Kết quả chính là controlled comparison trên cohort đã xem. S4 đo nhóm
  tuyến, S5 chưa có positive control tốt, S6 giới hạn hai đích, S7 dùng
  common cover wrapper, S8 là fusion sau căn clock; không mở rộng kết luận
  sang identity người/xe hay mọi attacker.

## Nguồn và tái lập

Main benchmark: `artifacts/benchmarks/paper_comparison_20261028_v1/`.
GeoI-BQ ledger/diagnostics:
`artifacts/benchmarks/main_depth_20261014_v1/`.
Bổ sung S4–S8: `remaining_scenarios_20261109_v1` và
`s8_fusion_20261109_v2`. PDF là bản canonical trong `artifacts/reports/`.
Hash nguồn và mọi tệp bundle nằm trong `manifest.json`.

Runtime server kiểm tra hash bundle trước khi phục vụ. Exporter kiểm tra
108 research pins, receipts, kết quả nguồn và đối chiếu **267.840 ô utility**,
kể cả answer count/N/A. Nó tính lại scores/reference ranking từ bản đồ
công khai, không sinh lại Z/Q. Cần môi trường scientific của repository và
public resource cache để export lại:

```sh
python -m experiments.export_thesis_demo_20261111
python -m experiments.export_thesis_demo_20261111 --check
python -m pytest -q tests/test_thesis_replay_web.py
node tests/test_thesis_replay_logic.mjs
```

Đã kèm bundle, nên **không cần export trước mỗi buổi trình bày**. Sau khi
sửa source/assets demo, export lại rồi khởi động lại server để manifest và
asset khớp nhau. Không sửa trực tiếp artifact benchmark nguồn.

Kiểm tra browser tùy chọn, dùng Playwright và Chrome headless riêng:

```sh
node tests/thesis_replay_browser.cjs
```

Nếu Playwright không ở local node_modules, đặt `NODE_PATH` tới installation
đã có; `CHROME_PATH` đổi executable, `DEMO_URL` đổi URL. Screenshots desktop/
mobile và receipt browser được ghi vào thư mục ignored
`build/thesis_demo_20261111/`. Không dùng profile browser cá nhân.
