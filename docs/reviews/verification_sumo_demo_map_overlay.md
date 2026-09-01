# Verification — lớp nền đường cho SUMO SOTA demo

**Ngày kiểm tra:** 01/09/2026

**Phạm vi:** khả năng đọc của `sota_demo_map.html` khi tile bản đồ trực tuyến
không tải được, cùng ảnh preview tĩnh mới.

**Kết luận:** **PASS cho demo cục bộ**. HTML không còn phụ thuộc vào raster tile
để hiển thị bối cảnh đường; vẫn cần người dùng kiểm tra tương tác cuối cùng trên
trình duyệt đích trước buổi trình bày.

## 1. Nguyên nhân lỗi cũ

File HTML cũ dùng OpenStreetMap raster tiles làm lớp nền. Khi mở bằng đường dẫn
`file://` mà trình duyệt hoặc mạng không tải được tile, Leaflet chỉ còn nền xám;
các đường quỹ đạo vẫn hiện nhưng không còn bối cảnh địa lý. Đồng thời cả bốn
mô hình đều bật mặc định nên các nét chồng lên nhau.

## 2. Thay đổi đã kiểm tra

- Đọc geometry trực tiếp từ OSMnx road graph đã pin trong repository.
- Cắt đường theo bounding box chứa ground truth và toàn bộ public output.
- Loại các cạnh hai chiều trùng geometry, rồi nhúng thành hai
  `MultiLineString`: đường chính và đường phụ.
- Khởi tạo Folium với `tiles=None`; online OSM tile chỉ là lớp tùy chọn và tắt
  mặc định.
- Bật mặc định mạng đường cục bộ, SUMO ground truth và mô hình đề xuất; tắt ba
  SOTA để người xem bật lần lượt.
- Thêm start/end markers, `fit_bounds`, nền sáng, panel thông tin gọn và layer
  control không bị panel che.
- Nối Semantic Dummy theo opaque public candidate ID ổn định qua thời gian;
  visualizer không dùng real-candidate label để chọn hoặc tô nổi một track.
- Sinh thêm `sota_demo_preview.png`: bốn panel có cùng bounds và cùng lớp đường,
  mỗi panel chỉ trình bày một mô hình.
- Gắn nhãn ground truth đúng theo nguồn được chọn (`SUMO` hoặc `GeoLife`) và ghi
  attribution OpenStreetMap/ODbL trên cả HTML lẫn PNG.
- JSON ghi SHA-256 của đúng HTML/PNG đã sinh và đánh dấu hai hình là artifact
  evaluator-only có chứa ground truth.

## 3. Kết quả kiểm tra

```text
python -m tests.run_all       -> 54 passed, 0 failed
git diff --check              -> PASS
SUMO quick end-to-end         -> PASS
PNG signature / nonempty      -> PASS
HTML local-road structure     -> PASS
online tile disabled default  -> PASS
```

Visual QA trên PNG end-to-end xác nhận đường chính/đường phụ nhìn thấy rõ, SUMO
truth và output của mô hình đủ tương phản, bốn panel cùng phạm vi, legend/footer
không chồng nội dung. HTML và PNG dùng cùng hàm trích xuất mạng đường.

Kiểm tra HTML còn xác nhận layer OSM trực tuyến không được gọi `.addTo(map)` khi
khởi tạo, còn local road layer được nhúng trong file dưới dạng GeoJSON. Vì kết
nối browser automation của môi trường kiểm tra không khả dụng, thao tác bật/tắt
layer trên trình duyệt chưa được tự động chụp lại; đây là caveat của verification,
không phải fallback sang tile mạng.

## 4. Claim boundary và đề xuất tiếp theo

1. Lớp nền hiện lấy từ protection candidate graph đa phương thức, trong khi
   SUMO chạy trên passenger-only graph. Nó phù hợp để định hướng trên demo nhưng
   chưa chứng minh hai graph có topology tương đương.
2. HTML vẫn tải thư viện Leaflet/Folium từ CDN theo template mặc định. Road
   geometry không cần mạng, nhưng muốn chạy hoàn toàn air-gapped thì cần bundle
   cả JavaScript/CSS cục bộ.
3. Số lượng geometry khiến HTML lớn hơn trước; nếu mở chậm trên máy trình chiếu,
   có thể simplify đường phụ theo zoom hoặc chỉ giữ các highway class cần thiết.
4. Trước buổi report, mở artifact cuối trên đúng trình duyệt/máy chiếu và thử:
   zoom, pan, bật riêng từng SOTA, bật/tắt online tile, và kiểm tra tooltip.
