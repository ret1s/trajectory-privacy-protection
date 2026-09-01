# Verification — SUMO-first SOTA demo v2

**Ngày kiểm tra:** 01/09/2026

**Phạm vi:** nguồn mobility cho demo ba SOTA prototype và mô hình đề xuất

**Kết luận:** **PASS cho mục tiêu demo sơ bộ**, chưa đủ để dùng như benchmark
khoa học hoặc kết quả reproduction của các paper.

## 1. Thay đổi đã kiểm tra

- Mobility source mặc định đã chuyển từ GeoLife sang Eclipse SUMO 1.27.1.
- Pipeline chạy thật theo thứ tự:
  `Beijing OSM -> netconvert -> randomTrips.py -> sumo -> geographic FCD`.
- `--quick` vẫn gọi SUMO; không còn synthetic/GeoLife fallback âm thầm.
- GeoLife chỉ chạy khi chỉ định rõ `--mobility-source geolife`.
- SUMO network chỉ giữ đường dành cho passenger vehicle trong bbox nghiên cứu.
- Route seed, simulation seed và protection seed được tách riêng.
- File trung gian chỉ được ghi dưới `cache/sumo_demo/`, không rơi vào repository
  root và không được commit.
- Artifact schema được nâng thành `msc-sota-demo-v2`.

## 2. Kết quả chạy end-to-end

Lệnh chính:

```bash
venv/bin/python -m experiments.run_sota_demo --quick
```

Kết quả pilot:

- SUMO version: `Eclipse SUMO sumo 1.27.1`;
- netconvert version: `Eclipse SUMO netconvert 1.27.1`;
- selected vehicle: `smoke_2`;
- 8 samples sau resampling;
- time span: 540 giây;
- actual route: 87 SUMO edges;
- khoảng cách của 8 ground-truth samples đến OSMnx road graph đã pin: dưới 2 m;
- cả bốn prototype chạy hết trên đúng cùng một SUMO trajectory.

Hai lần chạy độc lập với cùng input/version/seeds cho cùng `record_id`, points,
timestamps và route-edge sequence. Timestamp comment bên trong XML có thể khác,
vì vậy kiểm tra reproducibility dựa trên semantic trace thay vì byte hash của
toàn file.

## 3. Kiểm tra boundary attacker/evaluator

`attacker_view` của cả bốn mô hình không chứa:

- SUMO vehicle ID;
- route edges;
- speed, edge ID hoặc lane ID;
- real candidate index/ID;
- real trajectory hoặc REM anchors.

Các trường trên chỉ nằm trong `mobility_evaluator_only` hoặc
`evaluator_truth`. Artifact vẫn lưu chúng để chấm điểm offline, nhưng attack
code phải chỉ nhận `attacker_view`.

## 4. Provenance đã kiểm tra

JSON v2 ghi:

- source commit và trạng thái dirty trước khi chạy;
- phiên bản `eclipse-sumo` và `sumo-data`;
- ba argv arrays chính xác của `netconvert`, `randomTrips.py` và `sumo`;
- route seed, simulation seed và protection seed;
- SHA-256 của OSM, SUMO network, trips, routes, FCD, actual vehicle routes và
  `randomTrips.py`;
- selected record ID và số samples;
- cảnh báo không được so các output contract như một leaderboard.

## 5. Automated verification

```text
python -m tests.run_all        -> 49 passed, 0 failed
py_compile                     -> PASS
git diff --check               -> PASS
SUMO artifact leakage audit    -> PASS
SUMO semantic reproducibility  -> PASS
```

Tests mới cover FCD coordinate order (`x=longitude`, `y=latitude`), route
parsing, deterministic selection/resampling, missing-tool failure, subprocess
argv/environment, SUMO-default CLI và no-fallback behavior.

## 6. Claim boundary và việc phải cải thiện

1. `randomTrips.py` chỉ phù hợp cho smoke demo; nó chưa đại diện cho population
   mobility thực tế và chưa tạo có chủ đích từng threat scenario S1--S7.
2. SUMO truth chạy trên passenger-only network, còn protection candidates hiện
   dùng OSMnx graph đa phương thức của benchmark cũ. Hình học pilot khớp tốt,
   nhưng benchmark chính phải tạo `RoadNetwork` từ đúng SUMO network hoặc một
   OSMnx drive graph tương ứng.
3. Pilot chỉ chọn một trajectory. Benchmark cần nhiều users, routes, seeds,
   stop/revisit/group schedules và train/test split.
4. Ba comparator vẫn là paper-inspired `*Lite`, không phải official/faithful
   reproduction. Không được gọi các số hiện tại là kết quả SOTA.
5. Metrics hiện tại là sanity diagnostics; cần context-aware dummy-filtering
   attacker, metric đúng từng output contract và communication/service cost.

Do đó kết quả hiện tại đủ để **demo pipeline và kiến trúc**, nhưng chưa đủ để
report performance khoa học hoặc chứng minh mô hình đề xuất tốt hơn SOTA.
