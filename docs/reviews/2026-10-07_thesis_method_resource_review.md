# Review thuật toán và tài nguyên — 07/10/2026

Kết luận: **không có blocker về nội dung** trong hai bản cuối `thesis/current_algorithm.tex` và `thesis/current_resources.tex` đã pin bên dưới. Trình tự xử lý và các con số tài nguyên khớp implementation và readout. Review chỉ đọc source/evidence; chưa đánh giá bố cục PDF, peak RAM hoặc tính đúng của một chứng minh Geo-I mới. Không thay core, thesis hay artifact đã khóa.

## Đối chiếu thuật toán

| Nội dung | Bằng chứng và kết luận |
| --- | --- |
| Dành ngân sách trước phiên | `PersistentEpochBudget.reserve` và `FixedEpochProtectedSessions.start_session` dành toàn bộ slot trước khi tạo engine/GPS. Không hoàn lại theo nhánh riêng tư; phiên bị từ chối không gọi supplier và không gửi Q. |
| Kiểm tra trước khi đọc GPS | `protect_step` tính `reserve=1` khi chưa có neo, nếu không là `2`; chỉ gọi supplier khi đủ khoảng cách 60 s và `spent+reserve<=max_units`. Thời gian công khai phải tăng, nằm trong epoch, không chồng phiên. |
| Chi phí tạo/giữ neo | REM đầu tiên chi một đơn vị. Các lần sau phép thử có nhiễu chi một; nếu refresh thì thêm một. Với policy hiện tại tám slot, H12, cap 0,23: U=23, u=0,00125 và cap mỗi slot 0,02875. Đây là cách hạch toán của cơ chế hiện tại, không phải số lần GNSS thực của điện thoại. |
| Ước lượng sau quan sát | `_FilterBelief.update` dùng `privacy_read_this_step`. Một phép thử giữ Z vẫn có emission; event không đọc chỉ dự đoán. Bước 4 cuối đã phân biệt đúng hai trường hợp này. |
| Hết cap khác hết slot | Phiên đã nhận hết cap tiếp tục postprocessing từ neo/belief đã bảo vệ. Phiên không được nhận trả tọa độ rỗng; `BudgetedGeoILbsClient.public_tick` không gọi retrieval. |
| Q và trả lời local | Q dùng belief, graph, lịch sử và context công khai; slack chỉ giới hạn objective proxy. Request không nhận QuerySpec riêng tư. `answer` dùng vị trí/purpose local và đáp án đã nhận, không gọi engine hay server, không thay sổ ngân sách. L30 là cấu hình được chọn trong nghiên cứu, không phải mặc định chung của mọi wrapper. |
| Hai nguồn vị trí | Vị trí cấp cho cơ chế riêng tư và vị trí dùng xếp hạng local khác vai trò. Phép thử local sensor chỉ thay local ranking, giữ protected tape; không chứng minh sampler Geo-I đã được đánh giá với GPS đầu vào nhiễu. Các phép đếm supplier/local-fix không phải phép đo tổng GNSS, pin hoặc năng lượng. |

Mô tả chi phí tính toán cũng đúng phạm vi: REM xét toàn miền đường cố định; belief sử dụng lưới/transition thưa; planner giới hạn vòng cải thiện; ranking dùng đường có hướng và cache. Những điểm này chưa cho số latency, CPU hoặc hiệu năng điện thoại. Giới hạn sampler lý tưởng, số học dấu phẩy động và approximation của belief vẫn cần giữ ở phần giới hạn chung.

## Đối chiếu tài nguyên

Readout `public-array-footprint-accounting-v1` đọc header/hash của NPZ công khai. Signature có shape `[66189,6,60]`, int32: 95.312.160 byte; access có 264.756 byte. File nén chỉ có 1.707.502 byte. Các số này được ghi đúng bằng MB thập phân (`10^6` byte).

| Giá trị | Phạm vi đúng |
| --- | --- |
| L10: 15.885.360 byte, 15,89 MB | Payload signature theo shape nếu materialize riêng; chưa gồm graph, access, belief, Python objects hoặc mảng tạm; không phải peak RAM planner. |
| L30: 47.656.080 byte | Payload bảng toàn miền giả định. Reply L30 tại năm Q không buộc điện thoại giữ bảng toàn bản đồ này. |
| Cache 256: 135.555.072 byte | `256*66189*8`, giả định đủ vector float64; không khẳng định cache đã đầy. |
| Cache 4.096: 2.168.881.152 byte | Cấu hình evaluator offline nếu đầy; không chuyển thành yêu cầu bộ nhớ cho client. |

Planner L10 thực tế được `native_resources` tạo/nạp riêng từ `reply10.npz`. Trong replay response-depth, `PrefixContext` cho L20/L30/L40 lấy view của `full.signatures[:,:,:depth]`, nên vẫn giữ allocation L60. Hai trường hợp này đã được phân biệt trong đoạn cuối của `current_resources.tex`. Audit không đo RSS, memory peak, tải context, CPU, latency hoặc năng lượng. Table chỉ là payload/ước tính cấu trúc; không có khẳng định triển khai trên điện thoại.

## Kiểm tra đã chạy

Hai lệnh chỉ đọc hoàn tất với exit code 0:

```text
/private/tmp/trajectory-research-20261005-venv/bin/python -m experiments.audit_public_resource_footprint_20261007 --check
PASS: public archive/array accounting; prefix/cache rows are analytic, not live memory benchmarks

/private/tmp/trajectory-research-20261005-venv/bin/python -m experiments.export_thesis_extensions_20261007 --check
PASS: 10 source readouts, 3 deterministic TeX fragments
```

Không fit model, chấm lại benchmark hay đọc private keys. Chưa kiểm tra render PDF; nếu chỉnh LaTeX sau review thì các hash thesis dưới đây cần được đọc như pin của bản đã review, không tự động áp dụng cho bản mới.

## Source pins

SHA-256 của bản cuối và nguồn chính đã đối chiếu:

| Path | SHA-256 |
| --- | --- |
| `thesis/current_algorithm.tex` | `e732a65530c1263e30ca28f3195d264dfaf7ada10104334c8995f20be7682cae` |
| `thesis/current_resources.tex` | `add9fd52e6da82e84159a59249d6e867663475ac3bf089114143f6e7a8bfe77e` |
| `core/session_budget.py` | `7bbb7ed8fd16252d0ed6280375c212e67c9216128d65e2a67f9e87e2e0039171` |
| `core/mechanisms.py` | `2c950335809507f0551eeeea62089260e74fa94624b3a84eb64b4666ca0cadfa` |
| `benchmark/engines/filtered_cover.py` | `59633ff32d36de8be5d20e88d62fa579d6c7bc24f9f081904326140ddc00ee7c` |
| `benchmark/engines/matched_filter.py` | `147c298c6f128c680ce899465a6dc2f980bb8f8456d38c2cc98726f47cd7e5e8` |
| `benchmark/engines/paced_guard.py` | `740b6b8eabce33a8b50bca55f33b22b868f5f42211f86600924d4d8d43b6ecf0` |
| `benchmark/anchor_belief.py` | `9a4d99261ce43c3ebc6d0ce118e974ea3778ca82d0a02ac36b428e669d43dc61` |
| `benchmark/engines/slack_progress.py` | `655e2a730323d1e365fef4ac8a14cbd67e7049b27816cf41fcae850651a97963` |
| `benchmark/budgeted_geoi_lbs.py` | `33fbb7344d5e4aed41943128d3e03d814a86bda496bc1271ba49b80cd09c8e10` |
| `benchmark/public_poi_context.py` | `98aa1301ce921630dbb77ef1bb7a19c6b70241e1689cccaa132192019b88ae86` |
| `benchmark/query_purpose.py` | `110ce445bb1a89283728b263c1994eda2cf601b3c6ae1eab49e31b2b6bad4443` |
| `experiments/future_sumo_eval.py` | `8705492563025bb6c4207c3b644d7db9f5366f466b7963bee4e824b2c6395e7c` |
| `experiments/qplanner_response_depth_20261006.py` | `8ee62128e52cbde68852f09fbfc1260b6441c03fe8167fe416e52f5a567c69b2` |
| `experiments/qplanner_study_20261006_v2.py` | `a3e177d4c3c704ce94878f8156b19f0e6a950a5b8a75ec634b9bfbdf28389102` |
| `experiments/audit_public_resource_footprint_20261007.py` | `e23a9150063940d2117b386800f7a5f45a5985ce4bfc0147564a2b5d937027b1` |
| `experiments/export_thesis_extensions_20261007.py` | `044e1b72472d83d506d1797b7ba766648e01882f1fd75b5a2a3f1ccc3d9b3002` |
| `artifacts/benchmarks/public_resource_footprint_20261007_v1/readout.json` | `20c10065c86fd1163a11e266b960633fc17755f28445f33723f518ecd53ff549` |
| `thesis/current_extensions_generated/resource_rows.tex` | `edc30f952902a4b6897d237bdfb584976ffc030cf28563658a010b0e72ca4958` |

Hash NPZ L60 được readout và lệnh audit xác thực: `ed341414e02173a430ea84eba0447466c3092b12a3addb8e2fb977c473119f94`. Các giới hạn còn lại là phạm vi đã khai báo, không phải lý do bỏ qua phép đo triển khai sau này.
