# Vì sao objective normalized chưa cải thiện rõ utility?

Phân tích này dùng **development cũ**, không mở fresh data/kết quả. Nguồn là [DEVv3](../../artifacts/benchmarks/qplanner_development_20261006_v3/protocol.json): selection gồm 6 families, mỗi family có 3 draws và 8 sessions; 4.536 events/method. Trung bình tính trong từng draw, rồi cho các draws và families trọng số bằng nhau. Train được lưu riêng, không gộp vào selection.

Để nối objective với đúng event, diagnostic dùng **event macro**: tại một event, lấy trung bình Recall@5 của các purposes có reference. Đây khác **primary macro** của benchmark: tính conditional Recall của từng purpose qua các events trước, rồi trung bình các purposes. Within-radius có 30/4.536 events N/A nên hai phép tổng hợp hơi khác nhau. Không thay primary scores, selection gate hay kết luận của benchmark.

| Normalized arm | Proxy mean | Actual event macro | Trung bình \|actual − proxy\| | Pearson theo events, trung bình theo family/draw |
|---|---:|---:|---:|---:|
| Mean, slack .03 | 90,740% | 92,317% | 8,125 điểm % | 0,390 |
| Tight, slack 0 | 90,831% | 91,861% | 7,887 điểm % | 0,352 |
| Tail, λ=.25, slack 0 | 90,869% | 91,862% | 7,861 điểm % | 0,345 |

Proxy là kỳ vọng trên protected belief và public destination prior, còn actual là utility ở vị trí/destination thật của event. Khoảng cách và correlation trên chỉ mô tả độ khớp quan sát; **không phải kiểm định calibration**, bằng chứng causal, hay lỗi phải bằng 0. Legacy/aligned có objective khác nghĩa, nên không so trực tiếp giá trị proxy của chúng với normalized.

Ba quan sát cụ thể:

- Tight tăng proxy mean **0,091 điểm %** so với Mean nhưng giảm actual event macro **0,456 điểm %**. Q movement cùng chỉ số giảm từ **115,20 xuống 31,46 m/event**. Đây là khoảng cách đường chim bay giữa hai Q liên tiếp, không phải directed road distance hay chứng minh feasibility. Kết quả gợi ý cần kiểm tra utility qua thời gian; chưa chứng minh giảm chuyển động gây giảm Recall.
- Tail nhận exchanges ở **57/4.536 events (1,257%)**, tổng cộng 73 exchanges. Lợi ích proxy CVaR trong chính bước refine chỉ **0,00669 điểm %/event** khi trung bình cả events không đổi. So với Tight, actual event macro tăng **0,00119 điểm %**; ba draw lần lượt **+0,0565 / −0,1284 / +0,0755 điểm %**. Không thấy cải thiện ổn định qua draws. Không lưu counterfactual local Recall của bộ Q trước refine, nên không quy hiệu quả này cho từng exchange.
- Lúc khởi tạo, mỗi track có **64.633** reachable states. Các bước sau chỉ còn median **17–19 states/track**, mean **21,86–26,74** ở ba normalized arms. Đây là bucket được lưu, không phải audit lại directed routes. Bộ tối ưu có ít lựa chọn hơn sau khởi tạo; Tail còn 5 events chạm iteration limit, nên không có claim tối ưu toàn cục.

Bước tiếp theo nên **đánh giá độ khớp của public/protected belief proxy trước khi tiếp tục chỉnh objective**: tách theo purpose, cold/tail và uncertainty; kiểm tra public destination mixture có đại diện được local query workload hay không; đánh giá cả utility hiện tại và utility qua các bước. Giữ Geo-I, clock và private-read accounting. Vị trí/destination thật chỉ dùng để đánh giá, không đưa vào planner. Chưa đề xuất thêm threshold hay chạy thêm fresh trial từ diagnostic này.

[Diagnostic JSON](../../artifacts/benchmarks/qplanner_proxy_diagnostic_development_20261006_v1/diagnostic.json) chứa từng family/draw, phases và paired comparisons; [input/source manifest](../../artifacts/benchmarks/qplanner_proxy_diagnostic_development_20261006_v1/analysis_protocol.json), [exact saved counts](../../artifacts/benchmarks/qplanner_proxy_diagnostic_development_20261006_v1/exact_saved_counts.json) và [source](../../experiments/qplanner_proxy_diagnostic_20261006.py) cho phép kiểm tra lại. Ba focused tests pass; [58 phép đối chiếu số độc lập](../../artifacts/benchmarks/qplanner_proxy_diagnostic_development_20261006_v1/validation.json) khớp source arrays và primary readout theo từng purpose. [CLI replay](../../artifacts/benchmarks/qplanner_proxy_diagnostic_development_20261006_v1/cli_recheck_receipt.json) tái tạo toàn bộ numeric summaries/family-draw records chính xác; không mở fresh artifacts.

Để replay với đường dẫn tương đối, dùng [CLI adapter v2](../../experiments/qplanner_proxy_diagnostic_cli_20261006_v2.py): `python -m experiments.qplanner_proxy_diagnostic_cli_20261006_v2 --output artifacts/benchmarks/new_proxy_diagnostic`. Adapter giữ nguyên source/arithmetic v1, kiểm tra output vẫn nằm trong repository trước khi ghi, kể cả symlink. [Direct CLI validation](../../artifacts/benchmarks/qplanner_proxy_diagnostic_cli_replay_20261006_v3/relative_cli_validation.json) pass: output tương đối chạy thành công, numeric records khớp chính xác; write-once retry và outside/symlink output bị từ chối trước ghi. Tám focused diagnostic/layout tests pass.
