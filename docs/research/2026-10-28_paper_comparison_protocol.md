# So sánh paper adaptations với GeoI-BQ: protocol 28/10/2026

Protocol được viết trước khi sinh outputs đối chứng mới. Mô hình đề xuất,
dataset, các kết quả cũ và 108 pins của GeoI-BQ v1 giữ nguyên. Đây là controlled
comparison trên cohort đã xem, không phải independent geometry confirmation.

## Phương pháp và fidelity

| ID | Public output | Cấu hình đã audit / phần thích nghi |
|---|---|---|
| raw | GPS tại mốc public 20 s | Không bảo vệ; sanity control |
| geo_i_Z_k1 | Một Z mỗi mốc | Dùng đúng GeoI_anchors từ source trip, không lấy mẫu lại |
| transprotect_markov | Một tọa độ thay thế | Candidate K10, 8 TRAIN-population targets, alpha10000, epsilon0.005/m; Markov proxy thay Node2Vec/GCN/Transformer của paper |
| semantic_poi | Điểm thật và K−1 dummy | K5, pool128, nearest public POI semantics và empirical predictor; không phải AMap/LSTM của paper |
| fake_queries | Real+dummy sets và các fake packets | K5, tối đa2 fake giữa lần thật, waits5–15s, 24 candidates/reference, similarity0.75; road/population-history adaptation |
| geoi_bq | Năm Q-only | Dùng đúng saved events của cache_target_g1 từ main_depth_20261014_v1; không chạy lại cơ chế |

Source mapping và gaps lấy từ `docs/reproduction/`. Các adaptation không được
gọi là official reproduction hoặc kết quả gốc của paper. AnotherMe offline,
DLS/RDG lịch sử không là main recent-online comparison.

## Cohort, clock và dịch vụ

48 family chia20 TRAIN /8 SELECTION /20 TEST; hai chuyến/family; TRAIN/SELECTION
một draw, TEST ba draws. Giữ đủ176 trip-draw, trong đó120 TEST. Không chọn trip
theo attack/utility score. Tất cả dùng cùng native graph,418 POI, bốn purposes,
reference top5, radius1000m và local destination đã biết.

Mỗi tọa độ hỏi mọi category với L60. Primary utility dùng exact evaluator
GPS tại grid0,20,...,600 s; bảo vệ GPS của GeoI-BQ vẫn pacing60s và cap0.23/m.
Giữ output count và fake timing của từng adaptation; mọi fake request trong
0..600 s đều đi vào chi phí, cache và cửa sổ attacker tương ứng. Metadata tĩnh
hợp lệ: báo current, epoch60 và session cache; session là primary.

**Discrepancy đã phát hiện trước khi chạy:** main runner cũ gửi trường public
`response_l=30` nhưng callback provider thực sự dùng reply60 và không đọc
trường này. Vòng mới thống nhất trường request thành60 cho tất cả arms,
không đổi source artifact. Kiểm tra exact Q/time, POI masks, utility và byte
counts của proposed so với source; hai giá trị30/60 cùng độ dài JSON. Đây là
sửa metadata hợp đồng công khai trong replay, không lấy mẫu hay tuning defense.

## Observation và attacker

- S1: một packet ở mỗi mốc service public; không nhìn trước/sau mốc đó.
- S2: toàn wire trong540..600 s; target là mean bốn native GPS ticks trong
  cửa sổ, chưa là semantic stop inference. Không lấy last-four packets khi
  fake-query có nhiều packet hơn.
- S3: toàn wire0..600 s; score tái dựng31 native GPS ticks cùng timestamp.
- S9/S10: toàn wire và public start/close600 s; target first/last native GPS.

Chỉ coordinates/timestamps vào feature API. Source index, real/fake flags,
Z, belief, private keys và labels không là attacker input. Public clock được
biết, nên fake timing không được giả định là một kênh bí mật.

Cùng họ bank: public geometry/history, ExtraTrees64, kNN1/kNN5, prior; endpoint
thêm observed-slot và geometry-linked sequences. Fit TRAIN, loss-specific
robust selection trên SELECTION: mean+SE cho MAE, mean−SE cho Hit, lexical
ties. Chọn và lưu model trước TEST predictions. Không coi finite bank là Bayes
optimal attacker. Không thay đổi defense sau khi đọc kết quả.

## Metrics, failures và phạm vi cost

Hit100/MAE, CI từng method và CI chênh utility: mean theo trip rồi family;
10.000 paired-family percentile bootstraps, seed public20261028.

Native Eq13 dùng cùng TRAIN-selected targets/prior của TransProtect adaptation,
directed shortest-path metres và disconnected-pair convention của provider.
Áp singleton outputs, dùng public coordinate snapping; không lấy centroid hoặc
secret-nearest Q để chế thành metric singleton. Đó là local-target-proxy
measurement, không phải parity với Rome/SF original experiment. ASR, DER và
indistinguishable-path count giữ N/A khi chưa có posterior/Sim/source-compatible
test. EIE point-estimator là MAE tương ứng, không là bằng chứng độc lập mới.

Giữ mọi failure và mẫu số. Privacy thiếu coverage không tính là thành công;
delivered utility của arm failed là0. Request/reply-index JSON bytes không gồm
metadata POI đầy đủ, HTTP/TLS, network latency, CPU/RAM/energy của điện thoại.
Thời gian generation chỉ đo cho adaptations mới; replay không phải thời gian
sinh Q của proposed. Shared setup và service replay được ghi riêng.

## Kiểm chứng và tính bất biến

Tests observation-window, label stripping, cache causal, all-packet accounting,
source pins và TRAIN-only prefix-causality smoke phải đạt trước freeze/run.
Verifier độc lập tái dựng requests/cache từ events, tính Recall từ reference
intersection, Eq13 và family/CI từ error tapes. Raw control phải khôi phục gần
như hoàn hảo các target trực tiếp quan sát được. Output mới ghi ở
`artifacts/benchmarks/paper_comparison_20261028_v1`; private keys chỉ ở
`/private/tmp/trajectory-paper-comparison-20261028-v1/private`.
